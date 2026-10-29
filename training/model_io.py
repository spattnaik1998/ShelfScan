"""Native Transformers VLM loading and bounded generation, authenticated through HF."""

import gc
import io
import json
import time

from PIL import Image

from annotation.contracts import (
    Adjudication,
    AdjudicationRequest,
    Candidate,
    RoleSpec,
    TeacherRequest,
)
from annotation.teachers import ModelCallError
from training.runtime import load_json_object, require_hf_token


def messages(prompt: str, answer: str | None = None) -> list[dict]:
    result = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    if answer is not None:
        result.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return result


def make_batch(processor, raw: bytes, prompt: str, answer: str | None = None):
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    rendered = processor.apply_chat_template(
        messages(prompt, answer),
        tokenize=False,
        add_generation_prompt=answer is None,
    )
    return processor(text=[rendered], images=[image], return_tensors="pt", padding=True)


class VisionSession:
    def __init__(self, spec: RoleSpec):
        self.spec = spec
        self.model = None
        self.processor = None
        self.compute_dtype = None
        self.calls = 0
        self.generation_seconds = 0.0
        self.output_tokens = 0
        self.peak_vram_bytes = 0

    def load(self):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

        if self.model is not None:
            return self
        if not torch.cuda.is_available():
            raise RuntimeError("VisionSession requires a CUDA Colab runtime")
        if hasattr(torch.cuda, "reset_peak_memory_stats"):
            torch.cuda.reset_peak_memory_stats()
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.compute_dtype = dtype
        kwargs = {
            "revision": self.spec.revision,
            "token": require_hf_token(),
            "trust_remote_code": False,
            "torch_dtype": dtype,
            "device_map": {"": 0},
            "attn_implementation": "sdpa",
        }
        if self.spec.parameters.get("quantize", False):
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype,
            )
        processor_kwargs = {
            "revision": self.spec.revision,
            "token": require_hf_token(),
            "trust_remote_code": False,
        }
        if self.spec.family == "qwen2_5_vl":
            processor_kwargs.update(min_pixels=256 * 28 * 28, max_pixels=768 * 768)
        self.processor = AutoProcessor.from_pretrained(self.spec.model_id, **processor_kwargs)
        if self.spec.family == "smolvlm":
            self.processor.image_processor.size = {"longest_edge": 1024}
        self.model = AutoModelForImageTextToText.from_pretrained(self.spec.model_id, **kwargs)
        self.model.eval()
        return self

    def generate(self, raw: bytes, prompt: str, max_new_tokens: int | None = None) -> str:
        import torch

        self.load()
        batch = make_batch(self.processor, raw, prompt).to(
            device=self.model.device,
            dtype=self.compute_dtype,
        )
        started = time.monotonic()
        try:
            with torch.inference_mode():
                output = self.model.generate(
                    **batch,
                    do_sample=False,
                    use_cache=True,
                    max_new_tokens=max_new_tokens or int(self.spec.parameters["max_new_tokens"]),
                    max_time=float(self.spec.parameters.get("max_seconds", 120)),
                )
            generated = output[:, batch["input_ids"].shape[1] :]
            self.output_tokens += generated.shape[1]
            self.calls += 1
            return self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        except torch.cuda.OutOfMemoryError as exc:
            raise ModelCallError("gpu_oom", retryable=False) from exc
        finally:
            self.generation_seconds += time.monotonic() - started

    def close(self):
        import torch

        if self.model is not None and torch.cuda.is_available():
            self.peak_vram_bytes = max(self.peak_vram_bytes, torch.cuda.max_memory_allocated())
        self.model = None
        self.processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def metrics(self):
        return {
            "generation_calls": self.calls,
            "output_tokens": self.output_tokens,
            "inference_wall_seconds": round(self.generation_seconds, 3),
            "peak_allocated_vram_gib": round(self.peak_vram_bytes / 1024**3, 3),
            "billed_api_cost": 0.0,
            "gpu_billing_cost": None,
        }

    def load_adapter(self, directory):
        from peft import PeftModel, prepare_model_for_kbit_training
        from transformers import AutoProcessor

        self.load()
        # Match the base precision used during the smoke update (including float norms).
        if self.spec.parameters.get("quantize", False):
            self.model = prepare_model_for_kbit_training(
                self.model,
                use_gradient_checkpointing=False,
            )
        self.processor = AutoProcessor.from_pretrained(directory, trust_remote_code=False)
        self.model = PeftModel.from_pretrained(self.model, directory, is_trainable=False)
        self.model.eval()
        return self


class HubTeacher:
    def __init__(self, session: VisionSession):
        self.session = session

    def extract(self, request: TeacherRequest) -> Candidate:
        prompt = (
            request.prompt
            + (
                "\nReturn the annotation envelope, NOT BookRecord. "
                "Required fields: title, authors, publisher, edition, year, price, other_text. "
                "Every field needs status, value, reason, evidence. For absent/unknown use null. "
                "Present authors is a JSON array of names. Boxes use x0,y0,x1,y1 in [0,1]. "
                "Use null box when localization is uncertain. JSON schema:\n"
            )
            + json.dumps(Candidate.model_json_schema())
        )
        return Candidate.model_validate(
            load_json_object(self.session.generate(request.image.image_bytes, prompt))
        )


class HubAdjudicator:
    def __init__(self, session: VisionSession):
        self.session = session

    def resolve(self, request: AdjudicationRequest) -> Adjudication:
        context = request.model_dump(mode="json", exclude={"image", "prompt"})
        prompt = request.prompt + "\nReturn only {resolutions:{field:claim}}. Context:\n"
        prompt += json.dumps(context, ensure_ascii=False)
        prompt += "\nJSON schema:\n" + json.dumps(Adjudication.model_json_schema())
        return Adjudication.model_validate(
            load_json_object(self.session.generate(request.image.image_bytes, prompt))
        )
