"""Start authenticated loopback services, verify one image, and always shut them down."""

import base64
import os
import secrets
import shutil
import subprocess
import threading
import time
from pathlib import Path

import yaml

from annotation.cache import atomic_json
from serving.app import create_app
from shelfscan.prompt import EXTRACT_PROMPT
from shelfscan.schemas import BookRecord
from training.model_io import VisionSession
from training.runtime import load_json_object


def wait_ready(url: str, key: str, process=None, seconds: int = 45):
    import httpx

    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError("Gateway exited; inspect private gateway startup log")
        try:
            response = httpx.get(url, headers={"Authorization": f"Bearer {key}"}, timeout=2)
            if response.status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    raise RuntimeError("Loopback service did not become ready")


def gateway_smoke(spec, raw: bytes, output: Path) -> dict:
    import httpx
    import uvicorn

    executable = shutil.which("litellm")
    if executable is None:
        raise RuntimeError("LiteLLM proxy CLI missing from pilot environment")
    session = VisionSession(spec).load()
    server, thread, process, log = None, None, None, None
    serving_key, gateway_key = secrets.token_urlsafe(32), "sk-" + secrets.token_urlsafe(32)
    try:
        session.load_adapter(output / "smoke_adapter")
        server = uvicorn.Server(
            uvicorn.Config(
                create_app(session.generate, serving_key),
                host="127.0.0.1",
                port=8001,
                log_level="error",
                access_log=False,
            )
        )
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        wait_ready("http://127.0.0.1:8001/health", serving_key)
        config = {
            "model_list": [
                {
                    "model_name": "shelfscan-local",
                    "litellm_params": {
                        "model": "openai/shelfscan-student",
                        "api_base": "http://127.0.0.1:8001/v1",
                        "api_key": "os.environ/SHELFSCAN_UPSTREAM_KEY",
                    },
                }
            ],
            "general_settings": {"master_key": "os.environ/LITELLM_MASTER_KEY"},
        }
        path = output / "gateway.pilot.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        env = {
            **os.environ,
            "SHELFSCAN_UPSTREAM_KEY": serving_key,
            "LITELLM_MASTER_KEY": gateway_key,
            "LITELLM_TELEMETRY": "False",
        }
        # The proxy process needs neither Hub authentication nor a public listener.
        env.pop("HF_TOKEN", None)
        log = (output / "gateway.private.log").open("w", encoding="utf-8")
        process = subprocess.Popen(
            [executable, "--config", str(path), "--host", "127.0.0.1", "--port", "4000"],
            env=env,
            stdout=log,
            stderr=log,
        )
        wait_ready("http://127.0.0.1:4000/v1/models", gateway_key, process)
        unauthorized = httpx.get("http://127.0.0.1:4000/v1/models", timeout=10)
        direct_unauthorized = httpx.get("http://127.0.0.1:8001/health", timeout=10)
        payload = {
            "model": "shelfscan-local",
            "temperature": 0,
            "max_tokens": 450,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": EXTRACT_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/jpeg;base64," + base64.b64encode(raw).decode(),
                            },
                        },
                    ],
                }
            ],
        }
        response = httpx.post(
            "http://127.0.0.1:4000/v1/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {gateway_key}"},
            timeout=180,
        )
        response.raise_for_status()
        record = BookRecord.model_validate(
            load_json_object(response.json()["choices"][0]["message"]["content"])
        )
        report = {
            "ok": unauthorized.status_code in (401, 403) and direct_unauthorized.status_code == 401,
            "gateway_auth_status": unauthorized.status_code,
            "upstream_auth_status": direct_unauthorized.status_code,
            "record_schema_valid": True,
            "model_alias": "shelfscan-local",
            "scope": "loopback only; no public tunnel",
            "title": record.title,
        }
        atomic_json(output / "gateway_report.json", report)
        return report
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if server is not None:
            server.should_exit = True
        if thread is not None:
            thread.join(timeout=10)
        if log is not None:
            log.close()
        session.close()
