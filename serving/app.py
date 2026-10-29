"""Minimal authenticated OpenAI-compatible vision endpoint for a Colab loopback smoke."""

import base64
import binascii
import secrets
from threading import Lock
from typing import Annotated

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from shelfscan.images import ImageInputError, prepare_image


class ChatRequest(BaseModel):
    model: str
    messages: list[dict] = Field(min_length=1, max_length=2)
    max_tokens: int = Field(default=450, ge=1, le=600)
    temperature: float = 0.0
    stream: bool = False


def parse_message(messages: list[dict]) -> tuple[bytes, str]:
    if len(messages) != 1 or messages[0].get("role") != "user":
        raise ValueError("Pilot accepts one user message only")
    content = messages[0].get("content")
    if not isinstance(content, list) or len(content) != 2:
        raise ValueError("Expected exactly one text item and one image item")
    texts, images = [], []
    for item in content:
        if item.get("type") == "text" and isinstance(item.get("text"), str):
            texts.append(item["text"])
        elif item.get("type") == "image_url":
            images.append(item.get("image_url", {}).get("url", ""))
    if len(texts) != 1 or len(images) != 1 or len(texts[0]) > 12000:
        raise ValueError("Invalid pilot message")
    url = images[0]
    if not isinstance(url, str) or not url.startswith(
        ("data:image/jpeg;base64,", "data:image/png;base64,")
    ):
        raise ValueError("Only inline JPEG/PNG data URLs are accepted; remote fetch is disabled")
    if len(url) > 21 * 1024 * 1024:
        raise ValueError("Image data URL exceeds limit")
    raw = base64.b64decode(url.split(",", 1)[1], validate=True)
    return prepare_image(raw).jpeg, texts[0]


def create_app(generate, token: str) -> FastAPI:
    if len(token) < 24:
        raise ValueError("Use a strong randomly generated serving key")
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    busy = Lock()

    def auth(value):
        if not secrets.compare_digest(value or "", f"Bearer {token}"):
            raise HTTPException(status_code=401, detail="Invalid serving key")

    @app.get("/health")
    def health(authorization: Annotated[str | None, Header()] = None):
        auth(authorization)
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    def chat(request: ChatRequest, authorization: Annotated[str | None, Header()] = None):
        auth(authorization)
        if request.model != "shelfscan-student" or request.stream or request.temperature != 0:
            raise HTTPException(status_code=400, detail="Unsupported pilot model/decoding mode")
        try:
            raw, prompt = parse_message(request.messages)
        except (ValueError, TypeError, AttributeError, binascii.Error, ImageInputError) as exc:
            raise HTTPException(status_code=400, detail="Invalid image message") from exc
        if not busy.acquire(blocking=False):
            raise HTTPException(status_code=429, detail="GPU is busy")
        try:
            result = generate(raw, prompt, request.max_tokens)
        except Exception as exc:
            raise HTTPException(status_code=503, detail="GPU inference failed") from exc
        finally:
            busy.release()
        return {
            "id": "pilot",
            "object": "chat.completion",
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result},
                    "finish_reason": "stop",
                }
            ],
        }

    return app
