from __future__ import annotations
from typing import Any
from voice_rag.core.models import ChatMessage

class DeepgramAdapter:
    """Adapter for Deepgram voice agent webhook.
    Deepgram's agent API uses an OpenAI-compatible chat completion format."""

    def parse_request(self, payload: dict[str, Any]) -> list[ChatMessage]:
        messages = payload.get("messages", [])
        return [ChatMessage(role=m["role"], content=m.get("content")) for m in messages]

    def format_response_headers(self) -> dict[str, str]:
        return {"content-type": "text/event-stream"}
