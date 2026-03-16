from __future__ import annotations
from typing import Any
from voice_rag.core.models import ChatMessage

class ElevenLabsAdapter:
    def parse_request(self, payload: dict[str, Any]) -> list[ChatMessage]:
        if not payload.get("stream", True):
            raise ValueError("Only stream=true is supported for the ElevenLabs webhook.")
        messages = payload.get("messages", [])
        return [ChatMessage(role=m["role"], content=m.get("content")) for m in messages]

    def format_response_headers(self) -> dict[str, str]:
        return {"content-type": "text/event-stream"}
