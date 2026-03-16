from __future__ import annotations
from typing import Any, Protocol
from voice_rag.core.models import ChatMessage

class VoiceAdapter(Protocol):
    def parse_request(self, payload: dict[str, Any]) -> list[ChatMessage]: ...
    def format_response_headers(self) -> dict[str, str]: ...
