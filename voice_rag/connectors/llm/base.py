from __future__ import annotations
from typing import Any, AsyncIterator, Protocol

class LLMClient(Protocol):
    async def stream_chat_completion(self, messages: list[dict[str, Any]], model: str) -> AsyncIterator[str]: ...
