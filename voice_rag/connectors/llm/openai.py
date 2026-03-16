from __future__ import annotations
import json, time, uuid
from typing import Any, AsyncIterator
from openai import AsyncOpenAI
from voice_rag.core.streaming import format_sse

class OpenAIChatClient:
    def __init__(self, async_client: AsyncOpenAI | None = None, api_key: str = "", base_url: str = "https://api.openai.com/v1"):
        if async_client is not None:
            self._client = async_client
        else:
            self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def stream_chat_completion(self, messages: list[dict[str, Any]], model: str) -> AsyncIterator[str]:
        stream = await self._client.chat.completions.create(model=model, messages=messages, stream=True)
        saw_terminal_chunk = False
        async for chunk in stream:
            payload = chunk.model_dump(mode="json")
            if not payload.get("model"):
                payload["model"] = model
            if not payload.get("created"):
                payload["created"] = int(time.time())
            if not payload.get("object"):
                payload["object"] = "chat.completion.chunk"
            if not payload.get("id"):
                payload["id"] = f"chatcmpl-{uuid.uuid4().hex}"
            if any(choice.get("finish_reason") for choice in payload.get("choices", [])):
                saw_terminal_chunk = True
            yield format_sse(json.dumps(payload))

        if not saw_terminal_chunk:
            terminal_payload = {
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield format_sse(json.dumps(terminal_payload))
        yield format_sse("[DONE]")
