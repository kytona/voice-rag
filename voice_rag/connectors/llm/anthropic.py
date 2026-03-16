from __future__ import annotations
import json, time, uuid
from typing import Any, AsyncIterator
from voice_rag.core.streaming import format_sse, extract_text_content

class AnthropicChatClient:
    """Anthropic LLM client that converts to OpenAI-compatible SSE streaming."""

    def __init__(self, api_key: str = "", client: Any = None):
        if client is not None:
            self._client = client
        else:
            try:
                import anthropic
            except ImportError as e:
                raise ImportError("anthropic is required. Install with: pip install voice-rag[anthropic]") from e
            self._client = anthropic.Anthropic(api_key=api_key)

    async def stream_chat_completion(self, messages: list[dict[str, Any]], model: str) -> AsyncIterator[str]:
        system_content = None
        non_system = []
        for msg in messages:
            if msg.get("role") == "system":
                system_content = extract_text_content(msg.get("content", ""))
            else:
                # Anthropic requires content to be a plain string or Anthropic content blocks,
                # not OpenAI-style input_text/output_text arrays — flatten to string.
                text = extract_text_content(msg.get("content", ""))
                non_system.append({"role": msg["role"], "content": text})

        kwargs: dict[str, Any] = {"model": model, "messages": non_system, "max_tokens": 4096}
        if system_content:
            kwargs["system"] = system_content

        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())

        with self._client.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                payload = {
                    "id": completion_id, "object": "chat.completion.chunk", "created": created, "model": model,
                    "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
                }
                yield format_sse(json.dumps(payload))

        terminal = {
            "id": completion_id, "object": "chat.completion.chunk", "created": created, "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield format_sse(json.dumps(terminal))
        yield format_sse("[DONE]")
