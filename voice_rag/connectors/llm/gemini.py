from __future__ import annotations
import json, time, uuid
from typing import Any, AsyncIterator
from voice_rag.core.streaming import format_sse, extract_text_content

class GeminiChatClient:
    """Google Gemini LLM client that converts to OpenAI-compatible SSE streaming."""

    def __init__(self, api_key: str = "", client: Any = None):
        self._api_key = api_key
        self._client_override = client

    def _get_model(self, model: str) -> Any:
        if self._client_override is not None:
            return self._client_override
        try:
            from google import genai
        except ImportError as e:
            raise ImportError("google-genai is required. Install with: pip install voice-rag[gemini]") from e
        client = genai.Client(api_key=self._api_key)
        return client.models

    async def stream_chat_completion(self, messages: list[dict[str, Any]], model: str) -> AsyncIterator[str]:
        system_instruction = None
        contents = []
        for msg in messages:
            role = msg.get("role", "user")
            # Gemini expects plain strings in parts, not OpenAI content arrays — flatten first.
            text = extract_text_content(msg.get("content", ""))
            if role == "system":
                system_instruction = text
            else:
                gemini_role = "model" if role == "assistant" else "user"
                contents.append({"role": gemini_role, "parts": [{"text": text}]})

        genai_model = self._get_model(model)
        kwargs: dict[str, Any] = {"model": model, "contents": contents}
        if system_instruction:
            kwargs["config"] = {"system_instruction": system_instruction}

        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())

        response = genai_model.generate_content_stream(**kwargs)
        for chunk in response:
            text = chunk.text if hasattr(chunk, "text") else str(chunk)
            if text:
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
