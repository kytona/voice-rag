import asyncio, json
from types import SimpleNamespace
from voice_rag.connectors.llm.openai import OpenAIChatClient

class FakeAsyncStream:
    def __init__(self, chunks): self._chunks = chunks
    def __aiter__(self): self._it = iter(self._chunks); return self
    async def __anext__(self):
        try: return next(self._it)
        except StopIteration as e: raise StopAsyncIteration from e

class FakeAsyncOpenAI:
    def __init__(self, chunks):
        async def create(**_kw): return FakeAsyncStream(chunks)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))

def test_stream_produces_sse():
    upstream = [
        SimpleNamespace(model_dump=lambda mode="json": {"id": "1", "object": "chat.completion.chunk", "created": 1, "model": "gpt-4o-mini", "choices": [{"index": 0, "delta": {"content": "Hi"}, "finish_reason": None}]}),
        SimpleNamespace(model_dump=lambda mode="json": {"id": "1", "object": "chat.completion.chunk", "created": 1, "model": "gpt-4o-mini", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
    ]
    client = OpenAIChatClient(async_client=FakeAsyncOpenAI(upstream))
    events = asyncio.run(_collect(client))
    assert events[-1] == "data: [DONE]\n\n"
    assert len(events) == 3

def test_stream_synthesizes_stop():
    upstream = [SimpleNamespace(model_dump=lambda mode="json": {"id": "2", "object": "chat.completion.chunk", "created": 1, "choices": [{"index": 0, "delta": {"content": "Hi"}, "finish_reason": None}]})]
    client = OpenAIChatClient(async_client=FakeAsyncOpenAI(upstream))
    events = asyncio.run(_collect(client))
    assert events[-1] == "data: [DONE]\n\n"
    second_last = json.loads(events[-2].removeprefix("data: ").strip())
    assert second_last["choices"][0]["finish_reason"] == "stop"

async def _collect(client):
    return [item async for item in client.stream_chat_completion(messages=[{"role": "user", "content": "Hi"}], model="gpt-4o-mini")]
