import asyncio, json
from unittest.mock import MagicMock, patch
from voice_rag.connectors.llm.anthropic import AnthropicChatClient

def test_stream_produces_sse():
    mock_anthropic_client = MagicMock()
    client = AnthropicChatClient(client=mock_anthropic_client)

    mock_stream = MagicMock()
    mock_stream.text_stream = iter(["Hello", " world"])
    mock_stream.__enter__ = MagicMock(return_value=mock_stream)
    mock_stream.__exit__ = MagicMock(return_value=False)

    mock_anthropic_client.messages.stream.return_value = mock_stream

    async def collect():
        return [item async for item in client.stream_chat_completion(
            messages=[{"role": "user", "content": "Hi"}], model="claude-sonnet-4-20250514",
        )]

    events = asyncio.run(collect())

    assert events[-1] == "data: [DONE]\n\n"
    assert len(events) >= 3
    first = json.loads(events[0].removeprefix("data: ").strip())
    assert first["choices"][0]["delta"]["content"] == "Hello"
