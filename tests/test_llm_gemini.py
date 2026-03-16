import asyncio, json
from unittest.mock import MagicMock, patch
from voice_rag.connectors.llm.gemini import GeminiChatClient

def test_stream_produces_sse():
    client = GeminiChatClient(api_key="test-key")

    mock_chunk1 = MagicMock()
    mock_chunk1.text = "Hello"
    mock_chunk2 = MagicMock()
    mock_chunk2.text = " world"
    mock_response = MagicMock()
    mock_response.__iter__ = MagicMock(return_value=iter([mock_chunk1, mock_chunk2]))

    mock_model = MagicMock()
    mock_model.generate_content_stream.return_value = mock_response

    with patch.object(client, "_get_model", return_value=mock_model):
        async def collect():
            return [item async for item in client.stream_chat_completion(
                messages=[{"role": "user", "content": "Hi"}], model="gemini-2.0-flash",
            )]

        events = asyncio.run(collect())

    assert events[-1] == "data: [DONE]\n\n"
    first = json.loads(events[0].removeprefix("data: ").strip())
    assert first["choices"][0]["delta"]["content"] == "Hello"
