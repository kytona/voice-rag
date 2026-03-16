import pytest
from voice_rag.connectors.voice.elevenlabs import ElevenLabsAdapter

def test_parse_request():
    adapter = ElevenLabsAdapter()
    payload = {"model": "custom", "messages": [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "What is RAG?"}], "stream": True}
    messages = adapter.parse_request(payload)
    assert len(messages) == 2
    assert messages[1].content == "What is RAG?"

def test_parse_request_rejects_non_stream():
    adapter = ElevenLabsAdapter()
    with pytest.raises(ValueError, match="stream=true"):
        adapter.parse_request({"model": "custom", "messages": [{"role": "user", "content": "Hi"}], "stream": False})

def test_format_response_headers():
    assert ElevenLabsAdapter().format_response_headers()["content-type"] == "text/event-stream"
