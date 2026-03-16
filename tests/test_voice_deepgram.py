from voice_rag.connectors.voice.deepgram import DeepgramAdapter

def test_parse_request():
    adapter = DeepgramAdapter()
    payload = {"messages": [{"role": "user", "content": "Hello from Deepgram"}], "stream": True}
    messages = adapter.parse_request(payload)
    assert len(messages) == 1
    assert messages[0].content == "Hello from Deepgram"

def test_format_response_headers():
    assert DeepgramAdapter().format_response_headers()["content-type"] == "text/event-stream"
