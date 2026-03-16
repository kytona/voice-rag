import pytest
from voice_rag.core.streaming import format_sse, extract_text_content, extract_latest_user_message

def test_format_sse():
    assert format_sse('{"key": "value"}') == 'data: {"key": "value"}\n\n'

def test_format_sse_done():
    assert format_sse("[DONE]") == "data: [DONE]\n\n"

def test_extract_text_content_string():
    assert extract_text_content("Hello world") == "Hello world"

def test_extract_text_content_list():
    content = [{"type": "text", "text": "Primary"}, {"type": "input_text", "text": "Fallback"}]
    assert extract_text_content(content) == "Primary"

def test_extract_text_content_list_fallback():
    content = [{"type": "input_text", "text": "Only fallback"}]
    assert extract_text_content(content) == "Only fallback"

def test_extract_text_content_empty():
    assert extract_text_content(None) == ""
    assert extract_text_content([]) == ""

def test_extract_latest_user_message():
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What is RAG?"},
    ]
    assert extract_latest_user_message(messages) == "What is RAG?"

def test_extract_latest_user_message_raises_when_missing():
    with pytest.raises(ValueError, match="user message"):
        extract_latest_user_message([{"role": "assistant", "content": "No user"}])
