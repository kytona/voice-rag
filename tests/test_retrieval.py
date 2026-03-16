from voice_rag.core.retrieval import build_augmented_messages
from voice_rag.core.models import RetrievedChunk

def test_build_augmented_appends_to_system():
    messages = [{"role": "system", "content": "Be concise."}, {"role": "user", "content": "Q"}]
    chunks = [RetrievedChunk(source="doc.md", chunk_index=0, text="Answer", score=0.99)]
    result = build_augmented_messages(messages, chunks)
    assert result[0]["role"] == "system"
    assert "Be concise." in result[0]["content"]
    assert "Answer" in result[0]["content"]

def test_build_augmented_inserts_system_if_missing():
    messages = [{"role": "user", "content": "Q"}]
    chunks = [RetrievedChunk(source="doc.md", chunk_index=0, text="Answer", score=0.99)]
    result = build_augmented_messages(messages, chunks)
    assert result[0]["role"] == "system"
    assert "Answer" in result[0]["content"]

def test_build_augmented_no_chunks():
    messages = [{"role": "user", "content": "Q"}]
    assert build_augmented_messages(messages, []) == messages
