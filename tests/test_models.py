from voice_rag.core.models import Document, Chunk, RetrievedChunk, ChatMessage

def test_document_creation():
    doc = Document(source="test.md", content="Hello world", metadata={"type": "md"})
    assert doc.source == "test.md"
    assert doc.content == "Hello world"

def test_chunk_creation():
    chunk = Chunk(text="Some text", source="test.md", chunk_index=0, metadata={})
    assert chunk.text == "Some text"

def test_retrieved_chunk_creation():
    rc = RetrievedChunk(source="test.md", chunk_index=0, text="Some text", score=0.95)
    assert rc.score == 0.95

def test_chat_message_creation():
    msg = ChatMessage(role="user", content="Hello")
    assert msg.role == "user"

def test_chat_message_with_list_content():
    msg = ChatMessage(role="user", content=[{"type": "text", "text": "Hello"}])
    assert isinstance(msg.content, list)
