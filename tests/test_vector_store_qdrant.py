from types import SimpleNamespace
from unittest.mock import MagicMock
import pytest
from voice_rag.connectors.vector_stores.qdrant import QdrantStore
from voice_rag.core.models import Chunk
from qdrant_client.models import SparseVector

def test_ensure_collection_creates():
    client = MagicMock()
    client.get_collections.return_value = SimpleNamespace(collections=[])
    store = QdrantStore(client=client, collection_name="test")
    store.ensure_collection(1536)
    client.create_collection.assert_called_once()

def test_ensure_collection_raises_old_schema():
    client = MagicMock()
    client.get_collections.return_value = SimpleNamespace(collections=[SimpleNamespace(name="test")])
    client.get_collection.return_value = SimpleNamespace(config=SimpleNamespace(params=SimpleNamespace(vectors=SimpleNamespace(size=1536))))
    store = QdrantStore(client=client, collection_name="test")
    with pytest.raises(ValueError, match="old single-vector schema"):
        store.ensure_collection(1536)

def test_upsert_returns_count():
    client = MagicMock()
    client.get_collections.return_value = SimpleNamespace(collections=[SimpleNamespace(name="test")])
    client.get_collection.return_value = SimpleNamespace(config=SimpleNamespace(params=SimpleNamespace(vectors={"dense": SimpleNamespace(size=3)}, sparse_vectors={"bm25": SimpleNamespace()})))
    store = QdrantStore(client=client, collection_name="test")
    count = store.upsert([Chunk(text="hello", source="doc.md", chunk_index=0)], [[0.1, 0.2, 0.3]], [SparseVector(indices=[0], values=[1.0])])
    assert count == 1

def test_query_returns_chunks():
    client = MagicMock()
    client.query_points.return_value = SimpleNamespace(points=[SimpleNamespace(payload={"source": "doc.md", "chunk_index": 0, "text": "hello"}, score=0.9)])
    store = QdrantStore(client=client, collection_name="test")
    chunks = store.query([0.1], SparseVector(indices=[0], values=[1.0]))
    assert len(chunks) == 1
    assert chunks[0].text == "hello"

def test_collection_stats_missing():
    from qdrant_client.http.exceptions import UnexpectedResponse
    client = MagicMock()
    client.get_collection.side_effect = UnexpectedResponse(status_code=404, reason_phrase="Not Found", content=b"not found", headers={})
    store = QdrantStore(client=client, collection_name="test")
    stats = store.collection_stats()
    assert stats["collection_exists"] is False
