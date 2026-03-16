from types import SimpleNamespace
from unittest.mock import MagicMock
import numpy as np
from voice_rag.connectors.embeddings.openai import OpenAIDenseEmbedding, FastEmbedSparseEmbedding
from qdrant_client.models import SparseVector

def test_dense_embedding():
    mock = MagicMock()
    mock.embeddings.create.return_value = SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3])])
    embedder = OpenAIDenseEmbedding(client=mock, model="text-embedding-3-small")
    result = embedder.embed(["hello"])
    assert result == [[0.1, 0.2, 0.3]]

def test_sparse_embedding():
    result = SimpleNamespace(indices=np.array([0, 5, 10]), values=np.array([0.3, 0.7, 0.1]))
    mock = MagicMock()
    mock.embed.return_value = iter([result])
    embedder = FastEmbedSparseEmbedding(embedder=mock)
    vectors = embedder.embed(["hello"])
    assert len(vectors) == 1
    assert isinstance(vectors[0], SparseVector)
    assert vectors[0].indices == [0, 5, 10]
