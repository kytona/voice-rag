from __future__ import annotations
from openai import OpenAI
from fastembed import SparseTextEmbedding
from qdrant_client.models import SparseVector

class OpenAIDenseEmbedding:
    def __init__(self, client: OpenAI | None = None, model: str = "text-embedding-3-small", api_key: str = "", base_url: str = "https://api.openai.com/v1"):
        self._client = client if client is not None else OpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(model=self._model, input=texts)
        return [item.embedding for item in response.data]

class FastEmbedSparseEmbedding:
    def __init__(self, embedder: SparseTextEmbedding | None = None, model: str = "Qdrant/bm25"):
        self._embedder = embedder if embedder is not None else SparseTextEmbedding(model_name=model)

    def embed(self, texts: list[str]) -> list[SparseVector]:
        results = list(self._embedder.embed(texts))
        sparse_vectors: list[SparseVector] = []
        for result in results:
            indices = result.indices.tolist() if hasattr(result.indices, "tolist") else list(result.indices)
            values = result.values.tolist() if hasattr(result.values, "tolist") else list(result.values)
            sparse_vectors.append(SparseVector(indices=indices, values=values))
        return sparse_vectors
