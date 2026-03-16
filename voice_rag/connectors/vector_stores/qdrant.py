from __future__ import annotations
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Fusion, FusionQuery, Prefetch, SparseVector
from voice_rag.core.models import Chunk, RetrievedChunk

class QdrantStore:
    def __init__(self, client: QdrantClient | None = None, collection_name: str = "knowledge_base",
                 url: str = "http://localhost:6333", in_memory: bool = False, local_path: str = ".qdrant",
                 dense_prefetch_limit: int = 20, sparse_prefetch_limit: int = 20):
        if client is not None:
            self._client = client
        elif in_memory:
            self._client = QdrantClient(path=local_path)
        else:
            self._client = QdrantClient(url=url)
        self._collection_name = collection_name
        self._dense_prefetch_limit = dense_prefetch_limit
        self._sparse_prefetch_limit = sparse_prefetch_limit

    def ensure_collection(self, vector_size: int) -> None:
        collections = self._client.get_collections().collections
        if any(c.name == self._collection_name for c in collections):
            info = self._client.get_collection(self._collection_name)
            params = getattr(getattr(info, "config", None), "params", None)
            vectors_config = getattr(params, "vectors", None)
            sparse_config = getattr(params, "sparse_vectors", None)

            # Must have named dense+bm25 schema (not legacy single-vector).
            has_named_dense = isinstance(vectors_config, dict) and "dense" in vectors_config
            if not has_named_dense:
                raise ValueError(
                    f"Collection '{self._collection_name}' exists but uses the old single-vector schema. "
                    "Re-ingest with recreate=True to upgrade to the hybrid dense+BM25 schema."
                )

            # Dense dimension must match the current embedding model.
            existing_size = getattr(vectors_config.get("dense"), "size", None)
            if existing_size is not None and existing_size != vector_size:
                raise ValueError(
                    f"Collection '{self._collection_name}' has dense vectors of size {existing_size} "
                    f"but current embedding model produces size {vector_size}. "
                    "Re-ingest with recreate=True to rebuild the collection."
                )

            # Must have the bm25 sparse index for hybrid retrieval.
            has_bm25 = isinstance(sparse_config, dict) and "bm25" in sparse_config
            if not has_bm25:
                raise ValueError(
                    f"Collection '{self._collection_name}' is missing the 'bm25' sparse vector index "
                    "required for hybrid retrieval. Re-ingest with recreate=True."
                )
            return
        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config={"dense": rest.VectorParams(size=vector_size, distance=rest.Distance.COSINE)},
            sparse_vectors_config={"bm25": rest.SparseVectorParams(modifier=rest.Modifier.IDF)},
        )

    def upsert(self, chunks: list[Chunk], dense_vectors: list[list[float]],
               sparse_vectors: list[SparseVector] | None = None, recreate: bool = False) -> int:
        if not chunks:
            return 0
        if recreate:
            self._client.recreate_collection(
                collection_name=self._collection_name,
                vectors_config={"dense": rest.VectorParams(size=len(dense_vectors[0]), distance=rest.Distance.COSINE)},
                sparse_vectors_config={"bm25": rest.SparseVectorParams(modifier=rest.Modifier.IDF)},
            )
        else:
            self.ensure_collection(len(dense_vectors[0]))
        points = []
        for i, chunk in enumerate(chunks):
            vector: dict = {"dense": dense_vectors[i]}
            if sparse_vectors:
                vector["bm25"] = sparse_vectors[i]
            points.append(rest.PointStruct(
                id=str(uuid.uuid4()), vector=vector,
                payload={"source": chunk.source, "chunk_index": chunk.chunk_index, "text": chunk.text},
            ))
        self._client.upsert(collection_name=self._collection_name, points=points)
        return len(points)

    def query(self, dense_vector: list[float], sparse_vector: SparseVector | None = None,
              limit: int = 5, score_threshold: float = 0.0) -> list[RetrievedChunk]:
        if sparse_vector is not None:
            search_result = self._client.query_points(
                collection_name=self._collection_name,
                prefetch=[
                    Prefetch(query=sparse_vector, using="bm25", limit=self._sparse_prefetch_limit),
                    Prefetch(query=dense_vector, using="dense", limit=self._dense_prefetch_limit),
                ],
                query=FusionQuery(fusion=Fusion.RRF), limit=limit,
                score_threshold=score_threshold, with_payload=True,
            )
        else:
            search_result = self._client.query_points(
                collection_name=self._collection_name, query=dense_vector, using="dense",
                limit=limit, score_threshold=score_threshold, with_payload=True,
            )
        points = getattr(search_result, "points", search_result)
        results: list[RetrievedChunk] = []
        for point in points:
            payload = point.payload or {}
            text = str(payload.get("text", ""))
            if text:
                results.append(RetrievedChunk(
                    source=str(payload.get("source", "unknown")),
                    chunk_index=int(payload.get("chunk_index", 0)),
                    text=text, score=float(getattr(point, "score", 0.0)),
                ))
        return results

    def collection_stats(self) -> dict[str, bool | int | None]:
        try:
            info = self._client.get_collection(self._collection_name)
        except UnexpectedResponse as exc:
            if getattr(exc, "status_code", None) == 404:
                return {"collection_exists": False, "points_count": 0}
            raise
        points_count = getattr(info, "points_count", None)
        if points_count is None:
            result = getattr(info, "result", None)
            points_count = getattr(result, "points_count", None)
        return {"collection_exists": True, "points_count": int(points_count) if points_count is not None else None}

    def close(self) -> None:
        self._client.close()
