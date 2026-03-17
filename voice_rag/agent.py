from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any

from voice_rag.core.config import VoiceRagConfig
from voice_rag.core.models import Chunk, Document, RetrievedChunk

logger = logging.getLogger(__name__)

_VOICE_ADAPTERS = {
    "elevenlabs": "voice_rag.connectors.voice.elevenlabs:ElevenLabsAdapter",
    "deepgram": "voice_rag.connectors.voice.deepgram:DeepgramAdapter",
}
_LLM_CLIENTS = {
    "openai": "voice_rag.connectors.llm.openai:OpenAIChatClient",
    "anthropic": "voice_rag.connectors.llm.anthropic:AnthropicChatClient",
    "gemini": "voice_rag.connectors.llm.gemini:GeminiChatClient",
}
_PARSERS = {
    ".txt": "voice_rag.connectors.parsers.text:TextLoader",
    ".md": "voice_rag.connectors.parsers.markdown:MarkdownLoader",
    ".pdf": "voice_rag.connectors.parsers.pdf:PdfLoader",
    ".docx": "voice_rag.connectors.parsers.docx:DocxLoader",
}


def _supported_providers(values: dict[str, str]) -> str:
    return ", ".join(sorted(values))


def _import_class(dotted_path: str) -> type:
    module_path, class_name = dotted_path.rsplit(":", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


class KnowledgeAgent:
    """High-level API for voice-rag.

    Usage:
        agent = KnowledgeAgent(llm="openai", voice="elevenlabs")
        agent.ingest("./docs/")
        app = agent.create_app()
    """

    def __init__(
        self,
        config: VoiceRagConfig | None = None,
        llm: str | None = None,
        voice: str | None = None,
        vector_store: str | None = None,
        _lazy_init: bool = False,
    ):
        if config is None:
            config = VoiceRagConfig()
        if llm:
            config.llm.provider = llm
        if voice:
            config.voice.provider = voice
        if vector_store:
            config.vector_store.provider = vector_store

        self.config = config
        self._voice_adapter = None
        self._llm_client = None
        self._dense_embedder = None
        self._sparse_embedder = None
        self._vector_store = None

        if not _lazy_init:
            self._init_components()

    def _init_components(self) -> None:
        # Voice adapter
        adapter_path = _VOICE_ADAPTERS.get(self.config.voice.provider)
        if not adapter_path:
            raise ValueError(
                f"Unsupported voice provider '{self.config.voice.provider}'. "
                f"Supported: {_supported_providers(_VOICE_ADAPTERS)}."
            )
        cls = _import_class(adapter_path)
        self._voice_adapter = cls()

        # LLM client
        llm_path = _LLM_CLIENTS.get(self.config.llm.provider)
        if not llm_path:
            raise ValueError(
                f"Unsupported llm provider '{self.config.llm.provider}'. "
                f"Supported: {_supported_providers(_LLM_CLIENTS)}."
            )
        cls = _import_class(llm_path)
        if self.config.llm.provider == "openai":
            self._llm_client = cls(api_key=self.config.llm.api_key, base_url=self.config.llm.base_url)
        else:
            self._llm_client = cls(api_key=self.config.llm.api_key)

        # Embeddings
        if self.config.embedding.provider != "openai":
            raise ValueError(
                f"Unsupported embedding provider '{self.config.embedding.provider}'. Supported: openai."
            )
        from voice_rag.connectors.embeddings.openai import OpenAIDenseEmbedding, FastEmbedSparseEmbedding
        self._dense_embedder = OpenAIDenseEmbedding(
            api_key=self.config.embedding.api_key,
            base_url=self.config.embedding.base_url,
            model=self.config.embedding.model,
        )
        self._sparse_embedder = FastEmbedSparseEmbedding(model=self.config.ingestion.bm25_model)

        # Vector store
        if self.config.vector_store.provider != "qdrant":
            raise ValueError(
                f"Unsupported vector store provider '{self.config.vector_store.provider}'. Supported: qdrant."
            )
        from voice_rag.connectors.vector_stores.qdrant import QdrantStore
        self._vector_store = QdrantStore(
            collection_name=self.config.vector_store.collection_name,
            url=self.config.vector_store.url,
            local_path=self.config.vector_store.local_path,
            dense_prefetch_limit=self.config.vector_store.dense_prefetch_limit,
            sparse_prefetch_limit=self.config.vector_store.sparse_prefetch_limit,
        )

    def _get_loader(self, suffix: str):
        parser_path = _PARSERS.get(suffix)
        if not parser_path:
            raise ValueError(f"No parser for '{suffix}'. Supported: {list(_PARSERS.keys())}")
        cls = _import_class(parser_path)
        return cls()

    def _document_is_markdown(self, doc: Document) -> bool:
        content_type = str(doc.metadata.get("content_type", "")).lower()
        doc_format = str(doc.metadata.get("format", "")).lower()
        return doc.source.lower().endswith(".md") or "markdown" in content_type or doc_format == "markdown"

    def _build_chunks(self, doc: Document) -> list[Chunk]:
        from voice_rag.core.chunking import chunk_markdown, chunk_text

        chunk_size = self.config.ingestion.chunk_size
        chunk_overlap = self.config.ingestion.chunk_overlap

        if self._document_is_markdown(doc):
            text_chunks = chunk_markdown(doc.content, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        else:
            text_chunks = chunk_text(doc.content, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        return [
            Chunk(text=text, source=doc.source, chunk_index=index, metadata=dict(doc.metadata))
            for index, text in enumerate(text_chunks)
        ]

    def ingest_documents(self, documents: list[Document], recreate: bool = False) -> int:
        total = 0
        should_recreate = recreate

        for doc in documents:
            chunks = self._build_chunks(doc)
            if not chunks:
                continue

            dense_vecs = self._dense_embedder.embed([chunk.text for chunk in chunks])
            sparse_vecs = self._sparse_embedder.embed([chunk.text for chunk in chunks])
            count = self._vector_store.upsert(chunks, dense_vecs, sparse_vecs, recreate=should_recreate)
            should_recreate = False
            total += count

        return total

    def ingest(self, path: str | Path, recreate: bool = False) -> int:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Path not found: {path}")

        if path.is_file():
            files = [path]
        else:
            supported = set(_PARSERS.keys())
            files = sorted(f for f in path.rglob("*") if f.suffix.lower() in supported)

        if not files:
            raise ValueError("No supported documents found.")

        total = 0
        should_recreate = recreate

        for file_path in files:
            loader = self._get_loader(file_path.suffix.lower())
            doc = loader.load(file_path)
            count = self.ingest_documents([doc], recreate=should_recreate)
            if count > 0:
                should_recreate = False
            total += count
            logger.info("Ingested %d chunks from %s", count, file_path)

        return total

    def query(self, text: str) -> list[RetrievedChunk]:
        dense_vec = self._dense_embedder.embed([text])[0]
        sparse_vec = self._sparse_embedder.embed([text])[0]
        return self._vector_store.query(
            dense_vector=dense_vec,
            sparse_vector=sparse_vec,
            limit=self.config.vector_store.retrieval_limit,
            score_threshold=self.config.vector_store.score_threshold,
        )

    def create_app(self):
        from voice_rag.server import create_app
        return create_app(self)
