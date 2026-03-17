from unittest.mock import MagicMock

from voice_rag.agent import KnowledgeAgent
from voice_rag.core.config import VoiceRagConfig
from voice_rag.core.models import Document


def test_agent_default_creation():
    config = VoiceRagConfig()
    agent = KnowledgeAgent(config=config, _lazy_init=True)
    assert agent.config.llm.provider == "openai"
    assert agent.config.voice.provider == "elevenlabs"


def test_agent_with_string_providers():
    agent = KnowledgeAgent(llm="anthropic", voice="deepgram", _lazy_init=True)
    assert agent.config.llm.provider == "anthropic"
    assert agent.config.voice.provider == "deepgram"


def test_agent_create_app():
    agent = KnowledgeAgent(_lazy_init=True)
    app = agent.create_app()
    route_paths = [route.path for route in app.routes]
    assert "/v1/chat/completions" in route_paths
    assert "/chat/completions" in route_paths
    assert "/health" in route_paths


def test_agent_ingest_documents_uses_markdown_chunking_metadata() -> None:
    config = VoiceRagConfig()
    config.ingestion.chunk_size = 40
    config.ingestion.chunk_overlap = 0

    agent = KnowledgeAgent(config=config, _lazy_init=True)
    agent._dense_embedder = MagicMock()
    agent._sparse_embedder = MagicMock()
    agent._vector_store = MagicMock()

    agent._dense_embedder.embed.return_value = [[0.1]]
    agent._sparse_embedder.embed.return_value = [MagicMock()]
    agent._vector_store.upsert.return_value = 1

    count = agent.ingest_documents(
        [
            Document(
                source="https://docs.example.com/page",
                content="# Intro\n\nGitBook content.",
                metadata={"content_type": "text/markdown", "format": "markdown"},
            )
        ],
        recreate=True,
    )

    assert count == 1
    upsert_chunks = agent._vector_store.upsert.call_args.args[0]
    assert upsert_chunks[0].text.startswith("[# Intro]")
    assert agent._vector_store.upsert.call_args.kwargs["recreate"] is True


def test_ingest_preserves_recreate_until_non_empty_file(tmp_path) -> None:
    empty = tmp_path / "empty.md"
    empty.write_text("")
    non_empty = tmp_path / "real.md"
    non_empty.write_text("# Intro\n\nBody text.")

    agent = KnowledgeAgent(config=VoiceRagConfig(), _lazy_init=True)
    agent._dense_embedder = MagicMock()
    agent._sparse_embedder = MagicMock()
    agent._vector_store = MagicMock()

    agent._dense_embedder.embed.return_value = [[0.1]]
    agent._sparse_embedder.embed.return_value = [MagicMock()]
    agent._vector_store.upsert.return_value = 1

    count = agent.ingest(tmp_path, recreate=True)

    assert count == 1
    assert agent._vector_store.upsert.call_args.kwargs["recreate"] is True
