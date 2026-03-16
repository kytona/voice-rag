from __future__ import annotations

import os
import sys
from pathlib import Path

import click
import yaml

from voice_rag.agent import KnowledgeAgent


DEFAULT_CONFIG = {
    "llm": {"provider": "openai", "model": "gpt-4o-mini"},
    "voice": {"provider": "elevenlabs"},
    "embedding": {"provider": "openai", "model": "text-embedding-3-small"},
    "vector_store": {
        "provider": "qdrant",
        "collection_name": "knowledge_base",
        "in_memory": True,
    },
    "ingestion": {"chunk_size": 800, "chunk_overlap": 120},
    "server": {"host": "0.0.0.0", "port": 8000},
}


def _load_config(config_path: str | None):
    from voice_rag.core.config import VoiceRagConfig
    if config_path:
        return VoiceRagConfig.from_yaml(config_path)
    default_yaml = Path("voice-rag.yaml")   # hyphen, not underscore
    if default_yaml.exists():
        return VoiceRagConfig.from_yaml(default_yaml)
    return VoiceRagConfig()


def _format_serve_url(host: str, port: int) -> str:
    display_host = "localhost" if host == "0.0.0.0" else host
    return f"http://{display_host}:{port}/v1"


@click.command()
@click.option("--dir", "directory", default=".", help="Directory to create config in.")
def init_cmd(directory: str):
    """Initialize a voice-rag project with a config file."""
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)
    config_path = dir_path / "voice-rag.yaml"

    if config_path.exists():
        click.echo(f"Config already exists at {config_path}")
        return

    with open(config_path, "w") as f:
        yaml.dump(DEFAULT_CONFIG, f, default_flow_style=False, sort_keys=False)

    click.echo(f"Created config at {config_path}")
    click.echo("Edit the file to set your API keys and provider preferences.")


@click.command()
@click.argument("path")
@click.option("--recreate", is_flag=True, help="Recreate the collection before ingesting.")
@click.option("--config", "config_path", default=None, help="Path to voice-rag.yaml config file.")
def ingest_cmd(path: str, recreate: bool, config_path: str | None):
    """Ingest documents from a file or directory."""
    import logging

    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)

    config = _load_config(config_path)
    agent = KnowledgeAgent(config=config)

    try:
        total = agent.ingest(path, recreate=recreate)
        click.echo(f"Total: {total} chunks")
    except (FileNotFoundError, ValueError) as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@click.command()
@click.option("--host", default=None, help="Host to bind to.")
@click.option("--port", default=None, type=int, help="Port to listen on.")
@click.option("--voice", "voice_provider", default=None, help="Voice provider (elevenlabs, deepgram).")
@click.option("--reload", is_flag=True, help="Enable auto-reload (dev mode).")
@click.option("--config", "config_path", default=None, help="Path to voice-rag.yaml config file.")
def serve_cmd(host: str | None, port: int | None, voice_provider: str | None, reload: bool, config_path: str | None):
    """Start the voice RAG webhook server."""
    import uvicorn

    config = _load_config(config_path)
    if voice_provider:
        config.voice.provider = voice_provider

    agent = KnowledgeAgent(config=config)
    app = agent.create_app()

    final_host = host or config.server.host
    final_port = port or config.server.port
    url = _format_serve_url(final_host, final_port)
    click.echo(f"voice-rag serving at {url}  (voice={config.voice.provider}, llm={config.llm.provider})")
    uvicorn.run(app, host=final_host, port=final_port, reload=reload)


@click.command()
@click.option("--config", "config_path", default=None, help="Path to voice-rag.yaml config file.")
def inspect_cmd(config_path: str | None):
    """Show collection stats and configuration."""
    config = _load_config(config_path)
    agent = KnowledgeAgent(config=config)

    stats = agent._vector_store.collection_stats()
    click.echo(f"Collection: {config.vector_store.collection_name}")
    click.echo(f"Exists: {stats.get('collection_exists', False)}")
    click.echo(f"Points: {stats.get('points_count', 0)}")
    click.echo(f"LLM: {config.llm.provider} ({config.llm.model})")
    click.echo(f"Voice: {config.voice.provider}")
    click.echo(f"Embeddings: {config.embedding.provider} ({config.embedding.model})")


@click.command()
def doctor_cmd():
    """Check environment and connectivity."""
    checks = {
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
        "VOICE_RAG_LLM_API_KEY": os.environ.get("VOICE_RAG_LLM_API_KEY"),
    }

    for key, value in checks.items():
        if value:
            click.echo(f"  [OK] {key} is set")
        else:
            click.echo(f"  [--] {key} not set")

    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url="http://localhost:6333", timeout=3)
        client.get_collections()
        click.echo("  [OK] Qdrant reachable at localhost:6333")
    except Exception:
        click.echo("  [--] Qdrant not reachable at localhost:6333")

    config_path = Path("voice-rag.yaml")
    if config_path.exists():
        click.echo(f"  [OK] Config file found: {config_path}")
    else:
        click.echo("  [--] No voice-rag.yaml found (run: voice-rag init)")


@click.command()
@click.argument("text")
@click.option("--config", "config_path", default=None, help="Path to voice-rag.yaml config file.")
@click.option("--limit", default=None, type=int, help="Number of chunks to return.")
def query_cmd(text: str, config_path: str | None, limit: int | None):
    """Run a retrieval query against the vector store (no server required)."""
    config = _load_config(config_path)
    if limit is not None:
        config.vector_store.retrieval_limit = limit

    agent = KnowledgeAgent(config=config)
    chunks = agent.query(text)

    if not chunks:
        click.echo("No results found.")
        sys.exit(1)

    for i, chunk in enumerate(chunks, start=1):
        preview = chunk.text[:300] + "..." if len(chunk.text) > 300 else chunk.text
        click.echo(f"[{i}] score={chunk.score:.2f}  source={chunk.source}")
        click.echo(preview)
        click.echo("---")
