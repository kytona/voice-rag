# voice-rag

[![PyPI version](https://img.shields.io/pypi/v/voice-rag)](https://pypi.org/project/voice-rag/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/pypi/pyversions/voice-rag)](https://pypi.org/project/voice-rag/)

`voice-rag` is a Python package for voice-oriented RAG pipelines: ingest local documents, store them in Qdrant, serve an OpenAI-style chat-completions webhook, and swap voice or LLM providers behind a stable interface.

## What it includes

- Python SDK via `KnowledgeAgent`
- CLI for `init`, `ingest`, `serve`, `query`, `inspect`, and `doctor`
- Built-in voice adapters for `elevenlabs` and `deepgram`
- Built-in LLM clients for `openai`, `anthropic`, and `gemini`
- Built-in parsers for `.txt`, `.md`, `.pdf`, and `.docx`
- Qdrant vector store integration with hybrid dense + BM25 retrieval

## Quickstart

```bash
pip install "voice-rag[elevenlabs]"
voice-rag init
```

Edit `voice-rag.yaml`, then ingest and serve:

```bash
voice-rag ingest ./data/sample_docs/claude-code-changelog.md --recreate
voice-rag serve
```

Point your voice platform to `http://localhost:8000/v1` if it expects an OpenAI-style Custom LLM endpoint.

## CLI

```bash
voice-rag init [--dir PATH]
voice-rag ingest <path> [--recreate] [--config PATH]
voice-rag serve [--host HOST] [--port PORT] [--reload] [--config PATH]
voice-rag query <text> [--limit N] [--config PATH]
voice-rag inspect [--config PATH]
voice-rag doctor
```

## Python API

```python
from voice_rag import KnowledgeAgent, VoiceRagConfig

config = VoiceRagConfig()
agent = KnowledgeAgent(config=config)
agent.ingest("./docs", recreate=True)
app = agent.create_app()
```

## Configuration

`voice-rag` reads configuration from `voice-rag.yaml` or environment variables.

| Key | Env | Default |
| --- | --- | --- |
| `llm.api_key` / `embedding.api_key` | `OPENAI_API_KEY` | (required for OpenAI) |
| `llm.provider` | `LLM_PROVIDER` | `openai` |
| `llm.model` | `LLM_MODEL` | `gpt-4o-mini` |
| `embedding.model` | `EMBEDDING_MODEL` | `text-embedding-3-small` |
| `vector_store.collection_name` | `VECTOR_STORE_COLLECTION_NAME` | `knowledge_base` |
| `vector_store.local_path` | `VECTOR_STORE_LOCAL_PATH` | `.qdrant` |
| `server.port` | `SERVER_PORT` | `8000` |

See [voice-rag.yaml](voice-rag.yaml) for the full schema.

## Development

```bash
pip install -e ".[all,dev]"
pytest tests/ -v
```

Use [CONTRIBUTING.md](CONTRIBUTING.md) for connector and packaging guidelines.
