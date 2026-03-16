# voice-rag

[![PyPI version](https://img.shields.io/pypi/v/voice-rag)](https://pypi.org/project/voice-rag/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/kytona/voice-rag/blob/main/LICENSE)
[![Python](https://img.shields.io/pypi/pyversions/voice-rag)](https://pypi.org/project/voice-rag/)
[![CI](https://github.com/kytona/voice-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/kytona/voice-rag/actions/workflows/ci.yml)

**Ingest your docs. Answer questions by voice. Deploy in minutes.**

`voice-rag` is a Python library and CLI for building voice-powered RAG pipelines. Point it at a folder of documents, choose your LLM and voice provider, and get an OpenAI-compatible webhook ready to wire into ElevenLabs, Deepgram, or any voice platform.

```bash
pip install "voice-rag[elevenlabs]"
export OPENAI_API_KEY=sk-...
voice-rag init && voice-rag ingest ./docs --recreate && voice-rag serve
# → serving at http://localhost:8000/v1
```

---

## What it does

```
your docs  →  Qdrant (hybrid dense + BM25)  →  retrieved chunks
                                                       ↓
voice platform  →  speech-to-text  →  /v1/chat/completions  →  LLM  →  TTS
```

Each turn from your voice platform hits the webhook, embeds the user utterance, retrieves the most relevant chunks, injects them into the system prompt, and streams the LLM response back as SSE — all in one pip install.

---

## Install

```bash
# ElevenLabs voice + OpenAI LLM (most common)
pip install "voice-rag[elevenlabs]"

# All providers
pip install "voice-rag[all]"

# Pick only what you need
pip install "voice-rag[anthropic,pdf]"
```

| Extra | Adds |
|---|---|
| `elevenlabs` | ElevenLabs voice adapter |
| `deepgram` | Deepgram voice adapter |
| `anthropic` | Anthropic (Claude) LLM client |
| `gemini` | Google Gemini LLM client |
| `pdf` | PDF parser (PyMuPDF) |
| `docx` | Word document parser |
| `all` | Everything above |

---

## Quickstart

```bash
# 1. Create a config file
voice-rag init

# 2. Ingest your documents (supports .md, .txt, .pdf, .docx)
voice-rag ingest ./docs --recreate

# 3. Start the webhook server
voice-rag serve
```

Point your ElevenLabs agent's **Custom LLM URL** to `http://localhost:8000/v1`.

By default, vectors are stored locally in `.qdrant` — no separate Qdrant server needed. Set `vector_store.url` to connect to a remote instance.

---

## CLI reference

```bash
voice-rag init [--dir PATH]               # create voice-rag.yaml
voice-rag ingest <path> [--recreate]      # ingest a file or directory
voice-rag serve [--host] [--port] [--reload]
voice-rag query <text> [--limit N]        # test retrieval without a server
voice-rag inspect                         # show collection stats
voice-rag doctor                          # check API keys and Qdrant connectivity
```

---

## Python API

```python
from voice_rag import KnowledgeAgent, VoiceRagConfig

config = VoiceRagConfig()          # reads from voice-rag.yaml or env vars
agent = KnowledgeAgent(config=config)

agent.ingest("./docs", recreate=True)

app = agent.create_app()           # returns a FastAPI app
# run with: uvicorn app:app --port 8000
```

---

## Configuration

Config is loaded from `voice-rag.yaml` (run `voice-rag init` to generate one) or environment variables. Environment variables override the YAML file.

| Key | Env var | Default |
|---|---|---|
| `llm.provider` | `LLM_PROVIDER` | `openai` |
| `llm.model` | `LLM_MODEL` | `gpt-4o-mini` |
| `llm.api_key` / `embedding.api_key` | `OPENAI_API_KEY` | — |
| `llm.base_url` | `LLM_BASE_URL` | `https://api.openai.com/v1` |
| `embedding.model` | `EMBEDDING_MODEL` | `text-embedding-3-small` |
| `vector_store.url` | `VECTOR_STORE_URL` | empty → local `.qdrant` |
| `vector_store.collection_name` | `VECTOR_STORE_COLLECTION_NAME` | `knowledge_base` |
| `server.port` | `SERVER_PORT` | `8000` |
| `server.enable_debug_retrieval` | `SERVER_ENABLE_DEBUG_RETRIEVAL` | `false` |

See [`voice-rag.yaml`](voice-rag.yaml) for the full annotated schema.

---

## Providers

| Category | Supported |
|---|---|
| LLM | OpenAI, Anthropic, Gemini (any OpenAI-compatible URL via `llm.base_url`) |
| Voice | ElevenLabs, Deepgram |
| Embeddings | OpenAI |
| Vector store | Qdrant (local embedded or remote) |
| Parsers | `.txt`, `.md`, `.pdf`, `.docx` |

---

## Starter kit

Want a full working demo with a Next.js frontend and Railway deploy button? See [kytona/elevenlabs-knowledge-agent](https://github.com/kytona/elevenlabs-knowledge-agent) — a thin wrapper around `voice-rag` with an ElevenLabs voice UI.

---

## Development

```bash
git clone https://github.com/kytona/voice-rag
cd voice-rag
pip install -e ".[all,dev]"
pytest tests/ -v
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to add new LLM, voice, embedding, or vector store connectors.
