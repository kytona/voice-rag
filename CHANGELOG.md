# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-03-16

### Added

- `KnowledgeAgent` high-level Python API: `ingest()`, `query()`, `create_app()`
- CLI: `voice-rag init`, `ingest`, `serve`, `query`, `inspect`, `doctor`
- Voice adapters: ElevenLabs, Deepgram
- LLM clients: OpenAI, Anthropic, Gemini (with streaming)
- Embedding client: OpenAI (`text-embedding-3-small` default)
- Vector store: Qdrant with hybrid dense + BM25 retrieval
- Document parsers: `.txt`, `.md`, `.pdf` (via PyMuPDF), `.docx`
- `VoiceRagConfig` with YAML and environment variable support
- FastAPI webhook serving OpenAI-style `/v1/chat/completions` (SSE streaming)
- `voice-rag.yaml` annotated example config
- GitHub Actions CI (Python 3.11, 3.12) and publish pipeline (TestPyPI → PyPI)
