# Contributing to voice-rag

Thank you for contributing! This guide covers how to add new connectors.

## Project structure

```text
voice_rag/
  agent.py              # KnowledgeAgent high-level API; connector registries here
  core/
    config.py           # VoiceRagConfig pydantic model
    models.py           # Chunk, RetrievedChunk data models
    chunking.py         # Text/markdown chunking logic
    retrieval.py        # Prompt augmentation with retrieved chunks
    streaming.py        # SSE streaming helpers
  connectors/
    llm/                # LLM chat clients (openai, anthropic, gemini)
    voice/              # Voice adapters (elevenlabs, deepgram)
    embeddings/         # Embedding clients (openai)
    vector_stores/      # Vector store clients (qdrant)
    parsers/            # Document loaders (text, markdown, pdf, docx)
  cli/
    commands.py         # Click commands
    __main__.py         # CLI entry point
  server.py             # FastAPI app factory
tests/                  # pytest tests (mirror voice_rag/ structure)
```

## Adding a new LLM connector

1. Create `voice_rag/connectors/llm/<provider>.py`:

```python
from voice_rag.connectors.llm.base import BaseChatClient
from typing import Iterator

class MyProviderChatClient(BaseChatClient):
    def __init__(self, api_key: str):
        self.client = MyProviderSDK(api_key=api_key)

    def stream_chat_completion(self, messages: list[dict], model: str) -> Iterator[str]:
        # yield SSE-formatted chunks: "data: {...}\n\n"
        for chunk in self.client.stream(messages=messages, model=model):
            yield f"data: {chunk.to_json()}\n\n"
        yield "data: [DONE]\n\n"
```

2. Register it in `voice_rag/agent.py`:

```python
_LLM_CLIENTS = {
    "openai": "voice_rag.connectors.llm.openai:OpenAIChatClient",
    "anthropic": "voice_rag.connectors.llm.anthropic:AnthropicChatClient",
    "gemini": "voice_rag.connectors.llm.gemini:GeminiChatClient",
    "myprovider": "voice_rag.connectors.llm.myprovider:MyProviderChatClient",
}
```

3. Add the optional dependency in `pyproject.toml`:

```toml
[project.optional-dependencies]
myprovider = ["myprovider-sdk>=1.0.0"]
all = ["voice-rag[anthropic,gemini,deepgram,elevenlabs,myprovider,pdf,docx]"]
```

4. Add a test in `tests/test_llm_myprovider.py` (see `tests/test_llm_openai.py` for the pattern).

## Adding a new voice connector

Same pattern as LLM. Implement `BaseVoiceAdapter` from `voice_rag/connectors/voice/base.py` and register in `_VOICE_ADAPTERS` in `agent.py`.

## Adding a new vector store connector

Implement `BaseVectorStore` from `voice_rag/connectors/vector_stores/base.py`. There is no auto-registry yet — wire it manually in `KnowledgeAgent._init_components()` with a provider name check. Open a PR and we'll add registry support if needed.

## Adding a new document parser

Implement `BaseLoader` from `voice_rag/connectors/parsers/` and register the file extension in `_PARSERS` in `agent.py`.

## Running tests

**On macOS (to avoid arm64/x86_64 Rosetta issues):**

```bash
arch -arm64 .venv/bin/python -m pytest tests/ -v
```

**On Linux / CI:**

```bash
pytest tests/ -v
```

## Submitting a PR

- Branch name: `feat/<connector-name>` or `fix/<description>`
- Every new connector must have tests
- Run the full test suite before opening the PR
- Reviewers will check: does it implement the base class, is it registered, does it have tests, does it add the optional dep?
