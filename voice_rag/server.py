from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from voice_rag.core.retrieval import build_augmented_messages
from voice_rag.core.streaming import extract_latest_user_message

if TYPE_CHECKING:
    from voice_rag.agent import KnowledgeAgent

logger = logging.getLogger(__name__)


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="custom")
    messages: list[dict[str, Any]]
    stream: bool = True


def create_app(agent: KnowledgeAgent) -> FastAPI:
    app = FastAPI(title="voice-rag", version="0.1.0")

    @app.get("/health")
    def health():
        payload: dict[str, Any] = {"status": "ok"}
        if agent._vector_store:
            payload.update(agent._vector_store.collection_stats())
        return payload

    @app.get("/debug/retrieval")
    def debug_retrieval(q: str, limit: int | None = None, score_threshold: float | None = None):
        if not agent.config.server.enable_debug_retrieval:
            raise HTTPException(status_code=404, detail="Not found")

        effective_limit = limit or agent.config.vector_store.retrieval_limit
        effective_threshold = score_threshold if score_threshold is not None else agent.config.vector_store.score_threshold

        dense_vec = agent._dense_embedder.embed([q])[0]
        sparse_vec = agent._sparse_embedder.embed([q])[0]
        chunks = agent._vector_store.query(
            dense_vector=dense_vec, sparse_vector=sparse_vec,
            limit=effective_limit, score_threshold=effective_threshold,
        )
        return {
            "query": q, "collection": agent.config.vector_store.collection_name,
            "limit": effective_limit, "score_threshold": effective_threshold,
            "matches": [c.model_dump(mode="json") for c in chunks],
        }

    async def handle_chat_completion(payload: ChatCompletionRequest):
        if not payload.stream:
            raise HTTPException(status_code=400, detail="Only stream=true is supported.")

        try:
            user_query = extract_latest_user_message(payload.messages)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        try:
            chunks = agent.query(user_query)
        except Exception as exc:
            logger.warning("Retrieval failed, falling back to base model: %s", exc)
            chunks = []

        augmented_messages = build_augmented_messages(payload.messages, chunks)
        # Use the model from the request if it's meaningful, fall back to configured default.
        # ElevenLabs sends "custom" as the model name, so we treat that as "use config default".
        effective_model = agent.config.llm.model if payload.model in ("", "custom") else payload.model
        stream = agent._llm_client.stream_chat_completion(
            messages=augmented_messages, model=effective_model,
        )
        return StreamingResponse(stream, media_type="text/event-stream")

    @app.post("/v1/chat/completions")
    async def chat_completions(payload: ChatCompletionRequest):
        return await handle_chat_completion(payload)

    @app.post("/chat/completions")
    async def chat_completions_compat(payload: ChatCompletionRequest):
        return await handle_chat_completion(payload)

    return app
