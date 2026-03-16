from __future__ import annotations
from typing import Any

from voice_rag.core.models import RetrievedChunk
from voice_rag.core.streaming import extract_text_content


def build_augmented_messages(
    messages: list[dict[str, Any]],
    chunks: list[RetrievedChunk],
) -> list[dict[str, Any]]:
    if not chunks:
        return messages

    context_lines = [
        "Use the following retrieved context to answer the user's question.",
        "If the answer is not supported by the context, say that clearly.",
        "",
    ]
    for chunk in chunks:
        context_lines.append(
            f"[Source: {chunk.source} | Chunk: {chunk.chunk_index} | Score: {chunk.score:.3f}] {chunk.text}"
        )
    context_block = "\n".join(context_lines)

    updated_messages = [dict(msg) for msg in messages]
    for msg in updated_messages:
        if msg.get("role") == "system":
            existing = extract_text_content(msg.get("content"))
            msg["content"] = f"{existing}\n\n{context_block}".strip()
            return updated_messages

    return [{"role": "system", "content": context_block}, *updated_messages]
