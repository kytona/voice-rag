from __future__ import annotations
from typing import Any


def format_sse(data: str) -> str:
    return f"data: {data}\n\n"


def extract_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        primary_parts: list[str] = []
        fallback_parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type in {"text", "input_text", "output_text"}:
                value = item.get("text", "")
                if value:
                    target = primary_parts if item_type == "text" else fallback_parts
                    target.append(value)
        parts = primary_parts or fallback_parts
        return "\n".join(part.strip() for part in parts if part.strip()).strip()
    return ""


def extract_latest_user_message(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            content = extract_text_content(message.get("content"))
            if content:
                return content
    raise ValueError("Request must include a user message with text content.")
