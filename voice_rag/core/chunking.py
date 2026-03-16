from __future__ import annotations

import re


def _sliding_window(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split normalized text into overlapping windows."""
    chunks: list[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(length, start + chunk_size)
        chunks.append(text[start:end].strip())
        if end >= length:
            break
        start = max(0, end - chunk_overlap)
    return [c for c in chunks if c]


def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 120) -> list[str]:
    normalized = " ".join(text.split())
    if not normalized:
        return []
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    return _sliding_window(normalized, chunk_size, chunk_overlap)


# ---------------------------------------------------------------------------
# Markdown-aware chunking
# ---------------------------------------------------------------------------

_HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)", re.MULTILINE)
_FENCE_OPEN_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})")


def _fence_ranges(text: str) -> list[tuple[int, int]]:
    """Return (start, end) char ranges for every fenced code block in *text*."""
    ranges: list[tuple[int, int]] = []
    pos = 0
    fence_char: str | None = None
    fence_len: int = 0
    fence_start: int = 0

    for line in text.splitlines(keepends=True):
        m = _FENCE_OPEN_RE.match(line)
        if fence_char is None:
            if m:
                fence_char = m.group(1)[0]
                fence_len = len(m.group(1))
                fence_start = pos
        else:
            if m and m.group(1)[0] == fence_char and len(m.group(1)) >= fence_len:
                ranges.append((fence_start, pos + len(line)))
                fence_char = None
        pos += len(line)

    if fence_char is not None:  # unclosed fence — treat rest of text as fenced
        ranges.append((fence_start, len(text)))

    return ranges


def _parse_sections(text: str) -> list[tuple[list[str], str]]:
    """
    Split markdown text into sections.
    Returns list of (header_stack, body_text) pairs.
    The first item may have an empty header_stack for content before any header.
    Header lines inside fenced code blocks are ignored.
    """
    fenced = _fence_ranges(text)

    def _in_fence(pos: int) -> bool:
        return any(start <= pos < end for start, end in fenced)

    matches = [m for m in _HEADER_RE.finditer(text) if not _in_fence(m.start())]
    if not matches:
        return [([], text)]

    sections: list[tuple[list[str], str]] = []
    # Content before the first header
    preamble = text[: matches[0].start()]
    if preamble.strip():
        sections.append(([], preamble))

    header_stack: list[tuple[int, str]] = []  # (level, title)

    for i, match in enumerate(matches):
        level = len(match.group(1))
        title = match.group(2).strip()

        # Pop headers deeper than or equal to this level
        while header_stack and header_stack[-1][0] >= level:
            header_stack.pop()
        header_stack.append((level, title))

        body_start = match.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end]

        sections.append(([t for _, t in header_stack], body))

    return sections


def _build_header_prefix(header_stack: list[str]) -> str:
    if not header_stack:
        return ""
    hashes = ["#" * (i + 1) for i in range(len(header_stack))]
    parts = " > ".join(f"{h} {t}" for h, t in zip(hashes, header_stack))
    return f"[{parts}]\n"


def _is_list_heavy(body: str) -> bool:
    lines = [line for line in body.splitlines() if line.strip()]
    if not lines:
        return False
    bullet_lines = sum(
        1 for line in lines if re.match(r"^\s*[-*+]\s+|^\s*\d+\.\s+", line)
    )
    return bullet_lines / len(lines) >= 0.5


def _split_into_bullet_items(body: str) -> list[str]:
    """Split body text into individual bullet items (preserving sub-bullets and intro prose).

    Only top-level (unindented) bullets start a new item. Indented sub-bullets
    are continuation lines that stay attached to their parent bullet.
    """
    items: list[str] = []
    current: list[str] = []
    in_bullets = False

    for line in body.splitlines():
        # Top-level bullet: no leading whitespace before the marker
        if re.match(r"^[-*+]\s+|^\d+\.\s+", line):
            if current and not in_bullets:
                # flush intro prose as its own item
                items.append("\n".join(current).strip())
                current = []
            elif current:
                items.append("\n".join(current).strip())
                current = []
            in_bullets = True
            current = [line]
        elif current:
            current.append(line)
        else:
            current.append(line)

    if current:
        items.append("\n".join(current).strip())

    return [item for item in items if item]


def _chunk_list_section(body: str, prefix: str, chunk_size: int) -> list[str]:
    items = _split_into_bullet_items(body)
    if not items:
        text = " ".join(body.split())
        return [f"{prefix}{text}"] if text else []

    chunks: list[str] = []
    group: list[str] = []
    group_len = len(prefix)

    for item in items:
        item_len = len(item) + 1  # +1 for newline
        if group and group_len + item_len > chunk_size:
            chunks.append(f"{prefix}" + "\n".join(group))
            group = [item]
            group_len = len(prefix) + item_len
        else:
            group.append(item)
            group_len += item_len

    if group:
        chunks.append(f"{prefix}" + "\n".join(group))

    return chunks


def _chunk_prose_section(body: str, prefix: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    normalized = " ".join(body.split())
    if not normalized:
        return []
    # Subtract prefix length from the window size so the total chunk fits within chunk_size.
    # Clamp overlap to stay strictly below effective_size to keep _sliding_window advancing.
    effective_size = max(1, chunk_size - len(prefix))
    effective_overlap = min(chunk_overlap, max(0, effective_size - 1))
    windows = _sliding_window(normalized, effective_size, effective_overlap)
    return [f"{prefix}{w}" for w in windows]


def _merge_small_chunks(chunks: list[str], chunk_size: int) -> list[str]:
    """Merge adjacent chunks that are both smaller than chunk_size/2."""
    if not chunks:
        return chunks
    threshold = chunk_size // 2
    merged: list[str] = [chunks[0]]
    for chunk in chunks[1:]:
        prev = merged[-1]
        if len(prev) < threshold and len(chunk) < threshold and len(prev) + len(chunk) + 1 <= chunk_size:
            merged[-1] = prev + "\n" + chunk
        else:
            merged.append(chunk)
    return merged


def chunk_markdown(text: str, chunk_size: int = 800, chunk_overlap: int = 120) -> list[str]:
    """Chunk markdown text with structure-awareness."""
    if not text.strip():
        return []
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    sections = _parse_sections(text)
    raw_chunks: list[str] = []

    for header_stack, body in sections:
        prefix = _build_header_prefix(header_stack)
        if _is_list_heavy(body):
            raw_chunks.extend(_chunk_list_section(body, prefix, chunk_size))
        else:
            raw_chunks.extend(_chunk_prose_section(body, prefix, chunk_size, chunk_overlap))

    return _merge_small_chunks([c for c in raw_chunks if c.strip()], chunk_size)
