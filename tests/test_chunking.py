from voice_rag.core.chunking import chunk_text, chunk_markdown, _is_list_heavy, _fence_ranges, _parse_sections

def test_chunk_text_uses_overlap():
    text = "A" * 1000
    chunks = chunk_text(text, chunk_size=400, chunk_overlap=100)
    assert len(chunks) == 3
    assert all(chunks)

def test_chunk_markdown_empty():
    assert chunk_markdown("") == []
    assert chunk_markdown("   \n  ") == []

def test_chunk_markdown_no_headers():
    text = "Hello world. " * 100
    md_chunks = chunk_markdown(text, chunk_size=400, chunk_overlap=100)
    txt_chunks = chunk_text(text, chunk_size=400, chunk_overlap=100)
    assert len(md_chunks) == len(txt_chunks)

def test_chunk_markdown_preserves_header_context():
    text = "# Guide\n\n## Setup\n\nInstall the tool. Configure it. Run it.\n"
    chunks = chunk_markdown(text, chunk_size=800, chunk_overlap=0)
    assert any("[# Guide > ## Setup]" in chunk for chunk in chunks)

def test_chunk_markdown_list_detection():
    assert _is_list_heavy("- one\n- two\n- three\n- four\n") is True
    assert _is_list_heavy("Para one.\nPara two.\nPara three.\n") is False

def test_chunk_markdown_merges_small_sections():
    text = "# Doc\n\n## Tiny\n\nShort.\n\n## Also Tiny\n\nAlso short.\n"
    chunks = chunk_markdown(text, chunk_size=800, chunk_overlap=0)
    assert len(chunks) == 1

def test_chunk_markdown_ignores_headers_in_fenced_code():
    text = "# Real\n\nProse.\n\n```bash\n# comment\ngit clone x\n```\n\n## Another\n\nMore.\n"
    ranges = _fence_ranges(text)
    assert len(ranges) == 1
    sections = _parse_sections(text)
    titles = [s[-1] for s, _ in sections if s]
    assert "comment" not in " ".join(titles)
    assert "Real" in titles
    assert "Another" in titles
