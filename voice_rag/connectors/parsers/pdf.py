from __future__ import annotations
from pathlib import Path
from voice_rag.core.models import Document

class PdfLoader:
    supported_extensions = [".pdf"]
    def load(self, path: Path) -> Document:
        try:
            import pymupdf
        except ImportError as e:
            raise ImportError("pymupdf is required for PDF parsing. Install with: pip install voice-rag[pdf]") from e
        doc = pymupdf.open(str(path))
        text_parts = [page.get_text() for page in doc]
        doc.close()
        return Document(source=str(path), content="\n".join(text_parts))
