from voice_rag.connectors.parsers.text import TextLoader
from voice_rag.connectors.parsers.markdown import MarkdownLoader

__all__ = ["TextLoader", "MarkdownLoader"]

try:
    from voice_rag.connectors.parsers.pdf import PdfLoader
    __all__.append("PdfLoader")
except ImportError:
    pass

try:
    from voice_rag.connectors.parsers.docx import DocxLoader
    __all__.append("DocxLoader")
except ImportError:
    pass
