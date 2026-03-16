"""Package version helpers."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("voice-rag")
except PackageNotFoundError:
    __version__ = "0.1.0"
