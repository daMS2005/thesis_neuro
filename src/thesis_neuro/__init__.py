"""Transcript-grounded SAE and brain/LM comparison tools."""

from importlib.metadata import PackageNotFoundError, version

__all__ = ["__version__"]

try:
    __version__ = version("thesis-neuro")
except PackageNotFoundError:
    __version__ = "0+unknown"
