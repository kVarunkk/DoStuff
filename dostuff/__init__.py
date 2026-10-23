# Package init — lazy, doesn't pull agent loop on import
from importlib.metadata import PackageNotFoundError, version

try:
    # Single version source: pyproject.toml, via the installed distribution metadata
    __version__ = version("dostuff")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0+unknown"
