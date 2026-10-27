# Package init — lazy, doesn't pull agent loop on import
import os
from importlib.metadata import PackageNotFoundError, version

# litellm fetches its price list from GitHub on import (retrying, with warnings on timeout);
# use the copy bundled with litellm instead. Set before any submodule imports litellm.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

try:
    # Single version source: pyproject.toml, via the installed distribution metadata
    __version__ = version("dostuff")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0+unknown"
