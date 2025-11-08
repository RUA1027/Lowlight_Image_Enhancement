"""Project-local basicsr package proxying NAFNet_base/basicsr."""

from __future__ import annotations

import pathlib

_PKG_DIR = pathlib.Path(__file__).resolve().parent
_REAL_PKG_DIR = _PKG_DIR.parent / "NAFNet_base" / "basicsr"
_REAL_INIT = _REAL_PKG_DIR / "__init__.py"

if not _REAL_PKG_DIR.exists():
    raise ImportError(f"Expected basicsr sources at {_REAL_PKG_DIR}, but the directory is missing.")

__file__ = str(_REAL_INIT)
__path__ = [str(_REAL_PKG_DIR)]

if __spec__ is not None:  # pragma: no cover - defensive for exotic runtimes
    __spec__.origin = __file__
    __spec__.submodule_search_locations = __path__

_code = compile(_REAL_INIT.read_text(encoding="utf-8"), __file__, "exec")
exec(_code, globals(), globals())
