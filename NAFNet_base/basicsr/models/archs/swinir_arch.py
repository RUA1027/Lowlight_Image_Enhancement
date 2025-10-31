"""Thin wrapper that instantiates the official SwinIR architecture.

Requires the SwinIR repository to be cloned (e.g. into `external/SwinIR`) or
installed via `pip install swinir`. The Colab runbook documents the expected
clone path.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _prepare_import_path() -> None:
    current = Path(__file__).resolve()
    repo_root = current.parents[4]
    candidate_roots = [
        repo_root / "external" / "SwinIR",
        repo_root / "SwinIR",
    ]
    for candidate in candidate_roots:
        swinir_file = candidate / "models" / "swinir.py"
        if swinir_file.exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
            break


_prepare_import_path()

'''
try:
    from models.swinir import SwinIR  # type: ignore
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "Unable to import SwinIR. Please clone the official repository into 'external/SwinIR' or install it as a package."
    ) from exc
'''

try:
    from models.swinir import SwinIR  # type: ignore
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "Unable to import SwinIR. Please clone the official repository into 'external/SwinIR' or install it as a package."
    ) from exc

def SwinIRRestoration(**kwargs):
    return SwinIR(**kwargs)
