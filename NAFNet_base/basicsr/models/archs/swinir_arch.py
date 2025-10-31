"""Thin wrapper that instantiates the official SwinIR architecture.

Requires the SwinIR repository to be cloned (e.g. into `external/SwinIR`) or
installed via `pip install swinir`. The Colab runbook documents the expected
clone path.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _prepare_import_path() -> None:
    """Add SwinIR repo (or its models folder) into sys.path if found.

    Supports both of the following layouts:
    - external/SwinIR/models/network_swinir.py  (official repo)
    - external/SwinIR/models/swinir.py          (some forks)
    """
    current = Path(__file__).resolve()
    # Be defensive: fall back to the nearest parent if depth is shallow
    repo_root = current.parents[4] if len(current.parents) >= 5 else current.parents[-1]
    candidate_roots = [
        repo_root / "external" / "SwinIR",
        repo_root / "SwinIR",
    ]
    for candidate in candidate_roots:
        models_dir = candidate / "models"
        # Prefer adding the models directory when using the official layout
        if (models_dir / "network_swinir.py").exists():
            if str(models_dir) not in sys.path:
                sys.path.insert(0, str(models_dir))
            return
        # Legacy/fork layout
        if (models_dir / "swinir.py").exists():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return


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
    # Official repo path when models directory is on sys.path
    from network_swinir import SwinIR  # type: ignore
except Exception:
    try:
        # When the repository root is on sys.path
        from models.network_swinir import SwinIR  # type: ignore
    except Exception as exc2:  # pragma: no cover - optional dependency
        try:
            # Some forks use models/swinir.py
            from models.swinir import SwinIR  # type: ignore
        except Exception as exc3:  # pragma: no cover
            raise ImportError(
                "Unable to import SwinIR. Please clone the official repository into 'external/SwinIR' or install it as a package."
            ) from exc3

def SwinIRRestoration(**kwargs):
    return SwinIR(**kwargs)
