"""Utilities that keep SID paths portable between Windows/Linux environments."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Generator, Optional, Union

_SID_ROOT_CACHE: Optional[Path] = None


def _normalize_path(path_value: Union[str, os.PathLike[str]]) -> Path:
    """Expand environment markers/backslashes and return a pathlib.Path."""

    expanded = os.path.expandvars(os.fspath(path_value))
    expanded = expanded.replace("\\", "/")
    path = Path(expanded).expanduser()
    try:
        return path.resolve()
    except Exception:
        return path


def _candidate_roots() -> Generator[Path, None, None]:
    """Yield potential SID_ROOT locations ordered by confidence."""

    env_keys = ("SID_ROOT", "LOWLIGHT_ROOT")
    raw_candidates: list[Optional[Union[str, Path]]] = [os.environ.get(key) for key in env_keys]

    def _add_path_with_parents(path_obj: Path) -> None:
        raw_candidates.append(path_obj)
        raw_candidates.extend(list(path_obj.parents))

    here = Path(__file__).resolve()
    cwd = Path.cwd()
    _add_path_with_parents(here)
    _add_path_with_parents(cwd)
    raw_candidates.extend([
        Path("/root/autodl-tmp/Lowlight"),
        Path.home() / "Lowlight",
    ])

    seen: set[str] = set()
    for candidate in raw_candidates:
        if not candidate:
            continue
        if isinstance(candidate, Path):
            path = candidate
        else:
            path = _normalize_path(candidate)
        key = path.as_posix().lower()
        if key in seen:
            continue
        seen.add(key)
        yield path


def _looks_like_sid_root(path: Path) -> bool:
    if not path.is_dir():
        return False
    markers = [
        path / "SID_assets",
        path / "SID_lmdb",
        path / "SID_raw",
        path / "SID_experiments",
    ]
    present = sum(marker.exists() for marker in markers)
    return present >= 2


def detect_sid_root(force_refresh: bool = False) -> Optional[Path]:
    """Return a probable SID_ROOT directory (or ``None`` if detection fails)."""

    global _SID_ROOT_CACHE
    if not force_refresh and _SID_ROOT_CACHE is not None:
        return _SID_ROOT_CACHE

    for candidate in _candidate_roots():
        if _looks_like_sid_root(candidate):
            _SID_ROOT_CACHE = candidate
            os.environ.setdefault("SID_ROOT", str(candidate))
            return _SID_ROOT_CACHE
    return _SID_ROOT_CACHE


def expand_with_sid_root(path_value: Optional[Union[str, os.PathLike[str]]]) -> Optional[Path]:
    """Resolve ``path_value`` relative to SID_ROOT (if available)."""

    if path_value in {None, ""}:
        return None

    text = os.fspath(path_value) if isinstance(path_value, os.PathLike) else str(path_value)
    expanded = os.path.expandvars(text)
    expanded = expanded.replace("\\", "/")
    raw_path = Path(expanded).expanduser()
    if raw_path.is_absolute():
        return raw_path

    sid_root = detect_sid_root()
    base = sid_root if sid_root is not None else Path.cwd()
    try:
        return (base / raw_path).resolve()
    except Exception:
        return base / raw_path


__all__ = ["detect_sid_root", "expand_with_sid_root"]
