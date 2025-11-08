"""Lightweight registry system for this BasicSR fork."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional


class Registry:
    """Simple name → object mapping with decorator-friendly API."""

    def __init__(self, name: str):
        self._name = name
        self._obj_map: Dict[str, Any] = {}

    def _register_impl(self, obj: Any, name: Optional[str] = None, suffix: Optional[str] = None) -> Any:
        key = name or getattr(obj, "__name__", None)
        if key is None:
            raise KeyError(f"{self._name} registry: cannot infer name for object {obj!r}")

        if suffix:
            key = f"{key}_{suffix}"

        if key in self._obj_map:
            raise KeyError(f"{self._name} registry: '{key}' already registered with {self._obj_map[key]!r}")

        self._obj_map[key] = obj
        return obj

    def register(
        self,
        obj: Optional[Any] = None,
        *,
        name: Optional[str] = None,
        suffix: Optional[str] = None,
    ) -> Callable[[Any], Any] | Any:
        """Support decorator and direct-call registration."""

        if obj is not None:
            return self._register_impl(obj, name=name, suffix=suffix)

        def decorator(fn_or_cls: Any) -> Any:
            return self._register_impl(fn_or_cls, name=name, suffix=suffix)

        return decorator

    def get(self, key: str) -> Any:
        return self._obj_map.get(key)

    def __contains__(self, key: str) -> bool:  # pragma: no cover - convenience
        return key in self._obj_map

    def keys(self):  # pragma: no cover - debug helper
        return self._obj_map.keys()

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"{self.__class__.__name__}(name={self._name}, items={list(self._obj_map.keys())})"


# Global registries mirroring official BasicSR convention.
DATASET_REGISTRY = Registry("dataset")
ARCH_REGISTRY = Registry("arch")
MODEL_REGISTRY = Registry("model")
LOSS_REGISTRY = Registry("loss")
METRIC_REGISTRY = Registry("metric")
