"""Compatibility helpers for typing across Python versions."""

try:
    from typing import TypeAlias
except ImportError:  # pragma: no cover - Python <3.10
    from typing_extensions import TypeAlias  # type: ignore

__all__ = ["TypeAlias"]
