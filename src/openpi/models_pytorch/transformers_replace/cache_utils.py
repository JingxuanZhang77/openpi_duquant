"""Shim cache_utils module to satisfy transformers_replace relative imports."""

from transformers.cache_utils import Cache, DynamicCache  # noqa: F401

