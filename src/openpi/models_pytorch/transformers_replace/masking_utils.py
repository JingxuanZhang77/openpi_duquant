"""Shim masking_utils module for Gemma attention compatibility."""

from transformers.masking_utils import create_causal_mask  # noqa: F401
