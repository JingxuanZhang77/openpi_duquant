"""Shim module for Gemma modeling outputs."""

from transformers.modeling_outputs import (  # noqa: F401
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)

