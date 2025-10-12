#!/usr/bin/env python3
"""Test script to verify vision_tower layers are correctly excluded from DuQuant."""

import re

# Test layer names
test_layers = [
    # Should be EXCLUDED (vision tower)
    ("paligemma_with_expert.paligemma.model.vision_tower.encoder.layers.0.self_attn.q_proj", "vision", False),
    ("paligemma_with_expert.paligemma.model.vision_tower.encoder.layers.0.self_attn.k_proj", "vision", False),
    ("paligemma_with_expert.paligemma.model.multi_modal_projector.linear", "projector", False),
    
    # Should be INCLUDED (LLM)
    ("paligemma_with_expert.paligemma.model.language_model.layers.0.self_attn.q_proj", "llm", True),
    ("paligemma_with_expert.paligemma.model.language_model.layers.0.self_attn.k_proj", "llm", True),
    ("paligemma_with_expert.paligemma.model.language_model.layers.0.mlp.gate_proj", "llm", True),
    
    # Should be INCLUDED (DiT)
    ("paligemma_with_expert.gemma_expert.model.layers.0.self_attn.q_proj", "dit", True),
    ("paligemma_with_expert.gemma_expert.model.layers.0.mlp.gate_proj", "dit", True),
    
    # Should be EXCLUDED (embeddings and lm_head)
    ("paligemma_with_expert.paligemma.model.language_model.embed_tokens", "embed", False),
    ("paligemma_with_expert.paligemma.lm_head", "lm_head", False),
    
    # Should be EXCLUDED (normalization)
    ("paligemma_with_expert.paligemma.model.language_model.layers.0.input_layernorm", "norm", False),
]

# DuQuant filters from run_full_llm_dit_w4a8.sh
scope_prefix = "paligemma_with_expert."
include_regex = r'.*(q_proj|k_proj|v_proj|o_proj|out_proj|gate_proj|up_proj|down_proj).*'
exclude_regex = r'(?:^|\.)(norm|ln|layernorm|emb|embed|vision_tower|vision|multi_modal_projector|lm_head)(?:\.|$)'

inc = re.compile(include_regex)
exc = re.compile(exclude_regex)

print("=" * 80)
print("Vision Exclusion Test for DuQuant")
print("=" * 80)
print(f"\nScope: {scope_prefix}")
print(f"Include: {include_regex}")
print(f"Exclude: {exclude_regex}")
print("\n" + "=" * 80)

failures = []
vision_excluded_count = 0
llm_included_count = 0
dit_included_count = 0

for layer_name, category, should_include in test_layers:
    # Check scope
    if not layer_name.startswith(scope_prefix):
        actual_result = "out_of_scope"
        match_str = "❌ OUT OF SCOPE"
    # Check exclude (should NOT match for included layers)
    elif exc.search(layer_name):
        actual_result = "excluded"
        match_str = "❌ EXCLUDED"
        if category == "vision":
            vision_excluded_count += 1
    # Check include (should match for included layers)
    elif inc.search(layer_name):
        actual_result = "included"
        match_str = "✅ INCLUDED"
        if category == "llm":
            llm_included_count += 1
        elif category == "dit":
            dit_included_count += 1
    else:
        actual_result = "not_matched"
        match_str = "❌ NOT MATCHED"
    
    # Check if result matches expectation
    is_included = (actual_result == "included")
    if is_included != should_include:
        status = "❌ FAIL"
        failures.append((layer_name, should_include, is_included))
    else:
        status = "✅ PASS"
    
    # Truncate long names
    short_name = layer_name
    if len(short_name) > 60:
        short_name = "..." + short_name[-57:]
    
    print(f"{status} [{category:10s}] {match_str:15s} {short_name}")

print("=" * 80)
print(f"\n📊 Summary:")
print(f"  Vision layers excluded: {vision_excluded_count}/2")
print(f"  LLM layers included: {llm_included_count}/3")
print(f"  DiT layers included: {dit_included_count}/2")
print()

if failures:
    print(f"❌ FAILED {len(failures)} tests:")
    for name, expected, actual in failures:
        exp_str = "INCLUDED" if expected else "EXCLUDED"
        act_str = "INCLUDED" if actual else "EXCLUDED"
        print(f"  {name}")
        print(f"    Expected: {exp_str}, Got: {act_str}")
    exit(1)
else:
    print("✅ ALL TESTS PASSED!")
    print("✅ Vision layers are correctly excluded")
    print("✅ LLM and DiT layers are correctly included")
    exit(0)
