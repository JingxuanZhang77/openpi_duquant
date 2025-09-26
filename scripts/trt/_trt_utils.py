"""Shared helpers for TensorRT export/build workflows."""
from __future__ import annotations

import os
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.append(str(_SRC_DIR))


@dataclass(frozen=True)
class TrtSample:
    tensors: OrderedDict[str, torch.Tensor]
    pkv_structure: tuple[int, ...]


def _import_openpi():
    try:
        from openpi.models import model as _model  # noqa: WPS433
        from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks  # noqa: WPS433
        from openpi.policies import policy_config as _policy_config  # noqa: WPS433
        from openpi.training import config as _config  # noqa: WPS433
    except ModuleNotFoundError as exc:  # pragma: no cover - import guard
        missing = exc.name or "dependency"
        raise RuntimeError(
            f"Missing dependency '{missing}'. Install the OpenPI package and its requirements before running this script."
        ) from exc
    return _model, make_att_2d_masks, _policy_config, _config


def load_policy(config_name: str, checkpoint: str, device: str, *, duquant_scope: str | None = None) -> torch.nn.Module:
    _model, _, _policy_config, _config = _import_openpi()
    if duquant_scope:
        os.environ.setdefault("OPENPI_DUQUANT_SCOPE", duquant_scope)
    train_config = _config.get_config(config_name)
    policy = _policy_config.create_trained_policy(train_config, checkpoint, pytorch_device=device)
    policy._model.eval()  # noqa: SLF001
    return policy


def load_calibration_file(path: Path, *, batch_size: int | None = None, device: torch.device | None = None) -> dict:
    batch = torch.load(path)

    def _slice(value):
        if isinstance(value, dict):
            return {k: _slice(v) for k, v in value.items()}
        if isinstance(value, torch.Tensor):
            tensor = value if batch_size is None else value[:batch_size]
            if device is not None:
                tensor = tensor.to(device)
            return tensor
        return value

    return _slice(batch)


def build_observation(batch: dict, device: torch.device):
    _model, *_ = _import_openpi()
    observation_dict = {
        "image": batch["images"],
        "image_mask": batch["image_masks"],
        "state": batch["states"],
    }
    tokenized = batch.get("tokenized_prompts")
    if tokenized is not None:
        observation_dict["tokenized_prompt"] = tokenized
    token_masks = batch.get("tokenized_prompt_masks")
    if token_masks is not None:
        observation_dict["tokenized_prompt_mask"] = token_masks

    def _to_device(tree):
        if isinstance(tree, dict):
            return {k: _to_device(v) for k, v in tree.items()}
        if isinstance(tree, torch.Tensor):
            return tree.to(device)
        return tree

    observation_dict = _to_device(observation_dict)
    return _model.Observation.from_dict(observation_dict)


def prepare_trt_sample(
    policy,
    observation,
    *,
    device: torch.device,
    noise: torch.Tensor | None = None,
    time_value: float = 1.0,
) -> TrtSample:
    _, make_att_2d_masks, _, _ = _import_openpi()

    model = policy._model  # noqa: SLF001
    images, img_masks, lang_tokens, lang_masks, state = model._preprocess_observation(observation, train=False)
    images = [img.to(device) for img in images]
    img_masks = [mask.to(device) for mask in img_masks]
    lang_tokens = lang_tokens.to(device)
    lang_masks = lang_masks.to(device)
    state = state.to(device)

    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
    prefix_embs = prefix_embs.to(device)
    prefix_pad_masks = prefix_pad_masks.to(device)
    prefix_att_masks = prefix_att_masks.to(device)

    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    prefix_att_2d_masks_4d = model._prepare_attention_masks_4d(prefix_att_2d_masks)

    model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
    _, past_key_values = model.paligemma_with_expert.forward(
        attention_mask=prefix_att_2d_masks_4d,
        position_ids=prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=True,
    )

    bsize = state.shape[0]
    if noise is None:
        actions_shape = (bsize, model.config.action_horizon, model.config.action_dim)
        noise = model.sample_noise(actions_shape, device)
    timestep = torch.full((bsize,), time_value, dtype=torch.float32, device=device)

    suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = model.embed_suffix(state, noise, timestep)
    suffix_embs = suffix_embs.to(torch.float32)
    suffix_pad_masks = suffix_pad_masks.to(device)
    suffix_att_masks = suffix_att_masks.to(device)
    if adarms_cond is None:
        raise RuntimeError("AdaRMS conditioning tensor is missing; expected pi0.5 configuration")
    adarms_cond = adarms_cond.to(torch.float32)

    suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

    prefix_len = prefix_pad_masks.shape[1]
    suffix_len = suffix_pad_masks.shape[1]
    prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(bsize, suffix_len, prefix_len)
    full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
    full_att_2d_masks_4d = model._prepare_attention_masks_4d(full_att_2d_masks).to(torch.float32)

    position_ids = prefix_pad_masks.sum(dim=-1, keepdim=True).to(torch.int64) + torch.cumsum(suffix_pad_masks, dim=1) - 1
    position_ids = position_ids.to(torch.int64)

    flat_pkv: list[torch.Tensor] = []
    structure: list[int] = []
    for layer in past_key_values:
        tensors = []
        if isinstance(layer, (tuple, list)):
            tensors = [t.to(torch.float32) for t in layer]
        else:
            tensors = [layer.to(torch.float32)]
        structure.append(len(tensors))
        flat_pkv.extend(tensors)

    tensors = OrderedDict[str, torch.Tensor]()
    tensors["suffix_embs"] = suffix_embs.cpu().contiguous()
    tensors["attention_mask"] = full_att_2d_masks_4d.cpu().contiguous()
    tensors["position_ids"] = position_ids.cpu().contiguous()
    tensors["adarms_cond"] = adarms_cond.cpu().contiguous()

    for idx, tensor in enumerate(flat_pkv):
        tensors[f"pkv_{idx}"] = tensor.cpu().contiguous()

    return TrtSample(tensors=tensors, pkv_structure=tuple(structure))


def iter_calibration_files(calibration_dir: Path) -> Iterator[Path]:
    for path in sorted(calibration_dir.glob("batch_*.pt")):
        yield path
