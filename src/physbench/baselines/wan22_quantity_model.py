"""Trainable SI-aware quantity conditioning for WAN2.2.

This module deliberately injects after the frozen UMT5 encoder and before the
WAN DiT text projection. Back-propagating from a synthetic input embedding
through all 24 UMT5-XXL layers would dominate the memory cost of the small
quantity encoder. Each audited literal is therefore replaced by one native T5
sentinel in the model prompt; the sentinel's contextual vector is replaced by
``z_phys``.
"""

from __future__ import annotations

import math
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn


QUANTITY_CHECKPOINT_PREFIX = "pipe.quantity_encoder."
_DIT_LORA_KEY = re.compile(
    r"^(?P<target>.+)\.lora_(?P<side>A|B)"
    r"(?P<adapter>\.default)?\.weight$"
)
NUMERIC_FEATURE_NAMES = (
    "is_zero",
    "sign",
    "log1p_abs_si",
    "signed_log1p_abs_si",
    "tanh_si",
    "signed_base10_mantissa",
    "clipped_base10_exponent",
    "inverse_one_plus_abs_si",
)


def numeric_features(value_si: float) -> tuple[float, ...]:
    """Return fixed analytic features without Dataset-wide statistics."""

    value = float(value_si)
    if not math.isfinite(value):
        raise ValueError(f"quantity SI value must be finite, got {value!r}")
    magnitude = abs(value)
    is_zero = 1.0 if magnitude == 0.0 else 0.0
    sign = 0.0 if magnitude == 0.0 else math.copysign(1.0, value)
    log_magnitude = math.log1p(magnitude)
    if magnitude == 0.0:
        exponent = 0
        mantissa = 0.0
    else:
        exponent = math.floor(math.log10(magnitude))
        mantissa = sign * magnitude / (10.0**exponent)
    clipped_exponent = max(-12, min(12, exponent)) / 12.0
    return (
        is_zero,
        sign,
        log_magnitude,
        sign * log_magnitude,
        math.tanh(value),
        mantissa / 10.0,
        clipped_exponent,
        1.0 / (1.0 + magnitude),
    )


class QuantityEncoder(nn.Module):
    """Encode value, SI dimension and semantic kind into WAN text space."""

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.config = dict(config)
        numeric_size = int(config["numeric_feature_size"])
        if numeric_size != len(NUMERIC_FEATURE_NAMES):
            raise ValueError(
                "quantity encoder numeric_feature_size does not match "
                f"the fixed feature set ({len(NUMERIC_FEATURE_NAMES)})"
            )
        dimension_size = int(config["dimension_size"])
        if dimension_size != 7:
            raise ValueError("quantity encoder requires seven SI dimensions")
        type_count = int(config["quantity_type_count"])
        if type_count < 1:
            raise ValueError("quantity_type_count must be positive")
        numeric_hidden = int(config["numeric_hidden_size"])
        numeric_embedding = int(config["numeric_embedding_size"])
        dimension_hidden = int(config["dimension_hidden_size"])
        dimension_embedding = int(config["dimension_embedding_size"])
        type_embedding = int(config["quantity_type_embedding_size"])
        fusion_hidden = int(config["fusion_hidden_size"])
        text_hidden = int(config["text_hidden_size"])
        self.numeric_mlp = nn.Sequential(
            nn.Linear(numeric_size, numeric_hidden),
            nn.SiLU(),
            nn.Linear(numeric_hidden, numeric_embedding),
            nn.LayerNorm(numeric_embedding),
        )
        self.dimension_mlp = nn.Sequential(
            nn.Linear(dimension_size, dimension_hidden),
            nn.SiLU(),
            nn.Linear(dimension_hidden, dimension_embedding),
            nn.LayerNorm(dimension_embedding),
        )
        self.quantity_type_embedding = nn.Embedding(
            type_count,
            type_embedding,
        )
        self.fusion_mlp = nn.Sequential(
            nn.Linear(
                numeric_embedding + dimension_embedding + type_embedding,
                fusion_hidden,
            ),
            nn.SiLU(),
            nn.Linear(fusion_hidden, text_hidden),
            nn.LayerNorm(text_hidden),
        )

    def forward(
        self,
        numeric: torch.Tensor,
        dimension: torch.Tensor,
        quantity_type_id: torch.Tensor,
    ) -> torch.Tensor:
        e_num = self.numeric_mlp(numeric)
        e_dimension = self.dimension_mlp(dimension)
        e_type = self.quantity_type_embedding(quantity_type_id)
        return self.fusion_mlp(
            torch.cat((e_num, e_dimension, e_type), dim=-1)
        )

    def encode_records(
        self,
        records: Iterable[dict[str, Any]],
    ) -> torch.Tensor:
        items = list(records)
        if not items:
            raise ValueError("quantity encoder requires at least one record")
        parameter = next(self.parameters())
        device = parameter.device
        dtype = parameter.dtype
        numeric = torch.tensor(
            [numeric_features(item["si_value"]) for item in items],
            dtype=dtype,
            device=device,
        )
        dimension = torch.tensor(
            [item["dimension"] for item in items],
            dtype=dtype,
            device=device,
        )
        type_ids = torch.tensor(
            [item["quantity_type_id"] for item in items],
            dtype=torch.long,
            device=device,
        )
        if dimension.shape != (len(items), 7):
            raise ValueError(
                f"quantity dimensions must have shape ({len(items)}, 7)"
            )
        if torch.any(type_ids < 0) or torch.any(
            type_ids >= self.quantity_type_embedding.num_embeddings
        ):
            raise ValueError("quantity_type_id is outside the encoder registry")
        return self(numeric, dimension, type_ids)


def quantity_state_dict(path: str | Path) -> dict[str, torch.Tensor]:
    """Extract only quantity encoder tensors from a combined checkpoint."""

    from safetensors.torch import load_file

    return _quantity_state_dict(load_file(str(path), device="cpu"))


def _quantity_state_dict(
    state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    extracted = {
        key[len(QUANTITY_CHECKPOINT_PREFIX):]: value
        for key, value in state.items()
        if key.startswith(QUANTITY_CHECKPOINT_PREFIX)
    }
    if not extracted:
        alternate = "quantity_encoder."
        extracted = {
            key[len(alternate):]: value
            for key, value in state.items()
            if key.startswith(alternate)
        }
    return extracted


def _validate_dit_lora_state_dict(
    dit: nn.Module,
    state: dict[str, torch.Tensor],
) -> tuple[str, ...]:
    """Validate a DiffSynth/PEFT LoRA state before it can mutate the DiT."""

    pairs: dict[
        tuple[str, str],
        dict[str, tuple[str, torch.Tensor]],
    ] = {}
    invalid_keys = []
    for key, tensor in state.items():
        match = _DIT_LORA_KEY.fullmatch(key)
        if match is None:
            invalid_keys.append(key)
            continue
        pair_id = (
            match.group("target"),
            match.group("adapter") or "",
        )
        side = match.group("side")
        pair = pairs.setdefault(pair_id, {})
        if side in pair:
            raise ValueError(
                "combined checkpoint contains duplicate LoRA "
                f"{side} tensors for target {pair_id[0]!r}"
            )
        pair[side] = (key, tensor)
    if invalid_keys:
        raise ValueError(
            "combined checkpoint contains unsupported non-LoRA tensors: "
            f"{sorted(invalid_keys)[:10]}"
        )
    if not pairs:
        raise ValueError("checkpoint has no DiT LoRA tensor pairs")

    incomplete = []
    for (target, adapter), pair in sorted(pairs.items()):
        missing = sorted({"A", "B"} - set(pair))
        if missing:
            suffix = adapter or "<none>"
            incomplete.append(
                f"{target} (adapter={suffix}, missing={','.join(missing)})"
            )
    if incomplete:
        raise ValueError(
            "DiT LoRA A/B tensors are not one-to-one: "
            f"{incomplete[:10]}"
        )

    modules = dict(dit.named_modules())
    validated_targets = []
    normalized_targets: dict[str, tuple[str, str]] = {}
    for pair_id, pair in sorted(pairs.items()):
        raw_target, _adapter = pair_id
        target = (
            raw_target[len("diffusion_model."):]
            if raw_target.startswith("diffusion_model.")
            else raw_target
        )
        if target in normalized_targets:
            previous = normalized_targets[target]
            raise ValueError(
                "combined checkpoint maps more than one LoRA pair to DiT "
                f"target {target!r}: {previous!r} and {pair_id!r}"
            )
        normalized_targets[target] = pair_id
        module = modules.get(target)
        if module is None:
            raise ValueError(
                f"DiT LoRA target module does not exist: {target!r}"
            )
        weight = getattr(module, "weight", None)
        if not isinstance(weight, torch.Tensor):
            raise ValueError(
                f"DiT LoRA target module has no tensor weight: {target!r}"
            )
        if weight.ndim != 2:
            raise ValueError(
                "DiT LoRA target weight must be a matrix: "
                f"target={target!r}, shape={tuple(weight.shape)}"
            )

        tensor_a = pair["A"][1]
        tensor_b = pair["B"][1]
        if tensor_a.ndim != 2 or tensor_b.ndim != 2:
            raise ValueError(
                "DiT LoRA A/B tensors must be matrices: "
                f"target={target!r}, A={tuple(tensor_a.shape)}, "
                f"B={tuple(tensor_b.shape)}"
            )
        rank_a = int(tensor_a.shape[0])
        rank_b = int(tensor_b.shape[1])
        if rank_a < 1 or rank_a != rank_b:
            raise ValueError(
                "DiT LoRA rank mismatch: "
                f"target={target!r}, A={tuple(tensor_a.shape)}, "
                f"B={tuple(tensor_b.shape)}"
            )
        expected_a = (rank_a, int(weight.shape[1]))
        expected_b = (int(weight.shape[0]), rank_a)
        if tuple(tensor_a.shape) != expected_a or (
            tuple(tensor_b.shape) != expected_b
        ):
            raise ValueError(
                "DiT LoRA shape does not match target weight: "
                f"target={target!r}, weight={tuple(weight.shape)}, "
                f"A={tuple(tensor_a.shape)} (expected {expected_a}), "
                f"B={tuple(tensor_b.shape)} (expected {expected_b})"
            )
        validated_targets.append(target)
    return tuple(validated_targets)


def load_combined_quantity_checkpoint(
    pipe,
    encoder: QuantityEncoder,
    path: str | Path,
    *,
    lora_alpha: float,
) -> dict[str, Any]:
    """Load and strictly separate DiT LoRA and quantity-encoder tensors."""

    from safetensors.torch import load_file

    state = load_file(str(path), device="cpu")
    quantity = _quantity_state_dict(state)
    if not quantity:
        raise ValueError(
            f"checkpoint has no quantity encoder tensors: {path}"
        )
    quantity_keys = {
        key
        for key in state
        if key.startswith(QUANTITY_CHECKPOINT_PREFIX)
        or key.startswith("quantity_encoder.")
    }
    lora = {
        key: value
        for key, value in state.items()
        if key not in quantity_keys
    }
    if not lora:
        raise ValueError(f"checkpoint has no DiT LoRA tensors: {path}")
    validated_lora_targets = _validate_dit_lora_state_dict(
        pipe.dit,
        lora,
    )
    missing, unexpected = encoder.load_state_dict(quantity, strict=True)
    if missing or unexpected:
        raise ValueError(
            "quantity encoder checkpoint mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )
    pipe.load_lora(
        pipe.dit,
        state_dict=lora,
        alpha=float(lora_alpha),
    )
    return {
        "quantity_encoder_tensor_count": len(quantity),
        "dit_lora_tensor_count": len(lora),
        "dit_lora_validated_layer_count": len(validated_lora_targets),
        "dit_lora_fused_layer_count": len(validated_lora_targets),
    }


def load_quantity_encoder_checkpoint(
    encoder: QuantityEncoder,
    path: str | Path,
    *,
    required: bool,
) -> dict[str, Any]:
    state = quantity_state_dict(path)
    if not state:
        if required:
            raise ValueError(
                f"checkpoint has no quantity encoder tensors: {path}"
            )
        return {"loaded": False, "tensor_count": 0}
    missing, unexpected = encoder.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise ValueError(
            "quantity encoder checkpoint mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return {"loaded": True, "tensor_count": len(state)}


def _clean_prompt(pipe, prompt: str) -> str:
    cleaned = (
        pipe.tokenizer._clean(prompt)
        if getattr(pipe.tokenizer, "clean", None)
        else prompt
    )
    if cleaned != prompt:
        raise ValueError(
            "quantity model prompt changed under WAN tokenizer cleaning; "
            "the adapter must emit canonical single-space text"
        )
    return cleaned


def locate_sentinel_tokens(
    pipe,
    prompt: str,
    quantities: Iterable[dict[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    """Tokenize exactly as WAN does and locate one native T5 sentinel each."""

    _clean_prompt(pipe, prompt)
    ids, mask = pipe.tokenizer(
        prompt,
        return_mask=True,
        add_special_tokens=True,
    )
    if ids.shape[0] != 1:
        raise ValueError("quantity prompt embedder currently requires batch=1")
    audits: list[dict[str, Any]] = []
    seen_positions: set[int] = set()
    for quantity in quantities:
        sentinel = quantity["sentinel"]
        sentinel_id = int(
            pipe.tokenizer.tokenizer.convert_tokens_to_ids(sentinel)
        )
        if sentinel_id == int(pipe.tokenizer.tokenizer.unk_token_id):
            raise ValueError(f"unknown quantity sentinel token: {sentinel}")
        positions = torch.nonzero(
            ids[0] == sentinel_id,
            as_tuple=False,
        ).flatten().tolist()
        if len(positions) != 1:
            raise ValueError(
                f"quantity sentinel {sentinel!r} occurs {len(positions)} "
                "times after tokenization"
            )
        position = int(positions[0])
        if position in seen_positions:
            raise ValueError("quantity sentinels resolved to one token slot")
        seen_positions.add(position)
        audits.append({
            "name": quantity["name"],
            "sentinel": sentinel,
            "sentinel_token_id": sentinel_id,
            "token_span": [position, position + 1],
            "raw_value": quantity["raw_value"],
            "raw_unit": quantity["raw_unit"],
            "rendered_quantity": quantity["rendered_quantity"],
            "si_value": quantity["si_value"],
            "canonical_si_unit": quantity["canonical_si_unit"],
            "dimension": quantity["dimension"],
            "quantity_type": quantity["quantity_type"],
            "quantity_type_id": quantity["quantity_type_id"],
            "source_role": quantity["source_role"],
        })
    return ids, mask, audits


def encode_quantity_prompt(
    pipe,
    prompt: str,
    quantities: Iterable[dict[str, Any]] | None,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Encode text and replace sentinel context slots with physical vectors."""

    records = list(quantities or [])
    ids, mask, audits = locate_sentinel_tokens(pipe, prompt, records)
    ids = ids.to(pipe.device)
    mask = mask.to(pipe.device)
    sequence_lengths = mask.gt(0).sum(dim=1).long()
    with torch.no_grad():
        context = pipe.text_encoder(ids, mask)
        for index, length in enumerate(sequence_lengths):
            context[index, length:] = 0
    if records:
        physical = pipe.quantity_encoder.encode_records(records).to(
            dtype=context.dtype,
            device=context.device,
        )
        positions = torch.tensor(
            [audit["token_span"][0] for audit in audits],
            dtype=torch.long,
            device=context.device,
        )
        context = context.index_copy(
            1,
            positions,
            physical.unsqueeze(0),
        )
    return context, audits


@contextmanager
def quantity_inference_conditioning(
    pipe,
    quantities: Iterable[dict[str, Any]],
):
    """Bind quantities to the positive CFG branch for one pipeline call."""

    records = list(quantities)
    if not records:
        raise ValueError(
            "quantity inference conditioning requires at least one record"
        )
    missing = object()
    previous = getattr(pipe, "_active_quantity_conditioning", missing)
    pipe._active_quantity_conditioning = {
        "positive_quantities": records,
        "negative_quantities": None,
    }
    try:
        yield
    finally:
        if previous is missing:
            delattr(pipe, "_active_quantity_conditioning")
        else:
            pipe._active_quantity_conditioning = previous


def _resolve_cfg_quantities(
    pipe,
    inputs_posi: dict[str, Any],
    inputs_nega: dict[str, Any],
) -> tuple[
    Iterable[dict[str, Any]] | None,
    Iterable[dict[str, Any]] | None,
]:
    """Resolve explicit branch bindings without comparing prompt strings."""

    missing = object()
    positive = inputs_posi.get("quantities", missing)
    negative = inputs_nega.get("negative_quantities", missing)
    active = getattr(pipe, "_active_quantity_conditioning", None)
    if active is not None:
        if not isinstance(active, dict):
            raise TypeError(
                "active quantity conditioning must be a branch mapping"
            )
        required = {
            "positive_quantities",
            "negative_quantities",
        }
        if not required.issubset(active):
            raise ValueError(
                "active quantity conditioning must explicitly define "
                "positive_quantities and negative_quantities"
            )
        active_positive = active["positive_quantities"]
        active_negative = active["negative_quantities"]
        if positive is missing:
            positive = active_positive
        elif positive != active_positive:
            raise ValueError(
                "positive quantity bindings disagree between pipeline "
                "inputs and active inference conditioning"
            )
        if negative is missing:
            negative = active_negative
        elif negative != active_negative:
            raise ValueError(
                "negative quantity bindings disagree between pipeline "
                "inputs and active inference conditioning"
            )
    if positive is missing:
        positive = None
    if negative is missing:
        negative = None
    if negative:
        raise ValueError(
            "quantity embeddings are restricted to the positive CFG branch"
        )
    return positive, negative


def install_quantity_prompt_unit(
    pipe,
    encoder: QuantityEncoder,
) -> None:
    """Attach the encoder and replace WAN's stock prompt PipelineUnit."""

    from diffsynth.diffusion.base_pipeline import PipelineUnit

    class WanVideoUnitQuantityPromptEmbedder(PipelineUnit):
        def __init__(self):
            super().__init__(
                take_over=True,
                input_params_posi={
                    "prompt": "prompt",
                    "quantities": "quantities",
                },
                input_params_nega={
                    "prompt": "negative_prompt",
                    "quantities": "negative_quantities",
                },
                output_params=("context",),
                onload_model_names=(
                    "text_encoder",
                    "quantity_encoder",
                ),
            )

        def process(
            self,
            inner_pipe,
            inputs_shared,
            inputs_posi,
            inputs_nega,
        ):
            quantities, negative_quantities = _resolve_cfg_quantities(
                inner_pipe,
                inputs_posi,
                inputs_nega,
            )
            inner_pipe.load_models_to_device(self.onload_model_names)
            context, audit = encode_quantity_prompt(
                inner_pipe,
                inputs_posi["prompt"],
                quantities,
            )
            inputs_posi["context"] = context
            if quantities:
                inner_pipe._last_quantity_token_audit = audit
            if inputs_shared["cfg_scale"] != 1:
                negative_context, _ = encode_quantity_prompt(
                    inner_pipe,
                    inputs_nega["negative_prompt"],
                    negative_quantities,
                )
                inputs_nega["context"] = negative_context
            else:
                inputs_nega["context"] = context
            return inputs_shared, inputs_posi, inputs_nega

    pipe.quantity_encoder = encoder
    matches = [
        index
        for index, unit in enumerate(pipe.units)
        if unit.__class__.__name__ == "WanVideoUnit_PromptEmbedder"
    ]
    if matches != [2]:
        raise RuntimeError(
            "unsupported DiffSynth WAN unit layout; expected stock prompt "
            f"embedder at index 2, found {matches}"
        )
    pipe.units[2] = WanVideoUnitQuantityPromptEmbedder()


__all__ = [
    "NUMERIC_FEATURE_NAMES",
    "QUANTITY_CHECKPOINT_PREFIX",
    "QuantityEncoder",
    "encode_quantity_prompt",
    "install_quantity_prompt_unit",
    "load_combined_quantity_checkpoint",
    "load_quantity_encoder_checkpoint",
    "locate_sentinel_tokens",
    "numeric_features",
    "quantity_inference_conditioning",
    "quantity_state_dict",
]
