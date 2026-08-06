"""Symbol-semantic and SI-value cross-attention for WAN2.2 text context."""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Any, Iterable

import torch
from torch import nn


SYMBOL_VALUE_CHECKPOINT_PREFIX = "pipe.symbol_value_conditioner."
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
    """Return scale-robust analytic features in canonical SI units."""

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
    return (
        is_zero,
        sign,
        log_magnitude,
        sign * log_magnitude,
        math.tanh(value),
        mantissa / 10.0,
        max(-12, min(12, exponent)) / 12.0,
        1.0 / (1.0 + magnitude),
    )


def _validate_records(
    records: Iterable[dict[str, Any]],
    *,
    unit_count: int | None = None,
) -> list[dict[str, Any]]:
    items = list(records)
    if not items:
        raise ValueError("symbol-value conditioning requires at least one record")
    for index, item in enumerate(items):
        symbol = item.get("symbol")
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError(f"record {index} requires a non-empty symbol")
        dimension = item.get("dimension")
        if (
            not isinstance(dimension, list)
            or len(dimension) != 7
            or any(isinstance(value, bool) or not isinstance(value, int)
                   for value in dimension)
        ):
            raise ValueError(
                f"record {index} dimension must contain seven integer exponents"
            )
        try:
            value_si = float(item["si_value"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"record {index} requires numeric si_value") from exc
        if not math.isfinite(value_si):
            raise ValueError(f"record {index} si_value must be finite")
        unit_id = item.get("unit_id")
        if isinstance(unit_id, bool) or not isinstance(unit_id, int):
            raise ValueError(f"record {index} unit_id must be an integer")
        if unit_count is not None and not 0 <= unit_id < unit_count:
            raise ValueError(f"record {index} unit_id is outside the registry")
    return items


def pool_symbol_embeddings(
    tokenizer,
    token_embedding: nn.Embedding,
    records: Iterable[dict[str, Any]],
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Mean-pool frozen UMT5 subword embeddings for each physical symbol."""

    items = _validate_records(records)
    symbols = [item["symbol"].strip() for item in items]
    backend = getattr(tokenizer, "tokenizer", tokenizer)
    encoded = backend(
        symbols,
        add_special_tokens=False,
        padding=True,
        truncation=False,
        return_attention_mask=True,
        return_tensors="pt",
    )
    ids = encoded["input_ids"]
    mask = encoded["attention_mask"]
    if ids.ndim != 2 or mask.shape != ids.shape or ids.shape[0] != len(items):
        raise ValueError("symbol tokenizer returned an invalid batch shape")
    if torch.any(mask.sum(dim=1) < 1):
        raise ValueError("every symbol must tokenize to at least one subword")
    embedding_device = token_embedding.weight.device
    ids_device = ids.to(device=embedding_device)
    mask_device = mask.to(device=embedding_device)
    with torch.no_grad():
        token_vectors = token_embedding(ids_device)
        weights = mask_device.to(dtype=token_vectors.dtype).unsqueeze(-1)
        pooled = (token_vectors * weights).sum(dim=1) / weights.sum(dim=1)
        pooled = pooled.detach()
    audits = []
    for index, item in enumerate(items):
        active = mask[index].bool()
        audits.append({
            "name": item.get("name"),
            "symbol": item["symbol"],
            "symbol_token_ids": ids[index][active].tolist(),
            "symbol_subword_count": int(active.sum().item()),
        })
    return pooled, audits


class SymbolValueConditioner(nn.Module):
    """Fuse lexical symbols with SI values and cross-attend text to them."""

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.config = dict(config)
        numeric_size = int(config["numeric_feature_size"])
        if numeric_size != len(NUMERIC_FEATURE_NAMES):
            raise ValueError(
                "numeric_feature_size does not match the fixed feature set"
            )
        dimension_size = int(config["dimension_size"])
        if dimension_size != 7:
            raise ValueError("conditioner requires seven SI dimensions")
        text_hidden = int(config["text_hidden_size"])
        attention_hidden = int(config["attention_hidden_size"])
        attention_heads = int(config["attention_heads"])
        if attention_hidden % attention_heads:
            raise ValueError("attention_hidden_size must divide attention_heads")
        unit_count = int(config["unit_count"])
        if unit_count < 1:
            raise ValueError("unit_count must be positive")
        dropout = float(config.get("dropout", 0.0))
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        numeric_embedding = int(config["numeric_embedding_size"])
        dimension_embedding = int(config["dimension_embedding_size"])
        unit_embedding = int(config["unit_embedding_size"])
        self.numeric_mlp = nn.Sequential(
            nn.Linear(numeric_size, int(config["numeric_hidden_size"])),
            nn.SiLU(),
            nn.Linear(int(config["numeric_hidden_size"]), numeric_embedding),
            nn.LayerNorm(numeric_embedding),
        )
        self.dimension_mlp = nn.Sequential(
            nn.Linear(dimension_size, int(config["dimension_hidden_size"])),
            nn.SiLU(),
            nn.Linear(int(config["dimension_hidden_size"]), dimension_embedding),
            nn.LayerNorm(dimension_embedding),
        )
        self.unit_embedding = nn.Embedding(unit_count, unit_embedding)
        value_input = numeric_embedding + dimension_embedding + unit_embedding
        self.value_mlp = nn.Sequential(
            nn.Linear(value_input, int(config["value_fusion_hidden_size"])),
            nn.SiLU(),
            nn.Linear(int(config["value_fusion_hidden_size"]), text_hidden),
            nn.LayerNorm(text_hidden),
        )
        self.symbol_norm = nn.LayerNorm(text_hidden)
        self.fusion_norm = nn.LayerNorm(text_hidden)
        self.text_norm = nn.LayerNorm(text_hidden)
        self.physics_norm = nn.LayerNorm(text_hidden)
        self.query_projection = nn.Linear(text_hidden, attention_hidden)
        self.physics_projection = nn.Linear(text_hidden, attention_hidden)
        self.cross_attention = nn.MultiheadAttention(
            attention_hidden,
            attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.output_projection = nn.Linear(attention_hidden, text_hidden)
        self.output_norm = nn.LayerNorm(text_hidden)
        gate_init = float(config.get("residual_gate_init", 0.1))
        if not math.isfinite(gate_init):
            raise ValueError("residual_gate_init must be finite")
        self.residual_gate = nn.Parameter(torch.tensor(gate_init))

    def encode_value_records(
        self,
        records: Iterable[dict[str, Any]],
    ) -> torch.Tensor:
        items = _validate_records(
            records,
            unit_count=self.unit_embedding.num_embeddings,
        )
        parameter = next(self.parameters())
        numeric = torch.tensor(
            [numeric_features(item["si_value"]) for item in items],
            dtype=parameter.dtype,
            device=parameter.device,
        )
        dimensions = torch.tensor(
            [item["dimension"] for item in items],
            dtype=parameter.dtype,
            device=parameter.device,
        )
        unit_ids = torch.tensor(
            [item["unit_id"] for item in items],
            dtype=torch.long,
            device=parameter.device,
        )
        value = self.value_mlp(torch.cat((
            self.numeric_mlp(numeric),
            self.dimension_mlp(dimensions),
            self.unit_embedding(unit_ids),
        ), dim=-1))
        return value

    def build_physics_tokens(
        self,
        symbol_embeddings: torch.Tensor,
        records: Iterable[dict[str, Any]],
    ) -> torch.Tensor:
        items = list(records)
        value_embeddings = self.encode_value_records(items)
        symbols = symbol_embeddings.to(
            device=value_embeddings.device,
            dtype=value_embeddings.dtype,
        )
        if symbols.shape != value_embeddings.shape:
            raise ValueError(
                "symbol/value embedding shape mismatch: "
                f"symbol={tuple(symbols.shape)}, value={tuple(value_embeddings.shape)}"
            )
        tokens = self.fusion_norm(
            self.symbol_norm(symbols) + value_embeddings
        )
        if not self.training and not bool(torch.isfinite(tokens).all()):
            raise FloatingPointError("physical tokens contain non-finite values")
        return tokens

    def forward(
        self,
        text_context: torch.Tensor,
        text_mask: torch.Tensor,
        symbol_embeddings: torch.Tensor,
        records: Iterable[dict[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if text_context.ndim != 3 or text_context.shape[0] != 1:
            raise ValueError("text context must have shape [1, sequence, hidden]")
        if text_mask.shape != text_context.shape[:2]:
            raise ValueError("text mask shape must match text context sequence")
        physics = self.build_physics_tokens(symbol_embeddings, records).unsqueeze(0)
        queries = self.query_projection(self.text_norm(text_context.float()))
        keys_values = self.physics_projection(self.physics_norm(physics))
        attended, weights = self.cross_attention(
            queries,
            keys_values,
            keys_values,
            need_weights=True,
            average_attn_weights=False,
        )
        residual = self.output_projection(attended)
        output = self.output_norm(
            text_context.float() + self.residual_gate * residual
        )
        output = output.masked_fill(~text_mask.bool().unsqueeze(-1), 0.0)
        if not self.training and not bool(torch.isfinite(output).all()):
            raise FloatingPointError("conditioned text contains non-finite values")
        return output, weights


def encode_symbol_value_prompt(
    pipe,
    prompt: str,
    quantities: Iterable[dict[str, Any]] | None,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    records = list(quantities or [])
    ids, mask = pipe.tokenizer(
        prompt,
        return_mask=True,
        add_special_tokens=True,
    )
    if ids.shape[0] != 1:
        raise ValueError("symbol-value prompt embedder requires batch=1")
    ids = ids.to(pipe.device)
    mask = mask.to(pipe.device)
    with torch.no_grad():
        context = pipe.text_encoder(ids, mask)
        context = context.masked_fill(~mask.bool().unsqueeze(-1), 0.0)
    if not records:
        return context, []
    symbols, audits = pool_symbol_embeddings(
        pipe.tokenizer,
        pipe.text_encoder.token_embedding,
        records,
    )
    context, attention = pipe.symbol_value_conditioner(
        context,
        mask,
        symbols,
        records,
    )
    context = context.to(dtype=pipe.text_encoder.token_embedding.weight.dtype)
    mean_attention = attention.detach().float().mean(dim=(0, 1, 2))
    for index, audit in enumerate(audits):
        item = records[index]
        audit.update({
            "raw_value": item.get("raw_value"),
            "raw_unit": item.get("raw_unit"),
            "si_value": item["si_value"],
            "canonical_si_unit": item.get("canonical_si_unit"),
            "dimension": item["dimension"],
            "unit_id": item["unit_id"],
            "mean_cross_attention": float(mean_attention[index].item()),
        })
    return context, audits


@contextmanager
def symbol_value_inference_conditioning(
    pipe,
    quantities: Iterable[dict[str, Any]],
):
    records = _validate_records(quantities)
    missing = object()
    previous = getattr(pipe, "_active_symbol_value_conditioning", missing)
    pipe._active_symbol_value_conditioning = {
        "positive_quantities": records,
        "negative_quantities": None,
    }
    try:
        yield
    finally:
        if previous is missing:
            delattr(pipe, "_active_symbol_value_conditioning")
        else:
            pipe._active_symbol_value_conditioning = previous


def _resolve_cfg_quantities(
    pipe,
    inputs_posi: dict[str, Any],
    inputs_nega: dict[str, Any],
) -> tuple[Iterable[dict[str, Any]] | None, None]:
    missing = object()
    positive = inputs_posi.get("quantities", missing)
    negative = inputs_nega.get("negative_quantities", missing)
    active = getattr(pipe, "_active_symbol_value_conditioning", None)
    if active is not None:
        if not isinstance(active, dict) or set(active) != {
            "positive_quantities", "negative_quantities"
        }:
            raise ValueError("active symbol-value conditioning is malformed")
        if positive is missing:
            positive = active["positive_quantities"]
        elif positive != active["positive_quantities"]:
            raise ValueError("positive symbol-value bindings disagree")
        if negative is missing:
            negative = active["negative_quantities"]
        elif negative != active["negative_quantities"]:
            raise ValueError("negative symbol-value bindings disagree")
    if positive is missing:
        positive = None
    if negative is missing:
        negative = None
    if negative:
        raise ValueError("symbol-value tokens are restricted to positive CFG")
    return positive, None


def install_symbol_value_prompt_unit(
    pipe,
    conditioner: SymbolValueConditioner,
) -> None:
    from diffsynth.diffusion.base_pipeline import PipelineUnit

    class WanVideoUnitSymbolValuePromptEmbedder(PipelineUnit):
        def __init__(self):
            super().__init__(
                take_over=True,
                input_params_posi={"prompt": "prompt", "quantities": "quantities"},
                input_params_nega={
                    "prompt": "negative_prompt",
                    "quantities": "negative_quantities",
                },
                output_params=("context",),
                onload_model_names=("text_encoder", "symbol_value_conditioner"),
            )

        def process(self, inner_pipe, inputs_shared, inputs_posi, inputs_nega):
            quantities, negative = _resolve_cfg_quantities(
                inner_pipe, inputs_posi, inputs_nega
            )
            inner_pipe.load_models_to_device(self.onload_model_names)
            context, audit = encode_symbol_value_prompt(
                inner_pipe, inputs_posi["prompt"], quantities
            )
            inputs_posi["context"] = context
            if quantities:
                inner_pipe._last_symbol_value_token_audit = audit
            if inputs_shared["cfg_scale"] != 1:
                negative_context, _ = encode_symbol_value_prompt(
                    inner_pipe, inputs_nega["negative_prompt"], negative
                )
                inputs_nega["context"] = negative_context
            else:
                inputs_nega["context"] = context
            return inputs_shared, inputs_posi, inputs_nega

    pipe.symbol_value_conditioner = conditioner
    matches = [
        index for index, unit in enumerate(pipe.units)
        if unit.__class__.__name__ == "WanVideoUnit_PromptEmbedder"
    ]
    if matches != [2]:
        raise RuntimeError(
            "unsupported DiffSynth WAN unit layout; expected prompt embedder "
            f"at index 2, found {matches}"
        )
    pipe.units[2] = WanVideoUnitSymbolValuePromptEmbedder()


__all__ = [
    "NUMERIC_FEATURE_NAMES",
    "SYMBOL_VALUE_CHECKPOINT_PREFIX",
    "SymbolValueConditioner",
    "encode_symbol_value_prompt",
    "install_symbol_value_prompt_unit",
    "numeric_features",
    "pool_symbol_embeddings",
    "symbol_value_inference_conditioning",
]
