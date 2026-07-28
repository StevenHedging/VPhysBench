"""Trainable SI-aware quantity conditioning for WAN2.2.

This module deliberately injects after the frozen UMT5 encoder and before the
WAN DiT text projection. Back-propagating from a synthetic input embedding
through all 24 UMT5-XXL layers would dominate the memory cost of the small
quantity encoder. Each audited literal is therefore replaced by one native T5
sentinel in the model prompt; the sentinel's contextual vector is replaced by
``z_phys``.
"""

from __future__ import annotations

import errno
import hashlib
import math
import os
import re
import stat
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn

from ..io import canonical_sha256, load_json


QUANTITY_CHECKPOINT_PREFIX = "pipe.quantity_encoder."
WAN22_TI2V_5B_BLOCK_COUNT = 30
WAN22_TI2V_5B_LORA_RANK = 32
WAN22_TI2V_5B_LORA_PAIR_COUNT = 300
WAN22_TI2V_5B_LORA_TENSOR_COUNT = 600
QUANTITY_ENCODER_TENSOR_COUNT = 19
WAN22_TI2V_5B_LORA_TARGET_SUFFIXES = (
    "cross_attn.k",
    "cross_attn.o",
    "cross_attn.q",
    "cross_attn.v",
    "ffn.0",
    "ffn.2",
    "self_attn.k",
    "self_attn.o",
    "self_attn.q",
    "self_attn.v",
)
_PER_CALL_GENERATION_FIELDS = frozenset({
    "cfg_scale",
    "fps",
    "height",
    "negative_prompt",
    "num_frames",
    "num_inference_steps",
    "quality",
    "tiled",
    "width",
})
QUANTITY_ENCODER_STATE_KEYS = frozenset({
    "dimension_mlp.0.bias",
    "dimension_mlp.0.weight",
    "dimension_mlp.2.bias",
    "dimension_mlp.2.weight",
    "dimension_mlp.3.bias",
    "dimension_mlp.3.weight",
    "fusion_mlp.0.bias",
    "fusion_mlp.0.weight",
    "fusion_mlp.2.bias",
    "fusion_mlp.2.weight",
    "fusion_mlp.3.bias",
    "fusion_mlp.3.weight",
    "numeric_mlp.0.bias",
    "numeric_mlp.0.weight",
    "numeric_mlp.2.bias",
    "numeric_mlp.2.weight",
    "numeric_mlp.3.bias",
    "numeric_mlp.3.weight",
    "quantity_type_embedding.weight",
})
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


def expected_wan22_ti2v_5b_lora_targets() -> frozenset[str]:
    """Return the frozen LoRA target topology for WAN2.2-TI2V-5B."""

    return frozenset(
        f"blocks.{block}.{suffix}"
        for block in range(WAN22_TI2V_5B_BLOCK_COUNT)
        for suffix in WAN22_TI2V_5B_LORA_TARGET_SUFFIXES
    )


def distributed_sampling_contract(
    dataset_length: int,
    world_size: int,
) -> dict[str, Any]:
    """Return a no-padding DDP sampling contract or fail before training."""

    if (
        isinstance(dataset_length, bool)
        or not isinstance(dataset_length, int)
        or dataset_length < 1
    ):
        raise ValueError("training dataset length must be a positive integer")
    if (
        isinstance(world_size, bool)
        or not isinstance(world_size, int)
        or world_size < 1
    ):
        raise ValueError("distributed world size must be a positive integer")
    remainder = dataset_length % world_size
    if remainder:
        raise ValueError(
            "training dataset length must be divisible by distributed world "
            "size to prevent Accelerate even-batch padding and cross-rank "
            f"duplicates: dataset_length={dataset_length}, "
            f"world_size={world_size}, remainder={remainder}"
        )
    return {
        "dataset_length": dataset_length,
        "distributed_world_size": world_size,
        "samples_per_rank_per_epoch": dataset_length // world_size,
        "shard_strategy": (
            "common_seed_random_permutation_then_accelerate_"
            "batch_sampler_shard"
        ),
        "even_batch_padding_required": False,
        "cross_rank_duplicate_policy": "forbidden",
    }


def quantity_pipeline_shared_config(job: dict[str, Any]) -> dict[str, Any]:
    """Canonicalize state that a persistent worker loads exactly once."""

    checkpoint = job.get("checkpoint")
    manifest = job.get("checkpoint_manifest")
    if not isinstance(checkpoint, str) or not checkpoint:
        raise ValueError("quantity job requires a checkpoint path")
    if not isinstance(manifest, str) or not manifest:
        raise ValueError("quantity job requires a checkpoint manifest path")
    wan22 = job.get("wan22")
    if not isinstance(wan22, dict):
        raise ValueError("quantity job requires wan22 configuration")
    runtime = wan22.get("runtime")
    encoder = wan22.get("quantity_encoder")
    generation = wan22.get("generation")
    if not isinstance(runtime, dict):
        raise ValueError("quantity job requires wan22.runtime")
    if not isinstance(encoder, dict):
        raise ValueError("quantity job requires wan22.quantity_encoder")
    if not isinstance(generation, dict):
        raise ValueError("quantity job requires wan22.generation")
    try:
        alpha = float(generation.get("lora_alpha", 1.0))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("quantity job LoRA alpha must be numeric") from exc
    if not math.isfinite(alpha):
        raise ValueError(f"LoRA alpha must be finite, got {alpha!r}")
    shared_generation = {
        key: value
        for key, value in generation.items()
        if key not in _PER_CALL_GENERATION_FIELDS
    }
    shared_generation["lora_alpha"] = alpha
    return {
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_manifest": str(Path(manifest).resolve()),
        "runtime": runtime,
        "quantity_encoder": encoder,
        "pipeline_generation": shared_generation,
    }


def quantity_pipeline_shared_fingerprint(job: dict[str, Any]) -> str:
    return canonical_sha256(quantity_pipeline_shared_config(job))


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
        for index, item in enumerate(items):
            dimension_value = item.get("dimension")
            if (
                not isinstance(dimension_value, list)
                or len(dimension_value) != 7
                or any(
                    not isinstance(value, int) or isinstance(value, bool)
                    for value in dimension_value
                )
            ):
                raise ValueError(
                    "quantity dimension must contain seven finite integer "
                    f"exponents: record={index}"
                )
            type_value = item.get("quantity_type_id")
            if not isinstance(type_value, int) or isinstance(type_value, bool):
                raise ValueError(
                    "quantity_type_id must be an integer: "
                    f"record={index}"
                )
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
        encoded = self(numeric, dimension, type_ids)
        if (
            not self.training
            and not bool(torch.isfinite(encoded.detach()).all())
        ):
            raise FloatingPointError(
                "QuantityEncoder produced non-finite physical embeddings"
            )
        return encoded


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


def _validate_quantity_encoder_state_dict(
    encoder: QuantityEncoder,
    state: dict[str, torch.Tensor],
) -> None:
    if len(state) != QUANTITY_ENCODER_TENSOR_COUNT:
        raise ValueError(
            "quantity encoder checkpoint must contain exactly "
            f"{QUANTITY_ENCODER_TENSOR_COUNT} tensors, got {len(state)}"
        )
    if set(state) != QUANTITY_ENCODER_STATE_KEYS:
        missing = sorted(QUANTITY_ENCODER_STATE_KEYS - set(state))
        unexpected = sorted(set(state) - QUANTITY_ENCODER_STATE_KEYS)
        raise ValueError(
            "quantity encoder checkpoint topology mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )
    expected = encoder.state_dict()
    if set(expected) != QUANTITY_ENCODER_STATE_KEYS:
        raise RuntimeError(
            "QuantityEncoder implementation no longer matches the frozen "
            "19-tensor checkpoint topology"
        )
    for key, tensor in state.items():
        if tuple(tensor.shape) != tuple(expected[key].shape):
            raise ValueError(
                "quantity encoder checkpoint shape mismatch: "
                f"{key}={tuple(tensor.shape)}, "
                f"expected={tuple(expected[key].shape)}"
            )
        if not tensor.is_floating_point():
            raise ValueError(
                f"quantity encoder tensor must be floating point: {key}"
            )
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(
                f"quantity encoder tensor contains non-finite values: {key}"
            )


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
    if len(state) != WAN22_TI2V_5B_LORA_TENSOR_COUNT:
        raise ValueError(
            "WAN2.2-TI2V-5B checkpoint must contain exactly "
            f"{WAN22_TI2V_5B_LORA_TENSOR_COUNT} LoRA tensors, "
            f"got {len(state)}"
        )
    if len(pairs) != WAN22_TI2V_5B_LORA_PAIR_COUNT:
        raise ValueError(
            "WAN2.2-TI2V-5B checkpoint must contain exactly "
            f"{WAN22_TI2V_5B_LORA_PAIR_COUNT} LoRA A/B pairs, "
            f"got {len(pairs)}"
        )

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
    normalized_pairs: dict[
        str,
        dict[str, tuple[str, torch.Tensor]],
    ] = {}
    for pair_id, pair in pairs.items():
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
        normalized_pairs[target] = pair

    expected_targets = expected_wan22_ti2v_5b_lora_targets()
    actual_targets = set(normalized_targets)
    if actual_targets != expected_targets:
        missing = sorted(expected_targets - actual_targets)
        unexpected = sorted(actual_targets - expected_targets)
        raise ValueError(
            "WAN2.2-TI2V-5B LoRA target topology is incomplete or "
            f"unexpected: missing={missing[:10]}, "
            f"unexpected={unexpected[:10]}"
        )

    for target in sorted(expected_targets):
        pair = normalized_pairs[target]
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
        if (
            rank_a != WAN22_TI2V_5B_LORA_RANK
            or rank_b != WAN22_TI2V_5B_LORA_RANK
        ):
            raise ValueError(
                "DiT LoRA rank must be exactly "
                f"{WAN22_TI2V_5B_LORA_RANK}: "
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
        for side, tensor in (("A", tensor_a), ("B", tensor_b)):
            if not tensor.is_floating_point():
                raise ValueError(
                    "DiT LoRA tensor must be floating point: "
                    f"target={target!r}, side={side}"
                )
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(
                    "DiT LoRA tensor contains non-finite values: "
                    f"target={target!r}, side={side}"
                )
        validated_targets.append(target)
    return tuple(validated_targets)


def _checkpoint_stat_identity(
    value: os.stat_result,
) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


@contextmanager
def verified_quantity_checkpoint_bytes(
    checkpoint_path: str | Path,
    manifest_path: str | Path,
) -> Iterable[tuple[bytes, dict[str, Any]]]:
    """Yield one authenticated buffer while its descriptor stays stable."""

    checkpoint_argument = Path(checkpoint_path)
    checkpoint_open_path = Path(
        os.path.abspath(os.fspath(checkpoint_argument))
    )
    manifest = Path(manifest_path).resolve()
    if not manifest.is_file():
        raise FileNotFoundError(
            f"quantity checkpoint manifest not found: {manifest}"
        )
    value = load_json(manifest)
    if not isinstance(value, dict):
        raise ValueError("quantity checkpoint manifest must be an object")
    if value.get("schema_version") != "2.0":
        raise ValueError(
            "quantity checkpoint manifest must use schema_version=2.0"
        )
    if value.get("status") not in {"complete", "not_requested"}:
        raise ValueError(
            "quantity checkpoint manifest does not authorize inference: "
            f"status={value.get('status')!r}"
        )
    recorded_checkpoint = value.get("checkpoint")
    if not isinstance(recorded_checkpoint, str) or not recorded_checkpoint:
        raise ValueError(
            "quantity checkpoint manifest has no checkpoint path"
        )
    recorded_path = Path(recorded_checkpoint)
    if not recorded_path.is_absolute():
        recorded_path = manifest.parent / recorded_path
    try:
        checkpoint_lstat = checkpoint_open_path.lstat()
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"quantity checkpoint not found: {checkpoint_open_path}"
        ) from exc
    if stat.S_ISLNK(checkpoint_lstat.st_mode):
        raise ValueError(
            "quantity checkpoint path must not be a symbolic link: "
            f"{checkpoint_open_path}"
        )
    checkpoint = checkpoint_open_path.resolve(strict=True)
    if recorded_path.resolve(strict=True) != checkpoint:
        raise ValueError(
            "quantity checkpoint path differs from its manifest: "
            f"job={checkpoint}, manifest={recorded_path.resolve()}"
        )
    expected_size = value.get("checkpoint_size")
    if (
        not isinstance(expected_size, int)
        or isinstance(expected_size, bool)
        or expected_size < 1
    ):
        raise ValueError(
            "quantity checkpoint manifest has no valid checkpoint_size"
        )
    expected_sha256 = value.get("checkpoint_sha256")
    if (
        not isinstance(expected_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
    ):
        raise ValueError(
            "quantity checkpoint manifest has no valid checkpoint_sha256"
        )

    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(checkpoint_open_path, flags)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise ValueError(
                "quantity checkpoint path must not be a symbolic link: "
                f"{checkpoint_open_path}"
            ) from exc
        if exc.errno == errno.ENOENT:
            raise FileNotFoundError(
                f"quantity checkpoint not found: {checkpoint_open_path}"
            ) from exc
        raise
    with os.fdopen(descriptor, "rb", closefd=True) as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(
                "quantity checkpoint must be a regular file: "
                f"{checkpoint_open_path}"
            )
        if before.st_size != expected_size:
            raise ValueError(
                "quantity checkpoint size differs from its manifest: "
                f"expected={expected_size}, actual={before.st_size}"
            )
        checkpoint_bytes = handle.read(expected_size + 1)
        after_read = os.fstat(handle.fileno())
        if (
            _checkpoint_stat_identity(before)
            != _checkpoint_stat_identity(after_read)
        ):
            raise RuntimeError(
                "quantity checkpoint changed while its bytes were read"
            )
        if len(checkpoint_bytes) != expected_size:
            raise ValueError(
                "quantity checkpoint byte length differs from its manifest: "
                f"expected={expected_size}, actual={len(checkpoint_bytes)}"
            )
        actual_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
        if actual_sha256 != expected_sha256:
            raise ValueError(
                "quantity checkpoint SHA-256 differs from its manifest: "
                f"expected={expected_sha256}, actual={actual_sha256}"
            )
        audit = {
            "checkpoint": str(checkpoint),
            "manifest": str(manifest),
            "checkpoint_size": after_read.st_size,
            "checkpoint_sha256": actual_sha256,
            "checkpoint_identity": {
                "device": after_read.st_dev,
                "inode": after_read.st_ino,
                "size": after_read.st_size,
                "mtime_ns": after_read.st_mtime_ns,
                "ctime_ns": after_read.st_ctime_ns,
            },
            "checkpoint_read_mode": "single_fd_single_bytes",
            "nofollow_requested": bool(getattr(os, "O_NOFOLLOW", 0)),
            "descriptor_identity_verified_after_consume": False,
            "path_identity_verified_after_consume": False,
            "verified": False,
        }
        try:
            yield checkpoint_bytes, audit
        finally:
            after_consume = os.fstat(handle.fileno())
            if (
                _checkpoint_stat_identity(after_read)
                != _checkpoint_stat_identity(after_consume)
            ):
                raise RuntimeError(
                    "quantity checkpoint descriptor identity changed while "
                    "its authenticated bytes were consumed"
                )
            try:
                path_after_consume = checkpoint_open_path.lstat()
            except FileNotFoundError as exc:
                raise RuntimeError(
                    "quantity checkpoint path disappeared while its "
                    "authenticated bytes were consumed"
                ) from exc
            if stat.S_ISLNK(path_after_consume.st_mode):
                raise RuntimeError(
                    "quantity checkpoint path became a symbolic link while "
                    "its authenticated bytes were consumed"
                )
            if (
                _checkpoint_stat_identity(path_after_consume)
                != _checkpoint_stat_identity(after_consume)
            ):
                raise RuntimeError(
                    "quantity checkpoint path identity changed while its "
                    "authenticated bytes were consumed"
                )
            audit["descriptor_identity_verified_after_consume"] = True
            audit["path_identity_verified_after_consume"] = True
            audit["verified"] = True


def _read_verified_quantity_checkpoint(
    checkpoint_path: str | Path,
    manifest_path: str | Path,
) -> tuple[bytes, dict[str, Any]]:
    """Read and verify one descriptor-bound checkpoint byte buffer."""

    with verified_quantity_checkpoint_bytes(
        checkpoint_path,
        manifest_path,
    ) as verified:
        return verified


def verify_quantity_checkpoint_manifest(
    checkpoint_path: str | Path,
    manifest_path: str | Path,
) -> dict[str, Any]:
    """Fail closed unless exact checkpoint bytes match the run manifest."""

    _, audit = _read_verified_quantity_checkpoint(
        checkpoint_path,
        manifest_path,
    )
    return audit


def load_verified_combined_quantity_checkpoint(
    pipe,
    encoder: QuantityEncoder,
    checkpoint_path: str | Path,
    manifest_path: str | Path,
    *,
    lora_alpha: float,
) -> dict[str, Any]:
    """Verify and parse one descriptor-bound checkpoint byte buffer."""

    checkpoint_bytes, verification = _read_verified_quantity_checkpoint(
        checkpoint_path,
        manifest_path,
    )
    from safetensors.torch import load

    state = load(checkpoint_bytes)
    # safetensors.torch.load materializes independent tensor backing stores.
    # Drop the authenticated input buffer before topology validation/fusion.
    del checkpoint_bytes
    loaded = _load_combined_quantity_checkpoint_state(
        pipe,
        encoder,
        state,
        verification["checkpoint"],
        lora_alpha=lora_alpha,
    )
    return {
        **loaded,
        "checkpoint_verification": verification,
        "load_boundary_verified": True,
        "checkpoint_load_mode": "manifest_hash_and_safetensors_same_bytes",
    }


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
    return _load_combined_quantity_checkpoint_state(
        pipe,
        encoder,
        state,
        str(path),
        lora_alpha=lora_alpha,
    )


def _load_combined_quantity_checkpoint_state(
    pipe,
    encoder: QuantityEncoder,
    state: dict[str, torch.Tensor],
    source: str,
    *,
    lora_alpha: float,
) -> dict[str, Any]:
    """Validate and fuse an already materialized combined checkpoint."""

    alpha = float(lora_alpha)
    if not math.isfinite(alpha):
        raise ValueError(f"LoRA alpha must be finite, got {lora_alpha!r}")
    quantity_keys = {
        key
        for key in state
        if key.startswith(QUANTITY_CHECKPOINT_PREFIX)
        or key.startswith("quantity_encoder.")
    }
    if not quantity_keys:
        raise ValueError(
            f"checkpoint has no quantity encoder tensors: {source}"
        )
    if len(quantity_keys) != QUANTITY_ENCODER_TENSOR_COUNT:
        raise ValueError(
            "combined checkpoint must contain exactly "
            f"{QUANTITY_ENCODER_TENSOR_COUNT} quantity encoder tensors, "
            f"got {len(quantity_keys)}"
        )
    quantity_prefixes = {
        (
            QUANTITY_CHECKPOINT_PREFIX
            if key.startswith(QUANTITY_CHECKPOINT_PREFIX)
            else "quantity_encoder."
        )
        for key in quantity_keys
    }
    if len(quantity_prefixes) != 1:
        raise ValueError(
            "combined checkpoint mixes quantity encoder key prefixes"
        )
    quantity = _quantity_state_dict(state)
    _validate_quantity_encoder_state_dict(encoder, quantity)
    lora = {
        key: value
        for key, value in state.items()
        if key not in quantity_keys
    }
    if not lora:
        raise ValueError(f"checkpoint has no DiT LoRA tensors: {source}")
    validated_lora_targets = _validate_dit_lora_state_dict(
        pipe.dit,
        lora,
    )
    # Both sub-states are fully validated before either model is mutated.
    missing, unexpected = encoder.load_state_dict(quantity, strict=True)
    if missing or unexpected:
        raise ValueError(
            "quantity encoder checkpoint mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )
    pipe.load_lora(
        pipe.dit,
        state_dict=lora,
        alpha=alpha,
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
    from safetensors.torch import load_file

    combined = load_file(str(path), device="cpu")
    raw_quantity_keys = {
        key
        for key in combined
        if key.startswith(QUANTITY_CHECKPOINT_PREFIX)
        or key.startswith("quantity_encoder.")
    }
    if not raw_quantity_keys:
        if required:
            raise ValueError(
                f"checkpoint has no quantity encoder tensors: {path}"
            )
        return {"loaded": False, "tensor_count": 0}
    if len(raw_quantity_keys) != QUANTITY_ENCODER_TENSOR_COUNT:
        raise ValueError(
            "checkpoint must contain exactly "
            f"{QUANTITY_ENCODER_TENSOR_COUNT} quantity encoder tensors, "
            f"got {len(raw_quantity_keys)}"
        )
    quantity_prefixes = {
        (
            QUANTITY_CHECKPOINT_PREFIX
            if key.startswith(QUANTITY_CHECKPOINT_PREFIX)
            else "quantity_encoder."
        )
        for key in raw_quantity_keys
    }
    if len(quantity_prefixes) != 1:
        raise ValueError(
            "checkpoint mixes quantity encoder key prefixes"
        )
    state = _quantity_state_dict(combined)
    _validate_quantity_encoder_state_dict(encoder, state)
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
    "QUANTITY_ENCODER_STATE_KEYS",
    "QUANTITY_ENCODER_TENSOR_COUNT",
    "QUANTITY_CHECKPOINT_PREFIX",
    "QuantityEncoder",
    "WAN22_TI2V_5B_BLOCK_COUNT",
    "WAN22_TI2V_5B_LORA_PAIR_COUNT",
    "WAN22_TI2V_5B_LORA_RANK",
    "WAN22_TI2V_5B_LORA_TARGET_SUFFIXES",
    "WAN22_TI2V_5B_LORA_TENSOR_COUNT",
    "distributed_sampling_contract",
    "encode_quantity_prompt",
    "expected_wan22_ti2v_5b_lora_targets",
    "install_quantity_prompt_unit",
    "load_combined_quantity_checkpoint",
    "load_quantity_encoder_checkpoint",
    "load_verified_combined_quantity_checkpoint",
    "locate_sentinel_tokens",
    "numeric_features",
    "quantity_inference_conditioning",
    "quantity_pipeline_shared_config",
    "quantity_pipeline_shared_fingerprint",
    "quantity_state_dict",
    "verified_quantity_checkpoint_bytes",
    "verify_quantity_checkpoint_manifest",
]
