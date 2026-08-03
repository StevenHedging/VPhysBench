"""Validation for auditable Baseline adaptation input contracts.

An adapter owns the opaque ``native_inputs`` consumed by its Baseline.  The
lightweight ``input_contract`` validated here describes how text, media, and
optional physics controls are bound into that object without prescribing a
model-specific representation.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..baseline_api.input_policy import validate_input_policy
from .media_contract import validate_media_contract

INPUT_CONTRACT_SCHEMA_VERSION = "1.0"
GENERATION_MODES = frozenset({"t2v", "i2v", "v2v", "hybrid"})
MEDIA_KINDS = frozenset({"image", "video"})
NATIVE_INPUT_BINDING_PREFIX = "native_inputs."

FORBIDDEN_ASSET_KEYS = frozenset({
    "reference_video",
    "physics_reference_video",
    "source_video",
})
LARGE_PHYSICS_REPRESENTATIONS = frozenset({
    "mask",
    "optical_flow",
    "flow",
    "force_field",
    "trajectory",
    "state_sequence",
    "proxy_video",
    "control_video",
})
ARTIFACT_REFERENCE_PREFIXES = ("artifact://", "cache://")
ARTIFACT_REFERENCE_RE = re.compile(
    r"^(?:artifact|cache)://sha256/([0-9a-f]{64})$"
)
MAX_INLINE_CONTROL_BYTES = 4096

__all__ = [
    "ARTIFACT_REFERENCE_PREFIXES",
    "ARTIFACT_REFERENCE_RE",
    "FORBIDDEN_ASSET_KEYS",
    "GENERATION_MODES",
    "INPUT_CONTRACT_SCHEMA_VERSION",
    "LARGE_PHYSICS_REPRESENTATIONS",
    "MEDIA_KINDS",
    "MAX_INLINE_CONTROL_BYTES",
    "NATIVE_INPUT_BINDING_PREFIX",
    "resolve_binding",
    "resolve_dataset_asset_path",
    "validate_adaptation_record",
]


def resolve_dataset_asset_path(
    asset_root: str | Path,
    value: Any,
    *,
    label: str,
) -> Path:
    """Resolve a Dataset asset while rejecting absolute and escaping paths."""

    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty relative asset path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(
            f"{label} must stay inside the Dataset asset root: {value!r}"
        )
    root = Path(asset_root).resolve()
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"{label} escapes the Dataset asset root: {value!r}"
        ) from exc
    return resolved


def resolve_binding(
    record: Mapping[str, Any],
    binding: str,
) -> Any:
    """Resolve a ``native_inputs.*`` dot path in an adaptation record.

    Bindings deliberately address object members only.  This keeps contracts
    stable when a Baseline changes the size or internal layout of a list-valued
    tensor/control payload.

    Args:
        record: Complete adaptation record containing ``native_inputs``.
        binding: Dot-separated path rooted at ``native_inputs``.

    Returns:
        The value addressed by ``binding``.

    Raises:
        ValueError: If the binding is malformed or cannot be resolved.
    """

    if not isinstance(binding, str) or not binding:
        raise ValueError("binding must be a non-empty string")
    if not binding.startswith(NATIVE_INPUT_BINDING_PREFIX):
        raise ValueError(
            f"binding {binding!r} must start with "
            f"{NATIVE_INPUT_BINDING_PREFIX!r}"
        )
    parts = binding.split(".")
    if any(not part for part in parts):
        raise ValueError(f"binding {binding!r} contains an empty path segment")

    value: Any = record
    traversed: list[str] = []
    for part in parts:
        traversed.append(part)
        if not isinstance(value, Mapping) or part not in value:
            prefix = ".".join(traversed)
            raise ValueError(
                f"binding {binding!r} does not resolve at {prefix!r}"
            )
        value = value[part]
    return value


def _require_object(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _require_list(value: Any, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return value


def _require_non_empty_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _normalized_asset_key(value: str) -> str:
    """Return the logical asset key used for reserved-key checks."""

    key = value.split(":", 1)[0]
    if key.startswith("assets."):
        key = key[len("assets.") :]
    return key


def _validate_channel_id(
    value: Any,
    *,
    label: str,
    seen: dict[str, str],
) -> str:
    channel_id = _require_non_empty_string(value, label=f"{label}.id")
    previous = seen.get(channel_id)
    if previous is not None:
        raise ValueError(
            f"duplicate input_contract channel id {channel_id!r} "
            f"in {previous} and {label}"
        )
    seen[channel_id] = label
    return channel_id


def _artifact_digest(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    match = ARTIFACT_REFERENCE_RE.fullmatch(value)
    return match.group(1) if match else None


def _require_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _validate_artifact_provenance(
    *,
    bound_value: Any,
    provenance_value: Any,
    label: str,
) -> None:
    artifact_digest = _artifact_digest(bound_value)
    if artifact_digest is None:
        raise ValueError(
            f"{label}.binding must resolve to "
            "artifact://sha256/<digest> or cache://sha256/<digest>"
        )
    provenance = _require_object(
        provenance_value,
        label=f"{label}.artifact_provenance",
    )
    allowed = {
        "content_sha256",
        "producer_fingerprint",
        "source_digest",
    }
    unknown = sorted(set(provenance) - allowed)
    missing = sorted(allowed - set(provenance))
    if missing or unknown:
        raise ValueError(
            f"{label}.artifact_provenance fields mismatch: "
            f"missing={missing}, unknown={unknown}"
        )
    content_sha256 = _require_sha256(
        provenance.get("content_sha256"),
        label=f"{label}.artifact_provenance.content_sha256",
    )
    _require_sha256(
        provenance.get("producer_fingerprint"),
        label=f"{label}.artifact_provenance.producer_fingerprint",
    )
    _require_sha256(
        provenance.get("source_digest"),
        label=f"{label}.artifact_provenance.source_digest",
    )
    if content_sha256 != artifact_digest:
        raise ValueError(
            f"{label}.artifact_provenance.content_sha256 must match "
            "the digest in its artifact URI"
        )


def validate_adaptation_record(
    record: Mapping[str, Any],
    *,
    input_policy: Mapping[str, Any],
) -> None:
    """Validate one adapter-produced record and its input contract.

    The Baseline's fixed physics usage policy determines whether physical
    channels are forbidden, optional, or required. Every used physical
    parameter must be attributable to a channel. Large control
    representations are external artifact/cache references so task instances
    remain lightweight.

    Args:
        record: Complete adaptation record.
        input_policy: Normalized Baseline-owned Case input policy.

    Raises:
        ValueError: If the record or contract violates the declared protocol.
    """

    record = _require_object(record, label="adaptation record")
    policy = validate_input_policy(input_policy)
    physics_policy = policy["physics"]
    physics_usage = physics_policy["usage"]

    native_inputs = _require_object(
        record.get("native_inputs"),
        label="adaptation record.native_inputs",
    )
    used_parameters = _require_object(
        record.get("used_parameters"),
        label="adaptation record.used_parameters",
    )
    for key in used_parameters:
        _require_non_empty_string(
            key,
            label="adaptation record.used_parameters key",
        )

    contract = _require_object(
        record.get("input_contract"),
        label="adaptation record.input_contract",
    )
    contract_fields = {
        "schema_version",
        "generation_mode",
        "text",
        "media_channels",
        "physics_channels",
        "asset_access",
    }
    unknown_contract_fields = sorted(set(contract) - contract_fields)
    missing_contract_fields = sorted(contract_fields - set(contract))
    if unknown_contract_fields or missing_contract_fields:
        raise ValueError(
            "input_contract fields mismatch: "
            f"missing={missing_contract_fields}, "
            f"unknown={unknown_contract_fields}"
        )
    if contract.get("schema_version") != INPUT_CONTRACT_SCHEMA_VERSION:
        raise ValueError(
            "input_contract.schema_version must be "
            f"{INPUT_CONTRACT_SCHEMA_VERSION!r}"
        )
    generation_mode = contract.get("generation_mode")
    if generation_mode not in GENERATION_MODES:
        raise ValueError(
            "input_contract.generation_mode must be one of "
            f"{sorted(GENERATION_MODES)}, got {generation_mode!r}"
        )

    text_contract = _require_object(
        contract.get("text"),
        label="input_contract.text",
    )
    if set(text_contract) != {"required", "binding"}:
        raise ValueError(
            "input_contract.text must contain exactly required and binding"
        )
    if text_contract.get("required") is not True:
        raise ValueError("input_contract.text.required must be true")
    text_binding = _require_non_empty_string(
        text_contract.get("binding"),
        label="input_contract.text.binding",
    )
    text_value = resolve_binding(record, text_binding)
    if not isinstance(text_value, str) or not text_value.strip():
        raise ValueError(
            "input_contract.text.binding must resolve to a non-empty string"
        )

    media_channels = _require_list(
        contract.get("media_channels"),
        label="input_contract.media_channels",
    )
    physics_channels = _require_list(
        contract.get("physics_channels"),
        label="input_contract.physics_channels",
    )
    asset_access = _require_list(
        contract.get("asset_access"),
        label="input_contract.asset_access",
    )

    declared_assets: set[str] = set()
    for index, value in enumerate(asset_access):
        asset_key = _require_non_empty_string(
            value,
            label=f"input_contract.asset_access[{index}]",
        )
        normalized = _normalized_asset_key(asset_key)
        if normalized in FORBIDDEN_ASSET_KEYS:
            raise ValueError(
                f"input_contract.asset_access cannot include reserved "
                f"asset key {asset_key!r}"
            )
        declared_assets.add(asset_key)
    if len(declared_assets) != len(asset_access):
        raise ValueError(
            "input_contract.asset_access contains duplicates"
        )

    seen_channel_ids: dict[str, str] = {}
    media_kinds: set[str] = set()
    required_media_assets: set[str] = set()
    for index, raw_channel in enumerate(media_channels):
        label = f"input_contract.media_channels[{index}]"
        channel = _require_object(raw_channel, label=label)
        allowed_media_fields = {
            "id",
            "kind",
            "origin",
            "asset_key",
            "binding",
            "artifact_provenance",
        }
        unknown_media_fields = sorted(
            set(channel) - allowed_media_fields
        )
        if unknown_media_fields:
            raise ValueError(
                f"{label} contains unknown fields: "
                f"{unknown_media_fields}"
            )
        _validate_channel_id(
            channel.get("id"),
            label=label,
            seen=seen_channel_ids,
        )
        kind = channel.get("kind")
        if kind not in MEDIA_KINDS:
            raise ValueError(
                f"{label}.kind must be one of {sorted(MEDIA_KINDS)}, "
                f"got {kind!r}"
            )
        media_kinds.add(kind)
        origin = channel.get("origin", "dataset_asset")
        if origin not in {"dataset_asset", "derived_artifact"}:
            raise ValueError(
                f"{label}.origin must be dataset_asset or "
                "derived_artifact"
            )
        binding = _require_non_empty_string(
            channel.get("binding"),
            label=f"{label}.binding",
        )
        bound_media = resolve_binding(record, binding)
        if not isinstance(bound_media, str) or not bound_media.strip():
            raise ValueError(
                f"{label}.binding must resolve to a non-empty asset string"
            )
        if origin == "dataset_asset":
            asset_key = _require_non_empty_string(
                channel.get("asset_key"),
                label=f"{label}.asset_key",
            )
            if "artifact_provenance" in channel:
                raise ValueError(
                    f"{label}.artifact_provenance is only valid for "
                    "derived_artifact"
                )
            if (
                generation_mode == "v2v"
                and _normalized_asset_key(asset_key)
                in FORBIDDEN_ASSET_KEYS
            ):
                raise ValueError(
                    f"{label}.asset_key {asset_key!r} is reserved and "
                    "cannot be used as V2V input"
                )
            required_media_assets.add(asset_key)
        else:
            if "asset_key" in channel:
                raise ValueError(
                    f"{label}.asset_key must be omitted for "
                    "derived_artifact"
                )
            _validate_artifact_provenance(
                bound_value=bound_media,
                provenance_value=channel.get("artifact_provenance"),
                label=label,
            )

    missing_assets = required_media_assets - declared_assets
    if missing_assets:
        raise ValueError(
            "input_contract.asset_access must include every media asset key; "
            f"missing {sorted(missing_assets)}"
        )
    if generation_mode == "t2v" and media_channels:
        raise ValueError(
            "input_contract for generation_mode 't2v' must not declare "
            "media channels"
        )
    if generation_mode == "i2v" and media_kinds != {"image"}:
        raise ValueError(
            "input_contract for generation_mode 'i2v' requires one or more "
            "image channels and no video channels"
        )
    if generation_mode == "i2v":
        media_contract = validate_media_contract(
            native_inputs.get("media_contract")
        )
        conditioning_assets = {
            resolve_binding(record, channel["binding"])
            for channel in media_channels
        }
        if media_contract["conditioning"]["asset"] not in conditioning_assets:
            raise ValueError(
                "native_inputs.media_contract conditioning asset is not "
                "bound by an I2V media channel"
            )
    if generation_mode == "v2v" and media_kinds != {"video"}:
        raise ValueError(
            "input_contract for generation_mode 'v2v' requires one or more "
            "video channels and no image channels"
        )
    if generation_mode == "hybrid" and media_kinds != {"image", "video"}:
        raise ValueError(
            "input_contract for generation_mode 'hybrid' requires both "
            "image and video channels"
        )

    parameter_bindings: set[str] = set()
    supported_representations = frozenset(
        physics_policy["representations"]
    )
    for index, raw_channel in enumerate(physics_channels):
        label = f"input_contract.physics_channels[{index}]"
        channel = _require_object(raw_channel, label=label)
        allowed_physics_fields = {
            "id",
            "representation",
            "binding",
            "transport",
            "used_parameters",
            "artifact_provenance",
        }
        unknown_physics_fields = sorted(
            set(channel) - allowed_physics_fields
        )
        if unknown_physics_fields:
            raise ValueError(
                f"{label} contains unknown fields: "
                f"{unknown_physics_fields}"
            )
        _validate_channel_id(
            channel.get("id"),
            label=label,
            seen=seen_channel_ids,
        )
        representation = _require_non_empty_string(
            channel.get("representation"),
            label=f"{label}.representation",
        )
        if representation not in supported_representations:
            raise ValueError(
                f"{label}.representation {representation!r} is not declared "
                "in input_policy.physics.representations"
            )
        transport = _require_non_empty_string(
            channel.get("transport"),
            label=f"{label}.transport",
        )
        binding = _require_non_empty_string(
            channel.get("binding"),
            label=f"{label}.binding",
        )
        bound_value = resolve_binding(record, binding)
        if transport == "artifact_ref":
            _validate_artifact_provenance(
                bound_value=bound_value,
                provenance_value=channel.get("artifact_provenance"),
                label=label,
            )
        else:
            if "artifact_provenance" in channel:
                raise ValueError(
                    f"{label}.artifact_provenance requires "
                    "transport='artifact_ref'"
                )
            semantic_text = (
                representation == "structured_text"
                and binding == text_binding
            )
            if not semantic_text:
                try:
                    inline_size = len(
                        json.dumps(
                            bound_value,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    )
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"{label}.binding is not a JSON value"
                    ) from exc
                if inline_size > MAX_INLINE_CONTROL_BYTES:
                    raise ValueError(
                        f"{label}.binding exceeds "
                        f"{MAX_INLINE_CONTROL_BYTES} inline bytes and "
                        "therefore requires transport='artifact_ref'"
                    )
        if representation in LARGE_PHYSICS_REPRESENTATIONS:
            if transport != "artifact_ref":
                raise ValueError(
                    f"{label} uses large representation {representation!r} "
                    "and therefore requires transport='artifact_ref'"
                )

        raw_parameters = _require_list(
            channel.get("used_parameters"),
            label=f"{label}.used_parameters",
        )
        channel_parameters: list[str] = []
        for parameter_index, parameter in enumerate(raw_parameters):
            channel_parameters.append(_require_non_empty_string(
                parameter,
                label=(
                    f"{label}.used_parameters[{parameter_index}]"
                ),
            ))
        if len(channel_parameters) != len(set(channel_parameters)):
            raise ValueError(
                f"{label}.used_parameters contains duplicates"
            )
        parameter_bindings.update(channel_parameters)

    if physics_usage == "ignored":
        if used_parameters:
            raise ValueError(
                "physics-ignored adaptation record.used_parameters must be {}"
            )
        if physics_channels:
            raise ValueError(
                "physics-ignored adaptation input_contract.physics_channels "
                "must be empty"
            )
    elif physics_usage == "required" and not physics_channels:
        raise ValueError(
            "physics-required adaptation requires at least one physics channel"
        )
    elif physics_usage == "required" and not used_parameters:
        raise ValueError(
            "physics-required adaptation must consume at least one annotated "
            "physical parameter"
        )
    elif physics_usage == "optional" and bool(physics_channels) != bool(
        used_parameters
    ):
        raise ValueError(
            "physics-optional adaptation must declare channels exactly when "
            "it consumes physical parameters"
        )

    used_parameter_keys = set(used_parameters)
    if parameter_bindings != used_parameter_keys:
        missing = sorted(used_parameter_keys - parameter_bindings)
        extra = sorted(parameter_bindings - used_parameter_keys)
        raise ValueError(
            "physics channel used_parameters must match adaptation "
            f"record.used_parameters keys; missing={missing}, extra={extra}"
        )
