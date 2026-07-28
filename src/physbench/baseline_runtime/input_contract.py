"""Validation for auditable Baseline adaptation input contracts.

An adapter owns the opaque ``native_inputs`` consumed by its Baseline.  The
lightweight ``input_contract`` validated here describes how text, media, and
optional physics controls are bound into that object without prescribing a
model-specific representation.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


INPUT_CONTRACT_SCHEMA_VERSION = "1.0"
GENERATION_MODES = frozenset({"t2v", "i2v", "v2v", "hybrid"})
MEDIA_KINDS = frozenset({"image", "video"})
CONDITIONING_MODES = frozenset({"generic", "physics"})
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
DEFAULT_PHYSICS_REPRESENTATIONS = frozenset({"structured_text"})
MAX_INLINE_CONTROL_BYTES = 4096

__all__ = [
    "ARTIFACT_REFERENCE_PREFIXES",
    "CONDITIONING_MODES",
    "DEFAULT_PHYSICS_REPRESENTATIONS",
    "FORBIDDEN_ASSET_KEYS",
    "GENERATION_MODES",
    "INPUT_CONTRACT_SCHEMA_VERSION",
    "LARGE_PHYSICS_REPRESENTATIONS",
    "MEDIA_KINDS",
    "MAX_INLINE_CONTROL_BYTES",
    "NATIVE_INPUT_BINDING_PREFIX",
    "resolve_binding",
    "validate_adaptation_record",
]


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


def _validate_capability_representations(
    capabilities: Mapping[str, Any],
) -> frozenset[str]:
    raw = capabilities.get(
        "physics_representations",
        list(DEFAULT_PHYSICS_REPRESENTATIONS),
    )
    if not isinstance(raw, list) or not raw:
        raise ValueError(
            "capabilities.physics_representations must be a non-empty list"
        )
    representations: list[str] = []
    for index, value in enumerate(raw):
        representations.append(_require_non_empty_string(
            value,
            label=(
                "capabilities.physics_representations"
                f"[{index}]"
            ),
        ))
    if len(representations) != len(set(representations)):
        raise ValueError(
            "capabilities.physics_representations contains duplicates"
        )
    return frozenset(representations)


def _is_artifact_reference(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    return any(
        value.startswith(prefix) and len(value) > len(prefix)
        for prefix in ARTIFACT_REFERENCE_PREFIXES
    )


def validate_adaptation_record(
    record: Mapping[str, Any],
    *,
    conditioning: str,
    capabilities: Mapping[str, Any],
) -> None:
    """Validate one adapter-produced record and its input contract.

    ``generic`` adaptations may describe text and media but cannot declare or
    consume physical parameters.  ``physics`` adaptations must declare at
    least one physics channel, and every used physical parameter must be
    attributable to a channel.  Large control representations are external
    artifact/cache references so task instances remain lightweight.

    Args:
        record: Complete adaptation record.
        conditioning: Benchmark information-access arm (``generic`` or
            ``physics``), independent of the model's injection mechanism.
        capabilities: Baseline capability object.  Older manifests that omit
            ``physics_representations`` are treated as supporting only
            ``structured_text``.

    Raises:
        ValueError: If the record or contract violates the declared protocol.
    """

    record = _require_object(record, label="adaptation record")
    capabilities = _require_object(
        capabilities,
        label="capabilities",
    )
    if conditioning not in CONDITIONING_MODES:
        raise ValueError(
            f"conditioning must be one of {sorted(CONDITIONING_MODES)}, "
            f"got {conditioning!r}"
        )

    _require_object(
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
        asset_key = _require_non_empty_string(
            channel.get("asset_key"),
            label=f"{label}.asset_key",
        )
        if (
            generation_mode == "v2v"
            and _normalized_asset_key(asset_key)
            in FORBIDDEN_ASSET_KEYS
        ):
            raise ValueError(
                f"{label}.asset_key {asset_key!r} is reserved and cannot "
                "be used as V2V input"
            )
        required_media_assets.add(asset_key)
        binding = _require_non_empty_string(
            channel.get("binding"),
            label=f"{label}.binding",
        )
        bound_media = resolve_binding(record, binding)
        if not isinstance(bound_media, str) or not bound_media.strip():
            raise ValueError(
                f"{label}.binding must resolve to a non-empty asset string"
            )

    missing_assets = required_media_assets - declared_assets
    if missing_assets:
        raise ValueError(
            "input_contract.asset_access must include every media asset key; "
            f"missing {sorted(missing_assets)}"
        )
    if generation_mode == "i2v" and "image" not in media_kinds:
        raise ValueError(
            "input_contract for generation_mode 'i2v' requires at least "
            "one image media channel"
        )
    if generation_mode == "v2v" and "video" not in media_kinds:
        raise ValueError(
            "input_contract for generation_mode 'v2v' requires at least "
            "one video media channel"
        )

    parameter_bindings: set[str] = set()
    supported_representations = (
        _validate_capability_representations(capabilities)
        if conditioning == "physics"
        else DEFAULT_PHYSICS_REPRESENTATIONS
    )
    for index, raw_channel in enumerate(physics_channels):
        label = f"input_contract.physics_channels[{index}]"
        channel = _require_object(raw_channel, label=label)
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
                "in capabilities.physics_representations"
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
        if (
            transport != "artifact_ref"
            and isinstance(bound_value, (list, dict))
            and len(
                json.dumps(
                    bound_value,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            > MAX_INLINE_CONTROL_BYTES
        ):
            raise ValueError(
                f"{label}.binding exceeds {MAX_INLINE_CONTROL_BYTES} inline "
                "bytes and therefore requires transport='artifact_ref'"
            )
        if representation in LARGE_PHYSICS_REPRESENTATIONS:
            if transport != "artifact_ref":
                raise ValueError(
                    f"{label} uses large representation {representation!r} "
                    "and therefore requires transport='artifact_ref'"
                )
            if not _is_artifact_reference(bound_value):
                raise ValueError(
                    f"{label}.binding must resolve to a non-empty "
                    "artifact:// or cache:// URI"
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

    if conditioning == "generic":
        if used_parameters:
            raise ValueError(
                "generic adaptation record.used_parameters must be {}"
            )
        if physics_channels:
            raise ValueError(
                "generic adaptation input_contract.physics_channels "
                "must be empty"
            )
    elif not physics_channels:
        raise ValueError(
            "physics adaptation requires at least one physics channel"
        )

    used_parameter_keys = set(used_parameters)
    if parameter_bindings != used_parameter_keys:
        missing = sorted(used_parameter_keys - parameter_bindings)
        extra = sorted(parameter_bindings - used_parameter_keys)
        raise ValueError(
            "physics channel used_parameters must match adaptation "
            f"record.used_parameters keys; missing={missing}, extra={extra}"
        )
