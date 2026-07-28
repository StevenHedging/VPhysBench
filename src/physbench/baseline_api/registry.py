from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

from ..domain import BaselineBundle
from ..io import canonical_sha256, load_json, sha256_file
from .input_policy import validate_input_policy
from .interfaces import BaselinePlugin


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BASELINES_ROOT = PROJECT_ROOT / "baselines"
DESCRIPTOR_NAME = "baseline.json"
LOCAL_OVERRIDE_NAME = "baseline.local.json"
ALLOWED_LOCAL_OVERRIDE_KEYS = {"model", "runtime"}
COMMON_MANIFEST_KEYS = {
    "schema_version",
    "baseline_id",
    "baseline_version",
    "description",
    "implementation",
    "supported_scenes",
    "capabilities",
    "input_policy",
    "model",
    "runtime",
}
V5_MANIFEST_KEYS = {
    *COMMON_MANIFEST_KEYS,
    "adapter",
    "runner",
    "trainer",
}
V4_IMPLEMENTATION_KEYS = {
    "kind",
    "driver",
    "fingerprint_paths",
}
ALLOWED_CAPABILITY_KEYS = {
    "task_families",
    "generation_modes",
    "train",
    "finetune",
    "generate",
}


def _require_string(value: dict[str, Any], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result.strip():
        raise ValueError(f"baseline bundle requires a non-empty {key}")
    return result


def _validate_baseline_id(value: dict[str, Any]) -> str:
    baseline_id = _require_string(value, "baseline_id")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", baseline_id):
        raise ValueError(
            "baseline_id may only contain letters, digits, '.', '_' and '-'"
        )
    return baseline_id


def _validate_relative_path(root: Path, value: str, *, label: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} must be a bundle-relative path: {value}")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes the Baseline bundle: {value}") from exc
    return resolved


def _validate_fingerprint_patterns(
    implementation: dict[str, Any],
    *,
    required: bool,
) -> None:
    patterns = implementation.get("fingerprint_paths")
    if patterns is None and not required:
        return
    if (
        not isinstance(patterns, list)
        or (required and not patterns)
        or any(
            not isinstance(item, str) or not item or "\x00" in item
            for item in patterns
        )
    ):
        qualifier = "a non-empty" if required else "a"
        raise ValueError(
            f"implementation.fingerprint_paths must be {qualifier} string list"
        )
    for pattern in patterns:
        if Path(pattern).is_absolute() or ".." in Path(pattern).parts:
            raise ValueError(
                "implementation fingerprint glob must be bundle-relative: "
                f"{pattern}"
            )


def _validate_v5_implementation(
    implementation: dict[str, Any],
    descriptor_path: Path,
) -> None:
    unknown = sorted(set(implementation) - V4_IMPLEMENTATION_KEYS)
    if unknown:
        raise ValueError(
            f"baseline implementation contains unknown fields: {unknown}"
        )
    kind = implementation.get("kind")
    if kind not in {"managed", "submission"}:
        raise ValueError(
            "schema v5 baseline implementation kind must be managed or "
            f"submission, got {kind!r}"
        )
    _validate_fingerprint_patterns(implementation, required=False)
    if kind == "managed":
        driver = implementation.get("driver")
        if not isinstance(driver, str) or not driver:
            raise ValueError(
                "managed baseline requires implementation.driver"
            )
        path = _validate_relative_path(
            descriptor_path.parent,
            driver,
            label="implementation.driver",
        )
        if not path.is_file():
            raise FileNotFoundError(
                f"managed baseline driver not found: {path}"
            )
    elif "driver" in implementation:
        raise ValueError(
            "submission baseline must not declare implementation.driver"
        )


def _validate_manifest(value: dict[str, Any], descriptor_path: Path) -> None:
    schema_version = value.get("schema_version")
    if schema_version != "5.0":
        raise ValueError(
            "baseline bundle must use schema_version=5.0; schema-v3/v4 "
            "Bundles are legacy and cannot compile new Tasks"
        )
    allowed = V5_MANIFEST_KEYS
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"baseline bundle contains unknown fields: {unknown}")
    _validate_baseline_id(value)
    _require_string(value, "baseline_version")

    implementation = value.get("implementation")
    if not isinstance(implementation, dict):
        raise ValueError("baseline bundle requires an implementation object")
    kind = implementation.get("kind")
    _validate_v5_implementation(implementation, descriptor_path)

    capabilities = value.get("capabilities")
    if not isinstance(capabilities, dict):
        raise ValueError("baseline bundle requires a capabilities object")
    unknown = sorted(set(capabilities) - ALLOWED_CAPABILITY_KEYS)
    if unknown:
        raise ValueError(
            f"baseline capabilities contains unknown fields: {unknown}"
        )
    for key in ("task_families", "generation_modes"):
        items = capabilities.get(key)
        if (
            not isinstance(items, list)
            or not items
            or any(not isinstance(item, str) or not item for item in items)
        ):
            raise ValueError(f"capabilities.{key} must be a non-empty string list")
        if len(items) != len(set(items)):
            raise ValueError(f"capabilities.{key} contains duplicates")
    generation_modes = capabilities.get("generation_modes")
    if generation_modes is not None:
        unknown_modes = sorted(
            set(generation_modes) - {"t2v", "i2v", "v2v", "hybrid"}
        )
        if unknown_modes:
            raise ValueError(
                "capabilities.generation_modes contains unsupported modes: "
                f"{unknown_modes}"
            )
    for key in ("train", "finetune", "generate"):
        if key in capabilities and not isinstance(capabilities[key], bool):
            raise ValueError(f"capabilities.{key} must be a boolean")
    input_policy = validate_input_policy(value.get("input_policy"))

    supported_scenes = value.get("supported_scenes", "all")
    if supported_scenes != "all" and (
        not isinstance(supported_scenes, list)
        or not supported_scenes
        or any(
            not isinstance(item, str) or not item
            for item in supported_scenes
        )
    ):
        raise ValueError(
            "supported_scenes must be 'all' or a non-empty string list"
        )
    if isinstance(supported_scenes, list) and len(supported_scenes) != len(
        set(supported_scenes)
    ):
        raise ValueError("supported_scenes contains duplicates")

    adapter = value.get("adapter")
    if not isinstance(adapter, dict):
        raise ValueError(
            "schema v5 baseline requires an adapter object"
        )
    adapter_kind = adapter.get("kind", "standard")
    if (
        not isinstance(adapter_kind, str)
        or adapter_kind not in {"standard", "python"}
    ):
        raise ValueError(
            "adapter.kind must be 'standard' or 'python'"
        )
    if adapter_kind == "python":
        unknown_adapter_fields = sorted(
            set(adapter)
            - {"kind", "entrypoint", "config", "cache_policy"}
        )
        if unknown_adapter_fields:
            raise ValueError(
                "Python adapter contains unknown fields: "
                f"{unknown_adapter_fields}"
            )
        entrypoint = adapter.get("entrypoint")
        if not isinstance(entrypoint, str) or not entrypoint:
            raise ValueError(
                "Python adapter requires adapter.entrypoint"
            )
        path = _validate_relative_path(
            descriptor_path.parent,
            entrypoint,
            label="adapter.entrypoint",
        )
        if not path.is_file():
            raise FileNotFoundError(
                f"Baseline adapter entrypoint not found: {path}"
            )
        entrypoint_path = Path(entrypoint)
        module_parts = [
            *entrypoint_path.parts[:-1],
            entrypoint_path.stem,
        ]
        if entrypoint_path.suffix != ".py" or any(
            not part.isidentifier() for part in module_parts
        ):
            raise ValueError(
                "adapter.entrypoint must be a Python module path whose "
                f"components are identifiers: {entrypoint}"
            )
        if "config" in adapter and not isinstance(
            adapter["config"], dict
        ):
            raise ValueError("adapter.config must be an object")
        if (
            "cache_policy" in adapter
            and (
                not isinstance(adapter["cache_policy"], str)
                or not adapter["cache_policy"]
            )
        ):
            raise ValueError(
                "adapter.cache_policy must be a non-empty string"
            )
        if capabilities.get("generation_modes") is None:
            raise ValueError(
                "Python adapter requires capabilities.generation_modes"
            )
    else:
        allowed_standard_fields = {
            "kind",
            "preset",
            "physics_transform",
            "spatial",
            "temporal",
            "first_frame_policy",
            "video_asset_key",
            "cache_policy",
        }
        unknown_adapter_fields = sorted(
            set(adapter) - allowed_standard_fields
        )
        if unknown_adapter_fields:
            raise ValueError(
                "standard adapter contains unknown fields: "
                f"{unknown_adapter_fields}"
            )
        preset_modes = {
            "standard_t2v_v1": "t2v",
            "standard_i2v_v1": "i2v",
            "standard_v2v_v1": "v2v",
        }
        preset = adapter.get("preset")
        if preset not in preset_modes:
            raise ValueError(
                f"unsupported standard adapter preset {preset!r}"
            )
        if preset_modes[preset] not in generation_modes:
            raise ValueError(
                f"adapter preset {preset!r} is not declared in "
                "capabilities.generation_modes"
            )
        for field in ("spatial", "temporal"):
            if not isinstance(adapter.get(field), dict):
                raise ValueError(
                    f"standard adapter requires a {field} object"
                )
        if preset == "standard_i2v_v1":
            if adapter.get("first_frame_policy") != "require_asset":
                raise ValueError(
                    "standard I2V adapter requires "
                    "first_frame_policy=require_asset"
                )
        elif "first_frame_policy" in adapter:
            raise ValueError(
                "first_frame_policy is only valid for standard I2V"
            )
        if preset == "standard_v2v_v1":
            video_asset_key = adapter.get("video_asset_key")
            if (
                not isinstance(video_asset_key, str)
                or not video_asset_key
            ):
                raise ValueError(
                    "standard V2V adapter requires video_asset_key"
                )
        elif "video_asset_key" in adapter:
            raise ValueError(
                "video_asset_key is only valid for standard V2V"
            )
        if (
            "cache_policy" in adapter
            and (
                not isinstance(adapter["cache_policy"], str)
                or not adapter["cache_policy"]
            )
        ):
            raise ValueError(
                "adapter.cache_policy must be a non-empty string"
            )
        transform = adapter.get("physics_transform", {"type": "none"})
        if not isinstance(transform, dict):
            raise ValueError("adapter.physics_transform must be an object")
        transform_type = transform.get("type")
        usage = input_policy["physics"]["usage"]
        if usage == "ignored" and transform_type != "none":
            raise ValueError(
                "physics-ignored standard adapter requires "
                "physics_transform.type=none"
            )
        if usage != "ignored" and transform_type != "append_structured_text_v1":
            raise ValueError(
                "physics-using standard adapter requires "
                "physics_transform.type=append_structured_text_v1"
            )
        if usage != "ignored" and input_policy["physics"][
            "representations"
        ] != ["structured_text"]:
            raise ValueError(
                "append_structured_text_v1 requires exactly the "
                "structured_text representation"
            )
        if transform_type == "append_structured_text_v1":
            if set(transform) != {"type", "template_set"}:
                raise ValueError(
                    "append_structured_text_v1 requires exactly type and "
                    "template_set"
                )
            if not isinstance(transform["template_set"], str) or not transform[
                "template_set"
            ]:
                raise ValueError(
                    "physics_transform.template_set must be non-empty"
                )
        elif transform != {"type": "none"}:
            raise ValueError(
                "physics_transform.type=none accepts no extra fields"
            )
    if kind == "managed" and not isinstance(value.get("runner"), dict):
        raise ValueError(
            "managed baseline requires a runner object"
        )
    if "trainer" in value and not isinstance(value["trainer"], dict):
        raise ValueError(
            "baseline trainer must be an object when present"
        )


def discover_baseline_bundles(
    baselines_root: str | Path | None = None,
) -> dict[str, Path]:
    """Discover manifests without importing or executing Baseline code."""
    root = Path(baselines_root or DEFAULT_BASELINES_ROOT).resolve()
    discovered: dict[str, Path] = {}
    if not root.is_dir():
        return discovered
    descriptor_paths = {
        *root.glob(f"*/{DESCRIPTOR_NAME}"),
        *root.glob("*/*.baseline.json"),
    }
    for descriptor_path in sorted(descriptor_paths):
        value = load_json(descriptor_path)
        baseline_id = _validate_baseline_id(value)
        previous = discovered.get(baseline_id)
        if previous is not None:
            raise ValueError(
                f"duplicate baseline_id {baseline_id!r}: "
                f"{previous} and {descriptor_path.resolve()}"
            )
        discovered[baseline_id] = descriptor_path.resolve()
    return discovered


def resolve_baseline_descriptor(
    reference: str | Path,
    *,
    baselines_root: str | Path | None = None,
) -> Path:
    candidate = Path(reference).expanduser()
    if candidate.exists():
        descriptor = candidate / DESCRIPTOR_NAME if candidate.is_dir() else candidate
        if not descriptor.is_file():
            raise FileNotFoundError(f"baseline descriptor not found: {descriptor}")
        return descriptor.resolve()

    raw = str(reference)
    if candidate.suffix == ".json" or "/" in raw or "\\" in raw:
        raise FileNotFoundError(f"baseline bundle not found: {candidate}")
    discovered = discover_baseline_bundles(baselines_root)
    try:
        return discovered[raw]
    except KeyError as exc:
        available = ", ".join(sorted(discovered)) or "(none)"
        raise ValueError(
            f"unknown baseline ID {raw!r}; discovered: {available}"
        ) from exc


def _fingerprinted_files(
    root: Path, patterns: list[str]
) -> list[dict[str, str]]:
    by_path: dict[str, Path] = {}
    for pattern in patterns:
        matches = sorted(path for path in root.glob(pattern) if path.is_file())
        if not matches:
            raise ValueError(
                f"baseline fingerprint glob matched no files: {pattern}"
            )
        for path in matches:
            resolved = path.resolve()
            try:
                relative = resolved.relative_to(root)
            except ValueError as exc:
                raise ValueError(
                    f"fingerprinted file escapes Baseline bundle: {path}"
                ) from exc
            by_path[relative.as_posix()] = resolved
    return [
        {"path": relative, "sha256": sha256_file(path)}
        for relative, path in sorted(by_path.items())
    ]


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        current = result.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            result[key] = _deep_merge(current, value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _load_local_override(root: Path) -> dict[str, Any]:
    path = root / LOCAL_OVERRIDE_NAME
    if not path.is_file():
        return {}
    value = load_json(path)
    unknown = sorted(set(value) - ALLOWED_LOCAL_OVERRIDE_KEYS)
    if unknown:
        raise ValueError(
            f"{LOCAL_OVERRIDE_NAME} may only override "
            f"{sorted(ALLOWED_LOCAL_OVERRIDE_KEYS)}; found {unknown}"
        )
    for key, item in value.items():
        if not isinstance(item, dict):
            raise ValueError(f"{LOCAL_OVERRIDE_NAME}.{key} must be an object")
    return value


def load_baseline_bundle(
    reference: str | Path,
    *,
    baselines_root: str | Path | None = None,
) -> BaselineBundle:
    descriptor_path = resolve_baseline_descriptor(
        reference, baselines_root=baselines_root
    )
    portable_value = load_json(descriptor_path)
    _validate_manifest(portable_value, descriptor_path)
    root = descriptor_path.parent.resolve()
    implementation = portable_value["implementation"]
    patterns = list(implementation.get("fingerprint_paths", []))
    driver = implementation.get("driver")
    if driver and driver not in patterns:
        patterns.append(driver)
    adapter = portable_value.get("adapter", {})
    adapter_path = (
        adapter.get("entrypoint")
        if adapter.get("kind", "standard") == "python"
        else None
    )
    if adapter_path and adapter_path not in patterns:
        patterns.append(adapter_path)
    if (
        portable_value["schema_version"] == "5.0"
        and "**/*.py" not in patterns
        and any(path.is_file() for path in root.rglob("*.py"))
    ):
        # Bundle-local driver, adapter and helper modules are implementation,
        # not deployment state. Cover transitive imports automatically rather
        # than trusting each Bundle author to enumerate them.
        patterns.append("**/*.py")
    files = _fingerprinted_files(root, patterns)
    bundle_digest = canonical_sha256({
        "manifest": portable_value,
        "files": files,
    })
    resolved_value = _deep_merge(portable_value, _load_local_override(root))
    deployment_digest = canonical_sha256(resolved_value)
    return BaselineBundle(
        root=root,
        value=resolved_value,
        digest=bundle_digest,
        deployment_digest=deployment_digest,
        descriptor_path=descriptor_path,
    )


def load_baseline_plugin(bundle: BaselineBundle) -> BaselinePlugin:
    kind = bundle.value["implementation"]["kind"]
    if kind == "managed":
        from ..baseline_runtime import ManagedBaselinePlugin

        return ManagedBaselinePlugin(bundle)
    if kind == "submission":
        from ..baseline_runtime import SubmissionBaselinePlugin

        return SubmissionBaselinePlugin(bundle)
    raise ValueError(f"unsupported baseline implementation kind {kind!r}")
