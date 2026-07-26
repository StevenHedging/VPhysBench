from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

from ..domain import BaselineBundle
from ..io import canonical_sha256, load_json, sha256_file
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
    "model",
    "runtime",
}
V3_MANIFEST_KEYS = {
    *COMMON_MANIFEST_KEYS,
    "components",
}
V4_MANIFEST_KEYS = {
    *COMMON_MANIFEST_KEYS,
    "adapter",
    "runner",
    "trainer",
}
V3_IMPLEMENTATION_KEYS = {
    "kind",
    "protocol",
    "entrypoint",
    "fingerprint_paths",
}
V4_IMPLEMENTATION_KEYS = {
    "kind",
    "driver",
    "fingerprint_paths",
}
ALLOWED_CAPABILITY_KEYS = {
    "task_families",
    "conditioning",
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


def _validate_command_implementation(
    implementation: dict[str, Any],
    descriptor_path: Path,
) -> None:
    unknown = sorted(set(implementation) - V3_IMPLEMENTATION_KEYS)
    if unknown:
        raise ValueError(
            f"baseline implementation contains unknown fields: {unknown}"
        )
    if implementation.get("protocol") != "physbench-baseline-v1":
        raise ValueError(
            "command baseline must use protocol=physbench-baseline-v1"
        )
    entrypoint = implementation.get("entrypoint")
    if (
        not isinstance(entrypoint, list)
        or not entrypoint
        or any(
            not isinstance(item, str) or not item or "\x00" in item
            for item in entrypoint
        )
    ):
        raise ValueError(
            "implementation.entrypoint must be a non-empty string list"
        )
    if entrypoint[0] == "{python}":
        if len(entrypoint) < 2:
            raise ValueError(
                "{python} entrypoint requires a bundle-local script"
            )
        script = _validate_relative_path(
            descriptor_path.parent,
            entrypoint[1],
            label="implementation.entrypoint script",
        )
        if not script.is_file():
            raise FileNotFoundError(
                f"baseline entrypoint not found: {script}"
            )
    else:
        executable = _validate_relative_path(
            descriptor_path.parent,
            entrypoint[0],
            label="implementation.entrypoint executable",
        )
        if not executable.is_file():
            raise FileNotFoundError(
                f"baseline entrypoint not found: {executable}"
            )
    _validate_fingerprint_patterns(implementation, required=True)


def _validate_v4_implementation(
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
            "schema v4 baseline implementation kind must be managed or "
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
    if schema_version not in {"3.0", "4.0"}:
        raise ValueError(
            "baseline bundle must use schema_version=3.0 or 4.0"
        )
    allowed = V3_MANIFEST_KEYS if schema_version == "3.0" else V4_MANIFEST_KEYS
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"baseline bundle contains unknown fields: {unknown}")
    _validate_baseline_id(value)
    _require_string(value, "baseline_version")

    implementation = value.get("implementation")
    if not isinstance(implementation, dict):
        raise ValueError("baseline bundle requires an implementation object")
    kind = implementation.get("kind")
    if schema_version == "3.0" and kind != "command":
        raise ValueError(
            "schema v3 baseline implementation kind must be command, "
            f"got {kind!r}"
        )
    if schema_version == "3.0":
        _validate_command_implementation(implementation, descriptor_path)
    else:
        _validate_v4_implementation(implementation, descriptor_path)

    capabilities = value.get("capabilities")
    if not isinstance(capabilities, dict):
        raise ValueError("baseline bundle requires a capabilities object")
    unknown = sorted(set(capabilities) - ALLOWED_CAPABILITY_KEYS)
    if unknown:
        raise ValueError(
            f"baseline capabilities contains unknown fields: {unknown}"
        )
    for key in ("task_families", "conditioning"):
        items = capabilities.get(key)
        if (
            not isinstance(items, list)
            or not items
            or any(not isinstance(item, str) or not item for item in items)
        ):
            raise ValueError(f"capabilities.{key} must be a non-empty string list")
        if len(items) != len(set(items)):
            raise ValueError(f"capabilities.{key} contains duplicates")
    for key in ("train", "finetune", "generate"):
        if key in capabilities and not isinstance(capabilities[key], bool):
            raise ValueError(f"capabilities.{key} must be a boolean")

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

    if schema_version == "3.0":
        components = value.get("components")
        if components is not None and not isinstance(components, dict):
            raise ValueError(
                "baseline components must be an object when present"
            )
        if isinstance(components, dict) and "condition_adapter" in components:
            raise ValueError(
                "condition_adapter is not a public Baseline component; keep "
                "all model-input adaptation inside the Baseline-owned data "
                "adapter"
            )
    else:
        adapter = value.get("adapter")
        if not isinstance(adapter, dict):
            raise ValueError(
                "schema v4 baseline requires an adapter object"
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
    for descriptor_path in sorted(root.glob(f"*/{DESCRIPTOR_NAME}")):
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
    if kind == "command":
        from .command import CommandBaselinePlugin

        return CommandBaselinePlugin(bundle)
    if kind == "managed":
        from ..baseline_runtime import ManagedBaselinePlugin

        return ManagedBaselinePlugin(bundle)
    if kind == "submission":
        from ..baseline_runtime import SubmissionBaselinePlugin

        return SubmissionBaselinePlugin(bundle)
    raise ValueError(f"unsupported baseline implementation kind {kind!r}")
