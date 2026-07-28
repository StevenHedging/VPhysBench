from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..baseline_api.interfaces import DataAdapter
from ..domain import BaselineBundle
from .bundle_loader import (
    load_bundle_module,
    resolve_bundle_module_path,
)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_adapter(
    adapter: DataAdapter,
    bundle: BaselineBundle,
) -> DataAdapter:
    fingerprint = adapter.fingerprint
    materialization_fingerprint = adapter.materialization_fingerprint
    for label, value in (
        ("fingerprint", fingerprint),
        (
            "materialization_fingerprint",
            materialization_fingerprint,
        ),
    ):
        if not _is_sha256(value):
            raise TypeError(
                f"DataAdapter.{label} must be a lowercase SHA-256 digest"
            )
    description = adapter.describe()
    if not isinstance(description, dict):
        raise TypeError("DataAdapter.describe() must return an object")
    adapter_type = description.get("type")
    if not isinstance(adapter_type, str) or not adapter_type:
        raise TypeError(
            "DataAdapter.describe().type must be a non-empty string"
        )
    expected_identity = {
        "fingerprint": fingerprint,
        "materialization_fingerprint": materialization_fingerprint,
    }
    mismatched = sorted(
        key
        for key, expected in expected_identity.items()
        if description.get(key) != expected
    )
    if mismatched:
        raise TypeError(
            "DataAdapter.describe() identity differs from adapter "
            f"properties: {mismatched}"
        )
    try:
        json.dumps(description, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "DataAdapter.describe() must be JSON-serializable"
        ) from exc
    declared_modes = bundle.value["capabilities"].get(
        "generation_modes"
    )
    described_mode = description.get("generation_mode")
    if (
        described_mode is not None
        and declared_modes is not None
        and described_mode not in declared_modes
    ):
        raise TypeError(
            "DataAdapter.describe().generation_mode is not declared in "
            "Baseline capabilities"
        )
    described_representations = description.get(
        "physics_representations"
    )
    declared_representations = bundle.value["capabilities"].get(
        "physics_representations"
    )
    if described_representations is not None and (
        not isinstance(described_representations, list)
        or any(
            not isinstance(item, str) or not item
            for item in described_representations
        )
        or (
            declared_representations is not None
            and not set(described_representations)
            <= set(declared_representations)
        )
    ):
        raise TypeError(
            "DataAdapter.describe().physics_representations must be a "
            "declared string list"
        )
    dependencies = adapter.dependency_paths()
    if (
        not isinstance(dependencies, dict)
        or any(
            not isinstance(name, str)
            or not name
            or not isinstance(path, Path)
            for name, path in dependencies.items()
        )
    ):
        raise TypeError(
            "DataAdapter.dependency_paths() must map names to Path objects"
        )
    return adapter


def adapter_entrypoint(config: dict[str, Any]) -> str | None:
    """Return the Bundle-local Python adapter entrypoint, when configured."""
    kind = config.get("kind", "standard")
    if kind != "python":
        return None
    value = config.get("entrypoint")
    return value if isinstance(value, str) and value else None


def _entrypoint_path(bundle: BaselineBundle, relative: str) -> Path:
    return resolve_bundle_module_path(
        bundle,
        relative,
        label="adapter.entrypoint",
    )


def load_data_adapter(bundle: BaselineBundle) -> DataAdapter:
    """Construct the Baseline-owned adapter declared by a schema-v4 Bundle.

    Existing v4 manifests remain compatible: an omitted ``adapter.kind`` is
    interpreted as the built-in standard adapter. A Python adapter is trusted
    Bundle code and must expose ``create_adapter(bundle)``.
    """
    config = bundle.value["adapter"]
    kind = config.get("kind", "standard")
    if kind == "standard":
        from .adapter import StandardDataAdapter

        return _validate_adapter(StandardDataAdapter(config), bundle)
    if kind != "python":
        raise ValueError(
            f"unsupported Baseline adapter kind {kind!r}; "
            "expected 'standard' or 'python'"
        )
    relative = adapter_entrypoint(config)
    if relative is None:
        raise ValueError(
            "Python Baseline adapter requires adapter.entrypoint"
        )
    path = _entrypoint_path(bundle, relative)
    module = load_bundle_module(
        bundle,
        relative,
        label="adapter.entrypoint",
    )
    factory = getattr(module, "create_adapter", None)
    if not callable(factory):
        raise TypeError(
            f"Python Baseline adapter must export create_adapter(bundle): {path}"
        )
    adapter = factory(bundle)
    if not isinstance(adapter, DataAdapter):
        raise TypeError(
            "create_adapter(bundle) must return a DataAdapter; "
            f"got {type(adapter).__name__}"
        )
    return _validate_adapter(adapter, bundle)
