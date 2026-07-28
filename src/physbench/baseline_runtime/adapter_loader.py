from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

from ..baseline_api.interfaces import DataAdapter
from ..domain import BaselineBundle


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_adapter(adapter: DataAdapter) -> DataAdapter:
    for label, value in (
        ("fingerprint", adapter.fingerprint),
        (
            "materialization_fingerprint",
            adapter.materialization_fingerprint,
        ),
    ):
        if not _is_sha256(value):
            raise TypeError(
                f"DataAdapter.{label} must be a lowercase SHA-256 digest"
            )
    description = adapter.describe()
    if not isinstance(description, dict):
        raise TypeError("DataAdapter.describe() must return an object")
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
    value = Path(relative)
    if value.is_absolute() or ".." in value.parts:
        raise ValueError(
            f"adapter.entrypoint must be a bundle-relative path: {relative}"
        )
    path = (bundle.root / value).resolve()
    try:
        path.relative_to(bundle.root)
    except ValueError as exc:
        raise ValueError(
            f"adapter.entrypoint escapes the Baseline bundle: {relative}"
        ) from exc
    if not path.is_file():
        raise FileNotFoundError(f"Baseline adapter entrypoint not found: {path}")
    return path


def _load_module(path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load Baseline adapter module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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

        return _validate_adapter(StandardDataAdapter(config))
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
    module = _load_module(
        path,
        f"_physbench_adapter_{bundle.baseline_id}_{bundle.digest[:12]}",
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
    return _validate_adapter(adapter)
