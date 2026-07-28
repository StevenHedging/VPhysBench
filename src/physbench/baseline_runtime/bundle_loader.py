"""Import trusted Python modules from a Baseline Bundle namespace."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from ..domain import BaselineBundle


def resolve_bundle_module_path(
    bundle: BaselineBundle,
    relative: str,
    *,
    label: str,
) -> Path:
    value = Path(relative)
    if value.is_absolute() or ".." in value.parts:
        raise ValueError(f"{label} must be a bundle-relative path: {relative}")
    path = (bundle.root / value).resolve()
    try:
        relative_path = path.relative_to(bundle.root)
    except ValueError as exc:
        raise ValueError(
            f"{label} escapes the Baseline bundle: {relative}"
        ) from exc
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    module_parts = [*relative_path.parts[:-1], relative_path.stem]
    if path.suffix != ".py" or any(
        not part.isidentifier() for part in module_parts
    ):
        raise ValueError(
            f"{label} must be a Python module path whose components are "
            f"identifiers: {relative}"
        )
    return path


def _namespace(bundle: BaselineBundle) -> str:
    root_digest = hashlib.sha256(
        str(bundle.root.resolve()).encode("utf-8")
    ).hexdigest()
    return (
        f"_physbench_bundle_{bundle.digest[:20]}_"
        f"{bundle.deployment_digest[:16]}_{root_digest[:16]}"
    )


def _create_root_package(
    bundle: BaselineBundle,
    package_name: str,
) -> ModuleType:
    initializer = bundle.root / "__init__.py"
    if initializer.is_file():
        spec = importlib.util.spec_from_file_location(
            package_name,
            initializer,
            submodule_search_locations=[str(bundle.root)],
        )
        if spec is None or spec.loader is None:
            raise ImportError(
                f"cannot load Baseline package initializer: {initializer}"
            )
        package = importlib.util.module_from_spec(spec)
        sys.modules[package_name] = package
        spec.loader.exec_module(package)
        return package

    package = ModuleType(package_name)
    package.__package__ = package_name
    package.__path__ = [str(bundle.root)]  # type: ignore[attr-defined]
    spec = importlib.util.spec_from_loader(
        package_name,
        loader=None,
        is_package=True,
    )
    if spec is not None:
        spec.submodule_search_locations = [str(bundle.root)]
    package.__spec__ = spec
    sys.modules[package_name] = package
    return package


def load_bundle_module(
    bundle: BaselineBundle,
    relative: str,
    *,
    label: str,
) -> ModuleType:
    """Load one Bundle-local module with ordinary relative-import semantics.

    The synthetic package name includes both portable content identity and the
    resolved Bundle root. Identical Bundles copied to different directories
    therefore cannot accidentally reuse one another's imported helpers.
    """

    path = resolve_bundle_module_path(
        bundle,
        relative,
        label=label,
    )
    relative_path = path.relative_to(bundle.root)
    module_parts = [*relative_path.parts[:-1], relative_path.stem]
    package_name = _namespace(bundle)
    module_name = ".".join([package_name, *module_parts])
    prefix = f"{package_name}."
    previous = {
        name
        for name in sys.modules
        if name == package_name or name.startswith(prefix)
    }
    try:
        importlib.invalidate_caches()
        if package_name not in sys.modules:
            _create_root_package(bundle, package_name)
        if module_parts[:-1]:
            importlib.import_module(
                ".".join([package_name, *module_parts[:-1]])
            )
        module = importlib.import_module(module_name)
        loaded_path = Path(getattr(module, "__file__", "")).resolve()
        if loaded_path != path:
            raise ImportError(
                f"{label} resolved to unexpected module path: {loaded_path}"
            )
        return module
    except BaseException:
        for name in list(sys.modules):
            if (
                (name == package_name or name.startswith(prefix))
                and name not in previous
            ):
                sys.modules.pop(name, None)
        raise


__all__ = [
    "load_bundle_module",
    "resolve_bundle_module_path",
]
