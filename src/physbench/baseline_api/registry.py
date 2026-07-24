from __future__ import annotations

from pathlib import Path

from ..domain import BaselineBundle
from ..io import canonical_sha256, load_json
from .interfaces import BaselinePlugin


def load_baseline_bundle(path: str | Path) -> BaselineBundle:
    descriptor_path = Path(path).resolve()
    value = load_json(descriptor_path)
    if value.get("schema_version") != "2.0":
        raise ValueError("baseline bundle must use schema_version=2.0")
    required = {"baseline_id", "plugin", "capabilities", "components"}
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"baseline bundle missing {missing}")
    components = value["components"]
    required_components = {"task_builder", "trainer", "predictor"}
    missing_components = sorted(required_components - set(components))
    if missing_components:
        raise ValueError(
            f"baseline bundle components missing {missing_components}"
        )
    if "condition_adapter" in components:
        raise ValueError(
            "condition_adapter is not a v2 component; merge text and physics "
            "adaptation into the baseline-owned data_adapter"
        )
    if "data_adapter" in components:
        raise ValueError(
            "data_adapter is not a top-level v2 component; encapsulate it in "
            "the baseline-owned task_builder"
        )
    return BaselineBundle(descriptor_path.parent, value, canonical_sha256(value))


def load_baseline_plugin(bundle: BaselineBundle) -> BaselinePlugin:
    if bundle.value["plugin"] == "wan22_lora":
        from ..baseline_plugins.wan22 import Wan22BaselinePlugin

        return Wan22BaselinePlugin(bundle)
    raise ValueError(f"unknown baseline plugin {bundle.value['plugin']}")
