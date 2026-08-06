from __future__ import annotations

import copy
import math
import re
from pathlib import Path
from typing import Any

from physbench.baseline_api.interfaces import DataAdapter
from physbench.baseline_runtime.adapter import StandardDataAdapter
from physbench.datasets.physics import flat_physics_quantities
from physbench.io import canonical_sha256, load_json, sha256_file


REPRESENTATION = "symbol_value_cross_attention_v1"
_ASCII_IDENTIFIER = re.compile(r"^[A-Za-z0-9_]+$")


def _symbol_occurrences(prompt: str, symbol: str) -> list[list[int]]:
    if _ASCII_IDENTIFIER.fullmatch(symbol):
        pattern = re.compile(
            rf"(?<![A-Za-z0-9_]){re.escape(symbol)}(?![A-Za-z0-9_])"
        )
        return [[match.start(), match.end()] for match in pattern.finditer(prompt)]
    spans: list[list[int]] = []
    cursor = 0
    while True:
        start = prompt.find(symbol, cursor)
        if start < 0:
            return spans
        end = start + len(symbol)
        spans.append([start, end])
        cursor = end


class SymbolValueRegistry:
    """Select audited scalar quantities without changing the model prompt."""

    def __init__(self, path: Path):
        self.path = path
        self.value = load_json(path)
        self._validate()

    def _validate(self) -> None:
        if self.value.get("schema_version") != "1.0":
            raise ValueError("symbol-value registry requires schema_version=1.0")
        if self.value.get("dimension_basis") != [
            "L", "M", "T", "I", "Theta", "N_amount", "J_luminous"
        ]:
            raise ValueError("symbol-value registry requires seven SI dimensions")
        units = self.value.get("units")
        unit_ids = self.value.get("unit_ids")
        types = self.value.get("quantity_types")
        scenes = self.value.get("scenes")
        if not isinstance(units, dict) or not units:
            raise ValueError("symbol-value registry requires units")
        if set(unit_ids or {}) != set(units):
            raise ValueError("unit_ids must exactly cover units")
        expected_ids = set(range(len(units)))
        if set(unit_ids.values()) != expected_ids:
            raise ValueError("unit_ids must be unique contiguous integers")
        if not isinstance(types, dict) or types.get("unknown") != 0:
            raise ValueError("quantity_types.unknown must equal zero")
        if len(set(types.values())) != len(types):
            raise ValueError("quantity type indices must be unique")
        if not isinstance(scenes, dict) or not scenes:
            raise ValueError("symbol-value registry requires scenes")
        for unit, spec in units.items():
            dimension = spec.get("dimension")
            if (
                not isinstance(dimension, list)
                or len(dimension) != 7
                or any(isinstance(value, bool) or not isinstance(value, int)
                       for value in dimension)
            ):
                raise ValueError(f"unit {unit!r} has invalid SI dimensions")
            scale = spec.get("si_scale")
            if (
                isinstance(scale, bool)
                or not isinstance(scale, (int, float))
                or not math.isfinite(float(scale))
                or float(scale) <= 0
            ):
                raise ValueError(f"unit {unit!r} has invalid SI scale")
        for scene_id, scene in scenes.items():
            parameters = scene.get("parameters")
            if not isinstance(parameters, list) or not parameters:
                raise ValueError(f"scene {scene_id!r} requires parameters")
            names: set[str] = set()
            for parameter in parameters:
                name = parameter.get("name")
                if not isinstance(name, str) or not name or name in names:
                    raise ValueError(f"scene {scene_id!r} has invalid parameter name")
                names.add(name)
                if parameter.get("unit") not in units:
                    raise ValueError(f"parameter {scene_id}/{name} has unknown unit")
                if parameter.get("quantity_type") not in types:
                    raise ValueError(f"parameter {scene_id}/{name} has unknown type")
                precision = parameter.get("precision")
                if isinstance(precision, bool) or not isinstance(precision, int) or precision < 0:
                    raise ValueError(f"parameter {scene_id}/{name} has invalid precision")

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.value)

    def render(
        self,
        case: dict[str, Any],
        base_prompt: str,
    ) -> tuple[str, list[dict[str, Any]], dict[str, dict[str, Any]]]:
        prompt = base_prompt.strip()
        try:
            parameters = self.value["scenes"][case["scene_id"]]["parameters"]
        except KeyError as exc:
            raise ValueError(
                f"symbol-value registry does not support {case['scene_id']}"
            ) from exc
        flat = flat_physics_quantities(case)
        records: list[dict[str, Any]] = []
        used: dict[str, dict[str, Any]] = {}
        for parameter in parameters:
            name = parameter["name"]
            raw = flat.get(name)
            if not isinstance(raw, dict) or "value" not in raw:
                if parameter.get("required", False):
                    if isinstance(raw, dict) and "samples" in raw:
                        raise ValueError(
                            f"required physics.{name} must be a scalar quantity"
                        )
                    raise ValueError(f"required physics quantity missing: {name}")
                continue
            value = raw["value"]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"physics.{name} value must be finite numeric")
            unit = str(raw.get("unit", "")).strip()
            expected_unit = parameter["unit"]
            if unit != expected_unit:
                raise ValueError(
                    f"physics.{name} unit {unit!r} does not match {expected_unit!r}"
                )
            symbol = raw.get("symbol")
            if not isinstance(symbol, str) or not symbol.strip():
                raise ValueError(f"physics.{name} requires a non-empty symbol")
            symbol = symbol.strip()
            spans = _symbol_occurrences(prompt, symbol)
            if not spans:
                raise ValueError(
                    f"physics symbol {symbol!r} is absent from the original prompt"
                )
            precision = parameter["precision"]
            rendered_value = f"{float(value):.{precision}f}"
            rounded_value = float(rendered_value)
            unit_spec = self.value["units"][unit]
            si_value = rounded_value * float(unit_spec["si_scale"])
            if not math.isfinite(si_value):
                raise ValueError(f"physics.{name} SI value must be finite")
            record = {
                "name": name,
                "symbol": symbol,
                "symbol_char_spans": spans,
                "raw_value": value,
                "raw_unit": unit,
                "rendered_value": rendered_value,
                "rendered_quantity": (
                    rendered_value if unit == "1" else f"{rendered_value} {unit}"
                ),
                "si_value": si_value,
                "canonical_si_unit": unit_spec["canonical_si_unit"],
                "dimension": list(unit_spec["dimension"]),
                "dimension_basis": list(self.value["dimension_basis"]),
                "unit_id": self.value["unit_ids"][unit],
                "quantity_type": parameter["quantity_type"],
                "quantity_type_id": self.value["quantity_types"][parameter["quantity_type"]],
                "source_role": parameter["source_role"],
            }
            records.append(record)
            used[name] = {
                "value": value,
                "unit": unit,
                "symbol": symbol,
            }
        if not records:
            raise ValueError(f"case {case['case_id']} has no scalar symbol/value records")
        return prompt, records, used


class SymbolValueCrossAttentionDataAdapter(DataAdapter):
    def __init__(self, bundle):
        self.bundle = bundle
        self.config = copy.deepcopy(bundle.value["adapter"]["config"])
        self.registry = SymbolValueRegistry(
            Path(__file__).resolve().parent / self.config["quantity_registry"]
        )
        standard_config = {
            key: copy.deepcopy(self.config[key])
            for key in (
                "preset", "first_frame_policy", "spatial", "temporal", "cache_policy"
            )
            if key in self.config
        }
        standard_config["physics_transform"] = {"type": "none"}
        ignored_policy = copy.deepcopy(bundle.value["input_policy"])
        ignored_policy["physics"] = {
            "source": "case.physics", "usage": "ignored", "representations": []
        }
        self.standard = StandardDataAdapter(standard_config, ignored_policy)

    @property
    def fingerprint(self) -> str:
        return canonical_sha256({
            "type": "wan22_symbol_value_cross_attention_adapter_v1",
            "input_policy": self.bundle.value["input_policy"],
            "config": self.config,
            "registry": self.registry.value,
            "standard_materialization": self.standard.materialization_fingerprint,
            "implementation": sha256_file(Path(__file__)),
        })

    @property
    def materialization_fingerprint(self) -> str:
        return self.standard.materialization_fingerprint

    def dependency_paths(self) -> dict[str, Path]:
        shared = (
            Path(__file__).resolve().parents[2]
            / "src" / "physbench" / "baseline_runtime" / "adapter.py"
        )
        return {"src/physbench/baseline_runtime/adapter.py": shared}

    def describe(self) -> dict[str, Any]:
        return {
            "type": "wan22_symbol_value_cross_attention_adapter_v1",
            "fingerprint": self.fingerprint,
            "materialization_fingerprint": self.materialization_fingerprint,
            "generation_mode": "i2v",
            "input_policy": self.bundle.value["input_policy"],
            "physics_representations": [REPRESENTATION],
            "text_conditioning_required": True,
            "quantity_registry_id": self.registry.value["registry_id"],
            "quantity_registry_fingerprint": self.registry.fingerprint,
            "dimension_basis": self.registry.value["dimension_basis"],
            "stages": [
                "spatial", "temporal", "i2v_first_frame",
                "structured_symbol_value_selection", "model_side_cross_attention",
            ],
        }

    def adapt_case(self, case: dict[str, Any], *, role: str) -> dict[str, Any]:
        base = self.standard.adapt_case(case, role=role)
        source_prompt = case["text"]["prompt"].strip()
        prompt, records, used = self.registry.render(case, source_prompt)
        base.update({
            "source_prompt_sha256": canonical_sha256(source_prompt),
            "prompt_sha256": canonical_sha256(prompt),
            "text_transform_id": "identity_symbol_audited_v1",
            "prompt": prompt,
            "used_parameters": used,
            "data_adapter_fingerprint": self.fingerprint,
            "materialization_fingerprint": self.materialization_fingerprint,
        })
        base["native_inputs"]["text"] = {"prompt": prompt}
        base["native_inputs"]["physics"] = {
            "representation": REPRESENTATION,
            "registry_id": self.registry.value["registry_id"],
            "registry_fingerprint": self.registry.fingerprint,
            "quantities": records,
        }
        base["stages"]["text"] = {
            "type": "dataset_prompt_identity_v1",
            "source": "case.text.prompt",
            "source_prompt_sha256": canonical_sha256(source_prompt),
            "model_prompt_sha256": canonical_sha256(prompt),
            "numeric_literals_appended": False,
        }
        base["stages"]["physics"] = {
            "usage": "required",
            "strategy": REPRESENTATION,
            "registry_id": self.registry.value["registry_id"],
            "registry_fingerprint": self.registry.fingerprint,
            "used_parameters": used,
            "fusion_stage": "post_frozen_umt5_pre_dit",
        }
        base["input_contract"]["physics_channels"] = [{
            "id": "symbol_value_tokens",
            "representation": REPRESENTATION,
            "binding": "native_inputs.physics.quantities",
            "transport": "inline_json",
            "used_parameters": sorted(used),
        }]
        return base


def create_adapter(bundle):
    return SymbolValueCrossAttentionDataAdapter(bundle)
