from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from physbench.baseline_api.interfaces import DataAdapter
from physbench.baseline_runtime.adapter import StandardDataAdapter
from physbench.io import canonical_sha256, load_json, sha256_file


REPRESENTATION = "quantity_token_embedding_v1"


class QuantityRegistry:
    """Render compact quantity spans and freeze their SI interpretation."""

    def __init__(self, path: Path):
        self.path = path
        self.value = load_json(path)
        if self.value.get("schema_version") != "1.0":
            raise ValueError("quantity registry must use schema_version=1.0")
        basis = self.value.get("dimension_basis")
        if basis != [
            "L",
            "M",
            "T",
            "I",
            "Theta",
            "N_amount",
            "J_luminous",
        ]:
            raise ValueError("quantity registry requires the seven SI dimensions")
        units = self.value.get("units")
        types = self.value.get("quantity_types")
        scenes = self.value.get("scenes")
        if not isinstance(units, dict) or not units:
            raise ValueError("quantity registry requires units")
        if not isinstance(types, dict) or types.get("unknown") != 0:
            raise ValueError("quantity registry requires quantity_types.unknown=0")
        if len(set(types.values())) != len(types):
            raise ValueError("quantity type indices must be unique")
        if not isinstance(scenes, dict) or not scenes:
            raise ValueError("quantity registry requires scenes")
        for unit, spec in units.items():
            dimension = spec.get("dimension")
            if (
                not isinstance(dimension, list)
                or len(dimension) != 7
                or any(not isinstance(value, int) for value in dimension)
            ):
                raise ValueError(f"unit {unit!r} has an invalid SI dimension")
            scale = spec.get("si_scale")
            if not isinstance(scale, (int, float)) or scale <= 0:
                raise ValueError(f"unit {unit!r} has an invalid SI scale")
        for scene_id, scene in scenes.items():
            parameters = scene.get("parameters")
            if not isinstance(parameters, list) or not parameters:
                raise ValueError(f"scene {scene_id!r} requires parameters")
            names: set[str] = set()
            for parameter in parameters:
                name = parameter.get("name")
                if not isinstance(name, str) or not name or name in names:
                    raise ValueError(
                        f"scene {scene_id!r} has an invalid parameter name"
                    )
                names.add(name)
                if parameter.get("unit") not in units:
                    raise ValueError(
                        f"parameter {scene_id}/{name} uses an unknown unit"
                    )
                if parameter.get("quantity_type") not in types:
                    raise ValueError(
                        f"parameter {scene_id}/{name} has an unknown type"
                    )
                if "{quantity}" not in str(parameter.get("template", "")):
                    raise ValueError(
                        f"parameter {scene_id}/{name} requires {{quantity}}"
                    )
                precision = parameter.get("precision")
                if not isinstance(precision, int) or precision < 0:
                    raise ValueError(
                        f"parameter {scene_id}/{name} has invalid precision"
                    )

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.value)

    def render(
        self,
        case: dict[str, Any],
        base_prompt: str,
    ) -> tuple[
        str,
        str,
        list[dict[str, Any]],
        dict[str, dict[str, Any]],
    ]:
        try:
            scene = self.value["scenes"][case["scene_id"]]
        except KeyError as exc:
            raise ValueError(
                f"quantity registry does not support {case['scene_id']}"
            ) from exc
        clauses: list[tuple[str, dict[str, Any]]] = []
        used: dict[str, dict[str, Any]] = {}
        for parameter in scene["parameters"]:
            name = parameter["name"]
            raw = case["physics"].get(name)
            usable = (
                isinstance(raw, dict)
                and raw.get("annotated") is True
                and raw.get("value") is not None
            )
            if not usable:
                if parameter.get("required", False):
                    raise ValueError(
                        f"case {case['case_id']} lacks required physics.{name}"
                    )
                continue
            value = raw["value"]
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
            ):
                raise ValueError(
                    f"case {case['case_id']} physics.{name} must be numeric"
                )
            unit = str(raw.get("unit", "")).strip()
            if unit != parameter["unit"]:
                raise ValueError(
                    f"case {case['case_id']} physics.{name} unit {unit!r} "
                    f"!= expected {parameter['unit']!r}"
                )
            rendered_value = f"{float(value):.{parameter['precision']}f}"
            rendered_quantity = (
                rendered_value if unit == "1" else f"{rendered_value} {unit}"
            )
            unit_spec = self.value["units"][unit]
            quantity = {
                "name": name,
                "raw_value": value,
                "raw_unit": unit,
                "rendered_value": rendered_value,
                "rendered_quantity": rendered_quantity,
                "si_value": (
                    float(rendered_value) * float(unit_spec["si_scale"])
                ),
                "canonical_si_unit": unit_spec["canonical_si_unit"],
                "dimension": list(unit_spec["dimension"]),
                "dimension_basis": list(self.value["dimension_basis"]),
                "quantity_type": parameter["quantity_type"],
                "quantity_type_id": self.value["quantity_types"][
                    parameter["quantity_type"]
                ],
                "source_role": parameter["source_role"],
            }
            clause = parameter["template"].format(
                quantity=rendered_quantity
            )
            clauses.append((clause, quantity))
            used[name] = {"value": value, "unit": unit}

        if not clauses:
            raise ValueError(
                f"case {case['case_id']} has no registry-backed quantities"
            )
        prefix = base_prompt.rstrip() + str(scene["parameter_intro"])
        separator = str(scene.get("parameter_separator", "; "))
        audited_parts = [prefix]
        model_parts = [prefix]
        quantities: list[dict[str, Any]] = []
        audited_cursor = len(prefix)
        model_cursor = len(prefix)
        for index, (clause, quantity) in enumerate(clauses):
            if index:
                audited_parts.append(separator)
                model_parts.append(separator)
                audited_cursor += len(separator)
                model_cursor += len(separator)
            local_start = clause.index(quantity["rendered_quantity"])
            audited_start = audited_cursor + local_start
            audited_end = (
                audited_start + len(quantity["rendered_quantity"])
            )
            sentinel = f"<extra_id_{index}>"
            model_clause = (
                clause[:local_start]
                + sentinel
                + clause[local_start + len(quantity["rendered_quantity"]):]
            )
            model_start = model_cursor + local_start
            model_end = model_start + len(sentinel)
            item = dict(quantity)
            item["sentinel"] = sentinel
            item["audited_char_span"] = [audited_start, audited_end]
            item["model_char_span"] = [model_start, model_end]
            quantities.append(item)
            audited_parts.append(clause)
            model_parts.append(model_clause)
            audited_cursor += len(clause)
            model_cursor += len(model_clause)
        outro = str(scene.get("parameter_outro", "."))
        audited_parts.append(outro)
        model_parts.append(outro)
        audited_prompt = "".join(audited_parts)
        model_prompt = "".join(model_parts)
        for item in quantities:
            start, end = item["audited_char_span"]
            if audited_prompt[start:end] != item["rendered_quantity"]:
                raise AssertionError(
                    "audited quantity character span is inconsistent"
                )
            start, end = item["model_char_span"]
            if model_prompt[start:end] != item["sentinel"]:
                raise AssertionError(
                    "model sentinel character span is inconsistent"
                )
        return audited_prompt, model_prompt, quantities, used


class QuantityEmbeddingDataAdapter(DataAdapter):
    """I2V media adaptation plus a model-side structured quantity channel."""

    def __init__(self, bundle):
        self.bundle = bundle
        self.config = copy.deepcopy(bundle.value["adapter"]["config"])
        registry_name = self.config["quantity_registry"]
        self.registry = QuantityRegistry(
            Path(__file__).resolve().parent / registry_name
        )
        standard_config = {
            key: copy.deepcopy(self.config[key])
            for key in (
                "preset",
                "first_frame_policy",
                "spatial",
                "temporal",
                "cache_policy",
            )
            if key in self.config
        }
        standard_config["physics_transform"] = {"type": "none"}
        ignored_policy = copy.deepcopy(bundle.value["input_policy"])
        ignored_policy["physics"] = {
            "source": "case.physics[annotated=true]",
            "usage": "ignored",
            "representations": [],
        }
        self.standard = StandardDataAdapter(
            standard_config,
            ignored_policy,
        )

    @property
    def fingerprint(self) -> str:
        return canonical_sha256({
            "type": "wan22_quantity_embedding_adapter_v1",
            "input_policy": self.bundle.value["input_policy"],
            "config": self.config,
            "registry": self.registry.value,
            "standard_materialization": (
                self.standard.materialization_fingerprint
            ),
            "implementation": sha256_file(Path(__file__)),
        })

    @property
    def materialization_fingerprint(self) -> str:
        return self.standard.materialization_fingerprint

    def dependency_paths(self) -> dict[str, Path]:
        shared = (
            Path(__file__).resolve().parents[2]
            / "src"
            / "physbench"
            / "baseline_runtime"
            / "adapter.py"
        )
        return {
            "src/physbench/baseline_runtime/adapter.py": shared,
        }

    def describe(self) -> dict[str, Any]:
        return {
            "type": "wan22_quantity_embedding_adapter_v1",
            "fingerprint": self.fingerprint,
            "materialization_fingerprint": (
                self.materialization_fingerprint
            ),
            "generation_mode": "i2v",
            "input_policy": self.bundle.value["input_policy"],
            "physics_representations": [REPRESENTATION],
            "text_conditioning_required": True,
            "quantity_registry_id": self.registry.value["registry_id"],
            "quantity_registry_fingerprint": self.registry.fingerprint,
            "dimension_basis": self.registry.value["dimension_basis"],
            "stages": [
                "spatial",
                "temporal",
                "i2v_first_frame",
                "structured_quantity_rendering",
                "model_side_quantity_embedding",
            ],
        }

    def adapt_case(
        self,
        case: dict[str, Any],
        *,
        role: str,
    ) -> dict[str, Any]:
        base = self.standard.adapt_case(case, role=role)
        source_prompt = case["text"]["prompt"].strip()
        audited_prompt, prompt, quantities, used = self.registry.render(
            case,
            source_prompt,
        )
        base.update({
            "source_prompt_sha256": canonical_sha256(source_prompt),
            "prompt_sha256": canonical_sha256(prompt),
            "audited_prompt_sha256": canonical_sha256(audited_prompt),
            "text_transform_id": "append_quantity_spans_v1",
            "prompt": prompt,
            "used_parameters": used,
            "data_adapter_fingerprint": self.fingerprint,
            "materialization_fingerprint": (
                self.materialization_fingerprint
            ),
        })
        base["native_inputs"]["text"] = {
            "prompt": prompt,
            "audited_prompt": audited_prompt,
        }
        base["native_inputs"]["physics"] = {
            "representation": REPRESENTATION,
            "registry_id": self.registry.value["registry_id"],
            "registry_fingerprint": self.registry.fingerprint,
            "quantities": quantities,
        }
        base["stages"]["text"] = {
            "type": "dataset_prompt_plus_quantity_clauses_v1",
            "source": "case.text.prompt",
            "source_prompt_sha256": canonical_sha256(source_prompt),
            "audited_prompt_sha256": canonical_sha256(audited_prompt),
            "model_prompt_sha256": canonical_sha256(prompt),
            "literal_to_sentinel": True,
        }
        base["stages"]["physics"] = {
            "usage": "required",
            "strategy": REPRESENTATION,
            "registry_id": self.registry.value["registry_id"],
            "registry_fingerprint": self.registry.fingerprint,
            "used_parameters": used,
            "span_stage": "post_frozen_text_encoder_pre_dit",
        }
        base["input_contract"]["physics_channels"] = [{
            "id": "quantity_token_embeddings",
            "representation": REPRESENTATION,
            "binding": "native_inputs.physics.quantities",
            "transport": "inline_json",
            "used_parameters": sorted(used),
        }]
        return base


def create_adapter(bundle):
    return QuantityEmbeddingDataAdapter(bundle)
