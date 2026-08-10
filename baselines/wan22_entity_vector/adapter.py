from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

from physbench.baseline_api.interfaces import DataAdapter
from physbench.baseline_runtime.adapter import StandardDataAdapter
from physbench.io import canonical_sha256, sha256_file


REPRESENTATION = "first_entity_vector_mlp_v1"
ENTITY_VECTOR_COMPONENTS = (
    "mass_kg",
    "size_m",
    "initial_velocity_m_per_s",
)
_SIZE_FIELDS = ("radius", "length", "height", "orbit_radius")
_VELOCITY_FIELDS = ("initial_velocity", "initial_horizontal_velocity")
_PROJECTED_OBJECT_ONE_FIELDS = {
    "collision_1d": {
        "mass": "ball_1_mass",
        "radius": "ball_1_radius",
        "initial_velocity": "ball_1_initial_velocity",
    },
    "inclined_plane_slide": {
        "mass": "block_mass",
        "length": "block_length",
    },
    "parabolic_motion": {
        "mass": "ball_mass",
        "radius": "ball_radius",
        "initial_horizontal_velocity": "initial_horizontal_velocity",
    },
    "pendulum": {
        "mass": "bob_mass",
        "radius": "bob_radius",
    },
    "push_bottle": {
        "mass": "bottle_mass",
        "height": "bottle_height",
    },
    "uniform_circular_motion": {
        "orbit_radius": "object_1_orbit_radius",
    },
    "vertical_spring_oscillator": {
        "mass": "oscillator_mass",
        "radius": "ball_radius",
    },
}


def _read_nonnegative_quantity(
    record: dict[str, Any],
    field: str,
    expected_unit: str,
    *,
    case_id: str,
) -> float:
    quantity = record[field]
    if not isinstance(quantity, dict):
        raise ValueError(
            f"case {case_id} physics.objects.object_1.{field} must be an object"
        )
    value = quantity.get("value")
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
    ):
        raise ValueError(
            f"case {case_id} physics.objects.object_1.{field}.value "
            "must be numeric"
        )
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0:
        raise ValueError(
            f"case {case_id} physics.objects.object_1.{field}.value "
            "must be finite and nonnegative"
        )
    unit = quantity.get("unit")
    if unit != expected_unit:
        raise ValueError(
            f"case {case_id} physics.objects.object_1.{field}.unit "
            f"must be {expected_unit!r}, got {unit!r}"
        )
    return numeric


def _read_angular_velocity(
    environment: dict[str, Any],
    *,
    case_id: str,
) -> float:
    quantity = environment.get("angular_velocity")
    if not isinstance(quantity, dict):
        raise ValueError(
            f"case {case_id} circular velocity derivation requires "
            "physics.environment.angular_velocity"
        )
    value = quantity.get("value")
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
    ):
        raise ValueError(
            f"case {case_id} angular_velocity.value must be numeric"
        )
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0:
        raise ValueError(
            f"case {case_id} angular_velocity.value must be finite and "
            "nonnegative"
        )
    unit = quantity.get("unit")
    if unit == "deg/s":
        return math.radians(numeric)
    if unit == "rad/s":
        return numeric
    raise ValueError(
        f"case {case_id} angular_velocity.unit must be 'deg/s' or "
        f"'rad/s', got {unit!r}"
    )


def extract_first_entity_vector(case: dict[str, Any]) -> dict[str, Any]:
    """Return ``[mass, size, initial velocity]`` for ``object_1`` in SI."""

    case_id = str(case.get("case_id", "<unknown>"))
    physics = case.get("physics")
    if not isinstance(physics, dict):
        raise ValueError(f"case {case_id} requires physics")
    if set(physics) == {"objects", "environment"}:
        objects = physics.get("objects")
        if not isinstance(objects, dict):
            raise ValueError(f"case {case_id} requires physics.objects")
        object_1 = objects.get("object_1")
        if not isinstance(object_1, dict):
            raise ValueError(
                f"case {case_id} requires physics.objects.object_1"
            )
        environment = physics.get("environment", {})
        if not isinstance(environment, dict):
            raise ValueError(
                f"case {case_id} physics.environment must be an object"
            )
    else:
        scene_id = str(case.get("scene_id"))
        try:
            projected_fields = _PROJECTED_OBJECT_ONE_FIELDS[scene_id]
        except KeyError as exc:
            raise ValueError(
                f"case {case_id} has no object_1 projection for {scene_id}"
            ) from exc
        object_1 = {
            field: physics[semantic_name]
            for field, semantic_name in projected_fields.items()
            if semantic_name in physics
        }
        environment = {
            "angular_velocity": physics["angular_velocity"]
        } if "angular_velocity" in physics else {}

    source_fields: list[str | None] = []
    provenance: list[str] = []

    if "mass" in object_1:
        mass = _read_nonnegative_quantity(
            object_1, "mass", "kg", case_id=case_id
        )
        source_fields.append("mass")
        provenance.append("observed")
    else:
        mass = 0.0
        source_fields.append(None)
        provenance.append("imputed_zero")

    size_field = next(
        (field for field in _SIZE_FIELDS if field in object_1),
        None,
    )
    if size_field is None:
        size = 0.0
        source_fields.append(None)
        provenance.append("imputed_zero")
    else:
        size = _read_nonnegative_quantity(
            object_1, size_field, "m", case_id=case_id
        )
        source_fields.append(size_field)
        provenance.append("observed")

    velocity_field = next(
        (field for field in _VELOCITY_FIELDS if field in object_1),
        None,
    )
    if velocity_field is not None:
        velocity = _read_nonnegative_quantity(
            object_1, velocity_field, "m/s", case_id=case_id
        )
        source_fields.append(velocity_field)
        provenance.append("observed")
    elif case.get("scene_id") == "uniform_circular_motion":
        if size_field != "orbit_radius":
            raise ValueError(
                f"case {case_id} circular velocity derivation requires "
                "object_1.orbit_radius"
            )
        velocity = size * _read_angular_velocity(
            environment,
            case_id=case_id,
        )
        source_fields.append("orbit_radius*angular_velocity")
        provenance.append("derived")
    else:
        velocity = 0.0
        source_fields.append(None)
        provenance.append("imputed_rest")

    values = [mass, size, velocity]
    if len(values) != len(ENTITY_VECTOR_COMPONENTS) or not all(
        math.isfinite(value) and value >= 0 for value in values
    ):
        raise AssertionError("first-entity vector invariant failed")
    return {
        "schema_version": "1.0",
        "representation": REPRESENTATION,
        "components": list(ENTITY_VECTOR_COMPONENTS),
        "values_si": values,
        "sentinel": "<extra_id_0>",
        "source_object": "object_1",
        "source_fields": source_fields,
        "provenance": provenance,
    }


class FirstEntityVectorDataAdapter(DataAdapter):
    """Append seven-scene physics text and one first-object vector token."""

    def __init__(self, bundle):
        self.bundle = bundle
        adapter_value = copy.deepcopy(bundle.value["adapter"])
        self.config = copy.deepcopy(
            adapter_value.get("config", adapter_value)
        )
        standard_config = {
            key: copy.deepcopy(self.config[key])
            for key in (
                "preset",
                "first_frame_policy",
                "spatial",
                "temporal",
                "cache_policy",
                "physics_transform",
            )
            if key in self.config
        }
        transform = standard_config.get("physics_transform")
        if transform != {
            "type": "append_structured_text_v1",
            "template_set": "seven_scene_physics_text_v1",
        }:
            raise ValueError(
                "first-entity vector adapter requires the frozen "
                "seven_scene_physics_text_v1 transform"
            )
        structured_policy = copy.deepcopy(bundle.value["input_policy"])
        structured_policy["physics"] = {
            "source": "case.physics",
            "usage": "required",
            "representations": ["structured_text"],
        }
        self.standard = StandardDataAdapter(
            standard_config,
            structured_policy,
        )

    @property
    def fingerprint(self) -> str:
        return canonical_sha256({
            "type": "wan22_first_entity_vector_adapter_v1",
            "input_policy": self.bundle.value["input_policy"],
            "config": self.config,
            "standard_fingerprint": self.standard.fingerprint,
            "implementation": sha256_file(Path(__file__)),
        })

    @property
    def materialization_fingerprint(self) -> str:
        return self.standard.materialization_fingerprint

    def dependency_paths(self) -> dict[str, Path]:
        return self.standard.dependency_paths()

    def describe(self) -> dict[str, Any]:
        return {
            "type": "wan22_first_entity_vector_adapter_v1",
            "fingerprint": self.fingerprint,
            "materialization_fingerprint": self.materialization_fingerprint,
            "generation_mode": "i2v",
            "input_policy": self.bundle.value["input_policy"],
            "physics_representations": [
                "structured_text",
                REPRESENTATION,
            ],
            "text_conditioning_required": True,
            "vector_components": list(ENTITY_VECTOR_COMPONENTS),
            "vector_source_object": "object_1",
            "stages": [
                "spatial",
                "temporal",
                "i2v_first_frame",
                "structured_physics_text",
                "first_entity_vector_embedding",
            ],
        }

    def adapt_case(
        self,
        case: dict[str, Any],
        *,
        role: str,
    ) -> dict[str, Any]:
        base = self.standard.adapt_case(case, role=role)
        vector = extract_first_entity_vector(case)
        mass, size, velocity = vector["values_si"]
        audited_clause = (
            "\nFirst entity vector: "
            f"[m={mass:.5f} kg, size={size:.5f} m, "
            f"v={velocity:.6f} m/s]."
        )
        model_clause = (
            "\nFirst entity vector embedding: "
            f"{vector['sentinel']}."
        )
        structured_prompt = base["native_inputs"]["text"]["prompt"]
        audited_prompt = structured_prompt + audited_clause
        model_prompt = " ".join(
            (structured_prompt + model_clause).split()
        )
        if model_prompt.count(vector["sentinel"]) != 1:
            raise AssertionError("entity-vector model prompt needs one sentinel")

        source_prompt = case["text"]["prompt"].strip()
        base.update({
            "source_prompt_sha256": canonical_sha256(source_prompt),
            "prompt_sha256": canonical_sha256(model_prompt),
            "audited_prompt_sha256": canonical_sha256(audited_prompt),
            "text_transform_id": (
                "append_seven_scene_physics_text_and_entity_vector_v1"
            ),
            "prompt": model_prompt,
            "data_adapter_fingerprint": self.fingerprint,
            "materialization_fingerprint": self.materialization_fingerprint,
        })
        base["native_inputs"]["text"] = {
            "prompt": model_prompt,
            "audited_prompt": audited_prompt,
        }
        base["native_inputs"]["physics"] = {
            "representation": REPRESENTATION,
            "entity_vector": vector,
            "quantities": [vector],
            "registry_id": REPRESENTATION,
            "registry_fingerprint": canonical_sha256({
                "representation": REPRESENTATION,
                "components": list(ENTITY_VECTOR_COMPONENTS),
                "source_object": "object_1",
            }),
        }
        base["stages"]["text"] = {
            "type": "seven_scene_physics_text_plus_entity_sentinel_v1",
            "source": "case.text.prompt+case.physics",
            "source_prompt_sha256": canonical_sha256(source_prompt),
            "audited_prompt_sha256": canonical_sha256(audited_prompt),
            "model_prompt_sha256": canonical_sha256(model_prompt),
            "literal_to_sentinel": True,
        }
        base["stages"]["physics"]["entity_vector"] = {
            "representation": REPRESENTATION,
            "source_object": "object_1",
            "components": list(ENTITY_VECTOR_COMPONENTS),
            "span_stage": "post_frozen_text_encoder_pre_dit",
        }
        base["input_contract"]["physics_channels"].append({
            "id": "first_entity_vector",
            "representation": REPRESENTATION,
            "binding": "native_inputs.physics.entity_vector",
            "transport": "inline_json",
            "used_parameters": [],
        })
        return base


def create_adapter(bundle):
    return FirstEntityVectorDataAdapter(bundle)
