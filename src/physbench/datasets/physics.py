from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any


GROUPED_PHYSICS_SCENES = frozenset({
    "collision_1d",
    "inclined_plane_slide",
    "parabolic_motion",
    "pendulum",
    "push_bottle",
    "uniform_circular_motion",
    "vertical_spring_oscillator",
})

_OBJECT_PARAMETERS = {
    "collision_1d": {"mass", "radius", "initial_velocity"},
    "inclined_plane_slide": {"mass", "length"},
    "parabolic_motion": {
        "mass",
        "radius",
        "initial_horizontal_velocity",
        "launch_height",
    },
    "pendulum": {"mass", "radius", "initial_angle"},
    "push_bottle": {"mass", "height", "applied_force"},
    "uniform_circular_motion": {"orbit_radius"},
    "vertical_spring_oscillator": {"initial_displacement", "mass", "radius"},
}
_ENVIRONMENT_PARAMETERS = {
    "collision_1d": set(),
    "inclined_plane_slide": {"incline_angle", "gravity_acceleration"},
    "parabolic_motion": set(),
    "pendulum": {"string_length"},
    "push_bottle": set(),
    "uniform_circular_motion": {"angular_velocity"},
    "vertical_spring_oscillator": {
        "gravity_acceleration",
        "natural_spring_length",
        "spring_stiffness",
    },
}


def is_scalar_quantity(value: Any) -> bool:
    return isinstance(value, Mapping) and set(value) == {"value", "unit", "symbol"}


def is_time_series_quantity(value: Any) -> bool:
    return isinstance(value, Mapping) and set(value) == {
        "samples",
        "time_unit",
        "unit",
        "symbol",
    }


def is_projected_scalar_quantity(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and {"value", "unit"} <= set(value)
        and set(value) <= {"value", "unit", "symbol"}
    )


def _semantic_object_name(scene_id: str, object_index: int, name: str) -> str:
    if scene_id == "collision_1d":
        return f"ball_{object_index}_{name}"
    if scene_id == "inclined_plane_slide":
        return {"mass": "block_mass", "length": "block_length"}[name]
    if scene_id == "parabolic_motion":
        return {
            "mass": "ball_mass",
            "radius": "ball_radius",
            "initial_horizontal_velocity": "initial_horizontal_velocity",
            "launch_height": "launch_height",
        }[name]
    if scene_id == "pendulum":
        return {
            "mass": "bob_mass",
            "radius": "bob_radius",
            "initial_angle": "initial_angle",
        }[name]
    if scene_id == "push_bottle":
        return {
            "mass": "bottle_mass",
            "height": "bottle_height",
            "applied_force": "applied_force",
        }[name]
    if scene_id == "uniform_circular_motion":
        return f"object_{object_index}_orbit_radius"
    if scene_id == "vertical_spring_oscillator":
        return {
            "initial_displacement": "initial_displacement",
            "mass": "oscillator_mass",
            "radius": "ball_radius",
        }[name]
    raise ValueError(f"scene {scene_id} does not use grouped physics")


def iter_physics_quantities(
    case: Mapping[str, Any],
) -> Iterator[tuple[str, str, Mapping[str, Any]]]:
    """Yield canonical path, stable semantic name, and quantity."""

    scene_id = case.get("scene_id")
    physics = case.get("physics")
    if not isinstance(physics, Mapping) or not physics:
        raise ValueError(f"case {case.get('case_id')} requires structured physics")

    if scene_id not in GROUPED_PHYSICS_SCENES:
        raise ValueError(f"scene {scene_id} has no current physics contract")
    if set(physics) != {"objects", "environment"}:
        raise ValueError(
            f"case {case.get('case_id')} grouped physics must contain exactly "
            "objects and environment"
        )
    objects = physics["objects"]
    environment = physics["environment"]
    if not isinstance(objects, Mapping) or not objects:
        raise ValueError(f"case {case.get('case_id')} physics.objects must be non-empty")
    if not isinstance(environment, Mapping):
        raise ValueError(
            f"case {case.get('case_id')} physics.environment must be an object"
        )
    expected_ids = [f"object_{index}" for index in range(1, len(objects) + 1)]
    if sorted(objects) != expected_ids:
        raise ValueError(
            f"case {case.get('case_id')} physics object IDs must be contiguous"
        )

    expected_object_parameters = _OBJECT_PARAMETERS[str(scene_id)]
    for object_index, object_id in enumerate(expected_ids, start=1):
        quantities = objects[object_id]
        if not isinstance(quantities, Mapping):
            raise ValueError(
                f"case {case.get('case_id')} physics.objects.{object_id} "
                "must be an object"
            )
        names = set(quantities)
        if scene_id == "pendulum":
            allowed = (
                {"radius", "initial_angle"},
                {"mass", "radius", "initial_angle"},
            )
            if names not in allowed:
                raise ValueError(
                    f"case {case.get('case_id')} has invalid pendulum object physics"
                )
        elif names != expected_object_parameters:
            raise ValueError(
                f"case {case.get('case_id')} has invalid {scene_id} object physics"
            )
        for name, quantity in quantities.items():
            if not isinstance(quantity, Mapping):
                raise ValueError(
                    f"case {case.get('case_id')} physics.objects.{object_id}."
                    f"{name} must be an object"
                )
            yield (
                f"objects.{object_id}.{name}",
                _semantic_object_name(str(scene_id), object_index, name),
                quantity,
            )

    expected_environment = _ENVIRONMENT_PARAMETERS[str(scene_id)]
    if set(environment) != expected_environment:
        raise ValueError(
            f"case {case.get('case_id')} has invalid {scene_id} environment physics"
        )
    for name, quantity in environment.items():
        if not isinstance(quantity, Mapping):
            raise ValueError(
                f"case {case.get('case_id')} physics.environment.{name} "
                "must be an object"
            )
        yield f"environment.{name}", name, quantity


def flat_physics_quantities(
    case: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Project current physics to stable semantic names without storing a copy.

    Managed runtime Cases already contain this projection. Accept that exact
    three-field form so compiled Tasks can be validated without reconstructing
    the original grouped document.
    """

    physics = case.get("physics")
    if (
        case.get("scene_id") in GROUPED_PHYSICS_SCENES
        and isinstance(physics, Mapping)
        and set(physics) != {"objects", "environment"}
        and physics
        and all(
            is_projected_scalar_quantity(quantity)
            or is_time_series_quantity(quantity)
            for quantity in physics.values()
        )
    ):
        return {str(name): dict(quantity) for name, quantity in physics.items()}
    if (
        case.get("scene_id") in GROUPED_PHYSICS_SCENES
        and isinstance(physics, Mapping)
        and physics
        and all(
            isinstance(quantity, Mapping)
            and isinstance(quantity.get("annotated"), bool)
            for quantity in physics.values()
        )
    ):
        # Compatibility for pre-V5 in-memory fixtures. Current Dataset files
        # never enter this branch because the Loader enforces grouped physics.
        return {
            str(name): {
                key: value
                for key, value in quantity.items()
                if key != "annotated"
            }
            for name, quantity in physics.items()
            if quantity["annotated"] is True
        }

    result: dict[str, dict[str, Any]] = {}
    for _, semantic_name, quantity in iter_physics_quantities(case):
        if semantic_name in result:
            raise ValueError(
                f"case {case.get('case_id')} repeats physics name {semantic_name}"
            )
        result[semantic_name] = dict(quantity)
    return result


__all__ = [
    "GROUPED_PHYSICS_SCENES",
    "flat_physics_quantities",
    "is_projected_scalar_quantity",
    "is_scalar_quantity",
    "is_time_series_quantity",
    "iter_physics_quantities",
]
