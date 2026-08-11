from __future__ import annotations

import copy
import unittest

from physbench.evaluation.common.entities.contracts import (
    LifecyclePolicy,
    ReferenceCapability,
)
from physbench.evaluation.common.entities.manifest import (
    EntityManifest,
    materialize_entity_manifest,
    parse_entity_manifest,
)


def _quantity(
    value: float,
    unit: str,
    *,
    annotated: bool = True,
    symbol: str | None = None,
) -> dict[str, object]:
    quantity: dict[str, object] = {
        "value": value,
        "unit": unit,
        "annotated": annotated,
    }
    if symbol is not None:
        quantity["symbol"] = symbol
    return quantity


def _collision_case(count: int) -> dict[str, object]:
    physics: dict[str, object] = {}
    labels: list[str] = []
    materials: list[str] = []
    for index in range(1, count + 1):
        label = "steel_ball" if index < count else "glass_ball"
        labels.append(label)
        materials.append("steel" if index < count else "glass")
        physics[f"ball_{index}_mass"] = _quantity(
            0.01 * index,
            "kg",
        )
        physics[f"ball_{index}_radius"] = _quantity(
            0.005 * index,
            "m",
        )
        physics[f"ball_{index}_initial_velocity"] = _quantity(
            0.2 if index in {1, count} else 0.0,
            "m/s",
        )
    physics["striker_initial_velocity"] = copy.deepcopy(
        physics["ball_1_initial_velocity"]
    )
    return {
        "case_id": f"collision_{count}_body",
        "scene_id": "collision_1d",
        "appearance": {
            "ball_sequence": labels,
            "ball_materials": materials,
        },
        "physics": physics,
        "has_real_reference_video": True,
        "provenance": {
            "parent_case_id": None,
        },
    }


def _spring_case() -> dict[str, object]:
    return {
        "case_id": "vertical_spring_case",
        "scene_id": "vertical_spring_oscillator",
        "appearance": {
            "spring_id": "spring_a",
            "release_side": "below_equilibrium",
        },
        "physics": {
            "objects": {
                "object_1": {
                    "initial_displacement": {
                        "value": 0.03,
                        "unit": "m",
                        "symbol": "x_0",
                    },
                    "mass": {
                        "value": 0.1,
                        "unit": "kg",
                        "symbol": "m",
                    },
                    "radius": {
                        "value": 0.01,
                        "unit": "m",
                        "symbol": "r",
                    },
                },
            },
            "environment": {
                "gravity_acceleration": {
                    "value": 9.81,
                    "unit": "m/s^2",
                    "symbol": "g",
                },
                "natural_spring_length": {
                    "value": 0.2,
                    "unit": "m",
                    "symbol": "L_0",
                },
                "spring_stiffness": {
                    "value": 12.0,
                    "unit": "N/m",
                    "symbol": "k",
                },
            },
        },
        "assets": {"reference_video": "spring.mp4"},
    }


class EntityManifestTests(unittest.TestCase):
    def test_vertical_spring_materializes_frozen_ball_and_apparatus(
        self,
    ) -> None:
        manifest = materialize_entity_manifest(_spring_case())

        self.assertEqual("oscillator_ball", manifest.entities[0].entity_id)
        self.assertEqual("spring_oscillator", manifest.entities[0].role_id)
        self.assertEqual("steel_ball", manifest.entities[0].entity_class)
        self.assertEqual(("spring",), manifest.entities[0].parts)
        self.assertEqual(
            {"initial_displacement", "mass", "radius"},
            {
                value.name
                for value in manifest.entities[0].physical_attributes
            },
        )
        self.assertEqual(
            "vertical_spring_and_support",
            manifest.apparatus[0].apparatus_class,
        )
        self.assertIs(
            manifest.reference_capability,
            ReferenceCapability.SAME_CASE_GT,
        )

    def test_vertical_spring_rejects_missing_required_physics(
        self,
    ) -> None:
        missing_parameters = {
            "mass": ("objects", "object_1", "mass", "oscillator_mass"),
            "radius": ("objects", "object_1", "radius", "ball_radius"),
            "displacement": (
                "objects",
                "object_1",
                "initial_displacement",
                "initial_displacement",
            ),
            "stiffness": (
                "environment",
                "spring_stiffness",
                "spring_stiffness",
            ),
            "natural length": (
                "environment",
                "natural_spring_length",
                "natural_spring_length",
            ),
            "gravity": (
                "environment",
                "gravity_acceleration",
                "gravity_acceleration",
            ),
        }
        for name, path in missing_parameters.items():
            with self.subTest(parameter=name):
                case = _spring_case()
                target: object = case["physics"]
                for key in path[:-2]:
                    target = target[key]  # type: ignore[index]
                del target[path[-2]]  # type: ignore[index]
                with self.assertRaises(ValueError):
                    materialize_entity_manifest(case)

    def test_collision_cardinality_is_derived_for_two_and_four_bodies(
        self,
    ) -> None:
        for count in (2, 4):
            with self.subTest(count=count):
                manifest = materialize_entity_manifest(
                    _collision_case(count)
                )
                self.assertEqual(
                    tuple(
                        f"ball_{index}"
                        for index in range(1, count + 1)
                    ),
                    tuple(
                        entity.entity_id for entity in manifest.entities
                    ),
                )
                self.assertEqual(
                    tuple(
                        f"body_{index}"
                        for index in range(1, count + 1)
                    ),
                    tuple(entity.role_id for entity in manifest.entities),
                )
                self.assertEqual(
                    count,
                    len(manifest.entity_specs),
                )
                active = {
                    entity.entity_id
                    for entity in manifest.entities
                    for attribute in entity.physical_attributes
                    if (
                        attribute.name == "initial_velocity"
                        and attribute.value != 0.0
                    )
                }
                self.assertEqual(
                    {"ball_1", f"ball_{count}"},
                    active,
                )
                self.assertEqual(
                    "legacy_collision_entities_v1",
                    manifest.materializer_id,
                )

    def test_collision_rejects_sequence_physics_cardinality_mismatch(
        self,
    ) -> None:
        missing = _collision_case(3)
        del missing["physics"]["ball_2_radius"]  # type: ignore[index]
        with self.assertRaisesRegex(
            ValueError,
            "requires exactly mass, radius, and initial_velocity",
        ):
            materialize_entity_manifest(missing)

        extra = _collision_case(3)
        extra["physics"]["ball_4_mass"] = _quantity(0.02, "kg")  # type: ignore[index]
        with self.assertRaisesRegex(
            ValueError,
            "indexed physics disagree",
        ):
            materialize_entity_manifest(extra)

        malformed_index = _collision_case(2)
        malformed_index["physics"]["ball_01_mass"] = _quantity(  # type: ignore[index]
            0.01,
            "kg",
        )
        with self.assertRaisesRegex(
            ValueError,
            "non-canonical ball index",
        ):
            materialize_entity_manifest(malformed_index)

    def test_collision_rejects_uncertain_or_inconsistent_physics(
        self,
    ) -> None:
        unannotated = _collision_case(2)
        unannotated["physics"]["ball_2_mass"]["annotated"] = False  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "must be annotated"):
            materialize_entity_manifest(unannotated)

        bad_unit = _collision_case(2)
        bad_unit["physics"]["ball_2_radius"]["unit"] = "cm"  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "unit must be 'm'"):
            materialize_entity_manifest(bad_unit)

        inconsistent_alias = _collision_case(2)
        inconsistent_alias["physics"]["striker_initial_velocity"][  # type: ignore[index]
            "value"
        ] = 9.0
        with self.assertRaisesRegex(
            ValueError,
            "striker_initial_velocity disagrees",
        ):
            materialize_entity_manifest(inconsistent_alias)

    def test_collision_striker_alias_can_identify_right_hand_ball(
        self,
    ) -> None:
        case = _collision_case(2)
        case["appearance"]["striker_ball_index"] = 2  # type: ignore[index]
        case["physics"]["striker_initial_velocity"] = copy.deepcopy(  # type: ignore[index]
            case["physics"]["ball_2_initial_velocity"]  # type: ignore[index]
        )
        manifest = materialize_entity_manifest(case)
        self.assertEqual(2, len(manifest.entities))

        case["appearance"]["striker_ball_index"] = 3  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "striker_ball_index"):
            materialize_entity_manifest(case)

    def test_schema_5_collision_alias_uses_magnitude_and_preserves_symbols(
        self,
    ) -> None:
        case = _collision_case(2)
        case["schema_version"] = "5.0"
        case["appearance"]["striker_ball_index"] = 2  # type: ignore[index]
        case["physics"]["ball_1_initial_velocity"] = _quantity(  # type: ignore[index]
            0.0, "m/s", symbol="v_1"
        )
        case["physics"]["ball_2_initial_velocity"] = _quantity(  # type: ignore[index]
            -0.2, "m/s", symbol="v_2"
        )
        case["physics"]["ball_1_mass"]["symbol"] = "m_1"  # type: ignore[index]
        case["physics"]["ball_1_radius"]["symbol"] = "r_1"  # type: ignore[index]
        case["physics"]["ball_2_mass"]["symbol"] = "m_2"  # type: ignore[index]
        case["physics"]["ball_2_radius"]["symbol"] = "r_2"  # type: ignore[index]
        case["physics"]["striker_initial_velocity"] = _quantity(  # type: ignore[index]
            0.2, "m/s", annotated=False, symbol="v_s"
        )

        manifest = materialize_entity_manifest(case)
        velocity = next(
            attribute
            for attribute in manifest.entities[1].physical_attributes
            if attribute.name == "initial_velocity"
        )
        self.assertEqual("v_2", velocity.symbol)
        self.assertEqual(
            "v_2",
            manifest.to_canonical_dict()["entities"][1][
                "physical_attributes"
            ]["initial_velocity"]["symbol"],
        )

        historical = copy.deepcopy(case)
        historical["schema_version"] = "4.0"
        for quantity in historical["physics"].values():  # type: ignore[union-attr]
            quantity.pop("symbol")
        historical["physics"]["striker_initial_velocity"][  # type: ignore[index]
            "annotated"
        ] = True
        with self.assertRaisesRegex(ValueError, "striker_initial_velocity disagrees"):
            materialize_entity_manifest(historical)

    def test_parabolic_motion_has_projectile_and_launch_frame(self) -> None:
        case = {
            "case_id": "parabolic_case",
            "scene_id": "parabolic_motion",
            "appearance": {
                "ball_material": "steel",
                "ball_size_class": "medium",
            },
            "physics": {
                "ball_mass": _quantity(0.014, "kg"),
                "ball_radius": _quantity(0.00685, "m"),
                "launch_height": _quantity(0.77, "m"),
                "initial_horizontal_velocity": _quantity(0.82, "m/s"),
                "photogate_distance_before_launch": _quantity(0.012, "m"),
            },
            "has_real_reference_video": True,
            "provenance": {"parent_case_id": None},
        }
        manifest = materialize_entity_manifest(case)
        self.assertEqual("projectile_ball", manifest.entities[0].entity_id)
        self.assertEqual("launch_frame", manifest.apparatus[0].apparatus_id)

    def test_explicit_entities_resolve_physics_references_and_apparatus(
        self,
    ) -> None:
        case = {
            "case_id": "custom_case",
            "scene_id": "custom_scene",
            "physics": {
                "body_mass": _quantity(2.5, "kg"),
            },
            "entities": [
                {
                    "entity_id": "body",
                    "role_id": "participant",
                    "entity_class": "rigid_body",
                    "physical_attributes": {
                        "mass": "body_mass",
                    },
                    "parts": ["shell"],
                    "exchangeability_group": None,
                    "lifecycle": "may_enter_and_exit",
                    "condition_anchor": {
                        "source": "condition_frame",
                        "order": 0,
                    },
                }
            ],
            "apparatus": [
                {
                    "apparatus_id": "surface",
                    "apparatus_class": "support_plane",
                    "physical_attributes": {},
                    "condition_anchor": {
                        "source": "condition_frame",
                    },
                }
            ],
            "has_real_reference_video": False,
            "provenance": {
                "parent_case_id": "custom_parent",
            },
        }
        manifest = materialize_entity_manifest(case)
        self.assertEqual("explicit_case_entities_v1", manifest.materializer_id)
        self.assertEqual(
            ReferenceCapability.PHYSICS_PARENT,
            manifest.reference_capability,
        )
        entity = manifest.entities[0]
        self.assertEqual(LifecyclePolicy.MAY_ENTER_AND_EXIT, entity.lifecycle)
        self.assertEqual("body_mass", entity.physical_attributes[0].source_parameter)
        self.assertEqual(2.5, entity.physical_attributes[0].value)
        self.assertEqual("surface", manifest.apparatus[0].apparatus_id)

    def test_explicit_inline_attribute_must_match_referenced_case_value(
        self,
    ) -> None:
        case = {
            "case_id": "bad_explicit",
            "scene_id": "custom_scene",
            "physics": {
                "body_mass": _quantity(2.5, "kg"),
            },
            "entities": [
                {
                    "entity_id": "body",
                    "entity_class": "rigid_body",
                    "physical_attributes": {
                        "mass": {
                            "value": 9.0,
                            "unit": "kg",
                            "annotated": True,
                            "source_parameter": "body_mass",
                        }
                    },
                }
            ],
        }
        with self.assertRaisesRegex(ValueError, "disagrees"):
            materialize_entity_manifest(case)

    def test_canonical_round_trip_and_digest_are_stable(self) -> None:
        manifest = materialize_entity_manifest(_collision_case(4))
        document = manifest.to_canonical_dict()
        restored = parse_entity_manifest(document)
        self.assertIsInstance(restored, EntityManifest)
        self.assertEqual(document, restored.to_canonical_dict())
        self.assertEqual(manifest.digest, restored.digest)
        self.assertEqual(64, len(manifest.digest))

        reordered = copy.deepcopy(document)
        first_attribute = reordered["entities"][0][  # type: ignore[index]
            "physical_attributes"
        ]
        reordered["entities"][0]["physical_attributes"] = {  # type: ignore[index]
            name: first_attribute[name]
            for name in reversed(list(first_attribute))
        }
        self.assertEqual(
            manifest.digest,
            parse_entity_manifest(reordered).digest,
        )
        with self.assertRaises(TypeError):
            manifest.entities[0].condition_anchor["new"] = "value"  # type: ignore[index]

    def test_manifest_schema_rejects_duplicates_and_nonfinite_anchor(
        self,
    ) -> None:
        document = materialize_entity_manifest(
            _collision_case(2)
        ).to_canonical_dict()
        duplicate = copy.deepcopy(document)
        duplicate["entities"][1]["entity_id"] = "ball_1"  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "duplicate entity_id"):
            parse_entity_manifest(duplicate)

        nonfinite = copy.deepcopy(document)
        nonfinite["entities"][0]["condition_anchor"]["x"] = float("nan")  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "non-finite"):
            parse_entity_manifest(nonfinite)

        unknown = copy.deepcopy(document)
        unknown["entities"][0]["unsupported"] = True  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "unknown fields"):
            parse_entity_manifest(unknown)

    def test_existing_scene_defaults_distinguish_entities_and_apparatus(
        self,
    ) -> None:
        cases = [
            {
                "case_id": "freefall",
                "scene_id": "free_fall",
                "appearance": {
                    "ball_material": "steel",
                    "ball_size_class": "L",
                },
                "physics": {
                    "ball_mass": _quantity(0.02, "kg"),
                    "ball_radius": _quantity(0.01, "m"),
                    "initial_height": _quantity(0.6, "m"),
                    "initial_velocity": _quantity(0.0, "m/s"),
                },
            },
            {
                "case_id": "incline",
                "scene_id": "inclined_plane_slide",
                "appearance": {
                    "block": "wood",
                    "release_point": "top",
                },
                "physics": {
                    "block_length": _quantity(0.1, "m"),
                    "block_mass": _quantity(0.1, "kg"),
                    "initial_velocity": _quantity(
                        0.0,
                        "m/s",
                        annotated=False,
                    ),
                    "incline_angle": _quantity(30.0, "deg"),
                },
            },
            {
                "case_id": "pendulum",
                "scene_id": "pendulum",
                "appearance": {
                    "bob_material": "steel",
                },
                "physics": {
                    "bob_radius": _quantity(0.01, "m"),
                    "initial_angle": _quantity(20.0, "deg"),
                    "pendulum_length": _quantity(0.11, "m"),
                    "string_length": _quantity(0.1, "m"),
                },
            },
            {
                "case_id": "circular",
                "scene_id": "uniform_circular_motion",
                "appearance": {
                    "object_count": 2,
                    "moving_objects": ["silver", "wood"],
                },
                "physics": {
                    "angular_velocity": _quantity(55.0, "deg/s"),
                    "angular_velocity_rad_s": _quantity(
                        0.96,
                        "rad/s",
                        annotated=False,
                    ),
                    "object_1_orbit_radius": _quantity(0.02, "m"),
                    "object_2_orbit_radius": _quantity(0.06, "m"),
                },
            },
        ]
        expected = {
            "free_fall": ("falling_body", "gravity_frame", 1),
            "inclined_plane_slide": (
                "sliding_block",
                "incline_track",
                1,
            ),
            "pendulum": ("bob", "pivot_support", 1),
            "uniform_circular_motion": (
                "object_1",
                "rotation_platform",
                2,
            ),
        }
        for case in cases:
            with self.subTest(scene=case["scene_id"]):
                manifest = materialize_entity_manifest(case)
                entity_id, apparatus_id, count = expected[case["scene_id"]]
                self.assertEqual(count, len(manifest.entities))
                self.assertEqual(entity_id, manifest.entities[0].entity_id)
                self.assertEqual(
                    apparatus_id,
                    manifest.apparatus[0].apparatus_id,
                )

    def test_unknown_legacy_scene_requires_explicit_entities(self) -> None:
        with self.assertRaisesRegex(ValueError, "declare case.entities"):
            materialize_entity_manifest(
                {
                    "case_id": "unknown",
                    "scene_id": "new_scene",
                    "physics": {
                        "speed": _quantity(1.0, "m/s"),
                    },
                }
            )


if __name__ == "__main__":
    unittest.main()
