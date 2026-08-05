from __future__ import annotations

import copy
import json
import subprocess
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from physbench.io import canonical_sha256, load_json, load_jsonl
from scripts import build_dataset_v11 as builder


class SymbolicPhysicsMigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = load_jsonl(builder.BASE_RELEASE_ROOT / "cases.jsonl")
        cls.by_scene = {
            scene_id: [case for case in cls.cases if case["scene_id"] == scene_id]
            for scene_id in {case["scene_id"] for case in cls.cases}
        }

    def test_symbol_table_exactly_covers_real_parameter_universe(self) -> None:
        actual = {
            (case["scene_id"], parameter)
            for case in self.cases
            for parameter in case["physics"]
        }
        self.assertEqual(actual, set(builder.SYMBOLS))
        self.assertTrue(all(builder.SYMBOLS[key].strip() for key in actual))
        for case in self.cases:
            migrated = builder.migrate_case(case)
            symbols = [quantity["symbol"] for quantity in migrated["physics"].values()]
            self.assertEqual(len(symbols), len(set(symbols)), case["case_id"])

    def test_value_and_annotation_migration_is_exact(self) -> None:
        negative_before = Counter(
            parameter
            for case in self.cases
            for parameter, quantity in case["physics"].items()
            if quantity["value"] < 0
        )
        self.assertEqual(
            Counter(
                {
                    "ball_2_initial_velocity": 264,
                    "striker_initial_velocity": 230,
                }
            ),
            negative_before,
        )

        value_changes = 0
        flag_changes = Counter()
        for case in self.cases:
            migrated = builder.migrate_case(case)
            self.assertEqual("5.0", migrated["schema_version"])
            for name, old in case["physics"].items():
                new = migrated["physics"][name]
                self.assertEqual(old["unit"], new["unit"])
                self.assertGreaterEqual(new["value"], 0)
                if old["value"] < 0:
                    self.assertEqual(abs(old["value"]), new["value"])
                    value_changes += 1
                else:
                    self.assertEqual(old["value"], new["value"])
                if old["annotated"] != new["annotated"]:
                    self.assertTrue(old["annotated"])
                    self.assertFalse(new["annotated"])
                    flag_changes[(case["scene_id"], name)] += 1
        self.assertEqual(494, value_changes)
        self.assertEqual(
            Counter(
                {
                    ("collision_1d", "striker_initial_velocity"): 330,
                    ("inclined_plane_slide", "calibration_length"): 95,
                    ("inclined_plane_slide", "friction_force"): 95,
                    ("inclined_plane_slide", "theoretical_acceleration"): 95,
                    ("pendulum", "pendulum_length"): 100,
                }
            ),
            flag_changes,
        )

    def test_exact_prompts_cover_every_scene_branch(self) -> None:
        fixtures = {
            "two_ball_single_incident": next(
                case
                for case in self.by_scene["collision_1d"]
                if case["appearance"]["collision_structure"]
                == "two_ball_single_incident"
            ),
            "two_ball_opposed_incident": next(
                case
                for case in self.by_scene["collision_1d"]
                if case["appearance"]["collision_structure"]
                == "two_ball_opposed_incident"
            ),
            "three_ball_single_incident": next(
                case
                for case in self.by_scene["collision_1d"]
                if case["appearance"]["collision_structure"]
                == "three_ball_single_incident"
            ),
            "pendulum_without_mass": next(
                case for case in self.by_scene["pendulum"] if "bob_mass" not in case["physics"]
            ),
            "pendulum_with_mass": next(
                case for case in self.by_scene["pendulum"] if "bob_mass" in case["physics"]
            ),
            "circular_one": next(
                case
                for case in self.by_scene["uniform_circular_motion"]
                if "object_2_orbit_radius" not in case["physics"]
            ),
            "circular_two": next(
                case
                for case in self.by_scene["uniform_circular_motion"]
                if "object_2_orbit_radius" in case["physics"]
            ),
            "incline": self.by_scene["inclined_plane_slide"][0],
            "projectile": self.by_scene["parabolic_motion"][0],
            "push_bottle": self.by_scene["push_bottle"][0],
        }
        expected = {
            "two_ball_single_incident": (
                "The left ball has mass m_1, radius r_1, and initial speed v_1, "
                "while the right ball has mass m_2, radius r_2, and initial speed "
                "v_2. The left ball is initially stationary, the right ball moves "
                "left, and they undergo a one-dimensional central collision."
            ),
            "two_ball_opposed_incident": (
                "The left ball has mass m_1, radius r_1, and initial speed v_1, "
                "while the right ball has mass m_2, radius r_2, and initial speed "
                "v_2. The left ball moves right, the right ball moves left, and they "
                "undergo a one-dimensional central collision."
            ),
            "three_ball_single_incident": (
                "The left ball has mass m_1, radius r_1, and initial speed v_1, the "
                "middle ball has mass m_2, radius r_2, and initial speed v_2, and the "
                "right ball has mass m_3, radius r_3, and initial speed v_3. The left "
                "ball moves right while the middle and right balls are initially "
                "stationary, and the left ball first undergoes a one-dimensional "
                "central collision with the middle ball."
            ),
            "incline": (
                "A block of mass m and length l starts from rest and slides down an "
                "incline of angle θ. The kinetic friction coefficient between the "
                "block and incline is μ_k, and the gravitational acceleration is g."
            ),
            "projectile": (
                "A ball of mass m and radius r is launched horizontally to the left "
                "with initial speed v_0 from a vertical height h, then follows a "
                "downward parabolic path."
            ),
            "pendulum_without_mass": (
                "A pendulum bob of radius r is released from rest at initial angle "
                "θ_0 on a string of length l_s, then swings back and forth about the "
                "fixed pivot."
            ),
            "pendulum_with_mass": (
                "A pendulum bob of mass m and radius r is released from rest at "
                "initial angle θ_0 on a string of length l_s, then swings back and "
                "forth about the fixed pivot."
            ),
            "push_bottle": (
                "An upright bottle of mass m and height h is pushed near its top by a "
                "force with mean magnitude F_mean and peak magnitude F_peak, then "
                "tips and falls onto its side."
            ),
            "circular_one": (
                "An object follows a circular orbit of radius r_1 with angular speed ω."
            ),
            "circular_two": (
                "Two objects follow circular orbits of radii r_1 and r_2 with angular "
                "speed ω."
            ),
        }
        for branch, case in fixtures.items():
            with self.subTest(branch=branch):
                text = builder.migrate_case(case)["text"]
                self.assertEqual(expected[branch], text["prompt"])
                self.assertEqual("en", text["language"])
                self.assertEqual(
                    "symbolic_physics_prompt_v1", text["annotation_source"]
                )

    def test_every_prompt_has_only_independent_symbols_and_no_hints(self) -> None:
        for case in self.cases:
            migrated = builder.migrate_case(case)
            builder.validate_prompt(migrated)
            prompt = migrated["text"]["prompt"]
            for quantity in migrated["physics"].values():
                if quantity["annotated"]:
                    self.assertTrue(
                        builder.symbol_in_prompt(quantity["symbol"], prompt),
                        (case["case_id"], quantity["symbol"], prompt),
                    )
                else:
                    self.assertFalse(
                        builder.symbol_in_prompt(quantity["symbol"], prompt),
                        (case["case_id"], quantity["symbol"], prompt),
                    )

    def test_collision_direction_contradictions_abort(self) -> None:
        single = next(
            case
            for case in self.by_scene["collision_1d"]
            if case["appearance"]["collision_structure"]
            == "two_ball_single_incident"
        )
        opposed = next(
            case
            for case in self.by_scene["collision_1d"]
            if case["appearance"]["collision_structure"]
            == "two_ball_opposed_incident"
        )
        bad_single = copy.deepcopy(single)
        bad_single["physics"]["ball_2_initial_velocity"]["value"] *= -1
        bad_single["physics"]["striker_initial_velocity"]["value"] *= -1
        bad_opposed = copy.deepcopy(opposed)
        bad_opposed["physics"]["ball_2_initial_velocity"]["value"] *= -1
        for case in (bad_single, bad_opposed):
            with self.subTest(case=case["case_id"]), self.assertRaisesRegex(
                ValueError, "signed velocity contradicts audited direction"
            ):
                builder.migrate_case(case)

    def test_circular_prompt_does_not_invent_rotation_direction(self) -> None:
        for case in self.by_scene["uniform_circular_motion"]:
            prompt = builder.migrate_case(case)["text"]["prompt"].lower()
            self.assertNotIn("clockwise", prompt)
            self.assertNotIn("counterclockwise", prompt)

    def test_physics_document_bytes_and_scene_classification_are_deterministic(self) -> None:
        migrated = [builder.migrate_case(case) for case in self.cases]
        with tempfile.TemporaryDirectory() as temporary:
            writes, output_cases = builder.prepare_v11(
                self.cases,
                [
                    json.loads(path.read_text(encoding="utf-8"))
                    for path in sorted((builder.BASE_RELEASE_ROOT / "scenes").glob("*.json"))
                ],
                Path(temporary),
            )
        self.assertEqual(799, len(writes))
        self.assertEqual(799, len(output_cases))
        first = writes[0]
        self.assertTrue(first.payload.endswith(b"\n"))
        self.assertFalse(first.payload.endswith(b"\n\n"))
        document = json.loads(first.payload)
        self.assertEqual("2.0", document["schema_version"])
        self.assertEqual(first.case_id, document["case_id"])

        by_scene = {}
        for case in migrated:
            by_scene.setdefault(case["scene_id"], []).append(case)
        for scene_path in sorted((builder.BASE_RELEASE_ROOT / "scenes").glob("*.json")):
            scene = json.loads(scene_path.read_text(encoding="utf-8"))
            migrated_scene = builder.migrate_scene(scene, by_scene[scene["scene_id"]])
            true_names = sorted(
                {
                    name
                    for case in by_scene[scene["scene_id"]]
                    for name, quantity in case["physics"].items()
                    if quantity["annotated"]
                }
            )
            false_names = sorted(
                {
                    name
                    for case in by_scene[scene["scene_id"]]
                    for name, quantity in case["physics"].items()
                    if not quantity["annotated"]
                }
            )
            self.assertEqual(true_names, migrated_scene["structured_physics_parameters"])
            self.assertEqual(
                false_names, migrated_scene["non_conditionable_physics_parameters"]
            )

    def test_conflicting_document_aborts_before_any_missing_write(self) -> None:
        cases = self.cases[:2]
        scenes = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted((builder.BASE_RELEASE_ROOT / "scenes").glob("*.json"))
        ]
        with tempfile.TemporaryDirectory() as temporary:
            writes, _ = builder.prepare_v11(cases, [
                scene for scene in scenes if scene["scene_id"] in {
                    case["scene_id"] for case in cases
                }
            ], Path(temporary))
            writes[1].absolute_path.parent.mkdir(parents=True)
            writes[1].absolute_path.write_bytes(b"conflict\n")
            with self.assertRaisesRegex(ValueError, "conflicting physics document"):
                builder.materialize_physics_documents(writes)
            self.assertFalse(writes[0].absolute_path.exists())
            self.assertEqual(b"conflict\n", writes[1].absolute_path.read_bytes())

    def test_git_ignore_admits_v10_and_v11_physics_but_not_media(self) -> None:
        paths = {
            "datasets/assets/pendulum/example/physics.json": False,
            "datasets/assets/pendulum/example/physics.v11.json": False,
            "datasets/assets/pendulum/example/canonical/reference.mp4": True,
            "datasets/assets/pendulum/example/canonical/masks/01.npz": True,
        }
        for path, ignored in paths.items():
            with self.subTest(path=path):
                result = subprocess.run(
                    ["git", "check-ignore", "--no-index", "-q", path],
                    cwd=builder.ROOT,
                    check=False,
                )
                self.assertEqual(0 if ignored else 1, result.returncode)


class DatasetV11ReleaseTests(unittest.TestCase):
    def test_release_is_minimal_and_preserves_non_symbolic_case_facts(self) -> None:
        self.assertEqual(
            {
                "README.md",
                "dataset.json",
                "release.json",
                "cases.jsonl",
                "assets.lock.json",
                "scenes",
                "views",
            },
            {path.name for path in builder.OUTPUT_RELEASE_ROOT.iterdir()},
        )
        old_cases = load_jsonl(builder.BASE_RELEASE_ROOT / "cases.jsonl")
        new_cases = load_jsonl(builder.OUTPUT_RELEASE_ROOT / "cases.jsonl")
        self.assertEqual(799, len(new_cases))
        physics_paths = {
            case["assets"]["physics_annotation"] for case in new_cases
        }
        self.assertEqual(799, len(physics_paths))
        self.assertTrue(
            all(path.endswith("/physics.v11.json") for path in physics_paths)
        )
        for old, new in zip(old_cases, new_cases, strict=True):
            self.assertEqual(old["case_id"], new["case_id"])
            comparable_old = copy.deepcopy(old)
            comparable_new = copy.deepcopy(new)
            for value in (comparable_old, comparable_new):
                value.pop("schema_version")
                value.pop("text")
                value.pop("physics")
                value["assets"].pop("physics_annotation")
            self.assertEqual(comparable_old, comparable_new)

    def test_release_lock_replaces_only_v10_physics_documents(self) -> None:
        old = load_json(builder.BASE_RELEASE_ROOT / "assets.lock.json")
        new = load_json(builder.OUTPUT_RELEASE_ROOT / "assets.lock.json")
        old_nonphysics = {
            item["path"]: item
            for item in old["files"]
            if not item["path"].endswith("/physics.json")
        }
        new_nonphysics = {
            item["path"]: item
            for item in new["files"]
            if not item["path"].endswith("/physics.v11.json")
        }
        self.assertEqual(5239, len(old_nonphysics))
        self.assertEqual(old_nonphysics, new_nonphysics)
        self.assertEqual(6038, len(new["files"]))

    def test_views_are_byte_identical_and_scenes_change_only_classification(self) -> None:
        for old_path in sorted((builder.BASE_RELEASE_ROOT / "views").glob("*.json")):
            self.assertEqual(
                old_path.read_bytes(),
                (builder.OUTPUT_RELEASE_ROOT / "views" / old_path.name).read_bytes(),
            )
        for old_path in sorted((builder.BASE_RELEASE_ROOT / "scenes").glob("*.json")):
            old = load_json(old_path)
            new = load_json(builder.OUTPUT_RELEASE_ROOT / "scenes" / old_path.name)
            for value in (old, new):
                value.pop("structured_physics_parameters")
                value.pop("non_conditionable_physics_parameters")
            self.assertEqual(old, new)

    def test_migration_evidence_has_exact_counts_and_digests(self) -> None:
        evidence = load_json(builder.PROVENANCE_ROOT / "migration.json")
        self.assertEqual(
            {
                "cases": 799,
                "physics_documents": 799,
                "locked_assets": 6038,
                "value_changes": 494,
                "annotation_flag_changes": 715,
                "prompt_changes": 799,
                "media_changes": 0,
            },
            evidence["counts"],
        )
        expected_symbol_digest = canonical_sha256(
            [
                {"scene_id": scene, "parameter": name, "symbol": symbol}
                for (scene, name), symbol in sorted(builder.SYMBOLS.items())
            ]
        )
        self.assertEqual(expected_symbol_digest, evidence["symbol_table_sha256"])
        self.assertEqual(
            {
                path.name: canonical_sha256(load_json(path))
                for path in sorted((builder.BASE_RELEASE_ROOT / "views").glob("*.json"))
            },
            evidence["view_digests"],
        )
        self.assertEqual(
            load_json(builder.BASE_RELEASE_ROOT / "release.json")["dataset_digest"],
            evidence["base"]["dataset_digest"],
        )
        self.assertEqual(
            load_json(builder.OUTPUT_RELEASE_ROOT / "release.json")["dataset_digest"],
            evidence["output"]["dataset_digest"],
        )


if __name__ == "__main__":
    unittest.main()
