from __future__ import annotations

from collections import Counter
import json
import math
import os
from pathlib import Path
import re
import unittest

from physbench.data_layout import LATEST_DATASET, V5_DATASET, V51_DATASET
from physbench.datasets import load_dataset


ROOT = Path(__file__).resolve().parents[1]
RELEASE_ROOT = ROOT / "datasets/physics_video/releases/5.1.0"
DATA_ROOT = ROOT / "datasets/physics_video"
BACKGROUND_TERMS = re.compile(
    r"(?:^|[-_])(?:bg|background|black|white|oil|foam|lab|tabletop)(?:[-_]|$)",
    re.IGNORECASE,
)
SPECS = {
    "steel_ball_small_d15mm_m14g": {
        "mass": 0.014,
        "radius": 0.0075,
        "prompt": "small steel ball (15 mm diameter, 14.00 g)",
    },
    "steel_ball_medium_d20mm_m33p13g": {
        "mass": 0.03313,
        "radius": 0.010,
        "prompt": "medium steel ball (20 mm diameter, 33.13 g)",
    },
    "steel_ball_large_d25mm_m64p77g": {
        "mass": 0.06477,
        "radius": 0.0125,
        "prompt": "large steel ball (25 mm diameter, 64.77 g)",
    },
    "glass_ball_d13p5mm_m3p46g": {
        "mass": 0.00346,
        "radius": 0.00675,
        "prompt": "glass ball (13.5 mm diameter, 3.46 g)",
    },
}


class SixSceneDatasetV51Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.v5 = load_dataset(V5_DATASET)
        cls.v51 = load_dataset(
            V51_DATASET,
            check_assets=True,
        )
        cls.by_id = {
            case["case_id"]: case
            for case in cls.v51.cases
        }

    def test_v51_is_latest_and_preserves_case_identity(self) -> None:
        self.assertEqual(V51_DATASET, LATEST_DATASET)
        self.assertEqual(
            "physics_video_six_scene_v5p1",
            self.v51.descriptor["dataset_id"],
        )
        self.assertEqual("5.1.0", self.v51.descriptor["release"])
        self.assertEqual(609, len(self.v51.cases))
        self.assertEqual(
            {case["case_id"] for case in self.v5.cases},
            set(self.by_id),
        )
        self.assertEqual(
            Counter(case["scene_id"] for case in self.v5.cases),
            Counter(case["scene_id"] for case in self.v51.cases),
        )

    def test_collision_ball_specs_are_canonical_and_prompts_are_case_local(
        self,
    ) -> None:
        collisions = [
            case
            for case in self.v51.cases
            if case["scene_id"] == "collision_1d"
        ]
        seen = set()
        for case in collisions:
            sequence = case["appearance"]["ball_sequence"]
            seen.update(sequence)
            prompt = case["text"]["prompt"]
            self.assertEqual(
                "collision_case_prompt_v2",
                case["text"]["annotation_source"],
            )
            self.assertIn("From left to right", prompt)
            count_word = {2: "two", 3: "three"}[len(sequence)]
            self.assertIn(f"all {count_word} balls are fully visible", prompt)
            for index, spec_id in enumerate(sequence, 1):
                spec = SPECS[spec_id]
                self.assertEqual(
                    spec["mass"],
                    case["physics"][f"ball_{index}_mass"]["value"],
                )
                self.assertEqual(
                    spec["radius"],
                    case["physics"][f"ball_{index}_radius"]["value"],
                )
                self.assertIn(spec["prompt"], prompt)
                velocity = case["physics"][
                    f"ball_{index}_initial_velocity"
                ]["value"]
                if velocity > 0:
                    self.assertIn(
                        f"moves right at {abs(velocity):.4f} m/s",
                        prompt,
                    )
                elif velocity < 0:
                    self.assertIn(
                        f"moves left at {abs(velocity):.4f} m/s",
                        prompt,
                    )
                else:
                    self.assertIn(
                        f"ball {index} is a {spec['prompt']} and is "
                        "initially stationary",
                        prompt,
                    )
        self.assertEqual(set(SPECS), seen)

    def test_parabolic_specs_velocities_and_conditionable_fields(self) -> None:
        for case in self.v51.cases:
            if case["scene_id"] != "parabolic_motion":
                continue
            spec_id = case["appearance"]["ball_spec_id"]
            spec = SPECS[spec_id]
            physics = case["physics"]
            self.assertEqual(spec["mass"], physics["ball_mass"]["value"])
            self.assertEqual(spec["radius"], physics["ball_radius"]["value"])
            expected = (
                spec["radius"] * 2
                / physics["photogate_block_time"]["value"]
            )
            self.assertTrue(math.isclose(
                expected,
                physics["initial_horizontal_velocity"]["value"],
                rel_tol=0,
                abs_tol=1e-12,
            ))
            for field in (
                "photogate_block_time",
                "photogate_distance_before_launch",
                "ramp_angle",
                "release_distance",
            ):
                if field in physics:
                    self.assertFalse(physics[field]["annotated"])

    def test_parabolic_view_a_has_no_physical_signature_leakage(self) -> None:
        groups = self.v51.views["view_a"]["scenes"]["parabolic_motion"]

        def signature(case_id: str) -> tuple[str, float, float]:
            case = self.by_id[case_id]
            return (
                case["appearance"]["ball_spec_id"],
                round(case["physics"]["launch_height"]["value"], 9),
                round(
                    case["physics"][
                        "initial_horizontal_velocity"
                    ]["value"],
                    12,
                ),
            )

        train = {signature(case_id) for case_id in groups["train"]}
        test = {signature(case_id) for case_id in groups["test_id"]}
        self.assertFalse(train & test)
        self.assertEqual(70, len(groups["train"]))
        self.assertEqual(27, len(groups["test_id"]))

    def test_every_case_uses_one_descriptive_non_background_directory(
        self,
    ) -> None:
        for case in self.v51.cases:
            paths = [
                value
                for role, value in case["assets"].items()
                if value is not None and role != "source_archive"
            ]
            directories = {
                "/".join(Path(value).parts[:3])
                for value in paths
            }
            self.assertEqual(1, len(directories), case["case_id"])
            directory = next(iter(directories))
            self.assertIn("/v51_", directory)
            self.assertIsNone(
                BACKGROUND_TERMS.search(directory),
                directory,
            )
            self.assertFalse(any(
                "background" in key.lower()
                for key in case["physics"]
            ))

    def test_new_asset_paths_are_hard_links_to_v5_assets(self) -> None:
        mapping = json.loads(
            (RELEASE_ROOT / "asset_directory_mapping.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(609, len(mapping["records"]))
        self.assertEqual(0, mapping["media_payload_copied_bytes"])
        directories = {
            record["corrected_case_directory"]
            for record in mapping["records"]
        }
        self.assertEqual(609, len(directories))
        for record in mapping["records"]:
            self.assertTrue(record["physical_tokens"])
            for token in record["physical_tokens"]:
                self.assertIn(token, record["corrected_case_directory"])
            for path_record in record["paths"].values():
                previous = DATA_ROOT / path_record["previous"]
                corrected = DATA_ROOT / path_record["corrected"]
                self.assertTrue(os.path.samefile(previous, corrected))

    def test_migration_audit_records_frozen_base_and_zero_leakage(self) -> None:
        audit = json.loads(
            (RELEASE_ROOT / "migration_audit.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(self.v5.digest, audit["base_dataset_digest"])
        self.assertEqual(0, audit[
            "parabolic_train_test_signature_overlap_count"
        ])
        self.assertFalse(audit["background_in_physics"])
        self.assertFalse(audit["background_in_asset_directory_names"])


if __name__ == "__main__":
    unittest.main()
