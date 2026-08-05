from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import unittest

from physbench.data_layout import LATEST_DATASET, V5_DATASET, V51_DATASET
from physbench.datasets import load_dataset


ROOT = Path(__file__).resolve().parents[1]
RELEASE_ROOT = ROOT / "datasets/releases/5.1.0"
DATA_ROOT = ROOT / "datasets"
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
RETIRED_SYNTHETIC_PENDULUM_OOD_CASE_IDS = {
    "pendulum_ltot0110mm_lrope0100mm_r010mm_a010deg_ood01",
    "pendulum_ltot0130mm_lrope0120mm_r010mm_a030deg_ood02",
    "pendulum_ltot0130mm_lrope0120mm_r010mm_a030deg_ood03",
    "pendulum_ltot0155mm_lrope0145mm_r010mm_a020deg_ood04",
    "pendulum_ltot0155mm_lrope0145mm_r010mm_a020deg_ood05",
}
RETAINED_PENDULUM_PARENT_CASE_IDS = {
    "pendulum_r2_ltot0110mm_lrope0100mm_r010mm_a010deg",
    "pendulum_r2_ltot0130mm_lrope0120mm_r010mm_a030deg",
    "pendulum_r2_ltot0155mm_lrope0145mm_r010mm_a020deg",
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

    def test_v51_retires_only_invalid_synthetic_cases(
        self,
    ) -> None:
        self.assertNotEqual(V51_DATASET, LATEST_DATASET)
        self.assertEqual(
            "physics_video_six_scene_v5p1",
            self.v51.descriptor["dataset_id"],
        )
        self.assertEqual("5.1.0", self.v51.descriptor["release"])
        self.assertEqual(604, len(self.v51.cases))
        self.assertEqual(
            {
                case["case_id"]
                for case in self.v5.cases
            } - RETIRED_SYNTHETIC_PENDULUM_OOD_CASE_IDS,
            set(self.by_id),
        )
        self.assertFalse(
            RETIRED_SYNTHETIC_PENDULUM_OOD_CASE_IDS & self.by_id.keys()
        )
        self.assertTrue(
            RETAINED_PENDULUM_PARENT_CASE_IDS <= self.by_id.keys()
        )

    def test_collision_ball_specs_are_canonical_and_prompts_describe_process(
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
                "collision_process_prompt_v3",
                case["text"]["annotation_source"],
            )
            self.assertIsNone(re.search(r"\d", prompt), prompt)
            structure = case["appearance"]["collision_structure"]
            expected_process = {
                "three_ball_single_incident": (
                    "left ball moves right toward two initially stationary "
                    "balls"
                ),
                "two_ball_single_incident": (
                    "left ball is initially stationary while the right ball "
                    "moves left"
                ),
                "two_ball_opposed_incident": (
                    "left ball moves right while the right ball moves left"
                ),
            }[structure]
            self.assertIn(expected_process, prompt)
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
                self.assertNotIn(spec["prompt"], prompt)
        self.assertEqual(set(SPECS), seen)

    def test_process_prompts_do_not_leak_values_or_visual_hints(self) -> None:
        forbidden = re.compile(
            r"\d|diameter|mass|m/s|shiny|steel|wooden|green|blue|pink|"
            r"background|outside the crop|side of the image",
            re.IGNORECASE,
        )
        for case in self.v51.cases:
            if case["scene_id"] == "free_fall":
                continue
            prompt = case["text"]["prompt"]
            self.assertIsNone(forbidden.search(prompt), case["case_id"])
        expected_sources = {
            "collision_1d": "collision_process_prompt_v3",
            "parabolic_motion": "scene_process_prompt_v2",
            "inclined_plane_slide": "scene_process_prompt_v2",
            "uniform_circular_motion": "scene_process_prompt_v2",
        }
        for case in self.v51.cases:
            source = expected_sources.get(case["scene_id"])
            if source is not None:
                self.assertEqual(source, case["text"]["annotation_source"])

    def test_dataset_assets_preserve_native_timing(self) -> None:
        audit_path = (
            ROOT
            / "datasets/provenance/alignment"
            / "native_timing_20260731_v1/audit.jsonl"
        )
        records = [
            json.loads(line)
            for line in audit_path.read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(406, len(records))
        self.assertEqual(
            {
                "collision_1d": 298,
                "free_fall": 11,
                "parabolic_motion": 97,
            },
            dict(Counter(record["scene_id"] for record in records)),
        )
        for record in records:
            self.assertTrue(record["frame_count_verified"])
            self.assertTrue(record["nominal_frame_rate_verified"])
            self.assertEqual(
                record["expected_reference_frames"],
                record["reference_probe"]["frames"],
            )
            self.assertEqual(
                record["source_probe"]["nominal_frame_rate"],
                record["reference_probe"]["nominal_frame_rate"],
            )
            case = self.by_id[record["case_id"]]
            self.assertEqual(
                "source_timing",
                case["temporal"]["time_scale"],
            )
            if record["kind"] == "byte_identical_source_timing_hardlink":
                source = DATA_ROOT / case["assets"]["source_video"]
                reference = DATA_ROOT / case["assets"]["reference_video"]
                self.assertTrue(source.samefile(reference))
            else:
                alignment = case["alignment"]
                self.assertEqual(
                    alignment["source_end_frame_exclusive"]
                    - alignment["source_start_frame"],
                    alignment["output_frames"],
                )
                self.assertEqual(
                    "preserve_source_fps_and_all_trimmed_frames",
                    alignment["timing_policy"],
                )
        self.assertEqual(
            8.0,
            self.by_id["parabolic_img_0539"]["temporal"][
                "encoded_to_physical_speed"
            ],
        )
        self.assertTrue(all(
            self.by_id[case_id]["temporal"]["encoded_to_physical_speed"]
            == 8.0
            for case_id in self.by_id
            if self.by_id[case_id]["scene_id"] == "free_fall"
        ))

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
            self.assertNotIn("/v51_", directory)
            self.assertIsNone(
                BACKGROUND_TERMS.search(directory),
                directory,
            )
            self.assertFalse(any(
                "background" in key.lower()
                for key in case["physics"]
            ))

    def test_descriptive_asset_paths_match_mapping(self) -> None:
        mapping = json.loads(
            (RELEASE_ROOT / "asset_directory_mapping.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(604, len(mapping["records"]))
        self.assertEqual(0, mapping["media_payload_copied_bytes"])
        directories = {
            record["corrected_case_directory"]
            for record in mapping["records"]
        }
        self.assertEqual(604, len(directories))
        for record in mapping["records"]:
            self.assertTrue(record["physical_tokens"])
            for token in record["physical_tokens"]:
                self.assertIn(token, record["corrected_case_directory"])
            for path_record in record["paths"].values():
                previous = DATA_ROOT / path_record["previous"]
                corrected = DATA_ROOT / path_record["corrected"]
                self.assertFalse(previous.exists())
                self.assertTrue(corrected.is_file())
                self.assertEqual(
                    path_record["sha256"],
                    hashlib.sha256(corrected.read_bytes()).hexdigest(),
                )

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
