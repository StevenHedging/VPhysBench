from __future__ import annotations

import copy
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts import build_dataset_v10 as builder


class CaseLocalPhysicsPreparationTests(unittest.TestCase):
    def test_v9_has_799_unique_case_directories(self) -> None:
        cases = builder.read_jsonl(builder.BASE_RELEASE_ROOT / "cases.jsonl")

        writes, output_cases = builder.prepare_v10_cases(
            cases, builder.DATASETS_ROOT
        )

        self.assertEqual(799, len(writes))
        self.assertEqual(799, len({item.relative_path for item in writes}))
        self.assertTrue(
            all(
                case["assets"]["physics_annotation"].endswith(
                    "/physics.json"
                )
                for case in output_cases
            )
        )

    def test_case_assets_must_resolve_to_one_case_directory(self) -> None:
        case = self._case(
            first_frame="assets/pendulum/case_a/canonical/first_frame.png",
            reference_video="assets/pendulum/case_b/canonical/reference.mp4",
        )
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(
                ValueError, "does not resolve to one Case directory"
            ):
                builder.prepare_v10_cases([case], Path(temporary))

    def test_physics_document_bytes_are_exact_and_deterministic(self) -> None:
        case = self._case()
        with tempfile.TemporaryDirectory() as temporary:
            first, _ = builder.prepare_v10_cases([case], Path(temporary))
            second, _ = builder.prepare_v10_cases([copy.deepcopy(case)], Path(temporary))

        self.assertEqual(first[0].payload, second[0].payload)
        self.assertTrue(first[0].payload.endswith(b"\n"))
        self.assertFalse(first[0].payload.endswith(b"\n\n"))
        self.assertEqual(
            {
                "schema_version": "1.0",
                "case_id": "pendulum_case_1",
                "scene_id": "pendulum",
                "physics": {
                    "gravity": {
                        "value": 9.8,
                        "unit": "m/s^2",
                        "annotated": True,
                    }
                },
            },
            json.loads(first[0].payload),
        )

    def test_conflict_aborts_before_any_missing_document_is_written(self) -> None:
        cases = [
            self._case(case_id="pendulum_case_1", case_directory="case_a"),
            self._case(case_id="pendulum_case_2", case_directory="case_b"),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            writes, _ = builder.prepare_v10_cases(cases, root)
            writes[1].absolute_path.parent.mkdir(parents=True)
            writes[1].absolute_path.write_bytes(b"conflicting bytes\n")

            with self.assertRaisesRegex(ValueError, "conflicting physics document"):
                builder.materialize_physics_documents(writes)

            self.assertFalse(writes[0].absolute_path.exists())
            self.assertEqual(
                b"conflicting bytes\n", writes[1].absolute_path.read_bytes()
            )

    def test_real_legacy_cleanup_preflight_enumerates_exact_32_directories(self) -> None:
        cases = builder.read_jsonl(builder.BASE_RELEASE_ROOT / "cases.jsonl")

        removals = builder.preflight_legacy_cleanup(cases)

        self.assertEqual(32, len(removals))
        self.assertEqual(32, len({item.relative_directory for item in removals}))
        for item in removals:
            self.assertTrue(item.absolute_directory.name.startswith("collision_r2_"))
            self.assertEqual(
                "source/first_frame_source.png", item.relative_file
            )
            active = next(case for case in cases if case["case_id"] == item.case_id)
            self.assertNotIn(
                f"assets/collision_1d/{item.absolute_directory.name}/canonical/",
                active["assets"]["first_frame"],
            )

    def test_legacy_cleanup_preflight_rejects_an_extra_file(self) -> None:
        case_id = "collision_r2_fixture"
        case = self._case(
            case_id=case_id,
            scene_id="collision_1d",
            case_directory="collision_descriptive_fixture",
        )
        with tempfile.TemporaryDirectory() as temporary:
            datasets_root = Path(temporary)
            legacy = datasets_root / "assets" / "collision_1d" / case_id
            (legacy / "source").mkdir(parents=True)
            (legacy / "source" / "first_frame_source.png").write_bytes(b"png")
            (legacy / "unexpected.txt").write_text("no", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "unexpected content"):
                builder.preflight_legacy_cleanup(
                    [case],
                    datasets_root=datasets_root,
                    historical_cases=[case],
                    expected_count=1,
                )

    def test_git_ignore_admits_only_case_root_physics_json(self) -> None:
        paths = {
            "datasets/assets/pendulum/example/physics.json": False,
            "datasets/assets/pendulum/example/canonical/reference.mp4": True,
            "datasets/assets/pendulum/example/canonical/masks/01.npz": True,
            "datasets/assets/pendulum/example/source/reference.mov": True,
        }
        for path, ignored in paths.items():
            with self.subTest(path=path):
                result = subprocess.run(
                    ["git", "check-ignore", "--no-index", "-q", path],
                    cwd=builder.ROOT,
                    check=False,
                )
                self.assertEqual(0 if ignored else 1, result.returncode)

    @staticmethod
    def _case(
        *,
        case_id: str = "pendulum_case_1",
        scene_id: str = "pendulum",
        case_directory: str = "case_a",
        first_frame: str | None = None,
        reference_video: str | None = None,
    ) -> dict:
        first_frame = first_frame or (
            f"assets/{scene_id}/{case_directory}/canonical/first_frame.png"
        )
        reference_video = reference_video or (
            f"assets/{scene_id}/{case_directory}/canonical/reference.mp4"
        )
        return {
            "schema_version": "4.0",
            "case_id": case_id,
            "scene_id": scene_id,
            "assets": {
                "first_frame": first_frame,
                "reference_video": reference_video,
                "physics_reference_video": reference_video,
            },
            "physics": {
                "gravity": {
                    "value": 9.8,
                    "unit": "m/s^2",
                    "annotated": True,
                }
            },
        }


if __name__ == "__main__":
    unittest.main()
