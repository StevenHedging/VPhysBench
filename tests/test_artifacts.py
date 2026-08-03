from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from physbench.artifacts import (
    import_prediction_video,
    prediction_artifact_manifest,
    validate_prediction_records,
)
from physbench.baseline_runtime import build_i2v_media_contract
from physbench.io import load_json, write_json
from physbench.orchestration import reevaluate_atomic


class PredictionArtifactTests(unittest.TestCase):
    @staticmethod
    def _job() -> dict:
        return {
            "job_id": "job_1",
            "case_id": "case_1",
            "evaluation_partition": "test_id",
            "seed": 42,
        }

    def test_external_prediction_is_imported_with_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "external" / "prediction.mp4"
            source.parent.mkdir()
            source.write_bytes(b"fixture-video")
            run_dir = root / "runs_v2" / "run"

            first = import_prediction_video(
                source=source,
                run_dir=run_dir,
                baseline_id="fixture",
                case_id="case_1",
                job_id="job_1",
                seed=42,
            )
            second = import_prediction_video(
                source=source,
                run_dir=run_dir,
                baseline_id="fixture",
                case_id="case_1",
                job_id="job_1",
                seed=42,
            )

            destination = Path(first["destination_path"])
            self.assertTrue(destination.is_file())
            self.assertEqual(source.read_bytes(), destination.read_bytes())
            self.assertEqual(first["sha256"], second["sha256"])
            manifest = load_json(first["manifest_path"])
            self.assertEqual(1, len(manifest["imports"]))
            self.assertFalse(
                manifest["policy"]["external_prediction_references_allowed"]
            )

    def test_completed_prediction_must_be_run_local(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "run"
            run_dir.mkdir()
            external = root / "external.mp4"
            external.write_bytes(b"external")
            prediction = {
                "baseline_id": "fixture",
                "case_id": "case_1",
                "job_id": "job_1",
                "status": "complete",
                "video_path": str(external),
            }
            with self.assertRaisesRegex(
                ValueError, "must be stored inside AtomicRun"
            ):
                prediction_artifact_manifest([prediction], run_dir)

    def test_run_local_prediction_is_fingerprinted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            video = run_dir / "predictions" / "fixture" / "job_1.mp4"
            video.parent.mkdir(parents=True)
            video.write_bytes(b"run-local")
            manifest = prediction_artifact_manifest([{
                "baseline_id": "fixture",
                "case_id": "case_1",
                "job_id": "job_1",
                "status": "complete",
                "video_path": str(video),
            }], run_dir)
            record = manifest["prediction_videos"][0]
            self.assertEqual(
                "predictions/fixture/job_1.mp4", record["path"]
            )
            self.assertEqual(video.stat().st_size, record["size_bytes"])

    def test_complete_prediction_requires_video_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(
                ValueError, "complete prediction has no run-local video_path"
            ):
                prediction_artifact_manifest(
                    [{"job_id": "job_1", "status": "complete"}],
                    Path(temporary),
                )

    def test_prediction_identity_is_bound_to_frozen_job(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            prediction = {
                "job_id": "job_1",
                "case_id": "different_case",
                "baseline_id": "fixture",
                "evaluation_partition": "test_id",
                "seed": 42,
                "status": "planned",
                "video_path": None,
            }
            with self.assertRaisesRegex(
                ValueError, "prediction identity mismatch"
            ):
                validate_prediction_records(
                    [prediction],
                    jobs=[self._job()],
                    baseline_id="fixture",
                    run_dir=temporary,
                )

    def test_prediction_media_contract_is_bound_to_frozen_job(self) -> None:
        contract = build_i2v_media_contract(
            conditioning_asset="assets/first.png",
            width=480,
            height=832,
            temporal={"fps": 24, "num_frames": 121},
        )
        job = {
            **self._job(),
            "native_inputs": {"media_contract": contract},
        }
        prediction = {
            "job_id": "job_1",
            "case_id": "case_1",
            "baseline_id": "fixture",
            "evaluation_partition": "test_id",
            "seed": 42,
            "status": "planned",
            "video_path": None,
            "media_contract": {
                **contract,
                "output": {
                    **contract["output"],
                    "canvas": {"width": 832, "height": 480},
                },
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(
                ValueError,
                "media_contract differs",
            ):
                validate_prediction_records(
                    [prediction],
                    jobs=[job],
                    baseline_id="fixture",
                    run_dir=temporary,
                )

    def test_validation_detects_prediction_artifact_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            video = run_dir / "predictions" / "fixture" / "job_1.mp4"
            video.parent.mkdir(parents=True)
            video.write_bytes(b"original")
            prediction = {
                "job_id": "job_1",
                "case_id": "case_1",
                "baseline_id": "fixture",
                "evaluation_partition": "test_id",
                "seed": 42,
                "status": "complete",
                "video_path": str(video),
            }
            frozen_artifacts = validate_prediction_records(
                [prediction],
                jobs=[self._job()],
                baseline_id="fixture",
                run_dir=run_dir,
            )
            video.write_bytes(b"modified")
            with self.assertRaisesRegex(
                ValueError, "differ from the frozen AtomicRun manifest"
            ):
                validate_prediction_records(
                    [prediction],
                    jobs=[self._job()],
                    baseline_id="fixture",
                    run_dir=run_dir,
                    expected_artifact_manifest=frozen_artifacts,
                )

    def test_schema_v2_in_place_reevaluation_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            evaluation = run_dir / "evaluation"
            evaluation.mkdir(parents=True)
            write_json(
                run_dir / "run.json",
                {"schema_version": "2.0", "run_id": "sealed-run"},
            )
            sentinel = evaluation / "task_result.json"
            sentinel.write_bytes(b"canonical-evaluation")

            with self.assertRaisesRegex(
                RuntimeError,
                "reevaluate_atomic_variant",
            ):
                reevaluate_atomic(run_dir)

            self.assertEqual(b"canonical-evaluation", sentinel.read_bytes())


if __name__ == "__main__":
    unittest.main()
