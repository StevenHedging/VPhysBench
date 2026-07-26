from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from physbench.artifacts import (
    import_prediction_video,
    prediction_artifact_manifest,
)
from physbench.io import load_json


class PredictionArtifactTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
