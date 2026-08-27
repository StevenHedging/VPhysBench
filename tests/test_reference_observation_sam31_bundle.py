from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import cv2
import numpy as np


def _disk(center_x: int, *, shape: tuple[int, int] = (40, 80)) -> np.ndarray:
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return ((xx - center_x) ** 2 + (yy - 25) ** 2 <= 5**2).astype(np.uint8)


def _fixture(root: Path, *, accepted: bool):
    from physbench.reference_observations.curation.sam31_gt import (
        GtCandidate,
        OrderedGtIdentity,
    )
    from physbench.reference_observations.curation.sam31_rebuild import (
        GtQualityFinding,
        Sam31GtCaseCandidate,
    )

    case_dir = root / "assets/collision_1d/collision_fixture"
    masks_dir = case_dir / "canonical/masks"
    observation_dir = case_dir / "canonical/reference_observation"
    masks_dir.mkdir(parents=True)
    observation_dir.mkdir(parents=True)
    frames = tuple(np.zeros((40, 80, 3), np.uint8) for _ in range(3))
    self_frame = case_dir / "canonical/first_frame.png"
    cv2.imwrite(str(self_frame), frames[0])
    reference = case_dir / "canonical/reference.mp4"
    reference.write_bytes(b"reference-video-fixture")
    timeline = observation_dir / "timeline.json"
    timeline.write_text('{"case_id":"collision_fixture"}\n', encoding="utf-8")
    mask_manifest = masks_dir / "manifest.json"
    mask_manifest.write_text(
        json.dumps(
            {
                "schema_version": "1.2",
                "case_id": "collision_fixture",
                "scene_id": "collision_1d",
                "frame_index": 0,
                "frame_scope": "first_frame_only",
                "image_shape_hw": [40, 80],
                "instances": [],
                "source_first_frame": "assets/collision_1d/collision_fixture/canonical/first_frame.png",
                "storage": {},
            }
        ),
        encoding="utf-8",
    )
    case = SimpleNamespace(
        case_id="collision_fixture",
        scene_id="collision_1d",
        asset_root=root,
        first_frame_path=self_frame,
        reference_video_path=reference,
        mask_manifest_path=mask_manifest,
        observation_manifest_path=observation_dir / "manifest.json",
        visualization_manifest_path=observation_dir / "visualization/manifest.json",
        mask_manifest=json.loads(mask_manifest.read_text()),
        timeline={"sampling_rate_hz": 24.0, "samples": [{}, {}, {}]},
        physics={
            "objects": {
                f"object_{index}": {
                    "mass": {"symbol": f"m_{index}", "value": 0.01},
                    "radius": {"symbol": f"r_{index}", "value": 0.005},
                    "initial_velocity": {"symbol": f"v_{index}", "value": 0.0},
                }
                for index in (1, 2)
            }
        },
        assets={
            "first_frame": "assets/collision_1d/collision_fixture/canonical/first_frame.png",
            "reference_video": "assets/collision_1d/collision_fixture/canonical/reference.mp4",
            "first_frame_mask_manifest": "assets/collision_1d/collision_fixture/canonical/masks/manifest.json",
            "reference_observation_manifest": "assets/collision_1d/collision_fixture/canonical/reference_observation/manifest.json",
            "reference_observation_visualization_manifest": "assets/collision_1d/collision_fixture/canonical/reference_observation/visualization/manifest.json",
        },
    )
    masks = {
        "object_1": np.stack([_disk(15), _disk(17), _disk(19)]),
        "object_2": np.stack([_disk(55), _disk(55), _disk(55)]),
    }
    identities = tuple(
        OrderedGtIdentity(
            object_id=f"object_{index}",
            evaluator_object_id=f"ball_{index}",
            mask_id=f"{index:02d}",
            candidate=GtCandidate.from_mask(
                candidate_id=f"candidate-{index}",
                prompt="small round object",
                backend_object_id=index,
                mask=masks[f"object_{index}"][0],
                confidence=0.9,
            ),
        )
        for index in (1, 2)
    )
    findings = () if accepted else (
        GtQualityFinding(
            code="internal_track_gap",
            severity="error",
            message="synthetic failure",
            object_ids=("object_1",),
            observation_indices=(1,),
        ),
    )
    candidate = Sam31GtCaseCandidate(
        case_id="collision_fixture",
        identities=identities,
        masks_by_object=masks,
        states_by_object={
            "object_1": np.zeros(3, np.uint8),
            "object_2": np.zeros(3, np.uint8),
        },
        source_frame_indices=(0, 10, 20),
        physical_times=(0.0, 1 / 24, 2 / 24),
        findings=findings,
        discovery_attempts=(
            {"source": "primary", "prompts": ["small round object"], "candidate_count": 2},
        ),
        predictor_provenance={"backend": "fixture"},
    )
    return case, candidate, frames


def _target_sources(bundle_path: Path) -> dict[str, Path]:
    value = json.loads(bundle_path.read_text(encoding="utf-8"))
    return {
        item["target_path"]: bundle_path.parent / item["candidate_path"]
        for item in value["files"]
    }


class Sam31GtBundleTests(unittest.TestCase):
    @staticmethod
    def _api():
        from physbench.reference_observations.curation import sam31_bundle

        return sam31_bundle

    def test_bundle_writes_row_major_anchor_ids_and_tube_zero_equality(self) -> None:
        api = self._api()
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            case, candidate, frames = _fixture(root, accepted=True)

            bundle_path = api.write_sam31_candidate_bundle(
                case,
                candidate,
                frames=frames,
                output_root=root / "staged",
                config_fingerprint="1" * 64,
                code_revision="fixture-revision",
            )

            files = _target_sources(bundle_path)
            mask_manifest = json.loads(files["canonical/masks/manifest.json"].read_text())
            self.assertEqual(
                [("object_1", "01"), ("object_2", "02")],
                [(item["object_id"], item["mask_id"]) for item in mask_manifest["instances"]],
            )
            with np.load(files["canonical/masks/01.npz"], allow_pickle=False) as anchor:
                anchor_mask = np.asarray(anchor["masks"])[0]
                self.assertEqual("ball_1", str(anchor["object_ids"][0]))
            from physbench.reference_observations import load_entity_observation

            entity = load_entity_observation(
                files["canonical/reference_observation/entities/object_1/mask_tube.npz"],
                files["canonical/reference_observation/entities/object_1/trajectory.npz"],
                object_id="object_1",
                mask_id="01",
                expected_samples=3,
            )
            np.testing.assert_array_equal(anchor_mask, entity.masks[0])

    def test_bundle_is_hash_closed_and_records_sam31_provenance(self) -> None:
        api = self._api()
        from physbench.reference_observations.curation.bundle import validate_candidate_bundle

        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            case, candidate, frames = _fixture(root, accepted=True)
            bundle_path = api.write_sam31_candidate_bundle(
                case,
                candidate,
                frames=frames,
                output_root=root / "staged",
                config_fingerprint="2" * 64,
                code_revision="fixture-revision",
            )

            bundle = validate_candidate_bundle(bundle_path)
            files = _target_sources(bundle_path)
            observation = json.loads(
                files["canonical/reference_observation/manifest.json"].read_text()
            )

            self.assertEqual("collision_fixture", bundle.case_id)
            self.assertEqual("sam31_collision_gt_curation_v1", observation["generator"]["id"])
            self.assertEqual("SAM 3.1 multiplex", observation["generator"]["model_id"])
            self.assertEqual("fixture-revision", observation["generator"]["code_revision"])

    def test_rejected_candidate_remains_visible_and_pending(self) -> None:
        api = self._api()
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            case, candidate, frames = _fixture(root, accepted=False)
            bundle_path = api.write_sam31_candidate_bundle(
                case,
                candidate,
                frames=frames,
                output_root=root / "staged",
                config_fingerprint="3" * 64,
                code_revision="fixture-revision",
            )
            files = _target_sources(bundle_path)
            quality = json.loads(
                files["canonical/reference_observation/quality.json"].read_text()
            )
            review = json.loads(
                files["canonical/reference_observation/review.json"].read_text()
            )

            self.assertEqual("fail", quality["status"])
            self.assertEqual(["internal_track_gap"], [item["code"] for item in quality["failures"]])
            self.assertEqual("pending", review["decision"])
            self.assertEqual("changes_requested", review["entity_reviews"]["object_1"]["decision"])


if __name__ == "__main__":
    unittest.main()
