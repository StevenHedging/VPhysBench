from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from physbench.evaluation.contracts import CaseEvaluationRequest
from test_reference_observation_storage import (
    ReferenceObservationManifestLoaderTests,
)


class FrozenReferenceObservationTests(unittest.TestCase):
    @staticmethod
    def _request(
        asset_root: Path,
        manifest_path: Path,
        *,
        case_id: str = "case_1",
        scene_id: str = "pendulum",
    ) -> CaseEvaluationRequest:
        return CaseEvaluationRequest(
            job={"job_id": "job_1"},
            case={
                "case_id": case_id,
                "scene_id": scene_id,
                "assets": {
                    "reference_observation_manifest": manifest_path.relative_to(
                        asset_root
                    ).as_posix(),
                },
            },
            case_catalog={},
            prediction=None,
            asset_root=asset_root,
            artifact_dir=asset_root / "artifacts",
            evaluator_config={},
        )

    def test_loads_time_aligned_masks_in_evaluator_canvas(self) -> None:
        from physbench.evaluation.common.frozen_reference import (
            load_frozen_reference_observation,
        )

        with tempfile.TemporaryDirectory() as temporary:
            asset_root, _, manifest_path = (
                ReferenceObservationManifestLoaderTests()._bundle(
                    Path(temporary)
                )
            )
            result = load_frozen_reference_observation(
                self._request(asset_root, manifest_path),
                times_s=[0.0, 1.0],
                spatial_transform={
                    "policy": "preserve_aspect_ratio_letterbox",
                    "source_size": [9, 4],
                    "target_size": [18, 8],
                    "scale": 2.0,
                    "offset_xy": [0, 0],
                },
                expected_entity_ids=["object_1"],
            )

        entity = result.entities["object_1"]
        self.assertEqual((2, 8, 18), entity.masks.shape)
        self.assertEqual(np.uint8, entity.masks.dtype)
        self.assertEqual([24, 24], entity.area_pixels.tolist())
        np.testing.assert_allclose(
            [[6.5, 3.5], [6.5, 3.5]],
            entity.centroid_xy,
        )
        self.assertEqual([0, 1], result.source_observation_indices.tolist())
        self.assertEqual("frozen_dataset_reference_observation_v1", result.policy)

    def test_allows_only_an_off_grid_terminal_sample_to_reuse_nearest_gt(self) -> None:
        from physbench.evaluation.common.frozen_reference import _source_indices

        source = np.arange(5, dtype=np.float64) / 24.0
        requested = np.asarray([0.0, 1.0 / 24.0, 0.05], dtype=np.float64)

        indices = _source_indices(source, requested, sampling_rate_hz=24.0)

        self.assertEqual([0, 1, 1], indices.tolist())

    def test_still_rejects_duplicate_nonterminal_gt_mapping(self) -> None:
        from physbench.evaluation.common.frozen_reference import _source_indices

        source = np.arange(5, dtype=np.float64) / 24.0
        requested = np.asarray([0.0, 0.01, 1.0 / 24.0], dtype=np.float64)

        with self.assertRaisesRegex(ValueError, "duplicate frozen"):
            _source_indices(source, requested, sampling_rate_hz=24.0)

    def test_rejects_case_identity_mismatch(self) -> None:
        from physbench.evaluation.common.frozen_reference import (
            load_frozen_reference_observation,
        )
        from physbench.evaluation.common.errors import ReferenceAnalysisError

        with tempfile.TemporaryDirectory() as temporary:
            asset_root, _, manifest_path = (
                ReferenceObservationManifestLoaderTests()._bundle(
                    Path(temporary)
                )
            )
            with self.assertRaisesRegex(
                ReferenceAnalysisError,
                "Case identity",
            ):
                load_frozen_reference_observation(
                    self._request(
                        asset_root,
                        manifest_path,
                        case_id="case_2",
                    ),
                    times_s=[0.0],
                    spatial_transform={
                        "policy": "preserve_aspect_ratio_letterbox",
                        "source_size": [9, 4],
                        "target_size": [9, 4],
                        "scale": 1.0,
                        "offset_xy": [0, 0],
                    },
                    expected_entity_ids=["object_1"],
                )

    def test_rejects_unrepresented_time_instead_of_interpolating_gt(self) -> None:
        from physbench.evaluation.common.frozen_reference import (
            load_frozen_reference_observation,
        )
        from physbench.evaluation.common.errors import ReferenceAnalysisError

        with tempfile.TemporaryDirectory() as temporary:
            asset_root, _, manifest_path = (
                ReferenceObservationManifestLoaderTests()._bundle(
                    Path(temporary)
                )
            )
            with self.assertRaisesRegex(
                ReferenceAnalysisError,
                "physical-time grid",
            ):
                load_frozen_reference_observation(
                    self._request(asset_root, manifest_path),
                    times_s=[0.5],
                    spatial_transform={
                        "policy": "preserve_aspect_ratio_letterbox",
                        "source_size": [9, 4],
                        "target_size": [9, 4],
                        "scale": 1.0,
                        "offset_xy": [0, 0],
                    },
                    expected_entity_ids=["object_1"],
                )

    def test_rejects_entity_set_mismatch(self) -> None:
        from physbench.evaluation.common.frozen_reference import (
            load_frozen_reference_observation,
        )
        from physbench.evaluation.common.errors import ReferenceAnalysisError

        with tempfile.TemporaryDirectory() as temporary:
            asset_root, _, manifest_path = (
                ReferenceObservationManifestLoaderTests()._bundle(
                    Path(temporary)
                )
            )
            with self.assertRaisesRegex(
                ReferenceAnalysisError,
                "entity identities",
            ):
                load_frozen_reference_observation(
                    self._request(asset_root, manifest_path),
                    times_s=[0.0],
                    spatial_transform={
                        "policy": "preserve_aspect_ratio_letterbox",
                        "source_size": [9, 4],
                        "target_size": [9, 4],
                        "scale": 1.0,
                        "offset_xy": [0, 0],
                    },
                    expected_entity_ids=["object_1", "object_2"],
                )

    def test_binds_evaluator_identity_to_dataset_object_by_order(self) -> None:
        from physbench.evaluation.common.frozen_reference import (
            load_frozen_reference_observation,
        )

        with tempfile.TemporaryDirectory() as temporary:
            asset_root, _, manifest_path = (
                ReferenceObservationManifestLoaderTests()._bundle(
                    Path(temporary)
                )
            )
            result = load_frozen_reference_observation(
                self._request(asset_root, manifest_path),
                times_s=[0.0],
                spatial_transform={
                    "policy": "preserve_aspect_ratio_letterbox",
                    "source_size": [9, 4],
                    "target_size": [9, 4],
                    "scale": 1.0,
                    "offset_xy": [0, 0],
                },
                expected_entity_ids=["logical_ball"],
            )

        self.assertEqual({"logical_ball"}, set(result.entities))
        self.assertEqual(
            "object_1",
            result.entities["logical_ball"].dataset_object_id,
        )
        self.assertEqual(
            {"logical_ball": "object_1"},
            result.provenance["entity_binding"],
        )


if __name__ == "__main__":
    unittest.main()
