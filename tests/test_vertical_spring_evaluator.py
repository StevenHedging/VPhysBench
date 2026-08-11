from __future__ import annotations

import importlib.util
import json
import math
import os
import unittest
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import cv2
import numpy as np
from _pytest.mark.structures import Mark

try:
    from jsonschema import Draft202012Validator
except ImportError:  # pragma: no cover - exercised by minimal release installs
    Draft202012Validator = None  # type: ignore[assignment,misc]

from physbench.evaluation.common.errors import (
    ReferenceAnalysisError,
    SceneAnalysisError,
)
from physbench.evaluation.common.frozen_subject import FrozenSubjectAnchor
from physbench.evaluation.common.masks.sam2 import Sam2VideoSegmenter
from physbench.evaluation.common.media import (
    SampledVideo,
    VideoInfo,
    probe_video,
    resolve_evaluation_timeline,
)
from physbench.evaluation.contracts import CaseEvaluationRequest
from physbench.evaluation import load_evaluation_protocol
from physbench.evaluation.registry import SceneEvaluatorRegistry
from physbench.evaluation.scenes.vertical_spring_oscillator.observation import (
    observe_spring_topology,
    prompt_from_anchor,
    validate_mask_tube,
    validate_prediction_identity,
)
from physbench.io import write_json


IDENTITY = {
    "minimum_anchor_iou": 0.60,
    "maximum_centroid_distance_radii": 1.50,
    "minimum_anchor_area_ratio": 0.50,
    "maximum_anchor_area_ratio": 1.80,
}

MASK_QUALITY = {
    "minimum_mask_pixels": 40,
    "maximum_mask_area_ratio": 0.10,
    "minimum_anchor_area_ratio": 0.50,
    "maximum_anchor_area_ratio": 1.80,
}

TOPOLOGY = {
    "canny_low_threshold": 40,
    "canny_high_threshold": 120,
    "corridor_half_width_radius_ratio": 1.50,
    "minimum_edge_pixels_per_row": 2,
    "endpoint_height_radius_ratio": 1.00,
    "boundary_exclusion_px": 2,
    "connectivity_dilation_px": 1,
    "minimum_corridor_height_radius_ratio": 2.00,
    "minimum_connected_vertical_span_ratio": 0.60,
}

EVALUATOR_CONFIG = {
    "type": "vertical_spring_oscillator_v1",
    "evaluator_contract": "robust_subject_v3",
    "timeline": {
        "policy": "physical_overlap_common_fps_v1",
        "fps": 24,
        "minimum_evaluation_fps": 8,
        "minimum_source_fps": 8,
        "duration_tolerance_s": 0.02,
        "decode_policy": "sequential_forward",
    },
    "spatial": {
        "width": 480,
        "height": 832,
        "policy": "shared_reference_content_no_pad_v1",
        "pad_value": 0,
    },
    "sam2": {
        "model_id": "facebook/sam2.1-hiera-tiny",
        "device": "auto",
    },
    "quality": {
        "minimum_mask_pixels": 40,
        "minimum_mask_area_ratio": 0.002,
        "maximum_mask_area_ratio": 0.10,
        "minimum_anchor_area_ratio": 0.50,
        "maximum_anchor_area_ratio": 1.80,
        "minimum_valid_frame_ratio": 0.90,
        "minimum_amplitude_px": 5.0,
    },
    "period": {
        "minimum_s": 0.35,
        "maximum_s": 1.20,
        "minimum_period_correlation": 0.60,
        "minimum_period_peak_prominence": 0.001,
        "minimum_complete_cycles": 2.0,
        "minimum_subharmonic_residual_px": 0.15,
        "maximum_cadence_relative_deviation": 0.02,
    },
    "scoring": {
        "weights": {
            "vertical_trajectory": 0.30,
            "period": 0.15,
            "amplitude_envelope": 0.15,
            "equilibrium_release_phase": 0.15,
            "vertical_axis_confinement": 0.10,
            "oscillation_evidence": 0.15,
        },
        "trajectory_scale": 1.0,
        "amplitude_scale": 0.30,
        "equilibrium_scale": 0.30,
        "axis_drift_scale": 0.30,
        "oscillation_amplitude_scale": 0.30,
        "minimum_period_correlation": 0.60,
        "minimum_period_peak_prominence": 0.001,
        "minimum_complete_cycles": 2.0,
        "minimum_subharmonic_residual_px": 0.15,
    },
    "subject_scoring": {
        "minimum_observed_pixels": 4,
        "position_distance_scale": 0.08,
        "canonical_crop_size": 64,
        "boundary_tolerance_px": 2,
        "weights": {"position": 0.5, "shape": 0.2, "appearance": 0.3},
        "case_weights": {"physics_state": 0.55, "subject": 0.25},
        "parent_weights": {
            "physics_state": 0.7,
            "conditioned_appearance": 0.3,
        },
    },
    "identity": IDENTITY,
    "topology": TOPOLOGY,
    "content_weights": {
        "physics_state": 0.55,
        "subject": 0.25,
        "topology": 0.20,
    },
    "general_metrics": {
        "csti": {
            "enabled": True,
            "algorithm": "exact_full_tube_edt",
            "spatial_tolerance_fraction": 0.004204482076268572,
            "temporal_tolerance_s": 0.025,
            "condition_frame_policy": "exclude_initial_samples",
            "initial_frames_excluded": 3,
            "score_aggregation": "full_tube",
            "diagnostic_prefix_fractions": [0.25, 0.5, 0.75, 1],
            "case_aggregation": "mean_gt_entities",
            "timeline_policy": "physical_overlap",
            "mask_resolution": "scene_analysis_native",
        }
    },
}


def circle_mask(center_x: int, center_y: int, *, radius: int = 6) -> np.ndarray:
    """Rasterize a hand-specified disk without observation helpers."""
    y, x = np.ogrid[:96, :96]
    return (
        (x - center_x) ** 2 + (y - center_y) ** 2 <= radius**2
    ).astype(np.uint8)


def frozen_circle_anchor(
    *, center_x: int = 48, center_y: int = 72, radius: int = 6
) -> FrozenSubjectAnchor:
    mask = circle_mask(center_x, center_y, radius=radius)
    area = float(np.count_nonzero(mask))
    source = mask.copy()
    mask.flags.writeable = False
    source.flags.writeable = False
    return FrozenSubjectAnchor(
        case_id="spring_case",
        logical_entity_id="oscillator_ball",
        dataset_object_id="ball_0",
        entity_class="steel_ball",
        manifest_path=Path("mask_manifest.json"),
        npz_path=Path("ball_0.npz"),
        source_mask=source,
        mask=mask,
        centroid_xy=np.asarray([center_x, center_y], dtype=np.float64),
        area_px2=area,
        equivalent_radius_px=float(math.sqrt(area / math.pi)),
        provenance={"policy": "frozen-like-test-anchor"},
    )


def spring_frame(
    ball_mask: np.ndarray, *, attached: bool, ruler_and_border: bool = True
) -> np.ndarray:
    """Draw a literal current-frame zig-zag spring and unrelated scene edges."""
    frame = np.zeros((*ball_mask.shape, 3), dtype=np.uint8)
    ys, xs = np.where(ball_mask > 0)
    center_x = int(round(float(xs.mean())))
    ball_top = int(ys.min())
    if ruler_and_border:
        cv2.rectangle(frame, (1, 1), (94, 94), (180, 180, 180), 1)
        cv2.line(frame, (10, 4), (10, 88), (255, 255, 255), 2)
        for y in range(8, 72, 8):
            cv2.line(frame, (10, y), (16, y), (255, 255, 255), 1)
    if attached:
        points = [(center_x, 2)]
        direction = -1
        for y in range(6, ball_top, 4):
            points.append((center_x + direction * 4, y))
            direction *= -1
        points.append((center_x, ball_top))
        cv2.polylines(
            frame,
            [np.asarray(points, dtype=np.int32)],
            False,
            (255, 255, 255),
            2,
            lineType=cv2.LINE_8,
        )
    frame[ball_mask > 0] = (160, 160, 160)
    return frame


def sinusoidal_spring_masks(times_s: np.ndarray) -> list[np.ndarray]:
    """Rasterize an independent 0.8 s vertical oscillator fixture."""
    return [
        circle_mask(
            48,
            int(round(48.0 + 16.0 * math.cos(2.0 * math.pi * time_s / 0.8))),
        )
        for time_s in times_s
    ]


def sampled_video(
    frames: list[np.ndarray], *, available: list[bool] | None = None
) -> SampledVideo:
    count = len(frames)
    times = (np.arange(count, dtype=float) / 24.0).tolist()
    return SampledVideo(
        frames=frames,
        info=VideoInfo(
            frame_count=count,
            fps=24.0,
            width=96,
            height=96,
            last_frame_time_s=times[-1],
        ),
        sample_times_s=times,
        source_indices=list(range(count)),
        spatial_transform={
            "policy": "reference_content_crop_resize_no_pad",
            "crop_xywh": [0, 0, 96, 96],
            "scale": 1.0,
            "source_size": [96, 96],
            "target_size": [96, 96],
            "padding": None,
        },
        available=available,
        temporal_transform={"source_time_scale": 1.0},
    )


def spring_case() -> dict[str, object]:
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
                }
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
                    "value": 6.168502750680848,
                    "unit": "N/m",
                    "symbol": "k",
                },
            },
        },
        "assets": {
            "first_frame": "condition.png",
            "first_frame_mask_manifest": "masks/manifest.json",
            "reference_video": "reference.mp4",
        },
    }


def write_frozen_spring_anchor(
    root: Path, *, case: dict[str, object], mask: np.ndarray
) -> None:
    binary_mask = np.where(mask > 0, 1, 0).astype(np.uint8)
    height, width = binary_mask.shape
    npz_path = root / "masks/ball_0.npz"
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        masks=binary_mask[None, ...],
        mask_ids=np.asarray(["spring-ball-mask"]),
        object_ids=np.asarray(["oscillator_ball"]),
        frame_index=np.asarray(0, dtype=np.int64),
    )
    ys, xs = np.where(binary_mask > 0)
    write_json(
        root / "masks/manifest.json",
        {
            "schema_version": "1.2",
            "case_id": case["case_id"],
            "scene_id": case["scene_id"],
            "frame_index": 0,
            "frame_scope": "first_frame_only",
            "source_first_frame": "condition.png",
            "image_shape_hw": [height, width],
            "instances": [
                {
                    "mask_id": "spring-ball-mask",
                    "object_id": "ball_0",
                    "entity_class": "steel_ball",
                    "npz_asset": "masks/ball_0.npz",
                    "area_pixels": int(xs.size),
                    "bbox_xyxy": [
                        int(xs.min()),
                        int(ys.min()),
                        int(xs.max()) + 1,
                        int(ys.max()) + 1,
                    ],
                    "centroid_xy": [float(xs.mean()), float(ys.mean())],
                }
            ],
            "storage": {
                "model": {
                    "array_key": "masks",
                    "layout": "1HW",
                    "dtype": "uint8",
                    "values": [0, 1],
                }
            },
        },
    )


def write_video(path: Path, frames: list[np.ndarray]) -> None:
    if not frames:
        raise ValueError("test video requires at least one frame")
    height, width = frames[0].shape[:2]
    if any(frame.shape[:2] != (height, width) for frame in frames):
        raise ValueError("test video frames must share one canvas")
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 24.0, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError("test video writer could not open")
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()


def resize_mask_tube(
    masks: list[np.ndarray], *, width: int, height: int
) -> list[np.ndarray]:
    """Mirror SAM2's native evaluation-canvas mask contract."""
    return [
        cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        for mask in masks
    ]


def evaluation_sinusoidal_masks(
    times_s: list[float], *, width: int, height: int
) -> list[np.ndarray]:
    """Rasterize the same motion directly on the evaluation canvas."""
    center_x = int(round((48.0 + 0.5) * width / 96.0 - 0.5))
    equilibrium_y = (48.0 + 0.5) * height / 96.0 - 0.5
    amplitude_y = 16.0 * height / 96.0
    radius = int(round(6.0 * math.sqrt(width * height) / 96.0))
    output: list[np.ndarray] = []
    for time_s in times_s:
        mask = np.zeros((height, width), dtype=np.uint8)
        center_y = int(
            round(
                equilibrium_y
                + amplitude_y * math.cos(2.0 * math.pi * time_s / 0.8)
            )
        )
        cv2.circle(mask, (center_x, center_y), radius, 255, -1)
        output.append(mask)
    return output


@unittest.skipIf(
    Draft202012Validator is None,
    "jsonschema unavailable: install a Draft 2020-12 consumer",
)
class VerticalSpringProtocolSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = Path(__file__).resolve().parents[1]
        cls.schema = json.loads(
            (root / "schemas/evaluation_protocol.schema.json").read_text(
                encoding="utf-8"
            )
        )
        cls.protocol = json.loads(
            (
                root
                / "configs/evaluation/protocols/scene_default_v1.json"
            ).read_text(encoding="utf-8")
        )
        assert Draft202012Validator is not None
        Draft202012Validator.check_schema(cls.schema)
        cls.validator = Draft202012Validator(cls.schema)

    def assert_protocol_mutation_rejected(
        self,
        *,
        container_path: tuple[str, ...],
        key: str,
        mutation: str,
        validator_keyword: str,
    ) -> None:
        candidate = deepcopy(self.protocol)
        container = candidate
        for segment in container_path:
            container = container[segment]
        if mutation == "missing":
            container.pop(key)
        elif mutation == "unknown":
            container[key] = 0.5
        else:  # pragma: no cover - test helper misuse
            raise AssertionError(f"unknown schema mutation: {mutation}")

        matching = [
            error
            for error in self.validator.iter_errors(candidate)
            if error.validator == validator_keyword
            and tuple(error.absolute_path) == container_path
        ]
        self.assertTrue(
            matching,
            f"Draft 2020-12 accepted {mutation} {'.'.join((*container_path, key))}",
        )

    def test_draft202012_rejects_missing_and_unknown_spring_scene(self) -> None:
        """Would fail if the public consumer can omit or invent a scene route."""
        self.assertEqual([], list(self.validator.iter_errors(self.protocol)))
        self.assert_protocol_mutation_rejected(
            container_path=("scenes",),
            key="vertical_spring_oscillator",
            mutation="missing",
            validator_keyword="required",
        )
        self.assert_protocol_mutation_rejected(
            container_path=("scenes",),
            key="vertical_spring_oscillator_typo",
            mutation="unknown",
            validator_keyword="additionalProperties",
        )

    def test_draft202012_rejects_missing_and_unknown_nested_spring_keys(
        self,
    ) -> None:
        """Would fail if nested spring settings silently default or admit typos."""
        spring_path = ("scenes", "vertical_spring_oscillator")
        self.assert_protocol_mutation_rejected(
            container_path=(*spring_path, "topology"),
            key="minimum_connected_vertical_span_ratio",
            mutation="missing",
            validator_keyword="required",
        )
        self.assert_protocol_mutation_rejected(
            container_path=(*spring_path, "identity"),
            key="minimum_anchor_io",
            mutation="unknown",
            validator_keyword="additionalProperties",
        )


class VerticalSpringRealAssetTests(unittest.TestCase):
    pytestmark = Mark("real_assets", (), {}, _ispytest=True)

    @classmethod
    def setUpClass(cls) -> None:
        configured_root = os.environ.get("PHYSBENCH_FULL_ASSET_ROOT")
        if not configured_root:
            raise unittest.SkipTest("PHYSBENCH_FULL_ASSET_ROOT is unset")
        cls.full_asset_root = Path(configured_root).expanduser().resolve()
        if not cls.full_asset_root.is_dir():
            raise unittest.SkipTest(
                "PHYSBENCH_FULL_ASSET_ROOT is not a directory: "
                f"{cls.full_asset_root}"
            )
        for dependency in ("torch", "sam2", "huggingface_hub"):
            if importlib.util.find_spec(dependency) is None:
                raise unittest.SkipTest(
                    f"SAM2 dependency unavailable: {dependency} is not installed"
                )

        repository_root = Path(__file__).resolve().parents[1]
        indexed_cases = [
            json.loads(line)
            for line in (
                repository_root
                / "datasets/releases/13.0.0/cases.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        spring_cases = [
            candidate
            for candidate in indexed_cases
            if candidate.get("scene_id") == "vertical_spring_oscillator"
        ]
        protocol = load_evaluation_protocol("scene_default_v1")
        spring_config = protocol["scenes"]["vertical_spring_oscillator"]
        cadence_tolerance = float(
            spring_config["period"]["maximum_cadence_relative_deviation"]
        )
        cls.cases = {}
        for release_side in ("above", "below"):
            eligible: list[tuple[int, str, dict[str, object]]] = []
            for candidate in spring_cases:
                if (
                    candidate.get("appearance", {}).get("release_side")
                    != release_side
                ):
                    continue
                assets = candidate.get("assets")
                if not isinstance(assets, dict):
                    continue
                reference_path = cls._asset_path(
                    assets.get("reference_video"), role="reference_video"
                )
                info = probe_video(reference_path)
                timeline = resolve_evaluation_timeline(
                    reference_info=info,
                    prediction_info=info,
                    config=spring_config["timeline"],
                )
                steps = np.diff(timeline.sample_times_s)
                median_step = float(np.median(steps))
                deviation = float(
                    np.max(np.abs(steps - median_step)) / median_step
                )
                if deviation <= cadence_tolerance:
                    eligible.append(
                        (
                            len(timeline.sample_times_s),
                            str(candidate["case_id"]),
                            candidate,
                        )
                    )
            if not eligible:
                raise AssertionError(
                    "canonical Dataset has no spring identity control within "
                    f"the protocol cadence gate for release_side={release_side}"
                )
            selected = min(eligible)[2]
            cls.cases[release_side] = cls._materialize_case(selected)

        cls.evaluator = SceneEvaluatorRegistry(protocol).resolve(
            "vertical_spring_oscillator"
        )
        cls.temporary = TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.artifact_root = Path(cls.temporary.name)

    @classmethod
    def _asset_path(cls, relative: object, *, role: str) -> Path:
        if not isinstance(relative, str) or not relative:
            raise unittest.SkipTest(f"full asset missing: assets.{role} is unset")
        parts = Path(relative).parts
        if not parts or parts[0] != "assets":
            raise AssertionError(
                f"canonical assets.{role} is not rooted at assets/: {relative}"
            )
        resolved = cls.full_asset_root.joinpath(*parts[1:]).resolve()
        try:
            resolved.relative_to(cls.full_asset_root)
        except ValueError as exc:
            raise AssertionError(
                f"canonical assets.{role} escapes the full asset root"
            ) from exc
        if not resolved.is_file():
            raise unittest.SkipTest(
                f"full asset missing for assets.{role}: {relative}"
            )
        return resolved

    @classmethod
    def _materialize_case(cls, indexed: dict[str, object]) -> dict[str, object]:
        assets = indexed.get("assets")
        if not isinstance(assets, dict):
            raise AssertionError("canonical spring Case assets must be an object")
        for role in (
            "caption",
            "physics_annotation",
            "first_frame",
            "first_frame_mask_manifest",
            "reference_video",
        ):
            cls._asset_path(assets.get(role), role=role)

        caption = json.loads(
            cls._asset_path(assets["caption"], role="caption").read_text(
                encoding="utf-8"
            )
        )
        physics = json.loads(
            cls._asset_path(
                assets["physics_annotation"], role="physics_annotation"
            ).read_text(encoding="utf-8")
        )
        manifest = json.loads(
            cls._asset_path(
                assets["first_frame_mask_manifest"],
                role="first_frame_mask_manifest",
            ).read_text(encoding="utf-8")
        )
        for instance in manifest.get("instances", []):
            cls._asset_path(instance.get("npz_asset"), role="frozen_mask_npz")

        case = deepcopy(indexed)
        case["text"] = {"prompt": caption["caption"]}
        case["physics"] = physics["physics"]
        return case

    def assert_real_reference_identity(self, release_side: str) -> None:
        case = self.cases[release_side]
        assets = case["assets"]
        assert isinstance(assets, dict)
        reference_path = self._asset_path(
            assets["reference_video"], role="reference_video"
        )
        request = CaseEvaluationRequest(
            job={"job_id": f"real_spring_identity_{release_side}"},
            case=case,
            case_catalog={str(case["case_id"]): case},
            prediction={
                "status": "complete",
                "video_path": str(reference_path),
            },
            asset_root=self.full_asset_root.parent,
            artifact_dir=self.artifact_root / release_side,
            evaluator_config=deepcopy(self.evaluator.config),
        )
        try:
            result = self.evaluator.evaluate(request)
        except RuntimeError as exc:
            explicit_environment_blockers = (
                "SAM2 segmenter internal failure (sam2_dependency_missing):",
                "SAM2 segmenter internal failure (sam2_model_load_failed):",
                "SAM2 segmenter internal failure (cuda_unavailable):",
            )
            if str(exc).startswith(explicit_environment_blockers):
                self.skipTest(str(exc))
            raise

        self.assertEqual("evaluated", result.status, result.to_dict())
        assert result.score is not None
        self.assertGreaterEqual(result.score, 0.98, result.to_dict())
        self.assertGreaterEqual(
            result.metrics["vertical_spring_dynamics_similarity"]["score"],
            0.98,
            result.to_dict(),
        )
        self.assertEqual(1.0, result.metrics["csti"]["score"])

    def test_real_above_reference_identity_scores_near_one(self) -> None:
        self.assert_real_reference_identity("above")

    def test_real_below_reference_identity_scores_near_one(self) -> None:
        self.assert_real_reference_identity("below")


class VerticalSpringEvaluatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.times_array = np.arange(96, dtype=float) / 24.0
        self.times = self.times_array.tolist()
        self.masks = sinusoidal_spring_masks(self.times_array)
        self.frames = [
            spring_frame(mask, attached=True, ruler_and_border=False)
            for mask in self.masks
        ]
        self.case = spring_case()
        write_frozen_spring_anchor(
            self.root, case=self.case, mask=self.masks[0]
        )
        self.request = CaseEvaluationRequest(
            job={"job_id": "spring_eval"},
            case=self.case,
            case_catalog={str(self.case["case_id"]): self.case},
            prediction={"status": "complete", "video_path": "unused.mp4"},
            asset_root=self.root,
            artifact_dir=self.root / "artifacts",
            evaluator_config=deepcopy(EVALUATOR_CONFIG),
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def evaluator(config: dict[str, object] | None = None):
        from physbench.evaluation.scenes.vertical_spring_oscillator.evaluator import (
            VerticalSpringOscillatorCaseEvaluator,
        )

        return VerticalSpringOscillatorCaseEvaluator(
            deepcopy(EVALUATOR_CONFIG if config is None else config)
        )

    def test_real_portrait_mp4s_share_exact_no_pad_spring_transform(
        self,
    ) -> None:
        """Would fail if the evaluator pads, distorts, or splits portrait roles."""
        height, width = 832, 480
        portrait_masks = evaluation_sinusoidal_masks(
            self.times,
            width=width,
            height=height,
        )
        portrait_frames = [
            spring_frame(mask, attached=True, ruler_and_border=False)
            for mask in portrait_masks
        ]
        reference_path = self.root / "reference.mp4"
        prediction_path = self.root / "portrait-prediction.mp4"
        write_video(reference_path, portrait_frames)
        write_video(prediction_path, portrait_frames)
        write_frozen_spring_anchor(
            self.root,
            case=self.case,
            mask=portrait_masks[0],
        )
        self.request.prediction["video_path"] = str(prediction_path)  # type: ignore[index]
        config = load_evaluation_protocol("scene_default_v1")["scenes"][
            "vertical_spring_oscillator"
        ]
        with patch.object(
            Sam2VideoSegmenter,
            "segment",
            return_value=(
                portrait_masks,
                {"backend": "deterministic_test_sam2"},
            ),
        ):
            result = self.evaluator(config).evaluate(self.request)

        self.assertEqual("evaluated", result.status, result.to_dict())
        contract = result.provenance["shared_media_contract"]
        self.assertEqual([480, 832], contract["target_size"])
        self.assertEqual([0, 0, 480, 832], contract["reference_crop_xywh"])
        self.assertEqual(
            contract["reference_crop_xywh"],
            contract["prediction_crop_xywh"],
        )
        self.assertFalse(contract["padding_used_for_evaluation"])
        self.assertFalse(contract["aspect_ratio_distortion"])
        sampling = result.provenance["sampling"]
        self.assertEqual(
            sampling["reference"]["spatial_transform"],
            sampling["prediction"]["spatial_transform"],
        )
        self.assertEqual(
            {
                "policy": "reference_content_crop_resize_no_pad",
                "crop_xywh": [0, 0, 480, 832],
                "scale": 1.0,
                "source_size": [480, 832],
                "target_size": [480, 832],
                "padding": None,
            },
            sampling["reference"]["spatial_transform"],
        )

    def test_identity_analysis_scores_one_reuses_segmentation_and_binds_csti(
        self,
    ) -> None:
        """Would fail if identity, one-pass reuse, scoring, or CSTI binding regresses."""
        video = sampled_video(self.frames)
        with patch.object(
            Sam2VideoSegmenter,
            "segment",
            return_value=(self.masks, {"backend": "deterministic_test_sam2"}),
        ) as segment:
            analysis = self.evaluator().analyze(
                self.request,
                times_s=self.times,
                reference_video=video,
                prediction_video=video,
            )

        self.assertEqual(1, segment.call_count)
        self.assertEqual(1.0, analysis.score)
        self.assertEqual(
            1.0,
            analysis.metrics["vertical_spring_oscillator_similarity"]["score"],
        )
        self.assertIsNotNone(analysis.csti_input)
        assert analysis.csti_input is not None
        self.assertEqual(1, len(analysis.csti_input.entities))
        entity = analysis.csti_input.entities[0]
        self.assertEqual("oscillator_ball", entity.entity_id)
        self.assertEqual("spring_oscillator", entity.role_id)
        self.assertEqual(("bound_spring_ball",), entity.matched_prediction_track_ids)
        self.assertIsNotNone(entity.prediction_masks)
        self.assertEqual("same_case_gt", analysis.csti_input.reference_capability.value)
        self.assertEqual(
            {
                "per_frame_csv",
                "trajectory_curve",
                "subject_components_csv",
                "subject_similarity_curve",
                "audit_json",
            },
            set(analysis.artifacts),
        )

    def test_removed_spring_cannot_earn_identity_motion_credit(self) -> None:
        """Would fail if ball identity and motion can compensate for no spring."""
        reference = sampled_video(self.frames)
        removed_frames = [
            spring_frame(mask, attached=False, ruler_and_border=False)
            for mask in self.masks
        ]
        prediction = sampled_video(removed_frames)
        with patch.object(
            Sam2VideoSegmenter,
            "segment",
            side_effect=[
                (self.masks, {"role": "reference"}),
                (self.masks, {"role": "prediction"}),
            ],
        ):
            analysis = self.evaluator().analyze(
                self.request,
                times_s=self.times,
                reference_video=reference,
                prediction_video=prediction,
            )

        dynamics = analysis.metrics["vertical_spring_dynamics_similarity"]
        topology = analysis.metrics["vertical_spring_topology_integrity"]
        self.assertEqual(1.0, dynamics["score"])
        self.assertLess(topology["score"], 0.05)
        self.assertLess(analysis.score, 0.05)

    def test_prediction_unavailable_samples_are_empty_in_the_accepted_csti_tube(
        self,
    ) -> None:
        """Would fail if partial media availability leaks a fabricated subject mask."""
        availability = [True] * len(self.frames)
        availability[20] = False
        reference = sampled_video(self.frames)
        prediction_frames = [frame.copy() for frame in self.frames]
        prediction_frames[0][0, 0] = (1, 2, 3)
        prediction = sampled_video(prediction_frames, available=availability)
        with patch.object(
            Sam2VideoSegmenter,
            "segment",
            side_effect=[
                (self.masks, {"role": "reference"}),
                (self.masks, {"role": "prediction"}),
            ],
        ):
            analysis = self.evaluator().analyze(
                self.request,
                times_s=self.times,
                reference_video=reference,
                prediction_video=prediction,
            )

        assert analysis.csti_input is not None
        tube = analysis.csti_input.entities[0].prediction_masks
        assert tube is not None
        self.assertEqual(0, np.count_nonzero(tube[20]))
        self.assertGreater(np.count_nonzero(tube[19]), 0)

    def test_segmenter_internal_failures_are_not_charged_to_either_video(
        self,
    ) -> None:
        """Would fail if SAM2 infrastructure faults become benchmark outcomes."""
        reference = sampled_video(self.frames)
        prediction_frames = [frame.copy() for frame in self.frames]
        prediction_frames[0][0, 0] = (1, 2, 3)
        prediction = sampled_video(prediction_frames)
        with patch.object(
            Sam2VideoSegmenter,
            "segment",
            side_effect=RuntimeError("reference model crashed"),
        ):
            with self.assertRaises(RuntimeError) as caught:
                self.evaluator().analyze(
                    self.request,
                    times_s=self.times,
                    reference_video=reference,
                    prediction_video=prediction,
                )
        self.assertIs(type(caught.exception), RuntimeError)
        self.assertEqual("reference model crashed", str(caught.exception))

        with patch.object(
            Sam2VideoSegmenter,
            "segment",
            side_effect=[
                (self.masks, {"role": "reference"}),
                SceneAnalysisError(
                    "sam2_model_load_failed", "prediction model failed"
                ),
            ],
        ):
            with self.assertRaises(RuntimeError) as caught:
                self.evaluator().analyze(
                    self.request,
                    times_s=self.times,
                    reference_video=reference,
                    prediction_video=prediction,
                )
        self.assertNotIsInstance(caught.exception, SceneAnalysisError)
        self.assertIn("sam2_model_load_failed", str(caught.exception))

    def test_scorer_runtime_failure_is_not_a_prediction_zero(self) -> None:
        """Would fail if a scorer bug is relabelled as prediction invalidity."""
        video = sampled_video(self.frames)
        with (
            patch.object(
                Sam2VideoSegmenter,
                "segment",
                return_value=(self.masks, {"role": "reference"}),
            ),
            patch(
                "physbench.evaluation.scenes.vertical_spring_oscillator."
                "evaluator.score_spring_traces",
                side_effect=RuntimeError("scorer implementation defect"),
            ),
        ):
            with self.assertRaises(RuntimeError) as caught:
                self.evaluator().analyze(
                    self.request,
                    times_s=self.times,
                    reference_video=video,
                    prediction_video=video,
                )
        self.assertIs(type(caught.exception), RuntimeError)
        self.assertEqual("scorer implementation defect", str(caught.exception))

    def test_invalid_content_weight_config_is_an_internal_error(self) -> None:
        """Would fail if malformed evaluator config is charged to a prediction."""
        config = deepcopy(EVALUATOR_CONFIG)
        config["content_weights"] = {"physics_state": 1.0}
        video = sampled_video(self.frames)
        with patch.object(
            Sam2VideoSegmenter,
            "segment",
            return_value=(self.masks, {"role": "reference"}),
        ):
            with self.assertRaisesRegex(ValueError, "content_weights"):
                self.evaluator(config).analyze(
                    self.request,
                    times_s=self.times,
                    reference_video=video,
                    prediction_video=video,
                )

    def test_trace_exception_origin_is_reference_or_prediction_specific(self) -> None:
        """Would fail if invalid trace coverage crosses the reference/prediction boundary."""
        reference = sampled_video(self.frames)
        prediction_frames = [frame.copy() for frame in self.frames]
        prediction_frames[0][0, 0] = (1, 2, 3)
        prediction = sampled_video(prediction_frames)
        trace_defect = [self.masks[0], *[np.zeros((96, 96), np.uint8) for _ in self.masks[1:]]]
        with patch.object(
            Sam2VideoSegmenter,
            "segment",
            side_effect=[
                (trace_defect, {"role": "reference"}),
                (self.masks, {"role": "prediction"}),
            ],
        ):
            with self.assertRaises(ReferenceAnalysisError) as caught:
                self.evaluator().analyze(
                    self.request,
                    times_s=self.times,
                    reference_video=reference,
                    prediction_video=prediction,
                )
        self.assertEqual("reference_spring_trace_invalid", caught.exception.code)

        with patch.object(
            Sam2VideoSegmenter,
            "segment",
            side_effect=[
                (self.masks, {"role": "reference"}),
                (trace_defect, {"role": "prediction"}),
            ],
        ):
            with self.assertRaises(SceneAnalysisError) as caught:
                self.evaluator().analyze(
                    self.request,
                    times_s=self.times,
                    reference_video=reference,
                    prediction_video=prediction,
                )
        self.assertNotIsInstance(caught.exception, ReferenceAnalysisError)
        self.assertEqual("prediction_spring_trace_invalid", caught.exception.code)

    def test_identity_rejection_is_evaluated_zero_with_no_csti_match(self) -> None:
        """Would fail if subject/topology can bypass the frozen identity gate."""
        write_video(self.root / "reference.mp4", self.frames)
        prediction_frames = [frame.copy() for frame in self.frames]
        for frame in prediction_frames:
            frame[0:2, 0:2] = (10, 20, 30)
        prediction_path = self.root / "prediction.mp4"
        write_video(prediction_path, prediction_frames)
        self.request.prediction["video_path"] = str(prediction_path)  # type: ignore[index]
        reference_masks = resize_mask_tube(self.masks, width=480, height=480)
        displaced = resize_mask_tube(
            [circle_mask(70, int(round(mask_y(mask)))) for mask in self.masks],
            width=480,
            height=480,
        )
        with patch.object(
            Sam2VideoSegmenter,
            "segment",
            side_effect=[
                (reference_masks, {"role": "reference"}),
                (displaced, {"role": "prediction"}),
            ],
        ):
            result = self.evaluator().evaluate(self.request)

        self.assertEqual("evaluated", result.status)
        self.assertEqual(0.0, result.score)
        self.assertEqual("prediction_spring_identity_rejected", result.reason_code)
        self.assertEqual(0.0, result.metrics["csti"]["score"])
        self.assertFalse(result.metrics["csti"]["objects"][0]["matched"])
        self.assertEqual(
            [],
            result.metrics["csti"]["objects"][0]["matched_prediction_track_ids"],
        )

    def test_prediction_media_defect_is_evaluated_zero_for_robust_spring(self) -> None:
        """Would fail if corrupt prediction media bypasses the zero policy."""
        write_video(self.root / "reference.mp4", self.frames)
        corrupt = self.root / "corrupt-prediction.mp4"
        corrupt.write_bytes(b"not a video")
        self.request.prediction["video_path"] = str(corrupt)  # type: ignore[index]

        result = self.evaluator().evaluate(self.request)

        self.assertEqual("evaluated", result.status)
        self.assertEqual(0.0, result.score)
        self.assertEqual(0.0, result.metrics["csti"]["score"])
        self.assertFalse(result.metrics["csti"]["objects"][0]["matched"])
        self.assertEqual(
            [],
            result.metrics["csti"]["objects"][0][
                "matched_prediction_track_ids"
            ],
        )
        self.assertTrue(result.reason_code.startswith("prediction_"))

    def test_reference_media_defect_remains_unavailable(self) -> None:
        """Would fail if a broken benchmark reference is charged to a model."""
        corrupt = self.root / "corrupt-reference.mp4"
        corrupt.write_bytes(b"not a video")
        self.case["assets"]["reference_video"] = corrupt.name  # type: ignore[index]
        prediction = self.root / "prediction.mp4"
        write_video(prediction, self.frames)
        self.request.prediction["video_path"] = str(prediction)  # type: ignore[index]

        result = self.evaluator().evaluate(self.request)

        self.assertEqual("unavailable", result.status)
        self.assertIsNone(result.score)
        self.assertTrue(result.reason_code.startswith("reference_"))

    def test_nonrobust_prediction_media_defect_preserves_protocol_error(self) -> None:
        """Would fail if the spring policy silently changes older evaluators."""
        write_video(self.root / "reference.mp4", self.frames)
        corrupt = self.root / "corrupt-prediction.mp4"
        corrupt.write_bytes(b"not a video")
        self.request.prediction["video_path"] = str(corrupt)  # type: ignore[index]
        config = deepcopy(EVALUATOR_CONFIG)
        config.pop("evaluator_contract")

        result = self.evaluator(config).evaluate(self.request)

        self.assertEqual("protocol_error", result.status)
        self.assertIsNone(result.score)

    def test_artifact_failure_is_audited_without_changing_the_score(self) -> None:
        """Would fail if base artifact setup can alter a successful result."""
        reference_path = self.root / "reference.mp4"
        write_video(reference_path, self.frames)
        self.request.prediction["video_path"] = str(reference_path)  # type: ignore[index]
        blocked = self.root / "blocked-artifact-directory"
        blocked.write_text("not a directory", encoding="utf-8")
        self.request = replace(self.request, artifact_dir=blocked)
        evaluation_masks = evaluation_sinusoidal_masks(
            self.times, width=480, height=480
        )
        with patch.object(
            Sam2VideoSegmenter,
            "segment",
            return_value=(
                evaluation_masks,
                {"backend": "deterministic_test_sam2"},
            ),
        ):
            result = self.evaluator().evaluate(self.request)

        self.assertEqual("evaluated", result.status)
        self.assertEqual(1.0, result.score)
        self.assertEqual({}, result.artifacts)
        self.assertTrue(result.quality["artifact_failures"])
        self.assertTrue(result.provenance["artifact_failures"])

    def test_degraded_artifact_failure_preserves_zero_and_is_audited(self) -> None:
        """Would fail if degraded-curve I/O escapes or changes the zero."""
        reference_path = self.root / "reference.mp4"
        write_video(reference_path, self.frames)
        self.request.prediction.update(  # type: ignore[union-attr]
            {
                "video_path": str(reference_path),
                "media_contract": "invalid-contract",
            }
        )
        blocked = self.root / "blocked-degraded-artifacts"
        blocked.write_text("not a directory", encoding="utf-8")
        self.request = replace(self.request, artifact_dir=blocked)

        result = self.evaluator().evaluate(self.request)

        self.assertEqual("evaluated", result.status)
        self.assertEqual(0.0, result.score)
        self.assertEqual("media_contract_invalid", result.reason_code)
        self.assertTrue(result.quality["artifact_failures"])
        self.assertTrue(result.provenance["artifact_failures"])


def mask_y(mask: np.ndarray) -> float:
    ys, _ = np.where(mask > 0)
    return float(ys.mean())


class VerticalSpringObservationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.anchor = frozen_circle_anchor()

    def call_with_config(self, family: str, config: dict[str, object]) -> None:
        """Exercise config validation through each public observation boundary."""
        if family == "mask_quality":
            validate_mask_tube(
                [self.anchor.mask],
                availability=[True],
                anchor=self.anchor,
                config=config,
            )
            return
        if family == "identity":
            validate_prediction_identity(
                anchor_mask=self.anchor.mask,
                prediction_mask=self.anchor.mask,
                config=config,
            )
            return
        if family == "topology":
            validate_mask = self.anchor.mask
            observe_spring_topology(
                [spring_frame(validate_mask, attached=True)],
                [validate_mask],
                availability=[True],
                config=config,
            )
            return
        raise AssertionError(f"unknown test config family: {family}")

    def assert_irreversibly_readonly(self, values: np.ndarray) -> None:
        """Require immutable backing storage, not only a cleared array flag."""
        with self.assertRaises(ValueError):
            values.flags.writeable = True
        with self.assertRaises(ValueError):
            values.flat[0] = values.flat[0]

    def test_prompt_expands_around_mask_and_has_one_positive_centroid(self) -> None:
        """Would fail if the prompt box, expansion, or centroid point regresses."""
        prompt = prompt_from_anchor(self.anchor)
        ys, xs = np.where(self.anchor.mask > 0)
        x0, y0, x1, y1 = prompt.box_xyxy.tolist()

        self.assertEqual(0, prompt.frame_index)
        self.assertLess(x0, float(xs.min()))
        self.assertLess(y0, float(ys.min()))
        self.assertGreater(x1, float(xs.max()))
        self.assertGreater(y1, float(ys.max()))
        self.assertGreaterEqual(x0, 0.0)
        self.assertGreaterEqual(y0, 0.0)
        self.assertLessEqual(x1, 95.0)
        self.assertLessEqual(y1, 95.0)
        np.testing.assert_allclose(prompt.points_xy, [[48.0, 72.0]])
        np.testing.assert_array_equal(prompt.point_labels, [1])
        self.assertFalse(prompt.box_xyxy.flags.writeable)
        self.assertFalse(prompt.points_xy.flags.writeable)
        self.assertFalse(prompt.point_labels.flags.writeable)

    def test_identity_accepts_exact_anchor_with_auditable_thresholds(self) -> None:
        """Would fail if exact identity or threshold provenance is omitted."""
        decision = validate_prediction_identity(
            anchor_mask=self.anchor.mask,
            prediction_mask=self.anchor.mask.copy(),
            config=IDENTITY,
        )
        self.assertTrue(decision["accepted"])
        self.assertGreater(decision["anchor_iou"], 0.99)
        self.assertAlmostEqual(0.0, decision["centroid_distance_radii"])
        self.assertAlmostEqual(1.0, decision["area_ratio"])
        self.assertEqual(IDENTITY, dict(decision["thresholds"]))
        for key in ("anchor_iou", "centroid_distance_radii", "area_ratio"):
            self.assertTrue(math.isfinite(decision[key]))
        with self.assertRaises(TypeError):
            decision["accepted"] = False

    def test_identity_rejects_empty_prediction(self) -> None:
        """Would fail if an empty prediction bypasses the identity gate."""
        decision = validate_prediction_identity(
            anchor_mask=self.anchor.mask,
            prediction_mask=np.zeros_like(self.anchor.mask),
            config=IDENTITY,
        )
        self.assertFalse(decision["accepted"])
        self.assertEqual(0.0, decision["anchor_iou"])
        self.assertEqual(0.0, decision["area_ratio"])
        self.assertTrue(math.isfinite(decision["centroid_distance_radii"]))

    def test_identity_rejects_displaced_prediction(self) -> None:
        """Would fail if centroid displacement and overlap checks are removed."""
        displaced = circle_mask(68, 72)
        decision = validate_prediction_identity(
            anchor_mask=self.anchor.mask,
            prediction_mask=displaced,
            config=IDENTITY,
        )
        self.assertFalse(decision["accepted"])
        self.assertEqual(0.0, decision["anchor_iou"])
        self.assertGreater(
            decision["centroid_distance_radii"],
            IDENTITY["maximum_centroid_distance_radii"],
        )

    def test_identity_rejects_extreme_area_prediction(self) -> None:
        """Would fail if the frozen-anchor area-ratio gate is removed."""
        bloated = circle_mask(48, 72, radius=20)
        decision = validate_prediction_identity(
            anchor_mask=self.anchor.mask,
            prediction_mask=bloated,
            config=IDENTITY,
        )
        self.assertFalse(decision["accepted"])
        self.assertGreater(
            decision["area_ratio"], IDENTITY["maximum_anchor_area_ratio"]
        )

    def test_identity_rejects_complex_object_and_string_masks(self) -> None:
        """Would fail if non-real mask dtypes can satisfy identity geometry."""
        for prediction in (
            self.anchor.mask.astype(np.complex64),
            self.anchor.mask.astype(object),
            self.anchor.mask.astype("<U1"),
        ):
            with self.subTest(dtype=str(prediction.dtype)):
                decision = validate_prediction_identity(
                    anchor_mask=self.anchor.mask,
                    prediction_mask=prediction,
                    config=IDENTITY,
                )
                self.assertFalse(decision["accepted"])
                self.assertEqual(0.0, decision["anchor_iou"])
                self.assertEqual(0.0, decision["area_ratio"])

    def test_identity_mask_tube_blanks_complex_object_and_string_masks(
        self,
    ) -> None:
        """Would fail if non-real dtypes survive binary tube normalization."""
        candidates = (
            self.anchor.mask.astype(np.complex64),
            self.anchor.mask.astype(object),
            self.anchor.mask.astype("<U1"),
        )
        masks = validate_mask_tube(
            candidates,
            availability=[True] * len(candidates),
            anchor=self.anchor,
            config=MASK_QUALITY,
        )
        self.assertEqual(
            [0, 0, 0], [int(np.count_nonzero(mask)) for mask in masks]
        )

    def test_identity_mask_tube_normalizes_and_blanks_unavailable_or_bad_masks(
        self,
    ) -> None:
        """Would fail if nonbinary, unavailable, wrong-canvas, or NaN masks leak."""
        nonbinary = self.anchor.mask.astype(np.float32) * 7.5
        wrong_canvas = np.ones((48, 48), dtype=np.uint8)
        nonfinite = self.anchor.mask.astype(np.float32)
        nonfinite[0, 0] = np.nan
        masks = validate_mask_tube(
            [
                nonbinary,
                self.anchor.mask.astype(bool),
                self.anchor.mask,
                wrong_canvas,
                nonfinite,
            ],
            availability=[True, True, False, True, True],
            anchor=self.anchor,
            config=MASK_QUALITY,
        )

        self.assertIsInstance(masks, tuple)
        self.assertEqual(5, len(masks))
        self.assertEqual({0, 255}, set(np.unique(masks[0]).tolist()))
        np.testing.assert_array_equal(masks[0] > 0, self.anchor.mask > 0)
        np.testing.assert_array_equal(masks[1] > 0, self.anchor.mask > 0)
        for mask in masks[2:]:
            self.assertEqual(0, np.count_nonzero(mask))
        for mask in masks:
            self.assertEqual(np.uint8, mask.dtype)
            self.assertEqual((96, 96), mask.shape)
            self.assertFalse(mask.flags.writeable)

    def test_identity_mask_tube_blanks_area_failures_and_bad_config(
        self,
    ) -> None:
        """Would fail if pixel/frame/anchor area gates or config checks fail open."""
        tiny = np.zeros((96, 96), dtype=np.uint8)
        tiny[10:12, 10:12] = 1
        bloated = circle_mask(48, 48, radius=30)
        masks = validate_mask_tube(
            [tiny, bloated],
            availability=[True, True],
            anchor=self.anchor,
            config=MASK_QUALITY,
        )
        self.assertEqual([0, 0], [int(np.count_nonzero(mask)) for mask in masks])

        invalid = dict(MASK_QUALITY, maximum_mask_area_ratio=float("nan"))
        with self.assertRaises(ValueError):
            validate_mask_tube(
                [self.anchor.mask],
                availability=[True],
                anchor=self.anchor,
                config=invalid,
            )

    def test_configs_reject_unknown_keys_even_when_unknown_value_is_nan(self) -> None:
        """Would fail if typo thresholds are silently ignored by any validator."""
        for family, base in (
            ("mask_quality", MASK_QUALITY),
            ("identity", IDENTITY),
            ("topology", TOPOLOGY),
        ):
            with self.subTest(family=family):
                invalid = dict(base, typo_threshold=float("nan"))
                with self.assertRaises(ValueError):
                    self.call_with_config(family, invalid)

    def test_configs_reject_omitted_required_keys(self) -> None:
        """Would fail if a validator invents a default for an omitted threshold."""
        for family, base in (
            ("mask_quality", MASK_QUALITY),
            ("identity", IDENTITY),
            ("topology", TOPOLOGY),
        ):
            with self.subTest(family=family):
                invalid = dict(base)
                invalid.pop(next(iter(base)))
                with self.assertRaises(ValueError):
                    self.call_with_config(family, invalid)

    def test_configs_reject_bool_string_and_complex_scalars(self) -> None:
        """Would fail if float coercion admits values outside real scalars."""
        for family, base in (
            ("mask_quality", MASK_QUALITY),
            ("identity", IDENTITY),
            ("topology", TOPOLOGY),
        ):
            key = next(iter(base))
            for value in (True, "1.0", 1.0 + 0.0j):
                with self.subTest(family=family, value=repr(value)):
                    invalid = dict(base)
                    invalid[key] = value
                    with self.assertRaises(ValueError):
                        self.call_with_config(family, invalid)

    def test_topology_uses_each_current_ball_corridor(self) -> None:
        """Would fail if topology localizes from a frozen or future ball position."""
        first = circle_mask(48, 72)
        second = circle_mask(68, 66)
        result = observe_spring_topology(
            [spring_frame(first, attached=True), spring_frame(second, attached=True)],
            [first, second],
            availability=[True, True],
            config=TOPOLOGY,
        )
        self.assertTrue(np.all(result.valid))
        self.assertTrue(np.all(result.row_coverage > 0.80))
        np.testing.assert_array_equal(result.endpoint_support, [1.0, 1.0])
        self.assertGreater(result.score, 0.90)

    def test_topology_ignores_background_ruler_and_border_when_spring_removed(
        self,
    ) -> None:
        """Would fail if unrelated full-frame edges can earn topology credit."""
        mask = circle_mask(48, 72)
        attached = observe_spring_topology(
            [spring_frame(mask, attached=True)],
            [mask],
            availability=[True],
            config=TOPOLOGY,
        )
        removed = observe_spring_topology(
            [spring_frame(mask, attached=False)],
            [mask],
            availability=[True],
            config=TOPOLOGY,
        )

        self.assertGreater(attached.score, 0.90)
        self.assertLess(removed.row_coverage[0], 0.05)
        self.assertEqual(0.0, removed.endpoint_support[0])
        self.assertLess(removed.score, 0.05)

    def test_topology_rejects_near_top_ball_with_only_canvas_border(self) -> None:
        """Would fail if a one-row top border can masquerade as a spring."""
        near_top = circle_mask(48, 4, radius=3)
        result = observe_spring_topology(
            [spring_frame(near_top, attached=False)],
            [near_top],
            availability=[True],
            config=TOPOLOGY,
        )
        np.testing.assert_array_equal(result.valid, [False])
        np.testing.assert_array_equal(result.row_coverage, [0.0])
        np.testing.assert_array_equal(result.endpoint_support, [0.0])
        self.assertEqual(0.0, result.score)

    def test_topology_rejects_degenerate_short_attached_corridor(self) -> None:
        """Would fail if a tiny connected segment receives full-span credit."""
        short = circle_mask(48, 10, radius=6)
        result = observe_spring_topology(
            [spring_frame(short, attached=True)],
            [short],
            availability=[True],
            config=TOPOLOGY,
        )
        np.testing.assert_array_equal(result.valid, [False])
        self.assertEqual(0.0, result.score)

    def test_topology_rejects_disconnected_in_corridor_texture(self) -> None:
        """Would fail if unrelated row hits need not form one endpoint path."""
        mask = circle_mask(48, 72)
        frame = np.zeros((96, 96, 3), dtype=np.uint8)
        for y in range(5, 66, 6):
            cv2.line(frame, (40, y), (56, y), (255, 255, 255), 1)
        frame[mask > 0] = (160, 160, 160)
        result = observe_spring_topology(
            [frame],
            [mask],
            availability=[True],
            config=TOPOLOGY,
        )
        np.testing.assert_array_equal(result.valid, [True])
        np.testing.assert_array_equal(result.row_coverage, [0.0])
        np.testing.assert_array_equal(result.endpoint_support, [0.0])
        self.assertEqual(0.0, result.score)

    def test_topology_fails_closed_per_frame_and_returns_immutable_finite_arrays(
        self,
    ) -> None:
        """Would fail if bad canvases, unavailable frames, or mutable outputs leak."""
        mask = circle_mask(48, 72)
        wrong_frame = np.zeros((48, 48, 3), dtype=np.uint8)
        result = observe_spring_topology(
            [wrong_frame, spring_frame(mask, attached=True)],
            [mask, mask],
            availability=[True, False],
            config=TOPOLOGY,
        )
        np.testing.assert_array_equal(result.valid, [False, False])
        np.testing.assert_array_equal(result.row_coverage, [0.0, 0.0])
        np.testing.assert_array_equal(result.endpoint_support, [0.0, 0.0])
        self.assertEqual(0.0, result.score)
        self.assertTrue(np.all(np.isfinite(result.row_coverage)))
        self.assertTrue(np.all(np.isfinite(result.endpoint_support)))
        self.assertTrue(math.isfinite(result.score))
        self.assertFalse(result.row_coverage.flags.writeable)
        self.assertFalse(result.endpoint_support.flags.writeable)
        self.assertFalse(result.valid.flags.writeable)

        invalid = dict(TOPOLOGY, endpoint_height_radius_ratio=float("inf"))
        with self.assertRaises(ValueError):
            observe_spring_topology(
                [spring_frame(mask, attached=True)],
                [mask],
                availability=[True],
                config=invalid,
            )

    def test_published_arrays_cannot_be_made_writeable_or_mutated(self) -> None:
        """Would fail if readonly output is only a reversible NumPy flag."""
        prompt = prompt_from_anchor(self.anchor)
        tube = validate_mask_tube(
            [self.anchor.mask],
            availability=[True],
            anchor=self.anchor,
            config=MASK_QUALITY,
        )
        topology = observe_spring_topology(
            [spring_frame(self.anchor.mask, attached=True)],
            [self.anchor.mask],
            availability=[True],
            config=TOPOLOGY,
        )
        arrays = {
            "prompt_box": prompt.box_xyxy,
            "prompt_points": prompt.points_xy,
            "prompt_labels": prompt.point_labels,
            "tube_mask": tube[0],
            "topology_coverage": topology.row_coverage,
            "topology_endpoint": topology.endpoint_support,
            "topology_valid": topology.valid,
        }
        for name, values in arrays.items():
            with self.subTest(name=name):
                self.assert_irreversibly_readonly(values)

    def test_published_masks_do_not_alias_mutable_inputs(self) -> None:
        """Would fail if caller mutation can alter a published tube mask."""
        candidate = self.anchor.mask.copy()
        tube = validate_mask_tube(
            [candidate],
            availability=[True],
            anchor=self.anchor,
            config=MASK_QUALITY,
        )
        candidate[:] = 0
        self.assertEqual(113, np.count_nonzero(tube[0]))


if __name__ == "__main__":
    unittest.main()
