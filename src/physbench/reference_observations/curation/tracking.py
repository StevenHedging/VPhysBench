from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from ...evaluation.common.masks.sam2 import MaskPrompt
from .anchors import AnchorCandidate
from .overrides import CorrectionPrompt, LifecycleOverride


@dataclass(frozen=True)
class TrackingCandidate:
    masks_by_object: dict[str, np.ndarray]
    states_by_object: dict[str, np.ndarray]
    seed_frame_by_observation: np.ndarray
    propagation_metadata: tuple[dict[str, Any], ...]


def _candidate_prompt(anchor: AnchorCandidate, frame_index: int) -> MaskPrompt:
    x, y = anchor.centroid_xy
    return MaskPrompt(
        frame_index=frame_index,
        box_xyxy=np.asarray(anchor.bbox_xyxy, dtype=np.float32),
        points_xy=np.asarray([[x, y]], dtype=np.float32),
        point_labels=np.asarray([1], dtype=np.int32),
        metadata={"object_id": anchor.object_id, "source": "independent_anchor"},
    )


def _correction_prompt(prompt: CorrectionPrompt) -> MaskPrompt:
    return MaskPrompt(
        frame_index=prompt.frame_index,
        box_xyxy=np.asarray(prompt.box_xyxy, dtype=np.float32),
        points_xy=np.asarray(prompt.points_xy, dtype=np.float32),
        point_labels=np.asarray(prompt.point_labels, dtype=np.int32),
        metadata={"object_id": prompt.object_id, "source": "reviewed_correction"},
    )


def _mask_prompt(object_id: str, frame_index: int, mask: np.ndarray) -> MaskPrompt:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError(f"cannot reseed empty mask for {object_id} at {frame_index}")
    return MaskPrompt(
        frame_index=frame_index,
        box_xyxy=np.asarray([xs.min(), ys.min(), xs.max(), ys.max()], np.float32),
        points_xy=np.asarray([[xs.mean(), ys.mean()]], np.float32),
        point_labels=np.asarray([1], np.int32),
        metadata={"object_id": object_id, "source": "propagated_reseed"},
    )


class CuratedSam2Tracker:
    def __init__(self, segmenter: Any):
        self.segmenter = segmenter

    def track(
        self,
        frames: Sequence[np.ndarray],
        *,
        anchors: Sequence[AnchorCandidate],
        corrections: Sequence[CorrectionPrompt],
        lifecycle: Sequence[LifecycleOverride],
    ) -> TrackingCandidate:
        if not frames or not anchors:
            raise ValueError("tracking requires frames and independent anchors")
        object_ids = tuple(anchor.object_id for anchor in anchors)
        if len(set(object_ids)) != len(object_ids):
            raise ValueError("tracking anchors must have unique object ids")
        unknown = {item.object_id for item in (*corrections, *lifecycle)} - set(object_ids)
        if unknown:
            raise ValueError(f"tracking overrides reference unknown objects: {sorted(unknown)}")
        frame_count = len(frames)
        corrections_by_seed = {
            seed: {item.object_id: item for item in corrections if item.frame_index == seed}
            for seed in sorted({item.frame_index for item in corrections})
        }
        if any(seed >= frame_count for seed in corrections_by_seed):
            raise ValueError("correction prompt frame is outside the video")

        seed_tubes: dict[int, dict[str, np.ndarray]] = {}
        metadata: list[dict[str, Any]] = []
        initial_prompts = [_candidate_prompt(anchor, 0) for anchor in anchors]
        tubes, details = self.segmenter.segment_instances(
            list(frames), prompts=initial_prompts, exclusive_masks=True
        )
        seed_tubes[0] = {
            object_id: np.asarray(tube, dtype=np.uint8) > 0
            for object_id, tube in zip(object_ids, tubes, strict=True)
        }
        metadata.append(dict(details))

        anchor_by_id = {anchor.object_id: anchor for anchor in anchors}
        for seed, reviewed in corrections_by_seed.items():
            prompts = []
            for object_id in object_ids:
                correction = reviewed.get(object_id)
                if correction is not None:
                    prompts.append(_correction_prompt(correction))
                else:
                    earlier = max(existing for existing in seed_tubes if existing < seed)
                    mask = seed_tubes[earlier][object_id][seed]
                    if mask.any():
                        prompts.append(_mask_prompt(object_id, seed, mask))
                    else:
                        prompts.append(_candidate_prompt(anchor_by_id[object_id], seed))
            tubes, details = self.segmenter.segment_instances(
                list(frames), prompts=prompts, exclusive_masks=True
            )
            seed_tubes[seed] = {
                object_id: np.asarray(tube, dtype=np.uint8) > 0
                for object_id, tube in zip(object_ids, tubes, strict=True)
            }
            metadata.append(dict(details))

        seeds = np.asarray(sorted(seed_tubes), dtype=np.int32)
        selected = np.asarray(
            [int(seeds[np.argmin(np.abs(seeds - index))]) for index in range(frame_count)],
            dtype=np.int32,
        )
        masks: dict[str, np.ndarray] = {}
        states: dict[str, np.ndarray] = {}
        for object_id in object_ids:
            tube = np.stack(
                [seed_tubes[int(seed)][object_id][index] for index, seed in enumerate(selected)]
            ).astype(np.uint8)
            state = np.where(tube.reshape(frame_count, -1).any(axis=1), 0, 3).astype(np.int8)
            masks[object_id] = tube
            states[object_id] = state
        for override in lifecycle:
            if override.end_index >= frame_count:
                raise ValueError("lifecycle override range is outside the video")
            selection = slice(override.start_index, override.end_index + 1)
            states[override.object_id][selection] = override.state
            if override.state != 0:
                masks[override.object_id][selection] = 0
        return TrackingCandidate(
            masks_by_object=masks,
            states_by_object=states,
            seed_frame_by_observation=selected,
            propagation_metadata=tuple(metadata),
        )
