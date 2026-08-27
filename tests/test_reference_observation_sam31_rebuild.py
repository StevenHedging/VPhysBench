from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np


def _disk(
    center_xy: tuple[int, int],
    radius: int = 5,
    *,
    shape: tuple[int, int] = (40, 80),
) -> np.ndarray:
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    cx, cy = center_xy
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2


def _tube(
    centers: list[tuple[int, int] | None],
    *,
    radius: int = 5,
    shape: tuple[int, int] = (40, 80),
) -> np.ndarray:
    result = np.zeros((len(centers), *shape), dtype=bool)
    for index, center in enumerate(centers):
        if center is not None:
            result[index] = _disk(center, radius, shape=shape)
    return result


def _track(
    semantic_id: str,
    backend_id: int,
    centers: list[tuple[int, int] | None],
    *,
    radius: int = 5,
    shape: tuple[int, int] = (40, 80),
):
    from physbench.reference_observations.curation.sam31_predictor import Sam31GtTrack

    masks = _tube(centers, radius=radius, shape=shape)
    count = len(centers)
    return Sam31GtTrack(
        semantic_id=semantic_id,
        backend_object_id=backend_id,
        masks=masks,
        boxes_xywh=np.zeros((count, 4), np.float32),
        confidences=np.where(masks.reshape(count, -1).any(axis=1), 0.9, 0.0).astype(
            np.float32
        ),
        prompt="small round object",
        source="fixture",
    )


def _case(sample_count: int = 4):
    objects = {
        "object_1": {
            "mass": {"symbol": "m_1", "value": 0.01},
            "radius": {"symbol": "r_1", "value": 0.005},
            "initial_velocity": {"symbol": "v_1", "value": 0.1},
        },
        "object_2": {
            "mass": {"symbol": "m_2", "value": 0.01},
            "radius": {"symbol": "r_2", "value": 0.005},
            "initial_velocity": {"symbol": "v_2", "value": 0.0},
        },
    }
    return SimpleNamespace(
        case_id="collision_fixture",
        scene_id="collision_1d",
        physics={"objects": objects},
        caption="m_1 r_1 v_1 m_2 r_2 v_2",
        appearance={"ball_materials": ["steel", "steel"]},
        timeline={
            "sampling_rate_hz": 24.0,
            "samples": [
                {
                    "observation_index": index,
                    "source_frame_index": 10 * index,
                    "physical_time_seconds": index / 24.0,
                }
                for index in range(sample_count)
            ],
        },
        entities=property(lambda _: (_ for _ in ()).throw(AssertionError("old GT read"))),
    )


class _Predictor:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[tuple[tuple[int, int], tuple[str, ...]]] = []
        self.discovered_tracks = []
        self.box_seeds = ()

    def discover(self, frames, *, prompts):
        self.calls.append((frames[0].shape[:2], tuple(prompts)))
        if not self.responses:
            return ()
        result = tuple(self.responses.pop(0))
        self.discovered_tracks.extend(result)
        return result

    def track_boxes(self, frames, seeds, *, initial_iou_threshold):
        from physbench.reference_observations.curation.sam31_predictor import Sam31GtTrack

        self.box_seeds = tuple(seeds)
        result = {}
        for seed in seeds:
            scored = []
            for track in self.discovered_tracks:
                frame_zero = np.zeros(frames[0].shape[:2], dtype=bool)
                frame_zero[: track.masks.shape[1], : track.masks.shape[2]] = track.masks[0]
                intersection = np.logical_and(frame_zero, seed.reference_mask).sum()
                union = np.logical_or(frame_zero, seed.reference_mask).sum()
                scored.append((intersection / union if union else 0.0, track))
            source = max(scored, key=lambda value: value[0])[1]
            masks = np.zeros((len(frames), *frames[0].shape[:2]), dtype=bool)
            masks[:, : source.masks.shape[1], : source.masks.shape[2]] = source.masks
            result[seed.semantic_id] = Sam31GtTrack(
                semantic_id=seed.semantic_id,
                backend_object_id=source.backend_object_id,
                masks=masks,
                boxes_xywh=source.boxes_xywh.copy(),
                confidences=source.confidences.copy(),
                prompt=seed.text,
                source="fixture_text_box",
            )
        return result

    def describe(self):
        return {"backend": "fixture"}


class Sam31GtRebuildTests(unittest.TestCase):
    @staticmethod
    def _api():
        from physbench.reference_observations.curation import sam31_rebuild

        return sam31_rebuild

    def _frames(self, count: int = 4, shape: tuple[int, int] = (40, 80)):
        return tuple(np.zeros((*shape, 3), np.uint8) for _ in range(count))

    def test_rebuild_preserves_timeline_without_reading_old_gt_pixels(self) -> None:
        api = self._api()
        case = _case()
        predictor = _Predictor(
            [
                (
                    _track("left", 1, [(15, 25), (17, 25), (19, 25), (21, 25)]),
                    _track("right", 2, [(55, 25), (55, 25), (55, 25), (55, 25)]),
                )
            ]
        )

        result = api.rebuild_collision_case(
            case,
            frames=self._frames(),
            predictor=predictor,
            config=api.Sam31GtConfig(),
        )

        self.assertEqual((0, 10, 20, 30), result.source_frame_indices)
        self.assertEqual((4, 40, 80), result.masks_by_object["object_1"].shape)
        self.assertEqual(("object_1", "object_2"), tuple(result.masks_by_object))
        self.assertEqual(
            ("object_1", "object_2"),
            tuple(seed.semantic_id for seed in predictor.box_seeds),
        )

    def test_rebuild_uses_locked_box_tracks_instead_of_text_track_ids(self) -> None:
        api = self._api()
        discovery_left = _track("sam-text-91", 91, [(15, 25)] * 4)
        discovery_right = _track("sam-text-7", 7, [(55, 25)] * 4)
        predictor = _Predictor([(discovery_left, discovery_right)])

        result = api.rebuild_collision_case(
            _case(),
            frames=self._frames(),
            predictor=predictor,
            config=api.Sam31GtConfig(),
        )

        self.assertTrue(result.accepted)
        self.assertEqual(("object_1", "object_2"), tuple(seed.semantic_id for seed in predictor.box_seeds))

    def test_rebuild_uses_fallback_prompt_after_primary_shortage(self) -> None:
        api = self._api()
        predictor = _Predictor(
            [
                (_track("only", 1, [(15, 25)] * 4),),
                (
                    _track("left", 1, [(15, 25)] * 4),
                    _track("right", 2, [(55, 25)] * 4),
                ),
            ]
        )

        result = api.rebuild_collision_case(
            _case(),
            frames=self._frames(),
            predictor=predictor,
            config=api.Sam31GtConfig(fallback_prompts=("ball",)),
        )

        self.assertTrue(result.accepted)
        self.assertEqual(("small round object",), predictor.calls[0][1])
        self.assertEqual(("ball",), predictor.calls[1][1])

    def test_rebuild_recovers_boundary_candidate_from_crop(self) -> None:
        api = self._api()
        full_right = _track("right", 2, [(55, 25)] * 4)
        cropped_left = _track(
            "cropped-left",
            7,
            [(2, 25)] * 4,
            shape=(40, 48),
        )
        predictor = _Predictor(
            [
                (full_right,),
                (full_right,),
                (cropped_left,),
                (),
            ]
        )

        result = api.rebuild_collision_case(
            _case(),
            frames=self._frames(),
            predictor=predictor,
            config=api.Sam31GtConfig(
                fallback_prompts=("ball",),
                border_crop_fraction=0.6,
            ),
        )

        self.assertTrue(result.accepted)
        self.assertEqual(0, np.nonzero(result.masks_by_object["object_1"][0])[1].min())
        self.assertIn((40, 48), [shape for shape, _ in predictor.calls])

    def test_trailing_disappearance_at_boundary_is_out_of_frame(self) -> None:
        api = self._api()
        left = _track("left", 1, [(12, 25), (6, 25), (1, 25), None])
        right = _track("right", 2, [(55, 25)] * 4)

        result = api.rebuild_collision_case(
            _case(),
            frames=self._frames(),
            predictor=_Predictor([(left, right)]),
            config=api.Sam31GtConfig(),
        )

        self.assertTrue(result.accepted)
        self.assertEqual([0, 0, 0, 2], result.states_by_object["object_1"].tolist())

    def test_internal_track_drop_is_rejected_as_unresolved(self) -> None:
        api = self._api()
        left = _track("left", 1, [(15, 25), None, (19, 25), (21, 25)])
        right = _track("right", 2, [(55, 25)] * 4)

        result = api.rebuild_collision_case(
            _case(),
            frames=self._frames(),
            predictor=_Predictor([(left, right)]),
            config=api.Sam31GtConfig(),
        )

        self.assertFalse(result.accepted)
        self.assertIn("internal_track_gap", [finding.code for finding in result.findings])
        self.assertEqual(3, int(result.states_by_object["object_1"][1]))

    def test_identity_order_inversion_is_rejected(self) -> None:
        api = self._api()
        left = _track("left", 1, [(20, 25), (35, 25), (50, 25), (60, 25)])
        right = _track("right", 2, [(60, 25), (50, 25), (35, 25), (20, 25)])

        result = api.rebuild_collision_case(
            _case(),
            frames=self._frames(),
            predictor=_Predictor([(left, right)]),
            config=api.Sam31GtConfig(),
        )

        self.assertFalse(result.accepted)
        self.assertIn("identity_order_inversion", [item.code for item in result.findings])

    def test_near_duplicate_tubes_are_rejected_but_contact_overlap_is_allowed(self) -> None:
        api = self._api()
        left = _track("left", 1, [(20, 25), (27, 25), (34, 25), (38, 25)], radius=7)
        contact = _track("right", 2, [(55, 25), (48, 25), (42, 25), (46, 25)], radius=7)

        allowed = api.rebuild_collision_case(
            _case(),
            frames=self._frames(),
            predictor=_Predictor([(left, contact)]),
            config=api.Sam31GtConfig(),
        )
        duplicate = api.rebuild_collision_case(
            _case(),
            frames=self._frames(),
            predictor=_Predictor([(left, _track("duplicate", 2, [(55, 25), (27, 25), (34, 25), (38, 25)], radius=7))]),
            config=api.Sam31GtConfig(),
        )

        self.assertTrue(np.logical_and(left.masks, contact.masks).any())
        self.assertTrue(allowed.accepted)
        self.assertFalse(duplicate.accepted)
        self.assertIn("near_duplicate_tubes", [item.code for item in duplicate.findings])


if __name__ == "__main__":
    unittest.main()
