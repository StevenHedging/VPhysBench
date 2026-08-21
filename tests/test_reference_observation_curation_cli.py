from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from physbench.reference_observations.curation.anchors import AnchorCandidate
from physbench.reference_observations.curation.overrides import CorrectionPrompt


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = ROOT / "scripts" / "reference_observations" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReferenceObservationCurationCliTests(unittest.TestCase):
    def test_gpu_shards_cover_each_case_exactly_once(self) -> None:
        rebuild = _load_script("rebuild_cases.py")
        shards = rebuild.shard_case_ids(["c", "a", "b", "d"], shard_count=3)
        flattened = [item for shard in shards for item in shard]
        self.assertEqual(["a", "b", "c", "d"], sorted(flattened))
        self.assertEqual(4, len(set(flattened)))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "cases.txt"
            path.write_text("case_b\ncase_a\ncase_b\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate Case"):
                rebuild.read_case_id_list(path)

    def test_install_authorization_requires_accepted_matching_candidate(self) -> None:
        rebuild = _load_script("rebuild_cases.py")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = root / "candidate.json"
            candidate.write_text("{}", encoding="utf-8")
            digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
            ledger = root / "ledger.jsonl"
            ledger.write_text(
                json.dumps(
                    {
                        "case_id": "case_a",
                        "anchor_decision": "pass",
                        "tube_decision": "repair",
                        "candidate_digest": digest,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "accepted review decision"):
                rebuild.require_install_authorization(
                    ledger, case_id="case_a", candidate_manifest=candidate
                )

    def test_install_authorization_rejects_stale_candidate_digest(self) -> None:
        rebuild = _load_script("rebuild_cases.py")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = root / "candidate.json"
            candidate.write_text("{}", encoding="utf-8")
            ledger = root / "ledger.jsonl"
            ledger.write_text(
                json.dumps(
                    {
                        "case_id": "case_a",
                        "anchor_decision": "pass",
                        "tube_decision": "pass",
                        "candidate_digest": "0" * 64,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "digest"):
                rebuild.require_install_authorization(
                    ledger, case_id="case_a", candidate_manifest=candidate
                )

    def test_audit_records_are_written_in_release_order(self) -> None:
        audit = _load_script("audit_v14.py")
        self.assertEqual((0, 3, 6, 9), audit.shard_release_indices(10, 0, 3))
        self.assertEqual((1, 4, 7), audit.shard_release_indices(10, 1, 3))
        self.assertEqual((2, 5, 8), audit.shard_release_indices(10, 2, 3))
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "diagnostics.jsonl"
            audit.write_diagnostic_records(
                output,
                [
                    {"case_id": "case_b", "release_index": 1},
                    {"case_id": "case_a", "release_index": 0},
                ],
            )
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(["case_a", "case_b"], [row["case_id"] for row in rows])
            with self.assertRaisesRegex(ValueError, "duplicate release index"):
                audit.merge_diagnostic_records(output, [output, output])

    def test_audit_reports_collision_visual_physics_size_binding(self) -> None:
        audit = _load_script("audit_v14.py")
        physics = {
            "objects": {
                "object_1": {"radius": {"value": 0.0125}},
                "object_2": {"radius": {"value": 0.01}},
            }
        }
        issue = audit.audit_visual_physics_size_binding(
            scene_id="collision_1d",
            physics=physics,
            anchor_areas={"object_1": 900, "object_2": 1400},
        )
        self.assertEqual("visual_physics_size_order_mismatch", issue["code"])
        self.assertIsNone(
            audit.audit_visual_physics_size_binding(
                scene_id="collision_1d",
                physics=physics,
                anchor_areas={"object_1": 1400, "object_2": 900},
            )
        )
        self.assertIsNone(
            audit.audit_visual_physics_size_binding(
                scene_id="pendulum",
                physics=physics,
                anchor_areas={"object_1": 900, "object_2": 1400},
            )
        )
        self.assertIsNone(
            audit.audit_visual_physics_size_binding(
                scene_id="collision_1d",
                physics={
                    "objects": {
                        "object_1": {"radius": {"value": 0.0075}},
                        "object_2": {"radius": {"value": 0.0125}},
                    }
                },
                anchor_areas={"object_1": 1083, "object_2": 1107},
                anchor_bbox_areas={"object_1": 1280, "object_2": 2300},
            )
        )

    def test_semantic_swap_plan_is_complete_and_idempotent(self) -> None:
        remap = _load_script("remap_collision_semantics.py")
        physics_document = {
            "case_id": "case_a",
            "scene_id": "collision_1d",
            "physics": {
                "environment": {},
                "objects": {
                    "object_1": {
                        "initial_velocity": {"symbol": "v_1", "value": 0.0},
                        "mass": {"symbol": "m_1", "value": 0.06477},
                        "radius": {"symbol": "r_1", "value": 0.0125},
                    },
                    "object_2": {
                        "initial_velocity": {"symbol": "v_2", "value": 0.2},
                        "mass": {"symbol": "m_2", "value": 0.014},
                        "radius": {"symbol": "r_2", "value": 0.0075},
                    },
                },
            },
        }
        appearance = {
            "ball_sequence": ["large", "small"],
            "striker_ball_index": 2,
        }

        plan = remap.plan_two_object_collision_swap(
            physics_document=physics_document,
            appearance=appearance,
            bbox_areas={"object_1": 900, "object_2": 2400},
        )

        self.assertIsNotNone(plan)
        self.assertEqual(
            0.0075, plan.physics_document["physics"]["objects"]["object_1"]["radius"]["value"]
        )
        self.assertEqual(["small", "large"], plan.appearance["ball_sequence"])
        self.assertEqual(1, plan.appearance["striker_ball_index"])
        self.assertIn("left ball moves right", plan.caption)
        self.assertIsNone(
            remap.plan_two_object_collision_swap(
                physics_document=plan.physics_document,
                appearance=plan.appearance,
                bbox_areas={"object_1": 900, "object_2": 2400},
            )
        )
    def test_rebuild_decodes_only_timeline_source_frames(self) -> None:
        rebuild = _load_script("rebuild_cases.py")
        with tempfile.TemporaryDirectory() as temporary:
            video = Path(temporary) / "fixture.avi"
            writer = cv2.VideoWriter(
                str(video), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (16, 16)
            )
            self.assertTrue(writer.isOpened())
            for value in (0, 40, 80, 120, 160):
                writer.write(np.full((16, 16, 3), value, np.uint8))
            writer.release()
            frames = rebuild._decode_video(video, source_frame_indices=(0, 2, 4))
            self.assertEqual(3, len(frames))
            self.assertLess(float(frames[0].mean()), float(frames[1].mean()))
            self.assertLess(float(frames[1].mean()), float(frames[2].mean()))

    def test_rebuild_preserves_repeated_timeline_source_frames(self) -> None:
        rebuild = _load_script("rebuild_cases.py")
        with tempfile.TemporaryDirectory() as temporary:
            video = Path(temporary) / "fixture.avi"
            writer = cv2.VideoWriter(
                str(video), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (16, 16)
            )
            self.assertTrue(writer.isOpened())
            for value in (0, 40, 80):
                writer.write(np.full((16, 16, 3), value, np.uint8))
            writer.release()

            frames = rebuild._decode_video(
                video, source_frame_indices=(0, 1, 1, 2, 2)
            )

            self.assertEqual(5, len(frames))
            np.testing.assert_array_equal(frames[1], frames[2])
            np.testing.assert_array_equal(frames[3], frames[4])

    def test_rebuild_rejects_descending_timeline_source_frames(self) -> None:
        rebuild = _load_script("rebuild_cases.py")
        with self.assertRaisesRegex(ValueError, "nondecreasing"):
            rebuild._decode_video(
                Path("unused.mp4"), source_frame_indices=(0, 2, 1)
            )

    def test_finalize_preserves_repeated_timeline_source_frames(self) -> None:
        finalize = _load_script("finalize_cases.py")
        with tempfile.TemporaryDirectory() as temporary:
            video = Path(temporary) / "fixture.avi"
            writer = cv2.VideoWriter(
                str(video), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (16, 16)
            )
            self.assertTrue(writer.isOpened())
            for value in (0, 40, 80):
                writer.write(np.full((16, 16, 3), value, np.uint8))
            writer.release()

            case = type(
                "Case",
                (),
                {
                    "reference_video_path": video,
                    "case_id": "fixture",
                    "timeline": {
                        "samples": [
                            {"source_frame_index": index}
                            for index in (0, 1, 1, 2, 2)
                        ]
                    },
                },
            )()
            frames = finalize._decode_samples(case)

            self.assertEqual(5, len(frames))
            np.testing.assert_array_equal(frames[1], frames[2])
            np.testing.assert_array_equal(frames[3], frames[4])

    def test_reviewed_anchor_prompt_replaces_only_named_first_frame_anchor(self) -> None:
        rebuild = _load_script("rebuild_cases.py")
        anchors = (
            AnchorCandidate.from_mask(
                "object_1", np.pad(np.ones((2, 2), np.uint8), ((1, 5), (1, 5)))
            ),
            AnchorCandidate.from_mask(
                "object_2", np.pad(np.ones((2, 2), np.uint8), ((4, 2), (4, 2)))
            ),
        )
        prompt = CorrectionPrompt(
            object_id="object_1",
            frame_index=0,
            box_xyxy=(2.0, 3.0, 5.0, 6.0),
            points_xy=((3.5, 4.5),),
            point_labels=(1,),
        )

        updated = rebuild._apply_anchor_prompts(
            anchors, (prompt,), frame_shape=(8, 8)
        )

        self.assertEqual((2.0, 3.0, 5.0, 6.0), updated[0].bbox_xyxy)
        np.testing.assert_array_equal(updated[1].mask, anchors[1].mask)

    def test_reviewed_anchor_prompt_must_target_frame_zero(self) -> None:
        rebuild = _load_script("rebuild_cases.py")
        anchor = AnchorCandidate.from_mask("object_1", np.ones((8, 8), np.uint8))
        prompt = CorrectionPrompt(
            object_id="object_1",
            frame_index=1,
            box_xyxy=(2.0, 2.0, 4.0, 4.0),
            points_xy=((3.0, 3.0),),
            point_labels=(1,),
        )
        with self.assertRaisesRegex(ValueError, "frame zero"):
            rebuild._apply_anchor_prompts((anchor,), (prompt,), frame_shape=(8, 8))

    def test_splice_keeps_reviewed_canonical_tube_except_selected_samples(self) -> None:
        splice = _load_script("splice_case_entities.py")
        canonical_masks = np.zeros((3, 8, 8), np.uint8)
        canonical_masks[:, 1:3, 1:3] = 1
        canonical_states = np.asarray([0, 1, 0], np.uint8)
        candidate_masks = np.zeros_like(canonical_masks)
        candidate_masks[:, 4:6, 4:6] = 1
        candidate_states = np.asarray([0, 0, 2], np.uint8)

        masks, states = splice.splice_entity_samples(
            canonical_masks,
            canonical_states,
            candidate_masks,
            candidate_states,
            candidate_indices=(0,),
        )

        np.testing.assert_array_equal(masks[0], candidate_masks[0])
        np.testing.assert_array_equal(masks[1:], canonical_masks[1:])
        self.assertEqual([0, 1, 0], states.tolist())

    def test_splice_repairs_unresolved_visible_sample_from_candidate(self) -> None:
        splice = _load_script("splice_case_entities.py")
        canonical = np.zeros((3, 12, 12), np.uint8)
        canonical[[0, 2], 4:8, 4:8] = 1
        candidate = canonical.copy()
        candidate[1, 4:8, 5:9] = 1
        masks, states = splice.repair_unresolved_samples(
            canonical,
            np.asarray([0, 3, 0], np.uint8),
            candidate,
            np.asarray([0, 0, 0], np.uint8),
            occupied_masks=np.zeros_like(canonical),
        )
        np.testing.assert_array_equal(masks[1], candidate[1])
        self.assertEqual([0, 0, 0], states.tolist())

    def test_splice_repairs_failed_occluded_sample_from_round_candidate(self) -> None:
        splice = _load_script("splice_case_entities.py")
        canonical = np.zeros((3, 16, 16), np.uint8)
        canonical[[0, 2], 5:11, 5:11] = 1
        candidate = canonical.copy()
        candidate[1, 5:11, 6:12] = 1
        masks, states = splice.repair_unresolved_samples(
            canonical,
            np.asarray([0, 1, 0], np.uint8),
            candidate,
            np.asarray([0, 0, 0], np.uint8),
            occupied_masks=np.zeros_like(canonical),
        )
        np.testing.assert_array_equal(masks[1], candidate[1])
        self.assertEqual([0, 0, 0], states.tolist())

    def test_splice_full_candidate_authorization_precedes_generic_repair(self) -> None:
        splice = _load_script("splice_case_entities.py")
        canonical = np.zeros((3, 12, 12), np.uint8)
        canonical[:, 4:8, 4:8] = 1
        canonical_states = np.asarray([0, 1, 0], np.uint8)
        candidate = np.zeros_like(canonical)
        candidate[:, 2:10, 2:10] = 1
        candidate_states = np.zeros(3, np.uint8)
        masks, states = splice.select_entity_base(
            canonical,
            canonical_states,
            candidate,
            candidate_states,
            full_candidate=True,
            repair_unresolved=True,
            occupied_masks=np.zeros_like(canonical),
        )
        np.testing.assert_array_equal(masks, candidate)
        self.assertEqual([0, 0, 0], states.tolist())

    def test_splice_rejects_elongated_background_candidate(self) -> None:
        splice = _load_script("splice_case_entities.py")
        canonical = np.zeros((3, 32, 32), np.uint8)
        canonical[[0, 2], 10:18, 10:18] = 1
        candidate = canonical.copy()
        candidate[1, 14:16, 2:30] = 1
        masks, states = splice.repair_unresolved_samples(
            canonical,
            np.asarray([0, 1, 0], np.uint8),
            candidate,
            np.asarray([0, 0, 0], np.uint8),
            occupied_masks=np.zeros_like(canonical),
        )
        self.assertFalse(masks[1].any())
        self.assertEqual(1, states[1])

    def test_splice_interpolates_short_internal_tracking_gap(self) -> None:
        splice = _load_script("splice_case_entities.py")
        masks = np.zeros((3, 20, 20), np.uint8)
        masks[0, 7:13, 4:10] = 1
        masks[2, 7:13, 8:14] = 1
        repaired_masks, repaired_states = splice.interpolate_short_visible_gaps(
            masks,
            np.asarray([0, 1, 0], np.uint8),
            occupied_masks=np.zeros_like(masks),
        )
        self.assertEqual(0, repaired_states[1])
        self.assertEqual(36, int(repaired_masks[1].sum()))
        ys, xs = np.nonzero(repaired_masks[1])
        self.assertAlmostEqual(8.5, float(xs.mean()))

    def test_splice_uses_candidate_scale_when_canonical_anchor_is_too_small(self) -> None:
        splice = _load_script("splice_case_entities.py")
        canonical = np.zeros((3, 16, 16), np.uint8)
        canonical[[0, 2], 7:9, 7:9] = 1
        candidate = np.zeros_like(canonical)
        candidate[:, 4:12, 4:12] = 1
        masks, states = splice.repair_unresolved_samples(
            canonical,
            np.asarray([0, 3, 0], np.uint8),
            candidate,
            np.asarray([0, 0, 0], np.uint8),
            occupied_masks=np.zeros_like(canonical),
        )
        np.testing.assert_array_equal(masks[1], candidate[1])
        self.assertEqual(0, states[1])

    def test_splice_replaces_tiny_observation_zero_from_stable_candidate(self) -> None:
        splice = _load_script("splice_case_entities.py")
        candidate_areas = np.asarray([64, 60, 68], np.int64)
        self.assertTrue(
            splice.should_replace_observation_zero(
                canonical_area=4,
                candidate_area=64,
                candidate_visible_areas=candidate_areas,
            )
        )

    def test_splice_keeps_plausible_observation_zero_over_bad_candidate(self) -> None:
        splice = _load_script("splice_case_entities.py")
        candidate_areas = np.asarray([32, 64, 68], np.int64)
        self.assertFalse(
            splice.should_replace_observation_zero(
                canonical_area=60,
                candidate_area=32,
                candidate_visible_areas=candidate_areas,
            )
        )

    def test_splice_replaces_oversized_observation_zero_from_stable_candidate(self) -> None:
        splice = _load_script("splice_case_entities.py")
        self.assertTrue(
            splice.should_replace_observation_zero(
                canonical_area=500,
                candidate_area=64,
                candidate_visible_areas=np.asarray([60, 64, 68], np.int64),
            )
        )

    def test_splice_does_not_replace_observation_zero_without_review_authorization(self) -> None:
        splice = _load_script("splice_case_entities.py")
        canonical = np.zeros((2, 12, 12), np.uint8)
        canonical[0, 5:7, 5:7] = 1
        candidate = np.zeros_like(canonical)
        candidate[:, 2:10, 2:10] = 1
        masks, states = splice.maybe_replace_observation_zero(
            canonical,
            np.asarray([0, 0], np.uint8),
            candidate,
            np.asarray([0, 0], np.uint8),
            authorized=False,
        )
        np.testing.assert_array_equal(masks, canonical)
        self.assertEqual([0, 0], states.tolist())

    def test_splice_replaces_bad_observation_zero_when_review_authorizes_it(self) -> None:
        splice = _load_script("splice_case_entities.py")
        canonical = np.zeros((2, 12, 12), np.uint8)
        canonical[0, 5:7, 5:7] = 1
        candidate = np.zeros_like(canonical)
        candidate[:, 2:10, 2:10] = 1
        masks, states = splice.maybe_replace_observation_zero(
            canonical,
            np.asarray([0, 0], np.uint8),
            candidate,
            np.asarray([0, 0], np.uint8),
            authorized=True,
        )
        np.testing.assert_array_equal(masks[0], candidate[0])
        self.assertEqual([0, 0], states.tolist())

    def test_splice_applies_reviewed_nonvisible_lifecycle_range(self) -> None:
        splice = _load_script("splice_case_entities.py")
        masks = np.ones((5, 4, 4), np.uint8)
        states = np.zeros(5, np.uint8)
        result_masks, result_states = splice.apply_lifecycle_overrides(
            masks,
            states,
            ({"start_index": 3, "end_index": 4, "state": 2},),
        )
        self.assertTrue(result_masks[:3].all())
        self.assertFalse(result_masks[3:].any())
        self.assertEqual([0, 0, 0, 2, 2], result_states.tolist())

    def test_splice_rejects_visible_lifecycle_without_masks(self) -> None:
        splice = _load_script("splice_case_entities.py")
        with self.assertRaisesRegex(ValueError, "visible lifecycle"):
            splice.apply_lifecycle_overrides(
                np.zeros((2, 4, 4), np.uint8),
                np.zeros(2, np.uint8),
                ({"start_index": 1, "end_index": 1, "state": 0},),
            )

    def test_splice_applies_reviewed_frame_copy_and_circle(self) -> None:
        splice = _load_script("splice_case_entities.py")
        masks = np.zeros((3, 20, 20), np.uint8)
        masks[1, 6:10, 7:11] = 1
        states = np.asarray([1, 0, 1], np.uint8)
        copied_masks, copied_states = splice.apply_frame_overrides(
            masks,
            states,
            (
                {"target_index": 0, "source_index": 1},
                {"target_index": 2, "circle_xy_radius": [18, 10, 4]},
            ),
        )
        np.testing.assert_array_equal(copied_masks[0], masks[1])
        self.assertTrue(copied_masks[2, 10, 18])
        self.assertTrue(copied_masks[2, 10, 19])
        self.assertEqual([0, 0, 0], copied_states.tolist())

    def test_splice_cli_allows_generic_repair_with_case_plan(self) -> None:
        splice = _load_script("splice_case_entities.py")
        arguments = splice._parser().parse_args(
            [
                "--dataset", "dataset.json", "--case-list", "cases.txt",
                "--candidate-root", "candidates", "--output-root", "output",
                "--repair-unresolved", "--plan", "plan.json",
            ]
        )
        self.assertTrue(arguments.repair_unresolved)
        self.assertEqual(Path("plan.json"), arguments.plan)

    def test_splice_applies_reviewed_cross_entity_source_range(self) -> None:
        splice = _load_script("splice_case_entities.py")
        masks = np.zeros((4, 8, 8), np.uint8)
        states = np.ones(4, np.uint8)
        source_masks = np.zeros_like(masks)
        source_masks[:, 3:5, 4:6] = 1
        source_states = np.zeros(4, np.uint8)
        repaired_masks, repaired_states = splice.apply_source_ranges(
            masks,
            states,
            source_masks,
            source_states,
            ({"start_index": 1, "end_index": 2},),
        )
        self.assertFalse(repaired_masks[0].any())
        np.testing.assert_array_equal(repaired_masks[1:3], source_masks[1:3])
        self.assertEqual([1, 0, 0, 1], repaired_states.tolist())

    def test_splice_reconstructs_constant_radius_from_reviewed_source_centers(self) -> None:
        splice = _load_script("splice_case_entities.py")
        masks = np.zeros((3, 20, 20), np.uint8)
        states = np.ones(3, np.uint8)
        centers = np.asarray([[5.0, 10.0], [10.0, 10.0], [18.0, 10.0]])
        repaired_masks, repaired_states = splice.apply_circle_source_ranges(
            masks,
            states,
            centers,
            ({"start_index": 0, "end_index": 2, "radius": 4},),
        )
        self.assertEqual([0, 0, 0], repaired_states.tolist())
        self.assertTrue(repaired_masks[0, 10, 5])
        self.assertTrue(repaired_masks[2, 10, 19])
        self.assertLess(int(repaired_masks[2].sum()), int(repaired_masks[1].sum()))

    def test_splice_reconstructs_reviewed_constant_velocity_circle_range(self) -> None:
        splice = _load_script("splice_case_entities.py")
        masks = np.zeros((3, 20, 20), np.uint8)
        states = np.ones(3, np.uint8)
        repaired_masks, repaired_states = splice.apply_circle_motion_ranges(
            masks,
            states,
            ({
                "start_index": 0,
                "end_index": 2,
                "center_start_xy": [5, 10],
                "velocity_xy": [4, 0],
                "radius": 3,
            },),
        )
        self.assertEqual([0, 0, 0], repaired_states.tolist())
        self.assertTrue(repaired_masks[0, 10, 5])
        self.assertTrue(repaired_masks[1, 10, 9])
        self.assertTrue(repaired_masks[2, 10, 13])

    def test_splice_splits_reviewed_overlap_by_horizontal_object_order(self) -> None:
        splice = _load_script("splice_case_entities.py")
        left = np.zeros((20, 30), np.uint8)
        right = np.zeros_like(left)
        cv2.circle(left, (12, 10), 6, 1, -1)
        cv2.circle(right, (18, 10), 6, 1, -1)
        split_left, split_right = splice.split_ordered_mask_overlap(left, right)
        self.assertEqual(0, int(np.logical_and(split_left, split_right).sum()))
        self.assertTrue(split_left[10, 8])
        self.assertTrue(split_right[10, 22])
        self.assertFalse(split_left[10, 19])
        self.assertFalse(split_right[10, 11])

    def test_splice_relabels_unresolved_boundary_exit_without_hallucination(self) -> None:
        splice = _load_script("splice_case_entities.py")
        canonical = np.zeros((3, 12, 12), np.uint8)
        canonical[0, 4:8, 9:12] = 1
        candidate = np.zeros_like(canonical)
        candidate[1, 4:8, 4:8] = 1
        masks, states = splice.repair_unresolved_samples(
            canonical,
            np.asarray([0, 3, 2], np.uint8),
            candidate,
            np.asarray([0, 0, 2], np.uint8),
            occupied_masks=np.zeros_like(canonical),
        )
        self.assertFalse(masks[1].any())
        self.assertEqual([0, 2, 2], states.tolist())

    def test_splice_marks_unresolved_without_plausible_candidate_occluded(self) -> None:
        splice = _load_script("splice_case_entities.py")
        canonical = np.zeros((3, 12, 12), np.uint8)
        canonical[[0, 2], 4:8, 4:8] = 1
        masks, states = splice.repair_unresolved_samples(
            canonical,
            np.asarray([0, 3, 0], np.uint8),
            np.zeros_like(canonical),
            np.asarray([0, 3, 0], np.uint8),
            occupied_masks=np.zeros_like(canonical),
        )
        self.assertFalse(masks[1].any())
        self.assertEqual([0, 1, 0], states.tolist())

    def test_circle_tracker_prefers_motion_consistent_subject_over_distractor(self) -> None:
        tracker = _load_script("track_round_subject_circles.py")
        selected = tracker.select_circle_candidate(
            ((24.0, 20.0, 7.0), (50.0, 50.0, 9.0), (25.0, 20.0, 20.0)),
            predicted_xy=np.asarray([25.0, 20.0]),
            guide_xy=np.asarray([24.0, 21.0]),
            expected_radius=8.0,
            maximum_distance=20.0,
        )
        self.assertEqual((24.0, 20.0, 7.0), selected)

    def test_pendulum_circle_tracker_detects_tall_support_anchor(self) -> None:
        tracker = _load_script("track_round_subject_circles.py")
        support = np.zeros((120, 100), np.uint8)
        support[10:100, 42:58] = 1
        bob = np.zeros_like(support)
        cv2.circle(bob, (50, 80), 12, 1, -1)

        self.assertTrue(tracker.pendulum_anchor_is_support(support))
        self.assertFalse(tracker.pendulum_anchor_is_support(bob))

    def test_pendulum_circle_tracker_bootstraps_lowest_bob_below_fixture(self) -> None:
        tracker = _load_script("track_round_subject_circles.py")
        selected = tracker.select_pendulum_bob_candidate(
            (
                (605.0, 413.0, 26.0),
                (637.0, 653.0, 44.0),
                (607.0, 807.0, 43.0),
                (652.0, 1071.0, 34.0),
            ),
            expected_radius=43.0,
            frame_height=1920,
        )

        self.assertEqual((652.0, 1071.0, 34.0), selected)

    def test_pendulum_bootstrap_does_not_inherit_visible_fixture_state(self) -> None:
        tracker = _load_script("track_round_subject_circles.py")
        state = tracker.bootstrap_detection_state(
            np.asarray([True, False, True], bool)
        )

        self.assertEqual([0, 3, 0], state.tolist())

    def test_pendulum_bootstrap_interpolates_only_interior_hough_misses(self) -> None:
        tracker = _load_script("track_round_subject_circles.py")
        masks = np.ones((3, 40, 60), np.uint8)
        states = tracker.resolve_bootstrap_subject_lifecycle(
            masks,
            interpolated_guides=np.asarray(
                [[30.0, 20.0], [31.0, 20.0], [58.0, 20.0]], np.float64
            ),
            detected=np.asarray([True, False, False]),
            expected_radius=5.0,
        )

        self.assertEqual([0, 0, 2], states.tolist())
        self.assertTrue(masks[1].any())
        self.assertFalse(masks[2].any())

    def test_collision_hough_tracker_selects_physical_left_to_right_rank(self) -> None:
        tracker = _load_script("track_collision_hough_special.py")
        circles = ((1400.0, 940.0, 34.0), (36.0, 941.0, 35.0), (636.0, 939.0, 33.0))
        self.assertEqual(
            (636.0, 939.0, 33.0),
            tracker.select_ranked_circle(circles, rank_from_left=1),
        )
        with self.assertRaisesRegex(ValueError, "rank"):
            tracker.select_ranked_circle(circles, rank_from_left=3)
        self.assertEqual(
            ((36.0, 941.0, 35.0), (636.0, 939.0, 33.0)),
            tracker.filter_circles_by_x(circles, x_range=(0.0, 1000.0)),
        )

    def test_collision_hough_tracker_prefers_temporal_prediction_over_fixture(self) -> None:
        tracker = _load_script("track_collision_hough_special.py")
        selected = tracker.select_temporal_circle(
            ((1700.0, 941.0, 46.0), (1722.0, 948.0, 48.0)),
            predicted_xy=np.asarray([1720.0, 947.0]),
            expected_radius=47.0,
            maximum_distance=30.0,
        )
        self.assertEqual((1722.0, 948.0, 48.0), selected)
        self.assertIsNone(
            tracker.select_temporal_circle(
                ((1700.0, 941.0, 46.0),),
                predicted_xy=np.asarray([1800.0, 947.0]),
                expected_radius=47.0,
                maximum_distance=30.0,
            )
        )
        np.testing.assert_allclose(
            [12.0, 0.0], tracker.clamp_velocity(np.asarray([18.0, 0.0]), 12.0)
        )
        np.testing.assert_allclose(
            [1637.0, 947.0],
            tracker.enforce_direction(
                np.asarray([1632.0, 947.0]),
                previous_xy=np.asarray([1637.0, 947.0]),
                predicted_xy=np.asarray([1646.0, 947.0]),
                direction=1,
                maximum_backward_jitter=12.0,
            ),
        )

    def test_spring_circle_tracker_prefers_axis_aligned_ball_over_ruler_circle(self) -> None:
        tracker = _load_script("track_round_subject_circles.py")
        selected = tracker.select_spring_axis_circle_candidate(
            ((35.0, 90.0, 10.0), (49.0, 65.0, 10.0)),
            guide_xy=np.asarray([48.0, 92.0]),
            expected_radius=10.0,
        )
        self.assertEqual((49.0, 65.0, 10.0), selected)

    def test_spring_circle_tracker_rejects_hook_when_ball_reaches_lower_edge(self) -> None:
        tracker = _load_script("track_round_subject_circles.py")
        center, accepted = tracker.resolve_spring_boundary_candidate(
            np.asarray([51.0, 70.0]),
            guide_xy=np.asarray([52.0, 91.0]),
            frame_height=100,
            expected_radius=10.0,
        )
        np.testing.assert_array_equal(center, np.asarray([52.0, 91.0]))
        self.assertFalse(accepted)

    def test_circle_tracker_marks_unresolved_boundary_excursion_out_of_frame(self) -> None:
        tracker = _load_script("track_round_subject_circles.py")
        masks = np.ones((3, 40, 60), np.uint8)
        states = tracker.resolve_subject_lifecycle(
            masks,
            canonical_state=np.asarray([0, 3, 0], np.uint8),
            interpolated_guides=np.asarray(
                [[30.0, 20.0], [57.0, 20.0], [30.0, 20.0]], np.float64
            ),
            detected=np.asarray([True, True, True]),
            expected_radius=5.0,
        )
        self.assertEqual([0, 2, 0], states.tolist())
        self.assertFalse(masks[1].any())

    def test_circle_tracker_retains_confirmed_nonvisible_lifecycle(self) -> None:
        tracker = _load_script("track_round_subject_circles.py")
        masks = np.ones((3, 40, 60), np.uint8)
        states = tracker.resolve_subject_lifecycle(
            masks,
            canonical_state=np.asarray([0, 1, 2], np.uint8),
            interpolated_guides=np.asarray(
                [[30.0, 20.0], [30.0, 20.0], [30.0, 20.0]], np.float64
            ),
            detected=np.asarray([True, True, True]),
            expected_radius=5.0,
        )
        self.assertEqual([0, 1, 2], states.tolist())
        self.assertFalse(masks[1:].any())

    def test_audit_evidence_decode_does_not_depend_on_random_seek(self) -> None:
        audit = _load_script("audit_v14.py")

        class SequentialCapture:
            def __init__(self, _path: str):
                self.index = 0

            def isOpened(self):
                return True

            def get(self, _property):
                return 5

            def set(self, _property, _value):
                raise AssertionError("random seeking is not reliable")

            def read(self):
                if self.index == 5:
                    return False, None
                frame = np.full((8, 8, 3), self.index, np.uint8)
                self.index += 1
                return True, frame

            def release(self):
                return None

        with patch.object(audit.cv2, "VideoCapture", SequentialCapture):
            frames = audit._spaced_video_frames(Path("fixture.mp4"), count=3)
        self.assertEqual([0, 2, 4], [int(frame[0, 0, 0]) for frame in frames])


if __name__ == "__main__":
    unittest.main()
