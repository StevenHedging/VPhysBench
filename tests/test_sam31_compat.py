from __future__ import annotations

import inspect
from types import SimpleNamespace
import unittest

from physbench.evaluation.common.masks.sam31_compat import (
    install_birth_conditioning_fix as install_fix,
)


def predictor_for(tracker):
    return SimpleNamespace(model=SimpleNamespace(tracker=SimpleNamespace(model=tracker)))


class BoundaryTracker:
    """Small native-call boundary; state transitions are covered with vendor code."""

    def __init__(self, flag=False):
        self.add_all_frames_to_correct_as_cond = flag

    def add_new_masks(self, inference_state, frame_idx, obj_ids, masks,
                      add_mask_to_memory=False, reconditioning=False):
        if masks == "raise":
            raise RuntimeError(f"neural inference failed; conditioning={self.add_all_frames_to_correct_as_cond}")
        return self.add_all_frames_to_correct_as_cond, add_mask_to_memory, reconditioning


class BirthConditioningCompatibilityTest(unittest.TestCase):
    def test_new_birth_only_is_conditioning_and_arguments_are_preserved(self):
        tracker = BoundaryTracker()
        install_fix(predictor_for(tracker))
        state = {"obj_id_to_idx": {0: 0}}
        self.assertEqual((True, True, False), tracker.add_new_masks(state, 6, [1], None, True))
        self.assertFalse(tracker.add_all_frames_to_correct_as_cond)
        self.assertEqual((False, False, False), tracker.add_new_masks(state, 6, [0], None))
        self.assertEqual((False, False, True), tracker.add_new_masks(
            state, 6, [1], None, reconditioning=True))

    def test_flag_restored_after_inference_exception(self):
        for original in (False, True):
            tracker = BoundaryTracker(original)
            install_fix(predictor_for(tracker))
            with self.assertRaisesRegex(RuntimeError, "conditioning=True"):
                tracker.add_new_masks({"obj_id_to_idx": {}}, 5, [0], "raise")
            self.assertIs(tracker.add_all_frames_to_correct_as_cond, original)

    def test_installation_is_instance_only_idempotent_and_preserves_signature(self):
        tracker, untouched = BoundaryTracker(), BoundaryTracker()
        predictor = predictor_for(tracker)
        signature = inspect.signature(tracker.add_new_masks)
        self.assertTrue(install_fix(predictor))
        installed = tracker.add_new_masks
        self.assertTrue(install_fix(predictor))
        self.assertIs(installed, tracker.add_new_masks)
        self.assertEqual(signature, inspect.signature(tracker.add_new_masks))
        self.assertEqual((False, False, False), untouched.add_new_masks(
            {"obj_id_to_idx": {}}, 5, [0], None))

    def test_incompatible_interface_reports_not_installed(self):
        self.assertFalse(install_fix(SimpleNamespace()))

    def test_malformed_tracker_fails_clearly(self):
        tracker = SimpleNamespace(add_new_masks=lambda unexpected: None,
                                  add_all_frames_to_correct_as_cond=False)
        with self.assertRaisesRegex(RuntimeError, "SAM3.*add_new_masks"):
            install_fix(predictor_for(tracker))


if __name__ == "__main__":
    unittest.main()
