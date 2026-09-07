"""Opt-in CPU state regression against the pinned SAM3 checkout; no checkpoint/GPU.

Set VPHYSBENCH_SAM31_STATE_TEST=1 with the pinned vendor on PYTHONPATH.
Native init/index/add/consolidation/removal/propagation code runs unchanged.
Only video loading and neural inference are substituted with CPU tensors.
"""
from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from tests.test_sam31_compat import install_fix, predictor_for


@unittest.skipUnless(os.environ.get("VPHYSBENCH_SAM31_STATE_TEST") == "1",
                     "set VPHYSBENCH_SAM31_STATE_TEST=1 for pinned vendor state tests")
class Sam31NativeStateRegressionTest(unittest.TestCase):
    def setUp(self):
        import torch
        from sam3.model.multiplex_utils import MultiplexController
        from sam3.model.video_tracking_multiplex_demo import VideoTrackingMultiplexDemo

        class CpuTracker(VideoTrackingMultiplexDemo):
            def __init__(self):
                torch.nn.Module.__init__(self)
                self.apply_sigmoid_to_mask_logits_for_mem_enc = True
                self.image_size = self.input_mask_size = self.low_res_mask_size = 4
                self.is_dynamic_model = True
                self.add_all_frames_to_correct_as_cond = False
                self.clear_non_cond_mem_around_input = False
                self.always_start_from_first_ann_frame = False
                self.non_overlap_masks_for_output = False
                self.fill_hole_area = 0
                self.use_memory_selection = False
                self.multiplex_controller = MultiplexController(4).eval()

            def _run_single_frame_inference(self, inference_state, frame_idx,
                                            add_to_existing_state=False,
                                            new_obj_idxs=None, new_obj_ids=None, **kwargs):
                mux = inference_state["multiplex_state"]
                if add_to_existing_state:
                    # This operation normally happens inside the neural forward path.
                    mux.add_objects(new_obj_idxs, object_ids=new_obj_ids)
                count = len(inference_state["obj_ids"])
                packed = torch.ones((mux.num_buckets, 1, 1, 1))
                masks = torch.ones((count, 1, 4, 4))
                return {"pred_masks": masks,
                        "object_score_logits": torch.ones((count, 1)),
                        "maskmem_features": packed.clone(),
                        "maskmem_pos_enc": [packed.clone()],
                        "image_features": None, "image_pos_enc": None,
                        "obj_ptr": packed.clone(),
                        "conditioning_objects": set(new_obj_idxs or []),
                        }, masks

            def _run_memory_encoder(self, inference_state, **kwargs):
                packed = torch.ones((inference_state["multiplex_state"].num_buckets, 1, 1, 1))
                return packed, [packed.clone()], None, None

        self.torch = torch
        self.tracker = CpuTracker()
        with patch("sam3.model.video_tracking_multiplex_demo.load_video_frames",
                   return_value=(torch.zeros((18, 3, 4, 4)), 4, 4)):
            self.state = self.tracker.init_state("fixture", True, True)
        self.state["device"] = torch.device("cpu")

    def add(self, frame, ids):
        self.tracker.add_new_masks(self.state, frame, ids,
                                   self.torch.ones((len(ids), 4, 4)))
        self.tracker.propagate_in_video_preflight(self.state)

    def propagate(self, start, end):
        return list(self.tracker.propagate_in_video(
            self.state, start, end - start, False, tqdm_disable=True))

    def birth_sequence(self):
        self.add(5, [0])
        self.propagate(5, 6)
        self.add(6, [1])
        self.propagate(6, 16)
        self.tracker.remove_object(self.state, 0, need_output=False)

    def test_native_late_birth_survives_early_object_removal_and_propagates(self):
        install_fix(predictor_for(self.tracker))
        self.birth_sequence()
        self.assertEqual([1], self.state["obj_ids"])
        self.assertEqual({1: 0}, self.state["obj_id_to_idx"])
        self.assertEqual({6}, set(self.state["output_dict"]["cond_frame_outputs"]))
        self.assertEqual({6}, set(self.state["mask_inputs_per_obj"][0]))
        self.assertEqual([1], self.state["multiplex_state"].object_ids)
        output = self.propagate(17, 17)
        self.assertEqual((17, [1]), output[0][:2])
        self.assertEqual((1, 1, 4, 4), tuple(output[0][3].shape))

    def test_unwrapped_vendor_reproduces_survivor_reset(self):
        self.birth_sequence()
        self.assertEqual([1], self.state["obj_ids"])
        self.assertEqual({}, self.state["mask_inputs_per_obj"][0])
        with self.assertRaisesRegex(RuntimeError, "No points are provided"):
            self.propagate(17, 17)

    def test_shared_birth_frame_retains_surviving_input(self):
        install_fix(predictor_for(self.tracker))
        self.add(5, [0, 1])
        self.tracker.remove_object(self.state, 0, need_output=False)
        self.assertEqual({5}, set(self.state["mask_inputs_per_obj"][0]))
        self.assertEqual({5}, set(self.state["output_dict"]["cond_frame_outputs"]))
        self.assertEqual([1], self.propagate(6, 6)[0][1])

    def test_existing_object_reconditioning_keeps_native_correction_storage(self):
        install_fix(predictor_for(self.tracker))
        self.add(5, [0])
        self.propagate(5, 6)
        self.tracker.add_new_masks(
            inference_state=self.state,
            frame_idx=6,
            obj_ids=[0],
            masks=self.torch.ones((1, 4, 4)),
            reconditioning=True,
        )
        self.tracker.propagate_in_video_preflight(self.state)
        self.assertEqual({5}, set(self.state["output_dict"]["cond_frame_outputs"]))
        self.assertEqual({6}, set(self.state["consolidated_frame_inds"]["non_cond_frame_outputs"]))
        self.assertEqual([0], self.propagate(7, 7)[0][1])

    def test_remove_all_still_clears_conditioning(self):
        install_fix(predictor_for(self.tracker))
        self.add(5, [0, 1])
        self.tracker.remove_objects(self.state, [0, 1], need_output=False)
        self.assertEqual([], self.state["obj_ids"])
        self.assertEqual({}, self.state["output_dict"]["cond_frame_outputs"])
        self.assertTrue(all(not inputs for inputs in self.state["mask_inputs_per_obj"].values()))


if __name__ == "__main__":
    unittest.main()
