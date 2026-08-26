from __future__ import annotations

import os
from pathlib import Path
import time
import unittest

import cv2

from physbench.evaluation.common.csti.observation import PromptGroupConfig
from physbench.evaluation.common.masks.sam31_text import (
    SAM31_CHECKPOINT_SHA256,
    Sam31TextVideoSegmenter,
)


class Sam31TextCheckpointIntegrationTest(unittest.TestCase):
    def test_local_checkpoint_runs_text_only_forward_video_session(self) -> None:
        checkpoint_value = os.environ.get("VPHYSBENCH_SAM31_CHECKPOINT", "")
        video_value = os.environ.get("VPHYSBENCH_SAM31_SMOKE_VIDEO", "")
        if not checkpoint_value or not video_value:
            self.skipTest(
                "set VPHYSBENCH_SAM31_CHECKPOINT and "
                "VPHYSBENCH_SAM31_SMOKE_VIDEO for the real-model smoke test"
            )
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is not installed")
        if not torch.cuda.is_available():
            self.skipTest("CUDA is unavailable")
        checkpoint = Path(checkpoint_value)
        video = Path(video_value)
        self.assertTrue(checkpoint.is_file())
        self.assertTrue(video.is_file())
        capture = cv2.VideoCapture(str(video))
        frames = []
        try:
            while len(frames) < 3:
                success, frame = capture.read()
                if not success:
                    break
                frames.append(cv2.resize(frame, (480, 270)))
        finally:
            capture.release()
        self.assertEqual(3, len(frames))
        segmenter = Sam31TextVideoSegmenter(
            {
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": SAM31_CHECKPOINT_SHA256,
                "device": "cuda",
                "precision": "bfloat16",
                "output_probability_threshold": 0.3,
                "max_num_objects": 16,
                "multiplex_count": 16,
                "compile": False,
                "warm_up": False,
                "use_fa3": False,
                "use_rope_real": True,
                "async_loading_frames": False,
            }
        )
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()

        candidates = segmenter.segment(
            frames,
            (
                PromptGroupConfig(
                    group_id="subject",
                    text=os.environ.get(
                        "VPHYSBENCH_SAM31_SMOKE_PROMPT", "metal block"
                    ),
                    entity_classes=("subject",),
                ),
            ),
        )

        elapsed = time.perf_counter() - started
        self.assertGreaterEqual(len(candidates), 1)
        for candidate in candidates:
            self.assertEqual((3, 270, 480), candidate.masks.shape)
            self.assertEqual((3, 4), candidate.boxes_xywh.shape)
            self.assertEqual((3,), candidate.confidences.shape)
        diagnostics = {
            "elapsed_s": elapsed,
            "candidate_count": len(candidates),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "segmenter": segmenter.describe(),
        }
        print(f"SAM31_TEXT_SMOKE={diagnostics}")


if __name__ == "__main__":
    unittest.main()
