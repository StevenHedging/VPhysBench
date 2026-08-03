from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from physbench.baseline_runtime import (
    build_i2v_media_contract,
    materialize_i2v_conditioning,
)

class Cosmos3SpatialAlignmentTests(unittest.TestCase):
    def test_run_local_condition_uses_full_view_contain_and_edge_margin(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.png"
            output = root / "run" / "conditioning" / "condition.png"
            frame = np.zeros((2, 4, 3), dtype=np.uint8)
            frame[:, 0] = (10, 20, 30)
            frame[:, 1] = (40, 50, 60)
            frame[:, 2] = (70, 80, 90)
            frame[:, 3] = (100, 110, 120)
            self.assertTrue(cv2.imwrite(str(source), frame))

            audit = materialize_i2v_conditioning(
                source,
                output,
                build_i2v_media_contract(
                    conditioning_asset="source.png",
                    width=8,
                    height=8,
                    temporal={"fps": 8, "num_frames": 9},
                ),
            )

            actual = cv2.imread(str(output), cv2.IMREAD_COLOR)
            self.assertEqual([0, 2, 8, 4], audit["content_rect_xywh"])
            self.assertEqual((8, 8), (actual.shape[1], actual.shape[0]))
            np.testing.assert_array_equal(actual[0], actual[2])
            np.testing.assert_array_equal(actual[-1], actual[-3])


if __name__ == "__main__":
    unittest.main()
