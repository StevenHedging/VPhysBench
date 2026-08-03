from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DRIVER_PATH = ROOT / "baselines" / "cosmos3_nano_i2v" / "driver.py"


def _load_driver_module():
    spec = importlib.util.spec_from_file_location(
        "physbench_test_cosmos3_driver",
        DRIVER_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Cosmos3 driver: {DRIVER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Cosmos3SpatialAlignmentTests(unittest.TestCase):
    def test_run_local_condition_uses_full_view_contain_and_edge_margin(
        self,
    ) -> None:
        driver = _load_driver_module()
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

            driver._materialize_contain_condition(
                source,
                output,
                width=8,
                height=8,
            )

            actual = cv2.imread(str(output), cv2.IMREAD_COLOR)
            expected_content = cv2.resize(
                frame,
                (8, 4),
                interpolation=cv2.INTER_LINEAR,
            )
            expected = cv2.copyMakeBorder(
                expected_content,
                2,
                2,
                0,
                0,
                borderType=cv2.BORDER_REPLICATE,
            )
            np.testing.assert_array_equal(expected, actual)
            np.testing.assert_array_equal(actual[0], actual[2])
            np.testing.assert_array_equal(actual[-1], actual[-3])


if __name__ == "__main__":
    unittest.main()
