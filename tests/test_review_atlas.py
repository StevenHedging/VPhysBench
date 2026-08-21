import unittest
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from physbench.reference_observations.curation.review_atlas import (
    render_review_atlas_pages,
)


class ReviewAtlasTests(unittest.TestCase):
    def test_paginates_labeled_images_in_input_order(self) -> None:
        entries = [
            ("case_b", np.full((20, 40, 3), 20, np.uint8)),
            ("case_a", np.full((20, 40, 3), 40, np.uint8)),
            ("case_c", np.full((20, 40, 3), 60, np.uint8)),
        ]

        pages = render_review_atlas_pages(
            entries,
            columns=2,
            rows=1,
            tile_width=100,
            tile_height=80,
        )

        self.assertEqual(len(pages), 2)
        self.assertEqual(pages[0].shape, (80, 200, 3))
        self.assertEqual(pages[1].shape, (80, 200, 3))
        # The first page preserves the caller's case_b, case_a ordering.
        self.assertGreater(float(pages[0][35:70, 100:200].mean()),
                           float(pages[0][35:70, 0:100].mean()))

    def test_rejects_invalid_or_empty_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "entries must not be empty"):
            render_review_atlas_pages([])
        with self.assertRaisesRegex(ValueError, "uint8 HWC BGR"):
            render_review_atlas_pages(
                [("case", np.zeros((20, 20), np.uint8))]
            )

    def test_cli_builds_a_page_for_selected_release_case(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/reference_observations/build_review_atlas.py",
                    "datasets/releases/14.0.0/dataset.json",
                    temporary,
                    "--scene-id",
                    "uniform_circular_motion",
                    "--case-id",
                    "circular_r1_silver02cm_img_0370",
                    "--columns",
                    "1",
                    "--rows",
                    "1",
                    "--tile-width",
                    "160",
                    "--tile-height",
                    "120",
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            page = Path(temporary) / "uniform_circular_motion" / "page_001.png"
            self.assertTrue(page.is_file())

    def test_cli_builds_independent_anchor_evidence_page(self) -> None:
        case_id = "pendulum_r2_ltot0130mm_lrope0120mm_r010mm_a010deg"
        review_root = Path(
            ".local/reference_observation_curation/full/review"
        )
        anchor = review_root / "pendulum" / case_id / "anchor.png"
        if not anchor.is_file():
            self.skipTest("independent-anchor audit evidence is not available")

        with tempfile.TemporaryDirectory() as temporary:
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/reference_observations/build_review_atlas.py",
                    "datasets/releases/14.0.0/dataset.json",
                    temporary,
                    "--scene-id",
                    "pendulum",
                    "--case-id",
                    case_id,
                    "--evidence",
                    "independent-anchor",
                    "--review-root",
                    str(review_root),
                    "--columns",
                    "1",
                    "--rows",
                    "1",
                    "--tile-width",
                    "160",
                    "--tile-height",
                    "120",
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            page = Path(temporary) / "pendulum" / "page_001.png"
            self.assertTrue(page.is_file())


if __name__ == "__main__":
    unittest.main()
