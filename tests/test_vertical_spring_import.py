from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
import zipfile

import cv2
import numpy as np
from openpyxl import Workbook

from physbench.datasets.vertical_spring_import import (
    BallDetection,
    SourceMember,
    TrackSample,
    TrialAnnotation,
    ball_mask,
    detect_ball,
    detect_release_return,
    inventory_archive,
    load_trial_annotations,
    map_trials_to_sources,
)


def _write_workbook(path: Path, rows: list[tuple[object, ...]]) -> Path:
    workbook = Workbook()
    settings = workbook.active
    settings.title = "实验设置"
    spring = workbook.create_sheet("弹簧设置")
    spring.append(["弹簧ID", "S01"])
    trials = workbook.create_sheet("Trial记录")
    trials.append(["Trial ID", "弹簧ID", "视频文件", "初始位移 (mm)"])
    for row in rows:
        trials.append(row)
    workbook.save(path)
    return path


class IntakeTests(unittest.TestCase):
    def test_load_trials_discovers_real_header_row_and_adds_mov_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "spring.xlsx"
            workbook = Workbook()
            sheet = workbook.active
            sheet.title = "Trial记录"
            sheet.append(["Trial 记录｜每次释放一行"])
            sheet.append([])
            sheet.append(["整批统一静止释放。"])
            sheet.append([])
            sheet.append([
                "Trial编号",
                "弹簧编号",
                "重复编号",
                "视频文件名",
                "释放长度-平衡长度_mm",
            ])
            sheet.append(["T001", "S01", None, "IMG_1538", 10])
            workbook.save(path)

            trials = load_trial_annotations(path)

        self.assertEqual("IMG_1538.MOV", trials[0].video_name)
        self.assertEqual(6, trials[0].workbook_row)

    def test_load_trials_preserves_signed_displacement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workbook = _write_workbook(
                Path(directory) / "spring.xlsx",
                [
                    ("T001", "S01", "IMG_1518.MOV", 30.0),
                    ("T002", "S01", "IMG_1519.MOV", -40.0),
                    ("T003", "S01", None, None),
                ],
            )

            trials = load_trial_annotations(workbook)

        self.assertEqual(
            [
                ("T001", 30.0, "below", 0.03, 2),
                ("T002", -40.0, "above", 0.04, 3),
            ],
            [
                (
                    item.trial_id,
                    item.displacement_mm,
                    item.direction,
                    item.displacement_m,
                    item.workbook_row,
                )
                for item in trials
            ],
        )

    def test_mapping_excludes_missing_required_video(self) -> None:
        trial = TrialAnnotation("T001", "S01", "IMG_1652.MOV", 30.0, 2)

        result = map_trials_to_sources([trial], [])

        self.assertEqual((), result.accepted)
        self.assertEqual(1, len(result.exclusions))
        self.assertEqual(
            "workbook_video_missing_from_archive",
            result.exclusions[0].reason,
        )

    def test_inventory_uses_zip_crc_and_basename(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "source.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("batch/IMG_1518.MOV", b"video-a")
                handle.writestr("batch/notes.xlsx", b"workbook")

            members = inventory_archive(archive)

        self.assertEqual(1, len(members))
        self.assertEqual("IMG_1518.MOV", members[0].basename)
        self.assertEqual(len(b"video-a"), members[0].size)
        self.assertIsInstance(members[0].crc32, int)

    def test_mapping_rejects_nonidentical_duplicate_basenames(self) -> None:
        trial = TrialAnnotation("T001", "S01", "IMG_1518.MOV", 30.0, 2)
        sources = [
            SourceMember("a/IMG_1518.MOV", "IMG_1518.MOV", 10, 1),
            SourceMember("b/IMG_1518.MOV", "IMG_1518.MOV", 11, 2),
        ]

        result = map_trials_to_sources([trial], sources)

        self.assertEqual((), result.accepted)
        self.assertEqual("ambiguous_source_members", result.exclusions[0].reason)

    def test_load_trials_rejects_zero_displacement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workbook = _write_workbook(
                Path(directory) / "spring.xlsx",
                [("T001", "S01", "IMG_1518.MOV", 0.0)],
            )

            with self.assertRaisesRegex(ValueError, "invalid displacement"):
                load_trial_annotations(workbook)

    def test_load_trials_rejects_unsupported_spring(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workbook = _write_workbook(
                Path(directory) / "spring.xlsx",
                [("T001", "S02", "IMG_1518.MOV", 30.0)],
            )

            with self.assertRaisesRegex(ValueError, "unsupported spring ID"):
                load_trial_annotations(workbook)


class TrajectoryTests(unittest.TestCase):
    @staticmethod
    def _track(direction: str, *, irregular: bool = False) -> list[TrackSample]:
        times = np.arange(0.0, 2.0, 1.0 / 240.0)
        if irregular:
            times = times + np.linspace(0.0, 0.004, len(times))
        sign = 1.0 if direction == "below" else -1.0
        y = 600.0 + sign * 120.0 * np.exp(-0.08 * times) * np.cos(
            2 * np.pi * times / 0.8
        )
        return [
            TrackSample(index, float(time), 400.0, float(center_y), 60.0, 1.0)
            for index, (time, center_y) in enumerate(zip(times, y))
        ]

    def test_return_detector_selects_first_lower_return_for_below(self) -> None:
        candidate = detect_release_return(self._track("below"), "below", 0.8)

        self.assertLessEqual(abs(candidate.time_s - 0.8), 2 / 240)
        self.assertGreater(candidate.center_y, 700.0)

    def test_return_detector_selects_first_upper_return_for_above(self) -> None:
        candidate = detect_release_return(self._track("above"), "above", 0.8)

        self.assertLessEqual(abs(candidate.time_s - 0.8), 2 / 240)
        self.assertLess(candidate.center_y, 500.0)

    def test_return_detector_uses_timestamps_not_constant_fps(self) -> None:
        candidate = detect_release_return(
            self._track("below", irregular=True),
            "below",
            0.8,
        )

        self.assertLessEqual(abs(candidate.time_s - 0.8), 0.015)

    def test_return_detector_accepts_regular_analysis_stride(self) -> None:
        sampled = self._track("below")[::16]

        candidate = detect_release_return(sampled, "below", 0.8)

        self.assertLessEqual(abs(candidate.time_s - 0.8), 0.04)
        self.assertEqual(1.0, candidate.track_coverage)

    def test_return_detector_preserves_small_motion_at_sparse_stride(self) -> None:
        times = np.arange(0.0, 2.0, 1.0 / 240.0)
        y = 600.0 + 18.0 * np.cos(2 * np.pi * times / 0.8)
        full = [
            TrackSample(index, float(time), 400.0, float(center_y), 60.0, 1.0)
            for index, (time, center_y) in enumerate(zip(times, y))
        ]

        candidate = detect_release_return(full[::16], "below", 0.8)

        self.assertLessEqual(abs(candidate.time_s - 0.8), 0.04)

    def test_return_detector_skips_source_prehold(self) -> None:
        times = np.arange(0.0, 2.4, 1.0 / 240.0)
        motion_time = np.maximum(times - 0.6, 0.0)
        y = np.where(
            times < 0.6,
            720.0,
            600.0
            + 120.0
            * np.exp(-0.08 * motion_time)
            * np.cos(2 * np.pi * motion_time / 0.8),
        )
        track = [
            TrackSample(index, float(time), 400.0, float(center_y), 60.0, 1.0)
            for index, (time, center_y) in enumerate(zip(times, y))
        ]

        candidate = detect_release_return(track, "below", 0.8)

        self.assertLessEqual(abs(candidate.time_s - 1.4), 2 / 240)

    def test_return_detector_rejects_insufficient_vertical_motion(self) -> None:
        track = [
            TrackSample(index, index / 240.0, 400.0, 600.0, 60.0, 1.0)
            for index in range(480)
        ]

        with self.assertRaisesRegex(ValueError, "insufficient_vertical_motion"):
            detect_release_return(track, "below", 0.8)

    def test_detect_ball_finds_synthetic_circle(self) -> None:
        frame = np.full((480, 270, 3), 230, dtype=np.uint8)
        cv2.circle(frame, (145, 330), 24, (40, 40, 40), 3)
        cv2.circle(frame, (145, 330), 20, (130, 130, 130), -1)

        detection = detect_ball(frame)

        self.assertLessEqual(abs(detection.center_x - 145.0), 3.0)
        self.assertLessEqual(abs(detection.center_y - 330.0), 3.0)
        self.assertLessEqual(abs(detection.radius - 24.0), 4.0)

    def test_ball_mask_is_binary_filled_disk(self) -> None:
        detection = BallDetection(60.0, 50.0, 20.0, 1.0)

        mask = ball_mask((100, 120), detection)

        self.assertEqual({0, 1}, set(np.unique(mask)))
        self.assertEqual(1, int(mask[50, 60]))
        self.assertEqual(0, int(mask[20, 60]))
        self.assertGreater(int(mask.sum()), 1100)
        self.assertLess(int(mask.sum()), 1400)


if __name__ == "__main__":
    unittest.main()
