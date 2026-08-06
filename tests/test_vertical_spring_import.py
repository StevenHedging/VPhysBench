from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
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
    TurningFrameCandidate,
    VideoAnalysis,
    analyze_video,
    ball_mask,
    build_caption,
    build_physics,
    detect_ball,
    detect_release_return,
    inventory_archive,
    load_trial_annotations,
    map_trials_to_sources,
    materialize_case,
    probe_frame_timestamps,
    trim_video_exact,
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

    def test_detect_ball_does_not_follow_small_ring_distractor(self) -> None:
        frame = np.full((480, 270, 3), 230, dtype=np.uint8)
        cv2.circle(frame, (145, 200), 14, (40, 40, 40), 3)
        cv2.circle(frame, (165, 350), 26, (40, 40, 40), 3)
        cv2.circle(frame, (165, 350), 22, (130, 130, 130), -1)
        stale_ring = BallDetection(145.0, 200.0, 14.0, 1.0)

        detection = detect_ball(frame, stale_ring)

        self.assertGreater(detection.radius, 18.0)
        self.assertLessEqual(abs(detection.center_y - 350.0), 4.0)

    def test_ball_mask_is_binary_filled_disk(self) -> None:
        detection = BallDetection(60.0, 50.0, 20.0, 1.0)

        mask = ball_mask((100, 120), detection)

        self.assertEqual({0, 1}, set(np.unique(mask)))
        self.assertEqual(1, int(mask[50, 60]))
        self.assertEqual(0, int(mask[20, 60]))
        self.assertGreater(int(mask.sum()), 1100)
        self.assertLess(int(mask.sum()), 1400)


class MediaTests(unittest.TestCase):
    def test_physics_uses_positive_displacement_and_fixed_symbols(self) -> None:
        physics = build_physics("spring_t001", -40.0)

        objects = physics["physics"]["objects"]
        environment = physics["physics"]["environment"]
        self.assertEqual(
            0.04,
            objects["object_1"]["initial_displacement"]["value"],
        )
        quantities = [*objects["object_1"].values(), *environment.values()]
        self.assertTrue(all(quantity["value"] >= 0 for quantity in quantities))
        self.assertEqual(
            {"m", "r", "x_0", "k", "L_0", "g"},
            {quantity["symbol"] for quantity in quantities},
        )

    def test_caption_direction_and_symbols_match_physics(self) -> None:
        caption = build_caption("spring_t001", "above")["caption"]

        self.assertIn("above equilibrium", caption)
        self.assertNotIn("below equilibrium", caption)
        for symbol in ("m", "r", "x_0", "k", "L_0", "g"):
            self.assertIn(symbol, caption)

    def test_exact_trim_retains_frames_and_presentation_intervals(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            target = root / "target.mp4"
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc2=size=96x64:rate=12",
                    "-frames:v",
                    "12",
                    "-vf",
                    "setpts=if(lt(N\\,6)\\,N/(12*TB)\\,(0.5+(N-6)/6)/TB)",
                    "-vsync",
                    "0",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    str(source),
                ],
                check=True,
            )
            source_times = probe_frame_timestamps(source)

            trim_video_exact(source, 4, target)

            target_times = probe_frame_timestamps(target)
            self.assertEqual(8, len(target_times))
            self.assertAlmostEqual(0.0, target_times[0], places=5)
            expected_deltas = np.diff(source_times[4:])
            actual_deltas = np.diff(target_times)
            np.testing.assert_allclose(actual_deltas, expected_deltas, atol=1e-4)

    def test_inventory_cli_writes_trial_source_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workbook = _write_workbook(
                root / "spring.xlsx",
                [("T001", "S01", "IMG_1518.MOV", 30.0)],
            )
            archive = root / "source.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.write(workbook, "batch/spring.xlsx")
                handle.writestr("batch/IMG_1518.MOV", b"video")
                handle.writestr("batch/IMG_1518(1).MOV", b"video")
            output = root / "output"

            subprocess.run(
                [
                    sys.executable,
                    "scripts/import_vertical_spring_oscillator.py",
                    "inventory",
                    "--archive",
                    str(archive),
                    "--output-dir",
                    str(output),
                ],
                check=True,
                env={"PYTHONPATH": "src"},
            )

            payload = json.loads(
                (output / "normalized_annotations.json").read_text()
            )
            self.assertEqual("source.zip", payload["archive_name"])
            self.assertNotIn("archive", payload)
            self.assertEqual(
                [["batch/IMG_1518(1).MOV", "batch/IMG_1518.MOV"]],
                payload["duplicate_source_groups"],
            )
            self.assertEqual(
                [
                    {
                        "canonical_member": "batch/IMG_1518.MOV",
                        "excluded_members": ["batch/IMG_1518(1).MOV"],
                        "reason": "byte_identical_duplicate_source",
                        "size": 5,
                    }
                ],
                payload.get("duplicate_source_decisions"),
            )
            self.assertEqual(1, payload["summary"]["accepted_count"])
            self.assertEqual("T001", payload["accepted"][0]["trial_id"])
            self.assertEqual(
                "batch/IMG_1518.MOV",
                payload["accepted"][0]["source_member"],
            )

    def test_materialize_case_writes_atomic_case_members(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_video = root / "source.mp4"
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=c=gray:size=96x64:rate=12",
                    "-frames:v",
                    "12",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    str(source_video),
                ],
                check=True,
            )
            trial = TrialAnnotation("T001", "S01", "IMG_1538.MOV", -40.0, 6)
            source = SourceMember("batch/IMG_1538.MOV", "IMG_1538.MOV", 1, 2)
            analysis = VideoAnalysis(
                candidate=TurningFrameCandidate(4, 4 / 12, 48, 32, 12, 0.8, 1.0),
                full_resolution_ball=BallDetection(48, 32, 12, 1.0),
                frame_count=12,
                displayed_width=96,
                displayed_height=64,
                analysis_stride=1,
            )

            draft = materialize_case(
                source_video,
                trial,
                source,
                analysis,
                root,
            )

            self.assertEqual(
                "vertical_spring_s01_x40mm_above_img_1538",
                draft.case_id,
            )
            self.assertEqual(
                (
                    "assets/vertical_spring_oscillator/"
                    "spring_m515p6g_r25mm_x40mm_above_img1538"
                ),
                draft.audit["asset_directory"],
            )
            self.assertEqual(6, draft.audit["workbook_row"])
            self.assertEqual(1, draft.audit["source_size"])
            self.assertEqual(2, draft.audit["source_crc32"])
            self.assertEqual("batch/IMG_1538.MOV", draft.audit["source_group"])
            self.assertEqual(0.8, draft.audit["detector"]["observed_period_s"])
            self.assertEqual(1.0, draft.audit["detector"]["track_coverage"])
            required = {
                "caption.json",
                "physics.json",
                "canonical/reference.mp4",
                "canonical/first_frame.png",
                "canonical/masks/01.png",
                "canonical/masks/01.npz",
                "canonical/masks/manifest.json",
            }
            self.assertTrue(
                all((draft.case_directory / path).is_file() for path in required)
            )
            mask_payload = np.load(draft.case_directory / "canonical/masks/01.npz")
            self.assertEqual((1, 64, 96), mask_payload["masks"].shape)
            manifest = json.loads(
                (draft.case_directory / "canonical/masks/manifest.json").read_text()
            )
            self.assertTrue(manifest["source_first_frame"].startswith("assets/"))
            self.assertTrue(manifest["instances"][0]["asset"].startswith("assets/"))
            self.assertEqual(
                0.04,
                json.loads((draft.case_directory / "physics.json").read_text())[
                    "physics"
                ]["objects"]["object_1"]["initial_displacement"]["value"],
            )

    def test_analyze_video_refines_first_post_cycle_turning_frame(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "oscillator.mp4"
            writer = cv2.VideoWriter(
                str(video),
                cv2.VideoWriter_fourcc(*"mp4v"),
                120.0,
                (270, 480),
            )
            self.assertTrue(writer.isOpened())
            for frame_index in range(240):
                time_s = frame_index / 120.0
                motion_time = max(0.0, time_s - 0.2)
                center_y = (
                    330.0
                    if time_s < 0.2
                    else 270.0 + 60.0 * np.cos(2 * np.pi * motion_time / 0.8)
                )
                frame = np.full((480, 270, 3), 230, dtype=np.uint8)
                cv2.circle(frame, (150, round(center_y)), 24, (30, 30, 30), 3)
                cv2.circle(frame, (150, round(center_y)), 20, (130, 130, 130), -1)
                writer.write(frame)
            writer.release()

            analysis = analyze_video(
                video,
                "below",
                analysis_stride=4,
                theoretical_period_s=0.8,
            )

            self.assertLessEqual(abs(analysis.candidate.time_s - 1.0), 2 / 120)
            self.assertLessEqual(abs(analysis.full_resolution_ball.center_x - 150), 4)
            self.assertLessEqual(abs(analysis.full_resolution_ball.center_y - 330), 4)

    def test_analyze_cli_writes_pending_review_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workbook = _write_workbook(
                root / "spring.xlsx",
                [("T001", "S01", "IMG_1518.MOV", 30.0)],
            )
            video = root / "IMG_1518.MOV"
            writer = cv2.VideoWriter(
                str(video),
                cv2.VideoWriter_fourcc(*"mp4v"),
                120.0,
                (270, 480),
            )
            for frame_index in range(180):
                time_s = frame_index / 120.0
                motion_time = max(0.0, time_s - 0.2)
                center_y = (
                    330.0
                    if time_s < 0.2
                    else 270.0 + 60.0 * np.cos(2 * np.pi * motion_time / 0.8)
                )
                frame = np.full((480, 270, 3), 230, dtype=np.uint8)
                cv2.circle(frame, (150, round(center_y)), 24, (30, 30, 30), 3)
                cv2.circle(frame, (150, round(center_y)), 20, (130, 130, 130), -1)
                writer.write(frame)
            writer.release()
            archive = root / "source.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.write(video, "batch/IMG_1518.MOV")
            output = root / "output"

            subprocess.run(
                [
                    sys.executable,
                    "scripts/import_vertical_spring_oscillator.py",
                    "analyze",
                    "--archive",
                    str(archive),
                    "--workbook",
                    str(workbook),
                    "--output-dir",
                    str(output),
                    "--analysis-stride",
                    "4",
                    "--workers",
                    "2",
                ],
                check=True,
                env={"PYTHONPATH": "src"},
            )

            rows = [
                json.loads(line)
                for line in (output / "review_candidates.jsonl").read_text().splitlines()
            ]
            self.assertEqual(1, len(rows))
            self.assertEqual("pending", rows[0]["status"])
            self.assertEqual("T001", rows[0]["trial_id"])
            self.assertLessEqual(abs(rows[0]["source_start_time_s"] - 1.0), 0.02)

    def test_materialize_cli_uses_only_approved_reviews(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "IMG_1538.MOV"
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=c=gray:size=96x64:rate=12",
                    "-frames:v",
                    "12",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    str(video),
                ],
                check=True,
            )
            archive = root / "source.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.write(video, "batch/IMG_1538.MOV")
            analysis_path = root / "review_candidates.jsonl"
            analysis_path.write_text(json.dumps({
                "status": "pending",
                "trial_id": "T001",
                "spring_id": "S01",
                "workbook_row": 6,
                "video_name": "IMG_1538.MOV",
                "source_member": "batch/IMG_1538.MOV",
                "source_size": video.stat().st_size,
                "source_crc32": 123,
                "signed_displacement_mm": -40.0,
                "direction": "above",
                "source_start_frame": 4,
                "source_start_time_s": 4 / 12,
                "observed_period_s": 0.8,
                "track_coverage": 1.0,
                "analysis_ball_xyr": [12.0, 8.0, 3.0],
                "full_resolution_ball_xyr": [48.0, 32.0, 12.0],
                "source_frame_count": 12,
                "displayed_width": 96,
                "displayed_height": 64,
                "analysis_stride": 1,
            }) + "\n")
            review_path = root / "review_decisions.jsonl"
            review_path.write_text(json.dumps({
                "trial_id": "T001",
                "status": "approved",
                "reviewer": "visual_review",
            }) + "\n")

            subprocess.run(
                [
                    sys.executable,
                    "scripts/import_vertical_spring_oscillator.py",
                    "materialize",
                    "--archive",
                    str(archive),
                    "--analysis",
                    str(analysis_path),
                    "--review",
                    str(review_path),
                    "--repo-root",
                    str(root),
                    "--workers",
                    "2",
                ],
                check=True,
                env={"PYTHONPATH": "src"},
            )

            audit = root / (
                "datasets/provenance/imports/"
                "vertical_spring_oscillator_20260806_import_audit.jsonl"
            )
            rows = [json.loads(line) for line in audit.read_text().splitlines()]
            self.assertEqual(
                ["vertical_spring_s01_x40mm_above_img_1538"],
                [row["case_id"] for row in rows],
            )

    def test_review_sheets_cli_renders_candidate_overlay(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "IMG_1538.MOV"
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=c=gray:size=96x64:rate=12",
                    "-frames:v",
                    "12",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    str(video),
                ],
                check=True,
            )
            archive = root / "source.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.write(video, "batch/IMG_1538.MOV")
            analysis = root / "review_candidates.jsonl"
            analysis.write_text(json.dumps({
                "status": "pending",
                "trial_id": "T001",
                "source_member": "batch/IMG_1538.MOV",
                "source_start_frame": 4,
                "full_resolution_ball_xyr": [48.0, 32.0, 12.0],
                "direction": "below",
                "signed_displacement_mm": 30.0,
            }) + "\n")
            output = root / "review"

            subprocess.run(
                [
                    sys.executable,
                    "scripts/import_vertical_spring_oscillator.py",
                    "review-sheets",
                    "--archive",
                    str(archive),
                    "--analysis",
                    str(analysis),
                    "--output-dir",
                    str(output),
                ],
                check=True,
                env={"PYTHONPATH": "src"},
            )

            pages = sorted(output.glob("review_page_*.png"))
            self.assertEqual(1, len(pages))
            image = cv2.imread(str(pages[0]))
            self.assertIsNotNone(image)
            self.assertGreater(image.shape[0], 100)
            self.assertGreater(image.shape[1], 100)


if __name__ == "__main__":
    unittest.main()
