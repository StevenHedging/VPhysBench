from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from physbench.baselines.wan22_media import Wan22MediaAdapter


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg/ffprobe required")
class Wan22MediaTests(unittest.TestCase):
    def test_scene_buckets_preserve_portrait_and_landscape_layouts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            portrait_source = root / "portrait.mp4"
            landscape_source = root / "landscape.mp4"
            for source, size in ((portrait_source, "90x160"), (landscape_source, "160x90")):
                subprocess.run([
                    "ffmpeg", "-v", "error", "-y", "-f", "lavfi",
                    "-i", f"testsrc=size={size}:rate=24", "-t", "1",
                    "-c:v", "mpeg4", "-q:v", "3", str(source),
                ], check=True)
            adapter = Wan22MediaAdapter({
                "width": 96, "height": 160, "fps": 24,
                "max_frames": 21, "min_frames": 5, "pad_color": "black",
                "aspect_ratio_buckets": {
                    "enabled": True,
                    "buckets": {
                        "portrait": {
                            "width": 96, "height": 160,
                            "scene_ids": ["pendulum", "free_fall"],
                        },
                        "landscape": {
                            "width": 160, "height": 96,
                            "scene_ids": ["collision_1d"],
                        },
                    },
                },
            })
            portrait = adapter.normalize_video(
                portrait_source, root / "portrait_normalized.mp4",
                materialize=True, scene_id="pendulum",
            )
            landscape = adapter.normalize_video(
                landscape_source, root / "landscape_normalized.mp4",
                materialize=True, scene_id="collision_1d",
            )

            self.assertTrue(adapter.dynamic_resolution)
            self.assertEqual(160 * 96, adapter.max_pixels)
            self.assertEqual("portrait", portrait["aspect_ratio_bucket"]["name"])
            self.assertEqual((96, 160), (
                portrait["output_probe"]["width"], portrait["output_probe"]["height"],
            ))
            self.assertEqual("landscape", landscape["aspect_ratio_bucket"]["name"])
            self.assertEqual((160, 96), (
                landscape["output_probe"]["width"], landscape["output_probe"]["height"],
            ))

    def test_variable_media_is_adapted_without_mutating_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source_320x180_15fps.mp4"
            subprocess.run([
                "ffmpeg", "-v", "error", "-y", "-f", "lavfi",
                "-i", "testsrc=size=320x180:rate=15", "-t", "1.5",
                "-c:v", "mpeg4", "-q:v", "3", str(source),
            ], check=True)
            before = file_sha256(source)
            adapter = Wan22MediaAdapter({
                "width": 96, "height": 160, "fps": 24,
                "max_frames": 33, "min_frames": 17, "pad_color": "black",
            })
            normalized = root / "cache" / "normalized.mp4"
            record = adapter.normalize_video(source, normalized, materialize=True)
            first_frame = root / "cache" / "first.png"
            adapter.normalize_first_frame(source, first_frame, source_is_video=True, materialize=True)

            self.assertEqual(before, file_sha256(source), "source video was modified")
            self.assertEqual(320, record["source_probe"]["width"])
            self.assertEqual(180, record["source_probe"]["height"])
            self.assertAlmostEqual(15.0, record["source_probe"]["fps"])
            self.assertEqual(96, record["output_probe"]["width"])
            self.assertEqual(160, record["output_probe"]["height"])
            self.assertEqual(24, record["output_probe"]["fps"])
            self.assertEqual(33, record["output_probe"]["frames"])
            self.assertEqual(1, record["target_frames"] % 4)
            self.assertEqual(1.0, record["time_mapping"]["encoded_to_physical_speed"])
            self.assertTrue(first_frame.is_file())

    def test_slow_motion_is_restored_only_in_baseline_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "slow_motion.mp4"
            subprocess.run([
                "ffmpeg", "-v", "error", "-y", "-f", "lavfi",
                "-i", "testsrc=size=160x120:rate=30", "-t", "8",
                "-c:v", "mpeg4", "-q:v", "3", str(source),
            ], check=True)
            before = file_sha256(source)
            adapter = Wan22MediaAdapter({
                "width": 96, "height": 160, "fps": 24,
                "max_frames": 121, "min_frames": 17, "pad_color": "black",
            })
            normalized = root / "cache" / "physical_time.mp4"
            record = adapter.normalize_video(
                source, normalized, materialize=True, speed_factor=8.0,
            )

            self.assertEqual(before, file_sha256(source), "source video was modified")
            self.assertEqual(8.0, record["time_mapping"]["encoded_to_physical_speed"])
            self.assertAlmostEqual(1.0, record["time_mapping"]["physical_duration_s"], places=2)
            self.assertEqual(21, record["target_frames"])
            self.assertEqual(25, record["generation_target_frames"])
            self.assertIn("setpts=PTS/8", " ".join(record["command"]))
            self.assertEqual(21, record["output_probe"]["frames"])

    def test_generation_length_covers_reference_last_timestamp(self) -> None:
        adapter = Wan22MediaAdapter({
            "width": 96,
            "height": 160,
            "fps": 24,
            "max_frames": 121,
            "min_frames": 5,
            "pad_color": "black",
        })
        source_probe = {
            "duration_s": 4.421,
            "frames": 943,
            "fps": 213.30819981149858,
        }
        reference_frames = adapter.frame_count(source_probe)
        generation_frames = adapter.generation_frame_count(source_probe)

        self.assertEqual(105, reference_frames)
        self.assertEqual(109, generation_frames)
        self.assertLess(
            (reference_frames - 1) / adapter.fps,
            source_probe["duration_s"],
        )
        self.assertGreaterEqual(
            (generation_frames - 1) / adapter.fps,
            source_probe["duration_s"],
        )
        self.assertEqual(1, generation_frames % 4)

if __name__ == "__main__":
    unittest.main()
