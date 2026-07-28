from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from physbench.evaluation.common import media
from physbench.evaluation.common.media import (
    VideoInfo,
    VideoProtocolError,
    sample_video,
)


class FakeCapture:
    def __init__(
        self,
        frames: list[np.ndarray],
        *,
        opened: bool = True,
        fail_at: int | None = None,
    ):
        self.frames = [frame.copy() for frame in frames]
        self.opened = opened
        self.fail_at = fail_at
        self.position = 0
        self.read_calls = 0
        self.set_calls: list[tuple[int, float]] = []
        self.released = False

    def isOpened(self) -> bool:
        return self.opened

    def read(self) -> tuple[bool, np.ndarray | None]:
        index = self.position
        self.position += 1
        self.read_calls += 1
        if index == self.fail_at or index >= len(self.frames):
            return False, None
        return True, self.frames[index].copy()

    def set(self, property_id: int, value: float) -> bool:
        self.set_calls.append((property_id, value))
        self.position = int(value)
        return True

    def release(self) -> None:
        self.released = True


class ForwardVideoSamplingTests(unittest.TestCase):
    @staticmethod
    def _info(*, frame_count: int = 5, fps: float = 10.0) -> VideoInfo:
        return VideoInfo(
            frame_count=frame_count,
            fps=fps,
            width=4,
            height=2,
            last_frame_time_s=(frame_count - 1) / fps,
        )

    def test_single_forward_decode_preserves_order_repeats_and_letterbox(
        self,
    ) -> None:
        frames = [
            np.full((2, 4, 3), 10, np.uint8),
            np.full((2, 4, 3), 20, np.uint8),
            np.full((3, 3, 3), 30, np.uint8),
            np.full((4, 2, 3), 40, np.uint8),
            np.full((2, 4, 3), 50, np.uint8),
        ]
        capture = FakeCapture(frames)
        times = [0.3, 0.1, 0.1, 0.2]
        with (
            patch.object(media, "probe_video", return_value=self._info()),
            patch.object(media.cv2, "VideoCapture", return_value=capture),
        ):
            sampled = sample_video(
                Path("fake.mp4"),
                sample_times_s=times,
                width=6,
                height=6,
                pad_value=7,
                decode_policy="sequential_forward",
            )

        self.assertEqual([3, 1, 1, 2], sampled.source_indices)
        self.assertEqual(times, sampled.sample_times_s)
        self.assertEqual(4, capture.read_calls)
        self.assertEqual([], capture.set_calls)
        self.assertTrue(capture.released)
        self.assertIsNot(sampled.frames[1], sampled.frames[2])
        expected_indices = [3, 1, 1, 2]
        for actual, source_index in zip(sampled.frames, expected_indices):
            expected, _ = media._letterbox(
                frames[source_index],
                width=6,
                height=6,
                pad_value=7,
            )
            np.testing.assert_array_equal(expected, actual)
        _, first_transform = media._letterbox(
            frames[3],
            width=6,
            height=6,
            pad_value=7,
        )
        self.assertEqual(first_transform, sampled.spatial_transform)

    def test_forward_decode_failure_keeps_protocol_error_semantics(
        self,
    ) -> None:
        capture = FakeCapture(
            [np.zeros((2, 4, 3), np.uint8) for _ in range(5)],
            fail_at=1,
        )
        with (
            patch.object(media, "probe_video", return_value=self._info()),
            patch.object(media.cv2, "VideoCapture", return_value=capture),
            self.assertRaises(VideoProtocolError) as raised,
        ):
            sample_video(
                Path("broken.mp4"),
                sample_times_s=[0.3],
                width=4,
                height=2,
                decode_policy="sequential_forward",
            )

        self.assertEqual("video_decode_failed", raised.exception.code)
        self.assertIn("frame 3", str(raised.exception))
        self.assertEqual([], capture.set_calls)
        self.assertTrue(capture.released)

    def test_legacy_seek_does_not_decode_unrequested_broken_frame(
        self,
    ) -> None:
        frames = [
            np.full((2, 4, 3), value, np.uint8)
            for value in (10, 20, 30, 40, 50)
        ]
        capture = FakeCapture(frames, fail_at=1)
        with (
            patch.object(media, "probe_video", return_value=self._info()),
            patch.object(media.cv2, "VideoCapture", return_value=capture),
        ):
            sampled = sample_video(
                Path("legacy-broken-middle.mp4"),
                sample_times_s=[0.3],
                width=4,
                height=2,
                decode_policy="legacy_random_seek",
            )

        self.assertEqual([3], sampled.source_indices)
        self.assertEqual(
            [(media.cv2.CAP_PROP_POS_FRAMES, 3)],
            capture.set_calls,
        )
        np.testing.assert_array_equal(frames[3], sampled.frames[0])

    def test_preflight_errors_happen_before_decode(self) -> None:
        capture_factory_calls: list[str] = []

        def capture_factory(path: str) -> FakeCapture:
            capture_factory_calls.append(path)
            return FakeCapture([])

        with (
            patch.object(media, "probe_video", return_value=self._info()),
            patch.object(media.cv2, "VideoCapture", side_effect=capture_factory),
            self.assertRaises(VideoProtocolError) as raised,
        ):
            sample_video(
                Path("short.mp4"),
                sample_times_s=[0.5],
                width=4,
                height=2,
            )
        self.assertEqual("insufficient_duration", raised.exception.code)
        self.assertEqual([], capture_factory_calls)

        with (
            patch.object(media, "probe_video", return_value=self._info()),
            patch.object(media.cv2, "VideoCapture", side_effect=capture_factory),
            self.assertRaises(VideoProtocolError) as raised,
        ):
            sample_video(
                Path("negative.mp4"),
                sample_times_s=[-0.1],
                width=4,
                height=2,
                decode_policy="sequential_forward",
            )
        self.assertEqual("invalid_timeline", raised.exception.code)
        self.assertEqual([], capture_factory_calls)

        with self.assertRaises(VideoProtocolError) as raised:
            sample_video(
                Path("empty.mp4"),
                sample_times_s=[],
                width=4,
                height=2,
            )
        self.assertEqual("empty_timeline", raised.exception.code)

        low_fps = self._info(fps=0.5)
        with (
            patch.object(media, "probe_video", return_value=low_fps),
            patch.object(media.cv2, "VideoCapture", side_effect=capture_factory),
            self.assertRaises(VideoProtocolError) as raised,
        ):
            sample_video(
                Path("low-fps.mp4"),
                sample_times_s=[0.0],
                width=4,
                height=2,
                min_source_fps=1.0,
            )
        self.assertEqual("source_fps_too_low", raised.exception.code)
        self.assertEqual([], capture_factory_calls)

    def test_decode_capture_open_failure_keeps_error_code(self) -> None:
        capture = FakeCapture([], opened=False)
        with (
            patch.object(media, "probe_video", return_value=self._info()),
            patch.object(media.cv2, "VideoCapture", return_value=capture),
            self.assertRaises(VideoProtocolError) as raised,
        ):
            sample_video(
                Path("cannot-open.mp4"),
                sample_times_s=[0.0],
                width=4,
                height=2,
            )
        self.assertEqual("video_open_failed", raised.exception.code)

    def test_legacy_policy_preserves_random_seek_order_and_repeats(self) -> None:
        frames = [
            np.full((2, 4, 3), value, np.uint8)
            for value in (10, 20, 30, 40, 50)
        ]
        capture = FakeCapture(frames)
        times = [0.3, 0.1, 0.1, 0.2]
        with (
            patch.object(media, "probe_video", return_value=self._info()),
            patch.object(media.cv2, "VideoCapture", return_value=capture),
        ):
            sampled = sample_video(
                Path("legacy.mp4"),
                sample_times_s=times,
                width=4,
                height=2,
                decode_policy="legacy_random_seek",
            )

        self.assertEqual([3, 1, 1, 2], sampled.source_indices)
        self.assertEqual(4, capture.read_calls)
        self.assertEqual(
            [
                (media.cv2.CAP_PROP_POS_FRAMES, 3.0),
                (media.cv2.CAP_PROP_POS_FRAMES, 1.0),
                (media.cv2.CAP_PROP_POS_FRAMES, 1.0),
                (media.cv2.CAP_PROP_POS_FRAMES, 2.0),
            ],
            capture.set_calls,
        )
        for actual, source_index in zip(
            sampled.frames,
            sampled.source_indices,
        ):
            np.testing.assert_array_equal(frames[source_index], actual)


if __name__ == "__main__":
    unittest.main()
