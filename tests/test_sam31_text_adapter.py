from __future__ import annotations

import unittest

import numpy as np

from physbench.evaluation.common.csti.observation import PromptGroupConfig
from physbench.evaluation.common.masks.sam31_text import (
    Sam31TextVideoSegmenter,
    get_shared_sam31_text_segmenter,
)


class _FakePredictor:
    def __init__(self, *, fail_propagation: bool = False) -> None:
        self.fail_propagation = fail_propagation
        self.requests: list[dict[str, object]] = []
        self.closed_sessions: list[str] = []

    @staticmethod
    def _outputs(
        frame_index: int,
        *,
        height: int = 6,
        width: int = 10,
    ) -> dict[str, object]:
        ids = [7, 3] if frame_index == 0 else [7]
        masks = []
        boxes = []
        probabilities = []
        for object_id in ids:
            mask = np.zeros((height, width), dtype=bool)
            if object_id == 7:
                mask[1:4, 1 + frame_index : 4 + frame_index] = True
                boxes.append([0.1, 0.2, 0.3, 0.5])
                probabilities.append(0.9 - 0.1 * frame_index)
            else:
                mask[2:5, 6:9] = True
                boxes.append([0.6, 0.3, 0.3, 0.5])
                probabilities.append(0.7)
            masks.append(mask)
        return {
            "out_obj_ids": np.asarray(ids, dtype=np.int64),
            "out_binary_masks": np.stack(masks),
            "out_boxes_xywh": np.asarray(boxes, dtype=np.float32),
            "out_probs": np.asarray(probabilities, dtype=np.float32),
        }

    def handle_request(self, request: dict[str, object]) -> dict[str, object]:
        self.requests.append(dict(request))
        request_type = request["type"]
        if request_type == "start_session":
            return {"session_id": f"session-{len(self.requests)}"}
        if request_type == "add_prompt":
            return {"frame_index": 0, "outputs": self._outputs(0)}
        if request_type == "close_session":
            self.closed_sessions.append(str(request["session_id"]))
            return {"is_success": True}
        raise AssertionError(f"unexpected request: {request}")

    def handle_stream_request(self, request: dict[str, object]):
        self.requests.append(dict(request))
        if self.fail_propagation:
            raise RuntimeError("synthetic propagation failure")
        for frame_index in (1, 2):
            yield {
                "frame_index": frame_index,
                "outputs": self._outputs(frame_index),
            }


def _frames() -> tuple[np.ndarray, ...]:
    return tuple(np.zeros((6, 10, 3), dtype=np.uint8) for _ in range(3))


def _group(group_id: str = "ball", text: str = "ball") -> PromptGroupConfig:
    return PromptGroupConfig(
        group_id=group_id,
        text=text,
        entity_classes=(group_id,),
    )


def _config(unique: str = "default") -> dict[str, object]:
    return {
        "checkpoint_path_env": f"VPHYSBENCH_TEST_SAM31_{unique}",
        "checkpoint_sha256": "0" * 64,
        "device": "auto",
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


class Sam31TextVideoAdapterTest(unittest.TestCase):
    def test_uses_text_only_frame_zero_prompt_and_forward_propagation(self) -> None:
        predictor = _FakePredictor()
        segmenter = Sam31TextVideoSegmenter(
            _config(), predictor_factory=lambda: predictor
        )

        segmenter.segment(_frames(), (_group(),))

        add_prompt = next(
            request
            for request in predictor.requests
            if request["type"] == "add_prompt"
        )
        self.assertEqual(
            {
                "type",
                "session_id",
                "frame_index",
                "text",
                "output_prob_thresh",
            },
            set(add_prompt),
        )
        self.assertEqual(0, add_prompt["frame_index"])
        self.assertEqual("ball", add_prompt["text"])
        propagate = next(
            request
            for request in predictor.requests
            if request["type"] == "propagate_in_video"
        )
        self.assertEqual("forward", propagate["propagation_direction"])
        self.assertEqual(0, propagate["start_frame_index"])

    def test_preserves_frame_zero_candidates_masks_boxes_and_confidences(self) -> None:
        predictor = _FakePredictor()
        segmenter = Sam31TextVideoSegmenter(
            _config(), predictor_factory=lambda: predictor
        )

        result = segmenter.segment(_frames(), (_group(),))

        self.assertEqual(("ball:3", "ball:7"), tuple(x.candidate_id for x in result))
        by_id = {candidate.candidate_id: candidate for candidate in result}
        self.assertEqual((3, 6, 10), by_id["ball:7"].masks.shape)
        self.assertEqual((3, 4), by_id["ball:7"].boxes_xywh.shape)
        self.assertAlmostEqual(0.9, float(by_id["ball:7"].confidences[0]), places=6)
        self.assertFalse(np.any(by_id["ball:3"].masks[1:]))
        np.testing.assert_allclose(
            [0.7, 0.0, 0.0],
            by_id["ball:3"].confidences,
            rtol=0.0,
            atol=1e-6,
        )

    def test_closes_session_after_success(self) -> None:
        predictor = _FakePredictor()
        segmenter = Sam31TextVideoSegmenter(
            _config(), predictor_factory=lambda: predictor
        )

        segmenter.segment(_frames(), (_group(),))

        self.assertEqual(1, len(predictor.closed_sessions))
        self.assertEqual("close_session", predictor.requests[-1]["type"])

    def test_closes_session_when_propagation_raises(self) -> None:
        predictor = _FakePredictor(fail_propagation=True)
        segmenter = Sam31TextVideoSegmenter(
            _config(), predictor_factory=lambda: predictor
        )

        with self.assertRaisesRegex(RuntimeError, "synthetic propagation"):
            segmenter.segment(_frames(), (_group(),))

        self.assertEqual(1, len(predictor.closed_sessions))
        self.assertEqual("close_session", predictor.requests[-1]["type"])

    def test_distinct_text_groups_use_namespaced_object_ids(self) -> None:
        predictor = _FakePredictor()
        segmenter = Sam31TextVideoSegmenter(
            _config(), predictor_factory=lambda: predictor
        )

        result = segmenter.segment(
            _frames(),
            (_group("ball", "ball"), _group("block", "sliding block")),
        )

        self.assertEqual(4, len(result))
        self.assertEqual(
            {"ball:3", "ball:7", "block:3", "block:7"},
            {candidate.candidate_id for candidate in result},
        )
        self.assertEqual(2, len(predictor.closed_sessions))

    def test_shared_factory_reuses_one_segmenter_and_predictor(self) -> None:
        loads = 0
        predictor = _FakePredictor()

        def factory():
            nonlocal loads
            loads += 1
            return predictor

        config = _config("shared_factory")
        first = get_shared_sam31_text_segmenter(
            config, predictor_factory=factory
        )
        second = get_shared_sam31_text_segmenter(
            config, predictor_factory=factory
        )
        first.segment(_frames(), (_group(),))
        second.segment(_frames(), (_group(),))

        self.assertIs(first, second)
        self.assertEqual(1, loads)


if __name__ == "__main__":
    unittest.main()
