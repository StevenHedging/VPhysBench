from __future__ import annotations

import unittest

import numpy as np


class _FakePredictor:
    def __init__(self, *, fail_propagation: bool = False) -> None:
        self.fail_propagation = fail_propagation
        self.requests: list[dict[str, object]] = []
        self.session_objects: dict[str, list[int]] = {}
        self.session_counter = 0

    @staticmethod
    def _outputs(ids: list[int], frame_index: int) -> dict[str, np.ndarray]:
        masks = []
        boxes = []
        probabilities = []
        for object_id in ids:
            mask = np.zeros((8, 12), dtype=bool)
            left = object_id + frame_index
            mask[2:6, left : left + 5] = True
            masks.append(mask)
            boxes.append([left / 12, 2 / 8, 5 / 12, 4 / 8])
            probabilities.append(0.95 - 0.05 * frame_index)
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
            self.session_counter += 1
            session_id = f"session-{self.session_counter}"
            self.session_objects[session_id] = []
            return {"session_id": session_id}
        if request_type == "add_prompt":
            session_id = str(request["session_id"])
            if "text" in request:
                ids = [5, 8]
                self.session_objects[session_id] = ids
            else:
                object_id = int(request["obj_id"])
                if object_id not in self.session_objects[session_id]:
                    self.session_objects[session_id].append(object_id)
                ids = self.session_objects[session_id]
            return {"frame_index": 0, "outputs": self._outputs(ids, 0)}
        if request_type == "close_session":
            return {"is_success": True}
        raise AssertionError(f"unexpected request {request}")

    def handle_stream_request(self, request: dict[str, object]):
        self.requests.append(dict(request))
        if self.fail_propagation:
            raise RuntimeError("synthetic point propagation failure")
        ids = self.session_objects[str(request["session_id"])]
        for frame_index in (1, 2):
            yield {
                "frame_index": frame_index,
                "outputs": self._outputs(ids, frame_index),
            }


def _frames() -> tuple[np.ndarray, ...]:
    return tuple(np.zeros((8, 12, 3), dtype=np.uint8) for _ in range(3))


def _config() -> dict[str, object]:
    return {
        "checkpoint_path_env": "VPHYSBENCH_TEST_SAM31_GT",
        "checkpoint_sha256": "0" * 64,
        "device": "auto",
        "precision": "bfloat16",
        "output_probability_threshold": 0.3,
        "max_num_objects": 8,
        "multiplex_count": 8,
        "compile": False,
        "warm_up": False,
        "use_fa3": False,
        "use_rope_real": True,
        "async_loading_frames": False,
    }


class Sam31GtPredictorTests(unittest.TestCase):
    @staticmethod
    def _api():
        from physbench.reference_observations.curation import sam31_predictor

        return sam31_predictor

    def test_discovery_uses_frame_zero_text_and_closes_session(self) -> None:
        api = self._api()
        predictor = _FakePredictor()
        adapter = api.Sam31GtPredictor(
            _config(), predictor_factory=lambda: predictor
        )

        result = adapter.discover(_frames(), prompts=("small round object",))

        self.assertEqual(2, len(result))
        add_prompt = next(
            request for request in predictor.requests if request["type"] == "add_prompt"
        )
        self.assertEqual(0, add_prompt["frame_index"])
        self.assertEqual("small round object", add_prompt["text"])
        self.assertNotIn("points", add_prompt)
        self.assertEqual("close_session", predictor.requests[-1]["type"])

    def test_point_tracking_locks_explicit_ids_and_propagates_forward(self) -> None:
        api = self._api()
        predictor = _FakePredictor()
        adapter = api.Sam31GtPredictor(
            _config(), predictor_factory=lambda: predictor
        )
        seeds = (
            api.Sam31PointSeed("object_1", 1, (2.0, 4.0)),
            api.Sam31PointSeed("object_2", 2, (7.0, 4.0)),
        )

        result = adapter.track_points(_frames(), seeds)

        self.assertEqual(("object_1", "object_2"), tuple(result))
        point_requests = [
            request
            for request in predictor.requests
            if request["type"] == "add_prompt"
        ]
        self.assertEqual([1, 2], [request["obj_id"] for request in point_requests])
        self.assertTrue(all(request["rel_coordinates"] is False for request in point_requests))
        self.assertEqual(2, predictor.session_counter)
        self.assertEqual([[1, 0], [1, 0]], [request["point_labels"] for request in point_requests])
        self.assertNotEqual(
            point_requests[0]["session_id"],
            point_requests[1]["session_id"],
        )
        propagate = next(
            request
            for request in predictor.requests
            if request["type"] == "propagate_in_video"
        )
        self.assertEqual("forward", propagate["propagation_direction"])
        self.assertEqual(0, propagate["start_frame_index"])

    def test_point_tracking_keeps_overlapping_masks_independent(self) -> None:
        api = self._api()
        adapter = api.Sam31GtPredictor(
            _config(), predictor_factory=lambda: _FakePredictor()
        )
        seeds = (
            api.Sam31PointSeed("object_1", 1, (2.0, 4.0)),
            api.Sam31PointSeed("object_2", 2, (7.0, 4.0)),
        )

        result = adapter.track_points(_frames(), seeds)

        self.assertTrue(np.logical_and(result["object_1"].masks, result["object_2"].masks).any())

    def test_point_tracking_closes_session_when_propagation_raises(self) -> None:
        api = self._api()
        predictor = _FakePredictor(fail_propagation=True)
        adapter = api.Sam31GtPredictor(
            _config(), predictor_factory=lambda: predictor
        )

        with self.assertRaisesRegex(RuntimeError, "synthetic point propagation"):
            adapter.track_points(
                _frames(),
                (api.Sam31PointSeed("object_1", 1, (2.0, 4.0)),),
            )

        self.assertEqual("close_session", predictor.requests[-1]["type"])

    def test_point_tracking_rejects_duplicate_semantic_or_backend_ids(self) -> None:
        api = self._api()
        adapter = api.Sam31GtPredictor(
            _config(), predictor_factory=lambda: _FakePredictor()
        )

        with self.assertRaisesRegex(ValueError, "unique"):
            adapter.track_points(
                _frames(),
                (
                    api.Sam31PointSeed("object_1", 1, (2.0, 4.0)),
                    api.Sam31PointSeed("object_1", 2, (7.0, 4.0)),
                ),
            )

    def test_box_tracking_uses_independent_text_box_sessions_and_locks_best_mask(self) -> None:
        api = self._api()
        predictor = _FakePredictor()
        adapter = api.Sam31GtPredictor(
            _config(), predictor_factory=lambda: predictor
        )
        outputs = predictor._outputs([5, 8], 0)
        seeds = (
            api.Sam31BoxSeed(
                semantic_id="object_1",
                text="small round object",
                reference_mask=outputs["out_binary_masks"][0],
                box_xywh=(0.35, 0.2, 0.45, 0.6),
            ),
            api.Sam31BoxSeed(
                semantic_id="object_2",
                text="small round object",
                reference_mask=outputs["out_binary_masks"][1],
                box_xywh=(0.65, 0.2, 0.35, 0.6),
            ),
        )

        result = adapter.track_boxes(_frames(), seeds, initial_iou_threshold=0.5)

        self.assertEqual((5, 8), tuple(track.backend_object_id for track in result.values()))
        requests = [request for request in predictor.requests if request["type"] == "add_prompt"]
        self.assertEqual(2, len(requests))
        self.assertTrue(all(request["text"] == "small round object" for request in requests))
        self.assertEqual([[1, 0], [1, 0]], [request["bounding_box_labels"] for request in requests])
        self.assertEqual(2, predictor.session_counter)


if __name__ == "__main__":
    unittest.main()
