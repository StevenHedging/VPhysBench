from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from physbench.evaluation.common.csti.observation import PromptGroupConfig
from physbench.evaluation.common.errors import SceneAnalysisError
from physbench.evaluation.common.masks.sam31_text import (
    Sam31TextVideoSegmenter,
    get_shared_sam31_text_segmenter,
)


class _FakePredictor:
    def __init__(
        self,
        *,
        fail_start_once: bool = False,
        fail_prompt_once: bool = False,
        fail_propagation_once: bool = False,
        fail_propagation: bool = False,
        fail_close_once: bool = False,
        model: object | None = None,
    ) -> None:
        self.fail_start_once = fail_start_once
        self.fail_prompt_once = fail_prompt_once
        self.fail_propagation_once = fail_propagation_once
        self.fail_propagation = fail_propagation
        self.fail_close_once = fail_close_once
        self.requests: list[dict[str, object]] = []
        self.backend_policy_at_request: list[tuple[str, float, float, bool]] = []
        self.discovery_policy_at_request: list[tuple[str, int, bool]] = []
        self.closed_sessions: list[str] = []
        if model is not None:
            self.model = model

    def _record_backend_policy(self, request_type: object) -> None:
        model = getattr(self, "model", None)
        if all(
            hasattr(model, name)
            for name in (
                "score_threshold_detection",
                "new_det_thresh",
                "masklet_confirmation_enable",
            )
        ):
            self.backend_policy_at_request.append(
                (
                    str(request_type),
                    model.score_threshold_detection,
                    model.new_det_thresh,
                    model.masklet_confirmation_enable,
                )
            )
        if all(
            hasattr(model, name)
            for name in (
                "hotstart_delay",
                "suppress_unmatched_only_within_hotstart",
            )
        ):
            self.discovery_policy_at_request.append(
                (
                    str(request_type),
                    model.hotstart_delay,
                    model.suppress_unmatched_only_within_hotstart,
                )
            )

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
        self._record_backend_policy(request_type)
        if request_type == "start_session":
            if self.fail_start_once:
                self.fail_start_once = False
                raise RuntimeError("synthetic start failure")
            return {"session_id": f"session-{len(self.requests)}"}
        if request_type == "add_prompt":
            if self.fail_prompt_once:
                self.fail_prompt_once = False
                raise RuntimeError("synthetic prompt failure")
            return {"frame_index": 0, "outputs": self._outputs(0)}
        if request_type == "close_session":
            self.closed_sessions.append(str(request["session_id"]))
            if self.fail_close_once:
                self.fail_close_once = False
                raise RuntimeError("synthetic close failure")
            return {"is_success": True}
        raise AssertionError(f"unexpected request: {request}")

    def handle_stream_request(self, request: dict[str, object]):
        self.requests.append(dict(request))
        self._record_backend_policy(request["type"])
        if self.fail_propagation or self.fail_propagation_once:
            self.fail_propagation_once = False
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


def _backend_model(
    *,
    score_threshold: float = 0.5,
    new_object_threshold: float = 0.6,
    confirmation_enabled: bool = True,
) -> SimpleNamespace:
    return SimpleNamespace(
        score_threshold_detection=score_threshold,
        new_det_thresh=new_object_threshold,
        masklet_confirmation_enable=confirmation_enabled,
        hotstart_delay=4,
        suppress_unmatched_only_within_hotstart=False,
    )


class Sam31TextVideoAdapterTest(unittest.TestCase):
    def test_fixed_initial_discovery_policy_spans_session_and_is_described(
        self,
    ) -> None:
        model = _backend_model()
        predictor = _FakePredictor(model=model)
        config = _config()
        config["discovery_pruning_policy"] = "fixed_initial_ids_v1"
        segmenter = Sam31TextVideoSegmenter(
            config, predictor_factory=lambda: predictor
        )

        segmenter.segment(_frames(), (_group(),))
        segmenter.segment(_frames(), (_group(),))

        self.assertEqual(
            2
            * [
                ("start_session", 0, True),
                ("add_prompt", 0, True),
                ("propagate_in_video", 0, True),
                ("close_session", 0, True),
            ],
            predictor.discovery_policy_at_request,
        )
        self.assertEqual(4, model.hotstart_delay)
        self.assertFalse(model.suppress_unmatched_only_within_hotstart)
        self.assertEqual(
            {
                "policy": "fixed_initial_ids_v1",
                "requested": {
                    "hotstart_delay": 0,
                    "suppress_unmatched_only_within_hotstart": True,
                },
                "native": {
                    "hotstart_delay": 4,
                    "suppress_unmatched_only_within_hotstart": False,
                },
                "effective": {
                    "hotstart_delay": 0,
                    "suppress_unmatched_only_within_hotstart": True,
                },
            },
            segmenter.describe()["discovery_pruning"],
        )
        self.assertEqual(2, segmenter.describe()["segmenter_policy_revision"])

    def test_backend_native_discovery_policy_is_legacy_and_requires_no_interface(
        self,
    ) -> None:
        config = _config()
        config["discovery_pruning_policy"] = "backend_native_v1"
        segmenter = Sam31TextVideoSegmenter(
            config, predictor_factory=lambda: _FakePredictor()
        )

        segmenter.segment(_frames(), (_group(),))

        self.assertEqual(1, segmenter.describe()["segmenter_policy_revision"])
        self.assertEqual(
            {
                "policy": "backend_native_v1",
                "requested": None,
                "native": None,
                "effective": None,
            },
            segmenter.describe()["discovery_pruning"],
        )

    def test_legacy_omission_retains_native_discovery_pruning_values(self) -> None:
        model = _backend_model()
        predictor = _FakePredictor(model=model)
        segmenter = Sam31TextVideoSegmenter(
            _config(), predictor_factory=lambda: predictor
        )

        segmenter.segment(_frames(), (_group(),))

        self.assertTrue(
            all(
                item[1:] == (4, False)
                for item in predictor.discovery_policy_at_request
            )
        )
        native = {
            "hotstart_delay": 4,
            "suppress_unmatched_only_within_hotstart": False,
        }
        description = segmenter.describe()["discovery_pruning"]
        self.assertEqual("backend_native_v1", description["policy"])
        self.assertIsNone(description["requested"])
        self.assertEqual(native, description["native"])
        self.assertEqual(native, description["effective"])

    def test_rejects_invalid_discovery_pruning_policy(self) -> None:
        for invalid in (
            None,
            False,
            0,
            [],
            {},
            "fixed_initial_ids_v2",
            "",
        ):
            with self.subTest(invalid=invalid):
                config = _config()
                config["discovery_pruning_policy"] = invalid
                with self.assertRaisesRegex(ValueError, "discovery_pruning_policy"):
                    Sam31TextVideoSegmenter(config)

    def test_fixed_initial_discovery_policy_requires_compatible_backend(self) -> None:
        invalid_models = (
            SimpleNamespace(),
            SimpleNamespace(
                hotstart_delay=True,
                suppress_unmatched_only_within_hotstart=False,
            ),
            SimpleNamespace(
                hotstart_delay=4,
                suppress_unmatched_only_within_hotstart=0,
            ),
        )
        for model in invalid_models:
            with self.subTest(model=model):
                config = _config()
                config["discovery_pruning_policy"] = "fixed_initial_ids_v1"
                segmenter = Sam31TextVideoSegmenter(
                    config,
                    predictor_factory=lambda model=model: _FakePredictor(
                        model=model
                    ),
                )

                with self.assertRaises(SceneAnalysisError) as caught:
                    segmenter.segment(_frames(), (_group(),))

                self.assertEqual(
                    "sam31_model_interface_incompatible", caught.exception.code
                )

    def test_fixed_initial_discovery_policy_restores_on_every_session_failure(
        self,
    ) -> None:
        failures = (
            ("fail_start_once", "synthetic start failure"),
            ("fail_prompt_once", "synthetic prompt failure"),
            ("fail_propagation_once", "synthetic propagation failure"),
            ("fail_close_once", "synthetic close failure"),
        )
        for flag, message in failures:
            with self.subTest(flag=flag):
                model = _backend_model()
                predictor = _FakePredictor(model=model, **{flag: True})
                config = _config()
                config["discovery_pruning_policy"] = "fixed_initial_ids_v1"
                segmenter = Sam31TextVideoSegmenter(
                    config, predictor_factory=lambda: predictor
                )

                with self.assertRaisesRegex(RuntimeError, message):
                    segmenter.segment(_frames(), (_group(),))

                self.assertEqual(4, model.hotstart_delay)
                self.assertFalse(model.suppress_unmatched_only_within_hotstart)
                self.assertTrue(predictor.discovery_policy_at_request)
                self.assertTrue(
                    all(
                        item[1:] == (0, True)
                        for item in predictor.discovery_policy_at_request
                    )
                )

    def test_discovery_policy_assignment_failure_restores_both_native_values(
        self,
    ) -> None:
        class MutatingSetterModel:
            def __init__(self) -> None:
                self._hotstart_delay = 4
                self._suppress = False
                self.fail_next_suppress = True

            @property
            def hotstart_delay(self) -> int:
                return self._hotstart_delay

            @hotstart_delay.setter
            def hotstart_delay(self, value: int) -> None:
                self._hotstart_delay = value

            @property
            def suppress_unmatched_only_within_hotstart(self) -> bool:
                return self._suppress

            @suppress_unmatched_only_within_hotstart.setter
            def suppress_unmatched_only_within_hotstart(self, value: bool) -> None:
                self._suppress = value
                if value is True and self.fail_next_suppress:
                    self.fail_next_suppress = False
                    raise RuntimeError("synthetic pruning setter failure")

        model = MutatingSetterModel()
        predictor = _FakePredictor(model=model)
        config = _config()
        config["discovery_pruning_policy"] = "fixed_initial_ids_v1"
        segmenter = Sam31TextVideoSegmenter(
            config, predictor_factory=lambda: predictor
        )

        with self.assertRaises(SceneAnalysisError) as caught:
            segmenter.segment(_frames(), (_group(),))

        self.assertEqual("sam31_model_interface_incompatible", caught.exception.code)
        self.assertEqual(4, model.hotstart_delay)
        self.assertFalse(model.suppress_unmatched_only_within_hotstart)

    def test_discovery_policy_nests_with_initial_and_confirmation_without_leaks(
        self,
    ) -> None:
        model = _backend_model()
        predictor = _FakePredictor(model=model)
        config = _config()
        config["initial_detection"] = {
            "score_threshold": 0.2,
            "new_object_threshold": 0.2,
        }
        config["masklet_confirmation_enable"] = False
        config["discovery_pruning_policy"] = "fixed_initial_ids_v1"
        segmenter = Sam31TextVideoSegmenter(
            config, predictor_factory=lambda: predictor
        )

        segmenter.segment(_frames(), (_group(),))

        description = segmenter.describe()
        self.assertTrue(description["masklet_confirmation"]["native"])
        self.assertEqual(
            4,
            description["discovery_pruning"]["native"]["hotstart_delay"],
        )
        self.assertEqual(0.5, description["propagation_detection"]["score_threshold"])
        propagation = next(
            item
            for item in predictor.backend_policy_at_request
            if item[0] == "propagate_in_video"
        )
        self.assertEqual((0.5, 0.6), propagation[1:3])
        self.assertTrue(
            all(item[1:] == (0, True) for item in predictor.discovery_policy_at_request)
        )
        self.assertEqual(4, model.hotstart_delay)
        self.assertFalse(model.suppress_unmatched_only_within_hotstart)
        self.assertTrue(model.masklet_confirmation_enable)

    def test_empty_initial_detection_restores_discovery_policy(self) -> None:
        class EmptyPredictor(_FakePredictor):
            @staticmethod
            def _outputs(frame_index, **kwargs):
                return {
                    "out_obj_ids": [],
                    "out_binary_masks": np.zeros((0, 6, 10)),
                    "out_boxes_xywh": np.zeros((0, 4)),
                    "out_probs": [],
                }

        model = _backend_model()
        predictor = EmptyPredictor(model=model)
        config = _config()
        config["discovery_pruning_policy"] = "fixed_initial_ids_v1"
        segmenter = Sam31TextVideoSegmenter(
            config, predictor_factory=lambda: predictor
        )

        self.assertEqual((), segmenter.segment(_frames(), (_group(),)))

        self.assertEqual(4, model.hotstart_delay)
        self.assertFalse(model.suppress_unmatched_only_within_hotstart)
        self.assertFalse(
            any(r["type"] == "propagate_in_video" for r in predictor.requests)
        )
        self.assertEqual(
            [
                ("start_session", 0, True),
                ("add_prompt", 0, True),
                ("close_session", 0, True),
            ],
            predictor.discovery_policy_at_request,
        )

    def test_late_backend_ids_do_not_join_fixed_initial_candidates(self) -> None:
        class LateBirthPredictor(_FakePredictor):
            @staticmethod
            def _outputs(frame_index, **kwargs):
                result = _FakePredictor._outputs(frame_index, **kwargs)
                if frame_index > 0:
                    late_mask = np.zeros((6, 10), dtype=bool)
                    late_mask[0:2, 7:9] = True
                    result["out_obj_ids"] = np.asarray([7, 99], dtype=np.int64)
                    result["out_binary_masks"] = np.stack(
                        [result["out_binary_masks"][0], late_mask]
                    )
                    result["out_boxes_xywh"] = np.asarray(
                        [result["out_boxes_xywh"][0], [0.7, 0.0, 0.2, 0.2]],
                        dtype=np.float32,
                    )
                    result["out_probs"] = np.asarray(
                        [result["out_probs"][0], 0.8], dtype=np.float32
                    )
                return result

        model = _backend_model()
        predictor = LateBirthPredictor(model=model)
        config = _config()
        config["discovery_pruning_policy"] = "fixed_initial_ids_v1"
        segmenter = Sam31TextVideoSegmenter(
            config, predictor_factory=lambda: predictor
        )

        result = segmenter.segment(_frames(), (_group(),))

        self.assertEqual(("ball:3", "ball:7"), tuple(x.candidate_id for x in result))
        self.assertNotIn("ball:99", {x.candidate_id for x in result})
        self.assertTrue(
            all(item[1:] == (0, True) for item in predictor.discovery_policy_at_request)
        )
    def test_initial_detection_gates_apply_only_to_prompt_and_are_described(
        self,
    ) -> None:
        model = _backend_model()
        predictor = _FakePredictor(model=model)
        config = _config()
        config["initial_detection"] = {
            "score_threshold": 0.2,
            "new_object_threshold": 0.2,
        }
        segmenter = Sam31TextVideoSegmenter(
            config, predictor_factory=lambda: predictor
        )

        segmenter.segment(_frames(), (_group(),))

        self.assertEqual(
            [
                ("start_session", 0.5, 0.6, True),
                ("add_prompt", 0.2, 0.2, True),
                ("propagate_in_video", 0.5, 0.6, True),
                ("close_session", 0.5, 0.6, True),
            ],
            predictor.backend_policy_at_request,
        )
        self.assertEqual(0.5, model.score_threshold_detection)
        self.assertEqual(0.6, model.new_det_thresh)
        description = segmenter.describe()
        self.assertEqual(2, description["segmenter_policy_revision"])
        self.assertNotIn("observer_revision", description)
        self.assertEqual(
            {
                "policy": "frame_zero_override_v1",
                "score_threshold": 0.2,
                "new_object_threshold": 0.2,
            },
            description["initial_detection"],
        )
        self.assertEqual(
            {
                "policy": "backend_native_v1",
                "score_threshold": 0.5,
                "new_object_threshold": 0.6,
            },
            description["propagation_detection"],
        )

    def test_initial_detection_restores_after_prompt_failure_and_reuse(self) -> None:
        model = _backend_model()
        predictor = _FakePredictor(fail_prompt_once=True, model=model)
        config = _config()
        config["initial_detection"] = {
            "score_threshold": 0.2,
            "new_object_threshold": 0.3,
        }
        segmenter = Sam31TextVideoSegmenter(
            config, predictor_factory=lambda: predictor
        )

        with self.assertRaisesRegex(RuntimeError, "synthetic prompt failure"):
            segmenter.segment(_frames(), (_group(),))
        self.assertEqual(0.5, model.score_threshold_detection)
        self.assertEqual(0.6, model.new_det_thresh)

        segmenter.segment(_frames(), (_group(),))

        prompt_states = [
            item
            for item in predictor.backend_policy_at_request
            if item[0] == "add_prompt"
        ]
        self.assertEqual(
            [
                ("add_prompt", 0.2, 0.3, True),
                ("add_prompt", 0.2, 0.3, True),
            ],
            prompt_states,
        )
        self.assertEqual(0.5, model.score_threshold_detection)
        self.assertEqual(0.6, model.new_det_thresh)

    def test_legacy_omission_retains_backend_detection_gates(self) -> None:
        model = _backend_model(score_threshold=0.45, new_object_threshold=0.55)
        predictor = _FakePredictor(model=model)
        segmenter = Sam31TextVideoSegmenter(
            _config(), predictor_factory=lambda: predictor
        )

        segmenter.segment(_frames(), (_group(),))

        self.assertTrue(
            all(
                item[1:3] == (0.45, 0.55)
                for item in predictor.backend_policy_at_request
            )
        )
        self.assertEqual(
            {
                "policy": "backend_native_v1",
                "score_threshold": 0.45,
                "new_object_threshold": 0.55,
            },
            segmenter.describe()["initial_detection"],
        )

    def test_rejects_invalid_initial_detection_mappings(self) -> None:
        invalid_mappings = (
            None,
            {"score_threshold": 0.2},
            {"score_threshold": 0.2, "new_object_threshold": 0.2, "extra": 1},
            {"score_threshold": True, "new_object_threshold": 0.2},
            {"score_threshold": float("nan"), "new_object_threshold": 0.2},
            {"score_threshold": float("inf"), "new_object_threshold": 0.2},
            {"score_threshold": -0.1, "new_object_threshold": 0.2},
            {"score_threshold": 0.2, "new_object_threshold": 1.1},
            {"score_threshold": 0.3, "new_object_threshold": 0.2},
        )
        for initial_detection in invalid_mappings:
            with self.subTest(initial_detection=initial_detection):
                config = _config()
                config["initial_detection"] = initial_detection
                with self.assertRaises(ValueError):
                    Sam31TextVideoSegmenter(config)

    def test_initial_detection_accepts_integer_unit_endpoints_as_floats(self) -> None:
        config = _config()
        config["initial_detection"] = {
            "score_threshold": 0,
            "new_object_threshold": 1,
        }

        description = Sam31TextVideoSegmenter(config).describe()[
            "initial_detection"
        ]

        self.assertEqual(0.0, description["score_threshold"])
        self.assertIsInstance(description["score_threshold"], float)
        self.assertEqual(1.0, description["new_object_threshold"])
        self.assertIsInstance(description["new_object_threshold"], float)

    def test_explicit_initial_detection_requires_backend_gate_interface(self) -> None:
        config = _config()
        config["initial_detection"] = {
            "score_threshold": 0.2,
            "new_object_threshold": 0.2,
        }
        segmenter = Sam31TextVideoSegmenter(
            config, predictor_factory=lambda: _FakePredictor()
        )

        with self.assertRaises(SceneAnalysisError) as caught:
            segmenter.segment(_frames(), (_group(),))

        self.assertEqual("sam31_model_interface_incompatible", caught.exception.code)

    def test_confirmation_policy_spans_session_and_restores_on_reuse(self) -> None:
        model = _backend_model()
        predictor = _FakePredictor(fail_propagation_once=True, model=model)
        config = _config()
        config["masklet_confirmation_enable"] = False
        config["initial_detection"] = {
            "score_threshold": 0.2,
            "new_object_threshold": 0.2,
        }
        segmenter = Sam31TextVideoSegmenter(
            config, predictor_factory=lambda: predictor
        )

        with self.assertRaisesRegex(RuntimeError, "synthetic propagation"):
            segmenter.segment(_frames(), (_group(),))
        self.assertTrue(model.masklet_confirmation_enable)

        segmenter.segment(_frames(), (_group(),))

        self.assertTrue(model.masklet_confirmation_enable)
        self.assertTrue(
            all(item[3] is False for item in predictor.backend_policy_at_request)
        )
        self.assertEqual(
            {
                "policy": "whole_session_override_v1",
                "requested": False,
                "native": True,
                "effective": False,
            },
            segmenter.describe()["masklet_confirmation"],
        )

    def test_confirmation_policy_restores_after_prompt_failure(self) -> None:
        model = _backend_model()
        predictor = _FakePredictor(fail_prompt_once=True, model=model)
        config = _config()
        config["masklet_confirmation_enable"] = False
        segmenter = Sam31TextVideoSegmenter(
            config, predictor_factory=lambda: predictor
        )

        with self.assertRaisesRegex(RuntimeError, "synthetic prompt failure"):
            segmenter.segment(_frames(), (_group(),))

        self.assertTrue(model.masklet_confirmation_enable)
        self.assertTrue(
            all(item[3] is False for item in predictor.backend_policy_at_request)
        )

    def test_confirmation_true_is_supported_and_restored_after_close_failure(
        self,
    ) -> None:
        model = _backend_model(confirmation_enabled=False)
        predictor = _FakePredictor(fail_close_once=True, model=model)
        config = _config()
        config["masklet_confirmation_enable"] = True
        segmenter = Sam31TextVideoSegmenter(
            config, predictor_factory=lambda: predictor
        )

        with self.assertRaisesRegex(RuntimeError, "synthetic close failure"):
            segmenter.segment(_frames(), (_group(),))

        self.assertFalse(model.masklet_confirmation_enable)
        self.assertTrue(
            all(item[3] is True for item in predictor.backend_policy_at_request)
        )

    def test_rejects_non_boolean_confirmation_policy(self) -> None:
        for invalid in (None, 0, 1, "false", "true"):
            with self.subTest(invalid=invalid):
                config = _config()
                config["masklet_confirmation_enable"] = invalid
                with self.assertRaisesRegex(ValueError, "boolean"):
                    Sam31TextVideoSegmenter(config)

    def test_explicit_confirmation_requires_backend_interface(self) -> None:
        config = _config()
        config["masklet_confirmation_enable"] = False
        segmenter = Sam31TextVideoSegmenter(
            config, predictor_factory=lambda: _FakePredictor()
        )

        with self.assertRaises(SceneAnalysisError) as caught:
            segmenter.segment(_frames(), (_group(),))

        self.assertEqual("sam31_model_interface_incompatible", caught.exception.code)

    def test_confirmation_assignment_failure_restores_mutated_backend_value(
        self,
    ) -> None:
        class MutatingSetterModel:
            score_threshold_detection = 0.5
            new_det_thresh = 0.6

            def __init__(self) -> None:
                self._confirmation = True
                self.fail_next_disable = True

            @property
            def masklet_confirmation_enable(self) -> bool:
                return self._confirmation

            @masklet_confirmation_enable.setter
            def masklet_confirmation_enable(self, value: bool) -> None:
                self._confirmation = value
                if value is False and self.fail_next_disable:
                    self.fail_next_disable = False
                    raise RuntimeError("synthetic mutating setter failure")

        model = MutatingSetterModel()
        predictor = _FakePredictor(model=model)
        config = _config()
        config["masklet_confirmation_enable"] = False
        segmenter = Sam31TextVideoSegmenter(
            config, predictor_factory=lambda: predictor
        )

        with self.assertRaises(SceneAnalysisError) as caught:
            segmenter.segment(_frames(), (_group(),))

        self.assertEqual("sam31_model_interface_incompatible", caught.exception.code)
        self.assertTrue(model.masklet_confirmation_enable)

    def test_empty_initial_detection_closes_without_propagating(self) -> None:
        class EmptyPredictor(_FakePredictor):
            @staticmethod
            def _outputs(frame_index, **kwargs):
                return {"out_obj_ids": [], "out_binary_masks": np.zeros((0, 6, 10)),
                        "out_boxes_xywh": np.zeros((0, 4)), "out_probs": []}

        predictor = EmptyPredictor()
        segmenter = Sam31TextVideoSegmenter(_config(), predictor_factory=lambda: predictor)
        self.assertEqual((), segmenter.segment(_frames(), (_group(),)))
        self.assertEqual(1, len(predictor.closed_sessions))
        self.assertFalse(any(r["type"] == "propagate_in_video" for r in predictor.requests))

    def test_compatible_factory_installs_and_reports_birth_policy(self) -> None:
        from tests.test_sam31_compat import BoundaryTracker, predictor_for
        predictor = _FakePredictor()
        tracker = BoundaryTracker()
        predictor.model = predictor_for(tracker).model
        segmenter = Sam31TextVideoSegmenter(_config(), predictor_factory=lambda: predictor)
        segmenter.segment(_frames(), (_group(),))
        self.assertEqual((True, False, False), tracker.add_new_masks(
            {"obj_id_to_idx": {}}, 6, [1], None))
        self.assertEqual("new_object_conditioning_v1", segmenter.describe().get("birth_conditioning_policy"))

    def test_incompatible_present_tracker_cannot_be_reused_after_load_failure(self):
        from types import SimpleNamespace
        from tests.test_sam31_compat import predictor_for
        predictor = _FakePredictor()
        predictor.model = predictor_for(SimpleNamespace(
            add_new_masks=lambda unexpected: None,
            add_all_frames_to_correct_as_cond=False)).model
        segmenter = Sam31TextVideoSegmenter(_config(), predictor_factory=lambda: predictor)
        for _ in range(2):
            with self.assertRaises(SceneAnalysisError) as caught:
                segmenter.segment(_frames(), (_group(),))
            self.assertEqual("sam31_tracker_interface_incompatible", caught.exception.code)

    def test_rejects_precision_the_official_predictor_cannot_honor(self) -> None:
        config = _config()
        config["precision"] = "float32"

        with self.assertRaisesRegex(ValueError, "bfloat16"):
            Sam31TextVideoSegmenter(
                config,
                predictor_factory=lambda: _FakePredictor(),
            )

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
