"""Instance-local compatibility for the pinned SAM3 multiplex tracker."""

from __future__ import annotations

from functools import wraps
import inspect
from typing import Any


BIRTH_CONDITIONING_POLICY = "new_object_conditioning_v1"


def install_birth_conditioning_fix(predictor: Any) -> bool:
    """Keep every object's birth as conditioning, without editing vendor code.

    Return False for predictors without the pinned tracker interface (e.g. test
    factories). A present but incompatible method fails explicitly. Calls must
    be serialized by the caller; the video adapter already holds its RLock.
    """
    model = getattr(predictor, "model", None)
    tracker = getattr(getattr(model, "tracker", None), "model", None)
    method = getattr(tracker, "add_new_masks", None)
    if method is None or not hasattr(tracker, "add_all_frames_to_correct_as_cond"):
        return False
    if getattr(method, "_physbench_birth_conditioning_policy", None) == BIRTH_CONDITIONING_POLICY:
        return True
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("SAM3 add_new_masks signature is unavailable") from exc
    required = {"inference_state", "obj_ids", "reconditioning"}
    if not required.issubset(signature.parameters):
        raise RuntimeError("SAM3 add_new_masks lacks the pinned birth-conditioning interface")

    @wraps(method)
    def add_new_masks(*args: Any, **kwargs: Any) -> Any:
        arguments = signature.bind(*args, **kwargs)
        arguments.apply_defaults()
        state = arguments.arguments["inference_state"]
        obj_ids = arguments.arguments["obj_ids"]
        new_ids = any(obj_id not in state["obj_id_to_idx"] for obj_id in obj_ids)
        force_conditioning = new_ids and not arguments.arguments["reconditioning"]
        if not force_conditioning:
            return method(*args, **kwargs)
        previous = tracker.add_all_frames_to_correct_as_cond
        try:
            tracker.add_all_frames_to_correct_as_cond = True
            return method(*args, **kwargs)
        finally:
            tracker.add_all_frames_to_correct_as_cond = previous

    add_new_masks._physbench_birth_conditioning_policy = BIRTH_CONDITIONING_POLICY
    tracker.add_new_masks = add_new_masks
    return True
