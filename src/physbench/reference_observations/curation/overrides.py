from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping


def _finite_tuple(values: Any, *, length: int, field: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != length or not all(isfinite(value) for value in result):
        raise ValueError(f"{field} must contain {length} finite values")
    return result


@dataclass(frozen=True)
class CorrectionPrompt:
    object_id: str
    frame_index: int
    box_xyxy: tuple[float, float, float, float]
    points_xy: tuple[tuple[float, float], ...]
    point_labels: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.object_id or self.frame_index < 0:
            raise ValueError("correction prompt requires an object and nonnegative frame")
        if len(self.box_xyxy) != 4 or not all(isfinite(v) for v in self.box_xyxy):
            raise ValueError("correction prompt box must contain four finite values")
        if len(self.points_xy) != len(self.point_labels) or not self.points_xy:
            raise ValueError("correction prompt points and labels must align")
        if any(len(point) != 2 or not all(isfinite(v) for v in point) for point in self.points_xy):
            raise ValueError("correction prompt points must be finite xy pairs")
        if any(label not in (0, 1) for label in self.point_labels):
            raise ValueError("correction prompt labels must be zero or one")


@dataclass(frozen=True)
class LifecycleOverride:
    object_id: str
    start_index: int
    end_index: int
    state: int

    def __post_init__(self) -> None:
        if not self.object_id or self.start_index < 0 or self.end_index < self.start_index:
            raise ValueError("lifecycle override has an invalid object or range")
        if self.state not in (0, 1, 2, 3):
            raise ValueError("lifecycle override state must be in [0, 3]")


@dataclass(frozen=True)
class CaseOverride:
    schema_version: str
    case_id: str
    anchor_prompts: tuple[CorrectionPrompt, ...]
    corrections: tuple[CorrectionPrompt, ...]
    lifecycle: tuple[LifecycleOverride, ...]
    scene_refinement: str


def _prompt_from_mapping(value: Mapping[str, Any]) -> CorrectionPrompt:
    points = tuple(
        tuple(_finite_tuple(point, length=2, field="points_xy"))
        for point in value["points_xy"]
    )
    return CorrectionPrompt(
        object_id=str(value["object_id"]),
        frame_index=int(value["frame_index"]),
        box_xyxy=tuple(_finite_tuple(value["box_xyxy"], length=4, field="box_xyxy")),
        points_xy=points,
        point_labels=tuple(int(label) for label in value["point_labels"]),
    )


def validate_override(value: Mapping[str, Any]) -> CaseOverride:
    required = {
        "schema_version",
        "case_id",
        "anchor_prompts",
        "corrections",
        "lifecycle",
        "scene_refinement",
    }
    if set(value) != required:
        raise ValueError(f"override fields must be exactly {sorted(required)}")
    anchors = tuple(_prompt_from_mapping(item) for item in value["anchor_prompts"])
    corrections = tuple(_prompt_from_mapping(item) for item in value["corrections"])
    seen: set[tuple[str, int]] = set()
    for prompt in corrections:
        key = (prompt.object_id, prompt.frame_index)
        if key in seen:
            raise ValueError(
                f"duplicate correction prompt for {prompt.object_id} at frame {prompt.frame_index}"
            )
        seen.add(key)
    lifecycle = tuple(
        LifecycleOverride(
            object_id=str(item["object_id"]),
            start_index=int(item["start_index"]),
            end_index=int(item["end_index"]),
            state=int(item["state"]),
        )
        for item in value["lifecycle"]
    )
    return CaseOverride(
        schema_version=str(value["schema_version"]),
        case_id=str(value["case_id"]),
        anchor_prompts=anchors,
        corrections=corrections,
        lifecycle=lifecycle,
        scene_refinement=str(value["scene_refinement"]),
    )
