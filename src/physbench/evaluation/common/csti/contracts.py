"""In-memory contracts for CSTI trajectory evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np

from physbench.evaluation.common.entities import ReferenceCapability


class CSTIContractError(ValueError):
    """Raised when CSTI input violates the metric contract."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class CSTIConfig:
    enabled: bool
    algorithm: str
    spatial_tolerance_fraction: float
    temporal_tolerance_s: float
    condition_frame_policy: str
    initial_frames_excluded: int
    score_aggregation: str
    diagnostic_prefix_fractions: tuple[float, ...]
    case_aggregation: str
    timeline_policy: str
    mask_resolution: str

    def __post_init__(self) -> None:
        if self.enabled is not True:
            raise CSTIContractError(
                "csti_config_disabled",
                "CSTI configuration must be enabled",
            )
        fixed_values = {
            "algorithm": "exact_full_tube_edt",
            "condition_frame_policy": "exclude_initial_samples",
            "score_aggregation": "full_tube",
            "case_aggregation": "mean_gt_entities",
            "timeline_policy": "physical_overlap",
            "mask_resolution": "scene_analysis_native",
        }
        for key, expected in fixed_values.items():
            actual = getattr(self, key)
            if actual != expected:
                raise CSTIContractError(
                    "csti_config_value_unsupported",
                    f"Unsupported CSTI {key}: expected {expected!r}, got {actual!r}",
                )
        object.__setattr__(
            self,
            "spatial_tolerance_fraction",
            _positive_finite_float(
                self.spatial_tolerance_fraction,
                "spatial_tolerance_fraction",
            ),
        )
        object.__setattr__(
            self,
            "temporal_tolerance_s",
            _positive_finite_float(
                self.temporal_tolerance_s,
                "temporal_tolerance_s",
            ),
        )
        if (
            isinstance(self.initial_frames_excluded, bool)
            or not isinstance(self.initial_frames_excluded, int)
            or self.initial_frames_excluded < 1
        ):
            raise CSTIContractError(
                "csti_config_integer_invalid",
                "CSTI initial_frames_excluded must be a positive integer",
            )
        expected_fractions = (0.25, 0.5, 0.75, 1.0)
        if self.diagnostic_prefix_fractions != expected_fractions:
            raise CSTIContractError(
                "csti_config_value_unsupported",
                "Unsupported CSTI diagnostic_prefix_fractions: "
                f"expected {expected_fractions!r}, got "
                f"{self.diagnostic_prefix_fractions!r}",
            )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CSTIConfig":
        if not isinstance(value, Mapping):
            raise CSTIContractError("csti_config_not_mapping", "CSTI configuration must be a mapping")

        required = {
            "enabled",
            "algorithm",
            "spatial_tolerance_fraction",
            "temporal_tolerance_s",
            "condition_frame_policy",
            "initial_frames_excluded",
            "score_aggregation",
            "diagnostic_prefix_fractions",
            "case_aggregation",
            "timeline_policy",
            "mask_resolution",
        }
        actual = set(value)
        if actual != required:
            missing = sorted(required - actual)
            extra = sorted(actual - required)
            raise CSTIContractError(
                "csti_config_keys_invalid",
                f"CSTI configuration keys are invalid: missing={missing}, extra={extra}",
            )

        if value["enabled"] is not True:
            raise CSTIContractError("csti_config_disabled", "CSTI configuration must be enabled")

        fixed_values = {
            "algorithm": "exact_full_tube_edt",
            "condition_frame_policy": "exclude_initial_samples",
            "score_aggregation": "full_tube",
            "case_aggregation": "mean_gt_entities",
            "timeline_policy": "physical_overlap",
            "mask_resolution": "scene_analysis_native",
        }
        for key, expected in fixed_values.items():
            if value[key] != expected:
                raise CSTIContractError(
                    "csti_config_value_unsupported",
                    f"Unsupported CSTI {key}: expected {expected!r}, got {value[key]!r}",
                )

        fractions = value["diagnostic_prefix_fractions"]
        if not isinstance(fractions, list):
            raise CSTIContractError(
                "csti_config_value_unsupported",
                "CSTI diagnostic_prefix_fractions must be a JSON array",
            )
        return cls(
            enabled=True,
            algorithm="exact_full_tube_edt",
            spatial_tolerance_fraction=_positive_finite_float(
                value["spatial_tolerance_fraction"],
                "spatial_tolerance_fraction",
            ),
            temporal_tolerance_s=_positive_finite_float(
                value["temporal_tolerance_s"],
                "temporal_tolerance_s",
            ),
            condition_frame_policy="exclude_initial_samples",
            initial_frames_excluded=value["initial_frames_excluded"],
            score_aggregation="full_tube",
            diagnostic_prefix_fractions=tuple(fractions),
            case_aggregation="mean_gt_entities",
            timeline_policy="physical_overlap",
            mask_resolution="scene_analysis_native",
        )


def _positive_finite_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CSTIContractError(
            "csti_config_number_invalid",
            f"CSTI {field_name} must be a finite positive number",
        )
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise CSTIContractError(
            "csti_config_number_invalid",
            f"CSTI {field_name} must be a finite positive number",
        )
    return result


@dataclass(frozen=True)
class CSTIEntityTube:
    entity_id: str
    role_id: str
    reference_masks: tuple[np.ndarray, ...]
    prediction_masks: tuple[np.ndarray, ...] | None
    matched_prediction_track_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class CSTIInput:
    reference_capability: ReferenceCapability
    times_s: tuple[float, ...]
    frame_shape: tuple[int, int]
    entities: tuple[CSTIEntityTube, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.reference_capability, ReferenceCapability):
            raise CSTIContractError(
                "csti_reference_capability_invalid",
                "CSTI reference capability must be a ReferenceCapability",
            )
