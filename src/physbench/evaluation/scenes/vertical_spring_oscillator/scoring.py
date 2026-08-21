"""Fail-closed, timeline-preserving scoring for vertical spring motion."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


class SpringTraceError(RuntimeError):
    """A trace cannot safely support physical scoring."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class SpringTrace:
    times_s: np.ndarray
    xy: np.ndarray
    area_px2: np.ndarray
    valid: np.ndarray
    valid_ratio: float
    equilibrium_y_px: float
    amplitude_px: float
    envelope_px: np.ndarray
    period_s: float | None
    horizontal_drift_ratio: float
    release_sign: int

    def __post_init__(self) -> None:
        """Make the public trace model immutable and reject incoherent state."""
        values = _validated_trace_data(self)
        object.__setattr__(self, "times_s", _readonly(values["times_s"]))
        object.__setattr__(self, "xy", _readonly(values["xy"]))
        object.__setattr__(self, "area_px2", _readonly(values["area_px2"]))
        object.__setattr__(self, "valid", _readonly(values["valid"]))
        object.__setattr__(self, "envelope_px", _readonly(values["envelope_px"]))
        for name in (
            "valid_ratio",
            "equilibrium_y_px",
            "amplitude_px",
            "horizontal_drift_ratio",
        ):
            object.__setattr__(self, name, values[name])
        object.__setattr__(self, "period_s", values["period_s"])
        object.__setattr__(self, "release_sign", values["release_sign"])


def theoretical_period_s(*, mass_kg: float, stiffness_n_m: float) -> float:
    """Return the ideal small-amplitude mass-spring period in seconds."""
    if not math.isfinite(mass_kg) or mass_kg <= 0.0:
        raise ValueError("mass_kg must be finite and positive")
    if not math.isfinite(stiffness_n_m) or stiffness_n_m <= 0.0:
        raise ValueError("stiffness_n_m must be finite and positive")
    return float(2.0 * math.pi * math.sqrt(mass_kg / stiffness_n_m))


def _readonly(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values).copy()
    result.flags.writeable = False
    return result


_CADENCE_RELATIVE_TOLERANCE = 0.02
_MINIMUM_PERIOD_CORRELATION = 0.60
_MINIMUM_PERIOD_PROMINENCE = 0.001
_MINIMUM_COMPLETE_CYCLES = 2.0
_MINIMUM_SUBHARMONIC_RESIDUAL_PX = 0.15


def _invalid_trace(message: str) -> SpringTraceError:
    return SpringTraceError("invalid_trace", message)


def _validated_trace_data(trace: SpringTrace) -> dict[str, Any]:
    """Validate every externally constructible SpringTrace field before scoring."""
    try:
        times = np.asarray(trace.times_s, dtype=np.float64)
        xy = np.asarray(trace.xy, dtype=np.float64)
        areas = np.asarray(trace.area_px2, dtype=np.float64)
        valid = np.asarray(trace.valid)
        envelope = np.asarray(trace.envelope_px, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise _invalid_trace("trace arrays must be numeric") from exc
    if times.ndim != 1 or len(times) < 3 or not np.all(np.isfinite(times)):
        raise _invalid_trace("trace times must be a non-empty finite one-dimensional sequence")
    if not np.all(np.diff(times) > 0.0):
        raise SpringTraceError("non_monotonic_times", "trace times must be strictly increasing")
    _require_uniform_cadence(times)
    count = len(times)
    if xy.shape != (count, 2) or not np.all(np.isfinite(xy)):
        raise _invalid_trace("trace xy must be finite with shape (frames, 2)")
    if areas.shape != (count,) or not np.all(np.isfinite(areas)) or np.any(areas < 0.0):
        raise _invalid_trace("trace areas must be finite non-negative frame values")
    if valid.shape != (count,) or valid.dtype != np.bool_:
        raise _invalid_trace("trace valid flags must be a boolean value per frame")
    if int(valid.sum()) < 3:
        raise _invalid_trace("trace requires at least three valid samples")
    if envelope.shape != (count,) or not np.all(np.isfinite(envelope)) or np.any(envelope < 0.0):
        raise _invalid_trace("trace envelope must be finite non-negative frame values")
    try:
        valid_ratio = float(trace.valid_ratio)
        equilibrium = float(trace.equilibrium_y_px)
        amplitude = float(trace.amplitude_px)
        drift = float(trace.horizontal_drift_ratio)
    except (TypeError, ValueError) as exc:
        raise _invalid_trace("trace scalar fields must be numeric") from exc
    if not all(math.isfinite(value) for value in (valid_ratio, equilibrium, amplitude, drift)):
        raise _invalid_trace("trace scalar fields must be finite")
    if not 0.0 <= valid_ratio <= 1.0 or not math.isclose(
        valid_ratio, float(np.mean(valid)), rel_tol=0.0, abs_tol=1e-9
    ):
        raise _invalid_trace("trace valid_ratio must match the valid flags")
    if amplitude < 0.0 or drift < 0.0:
        raise _invalid_trace("trace amplitude and horizontal drift must be non-negative")
    expected_equilibrium = float(np.median(xy[valid, 1]))
    expected_amplitude = float(
        0.5
        * (
            np.quantile(xy[valid, 1] - expected_equilibrium, 0.95)
            - np.quantile(xy[valid, 1] - expected_equilibrium, 0.05)
        )
    )
    expected_envelope = np.abs(xy[:, 1] - expected_equilibrium)
    expected_horizontal_span = float(
        0.5 * (np.quantile(xy[valid, 0], 0.95) - np.quantile(xy[valid, 0], 0.05))
    )
    expected_drift = expected_horizontal_span / max(expected_amplitude, 1e-12)
    if not math.isclose(equilibrium, expected_equilibrium, rel_tol=0.0, abs_tol=1e-6):
        raise _invalid_trace("trace equilibrium is inconsistent with valid positions")
    if not math.isclose(amplitude, expected_amplitude, rel_tol=1e-6, abs_tol=1e-6):
        raise _invalid_trace("trace amplitude is inconsistent with valid positions")
    if not np.allclose(envelope, expected_envelope, rtol=1e-6, atol=1e-6):
        raise _invalid_trace("trace envelope is inconsistent with vertical displacement")
    if not math.isclose(drift, expected_drift, rel_tol=1e-6, abs_tol=1e-6):
        raise _invalid_trace("trace horizontal drift is inconsistent with valid positions")
    release_sign = trace.release_sign
    if isinstance(release_sign, bool) or release_sign not in (-1, 0, 1):
        raise _invalid_trace("trace release_sign must be -1, 0, or 1")
    expected_release = _release_sign(xy[:, 1] - expected_equilibrium, expected_amplitude)
    if release_sign != expected_release:
        raise _invalid_trace("trace release_sign is inconsistent with vertical displacement")
    period = trace.period_s
    if period is not None:
        try:
            period = float(period)
        except (TypeError, ValueError) as exc:
            raise _invalid_trace("trace period must be numeric or None") from exc
        if not math.isfinite(period) or period <= 0.0:
            raise _invalid_trace("trace period must be finite and positive")
    return {
        "times_s": times,
        "xy": xy,
        "area_px2": areas,
        "valid": valid,
        "valid_ratio": valid_ratio,
        "equilibrium_y_px": equilibrium,
        "amplitude_px": amplitude,
        "envelope_px": envelope,
        "period_s": period,
        "horizontal_drift_ratio": drift,
        "release_sign": int(release_sign),
    }


def _require_uniform_cadence(
    times_s: np.ndarray,
    *,
    relative_tolerance: float = _CADENCE_RELATIVE_TOLERANCE,
) -> float:
    if not math.isfinite(relative_tolerance) or relative_tolerance <= 0.0:
        raise ValueError("cadence relative tolerance must be finite and positive")
    steps = np.diff(times_s)
    step_s = float(np.median(steps))
    if not math.isfinite(step_s) or step_s <= 0.0:
        raise SpringTraceError("non_monotonic_times", "time cadence must be positive")
    deviation = float(np.max(np.abs(steps - step_s)) / step_s)
    if deviation > relative_tolerance:
        raise SpringTraceError(
            "irregular_cadence",
            "time cadence exceeds the supported relative deviation",
        )
    return step_s


def _require_quality(quality_config: dict[str, Any], name: str) -> float:
    try:
        value = float(quality_config[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"quality_config requires a finite {name!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"quality_config {name!r} must be finite")
    return value


def _interpolate(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Linearly fill invalid samples after coverage has already been accepted."""
    indices = np.arange(len(values), dtype=float)
    result = values.astype(np.float64, copy=True)
    for column in range(result.shape[1]):
        result[~valid, column] = np.interp(
            indices[~valid], indices[valid], result[valid, column]
        )
    return result


def _configured_period_threshold(
    quality_config: dict[str, Any], name: str, default: float
) -> float:
    value = float(quality_config.get(name, default))
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"quality_config {name!r} must be finite and positive")
    return value


def _configured_scoring_threshold(
    scoring_config: dict[str, Any], name: str, default: float
) -> float:
    try:
        value = float(scoring_config.get(name, default))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"scoring_config {name!r} must be finite and positive") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"scoring_config {name!r} must be finite and positive")
    return value


def _qualified_autocorrelation_period(
    displacement: np.ndarray,
    times_s: np.ndarray,
    *,
    minimum_correlation: float,
    minimum_prominence: float,
    minimum_complete_cycles: float,
    minimum_subharmonic_residual_px: float,
    minimum_period_s: float,
    maximum_period_s: float,
) -> float | None:
    """Return a repeat-supported fundamental candidate, never an ACF fallback."""
    if (
        not math.isfinite(minimum_period_s)
        or minimum_period_s <= 0.0
        or not math.isfinite(maximum_period_s)
        or maximum_period_s < minimum_period_s
    ):
        raise ValueError("period bounds must be finite, positive, and ordered")
    if len(displacement) < 5:
        return None
    centered = displacement - float(np.mean(displacement))
    energy = float(np.dot(centered, centered))
    if energy <= 1e-12:
        return None
    step_s = _require_uniform_cadence(times_s)
    maximum_lag = min(
        len(centered) - 2,
        int(math.floor((times_s[-1] - times_s[0]) / (minimum_complete_cycles * step_s))),
    )
    if maximum_lag < 3:
        return None
    correlations = np.full(maximum_lag + 1, np.nan, dtype=np.float64)
    for lag in range(1, maximum_lag + 1):
        left = centered[:-lag]
        right = centered[lag:]
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        if denominator > 1e-12:
            correlations[lag] = float(np.dot(left, right) / denominator)
    candidates: list[int] = []
    for lag in range(2, maximum_lag):
        correlation = correlations[lag]
        prominence = correlation - max(correlations[lag - 1], correlations[lag + 1])
        if (
            math.isfinite(correlation)
            and correlation >= minimum_correlation
            and prominence >= minimum_prominence
        ):
            candidates.append(lag)
    endpoint_correlation = correlations[maximum_lag]
    endpoint_prominence = (
        endpoint_correlation - correlations[maximum_lag - 1]
    )
    if (
        math.isfinite(endpoint_correlation)
        and endpoint_correlation >= minimum_correlation
        and endpoint_prominence >= minimum_prominence
    ):
        candidates.append(maximum_lag)
    if not candidates:
        return None
    refined = [
        _refine_periodic_delay(centered, step_s=step_s, lag=lag)
        for lag in candidates
    ]
    candidate_periods = sorted({period for period, _ in refined})
    shortest = candidate_periods[0]
    # The shortest repeat-supported peak is the fundamental.  If it is not a
    # physically admissible period, none of its harmonics may stand in for it.
    if not minimum_period_s <= shortest <= maximum_period_s:
        return None
    candidate_chain = [
        (
            shortest,
            _periodic_recurrence_residual(
                centered, step_s=step_s, period_s=shortest
            ),
        )
    ]
    for longer in candidate_periods[1:]:
        multiple = int(round(longer / shortest))
        if multiple < 2 or abs(longer - multiple * shortest) > step_s:
            continue
        candidate_chain.append(
            (
                longer,
                _periodic_recurrence_residual(
                    centered, step_s=step_s, period_s=longer
                ),
            )
        )
    bounded_chain = [
        (period, residual)
        for period, residual in candidate_chain
        if minimum_period_s <= period <= maximum_period_s
    ]
    if not bounded_chain:
        return None
    minimum_residual = min(residual for _, residual in bounded_chain)
    return float(
        next(
            period
            for period, residual in bounded_chain
            if residual <= minimum_residual + minimum_subharmonic_residual_px
        )
    )


def _refine_periodic_delay(
    centered: np.ndarray, *, step_s: float, lag: int
) -> tuple[float, float]:
    """Refine a frame-quantized correlation peak over one half-frame on either side."""
    indices = np.arange(len(centered), dtype=np.float64)
    best_period = float(lag * step_s)
    best_correlation = -1.0
    for offset_frames in np.linspace(lag - 0.5, lag + 0.5, 41):
        if offset_frames <= 0.0:
            continue
        last_start = (len(centered) - 1) - offset_frames
        if last_start < 2.0:
            continue
        starts = np.arange(int(math.floor(last_start)) + 1, dtype=np.float64)
        left = np.interp(starts, indices, centered)
        right = np.interp(starts + offset_frames, indices, centered)
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        if denominator <= 1e-12:
            continue
        correlation = float(np.dot(left, right) / denominator)
        if correlation > best_correlation:
            best_period = float(offset_frames * step_s)
            best_correlation = correlation
    return best_period, best_correlation


def _periodic_recurrence_residual(
    centered: np.ndarray, *, step_s: float, period_s: float
) -> float:
    """RMS centroid displacement left unexplained by a fractional recurrence."""
    offset_frames = period_s / step_s
    last_start = (len(centered) - 1) - offset_frames
    if offset_frames <= 0.0 or last_start < 2.0:
        return math.inf
    indices = np.arange(len(centered), dtype=np.float64)
    starts = np.arange(int(math.floor(last_start)) + 1, dtype=np.float64)
    differences = np.interp(starts, indices, centered) - np.interp(
        starts + offset_frames, indices, centered
    )
    return float(np.sqrt(np.mean(np.square(differences))))


def _release_sign(displacement: np.ndarray, amplitude_px: float) -> int:
    threshold = max(1e-9, 0.1 * amplitude_px)
    for value in displacement:
        if abs(float(value)) >= threshold:
            return 1 if value > 0.0 else -1
    return 0


def extract_spring_trace(
    masks: Sequence[np.ndarray],
    times_s: Sequence[float],
    *,
    quality_config: dict[str, Any],
) -> SpringTrace:
    """Extract one immutable, coverage-gated centroid trace from mask frames."""
    if len(masks) != len(times_s):
        raise SpringTraceError("shape_mismatch", "mask and time counts differ")
    if len(masks) < 3:
        raise SpringTraceError("insufficient_observations", "at least three masks are required")

    times = np.asarray(times_s, dtype=np.float64)
    if times.ndim != 1 or len(times) != len(masks) or not np.all(np.isfinite(times)):
        raise SpringTraceError("invalid_times", "times must be a finite one-dimensional sequence")
    if not np.all(np.diff(times) > 0.0):
        raise SpringTraceError("non_monotonic_times", "times must be strictly increasing")
    first = np.asarray(masks[0])
    if first.ndim != 2:
        raise SpringTraceError("invalid_mask_shape", "masks must be two-dimensional")
    frame_shape = first.shape
    if frame_shape[0] <= 0 or frame_shape[1] <= 0:
        raise SpringTraceError("invalid_mask_shape", "mask canvas must be non-empty")

    minimum_pixels = _require_quality(quality_config, "minimum_mask_pixels")
    minimum_area_ratio = _require_quality(quality_config, "minimum_mask_area_ratio")
    maximum_area_ratio = _require_quality(quality_config, "maximum_mask_area_ratio")
    required_valid_ratio = _require_quality(quality_config, "minimum_valid_frame_ratio")
    minimum_amplitude_px = _require_quality(quality_config, "minimum_amplitude_px")
    minimum_s = _require_quality(quality_config, "minimum_s")
    maximum_s = _require_quality(quality_config, "maximum_s")
    if not (0.0 <= minimum_area_ratio <= maximum_area_ratio <= 1.0):
        raise ValueError("mask area ratios must satisfy 0 <= minimum <= maximum <= 1")
    if not (0.0 <= required_valid_ratio <= 1.0):
        raise ValueError("minimum_valid_frame_ratio must be in [0, 1]")
    if minimum_pixels < 0.0 or minimum_amplitude_px < 0.0 or minimum_s <= 0.0:
        raise ValueError("minimum quality bounds must be non-negative and minimum_s positive")
    if maximum_s < minimum_s:
        raise ValueError("maximum_s must be at least minimum_s")
    minimum_correlation = _configured_period_threshold(
        quality_config, "minimum_period_correlation", _MINIMUM_PERIOD_CORRELATION
    )
    minimum_prominence = _configured_period_threshold(
        quality_config, "minimum_period_peak_prominence", _MINIMUM_PERIOD_PROMINENCE
    )
    minimum_complete_cycles = _configured_period_threshold(
        quality_config, "minimum_complete_cycles", _MINIMUM_COMPLETE_CYCLES
    )
    minimum_subharmonic_residual_px = _configured_period_threshold(
        quality_config,
        "minimum_subharmonic_residual_px",
        _MINIMUM_SUBHARMONIC_RESIDUAL_PX,
    )
    cadence_tolerance = _configured_period_threshold(
        quality_config,
        "maximum_cadence_relative_deviation",
        _CADENCE_RELATIVE_TOLERANCE,
    )
    if minimum_correlation > 1.0:
        raise ValueError("minimum_period_correlation must be at most one")
    if cadence_tolerance > 1.0:
        raise ValueError("maximum_cadence_relative_deviation must be at most one")
    _require_uniform_cadence(times, relative_tolerance=cadence_tolerance)

    frame_area = float(frame_shape[0] * frame_shape[1])
    minimum_area = max(minimum_pixels, frame_area * minimum_area_ratio)
    maximum_area = frame_area * maximum_area_ratio
    xy = np.full((len(masks), 2), np.nan, dtype=np.float64)
    areas = np.zeros(len(masks), dtype=np.float64)
    valid = np.zeros(len(masks), dtype=bool)
    for index, raw_mask in enumerate(masks):
        mask = np.asarray(raw_mask)
        if mask.ndim != 2 or mask.shape != frame_shape:
            raise SpringTraceError("shape_mismatch", "all mask canvases must have the same shape")
        ys, xs = np.nonzero(mask > 0)
        area = float(len(xs))
        areas[index] = area
        if area < minimum_area or area > maximum_area:
            continue
        xy[index] = (float(np.mean(xs)), float(np.mean(ys)))
        valid[index] = True

    valid_ratio = float(np.mean(valid))
    if int(valid.sum()) < 3 or valid_ratio < required_valid_ratio:
        raise SpringTraceError(
            "insufficient_valid_masks",
            f"valid spring masks {int(valid.sum())}/{len(valid)} ({valid_ratio:.3f}) "
            f"below required {required_valid_ratio:.3f}",
        )
    xy = _interpolate(xy, valid)
    equilibrium_y = float(np.median(xy[valid, 1]))
    displacement = xy[:, 1] - equilibrium_y
    amplitude = float(
        0.5 * (np.quantile(displacement[valid], 0.95) - np.quantile(displacement[valid], 0.05))
    )
    if amplitude < minimum_amplitude_px:
        raise SpringTraceError(
            "insufficient_amplitude",
            f"vertical amplitude {amplitude:.3f}px is below required {minimum_amplitude_px:.3f}px",
        )
    envelope = np.abs(displacement)
    horizontal_span = float(
        0.5 * (np.quantile(xy[valid, 0], 0.95) - np.quantile(xy[valid, 0], 0.05))
    )
    period = _qualified_autocorrelation_period(
        displacement,
        times,
        minimum_correlation=minimum_correlation,
        minimum_prominence=minimum_prominence,
        minimum_complete_cycles=minimum_complete_cycles,
        minimum_subharmonic_residual_px=minimum_subharmonic_residual_px,
        minimum_period_s=minimum_s,
        maximum_period_s=maximum_s,
    )
    if period is not None and not minimum_s <= period <= maximum_s:
        raise SpringTraceError(
            "period_out_of_bounds",
            "observable autocorrelation period is outside configured bounds",
        )

    return SpringTrace(
        times_s=_readonly(times),
        xy=_readonly(xy),
        area_px2=_readonly(areas),
        valid=_readonly(valid),
        valid_ratio=valid_ratio,
        equilibrium_y_px=equilibrium_y,
        amplitude_px=amplitude,
        envelope_px=_readonly(envelope),
        period_s=period,
        horizontal_drift_ratio=float(horizontal_span / max(amplitude, 1e-12)),
        release_sign=_release_sign(displacement, amplitude),
    )


def _bounded(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return float(min(1.0, max(0.0, value)))


def _exponential_similarity(error: float, *, scale: float) -> float:
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("similarity scales must be finite and positive")
    if not math.isfinite(error):
        return 0.0
    return _bounded(math.exp(-max(0.0, error) / scale))


def _relative_error(observed: float, expected: float) -> float:
    return abs(observed - expected) / max(abs(expected), 1e-12)


def _period_similarity(
    observed_s: float,
    expected_s: float,
    *,
    sample_resolution_s: float,
) -> float:
    """Treat sub-frame period differences as indistinguishable observations."""
    error = _period_relative_error(
        observed_s,
        expected_s,
        sample_resolution_s=sample_resolution_s,
    )
    return _exponential_similarity(error, scale=1.0)


def _period_relative_error(
    observed_s: float,
    expected_s: float,
    *,
    sample_resolution_s: float,
) -> float:
    """Return resolution-aware relative period error."""
    delta_s = max(0.0, abs(observed_s - expected_s) - 0.5 * sample_resolution_s)
    return delta_s / max(abs(expected_s), 1e-12)


def _periodicity_evidence(
    trace_data: dict[str, Any], *, scoring_config: dict[str, Any]
) -> float:
    """Return one only for an accepted, repeat-supported period in this trace."""
    period = trace_data["period_s"]
    if period is None:
        return 0.0
    resolution = _require_uniform_cadence(trace_data["times_s"])
    qualified = _qualified_autocorrelation_period(
        trace_data["xy"][:, 1] - trace_data["equilibrium_y_px"],
        trace_data["times_s"],
        minimum_correlation=float(
            scoring_config.get("minimum_period_correlation", _MINIMUM_PERIOD_CORRELATION)
        ),
        minimum_prominence=float(
            scoring_config.get("minimum_period_peak_prominence", _MINIMUM_PERIOD_PROMINENCE)
        ),
        minimum_complete_cycles=float(
            scoring_config.get("minimum_complete_cycles", _MINIMUM_COMPLETE_CYCLES)
        ),
        minimum_subharmonic_residual_px=_configured_scoring_threshold(
            scoring_config,
            "minimum_subharmonic_residual_px",
            _MINIMUM_SUBHARMONIC_RESIDUAL_PX,
        ),
        minimum_period_s=max(1e-12, float(period) - 0.5 * resolution),
        maximum_period_s=float(period) + 0.5 * resolution,
    )
    if qualified is None:
        return 0.0
    return 1.0 if abs(qualified - period) <= 0.5 * resolution else 0.0


def score_spring_traces(
    reference: SpringTrace,
    prediction: SpringTrace,
    *,
    mass_kg: float,
    stiffness_n_m: float,
    scoring_config: dict[str, Any],
) -> dict[str, Any]:
    """Score traces at their exact samples; no warping or reference realignment."""
    reference_data = _validated_trace_data(reference)
    prediction_data = _validated_trace_data(prediction)
    if reference_data["times_s"].shape != prediction_data["times_s"].shape or not np.allclose(
        reference_data["times_s"], prediction_data["times_s"], atol=1e-9, rtol=0.0
    ):
        raise ValueError("reference and prediction traces use different timelines")
    if reference_data["xy"].shape != prediction_data["xy"].shape:
        raise ValueError("reference and prediction trace coordinates differ in shape")
    theory = theoretical_period_s(mass_kg=mass_kg, stiffness_n_m=stiffness_n_m)
    try:
        configured_weights = {
            name: float(value) for name, value in scoring_config["weights"].items()
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("scoring_config requires numeric component weights") from exc
    component_names = (
        "vertical_trajectory",
        "period",
        "amplitude_envelope",
        "equilibrium_release_phase",
        "vertical_axis_confinement",
        "oscillation_evidence",
    )
    if set(configured_weights) != set(component_names):
        raise ValueError("scoring weights must name exactly the spring components")
    if any(not math.isfinite(weight) or weight < 0.0 for weight in configured_weights.values()):
        raise ValueError("scoring weights must be finite and non-negative")
    total_weight = float(sum(configured_weights.values()))
    if total_weight <= 0.0:
        raise ValueError("at least one scoring weight must be positive")
    weights = {name: weight / total_weight for name, weight in configured_weights.items()}

    trajectory_error = float(
        np.sqrt(
            np.mean(
                np.square(prediction_data["xy"][:, 1] - reference_data["xy"][:, 1])
            )
        )
        / max(reference_data["amplitude_px"], 1e-12)
    )
    trajectory = _exponential_similarity(
        trajectory_error, scale=float(scoring_config["trajectory_scale"])
    )
    if reference_data["period_s"] is None or prediction_data["period_s"] is None:
        period = 0.0
    else:
        sample_resolution_s = _require_uniform_cadence(reference_data["times_s"])
        reference_similarity = _period_similarity(
            prediction_data["period_s"],
            reference_data["period_s"],
            sample_resolution_s=sample_resolution_s,
        )
        reference_theory_error = _period_relative_error(
            reference_data["period_s"],
            theory,
            sample_resolution_s=sample_resolution_s,
        )
        prediction_theory_error = _period_relative_error(
            prediction_data["period_s"],
            theory,
            sample_resolution_s=sample_resolution_s,
        )
        theory_similarity = _exponential_similarity(
            max(0.0, prediction_theory_error - reference_theory_error),
            scale=1.0,
        )
        period = _bounded(math.sqrt(reference_similarity * theory_similarity))

    amplitude_similarity = _exponential_similarity(
        _relative_error(prediction_data["amplitude_px"], reference_data["amplitude_px"]),
        scale=float(scoring_config["amplitude_scale"]),
    )
    envelope_error = float(
        np.sqrt(
            np.mean(
                np.square(prediction_data["envelope_px"] - reference_data["envelope_px"])
            )
        )
        / max(reference_data["amplitude_px"], 1e-12)
    )
    envelope_similarity = _exponential_similarity(
        envelope_error, scale=float(scoring_config["amplitude_scale"])
    )
    amplitude_envelope = _bounded(0.5 * (amplitude_similarity + envelope_similarity))

    equilibrium_similarity = _exponential_similarity(
        abs(
            prediction_data["equilibrium_y_px"]
            - reference_data["equilibrium_y_px"]
        )
        / max(reference_data["amplitude_px"], 1e-12),
        scale=float(scoring_config["equilibrium_scale"]),
    )
    release_similarity = (
        1.0
        if prediction_data["release_sign"] != 0
        and prediction_data["release_sign"] == reference_data["release_sign"]
        else 0.0
    )
    equilibrium_release_phase = _bounded(equilibrium_similarity * release_similarity)
    oscillation_evidence = _exponential_similarity(
        _relative_error(prediction_data["amplitude_px"], reference_data["amplitude_px"]),
        scale=float(scoring_config["oscillation_amplitude_scale"]),
    ) * _periodicity_evidence(prediction_data, scoring_config=scoring_config)
    reference_horizontal = (
        reference_data["xy"][:, 0] - reference_data["xy"][0, 0]
    )
    prediction_horizontal = (
        prediction_data["xy"][:, 0] - prediction_data["xy"][0, 0]
    )
    horizontal_trajectory_error = float(
        np.sqrt(
            np.mean(np.square(prediction_horizontal - reference_horizontal))
        )
        / max(reference_data["amplitude_px"], 1e-12)
    )
    horizontal_trajectory_similarity = _exponential_similarity(
        horizontal_trajectory_error,
        scale=float(scoring_config["horizontal_trajectory_scale"]),
    )
    excess_horizontal_drift_similarity = _exponential_similarity(
        max(
            0.0,
            prediction_data["horizontal_drift_ratio"]
            - reference_data["horizontal_drift_ratio"],
        ),
        scale=float(scoring_config["axis_drift_scale"]),
    )
    # A stationary point is geometrically confined but does not establish a
    # vertical motion axis, so confinement requires observable oscillation.
    # The scalar span protects against excess drift while the centered exact-
    # timeline trajectory prevents equal-span horizontal motion from hiding a
    # different axis history. Centering preserves invariance to camera origin.
    vertical_axis_confinement = _bounded(
        excess_horizontal_drift_similarity
        * horizontal_trajectory_similarity
        * oscillation_evidence
    )

    components = {
        "vertical_trajectory": trajectory,
        "period": period,
        "amplitude_envelope": amplitude_envelope,
        "equilibrium_release_phase": equilibrium_release_phase,
        "vertical_axis_confinement": vertical_axis_confinement,
        "oscillation_evidence": oscillation_evidence,
    }
    score = _bounded(sum(weights[name] * components[name] for name in component_names))
    return {
        "score": score,
        "components": components,
        "weights": weights,
        "diagnostics": {
            "reference_period_s": reference_data["period_s"],
            "prediction_period_s": prediction_data["period_s"],
            "theoretical_period_s": theory,
            "reference_amplitude_px": reference_data["amplitude_px"],
            "prediction_amplitude_px": prediction_data["amplitude_px"],
            "reference_horizontal_drift_ratio": reference_data[
                "horizontal_drift_ratio"
            ],
            "prediction_horizontal_drift_ratio": prediction_data[
                "horizontal_drift_ratio"
            ],
            "horizontal_trajectory_error": horizontal_trajectory_error,
            "horizontal_trajectory_similarity": horizontal_trajectory_similarity,
            "excess_horizontal_drift_similarity": excess_horizontal_drift_similarity,
        },
    }
