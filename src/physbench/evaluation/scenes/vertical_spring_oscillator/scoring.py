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


def _autocorrelation_period(
    displacement: np.ndarray,
    times_s: np.ndarray,
    *,
    minimum_s: float,
    maximum_s: float,
) -> float | None:
    """Find the first significant autocorrelation maximum inside configured lags."""
    if len(displacement) < 5:
        return None
    centered = displacement - float(np.mean(displacement))
    energy = float(np.dot(centered, centered))
    if energy <= 1e-12:
        return None
    step_s = float(np.median(np.diff(times_s)))
    if not math.isfinite(step_s) or step_s <= 0.0:
        return None
    minimum_lag = max(2, int(math.ceil(minimum_s / step_s)))
    maximum_lag = min(len(centered) - 2, int(math.floor(maximum_s / step_s)))
    if minimum_lag > maximum_lag:
        return None
    correlations = np.correlate(centered, centered, mode="full")[len(centered) - 1 :]
    normalized = correlations / energy
    for lag in range(minimum_lag + 1, maximum_lag):
        if (
            normalized[lag] >= normalized[lag - 1]
            and normalized[lag] > normalized[lag + 1]
            and normalized[lag] >= 0.1
        ):
            return float(lag * step_s)
    window = normalized[minimum_lag : maximum_lag + 1]
    if not len(window):
        return None
    peak_lag = minimum_lag + int(np.argmax(window))
    return float(peak_lag * step_s) if normalized[peak_lag] >= 0.1 else None


def _first_autocorrelation_peak(
    displacement: np.ndarray, times_s: np.ndarray
) -> float | None:
    """Expose the fundamental peak only to reject observable out-of-range motion."""
    if len(displacement) < 5:
        return None
    centered = displacement - float(np.mean(displacement))
    energy = float(np.dot(centered, centered))
    step_s = float(np.median(np.diff(times_s)))
    if energy <= 1e-12 or not math.isfinite(step_s) or step_s <= 0.0:
        return None
    correlations = np.correlate(centered, centered, mode="full")[len(centered) - 1 :]
    normalized = correlations / energy
    for lag in range(2, len(normalized) - 1):
        if (
            normalized[lag] >= normalized[lag - 1]
            and normalized[lag] > normalized[lag + 1]
            and normalized[lag] >= 0.1
        ):
            return float(lag * step_s)
    return None


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
    observed_period = _first_autocorrelation_peak(displacement, times)
    if observed_period is not None and not minimum_s <= observed_period <= maximum_s:
        raise SpringTraceError(
            "period_out_of_bounds",
            "observable autocorrelation period is outside configured bounds",
        )
    period = _autocorrelation_period(
        displacement, times, minimum_s=minimum_s, maximum_s=maximum_s
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
    delta_s = max(0.0, abs(observed_s - expected_s) - 0.5 * sample_resolution_s)
    return _exponential_similarity(delta_s / max(abs(expected_s), 1e-12), scale=1.0)


def score_spring_traces(
    reference: SpringTrace,
    prediction: SpringTrace,
    *,
    mass_kg: float,
    stiffness_n_m: float,
    scoring_config: dict[str, Any],
) -> dict[str, Any]:
    """Score traces at their exact samples; no warping or reference realignment."""
    if reference.times_s.shape != prediction.times_s.shape or not np.allclose(
        reference.times_s, prediction.times_s, atol=1e-9, rtol=0.0
    ):
        raise ValueError("reference and prediction traces use different timelines")
    if reference.xy.shape != prediction.xy.shape:
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
        np.sqrt(np.mean(np.square(prediction.xy[:, 1] - reference.xy[:, 1])))
        / max(reference.amplitude_px, 1e-12)
    )
    trajectory = _exponential_similarity(
        trajectory_error, scale=float(scoring_config["trajectory_scale"])
    )
    if reference.period_s is None or prediction.period_s is None:
        period = 0.0
    else:
        sample_resolution_s = float(np.median(np.diff(reference.times_s)))
        reference_similarity = _period_similarity(
            prediction.period_s,
            reference.period_s,
            sample_resolution_s=sample_resolution_s,
        )
        theory_similarity = _period_similarity(
            prediction.period_s,
            theory,
            sample_resolution_s=sample_resolution_s,
        )
        period = _bounded(math.sqrt(reference_similarity * theory_similarity))

    amplitude_similarity = _exponential_similarity(
        _relative_error(prediction.amplitude_px, reference.amplitude_px),
        scale=float(scoring_config["amplitude_scale"]),
    )
    envelope_error = float(
        np.sqrt(np.mean(np.square(prediction.envelope_px - reference.envelope_px)))
        / max(reference.amplitude_px, 1e-12)
    )
    envelope_similarity = _exponential_similarity(
        envelope_error, scale=float(scoring_config["amplitude_scale"])
    )
    amplitude_envelope = _bounded(0.5 * (amplitude_similarity + envelope_similarity))

    equilibrium_similarity = _exponential_similarity(
        _relative_error(prediction.equilibrium_y_px, reference.equilibrium_y_px),
        scale=float(scoring_config["equilibrium_scale"]),
    )
    release_similarity = (
        1.0
        if prediction.release_sign != 0 and prediction.release_sign == reference.release_sign
        else 0.0
    )
    equilibrium_release_phase = _bounded(equilibrium_similarity * release_similarity)
    oscillation_evidence = _exponential_similarity(
        _relative_error(prediction.amplitude_px, reference.amplitude_px),
        scale=float(scoring_config["oscillation_amplitude_scale"]),
    )
    # A stationary point is geometrically confined but does not establish a
    # vertical motion axis, so confinement requires observable oscillation.
    vertical_axis_confinement = _bounded(
        _exponential_similarity(
            prediction.horizontal_drift_ratio,
            scale=float(scoring_config["axis_drift_scale"]),
        )
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
            "reference_period_s": reference.period_s,
            "prediction_period_s": prediction.period_s,
            "theoretical_period_s": theory,
            "reference_amplitude_px": reference.amplitude_px,
            "prediction_amplitude_px": prediction.amplitude_px,
        },
    }
