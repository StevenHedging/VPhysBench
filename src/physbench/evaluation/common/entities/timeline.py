from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class CommonTimeGrid:
    """A shared physical-time grid with integration weights per sample.

    Samples represent Voronoi cells on ``[interval_start_s, interval_end_s]``.
    Interior cell boundaries are the midpoints between adjacent timestamps.
    With the default interval bounds, the weights are therefore the familiar
    trapezoidal-rule weights and sum to ``times_s[-1] - times_s[0]``.

    An empty grid has an empty weight vector and zero duration.  A one-sample
    grid has zero duration unless explicit interval bounds are supplied.  This
    avoids inventing a frame duration when no sampling period is known.
    """

    times_s: np.ndarray
    cell_weights_s: np.ndarray
    interval_start_s: float
    interval_end_s: float

    def __post_init__(self) -> None:
        times = np.asarray(self.times_s, dtype=np.float64)
        weights = np.asarray(self.cell_weights_s, dtype=np.float64)
        start = float(self.interval_start_s)
        end = float(self.interval_end_s)
        if times.ndim != 1:
            raise ValueError("times_s must be one-dimensional")
        if weights.shape != times.shape:
            raise ValueError("cell_weights_s must have one value per sample")
        if not np.isfinite(times).all():
            raise ValueError("times_s must contain only finite values")
        if times.size and float(times.min()) < 0.0:
            raise ValueError("times_s must be non-negative")
        if times.size > 1 and np.any(np.diff(times) <= 0.0):
            raise ValueError("times_s must be strictly increasing")
        if not math.isfinite(start) or not math.isfinite(end):
            raise ValueError("timeline interval bounds must be finite")
        if start < 0.0 or end < start:
            raise ValueError(
                "timeline interval must be non-negative and ordered"
            )
        if times.size:
            tolerance = 1e-12
            if start > float(times[0]) + tolerance:
                raise ValueError(
                    "interval_start_s must not follow the first sample"
                )
            if end + tolerance < float(times[-1]):
                raise ValueError(
                    "interval_end_s must not precede the last sample"
                )
        elif abs(end - start) > 1e-12:
            raise ValueError(
                "an empty time grid cannot represent a non-empty interval"
            )
        if not np.isfinite(weights).all() or np.any(weights < 0.0):
            raise ValueError(
                "cell_weights_s must be finite and non-negative"
            )
        duration = end - start
        if not math.isclose(
            float(weights.sum()),
            duration,
            rel_tol=1e-10,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "cell_weights_s must sum to the timeline duration"
            )

        # A frozen dataclass alone does not make NumPy buffers immutable.
        # Copying prevents caller-owned arrays from changing the time contract.
        normalized_times = np.array(times, dtype=np.float64, copy=True)
        normalized_weights = np.array(weights, dtype=np.float64, copy=True)
        normalized_times.setflags(write=False)
        normalized_weights.setflags(write=False)
        object.__setattr__(self, "times_s", normalized_times)
        object.__setattr__(self, "cell_weights_s", normalized_weights)
        object.__setattr__(self, "interval_start_s", start)
        object.__setattr__(self, "interval_end_s", end)

    @property
    def duration_s(self) -> float:
        return float(self.interval_end_s - self.interval_start_s)

    def integrate(self, values: Sequence[float] | np.ndarray) -> float:
        """Integrate one finite scalar value per timeline cell."""
        array = np.asarray(values, dtype=np.float64)
        if array.shape != self.times_s.shape:
            raise ValueError("values must have one value per timeline sample")
        if not np.isfinite(array).all():
            raise ValueError("values must contain only finite values")
        return float(np.dot(array, self.cell_weights_s))

    def to_dict(self) -> dict[str, object]:
        return {
            "times_s": self.times_s.tolist(),
            "cell_weights_s": self.cell_weights_s.tolist(),
            "interval_start_s": self.interval_start_s,
            "interval_end_s": self.interval_end_s,
            "duration_s": self.duration_s,
        }


def build_common_time_grid(
    times_s: Sequence[float] | np.ndarray,
    *,
    interval_start_s: float | None = None,
    interval_end_s: float | None = None,
) -> CommonTimeGrid:
    """Construct physical-time integration cells around shared sample times.

    Evaluators should first choose one timeline for both reference and
    prediction and then use this function.  It deliberately performs no time
    warping or source-frame inference.
    """

    times = np.asarray(times_s, dtype=np.float64)
    if times.ndim != 1:
        raise ValueError("times_s must be one-dimensional")
    if not np.isfinite(times).all():
        raise ValueError("times_s must contain only finite values")
    if times.size and float(times.min()) < 0.0:
        raise ValueError("times_s must be non-negative")
    if times.size > 1 and np.any(np.diff(times) <= 0.0):
        raise ValueError("times_s must be strictly increasing")

    if not times.size:
        start = 0.0 if interval_start_s is None else float(interval_start_s)
        end = start if interval_end_s is None else float(interval_end_s)
        return CommonTimeGrid(
            times_s=np.empty(0, dtype=np.float64),
            cell_weights_s=np.empty(0, dtype=np.float64),
            interval_start_s=start,
            interval_end_s=end,
        )

    start = (
        float(times[0])
        if interval_start_s is None
        else float(interval_start_s)
    )
    end = (
        float(times[-1])
        if interval_end_s is None
        else float(interval_end_s)
    )
    if not math.isfinite(start) or not math.isfinite(end):
        raise ValueError("timeline interval bounds must be finite")
    if start < 0.0 or end < 0.0:
        raise ValueError("timeline interval bounds must be non-negative")
    if start > float(times[0]) + 1e-12:
        raise ValueError("interval_start_s must not follow the first sample")
    if end + 1e-12 < float(times[-1]):
        raise ValueError("interval_end_s must not precede the last sample")
    if end < start:
        raise ValueError("timeline interval must be ordered")

    if times.size == 1:
        weights = np.asarray([end - start], dtype=np.float64)
    else:
        boundaries = np.empty(times.size + 1, dtype=np.float64)
        boundaries[0] = start
        boundaries[-1] = end
        boundaries[1:-1] = times[:-1] + 0.5 * np.diff(times)
        weights = np.diff(boundaries)
    return CommonTimeGrid(
        times_s=times,
        cell_weights_s=weights,
        interval_start_s=start,
        interval_end_s=end,
    )
