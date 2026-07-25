from __future__ import annotations

import math

import numpy as np


def fit_polynomial(
    times_s: np.ndarray,
    values: np.ndarray,
    *,
    degree: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    times = np.asarray(times_s, dtype=np.float64)
    observed = np.asarray(values, dtype=np.float64)
    coefficients = np.polyfit(times, observed, degree)
    fitted = np.polyval(coefficients, times)
    rmse = float(np.sqrt(np.mean(np.square(observed - fitted))))
    return coefficients, fitted, rmse


def normalized_rmse(
    reference: np.ndarray,
    prediction: np.ndarray,
    *,
    minimum_scale: float,
) -> tuple[float, float]:
    reference = np.asarray(reference, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    rmse = float(np.sqrt(np.mean(np.square(reference - prediction))))
    scale = max(float(np.ptp(reference)), float(minimum_scale))
    return rmse, rmse / scale


def exponential_similarity(error: float, *, scale: float = 1.0) -> float:
    return float(math.exp(-max(float(error), 0.0) / max(float(scale), 1e-12)))
