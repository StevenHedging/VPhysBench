from __future__ import annotations

import itertools

import numpy as np


def assign_points(reference: np.ndarray, prediction: np.ndarray) -> list[int]:
    """Return the minimum-distance prediction permutation for small N."""
    reference = np.asarray(reference, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if reference.shape != prediction.shape or reference.ndim != 2:
        raise ValueError("point sets must have the same NxD shape")
    count = len(reference)
    if count > 8:
        raise ValueError("brute-force assignment is limited to eight objects")
    best_cost = float("inf")
    best: tuple[int, ...] | None = None
    for permutation in itertools.permutations(range(count)):
        reordered = prediction[list(permutation)]
        cost = float(np.linalg.norm(reference - reordered, axis=1).sum())
        if cost < best_cost:
            best_cost = cost
            best = permutation
    return list(best or ())
