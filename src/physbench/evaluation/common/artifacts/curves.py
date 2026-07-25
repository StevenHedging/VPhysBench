from __future__ import annotations

from pathlib import Path

import numpy as np

from ..errors import SceneAnalysisError


def _pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SceneAnalysisError(
            "matplotlib_dependency_missing",
            "matplotlib is required to save evaluation curves",
        ) from exc
    return plt


def save_iou_curve(
    path: Path,
    *,
    times_s: list[float],
    ious: list[float | None],
    case_id: str,
    scene_name: str,
) -> None:
    plt = _pyplot()
    x = np.arange(len(ious))
    fig, axis = plt.subplots(figsize=(20, 6))
    plotted_ious = np.asarray(
        [np.nan if value is None else float(value) for value in ious],
        dtype=np.float64,
    )
    axis.plot(
        x,
        plotted_ious,
        marker="o",
        markersize=3,
        linewidth=1.5,
        label="Physical-subject mask IoU (reference vs generation)",
    )
    axis.set_xlim(0, max(1, len(ious) - 1))
    axis.set_ylim(0, 1)
    tick_step = max(1, len(x) // 20)
    ticks = x[::tick_step]
    axis.set_xticks(ticks)
    axis.set_xticklabels(
        [f"{times_s[index]:.2f}" for index in ticks], fontsize=7
    )
    axis.set_yticks(np.linspace(0, 1, 11))
    axis.set_xlabel("Physical time (s)")
    axis.set_ylabel("IoU")
    axis.set_title(f"{scene_name} physical-subject IoU over time\n{case_id}")
    axis.grid(True, alpha=0.25)
    axis.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_series_comparison(
    path: Path,
    *,
    times_s: list[float],
    reference: np.ndarray,
    prediction: np.ndarray,
    ylabel: str,
    title: str,
) -> None:
    plt = _pyplot()
    fig, axis = plt.subplots(figsize=(12, 5))
    axis.plot(times_s, reference, label="Reference", linewidth=2)
    axis.plot(times_s, prediction, label="Generation", linewidth=2)
    axis.set_xlabel("Physical time (s)")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(True, alpha=0.25)
    axis.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
