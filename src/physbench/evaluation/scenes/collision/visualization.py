from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ....io import canonical_sha256, sha256_file, write_json
from ...common.artifacts import write_rows_csv


_ROLE_NAMES = ("striker", "target_1", "target_2")
_ROLE_COLORS = (
    (40, 80, 255),
    (40, 210, 80),
    (255, 150, 30),
)


def _artifact_directory(
    request: Any, *, config: dict[str, Any]
) -> tuple[Path, str]:
    environment_name = str(
        config.get("external_root_env", "PHYSBENCH_VISUALIZATION_ROOT")
    )
    root_value = os.environ.get(environment_name) or config["external_root"]
    root = Path(root_value).expanduser()
    namespace = str(config.get("namespace", "collision"))
    identity = hashlib.sha256(
        str(request.artifact_dir.resolve()).encode("utf-8")
    ).hexdigest()[:16]
    relative = (
        Path(namespace)
        / request.case["case_id"]
        / f"{request.job['job_id']}-{identity}"
    )
    return root / relative, relative.as_posix()


def _overlay_instances(
    frame: np.ndarray,
    masks: list[list[np.ndarray]],
    xy: np.ndarray,
    *,
    frame_index: int,
    title: str,
    seed_frame: int | None,
    event_frame: int | None,
) -> np.ndarray:
    output = frame.copy()
    for object_index, color in enumerate(_ROLE_COLORS):
        mask = masks[object_index][frame_index] > 0
        if np.any(mask):
            color_layer = np.zeros_like(output)
            color_layer[mask] = color
            output = cv2.addWeighted(output, 1.0, color_layer, 0.38, 0.0)
            contours, _ = cv2.findContours(
                mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(output, contours, -1, color, 2)
        history = xy[: frame_index + 1, object_index]
        finite = np.isfinite(history).all(axis=1)
        points = np.rint(history[finite]).astype(np.int32)
        if len(points) >= 2:
            cv2.polylines(output, [points], False, color, 2, cv2.LINE_AA)
        if len(points):
            center = tuple(points[-1])
            cv2.circle(output, center, 4, color, -1, cv2.LINE_AA)
            cv2.putText(
                output,
                _ROLE_NAMES[object_index],
                (center[0] + 6, center[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                color,
                1,
                cv2.LINE_AA,
            )
    cv2.rectangle(output, (0, 0), (output.shape[1], 36), (0, 0, 0), -1)
    labels = [title, f"frame={frame_index}"]
    if seed_frame is not None and frame_index == seed_frame:
        labels.append("SEED")
    if event_frame is not None and frame_index == event_frame:
        labels.append("CONTACT")
    cv2.putText(
        output,
        " | ".join(labels),
        (12, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def _overlap_panel(
    reference_frame: np.ndarray,
    reference_union: np.ndarray,
    prediction_union: np.ndarray,
    *,
    frame_index: int,
    union_iou: float | None,
) -> np.ndarray:
    gray = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY)
    output = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    output = (0.38 * output).astype(np.uint8)
    reference = reference_union > 0
    prediction = prediction_union > 0
    only_reference = reference & ~prediction
    only_prediction = prediction & ~reference
    intersection = reference & prediction
    output[only_reference] = (255, 80, 60)
    output[only_prediction] = (40, 70, 255)
    output[intersection] = (40, 230, 80)
    cv2.rectangle(output, (0, 0), (output.shape[1], 60), (0, 0, 0), -1)
    score_text = "n/a" if union_iou is None else f"{union_iou:.3f}"
    cv2.putText(
        output,
        f"mask agreement | IoU={score_text} | frame={frame_index}",
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        "blue=reference only  red=prediction only  green=intersection",
        (12, 49),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return output


def _dashboard(
    *,
    width: int,
    height: int,
    frame_index: int,
    time_s: float,
    union_iou: float | None,
    instance_ious: list[float | None],
    reference_event_frame: int | None,
    prediction_event_frame: int | None,
    reference_mode: str,
    prediction_mode: str,
) -> np.ndarray:
    panel = np.full((height, width, 3), 22, dtype=np.uint8)
    lines = [
        "Collision evaluator diagnostic",
        f"time={time_s:.3f}s  frame={frame_index}",
        "union IoU="
        + ("n/a" if union_iou is None else f"{union_iou:.3f}"),
        f"reference seed={reference_mode}",
        f"prediction seed={prediction_mode}",
        (
            "contact frames: "
            f"reference={reference_event_frame} "
            f"prediction={prediction_event_frame}"
        ),
    ]
    for index, value in enumerate(instance_ious):
        lines.append(
            f"{_ROLE_NAMES[index]} IoU="
            + ("n/a" if value is None else f"{value:.3f}")
        )
    y = 34
    for line_index, line in enumerate(lines):
        color = (255, 255, 255) if line_index == 0 else (205, 205, 205)
        scale = 0.68 if line_index == 0 else 0.55
        cv2.putText(
            panel,
            line,
            (18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            2 if line_index == 0 else 1,
            cv2.LINE_AA,
        )
        y += 34
    legend_y = height - 36
    for index, (name, color) in enumerate(zip(_ROLE_NAMES, _ROLE_COLORS)):
        x = 18 + index * max(150, width // 3)
        cv2.circle(panel, (x, legend_y), 7, color, -1)
        cv2.putText(
            panel,
            name,
            (x + 13, legend_y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
            cv2.LINE_AA,
        )
    return panel


def _write_video(
    path: Path,
    *,
    times_s: list[float],
    reference_frames: list[np.ndarray],
    prediction_frames: list[np.ndarray],
    reference_masks: list[list[np.ndarray]],
    prediction_masks: list[list[np.ndarray]],
    reference_union: list[np.ndarray],
    prediction_union: list[np.ndarray],
    reference_xy: np.ndarray,
    prediction_xy: np.ndarray,
    union_ious: list[float | None],
    instance_ious: list[list[float | None]],
    reference_observation: dict[str, Any],
    prediction_observation: dict[str, Any],
    reference_event_frame: int | None,
    prediction_event_frame: int | None,
    config: dict[str, Any],
) -> None:
    panel_width = int(config.get("panel_width", 640))
    panel_height = int(config.get("panel_height", 360))
    fps = float(config.get("fps", 16.0))
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (2 * panel_width, 2 * panel_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot create collision visualization: {path}")
    try:
        reference_seed = reference_observation["prompt_builder"].get(
            "seed_frame"
        )
        prediction_seed = prediction_observation["prompt_builder"].get(
            "seed_frame"
        )
        reference_mode = str(
            reference_observation["prompt_builder"].get(
                "seed_source", "frame_zero"
            )
        )
        prediction_mode = str(
            prediction_observation["prompt_builder"].get(
                "seed_source", "frame_zero"
            )
        )
        for frame_index, time_s in enumerate(times_s):
            reference = _overlay_instances(
                reference_frames[frame_index],
                reference_masks,
                reference_xy,
                frame_index=frame_index,
                title="REFERENCE",
                seed_frame=reference_seed,
                event_frame=reference_event_frame,
            )
            prediction = _overlay_instances(
                prediction_frames[frame_index],
                prediction_masks,
                prediction_xy,
                frame_index=frame_index,
                title="PREDICTION",
                seed_frame=prediction_seed,
                event_frame=prediction_event_frame,
            )
            overlap = _overlap_panel(
                reference_frames[frame_index],
                reference_union[frame_index],
                prediction_union[frame_index],
                frame_index=frame_index,
                union_iou=union_ious[frame_index],
            )
            dashboard = _dashboard(
                width=reference.shape[1],
                height=reference.shape[0],
                frame_index=frame_index,
                time_s=time_s,
                union_iou=union_ious[frame_index],
                instance_ious=instance_ious[frame_index],
                reference_event_frame=reference_event_frame,
                prediction_event_frame=prediction_event_frame,
                reference_mode=reference_mode,
                prediction_mode=prediction_mode,
            )
            panels = [
                cv2.resize(value, (panel_width, panel_height))
                for value in (reference, prediction, overlap, dashboard)
            ]
            writer.write(
                np.vstack(
                    [
                        np.hstack(panels[:2]),
                        np.hstack(panels[2:]),
                    ]
                )
            )
    finally:
        writer.release()


def _save_similarity_plot(
    path: Path,
    *,
    times_s: list[float],
    union_ious: list[float | None],
    instance_ious: list[list[float | None]],
    case_id: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    time = np.asarray(times_s)
    fig, axis = plt.subplots(figsize=(10, 5))
    union = np.asarray(
        [np.nan if value is None else value for value in union_ious]
    )
    axis.plot(time, union, color="black", linewidth=2.4, label="subject union")
    for object_index, (name, color) in enumerate(
        zip(_ROLE_NAMES, ("#ff5050", "#36b85c", "#3196e6"))
    ):
        values = np.asarray(
            [
                np.nan if row[object_index] is None else row[object_index]
                for row in instance_ious
            ]
        )
        axis.plot(time, values, linewidth=1.8, label=name, color=color)
    axis.set_ylim(-0.02, 1.02)
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Mask IoU")
    axis.set_title(f"Collision instance agreement — {case_id}")
    axis.grid(alpha=0.25)
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_event_plot(
    path: Path,
    *,
    times_s: list[float],
    reference_normalized: np.ndarray,
    prediction_normalized: np.ndarray,
    reference_event_frame: int | None,
    prediction_event_frame: int | None,
    case_id: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    time = np.asarray(times_s)
    fig, axis = plt.subplots(figsize=(10, 5))
    for object_index, (name, color) in enumerate(
        zip(_ROLE_NAMES, ("#ff5050", "#36b85c", "#3196e6"))
    ):
        axis.plot(
            time,
            reference_normalized[:, object_index],
            color=color,
            linewidth=2.0,
            label=f"reference {name}",
        )
        axis.plot(
            time,
            prediction_normalized[:, object_index],
            color=color,
            linewidth=1.5,
            linestyle="--",
            label=f"prediction {name}",
        )
    if reference_event_frame is not None:
        axis.axvline(
            time[reference_event_frame],
            color="black",
            linewidth=1.8,
            label="reference contact",
        )
    if prediction_event_frame is not None:
        axis.axvline(
            time[prediction_event_frame],
            color="#a020f0",
            linestyle=":",
            linewidth=1.8,
            label="prediction contact",
        )
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Normalized track position")
    axis.set_title(f"Collision roles and contact event — {case_id}")
    axis.grid(alpha=0.25)
    axis.legend(loc="best", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_collision_visualization(
    request: Any,
    *,
    config: dict[str, Any],
    times_s: list[float],
    reference_frames: list[np.ndarray],
    prediction_frames: list[np.ndarray],
    reference_masks: list[list[np.ndarray]],
    prediction_masks: list[list[np.ndarray]],
    reference_union: list[np.ndarray],
    prediction_union: list[np.ndarray],
    reference_xy: np.ndarray,
    prediction_xy: np.ndarray,
    reference_normalized: np.ndarray,
    prediction_normalized: np.ndarray,
    union_ious: list[float | None],
    instance_ious: list[list[float | None]],
    rows: list[dict[str, Any]],
    reference_observation: dict[str, Any],
    prediction_observation: dict[str, Any],
    reference_event_frame: int | None,
    prediction_event_frame: int | None,
) -> dict[str, Any]:
    """Write a best-effort external diagnostic and a sealed local manifest."""
    local_manifest = (
        request.artifact_dir / "collision_visualization_manifest.json"
    )
    if not config.get("enabled", True):
        manifest = {
            "schema_version": "1.0",
            "status": "disabled",
            "storage_policy": "external_unsealed_diagnostic",
        }
        write_json(local_manifest, manifest)
        return {"collision_visualization_manifest": str(local_manifest)}

    try:
        directory, relative = _artifact_directory(request, config=config)
        directory.mkdir(parents=True, exist_ok=True)
        paths = {
            "observation_video": directory / "collision_observation.mp4",
            "instance_similarity": (
                directory / "collision_instance_similarity.png"
            ),
            "event_timeline": directory / "collision_event_timeline.png",
            "tracks_csv": directory / "collision_tracks.csv",
            "observation_json": directory / "collision_observation.json",
        }
        _write_video(
            paths["observation_video"],
            times_s=times_s,
            reference_frames=reference_frames,
            prediction_frames=prediction_frames,
            reference_masks=reference_masks,
            prediction_masks=prediction_masks,
            reference_union=reference_union,
            prediction_union=prediction_union,
            reference_xy=reference_xy,
            prediction_xy=prediction_xy,
            union_ious=union_ious,
            instance_ious=instance_ious,
            reference_observation=reference_observation,
            prediction_observation=prediction_observation,
            reference_event_frame=reference_event_frame,
            prediction_event_frame=prediction_event_frame,
            config=config,
        )
        _save_similarity_plot(
            paths["instance_similarity"],
            times_s=times_s,
            union_ious=union_ious,
            instance_ious=instance_ious,
            case_id=request.case["case_id"],
        )
        _save_event_plot(
            paths["event_timeline"],
            times_s=times_s,
            reference_normalized=reference_normalized,
            prediction_normalized=prediction_normalized,
            reference_event_frame=reference_event_frame,
            prediction_event_frame=prediction_event_frame,
            case_id=request.case["case_id"],
        )
        write_rows_csv(paths["tracks_csv"], rows)
        write_json(
            paths["observation_json"],
            {
                "schema_version": "1.0",
                "case_id": request.case["case_id"],
                "job_id": request.job["job_id"],
                "roles": list(_ROLE_NAMES),
                "reference_event_frame": reference_event_frame,
                "prediction_event_frame": prediction_event_frame,
                "reference_observation": reference_observation,
                "prediction_observation": prediction_observation,
            },
        )
        repository_link = str(config.get("repository_link", "visualizations"))
        files = {
            name: {
                "path": str(path.resolve()),
                "repository_path": (
                    Path(repository_link) / relative / path.name
                ).as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for name, path in paths.items()
        }
        manifest = {
            "schema_version": "1.0",
            "status": "complete",
            "evaluator_config_sha256": canonical_sha256(
                request.evaluator_config
            ),
            "storage_policy": (
                "external_unsealed_diagnostic_with_local_hashed_manifest"
            ),
            "external_directory": str(directory.resolve()),
            "repository_directory": (
                Path(repository_link) / relative
            ).as_posix(),
            "files": files,
        }
    except Exception as exc:
        manifest = {
            "schema_version": "1.0",
            "status": "failed",
            "storage_policy": "best_effort_never_changes_case_score",
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }
    write_json(local_manifest, manifest)
    return {
        "collision_visualization_manifest": str(local_manifest),
        "collision_visualization": manifest,
    }


__all__ = ["write_collision_visualization"]
