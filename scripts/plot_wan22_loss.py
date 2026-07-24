#!/usr/bin/env python3
"""Export raw/smoothed WAN training loss and epoch-level descriptive statistics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def moving(values: list[float], window: int, median: bool = False) -> list[float]:
    output = []
    for index in range(len(values)):
        chunk = values[max(0, index - window + 1): index + 1]
        output.append(statistics.median(chunk) if median else statistics.fmean(chunk))
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    checkpoint_dir = args.run_dir / "artifacts" / "wan22" / "checkpoints"
    event_files = sorted((checkpoint_dir / "tensorboard_log").glob("events.*"))
    if not event_files:
        raise FileNotFoundError("TensorBoard event file is not available")

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    scalars = {}
    for event_file in event_files:
        accumulator = EventAccumulator(str(event_file))
        accumulator.Reload()
        if "loss" in accumulator.Tags().get("scalars", []):
            for event in accumulator.Scalars("loss"):
                scalars[event.step] = float(event.value)
    steps = sorted(scalars)
    losses = [scalars[step] for step in steps]
    if not losses or any(not math.isfinite(value) for value in losses):
        raise RuntimeError("loss series is empty or contains non-finite values")
    sampling = json.loads(
        (args.run_dir / "artifacts" / "wan22" / "training_sampling_plan.json").read_text()
    )
    steps_per_epoch = int(sampling["expected_optimizer_steps_per_epoch"])
    mean25 = moving(losses, 25)
    median25 = moving(losses, 25, median=True)
    mean100 = moving(losses, 100)
    ema = []
    for value in losses:
        ema.append(value if not ema else 0.05 * value + 0.95 * ema[-1])

    output = args.run_dir / "artifacts" / "wan22" / "loss_analysis"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "loss_curve.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "epoch", "loss", "rolling_mean_25", "rolling_median_25", "rolling_mean_100", "ema_005"])
        for values in zip(steps, losses, mean25, median25, mean100, ema):
            step, loss, avg25, med25, avg100, ema_value = values
            writer.writerow([step, (step - 1) // steps_per_epoch + 1, loss, avg25, med25, avg100, ema_value])

    epoch_summary = []
    for epoch in range(1, math.ceil(max(steps) / steps_per_epoch) + 1):
        selected = [
            loss for step, loss in zip(steps, losses)
            if (step - 1) // steps_per_epoch + 1 == epoch
        ]
        if selected:
            epoch_summary.append({
                "epoch": epoch,
                "count": len(selected),
                "mean": statistics.fmean(selected),
                "median": statistics.median(selected),
                "min": min(selected),
                "max": max(selected),
                "stdev": statistics.stdev(selected) if len(selected) > 1 else 0.0,
            })
    first = losses[: min(100, len(losses))]
    last = losses[-min(100, len(losses)):]
    summary = {
        "recorded_steps": len(losses),
        "first_step": steps[0],
        "last_step": steps[-1],
        "expected_total_steps": sampling["expected_total_optimizer_steps"],
        "steps_per_epoch": steps_per_epoch,
        "finite_fraction": 1.0,
        "mean": statistics.fmean(losses),
        "median": statistics.median(losses),
        "min": min(losses),
        "max": max(losses),
        "first_100_mean": statistics.fmean(first),
        "last_100_mean": statistics.fmean(last),
        "last_over_first_100_mean": statistics.fmean(last) / statistics.fmean(first),
        "per_epoch": epoch_summary,
        "checkpoints": [path.name for path in sorted(checkpoint_dir.glob("*.safetensors"))],
    }
    (output / "loss_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True, constrained_layout=True)
    axes[0].plot(steps, losses, color="#9aa0a6", linewidth=0.6, alpha=0.65, label="raw loss")
    axes[0].plot(steps, mean25, color="#1a73e8", linewidth=1.5, label="rolling mean (25)")
    axes[0].plot(steps, mean100, color="#d93025", linewidth=2.0, label="rolling mean (100)")
    axes[0].plot(steps, ema, color="#188038", linewidth=1.5, label="EMA (alpha=0.05)")
    axes[0].set_ylabel("FlowMatch SFT loss")
    axes[0].set_title("WAN2.2 three-scene LoRA training loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend(ncol=4)
    axes[1].plot(steps, median25, color="#9334e6", linewidth=1.5, label="rolling median (25)")
    axes[1].plot(steps, mean25, color="#1a73e8", linewidth=1.0, alpha=0.8, label="rolling mean (25)")
    axes[1].set_xlabel("optimizer step")
    axes[1].set_ylabel("smoothed loss")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    for axis in axes:
        for boundary in range(steps_per_epoch, max(steps) + 1, steps_per_epoch):
            axis.axvline(boundary, color="black", linewidth=0.5, alpha=0.18)
    figure.savefig(output / "loss_curve.png", dpi=180)
    plt.close(figure)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
