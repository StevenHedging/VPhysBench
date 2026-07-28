from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import load_json, load_jsonl, write_json, write_jsonl
from .metrics import evaluate_cases
from .reporting import render_report
from .validation import load_scene_configs


def reevaluate_run(run_dir: str | Path, scene_config_dir: str | Path) -> dict[str, Any]:
    """Re-evaluate a frozen pre-AtomicRun directory.

    This is a read-only compatibility path for historical schema-v1 runs. New
    execution must use ``run_atomic`` or ``run_matrix``; in particular this
    module no longer plans prompt-profile experiment arms.
    """
    directory = Path(run_dir)
    cases = load_jsonl(directory / "frozen_cases.jsonl")
    predictions = load_jsonl(directory / "predictions.jsonl")
    scenes = load_scene_configs(scene_config_dir)
    metric_config = load_json(directory / "frozen_metrics.json")
    results, summary = evaluate_cases(cases, predictions, scenes, metric_config)
    write_jsonl(directory / "case_metrics.jsonl", results)
    write_json(directory / "summary.json", summary)
    run = load_json(directory / "run.json")
    plan = load_json(directory / "plan.json")
    (directory / "report.md").write_text(render_report(run, plan, summary), encoding="utf-8")
    return summary
