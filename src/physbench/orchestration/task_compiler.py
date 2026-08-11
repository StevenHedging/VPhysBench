from __future__ import annotations

from pathlib import Path
from typing import Any

from ..baseline_api import load_baseline_bundle, load_baseline_plugin
from ..datasets import load_dataset
from ..domain import BaselineTaskInstance, DatasetSnapshot, TaskSpec
from ..tasks import load_task, plan_atomic_task


def compile_task_instance(
    plugin: Any,
    dataset: DatasetSnapshot,
    task: TaskSpec,
) -> BaselineTaskInstance:
    """Compile the one canonical plan selected by the Benchmark."""
    canonical_plan = plan_atomic_task(task, dataset)
    return plugin.task_builder.compile(dataset, task, canonical_plan)


def build_task_instance(
    *,
    dataset_path: str | Path,
    task_path: str | Path,
    baseline_path: str | Path,
    check_assets: bool = True,
) -> BaselineTaskInstance:
    """Public file-based facade for a Baseline-owned TaskBuilder."""
    dataset = load_dataset(dataset_path, check_assets=check_assets)
    task = load_task(task_path)
    baseline = load_baseline_bundle(baseline_path)
    plugin = load_baseline_plugin(baseline)
    instance = compile_task_instance(plugin, dataset, task)
    instance.verify()
    return instance
