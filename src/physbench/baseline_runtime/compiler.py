from __future__ import annotations

from pathlib import Path
from typing import Any

from ..baseline_api.interfaces import TaskBuilder
from ..domain import (
    AtomicPlan,
    BaselineBundle,
    BaselineTaskInstance,
    DatasetSnapshot,
    TaskSpec,
)
from ..io import canonical_sha256
from .adapter import StandardDataAdapter


class ManagedTaskBuilder(TaskBuilder):
    """Core compiler for declarative managed and submission Baselines."""

    TYPE = "managed_task_builder_v1"

    def __init__(
        self,
        bundle: BaselineBundle,
        dependency_fingerprints: dict[str, str],
    ):
        self.bundle = bundle
        self.data_adapter = StandardDataAdapter(bundle.value["adapter"])
        self._dependency_fingerprints = dict(
            sorted({
                **dependency_fingerprints,
                **self.data_adapter.dependency_fingerprints,
            }.items())
        )

    @property
    def dependency_fingerprints(self) -> dict[str, str]:
        return dict(self._dependency_fingerprints)

    @property
    def fingerprint(self) -> str:
        value = self.bundle.value
        return canonical_sha256({
            "type": self.TYPE,
            "implementation_kind": value["implementation"]["kind"],
            "adapter": self.data_adapter.fingerprint,
            "runner": value.get("runner"),
            "trainer": value.get("trainer"),
            "model": value.get("model", {}),
            "bundle_digest": self.bundle.digest,
            "deployment_digest": self.bundle.deployment_digest,
            "runtime_dependencies": self.dependency_fingerprints,
        })

    def describe(self) -> dict[str, Any]:
        return {
            "type": self.TYPE,
            "fingerprint": self.fingerprint,
            "ownership": "benchmark_compiler_baseline_recipe",
            "build_is_side_effect_free": True,
            "canonical_plan_owner": "benchmark",
            "runtime_dependency_fingerprints": (
                self.dependency_fingerprints
            ),
            "data_adapter": self.data_adapter.describe(),
            "output": "BaselineTaskInstance",
        }

    def _validate_compatibility(
        self,
        task: TaskSpec,
        canonical_plan: AtomicPlan,
    ) -> None:
        value = self.bundle.value
        capabilities = value["capabilities"]
        if task.family not in capabilities["task_families"]:
            raise ValueError(
                f"Baseline does not support task family {task.family}"
            )
        if task.conditioning not in capabilities["conditioning"]:
            raise ValueError(
                f"Baseline does not support conditioning {task.conditioning}"
            )
        supported = value.get("supported_scenes", "all")
        if supported != "all":
            requested = set(canonical_plan.value["scene_ids"]) | {
                job["scene_id"] for job in canonical_plan.jobs
            }
            unknown = requested - set(supported)
            if unknown:
                raise ValueError(
                    f"Baseline does not support scenes {sorted(unknown)}"
                )
        if task.family == "finetune_eval" and "trainer" not in value:
            raise ValueError(
                "managed finetune_eval baseline requires trainer recipe"
            )
        if (
            value["implementation"]["kind"] == "submission"
            and task.family != "direct_eval"
        ):
            raise ValueError("submission Baselines only support direct_eval")

    def compile(
        self,
        dataset: DatasetSnapshot,
        task: TaskSpec,
        canonical_plan: AtomicPlan,
    ) -> BaselineTaskInstance:
        self._validate_compatibility(task, canonical_plan)
        by_id = {case["case_id"]: case for case in dataset.cases}
        train_ids = list(canonical_plan.train_case_ids)
        eval_ids = sorted({job["case_id"] for job in canonical_plan.jobs})
        selected_ids = sorted(set(train_ids) | set(eval_ids))
        adaptations: list[dict[str, Any]] = []
        adaptation_by_key: dict[tuple[str, str], dict[str, Any]] = {}
        for role, case_ids in (("train", train_ids), ("eval", eval_ids)):
            for case_id in case_ids:
                adaptation = self.data_adapter.adapt_case(
                    by_id[case_id], task.conditioning, role=role
                )
                adaptation_id = (
                    f"{case_id}::{role}::{task.conditioning}"
                )
                adaptation["adaptation_id"] = adaptation_id
                adaptations.append(adaptation)
                adaptation_by_key[(case_id, role)] = adaptation
        adaptations.sort(key=lambda item: item["adaptation_id"])

        model_ref = (
            "artifact://train/model"
            if task.family == "finetune_eval"
            else "baseline://frozen_model"
        )
        inference_jobs = []
        for job in canonical_plan.jobs:
            adaptation = adaptation_by_key[(job["case_id"], "eval")]
            inference_jobs.append({
                **job,
                "adaptation_id": adaptation["adaptation_id"],
                "model_ref": model_ref,
                "native_inputs": adaptation["native_inputs"],
            })
        inference_jobs.sort(key=lambda item: item["job_id"])

        training = None
        operations = []
        if task.family == "finetune_eval":
            training = {
                "operation_id": "train",
                "case_ids": train_ids,
                "adaptation_ids": [
                    adaptation_by_key[(case_id, "train")][
                        "adaptation_id"
                    ]
                    for case_id in train_ids
                ],
                "seed": canonical_plan.value["training_seed"],
                "trainer": self.bundle.value["trainer"],
                "outputs": {"model": "artifact://train/model"},
            }
            operations.append({
                "operation_id": "train",
                "kind": "train",
                "depends_on": [],
                "outputs": ["artifact://train/model"],
            })
        operations.extend([
            {
                "operation_id": "infer",
                "kind": "infer",
                "depends_on": ["train"] if training else [],
                "model_ref": model_ref,
                "job_ids": [job["job_id"] for job in inference_jobs],
            },
            {
                "operation_id": "evaluate",
                "kind": "evaluate",
                "depends_on": ["infer"],
                "job_ids": [job["job_id"] for job in inference_jobs],
            },
        ])

        cache_root = (
            Path(__file__).resolve().parents[3]
            / "cache"
            / "baselines"
            / self.bundle.baseline_id
            / self.data_adapter.materialization_fingerprint
            / dataset.digest
        )
        instance_id = (
            f"{task.task_id}__{self.bundle.baseline_id}"
            f"__{self.fingerprint[:12]}"
        )
        value = self.bundle.value
        return BaselineTaskInstance.seal({
            "schema_version": "2.1",
            "instance_id": instance_id,
            "identity": {
                "dataset": {
                    "dataset_id": dataset.dataset_id,
                    "digest": dataset.digest,
                },
                "task": {
                    "task_id": task.task_id,
                    "digest": task.digest,
                },
                "baseline": {
                    "baseline_id": self.bundle.baseline_id,
                    "baseline_version": self.bundle.baseline_version,
                    "digest": self.bundle.digest,
                    "deployment_digest": self.bundle.deployment_digest,
                },
                "task_builder": {
                    "type": self.TYPE,
                    "fingerprint": self.fingerprint,
                },
                "data_adapter": {
                    "fingerprint": self.data_adapter.fingerprint,
                    "materialization_fingerprint": (
                        self.data_adapter.materialization_fingerprint
                    ),
                },
                "canonical_plan_digest": canonical_sha256(
                    canonical_plan.value
                ),
            },
            "semantics": {
                "family": task.family,
                "conditioning": task.conditioning,
                "scene_ids": canonical_plan.value["scene_ids"],
            },
            "canonical_plan": canonical_plan.value,
            "source": {
                "asset_root": str(dataset.asset_root),
                "cases": [by_id[case_id] for case_id in selected_ids],
            },
            "adaptations": adaptations,
            "training": training,
            "inference": {
                "predictor": value.get("runner", {
                    "type": "submission_v1",
                    "config": {},
                }),
                "jobs": inference_jobs,
            },
            "execution_graph": {"operations": operations},
            "cache_bindings": [{
                "kind": "media_derivatives",
                "policy": value["adapter"].get(
                    "cache_policy",
                    "content_addressed_shared_immutable",
                ),
                "root": str(cache_root),
                "dataset_digest": dataset.digest,
                "materialization_fingerprint": (
                    self.data_adapter.materialization_fingerprint
                ),
            }],
            "baseline_payload": {
                "type": "managed_baseline_task_v1",
                "implementation": value["implementation"],
                "model": value.get("model", {}),
                "runtime": value.get("runtime", {}),
                "adapter": value["adapter"],
                "runner": value.get("runner"),
                "trainer": value.get("trainer"),
            },
        })
