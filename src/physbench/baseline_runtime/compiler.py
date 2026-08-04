from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

from ..baseline_api.interfaces import DataAdapter, TaskBuilder
from ..domain import (
    AtomicPlan,
    BaselineBundle,
    BaselineTaskInstance,
    DatasetSnapshot,
    TaskSpec,
)
from ..io import canonical_sha256
from .input_contract import (
    FORBIDDEN_ASSET_KEYS,
    resolve_binding,
    resolve_dataset_asset_path,
    validate_adaptation_record,
)
from .media_contract import probe_media


_NON_RUNTIME_ASSET_KEYS = {
    *FORBIDDEN_ASSET_KEYS,
    "source_archive",
}


class ManagedTaskBuilder(TaskBuilder):
    """Core compiler for declarative managed and submission Baselines."""

    TYPE = "managed_task_builder_v1"

    def __init__(
        self,
        bundle: BaselineBundle,
        dependency_fingerprints: dict[str, str],
        data_adapter: DataAdapter,
    ):
        self.bundle = bundle
        self.data_adapter = data_adapter
        self._dependency_fingerprints = dict(
            sorted(dependency_fingerprints.items())
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

    @staticmethod
    def _adapter_case(
        case: dict[str, Any],
        *,
        target_physical_duration_s: float | None = None,
    ) -> dict[str, Any]:
        """Project a Case onto assets and facts available to an adapter."""
        projected = {
            key: copy.deepcopy(case[key])
            for key in (
                "schema_version",
                "case_id",
                "scene_id",
                "text",
                "appearance",
                "temporal",
                "ood",
            )
            if key in case
        }
        projected["assets"] = {
            key: copy.deepcopy(value)
            for key, value in case["assets"].items()
            if key not in _NON_RUNTIME_ASSET_KEYS
        }
        projected["physics"] = {
            name: copy.deepcopy(quantity)
            for name, quantity in case["physics"].items()
            if quantity.get("annotated") is True
        }
        if target_physical_duration_s is not None:
            projected.setdefault("temporal", {})[
                "target_physical_duration_s"
            ] = target_physical_duration_s
        # Every Baseline sees the same conditionable Case. None receives
        # provenance, source locators, alignment evidence, non-annotated
        # derived quantities, or evaluator-only assets.  The target physical
        # duration is an explicit output requirement, not a reference asset.
        return projected

    @staticmethod
    def _runtime_case(
        case: dict[str, Any],
        *,
        training_target: bool,
        target_physical_duration_s: float | None = None,
    ) -> dict[str, Any]:
        """Project source data embedded in the runnable TaskInstance."""
        projected = ManagedTaskBuilder._adapter_case(
            case,
            target_physical_duration_s=target_physical_duration_s,
        )
        if training_target:
            target_key = (
                "reference_video"
                if case["assets"].get("reference_video")
                else "physics_reference_video"
            )
            target = case["assets"].get(target_key)
            if not target:
                raise ValueError(
                    f"training case {case['case_id']} has no supervised "
                    "video target"
                )
            projected["supervised_targets"] = {
                "video": {
                    "asset_key": target_key,
                    "asset": target,
                    "role": "training_target_only",
                }
            }
        return projected

    @staticmethod
    def _target_physical_duration(
        case: dict[str, Any],
        *,
        catalog: dict[str, dict[str, Any]],
        asset_root: Path,
    ) -> float | None:
        """Resolve the frozen reference duration exposed as an output target.

        Current datasets store the value in ``case.temporal``.  Probing is a
        compatibility fallback for older snapshots that still have their
        reference media locally available; no reference path or pixels enter
        the compiled Baseline task.
        """

        declared = case.get("temporal", {}).get(
            "target_physical_duration_s"
        )
        if declared is not None:
            if (
                isinstance(declared, bool)
                or not isinstance(declared, (int, float))
                or not math.isfinite(float(declared))
                or float(declared) <= 0.0
            ):
                raise ValueError(
                    f"case {case['case_id']} has invalid "
                    "temporal.target_physical_duration_s"
                )
            return float(declared)

        reference_case = case
        value = case.get("assets", {}).get("physics_reference_video")
        if not value or not case.get("has_real_reference_video", False):
            parent_id = case.get("provenance", {}).get("parent_case_id")
            reference_case = catalog.get(parent_id) if parent_id else None
            if reference_case is None:
                return None
            if reference_case.get("physics") != case.get("physics"):
                raise ValueError(
                    f"case {case['case_id']} has a physics-mismatched "
                    "reference parent"
                )
            value = reference_case.get("assets", {}).get(
                "physics_reference_video"
            )
        if not isinstance(value, str) or not value:
            return None
        path = resolve_dataset_asset_path(
            asset_root,
            value,
            label=f"case {case['case_id']} physics reference",
        )
        if not path.is_file():
            return None
        probe = probe_media(path, count_frames=False)
        if probe.get("frames") is None:
            probe = probe_media(path, count_frames=True)
        frames, fps = probe.get("frames"), probe.get("fps")
        if frames is None or int(frames) < 2 or fps is None:
            raise ValueError(
                f"case {case['case_id']} reference duration cannot be "
                "resolved from frames and FPS"
            )
        scale = float(
            reference_case.get("temporal", {}).get(
                "encoded_to_physical_speed", 1.0
            )
        )
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(
                f"case {case['case_id']} reference time scale is invalid"
            )
        return float((int(frames) - 1) / float(fps) / scale)

    @staticmethod
    def _validate_asset_access(
        adaptation: dict[str, Any],
        case: dict[str, Any],
        *,
        asset_root: Path,
        asset_digests: dict[str, str],
        forbidden_media_paths: set[Path],
        forbidden_media_digests: set[str],
    ) -> None:
        requested = adaptation["input_contract"]["asset_access"]
        missing = [
            key for key in requested if not case["assets"].get(key)
        ]
        if missing:
            raise ValueError(
                f"adaptation for {case['case_id']} requests unavailable "
                f"assets: {sorted(missing)}"
            )
        for key in requested:
            resolve_dataset_asset_path(
                asset_root,
                case["assets"][key],
                label=f"assets.{key}",
            )
        for channel in adaptation["input_contract"]["media_channels"]:
            if channel.get("origin", "dataset_asset") != "dataset_asset":
                continue
            expected = case["assets"][channel["asset_key"]]
            actual = resolve_binding(adaptation, channel["binding"])
            if actual != expected:
                raise ValueError(
                    f"adaptation for {case['case_id']} binds "
                    f"assets.{channel['asset_key']} to an unrelated media "
                    f"value: {actual!r}"
                )
        if adaptation["input_contract"]["generation_mode"] == "v2v":
            for channel in adaptation["input_contract"]["media_channels"]:
                if channel["kind"] != "video":
                    continue
                origin = channel.get("origin", "dataset_asset")
                if origin == "dataset_asset":
                    selected = case["assets"][channel["asset_key"]]
                    selected_path = resolve_dataset_asset_path(
                        asset_root,
                        selected,
                        label=f"assets.{channel['asset_key']}",
                    )
                    selected_digest = asset_digests.get(selected)
                    aliases_reference = (
                        selected_path in forbidden_media_paths
                        or (
                            selected_digest is not None
                            and selected_digest
                            in forbidden_media_digests
                        )
                    )
                    label = channel["asset_key"]
                else:
                    selected_digest = channel[
                        "artifact_provenance"
                    ]["content_sha256"]
                    aliases_reference = (
                        selected_digest in forbidden_media_digests
                    )
                    label = channel["id"]
                if aliases_reference:
                    raise ValueError(
                        f"V2V conditioning media {label!r} aliases an "
                        "evaluator/source video anywhere in the Dataset"
                    )

    @staticmethod
    def _validate_physics_access(
        adaptation: dict[str, Any],
        case: dict[str, Any],
    ) -> None:
        physics = case.get("physics", {})
        invalid = [
            name
            for name in adaptation["used_parameters"]
            if (
                name not in physics
                or physics[name].get("annotated") is not True
            )
        ]
        if invalid:
            raise ValueError(
                f"adaptation for {case['case_id']} uses missing or "
                f"non-annotated physics fields: {sorted(invalid)}"
            )
        mismatched = []
        for name, recorded in adaptation["used_parameters"].items():
            source = physics[name]
            if (
                not isinstance(recorded, dict)
                or recorded.get("value") != source.get("value")
                or recorded.get("unit") != source.get("unit")
            ):
                mismatched.append(name)
        if mismatched:
            raise ValueError(
                f"adaptation for {case['case_id']} records physics values "
                f"or units that differ from Dataset truth: "
                f"{sorted(mismatched)}"
            )

    def _validate_artifact_producers(
        self,
        adaptation: dict[str, Any],
    ) -> None:
        """Bind declared artifacts to an active, fingerprinted producer."""

        accepted = {
            self.data_adapter.fingerprint,
            self.data_adapter.materialization_fingerprint,
            self.fingerprint,
        }
        invalid = sorted(
            channel["id"]
            for channel in (
                *adaptation["input_contract"]["media_channels"],
                *adaptation["input_contract"]["physics_channels"],
            )
            if (
                (provenance := channel.get("artifact_provenance"))
                is not None
                and provenance["producer_fingerprint"] not in accepted
            )
        )
        if invalid:
            raise ValueError(
                "artifact producer_fingerprint must identify the active "
                "DataAdapter, media materializer, or managed TaskBuilder; "
                f"invalid channels: {invalid}"
            )

    def _predictor_payload(self) -> dict[str, Any]:
        return copy.deepcopy(self.bundle.value.get("runner", {
            "type": "submission_v1",
            "config": {},
        }))

    def _baseline_payload(self) -> dict[str, Any]:
        value = self.bundle.value
        return copy.deepcopy({
            "type": "managed_baseline_task_v1",
            "implementation": value["implementation"],
            "model": value.get("model", {}),
            "runtime": value.get("runtime", {}),
            "adapter": value["adapter"],
            "input_policy": value["input_policy"],
            "runner": value.get("runner"),
            "trainer": value.get("trainer"),
        })

    def _cache_binding(self, dataset_digest: str) -> dict[str, Any]:
        value = self.bundle.value
        cache_root = (
            Path(__file__).resolve().parents[3]
            / "cache"
            / "baseline_materializations"
            / self.data_adapter.materialization_fingerprint
            / dataset_digest
        )
        return {
            "kind": "media_derivatives",
            "policy": value["adapter"].get(
                "cache_policy",
                "content_addressed_shared_immutable",
            ),
            "root": str(cache_root),
            "dataset_digest": dataset_digest,
            "materialization_fingerprint": (
                self.data_adapter.materialization_fingerprint
            ),
        }

    def validate_compiled_instance(
        self,
        document: dict[str, Any],
    ) -> None:
        """Revalidate managed-only contracts against the active deployment."""

        plan = document["canonical_plan"]
        train_ids = list(plan["train_case_ids"])
        eval_ids = sorted({
            job["case_id"] for job in plan["jobs"]
        })
        train_id_set = set(train_ids)
        expected_source_ids = train_id_set | set(eval_ids)
        source_cases = document["source"]["cases"]
        source_by_id = {
            case["case_id"]: case for case in source_cases
        }
        if set(source_by_id) != expected_source_ids:
            raise ValueError(
                "managed source cases must exactly match canonical train/eval "
                "cases"
            )

        allowed_case_fields = {
            "schema_version",
            "case_id",
            "scene_id",
            "text",
            "physics",
            "appearance",
            "temporal",
            "ood",
            "assets",
            "supervised_targets",
        }
        for case_id, case in source_by_id.items():
            unexpected = sorted(set(case) - allowed_case_fields)
            if unexpected:
                raise ValueError(
                    f"managed source case {case_id} contains forbidden or "
                    f"unknown fields: {unexpected}"
                )
            assets = case.get("assets")
            if not isinstance(assets, dict):
                raise ValueError(
                    f"managed source case {case_id} requires assets"
                )
            text = case.get("text")
            if (
                not isinstance(text, dict)
                or not isinstance(text.get("prompt"), str)
                or not text["prompt"].strip()
            ):
                raise ValueError(
                    f"managed source case {case_id} requires text.prompt"
                )
            physics = case.get("physics")
            if (
                not isinstance(physics, dict)
                or not physics
                or any(
                    not isinstance(quantity, dict)
                    or quantity.get("annotated") is not True
                    for quantity in physics.values()
                )
            ):
                raise ValueError(
                    f"managed source case {case_id} requires annotated physics"
                )
            leaked_assets = sorted(
                set(assets) & _NON_RUNTIME_ASSET_KEYS
            )
            if leaked_assets:
                raise ValueError(
                    f"managed source case {case_id} exposes evaluator/source "
                    f"assets: {leaked_assets}"
                )
            target = case.get("supervised_targets")
            if case_id not in train_id_set:
                if target is not None:
                    raise ValueError(
                        f"evaluation case {case_id} exposes supervised targets"
                    )
                continue
            if (
                not isinstance(target, dict)
                or set(target) != {"video"}
                or not isinstance(target["video"], dict)
                or set(target["video"])
                != {"asset_key", "asset", "role"}
                or target["video"].get("asset_key")
                not in {"reference_video", "physics_reference_video"}
                or not isinstance(target["video"].get("asset"), str)
                or not target["video"]["asset"]
                or target["video"].get("role") != "training_target_only"
            ):
                raise ValueError(
                    f"training case {case_id} has invalid supervised target"
                )

        expected_adaptations = {
            f"{case_id}__{role}": (case_id, role)
            for role, case_ids in (
                ("train", train_ids),
                ("eval", eval_ids),
            )
            for case_id in case_ids
        }
        adaptations = {
            item["adaptation_id"]: item
            for item in document["adaptations"]
        }
        if set(adaptations) != set(expected_adaptations):
            raise ValueError(
                "managed adaptations must exactly match canonical "
                "train/eval cases"
            )
        capabilities = self.bundle.value["capabilities"]
        declared_modes = capabilities.get("generation_modes")
        for adaptation_id, (case_id, role) in (
            expected_adaptations.items()
        ):
            adaptation = adaptations[adaptation_id]
            expected_identity = {
                "case_id": case_id,
                "role": role,
            }
            actual_identity = {
                key: adaptation.get(key)
                for key in expected_identity
            }
            if actual_identity != expected_identity:
                raise ValueError(
                    "managed adaptation identity mismatch: "
                    f"expected={expected_identity}, actual={actual_identity}"
                )
            validate_adaptation_record(
                adaptation,
                input_policy=self.bundle.value["input_policy"],
            )
            self._validate_artifact_producers(adaptation)
            mode = adaptation["input_contract"]["generation_mode"]
            if declared_modes is not None and mode not in declared_modes:
                raise ValueError(
                    f"managed adaptation mode {mode!r} is not declared by "
                    "the active Baseline"
                )
            self._validate_asset_access(
                adaptation,
                source_by_id[case_id],
                asset_root=Path(document["source"]["asset_root"]),
                asset_digests={},
                forbidden_media_paths=set(),
                forbidden_media_digests=set(),
            )
            self._validate_physics_access(
                adaptation,
                source_by_id[case_id],
            )

        if document["inference"]["predictor"] != self._predictor_payload():
            raise ValueError(
                "managed inference predictor differs from active Baseline "
                "runner recipe"
            )
        training = document["training"]
        if (
            training is not None
            and training.get("trainer") != self.bundle.value.get("trainer")
        ):
            raise ValueError(
                "managed trainer differs from active Baseline recipe"
            )
        if document["baseline_payload"] != self._baseline_payload():
            raise ValueError(
                "managed baseline payload differs from active deployment"
            )
        expected_cache = self._cache_binding(
            document["identity"]["dataset"]["digest"]
        )
        if document["cache_bindings"] != [expected_cache]:
            raise ValueError(
                "managed cache binding differs from active deployment"
            )

    def compile(
        self,
        dataset: DatasetSnapshot,
        task: TaskSpec,
        canonical_plan: AtomicPlan,
    ) -> BaselineTaskInstance:
        self._validate_compatibility(task, canonical_plan)
        by_id = {case["case_id"]: case for case in dataset.cases}
        train_ids = list(canonical_plan.train_case_ids)
        train_id_set = set(train_ids)
        asset_digests = {
            item["path"]: item["sha256"]
            for item in (
                dataset.asset_lock.get("files", [])
                if dataset.asset_lock is not None
                else []
            )
        }
        forbidden_asset_values = {
            value
            for raw_case in dataset.cases
            for key in FORBIDDEN_ASSET_KEYS
            if (value := raw_case["assets"].get(key))
        }
        forbidden_media_paths = {
            (dataset.asset_root / value).resolve()
            for value in forbidden_asset_values
        }
        forbidden_media_digests = {
            digest
            for value in forbidden_asset_values
            if (digest := asset_digests.get(value))
        }
        eval_ids = sorted({job["case_id"] for job in canonical_plan.jobs})
        overlap = train_id_set & set(eval_ids)
        if overlap:
            raise ValueError(
                "managed canonical plan exposes training targets to "
                f"evaluation cases: {sorted(overlap)}"
            )
        selected_ids = sorted(set(train_ids) | set(eval_ids))
        target_duration_by_case = {
            case_id: self._target_physical_duration(
                by_id[case_id],
                catalog=by_id,
                asset_root=dataset.asset_root,
            )
            for case_id in selected_ids
        }
        adaptations: list[dict[str, Any]] = []
        adaptation_by_key: dict[tuple[str, str], dict[str, Any]] = {}
        for role, case_ids in (("train", train_ids), ("eval", eval_ids)):
            for case_id in case_ids:
                case = self._adapter_case(
                    by_id[case_id],
                    target_physical_duration_s=(
                        target_duration_by_case[case_id]
                    ),
                )
                adapter_case = copy.deepcopy(case)
                adaptation = self.data_adapter.adapt_case(
                    adapter_case, role=role
                )
                if adapter_case != case:
                    raise ValueError(
                        f"DataAdapter mutated its immutable Case input: "
                        f"{case_id}"
                    )
                expected_identity = {
                    "case_id": case_id,
                    "role": role,
                }
                actual_identity = {
                    key: adaptation.get(key)
                    for key in expected_identity
                }
                if actual_identity != expected_identity:
                    raise ValueError(
                        "DataAdapter changed or omitted adaptation identity: "
                        f"expected={expected_identity}, "
                        f"actual={actual_identity}"
                    )
                validate_adaptation_record(
                    adaptation,
                    input_policy=self.bundle.value["input_policy"],
                )
                declared_modes = self.bundle.value[
                    "capabilities"
                ].get("generation_modes")
                actual_mode = adaptation["input_contract"][
                    "generation_mode"
                ]
                if (
                    declared_modes is not None
                    and actual_mode not in declared_modes
                ):
                    raise ValueError(
                        f"adapter generated mode {actual_mode!r}, which is "
                        "not declared in capabilities.generation_modes"
                    )
                self._validate_asset_access(
                    adaptation,
                    case,
                    asset_root=dataset.asset_root,
                    asset_digests=asset_digests,
                    forbidden_media_paths=forbidden_media_paths,
                    forbidden_media_digests=forbidden_media_digests,
                )
                self._validate_physics_access(
                    adaptation,
                    case,
                )
                self._validate_artifact_producers(adaptation)
                adaptation_id = f"{case_id}__{role}"
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

        instance_id = (
            f"{task.task_id}__{self.bundle.baseline_id}"
            f"__{self.fingerprint[:12]}"
        )
        value = self.bundle.value
        return BaselineTaskInstance.seal({
            "schema_version": "3.0",
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
                "scene_ids": canonical_plan.value["scene_ids"],
            },
            "canonical_plan": canonical_plan.value,
            "source": {
                "asset_root": str(dataset.asset_root),
                "cases": [
                    self._runtime_case(
                        by_id[case_id],
                        training_target=case_id in train_id_set,
                        target_physical_duration_s=(
                            target_duration_by_case[case_id]
                        ),
                    )
                    for case_id in selected_ids
                ],
            },
            "adaptations": adaptations,
            "training": training,
            "inference": {
                "predictor": self._predictor_payload(),
                "jobs": inference_jobs,
            },
            "execution_graph": {"operations": operations},
            "cache_bindings": [self._cache_binding(dataset.digest)],
            "baseline_payload": self._baseline_payload(),
        })
