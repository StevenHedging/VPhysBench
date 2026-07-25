from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from ..domain import (
    AtomicPlan,
    BaselineBundle,
    BaselineTaskInstance,
    DatasetSnapshot,
    TaskSpec,
)
from ..io import canonical_sha256, load_json, write_json
from .interfaces import BaselinePlugin, DataAdapter, TaskBuilder


PROTOCOL = "physbench-baseline-v1"


def _read_log_tail(path: Path, max_bytes: int = 4000) -> str:
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - max_bytes))
        return handle.read().decode("utf-8", errors="replace").strip()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _bundle_document(bundle: BaselineBundle) -> dict[str, Any]:
    return {
        "root": str(bundle.root),
        "value": bundle.value,
        "digest": bundle.digest,
        "deployment_digest": bundle.deployment_digest,
        "descriptor_path": str(bundle.descriptor_path),
    }


def _dataset_document(dataset: DatasetSnapshot) -> dict[str, Any]:
    return {
        "root": str(dataset.root),
        "descriptor": dataset.descriptor,
        "cases": list(dataset.cases),
        "views": dataset.views,
        "scene_configs": dataset.scene_configs,
        "asset_lock": dataset.asset_lock,
        "digest": dataset.digest,
        "asset_root": str(dataset.asset_root),
    }


def _task_document(task: TaskSpec) -> dict[str, Any]:
    return {
        "path": str(task.path),
        "value": task.value,
        "digest": task.digest,
    }


class CommandProtocolError(RuntimeError):
    pass


class _CommandClient:
    def __init__(self, bundle: BaselineBundle):
        self.bundle = bundle
        self.command = self._resolve_command()

    def _bundle_path(self, raw: str, *, label: str) -> Path:
        relative = Path(raw)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"{label} must be bundle-relative: {raw}")
        resolved = (self.bundle.root / relative).resolve()
        try:
            resolved.relative_to(self.bundle.root)
        except ValueError as exc:
            raise ValueError(f"{label} escapes Baseline bundle: {raw}") from exc
        return resolved

    def _resolve_command(self) -> list[str]:
        raw = self.bundle.value["implementation"]["entrypoint"]
        if raw[0] == "{python}":
            script = self._bundle_path(raw[1], label="command entrypoint")
            command = [sys.executable, str(script), *raw[2:]]
        else:
            executable = self._bundle_path(raw[0], label="command entrypoint")
            command = [str(executable), *raw[1:]]
        return command

    def invoke(self, operation: str, payload: dict[str, Any]) -> Any:
        request = {
            "protocol": PROTOCOL,
            "operation": operation,
            "bundle": _bundle_document(self.bundle),
            "payload": payload,
        }
        with tempfile.TemporaryDirectory(prefix="physbench-baseline-") as temporary:
            root = Path(temporary)
            request_path = root / "request.json"
            response_path = root / "response.json"
            write_json(request_path, request)
            command = [
                *self.command,
                "--request",
                str(request_path),
                "--response",
                str(response_path),
            ]
            environment = os.environ.copy()
            # The command runs from the Bundle root. Preserve the caller's
            # import meaning when PYTHONPATH contains relative entries such as
            # the repository-standard ``PYTHONPATH=src``.
            if environment.get("PYTHONPATH"):
                caller_root = Path.cwd()
                environment["PYTHONPATH"] = os.pathsep.join(
                    str(
                        entry
                        if (entry := Path(item)).is_absolute()
                        else (caller_root / entry).resolve()
                    )
                    for item in environment["PYTHONPATH"].split(os.pathsep)
                    if item
                )
            if operation == "run_task":
                log_root = Path(payload["run_dir"]) / "logs" / "baseline_command"
                log_root.mkdir(parents=True, exist_ok=True)
            else:
                log_root = root
            stdout_path = log_root / f"{operation}.stdout.log"
            stderr_path = log_root / f"{operation}.stderr.log"
            with stdout_path.open("w", encoding="utf-8") as stdout, (
                stderr_path.open("w", encoding="utf-8")
            ) as stderr:
                completed = subprocess.run(
                    command,
                    cwd=self.bundle.root,
                    env=environment,
                    text=True,
                    stdout=stdout,
                    stderr=stderr,
                    check=False,
                )
            command_output = _read_log_tail(stderr_path) or _read_log_tail(
                stdout_path
            )
            if not response_path.is_file():
                raise CommandProtocolError(
                    f"Baseline command produced no response for {operation}; "
                    f"return_code={completed.returncode}; "
                    f"output={command_output[-4000:]}"
                )
            response = load_json(response_path)
            if response.get("protocol") != PROTOCOL:
                raise CommandProtocolError(
                    f"Baseline command returned an invalid protocol for {operation}"
                )
            if response.get("operation") != operation:
                raise CommandProtocolError(
                    f"Baseline command response operation mismatch: "
                    f"expected={operation}, actual={response.get('operation')}"
                )
            if completed.returncode != 0 or response.get("ok") is not True:
                error = response.get("error", {})
                message = (
                    error.get("message")
                    if isinstance(error, dict)
                    else str(error)
                )
                raise CommandProtocolError(
                    f"Baseline command failed during {operation}: "
                    f"{message or command_output[-4000:] or completed.returncode}"
                )
            if "result" not in response:
                raise CommandProtocolError(
                    f"Baseline command omitted result for {operation}"
                )
            return response["result"]


class CommandDataAdapterProxy(DataAdapter):
    def __init__(
        self,
        client: _CommandClient,
        description: dict[str, Any],
    ):
        self._client = client
        self._description = description
        for key in ("fingerprint", "materialization_fingerprint"):
            if not _is_sha256(description.get(key)):
                raise CommandProtocolError(
                    f"Baseline describe.data_adapter requires SHA-256 {key}"
                )

    @property
    def fingerprint(self) -> str:
        return self._description["fingerprint"]

    @property
    def materialization_fingerprint(self) -> str:
        return self._description["materialization_fingerprint"]

    def describe(self) -> dict[str, Any]:
        return self._description

    def adapt_case(
        self, case: dict[str, Any], conditioning: str, *, role: str
    ) -> dict[str, Any]:
        result = self._client.invoke("adapt_case", {
            "case": case,
            "conditioning": conditioning,
            "role": role,
        })
        if not isinstance(result, dict):
            raise CommandProtocolError("adapt_case result must be an object")
        return result


class CommandTaskBuilderProxy(TaskBuilder):
    def __init__(
        self,
        bundle: BaselineBundle,
        client: _CommandClient,
        description: dict[str, Any],
        data_adapter: CommandDataAdapterProxy,
    ):
        self.bundle = bundle
        self._client = client
        self._description = description
        self.data_adapter = data_adapter
        if not _is_sha256(description.get("fingerprint")):
            raise CommandProtocolError(
                "Baseline describe.task_builder requires SHA-256 fingerprint"
            )

    @property
    def fingerprint(self) -> str:
        return self._description["fingerprint"]

    def describe(self) -> dict[str, Any]:
        return self._description

    def compile(
        self,
        dataset: DatasetSnapshot,
        task: TaskSpec,
        canonical_plan: AtomicPlan,
    ) -> BaselineTaskInstance:
        capabilities = self.bundle.value["capabilities"]
        if task.family not in capabilities["task_families"]:
            raise ValueError(
                f"Baseline does not support task family {task.family}"
            )
        if task.conditioning not in capabilities["conditioning"]:
            raise ValueError(
                f"Baseline does not support conditioning {task.conditioning}"
            )
        supported_scenes = self.bundle.value.get("supported_scenes", "all")
        if supported_scenes != "all":
            requested = set(canonical_plan.value.get("scene_ids", [])) | {
                job["scene_id"] for job in canonical_plan.jobs
            }
            unsupported = requested - set(supported_scenes)
            if unsupported:
                raise ValueError(
                    f"Baseline does not support scenes {sorted(unsupported)}"
                )
        result = self._client.invoke("build_task_instance", {
            "dataset": _dataset_document(dataset),
            "task": _task_document(task),
            "canonical_plan": canonical_plan.value,
        })
        if not isinstance(result, dict):
            raise CommandProtocolError(
                "build_task_instance result must be an object"
            )
        instance = BaselineTaskInstance.from_document(result)
        value = instance.value
        identity = value.get("identity", {})
        expected = {
            "dataset_id": dataset.dataset_id,
            "dataset_digest": dataset.digest,
            "task_id": task.task_id,
            "task_digest": task.digest,
            "baseline_id": self.bundle.baseline_id,
            "baseline_version": self.bundle.baseline_version,
            "baseline_digest": self.bundle.digest,
            "baseline_deployment_digest": self.bundle.deployment_digest,
            "task_builder_fingerprint": self.fingerprint,
            "data_adapter_fingerprint": self.data_adapter.fingerprint,
            "materialization_fingerprint": (
                self.data_adapter.materialization_fingerprint
            ),
            "canonical_plan_digest": canonical_sha256(canonical_plan.value),
        }
        actual = {
            "dataset_id": identity.get("dataset", {}).get("dataset_id"),
            "dataset_digest": identity.get("dataset", {}).get("digest"),
            "task_id": identity.get("task", {}).get("task_id"),
            "task_digest": identity.get("task", {}).get("digest"),
            "baseline_id": identity.get("baseline", {}).get("baseline_id"),
            "baseline_version": identity.get("baseline", {}).get(
                "baseline_version"
            ),
            "baseline_digest": identity.get("baseline", {}).get("digest"),
            "baseline_deployment_digest": identity.get("baseline", {}).get(
                "deployment_digest"
            ),
            "task_builder_fingerprint": identity.get("task_builder", {}).get(
                "fingerprint"
            ),
            "data_adapter_fingerprint": identity.get("data_adapter", {}).get(
                "fingerprint"
            ),
            "materialization_fingerprint": identity.get(
                "data_adapter", {}
            ).get("materialization_fingerprint"),
            "canonical_plan_digest": identity.get("canonical_plan_digest"),
        }
        if actual != expected:
            raise CommandProtocolError(
                "Baseline changed the frozen build identity: "
                f"expected={expected}, actual={actual}"
            )
        if value.get("canonical_plan") != canonical_plan.value:
            raise CommandProtocolError(
                "Baseline changed the Benchmark-owned canonical plan"
            )
        return instance


class CommandBaselinePlugin(BaselinePlugin):
    def __init__(self, bundle: BaselineBundle):
        self.bundle = bundle
        self._client = _CommandClient(bundle)
        description = self._client.invoke("describe", {})
        if not isinstance(description, dict):
            raise CommandProtocolError("describe result must be an object")
        task_builder_description = description.get("task_builder")
        data_adapter_description = description.get("data_adapter")
        if not isinstance(task_builder_description, dict):
            raise CommandProtocolError(
                "describe result requires task_builder object"
            )
        if not isinstance(data_adapter_description, dict):
            raise CommandProtocolError(
                "describe result requires data_adapter object"
            )
        adapter = CommandDataAdapterProxy(
            self._client, data_adapter_description
        )
        self.task_builder = CommandTaskBuilderProxy(
            bundle,
            self._client,
            task_builder_description,
            adapter,
        )

    def run_task(
        self,
        *,
        instance: BaselineTaskInstance,
        run_dir: Path,
        execute: bool,
        stop_after_training: bool,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        instance.verify()
        identity = instance.value.get("identity", {})
        baseline_identity = identity.get("baseline", {})
        builder_identity = identity.get("task_builder", {})
        adapter_identity = identity.get("data_adapter", {})
        if (
            baseline_identity.get("baseline_id") != self.bundle.baseline_id
            or baseline_identity.get("baseline_version")
            != self.bundle.baseline_version
            or baseline_identity.get("digest") != self.bundle.digest
            or baseline_identity.get("deployment_digest")
            != self.bundle.deployment_digest
        ):
            raise ValueError(
                "task instance targets a different Baseline deployment"
            )
        if (
            builder_identity.get("fingerprint")
            != self.task_builder.fingerprint
        ):
            raise ValueError(
                "task instance TaskBuilder fingerprint does not match deployment"
            )
        if (
            adapter_identity.get("fingerprint")
            != self.task_builder.data_adapter.fingerprint
            or adapter_identity.get("materialization_fingerprint")
            != self.task_builder.data_adapter.materialization_fingerprint
        ):
            raise ValueError(
                "task instance DataAdapter fingerprint does not match deployment"
            )
        result = self._client.invoke("run_task", {
            "instance": instance.value,
            "run_dir": str(run_dir.resolve()),
            "execute": execute,
            "stop_after_training": stop_after_training,
        })
        if not isinstance(result, dict):
            raise CommandProtocolError("run_task result must be an object")
        training = result.get("training")
        predictions = result.get("predictions")
        if not isinstance(training, dict):
            raise CommandProtocolError("run_task.training must be an object")
        if not isinstance(predictions, list) or any(
            not isinstance(item, dict) for item in predictions
        ):
            raise CommandProtocolError(
                "run_task.predictions must be an object list"
            )
        return training, predictions
