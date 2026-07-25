from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

from physbench.domain import (
    AtomicPlan,
    BaselineBundle,
    BaselineTaskInstance,
    DatasetSnapshot,
    TaskSpec,
)


PROTOCOL = "physbench-baseline-v1"


def _bundle(value: dict[str, Any]) -> BaselineBundle:
    return BaselineBundle(
        root=Path(value["root"]),
        value=value["value"],
        digest=value["digest"],
        deployment_digest=value["deployment_digest"],
        descriptor_path=Path(value["descriptor_path"]),
    )


def _dataset(value: dict[str, Any]) -> DatasetSnapshot:
    return DatasetSnapshot(
        root=Path(value["root"]),
        descriptor=value["descriptor"],
        cases=tuple(value["cases"]),
        views=value["views"],
        scene_configs=value["scene_configs"],
        asset_lock=value["asset_lock"],
        digest=value["digest"],
        asset_root=Path(value["asset_root"]),
    )


def _task(value: dict[str, Any]) -> TaskSpec:
    return TaskSpec(
        path=Path(value["path"]),
        value=value["value"],
        digest=value["digest"],
    )


def _dispatch(request: dict[str, Any], plugin_type: type) -> Any:
    if request.get("protocol") != PROTOCOL:
        raise ValueError(f"unsupported protocol {request.get('protocol')!r}")
    operation = request.get("operation")
    plugin = plugin_type(_bundle(request["bundle"]))
    payload = request.get("payload", {})
    if operation == "describe":
        return {
            "task_builder": plugin.task_builder.describe(),
            "data_adapter": plugin.task_builder.data_adapter.describe(),
        }
    if operation == "adapt_case":
        return plugin.task_builder.data_adapter.adapt_case(
            payload["case"],
            payload["conditioning"],
            role=payload["role"],
        )
    if operation == "build_task_instance":
        instance = plugin.task_builder.compile(
            _dataset(payload["dataset"]),
            _task(payload["task"]),
            AtomicPlan(payload["canonical_plan"]),
        )
        return instance.value
    if operation == "run_task":
        training, predictions = plugin.run_task(
            instance=BaselineTaskInstance.from_document(payload["instance"]),
            run_dir=Path(payload["run_dir"]),
            execute=bool(payload["execute"]),
            stop_after_training=bool(payload["stop_after_training"]),
        )
        return {"training": training, "predictions": predictions}
    raise ValueError(f"unsupported operation {operation!r}")


def main(plugin_type: type) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--response", required=True)
    args = parser.parse_args()
    request_path = Path(args.request)
    response_path = Path(args.response)
    operation: Any = None
    try:
        with request_path.open(encoding="utf-8") as handle:
            request = json.load(handle)
        operation = request.get("operation")
        response = {
            "protocol": PROTOCOL,
            "operation": operation,
            "ok": True,
            "result": _dispatch(request, plugin_type),
        }
        return_code = 0
    except BaseException as exc:
        response = {
            "protocol": PROTOCOL,
            "operation": operation,
            "ok": False,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        }
        return_code = 1
    response_path.write_text(
        json.dumps(response, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return return_code
