from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetSnapshot:
    root: Path
    descriptor: dict[str, Any]
    cases: tuple[dict[str, Any], ...]
    views: dict[str, dict[str, Any]]
    scene_configs: dict[str, dict[str, Any]]
    asset_lock: dict[str, Any] | None
    digest: str
    asset_root: Path

    @property
    def dataset_id(self) -> str:
        return str(self.descriptor["dataset_id"])


@dataclass(frozen=True)
class TaskSpec:
    path: Path
    value: dict[str, Any]
    digest: str

    @property
    def task_id(self) -> str:
        return str(self.value["task_id"])

    @property
    def family(self) -> str:
        return str(self.value["family"])

    @property
    def conditioning(self) -> str:
        return str(self.value["conditioning"])


@dataclass(frozen=True)
class BaselineBundle:
    root: Path
    value: dict[str, Any]
    digest: str

    @property
    def baseline_id(self) -> str:
        return str(self.value["baseline_id"])


@dataclass(frozen=True)
class AtomicPlan:
    value: dict[str, Any]

    @property
    def train_case_ids(self) -> list[str]:
        return list(self.value["train_case_ids"])

    @property
    def jobs(self) -> list[dict[str, Any]]:
        return list(self.value["jobs"])


@dataclass(frozen=True)
class BaselineTaskInstance:
    """Immutable, canonical JSON task compiled for one Baseline.

    The canonical JSON string is the source of truth. ``value`` returns a fresh
    object on every access, preventing executors from mutating the sealed
    instance in memory.
    """

    _canonical_json: str
    digest: str

    @classmethod
    def seal(cls, value: dict[str, Any]) -> "BaselineTaskInstance":
        document = dict(value)
        document.pop("instance_digest", None)
        payload = json.dumps(
            document,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        document["instance_digest"] = digest
        canonical = json.dumps(
            document,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return cls(canonical, digest)

    @classmethod
    def from_document(
        cls, value: dict[str, Any]
    ) -> "BaselineTaskInstance":
        recorded = value.get("instance_digest")
        instance = cls.seal(value)
        if recorded != instance.digest:
            raise ValueError(
                f"BaselineTaskInstance document was modified: "
                f"recorded={recorded}, actual={instance.digest}"
            )
        return instance

    @property
    def value(self) -> dict[str, Any]:
        value = json.loads(self._canonical_json)
        if not isinstance(value, dict):
            raise TypeError("BaselineTaskInstance must contain a JSON object")
        return value

    @property
    def instance_id(self) -> str:
        return str(self.value["instance_id"])

    @property
    def canonical_plan(self) -> AtomicPlan:
        return AtomicPlan(self.value["canonical_plan"])

    def verify(self) -> None:
        value = self.value
        recorded = value.pop("instance_digest", None)
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        actual = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        if recorded != self.digest or actual != self.digest:
            raise ValueError(
                f"BaselineTaskInstance digest mismatch: "
                f"recorded={recorded}, expected={self.digest}, actual={actual}"
            )
