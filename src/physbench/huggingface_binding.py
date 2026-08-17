from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


_FIELDS = {
    "schema_version",
    "provider",
    "repo_type",
    "repo_id",
    "revision",
    "dataset_id",
    "release",
    "delivery",
}
_REQUIRED_FIELDS = _FIELDS - {"delivery"}
_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class HuggingFaceDatasetBinding:
    repo_id: str
    revision: str
    dataset_id: str
    release: str
    delivery: str = "distribution_v1"


def load_huggingface_dataset_binding(
    path: str | Path,
    *,
    dataset_path: str | Path | None = None,
) -> HuggingFaceDatasetBinding:
    binding_path = Path(path)
    value = json.loads(binding_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Hugging Face Dataset binding must be a JSON object")

    unknown = sorted(set(value) - _FIELDS)
    if unknown:
        raise ValueError(f"Hugging Face Dataset binding has unknown fields: {unknown}")
    missing = sorted(_REQUIRED_FIELDS - set(value))
    if missing:
        raise ValueError(f"Hugging Face Dataset binding is missing fields: {missing}")

    if value["schema_version"] != "1.0":
        raise ValueError("Hugging Face Dataset binding requires schema_version=1.0")
    if value["provider"] != "huggingface" or value["repo_type"] != "dataset":
        raise ValueError("Hugging Face Dataset binding must target a Dataset repo")

    repo_id = value["repo_id"]
    if (
        not isinstance(repo_id, str)
        or repo_id.count("/") != 1
        or any(not part for part in repo_id.split("/"))
    ):
        raise ValueError("Hugging Face Dataset binding has an invalid repo_id")

    revision = value["revision"]
    if not isinstance(revision, str) or _COMMIT_PATTERN.fullmatch(revision) is None:
        raise ValueError("Hugging Face Dataset binding revision must be a commit SHA")

    dataset_id = value["dataset_id"]
    release = value["release"]
    if not isinstance(dataset_id, str) or not dataset_id:
        raise ValueError("Hugging Face Dataset binding requires a dataset_id")
    if not isinstance(release, str) or not release:
        raise ValueError("Hugging Face Dataset binding requires a release")
    delivery = value.get("delivery", "distribution_v1")
    if delivery not in {"distribution_v1", "direct_assets_v1"}:
        raise ValueError("Hugging Face Dataset binding has an invalid delivery")

    if dataset_path is not None:
        dataset = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
        if dataset.get("dataset_id") != dataset_id:
            raise ValueError("Hugging Face binding dataset_id differs from Dataset")
        if dataset.get("release") != release:
            raise ValueError("Hugging Face binding release differs from Dataset")

    return HuggingFaceDatasetBinding(
        repo_id=repo_id,
        revision=revision,
        dataset_id=dataset_id,
        release=release,
        delivery=delivery,
    )
