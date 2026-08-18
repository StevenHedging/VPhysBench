"""Durable, complete per-Case visual review records."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


_DECISIONS = frozenset({"pass", "repair", "needs_dense_review"})
_FIELDS = {
    "schema_version",
    "case_id",
    "scene_id",
    "anchor_decision",
    "tube_decision",
    "evidence",
    "reviewer",
    "notes",
    "issues",
    "repairs",
    "final_asset_digests",
}


def _strings(values: Iterable[str], *, label: str) -> tuple[str, ...]:
    output = tuple(values)
    if any(not isinstance(value, str) or not value.strip() for value in output):
        raise ValueError(f"{label} must contain non-empty strings")
    if len(output) != len(set(output)):
        raise ValueError(f"{label} must not contain duplicates")
    return output


@dataclass(frozen=True)
class ReviewDecision:
    case_id: str
    scene_id: str
    anchor_decision: str
    tube_decision: str
    evidence: tuple[str, ...]
    reviewer: str
    notes: str
    issues: tuple[str, ...]
    repairs: tuple[str, ...]
    final_asset_digests: Mapping[str, str]

    def __post_init__(self) -> None:
        for value, label in (
            (self.case_id, "case_id"),
            (self.scene_id, "scene_id"),
            (self.reviewer, "reviewer"),
            (self.notes, "notes"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"review {label} must be a non-empty string")
        for value, label in (
            (self.anchor_decision, "anchor_decision"),
            (self.tube_decision, "tube_decision"),
        ):
            if value not in _DECISIONS:
                raise ValueError(
                    f"review {label} must be one of {sorted(_DECISIONS)}"
                )
        evidence = _strings(self.evidence, label="review evidence")
        if len(evidence) < 2:
            raise ValueError(
                "review requires at least two evidence paths for anchor and tube"
            )
        _strings(self.issues, label="review issues")
        _strings(self.repairs, label="review repairs")
        if not isinstance(self.final_asset_digests, Mapping):
            raise ValueError("review final_asset_digests must be an object")
        for path, digest in self.final_asset_digests.items():
            if not isinstance(path, str) or not path:
                raise ValueError("review final asset paths must be non-empty strings")
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(
                    f"review final asset digest for {path} must be lowercase SHA-256"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "case_id": self.case_id,
            "scene_id": self.scene_id,
            "anchor_decision": self.anchor_decision,
            "tube_decision": self.tube_decision,
            "evidence": list(self.evidence),
            "reviewer": self.reviewer,
            "notes": self.notes,
            "issues": list(self.issues),
            "repairs": list(self.repairs),
            "final_asset_digests": dict(sorted(self.final_asset_digests.items())),
        }

    @classmethod
    def from_dict(cls, value: Any) -> "ReviewDecision":
        if not isinstance(value, dict) or set(value) != _FIELDS:
            raise ValueError("review ledger row fields are invalid")
        if value.get("schema_version") != "1.0":
            raise ValueError("review ledger row schema_version must be 1.0")
        for field in ("evidence", "issues", "repairs"):
            if not isinstance(value[field], list):
                raise ValueError(f"review ledger {field} must be an array")
        return cls(
            case_id=value["case_id"],
            scene_id=value["scene_id"],
            anchor_decision=value["anchor_decision"],
            tube_decision=value["tube_decision"],
            evidence=tuple(value["evidence"]),
            reviewer=value["reviewer"],
            notes=value["notes"],
            issues=tuple(value["issues"]),
            repairs=tuple(value["repairs"]),
            final_asset_digests=value["final_asset_digests"],
        )


def _validate_complete_set(
    decisions: Iterable[ReviewDecision],
    *,
    expected_case_ids: Iterable[str],
) -> tuple[ReviewDecision, ...]:
    values = tuple(decisions)
    seen: set[str] = set()
    duplicates: list[str] = []
    for decision in values:
        if decision.case_id in seen:
            duplicates.append(decision.case_id)
        seen.add(decision.case_id)
    if duplicates:
        raise ValueError(f"review ledger has duplicate Case rows: {sorted(set(duplicates))}")
    expected = set(expected_case_ids)
    missing = sorted(expected - seen)
    unknown = sorted(seen - expected)
    if missing:
        raise ValueError(f"review ledger is missing Case rows: {missing}")
    if unknown:
        raise ValueError(f"review ledger has unknown Case rows: {unknown}")
    return values


def read_review_ledger(
    path: str | Path,
    *,
    expected_case_ids: Iterable[str],
) -> tuple[ReviewDecision, ...]:
    values: list[ReviewDecision] = []
    for line_number, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            document = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"review ledger line {line_number} is not valid JSON"
            ) from exc
        values.append(ReviewDecision.from_dict(document))
    return _validate_complete_set(values, expected_case_ids=expected_case_ids)


def write_review_ledger(
    path: str | Path,
    decisions: Iterable[ReviewDecision],
    *,
    expected_case_ids: Iterable[str],
) -> None:
    order = tuple(expected_case_ids)
    values = _validate_complete_set(decisions, expected_case_ids=order)
    by_id = {value.case_id: value for value in values}
    payload = "".join(
        json.dumps(
            by_id[case_id].to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        for case_id in order
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_value = tempfile.mkstemp(
        prefix=f".{output.name}.",
        suffix=".tmp",
        dir=output.parent,
    )
    temporary = Path(temporary_value)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


__all__ = ["ReviewDecision", "read_review_ledger", "write_review_ledger"]
