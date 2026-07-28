from __future__ import annotations

from collections.abc import Mapping
from typing import Any


PHYSICS_USAGES = frozenset({"ignored", "optional", "required"})


def validate_input_policy(value: Any) -> dict[str, Any]:
    """Validate the normalized, Baseline-owned Case input policy."""

    if not isinstance(value, Mapping):
        raise ValueError("baseline input_policy must be an object")
    expected = {"schema_version", "case_view", "text", "physics"}
    if set(value) != expected:
        raise ValueError(
            "baseline input_policy fields mismatch: "
            f"missing={sorted(expected - set(value))}, "
            f"unknown={sorted(set(value) - expected)}"
        )
    if value.get("schema_version") != "1.0":
        raise ValueError("baseline input_policy.schema_version must be 1.0")
    if value.get("case_view") != "conditionable_case_v1":
        raise ValueError(
            "baseline input_policy.case_view must be conditionable_case_v1"
        )

    text = value.get("text")
    if not isinstance(text, Mapping) or set(text) != {"source", "usage"}:
        raise ValueError(
            "baseline input_policy.text must contain exactly source and usage"
        )
    if text.get("source") != "case.text.prompt":
        raise ValueError(
            "baseline input_policy.text.source must be case.text.prompt"
        )
    if text.get("usage") != "required":
        raise ValueError(
            "baseline input_policy.text.usage must be required"
        )

    physics = value.get("physics")
    physics_fields = {"source", "usage", "representations"}
    if not isinstance(physics, Mapping) or set(physics) != physics_fields:
        raise ValueError(
            "baseline input_policy.physics fields mismatch"
        )
    if physics.get("source") != "case.physics[annotated=true]":
        raise ValueError(
            "baseline input_policy.physics.source must be "
            "case.physics[annotated=true]"
        )
    usage = physics.get("usage")
    if usage not in PHYSICS_USAGES:
        raise ValueError(
            "baseline input_policy.physics.usage must be one of "
            f"{sorted(PHYSICS_USAGES)}"
        )
    representations = physics.get("representations")
    if (
        not isinstance(representations, list)
        or any(
            not isinstance(item, str) or not item
            for item in representations
        )
        or len(representations) != len(set(representations))
    ):
        raise ValueError(
            "baseline input_policy.physics.representations must be a "
            "unique string list"
        )
    if usage == "ignored" and representations:
        raise ValueError(
            "physics usage ignored requires representations=[]"
        )
    if usage != "ignored" and not representations:
        raise ValueError(
            f"physics usage {usage} requires at least one representation"
        )
    return {
        "schema_version": "1.0",
        "case_view": "conditionable_case_v1",
        "text": dict(text),
        "physics": {
            "source": physics["source"],
            "usage": usage,
            "representations": list(representations),
        },
    }


__all__ = ["PHYSICS_USAGES", "validate_input_policy"]
