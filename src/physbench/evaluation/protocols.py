from __future__ import annotations

from pathlib import Path
from typing import Any

from ..io import canonical_sha256, load_json


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_ROOT = PROJECT_ROOT / "configs" / "evaluation" / "protocols"


def load_evaluation_protocol(
    protocol_id: str,
    *,
    protocol_root: str | Path = PROTOCOL_ROOT,
) -> dict[str, Any]:
    if not protocol_id or Path(protocol_id).name != protocol_id:
        raise ValueError(f"invalid evaluation protocol ID: {protocol_id!r}")
    path = Path(protocol_root) / f"{protocol_id}.json"
    value = load_json(path)
    if value.get("protocol_id") != protocol_id:
        raise ValueError(
            f"evaluation protocol ID mismatch: requested={protocol_id}, "
            f"document={value.get('protocol_id')}"
        )
    if value.get("schema_version") != "1.0":
        raise ValueError("evaluation protocol must use schema_version=1.0")
    if not isinstance(value.get("scenes"), dict):
        raise ValueError("evaluation protocol requires a scenes mapping")
    return {
        **value,
        "path": str(path.resolve()),
        "fingerprint": canonical_sha256(value),
    }
