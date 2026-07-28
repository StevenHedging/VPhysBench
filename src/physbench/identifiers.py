from __future__ import annotations

import re
from typing import Any


SAFE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def require_safe_id(value: Any, *, label: str) -> str:
    """Validate an identifier before it can become part of a file path."""

    if (
        not isinstance(value, str)
        or SAFE_ID_PATTERN.fullmatch(value) is None
    ):
        raise ValueError(
            f"{label} must start with an alphanumeric character and contain "
            "only letters, digits, '.', '_' or '-'"
        )
    return value


__all__ = ["SAFE_ID_PATTERN", "require_safe_id"]
