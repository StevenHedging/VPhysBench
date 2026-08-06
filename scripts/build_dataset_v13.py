#!/usr/bin/env python3
"""Build the single active Dataset 13 release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from physbench.datasets.vertical_spring_import import select_spring_test_ids


__all__ = ["select_spring_test_ids"]


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Build Dataset 13.0.0")
    parser.add_argument("--repo-root", type=Path, default=root)
    parser.add_argument(
        "--audit",
        type=Path,
        default=(
            root
            / "datasets/provenance/imports/"
            "vertical_spring_oscillator_20260806_import_audit.jsonl"
        ),
    )
    args = parser.parse_args(argv)
    from import_vertical_spring_oscillator import build_release_v13

    audits = [
        json.loads(line)
        for line in args.audit.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    report = build_release_v13(args.repo_root, audits)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
