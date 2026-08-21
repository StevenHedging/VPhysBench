#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from physbench.evaluation.preflight import run_reference_preflight


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Certify that every official Task Case returns finite evaluated "
            "scores for a known-legal prediction."
        )
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = run_reference_preflight(
        dataset_path=args.dataset,
        task_path=args.task,
        output_dir=args.output,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
