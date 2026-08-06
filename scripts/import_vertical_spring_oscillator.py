#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

from physbench.datasets.vertical_spring_import import (
    analyze_video,
    inventory_archive,
    load_trial_annotations,
    map_trials_to_sources,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def inventory(archive: Path, output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as handle:
        workbook_members = [
            item
            for item in handle.infolist()
            if not item.is_dir() and item.filename.lower().endswith(".xlsx")
        ]
        if len(workbook_members) != 1:
            raise ValueError(
                f"expected exactly one XLSX member, found {len(workbook_members)}"
            )
        workbook_member = workbook_members[0]
        workbook_path = output_dir / "source_workbook.xlsx"
        with handle.open(workbook_member) as source, workbook_path.open("wb") as target:
            shutil.copyfileobj(source, target)

    trials = load_trial_annotations(workbook_path)
    sources = inventory_archive(archive)
    result = map_trials_to_sources(trials, sources)
    payload: dict[str, object] = {
        "archive": str(archive),
        "workbook_member": workbook_member.filename,
        "accepted": [
            {
                "trial_id": item.trial.trial_id,
                "spring_id": item.trial.spring_id,
                "video_name": item.trial.video_name,
                "signed_displacement_mm": item.trial.displacement_mm,
                "displacement_m": item.trial.displacement_m,
                "direction": item.trial.direction,
                "workbook_row": item.trial.workbook_row,
                "source_member": item.source.member,
                "source_size": item.source.size,
                "source_crc32": item.source.crc32,
                "duplicate_members": list(item.duplicate_members),
            }
            for item in result.accepted
        ],
        "exclusions": [
            {
                "trial_id": item.trial_id,
                "video_name": item.video_name,
                "reason": item.reason,
                "details": list(item.details),
            }
            for item in result.exclusions
        ],
        "source_members": [
            {
                "member": item.member,
                "basename": item.basename,
                "size": item.size,
                "crc32": item.crc32,
            }
            for item in sources
        ],
        "summary": {
            "trial_count": len(trials),
            "video_member_count": len(sources),
            "accepted_count": len(result.accepted),
            "excluded_count": len(result.exclusions),
        },
    }
    _write_json(output_dir / "normalized_annotations.json", payload)
    return payload


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def analyze(
    archive: Path,
    workbook: Path,
    output_dir: Path,
    *,
    analysis_stride: int,
) -> dict[str, int]:
    trials = load_trial_annotations(workbook)
    sources = inventory_archive(archive)
    intake = map_trials_to_sources(trials, sources)
    candidates: list[dict[str, object]] = []
    exclusions: list[dict[str, object]] = [
        {
            "trial_id": item.trial_id,
            "video_name": item.video_name,
            "reason": item.reason,
            "details": list(item.details),
        }
        for item in intake.exclusions
    ]
    with zipfile.ZipFile(archive) as handle:
        for item in intake.accepted:
            with tempfile.TemporaryDirectory(
                prefix="vphysbench_vertical_spring_"
            ) as directory:
                extracted = Path(directory) / item.source.basename
                with handle.open(item.source.member) as source, extracted.open(
                    "wb"
                ) as target:
                    shutil.copyfileobj(source, target)
                try:
                    result = analyze_video(
                        extracted,
                        item.trial.direction,
                        analysis_stride=analysis_stride,
                    )
                except (OSError, RuntimeError, ValueError) as error:
                    exclusions.append(
                        {
                            "trial_id": item.trial.trial_id,
                            "video_name": item.trial.video_name,
                            "source_member": item.source.member,
                            "reason": "trajectory_analysis_failed",
                            "details": [str(error)],
                        }
                    )
                    continue
            candidates.append(
                {
                    "status": "pending",
                    "trial_id": item.trial.trial_id,
                    "spring_id": item.trial.spring_id,
                    "workbook_row": item.trial.workbook_row,
                    "video_name": item.trial.video_name,
                    "source_member": item.source.member,
                    "source_size": item.source.size,
                    "source_crc32": item.source.crc32,
                    "signed_displacement_mm": item.trial.displacement_mm,
                    "direction": item.trial.direction,
                    "source_start_frame": result.candidate.frame_index,
                    "source_start_time_s": result.candidate.time_s,
                    "observed_period_s": result.candidate.observed_period_s,
                    "track_coverage": result.candidate.track_coverage,
                    "analysis_ball_xyr": [
                        result.candidate.center_x,
                        result.candidate.center_y,
                        result.candidate.radius,
                    ],
                    "full_resolution_ball_xyr": [
                        result.full_resolution_ball.center_x,
                        result.full_resolution_ball.center_y,
                        result.full_resolution_ball.radius,
                    ],
                    "source_frame_count": result.frame_count,
                    "displayed_width": result.displayed_width,
                    "displayed_height": result.displayed_height,
                    "analysis_stride": result.analysis_stride,
                }
            )
    _write_jsonl(output_dir / "review_candidates.jsonl", candidates)
    _write_json(
        output_dir / "analysis_exclusions.json",
        {"excluded": exclusions, "excluded_count": len(exclusions)},
    )
    summary = {
        "candidate_count": len(candidates),
        "excluded_count": len(exclusions),
    }
    _write_json(output_dir / "analysis_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Import the vertical spring oscillator Dataset Scene."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    inventory_parser = subparsers.add_parser("inventory")
    inventory_parser.add_argument("--archive", type=Path, required=True)
    inventory_parser.add_argument("--output-dir", type=Path, required=True)
    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--archive", type=Path, required=True)
    analyze_parser.add_argument("--workbook", type=Path, required=True)
    analyze_parser.add_argument("--output-dir", type=Path, required=True)
    analyze_parser.add_argument("--analysis-stride", type=int, default=8)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "inventory":
        payload = inventory(args.archive, args.output_dir)
        print(json.dumps(payload["summary"], sort_keys=True))
        return 0
    if args.command == "analyze":
        summary = analyze(
            args.archive,
            args.workbook,
            args.output_dir,
            analysis_stride=args.analysis_stride,
        )
        print(json.dumps(summary, sort_keys=True))
        return 0
    raise ValueError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
