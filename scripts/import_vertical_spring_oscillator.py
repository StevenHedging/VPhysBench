#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

import cv2
import numpy as np

from physbench.datasets.vertical_spring_import import (
    BallDetection,
    MappedTrial,
    SourceMember,
    TrialAnnotation,
    TurningFrameCandidate,
    VideoAnalysis,
    analyze_video,
    inventory_archive,
    load_trial_annotations,
    map_trials_to_sources,
    materialize_case,
    select_spring_test_ids,
)
from physbench.io import canonical_sha256
from physbench.splitters import build_view_b


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
    members_by_content: dict[tuple[int, int], list[str]] = {}
    for source in sources:
        members_by_content.setdefault((source.size, source.crc32), []).append(
            source.member
        )
    duplicate_source_groups = sorted(
        sorted(members)
        for members in members_by_content.values()
        if len(members) > 1
    )
    duplicate_source_decisions = []
    for (size, _), members in sorted(members_by_content.items()):
        if len(members) < 2:
            continue
        ordered = sorted(
            members,
            key=lambda member: (
                Path(member).stem.endswith(")"),
                member,
            ),
        )
        duplicate_source_decisions.append(
            {
                "canonical_member": ordered[0],
                "excluded_members": ordered[1:],
                "reason": "byte_identical_duplicate_source",
                "size": size,
            }
        )
    payload: dict[str, object] = {
        "archive_name": archive.name,
        "workbook_member": workbook_member.filename,
        "duplicate_source_groups": duplicate_source_groups,
        "duplicate_source_decisions": duplicate_source_decisions,
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
            "duplicate_source_group_count": len(duplicate_source_groups),
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
    workers: int = 1,
) -> dict[str, int]:
    if workers < 1:
        raise ValueError("workers must be positive")
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
    def analyze_item(
        item: MappedTrial,
    ) -> tuple[MappedTrial, VideoAnalysis | Exception]:
        with zipfile.ZipFile(archive) as handle, tempfile.TemporaryDirectory(
            prefix="vphysbench_vertical_spring_"
        ) as directory:
            extracted = Path(directory) / item.source.basename
            with handle.open(item.source.member) as source, extracted.open(
                "wb"
            ) as target:
                shutil.copyfileobj(source, target)
            try:
                result: VideoAnalysis | Exception = analyze_video(
                    extracted,
                    item.trial.direction,
                    analysis_stride=analysis_stride,
                )
            except (OSError, RuntimeError, ValueError) as error:
                result = error
        return item, result

    with ThreadPoolExecutor(max_workers=workers) as executor:
        analyzed = executor.map(analyze_item, intake.accepted)
        for item, result in analyzed:
            if isinstance(result, Exception):
                exclusions.append(
                    {
                        "trial_id": item.trial.trial_id,
                        "video_name": item.trial.video_name,
                        "source_member": item.source.member,
                        "reason": "trajectory_analysis_failed",
                        "details": [str(result)],
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


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def materialize(
    archive: Path,
    analysis_path: Path,
    review_path: Path,
    repo_root: Path,
    *,
    workers: int = 1,
) -> dict[str, int]:
    if workers < 1:
        raise ValueError("workers must be positive")
    candidates = _load_jsonl(analysis_path)
    decisions = {
        str(item["trial_id"]): item for item in _load_jsonl(review_path)
    }
    audits: list[dict[str, object]] = []
    exclusions: list[dict[str, object]] = []
    approved: list[tuple[dict[str, object], dict[str, object]]] = []
    for candidate in candidates:
        trial_id = str(candidate["trial_id"])
        decision = decisions.get(trial_id)
        if decision is None:
            exclusions.append(
                {"trial_id": trial_id, "reason": "missing_review_decision"}
            )
            continue
        if decision.get("status") != "approved":
            exclusions.append(
                {
                    "trial_id": trial_id,
                    "reason": "visual_review_rejected",
                    "details": [str(decision.get("reason", "unspecified"))],
                }
            )
            continue
        approved.append((candidate, decision))

    def materialize_item(
        item: tuple[dict[str, object], dict[str, object]],
    ) -> dict[str, object]:
        candidate, decision = item
        trial_id = str(candidate["trial_id"])
        values = dict(candidate)
        overrides = decision.get("overrides", {})
        if isinstance(overrides, dict):
            values.update(overrides)
        source_member = str(values["source_member"])
        source = SourceMember(
            source_member,
            str(values["video_name"]),
            int(values["source_size"]),
            int(values["source_crc32"]),
        )
        trial = TrialAnnotation(
            trial_id,
            str(values["spring_id"]),
            str(values["video_name"]),
            float(values["signed_displacement_mm"]),
            int(values["workbook_row"]),
        )
        analysis_ball = [float(value) for value in values["analysis_ball_xyr"]]
        full_ball = [
            float(value) for value in values["full_resolution_ball_xyr"]
        ]
        analysis = VideoAnalysis(
            candidate=TurningFrameCandidate(
                int(values["source_start_frame"]),
                float(values["source_start_time_s"]),
                analysis_ball[0],
                analysis_ball[1],
                analysis_ball[2],
                float(values["observed_period_s"]),
                float(values["track_coverage"]),
            ),
            full_resolution_ball=BallDetection(
                full_ball[0], full_ball[1], full_ball[2], 1.0
            ),
            frame_count=int(values["source_frame_count"]),
            displayed_width=int(values["displayed_width"]),
            displayed_height=int(values["displayed_height"]),
            analysis_stride=int(values["analysis_stride"]),
        )
        with zipfile.ZipFile(archive) as handle, tempfile.TemporaryDirectory(
            prefix="vphysbench_vertical_spring_materialize_"
        ) as directory:
            extracted = Path(directory) / source.basename
            with handle.open(source.member) as source_handle, extracted.open(
                "wb"
            ) as target:
                shutil.copyfileobj(source_handle, target)
            draft = materialize_case(
                extracted,
                trial,
                source,
                analysis,
                repo_root,
            )
        audit = dict(draft.audit)
        audit["review"] = decision
        return audit

    with ThreadPoolExecutor(max_workers=workers) as executor:
        audits.extend(executor.map(materialize_item, approved))
    import_root = repo_root / "datasets" / "provenance" / "imports"
    _write_jsonl(
        import_root
        / "vertical_spring_oscillator_20260806_import_audit.jsonl",
        audits,
    )
    _write_json(
        import_root / "vertical_spring_oscillator_20260806_exclusions.json",
        {"excluded": exclusions, "excluded_count": len(exclusions)},
    )
    summary = {
        "accepted_count": len(audits),
        "excluded_count": len(exclusions),
    }
    _write_json(
        import_root / "vertical_spring_oscillator_20260806_summary.json",
        summary,
    )
    return summary


def review_sheets(
    archive: Path,
    analysis_path: Path,
    output_dir: Path,
) -> int:
    candidates = _load_jsonl(analysis_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    cards: list[np.ndarray] = []
    with zipfile.ZipFile(archive) as handle:
        for candidate in candidates:
            member = str(candidate["source_member"])
            with tempfile.TemporaryDirectory(
                prefix="vphysbench_vertical_spring_review_"
            ) as directory:
                extracted = Path(directory) / Path(member).name
                with handle.open(member) as source, extracted.open("wb") as target:
                    shutil.copyfileobj(source, target)
                capture = cv2.VideoCapture(str(extracted))
                start = int(candidate["source_start_frame"])
                frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                indices = [max(0, start - 8), start, min(frame_count - 1, start + 8)]
                panels: list[np.ndarray] = []
                selected_original_shape: tuple[int, int] | None = None
                for frame_index in indices:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                    ok, frame = capture.read()
                    if not ok:
                        raise ValueError(
                            f"could not decode review frame {frame_index} for {member}"
                        )
                    if frame_index == start:
                        selected_original_shape = frame.shape[:2]
                    panels.append(
                        cv2.resize(frame, (270, 480), interpolation=cv2.INTER_AREA)
                    )
                capture.release()
            overlay = panels[1].copy()
            if selected_original_shape is None:
                raise ValueError("selected review frame was not decoded")
            height, width = selected_original_shape
            center_x, center_y, radius = [
                float(value) for value in candidate["full_resolution_ball_xyr"]
            ]
            cv2.circle(
                overlay,
                (round(center_x * 270 / width), round(center_y * 480 / height)),
                round(radius * min(270 / width, 480 / height)),
                (0, 0, 255),
                3,
            )
            body = np.hstack([*panels, overlay])
            label = np.full((44, body.shape[1], 3), 255, dtype=np.uint8)
            cv2.putText(
                label,
                (
                    f"{candidate['trial_id']} {candidate['direction']} "
                    f"x={candidate['signed_displacement_mm']}mm frame={start}"
                ),
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )
            cards.append(np.vstack([label, body]))
    cards_per_page = 8
    columns = 2
    for page_index, offset in enumerate(range(0, len(cards), cards_per_page), start=1):
        page_cards = cards[offset : offset + cards_per_page]
        cell_height, cell_width = page_cards[0].shape[:2]
        rows = (len(page_cards) + columns - 1) // columns
        page = np.full(
            (rows * cell_height, columns * cell_width, 3),
            245,
            dtype=np.uint8,
        )
        for cell_index, card in enumerate(page_cards):
            row, column = divmod(cell_index, columns)
            page[
                row * cell_height : (row + 1) * cell_height,
                column * cell_width : (column + 1) * cell_width,
            ] = card
        cv2.imwrite(str(output_dir / f"review_page_{page_index:03d}.png"), page)
    return len(cards)


def build_release_v13(
    repo_root: Path,
    audits: list[dict[str, object]],
) -> dict[str, int]:
    datasets_root = repo_root / "datasets"
    base_release = datasets_root / "releases" / "12.0.0"
    release = datasets_root / "releases" / "13.0.0"
    base_provenance = datasets_root / "provenance" / "releases" / "12.0.0"
    provenance = datasets_root / "provenance" / "releases" / "13.0.0"
    if release.exists() or provenance.exists():
        raise ValueError("Dataset 13.0.0 already exists")
    if not audits or any(
        item.get("scene_id") != "vertical_spring_oscillator"
        for item in audits
    ):
        raise ValueError("release build requires vertical spring import audits")

    shutil.copytree(base_release, release)
    provenance.mkdir(parents=True)
    base_cases = _load_jsonl(base_release / "cases.jsonl")
    base_provenance_cases = _load_jsonl(base_provenance / "cases.jsonl")
    spring_cases: list[dict[str, object]] = []
    spring_provenance: list[dict[str, object]] = []
    for audit in sorted(audits, key=lambda item: str(item["case_id"])):
        case_id = str(audit["case_id"])
        relative_case = str(audit["asset_directory"])
        if not relative_case.startswith("assets/vertical_spring_oscillator/"):
            raise ValueError(f"invalid spring asset directory: {relative_case}")
        case_root = datasets_root / relative_case
        for member in ("caption.json", "physics.json"):
            if not (case_root / member).is_file():
                raise ValueError(f"missing spring Case member: {case_id}/{member}")
        spring_cases.append(
            {
                "appearance": {
                    "background": "laboratory_rig",
                    "camera": "fixed_portrait",
                    "capture_session": "20260806_vertical_spring_oscillator",
                    "moving_objects": ["steel_ball"],
                    "object_count": 1,
                    "release_side": str(audit["direction"]),
                    "spring_id": "S01",
                },
                "assets": {
                    "caption": f"{relative_case}/caption.json",
                    "first_frame": f"{relative_case}/canonical/first_frame.png",
                    "first_frame_mask_manifest": (
                        f"{relative_case}/canonical/masks/manifest.json"
                    ),
                    "physics_annotation": f"{relative_case}/physics.json",
                    "reference_video": f"{relative_case}/canonical/reference.mp4",
                },
                "case_id": case_id,
                "scene_id": "vertical_spring_oscillator",
                "temporal": {"encoded_to_physical_speed": 1.0},
            }
        )
        spring_provenance.append(
            {
                "alignment": audit.get("alignment", {}),
                "case_id": case_id,
                "detector": {
                    "source_start_frame": audit["source_start_frame"],
                    "source_start_time_s": audit["source_start_time_s"],
                },
                "review": audit.get("review", {}),
                "scene_id": "vertical_spring_oscillator",
                "source_locator": {
                    "annotation_workbook": (
                        "provenance/source_docs/"
                        "20260806_vertical_spring_oscillator/source_workbook.xlsx"
                    ),
                    "archive": (
                        "provenance/source_archives/"
                        "20260806_vertical_spring_oscillator/"
                        "vertical_spring_oscillator.zip"
                    ),
                    "member": audit["source_member"],
                    "trial_id": audit["trial_id"],
                    "workbook_row": audit.get("workbook_row"),
                },
                "temporal_metadata": {
                    "canonical_frame_count": audit["canonical_frame_count"],
                    "source_frame_count": audit["source_frame_count"],
                    "time_scale": "real_time",
                },
            }
        )

    all_cases = [*base_cases, *spring_cases]
    _write_jsonl(release / "cases.jsonl", all_cases)
    descriptor = json.loads((release / "dataset.json").read_text(encoding="utf-8"))
    descriptor.update(
        {
            "dataset_id": "physics_video_seven_scene_v13",
            "release": "13.0.0",
        }
    )
    _write_json(release / "dataset.json", descriptor)
    _write_json(
        release / "scenes" / "vertical_spring_oscillator.json",
        {
            "constraints": [
                (
                    "Canonical frame zero is the first post-cycle return to "
                    "the release-side turning point"
                ),
                "The source tail is retained without spatial or temporal cropping",
            ],
            "display_name": "竖直弹簧振子",
            "generalization_factors": [
                {
                    "category": "physical_parameter",
                    "name": "initial_displacement",
                    "source": "physics.objects.object_1.initial_displacement",
                },
                {
                    "category": "interaction_structure",
                    "name": "release_side",
                    "source": "appearance.release_side",
                },
                {
                    "category": "acquisition",
                    "name": "capture_session",
                    "source": "appearance.capture_session",
                },
            ],
            "metric_spec": {
                "common_sense_checks": [
                    "continuous vertical oscillation",
                    "fixed upper support",
                    "periodic return through the equilibrium region",
                    "no spontaneous topology change",
                ],
                "prediction_quantities": [
                    "vertical trajectory",
                    "oscillation period",
                    "amplitude decay",
                ],
                "prediction_subjects": ["steel_ball", "spring"],
                "visual_attributes": [
                    "ball integrity",
                    "spring attachment",
                    "support stability",
                ],
            },
            "scene_id": "vertical_spring_oscillator",
            "schema_version": "2.0",
            "structured_physics_parameters": [
                "environment.gravity_acceleration",
                "environment.natural_spring_length",
                "environment.spring_stiffness",
                "objects.object_1.initial_displacement",
                "objects.object_1.mass",
                "objects.object_1.radius",
            ],
        },
    )

    all_case_ids = [str(item["case_id"]) for item in all_cases]
    case_digest = canonical_sha256(sorted(all_case_ids))
    view_a_path = release / "views" / "view_a.json"
    view_a = json.loads(view_a_path.read_text(encoding="utf-8"))
    audit_by_case = {str(item["case_id"]): item for item in audits}
    test_ids = set(
        select_spring_test_ids(
            [
                {
                    "case_id": case_id,
                    "direction": audit["direction"],
                    "signed_displacement_mm": audit["signed_displacement_mm"],
                    "source_group": audit["source_group"],
                }
                for case_id, audit in sorted(audit_by_case.items())
            ],
            limit=20,
        )
    )
    spring_ids = sorted(str(item["case_id"]) for item in spring_cases)
    view_a["scenes"]["vertical_spring_oscillator"] = {
        "test": sorted(test_ids),
        "train": [case_id for case_id in spring_ids if case_id not in test_ids],
    }
    rationale = (
        "Independent held-out trial whose release side and displacement value "
        "are represented in View A training."
    )
    for case_id in sorted(test_ids):
        view_a["test_annotations"][case_id] = {
            "co_varying_factors": [],
            "generalization_regime": "id",
            "ood_factors": [],
            "rationale": rationale,
        }
    view_a["case_set_sha256"] = case_digest
    _write_json(view_a_path, view_a)

    view_b_path = release / "views" / "view_b.json"
    view_b = json.loads(view_b_path.read_text(encoding="utf-8"))
    generated = build_view_b(spring_cases, groups=5, seed=int(view_b["seed"]))
    view_b["scenes"]["vertical_spring_oscillator"] = generated["scenes"][
        "vertical_spring_oscillator"
    ]
    view_b["case_set_sha256"] = case_digest
    _write_json(view_b_path, view_b)

    _write_jsonl(
        provenance / "cases.jsonl",
        [*base_provenance_cases, *spring_provenance],
    )
    report = {
        "base_case_count": len(base_cases),
        "case_count": len(all_cases),
        "spring_case_count": len(spring_cases),
        "spring_test_count": len(test_ids),
        "spring_train_count": len(spring_cases) - len(test_ids),
    }
    _write_json(provenance / "build.json", report)
    return report


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
    analyze_parser.add_argument("--workers", type=int, default=1)
    materialize_parser = subparsers.add_parser("materialize")
    materialize_parser.add_argument("--archive", type=Path, required=True)
    materialize_parser.add_argument("--analysis", type=Path, required=True)
    materialize_parser.add_argument("--review", type=Path, required=True)
    materialize_parser.add_argument("--repo-root", type=Path, required=True)
    materialize_parser.add_argument("--workers", type=int, default=1)
    review_parser = subparsers.add_parser("review-sheets")
    review_parser.add_argument("--archive", type=Path, required=True)
    review_parser.add_argument("--analysis", type=Path, required=True)
    review_parser.add_argument("--output-dir", type=Path, required=True)
    release_parser = subparsers.add_parser("build-release")
    release_parser.add_argument("--repo-root", type=Path, required=True)
    release_parser.add_argument("--audit", type=Path, required=True)
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
            workers=args.workers,
        )
        print(json.dumps(summary, sort_keys=True))
        return 0
    if args.command == "materialize":
        summary = materialize(
            args.archive,
            args.analysis,
            args.review,
            args.repo_root,
            workers=args.workers,
        )
        print(json.dumps(summary, sort_keys=True))
        return 0
    if args.command == "review-sheets":
        count = review_sheets(args.archive, args.analysis, args.output_dir)
        print(json.dumps({"review_card_count": count}, sort_keys=True))
        return 0
    if args.command == "build-release":
        report = build_release_v13(args.repo_root, _load_jsonl(args.audit))
        print(json.dumps(report, sort_keys=True))
        return 0
    raise ValueError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
