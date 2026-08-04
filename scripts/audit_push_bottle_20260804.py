#!/usr/bin/env python3
"""Create a read-only intake audit for the 2026-08-04 push-bottle archive."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from zipfile import ZipFile


ARCHIVE = Path("/root/Steven/推水瓶.zip")
ANNOTATION_SUFFIXES = {".xlsx", ".xls", ".csv", ".json", ".jsonl"}
VIDEO_SUFFIXES = {".mov", ".mp4", ".mkv", ".avi"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with ZipFile(args.archive) as source:
        files = [item for item in source.infolist() if not item.is_dir()]
    unsafe = []
    for item in files:
        path = PurePosixPath(item.filename)
        if path.is_absolute() or ".." in path.parts:
            unsafe.append(item.filename)
    videos = sorted(
        item.filename
        for item in files
        if PurePosixPath(item.filename).suffix.lower() in VIDEO_SUFFIXES
    )
    annotations = sorted(
        item.filename
        for item in files
        if PurePosixPath(item.filename).suffix.lower() in ANNOTATION_SUFFIXES
    )
    payload = {
        "schema_version": "1.0",
        "intake_id": "push_bottle_20260804",
        "source_archive": str(args.archive),
        "source_archive_size": args.archive.stat().st_size,
        "source_archive_sha256": sha256(args.archive),
        "archive_file_count": len(files),
        "video_count": len(videos),
        "video_members": videos,
        "annotation_member_count": len(annotations),
        "annotation_members": annotations,
        "unsafe_members": unsafe,
        "status": (
            "ready_for_annotation_mapping"
            if annotations and not unsafe
            else "blocked_missing_annotation_table"
            if not annotations and not unsafe
            else "blocked_unsafe_archive_paths"
        ),
        "decision": (
            "No Dataset Case or scene was created because the archive contains "
            "no table that can establish a video-to-physics-annotation mapping."
            if not annotations
            else None
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
