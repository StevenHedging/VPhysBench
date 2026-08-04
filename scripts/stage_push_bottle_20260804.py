#!/usr/bin/env python3
"""Audit the 2026-08-04 push-bottle videos and multi-sheet XLSX labels."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import xml.etree.ElementTree as ET
from zipfile import ZipFile


VIDEO_ARCHIVE = Path("/root/Steven/推水瓶.zip")
ANNOTATION_ARCHIVE = Path("/root/Steven/推水瓶实验物理标注.zip")
DEFAULT_MEDIA_ROOT = Path(
    "/mnt/nvme1/physics_video_benchmark_ingest/20260804_new_data/"
    "push_bottle/推水瓶"
)
PROMPT = (
    "An upright bottle is pushed near its top, tips over, and falls onto its side."
)
KGF_TO_NEWTON = 9.80665
MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"


def _cell_text(cell: ET.Element, shared: list[str]) -> str | None:
    value_type = cell.attrib.get("t")
    if value_type == "inlineStr":
        return "".join(
            node.text or "" for node in cell.iter(f"{{{MAIN_NS}}}t")
        )
    value = cell.find(f"{{{MAIN_NS}}}v")
    if value is None or value.text is None:
        return None
    if value_type == "s":
        return shared[int(value.text)]
    if value_type == "b":
        return "TRUE" if value.text == "1" else "FALSE"
    return value.text


def read_xlsx(payload: bytes) -> list[dict[str, object]]:
    ns = {"m": MAIN_NS, "r": REL_NS, "pr": PACKAGE_REL_NS}
    with ZipFile(BytesIO(payload)) as workbook:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in workbook.namelist():
            root = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
            for item in root.findall("m:si", ns):
                shared.append(
                    "".join(
                        node.text or ""
                        for node in item.iter(f"{{{MAIN_NS}}}t")
                    )
                )

        book = ET.fromstring(workbook.read("xl/workbook.xml"))
        relationships = ET.fromstring(
            workbook.read("xl/_rels/workbook.xml.rels")
        )
        targets = {
            item.attrib["Id"]: item.attrib["Target"]
            for item in relationships.findall("pr:Relationship", ns)
        }
        result: list[dict[str, object]] = []
        for sheet in book.findall("m:sheets/m:sheet", ns):
            target = targets[sheet.attrib[f"{{{REL_NS}}}id"]]
            sheet_path = (
                target.lstrip("/") if target.startswith("/") else f"xl/{target}"
            )
            root = ET.fromstring(workbook.read(sheet_path))
            rows = []
            for row in root.findall("m:sheetData/m:row", ns):
                values: dict[str, object] = {"_row": int(row.attrib["r"])}
                formulas: dict[str, str] = {}
                for cell in row.findall("m:c", ns):
                    reference = cell.attrib["r"]
                    column_match = re.match(r"[A-Z]+", reference)
                    if column_match is None:
                        raise ValueError(f"invalid XLSX cell reference: {reference}")
                    column = column_match.group(0)
                    text = _cell_text(cell, shared)
                    if text is not None:
                        values[column] = text
                    formula = cell.find(f"{{{MAIN_NS}}}f")
                    if formula is not None and formula.text:
                        formulas[column] = formula.text
                if formulas:
                    values["_formulas"] = formulas
                if len(values) > 1:
                    rows.append(values)
            result.append(
                {
                    "name": sheet.attrib["name"],
                    "state": sheet.attrib.get("state", "visible"),
                    "rows": rows,
                }
            )
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _decoded_zip_name(value: str) -> str:
    try:
        return value.encode("cp437").decode("gbk")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return value


def _workbook_identity(member: str) -> dict[str, object]:
    decoded = _decoded_zip_name(member)
    name = PurePosixPath(decoded).name
    match = re.fullmatch(
        r"质量为([0-9.]+)(g|cm)[,，]?(?:高|高度)为([0-9.]+)cm(?:的)?水瓶\.xlsx",
        name,
    )
    if match is None:
        raise ValueError(f"unrecognized workbook identity: {decoded}")
    mass_value = float(match.group(1))
    mass_unit = match.group(2)
    corrections = []
    if mass_unit == "cm":
        if mass_value != 348.75:
            raise ValueError(f"ambiguous mass unit in workbook: {decoded}")
        corrections.append(
            "filename unit 'cm' in the mass field was corrected to 'g'; "
            "the field label is 质量 and every peer workbook uses grams"
        )
    return {
        "decoded_member": decoded,
        "bottle_mass_kg": round(mass_value / 1000.0, 6),
        "bottle_height_m": round(float(match.group(3)) / 100.0, 6),
        "corrections": corrections,
    }


def _number_with_suffix(value: object, suffix: str) -> float:
    match = re.fullmatch(rf"([+-]?[0-9.]+){re.escape(suffix)}", str(value).strip())
    if match is None:
        raise ValueError(f"unrecognized numeric value {value!r}; expected {suffix}")
    return float(match.group(1))


def _force_with_unit(value: object) -> tuple[float, str]:
    match = re.fullmatch(r"([+-]?[0-9.]+)(Kgf|N)", str(value).strip())
    if match is None:
        raise ValueError(f"unrecognized force value: {value!r}")
    return float(match.group(1)), match.group(2)


def _force_to_newton(value: float, unit: str) -> float:
    if unit == "N":
        return value
    if unit == "Kgf":
        return value * KGF_TO_NEWTON
    raise ValueError(f"unsupported force unit: {unit}")


def _parse_sheet(sheet: dict[str, object]) -> dict[str, object]:
    rows = sheet["rows"]
    if not isinstance(rows, list):
        raise ValueError(f"invalid rows in sheet {sheet['name']}")
    row_by_number = {int(row["_row"]): row for row in rows}
    heading = row_by_number.get(1, {}).get("A")
    if heading != f"{sheet['name']} 测试数据":
        raise ValueError(f"sheet/heading mismatch: {sheet['name']}: {heading}")
    summary = row_by_number.get(3)
    if summary is None or summary.get("A") != "最大值":
        raise ValueError(f"missing force summary in sheet {sheet['name']}")
    maximum_source, maximum_unit = _force_with_unit(summary["B"])
    minimum_source, minimum_unit = _force_with_unit(summary["D"])
    mean_source, mean_unit = _force_with_unit(summary["F"])
    if len({maximum_unit, minimum_unit, mean_unit}) != 1:
        raise ValueError(f"mixed force units in sheet {sheet['name']}")
    source_unit = maximum_unit

    samples: list[dict[str, float | int]] = []
    for row in rows:
        for sequence_col, time_col, force_col in (("A", "B", "C"), ("D", "E", "F")):
            if not all(column in row for column in (sequence_col, time_col, force_col)):
                continue
            if not str(row[sequence_col]).strip().isdigit():
                continue
            time_s = _number_with_suffix(row[time_col], " s")
            force_source = float(str(row[force_col]).strip())
            samples.append(
                {
                    "sequence": int(str(row[sequence_col]).strip()),
                    "time_s": time_s,
                    "force_source_value": force_source,
                    "force_source_unit": source_unit,
                    "force_n": round(
                        _force_to_newton(force_source, source_unit), 6
                    ),
                }
            )
    samples.sort(key=lambda item: int(item["sequence"]))
    if not samples:
        raise ValueError(f"no force samples in sheet {sheet['name']}")
    sequences = [int(item["sequence"]) for item in samples]
    if sequences != list(range(1, len(samples) + 1)):
        raise ValueError(f"non-contiguous force samples in sheet {sheet['name']}")
    measured = [float(item["force_source_value"]) for item in samples]
    return {
        "source_sheet": sheet["name"],
        "source_force_unit": source_unit,
        "peak_force_source_value": maximum_source,
        "minimum_force_source_value": minimum_source,
        "mean_force_source_value": mean_source,
        "peak_force_n": round(_force_to_newton(maximum_source, source_unit), 6),
        "mean_force_n": round(_force_to_newton(mean_source, source_unit), 6),
        "force_samples": samples,
        "force_sample_count": len(samples),
        "summary_cross_check": {
            "source_force_unit": source_unit,
            "series_max_source_value": max(measured),
            "series_min_source_value": min(measured),
            "series_mean_source_value": sum(measured) / len(measured),
            "summary_max_matches_series": abs(maximum_source - max(measured)) < 1e-9,
            "summary_min_matches_series": abs(minimum_source - min(measured)) < 1e-9,
            "summary_mean_minus_series_mean_source_value": mean_source
            - sum(measured) / len(measured),
        },
    }


def _probe(path: Path) -> dict[str, object]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,pix_fmt,r_frame_rate,avg_frame_rate,"
            "nb_frames,duration:stream_tags=rotate:format=duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    stream = payload["streams"][0]
    return {
        "codec": stream.get("codec_name"),
        "pixel_format": stream.get("pix_fmt"),
        "stored_width": int(stream["width"]),
        "stored_height": int(stream["height"]),
        "rotation_deg": int(stream.get("tags", {}).get("rotate", 0)),
        "nominal_frame_rate": stream["r_frame_rate"],
        "average_frame_rate": stream["avg_frame_rate"],
        "frame_count": int(stream["nb_frames"]),
        "duration_s": float(stream.get("duration", payload["format"]["duration"])),
    }


def _source_audit(arguments: tuple[str, str]) -> tuple[str, dict[str, object]]:
    stem, media_root_value = arguments
    path = Path(media_root_value) / f"{stem}.MOV"
    return stem, {
        "source_path": str(path),
        "source_size": path.stat().st_size,
        "source_sha256": sha256(path),
        "source_probe": _probe(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-archive", type=Path, default=VIDEO_ARCHIVE)
    parser.add_argument(
        "--annotation-archive", type=Path, default=ANNOTATION_ARCHIVE
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, default=DEFAULT_MEDIA_ROOT)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    with ZipFile(args.video_archive) as videos_zip:
        video_members = sorted(
            item.filename
            for item in videos_zip.infolist()
            if not item.is_dir()
            and PurePosixPath(item.filename).suffix.lower() == ".mov"
        )
    with ZipFile(args.annotation_archive) as annotations_zip:
        workbook_members = sorted(
            item.filename
            for item in annotations_zip.infolist()
            if not item.is_dir()
            and PurePosixPath(item.filename).suffix.lower() == ".xlsx"
        )
        workbooks = []
        normalized_by_stem: dict[str, dict[str, object]] = {}
        for member in workbook_members:
            identity = _workbook_identity(member)
            sheets = read_xlsx(annotations_zip.read(member))
            workbooks.append(
                {
                    "member": member,
                    "decoded_member": identity["decoded_member"],
                    "identity": identity,
                    "sheets": sheets,
                }
            )
            for sheet in sheets:
                stem = str(sheet["name"]).upper()
                if stem in normalized_by_stem:
                    raise ValueError(f"duplicate sheet/video identity: {stem}")
                force = _parse_sheet(sheet)
                normalized_by_stem[stem] = {
                    "annotation": {
                        "source_workbook_member": identity["decoded_member"],
                        "source_sheet": stem,
                        "unit_corrections": identity["corrections"],
                    },
                    "normalized_physics": {
                        "bottle_mass": {
                            "value": identity["bottle_mass_kg"],
                            "unit": "kg",
                            "annotated": True,
                        },
                        "bottle_height": {
                            "value": identity["bottle_height_m"],
                            "unit": "m",
                            "annotated": True,
                        },
                        "peak_applied_force": {
                            "value": force["peak_force_n"],
                            "unit": "N",
                            "annotated": True,
                        },
                        "mean_applied_force": {
                            "value": force["mean_force_n"],
                            "unit": "N",
                            "annotated": True,
                        },
                    },
                    "force_annotation": force,
                    "prompt": PROMPT,
                }

    video_by_stem = {
        PurePosixPath(member).stem.upper(): member for member in video_members
    }
    missing_annotation = sorted(set(video_by_stem) - set(normalized_by_stem))
    missing_video = sorted(set(normalized_by_stem) - set(video_by_stem))
    matched_stems = sorted(set(video_by_stem) & set(normalized_by_stem))
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        source_audits = dict(
            executor.map(
                _source_audit,
                [(stem, str(args.media_root)) for stem in matched_stems],
            )
        )
    records = []
    for stem in matched_stems:
        record = dict(normalized_by_stem[stem])
        record.update(source_audits[stem])
        record["annotation"]["video_member"] = video_by_stem[stem]
        record["normalized_appearance"] = {
            "bottle_count": 1,
            "camera": "fixed",
            "capture_session": "push_bottle_20260804",
            "interaction": "push_near_top_until_toppling",
        }
        record["alignment"] = {
            "canonical_first_frame_event": "upright bottle before the push develops",
            "source_start_frame": 0,
            "source_end_frame_exclusive": record["source_probe"]["frame_count"],
            "spatial_crop": None,
            "temporal_trim": None,
        }
        record["status"] = "candidate_needs_visual_review"
        records.append(record)

    payload = {
        "schema_version": "1.0",
        "intake_id": "push_bottle_20260804",
        "video_archive": str(args.video_archive),
        "annotation_archive": str(args.annotation_archive),
        "video_count": len(video_members),
        "video_members": video_members,
        "workbook_count": len(workbooks),
        "workbooks": workbooks,
        "matched_record_count": len(records),
        "videos_missing_annotation": missing_annotation,
        "annotations_missing_video": missing_video,
        "field_decisions": {
            "workbook filename mass": "physics.bottle_mass; g converted to kg",
            "workbook filename height": "physics.bottle_height; cm converted to m",
            "sheet 最大值": "physics.peak_applied_force; source N retained or kgf converted to N",
            "sheet 平均值": "physics.mean_applied_force; source N retained or kgf converted to N",
            "sheet force time series": "preserved in provenance in its source unit and normalized N",
        },
        "prompt": PROMPT,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
