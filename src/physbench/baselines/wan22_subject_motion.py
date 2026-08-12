from __future__ import annotations

import json
import math
import os
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .wan22_st_tube_iou import Wan22STTubeIoULoraAdapter
from ..identifiers import require_safe_id
from ..io import load_json, sha256_file, write_json


class Wan22SubjectMotionAdapter(Wan22STTubeIoULoraAdapter):
    """Warm-start WAN training with validated reusable subject tubes."""

    @staticmethod
    def _paired_head_path(lora_checkpoint: Path) -> Path:
        return lora_checkpoint.with_name(
            f"{lora_checkpoint.stem}.st-head.safetensors"
        )

    def _resolve_warm_start_checkpoint(self) -> Path:
        configured = self.config.get("initial_lora_checkpoint")
        if not isinstance(configured, str) or not configured.strip():
            raise ValueError("subject-motion training requires a warm-start LoRA")
        checkpoint = Path(configured)
        if not checkpoint.is_absolute():
            checkpoint = Path(self.runtime["project_root"]) / checkpoint
        checkpoint = checkpoint.resolve()
        if checkpoint.name.endswith(".st-head.safetensors"):
            raise ValueError("warm-start checkpoint must be the LoRA, not the ST-head")
        if checkpoint.suffix != ".safetensors":
            raise ValueError("warm-start LoRA must use a .safetensors checkpoint")
        if self.execute and not checkpoint.is_file():
            raise FileNotFoundError(f"warm-start LoRA checkpoint is missing: {checkpoint}")
        head = self._paired_head_path(checkpoint)
        if self.execute and not head.is_file():
            raise FileNotFoundError(
                f"warm-start paired ST-head checkpoint is missing: {head}"
            )
        return checkpoint

    def _subject_tube_cache_dir(self) -> Path | None:
        configured = self.runtime.get("subject_tube_cache_dir")
        if configured is None:
            return None
        if not isinstance(configured, str) or not configured.strip():
            raise ValueError("runtime.subject_tube_cache_dir must be a path")
        path = Path(configured)
        if not path.is_absolute():
            path = Path(self.runtime["project_root"]) / path
        return path.resolve()

    def _validate_cached_tube(
        self,
        case_id: str,
        tube_path: Path,
        audit_path: Path,
    ) -> dict[str, Any]:
        require_safe_id(case_id, label="subject tube cache case_id")
        if not tube_path.is_file() or not audit_path.is_file():
            raise FileNotFoundError(
                f"subject tube cache pair is incomplete for {case_id}"
            )
        audit = load_json(audit_path)
        if audit.get("case_id") != case_id:
            raise ValueError(f"cached tube audit case_id mismatch for {case_id}")
        recorded_hash = audit.get("tube_sha256")
        actual_hash = sha256_file(tube_path)
        if recorded_hash != actual_hash:
            raise ValueError(
                f"cached tube hash mismatch for {case_id}: "
                f"recorded={recorded_hash}, actual={actual_hash}"
            )
        try:
            with np.load(tube_path, allow_pickle=False) as payload:
                required = {"masks", "layout", "case_id"}
                if not required.issubset(payload.files):
                    raise ValueError(
                        f"cached tube is missing arrays for {case_id}: "
                        f"{sorted(required - set(payload.files))}"
                    )
                masks = np.asarray(payload["masks"])
                layout = np.asarray(payload["layout"])
                stored_case = np.asarray(payload["case_id"])
        except (OSError, ValueError) as exc:
            raise ValueError(f"cached tube cannot be decoded for {case_id}") from exc
        if masks.ndim != 3 or masks.dtype != np.uint8:
            raise ValueError(f"cached tube must be uint8 THW for {case_id}")
        if set(np.unique(masks).tolist()) - {0, 1}:
            raise ValueError(f"cached tube must be binary for {case_id}")
        if not bool(masks.any()):
            raise ValueError(f"cached tube is empty for {case_id}")
        if layout.ndim != 0 or str(layout.item()) != "THW":
            raise ValueError(f"cached tube layout must be THW for {case_id}")
        if stored_case.ndim != 0 or str(stored_case.item()) != case_id:
            raise ValueError(f"cached tube payload case_id mismatch for {case_id}")
        if audit.get("shape_thw") != list(masks.shape):
            raise ValueError(f"cached tube shape audit mismatch for {case_id}")
        if audit.get("dtype") != "uint8":
            raise ValueError(f"cached tube dtype audit must be uint8 for {case_id}")
        if audit.get("values") != sorted(np.unique(masks).tolist()):
            raise ValueError(f"cached tube value audit mismatch for {case_id}")
        source = audit.get("source_fingerprint")
        if not isinstance(source, dict) or source.get("case_id") != case_id:
            raise ValueError(
                f"cached tube source_fingerprint case_id mismatch for {case_id}"
            )
        video_hash = source.get("normalized_video_sha256")
        if (
            not isinstance(video_hash, str)
            or re.fullmatch(r"[0-9a-f]{64}", video_hash) is None
        ):
            raise ValueError(
                f"cached tube source video hash is invalid for {case_id}"
            )
        return audit

    @staticmethod
    def _link_or_copy(source: Path, target: Path) -> str:
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)
            return "copy2"
        return "hardlink"

    def _seed_subject_tube_cache(
        self,
        case_ids: list[str],
        target_dir: Path,
    ) -> dict[str, Any]:
        cache_dir = self._subject_tube_cache_dir()
        result: dict[str, Any] = {
            "schema_version": "1.0",
            "source_cache_dir": str(cache_dir) if cache_dir is not None else None,
            "target_dir": str(target_dir.resolve()),
            "seeded_case_ids": [],
            "fallback_case_ids": [],
            "records": [],
        }
        if cache_dir is None or not cache_dir.is_dir():
            result["fallback_case_ids"] = sorted(case_ids)
            return result
        target_dir.mkdir(parents=True, exist_ok=True)
        for case_id in sorted(case_ids):
            require_safe_id(case_id, label="subject tube cache case_id")
            source_tube = cache_dir / f"{case_id}.npz"
            source_audit = cache_dir / f"{case_id}.audit.json"
            present = (source_tube.is_file(), source_audit.is_file())
            if present == (False, False):
                result["fallback_case_ids"].append(case_id)
                continue
            if present[0] != present[1]:
                raise RuntimeError(
                    f"subject tube cache pair is incomplete for {case_id}"
                )
            self._validate_cached_tube(case_id, source_tube, source_audit)
            target_tube = target_dir / source_tube.name
            target_audit = target_dir / source_audit.name
            if target_tube.exists() or target_audit.exists():
                if not target_tube.is_file() or not target_audit.is_file():
                    raise RuntimeError(
                        f"run-local subject tube cache pair is incomplete for {case_id}"
                    )
                self._validate_cached_tube(case_id, target_tube, target_audit)
                if (
                    sha256_file(target_tube) != sha256_file(source_tube)
                    or sha256_file(target_audit) != sha256_file(source_audit)
                ):
                    raise ValueError(
                        f"run-local subject tube cache conflicts for {case_id}"
                    )
                method = "existing"
            else:
                tube_method = self._link_or_copy(source_tube, target_tube)
                try:
                    audit_method = self._link_or_copy(source_audit, target_audit)
                except Exception:
                    target_tube.unlink(missing_ok=True)
                    raise
                method = (
                    "hardlink"
                    if tube_method == audit_method == "hardlink"
                    else "copy_or_mixed"
                )
            result["seeded_case_ids"].append(case_id)
            result["records"].append({
                "case_id": case_id,
                "source_tube": str(source_tube),
                "source_audit": str(source_audit),
                "target_tube": str(target_tube),
                "target_audit": str(target_audit),
                "tube_sha256": sha256_file(target_tube),
                "audit_sha256": sha256_file(target_audit),
                "method": method,
            })
        return result

    @staticmethod
    def _require_four_gpu_assignment(value: Any) -> str:
        devices = [item.strip() for item in str(value).split(",") if item.strip()]
        if devices != ["0", "1", "2", "3"]:
            raise ValueError(
                "subject-motion training requires CUDA_VISIBLE_DEVICES=0,1,2,3"
            )
        return ",".join(devices)

    def _balance_training_rows(
        self,
        rows: list[dict[str, Any]],
        artifact_root: Path,
    ) -> list[dict[str, Any]]:
        """Align equal-scene oversampling to complete global batches."""

        policy = self.config["lora"].get("scene_balancing")
        if policy != "oversample_each_scene_to_largest_world_aligned":
            raise ValueError(f"unsupported subject-motion balancing policy: {policy}")
        by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_scene[row["scene_id"]].append(row)
        if not by_scene:
            raise ValueError("subject-motion balancing requires training rows")
        global_batch = int(self.config["lora"]["global_batch_size"])
        scene_count = len(by_scene)
        alignment = global_batch // math.gcd(global_batch, scene_count)
        largest = max(len(items) for items in by_scene.values())
        target = ((largest + alignment - 1) // alignment) * alignment
        balanced: list[dict[str, Any]] = []
        for scene_id in sorted(by_scene):
            items = sorted(by_scene[scene_id], key=lambda row: row["case_id"])
            balanced.extend(
                dict(items[index % len(items)]) for index in range(target)
            )
        parallelism = self._training_parallelism(
            metadata_row_count=len(balanced)
        )
        lora = self.config["lora"]
        write_json(artifact_root / "training_sampling_plan.json", {
            "policy": policy,
            "unique_case_count": len(rows),
            "metadata_row_count": len(balanced),
            "input_scene_counts": dict(sorted(Counter(
                row["scene_id"] for row in rows
            ).items())),
            "balanced_scene_counts": dict(sorted(Counter(
                row["scene_id"] for row in balanced
            ).items())),
            "per_scene_target": target,
            "global_batch_alignment": alignment,
            "dataset_repeat": int(lora["dataset_repeat"]),
            "num_epochs": int(lora["num_epochs"]),
            **parallelism,
            "expected_optimizer_steps_per_epoch": parallelism[
                "optimizer_steps_per_epoch"
            ],
            "expected_total_optimizer_steps": parallelism[
                "total_optimizer_steps"
            ],
            "shuffle": True,
            "sampler_seed": int(lora["seed"]),
            "sampler_generator": "torch.Generator",
            "cross_rank_duplicate_policy": "forbidden_after_balancing",
        })
        return balanced

    def prepare_training(
        self,
        train_case_ids: list[str],
        run_dir: Path,
    ) -> dict[str, Any]:
        prepared = super().prepare_training(train_case_ids, run_dir)
        prepared.update({
            "adapter": "wan22_subject_motion",
            "subject_motion": self.config["st_tube_iou"],
            "initial_lora_checkpoint": str(
                self._resolve_warm_start_checkpoint()
            ),
            "mask_cache_policy": "validated_previous_run_seed_or_materialize",
        })
        return prepared

    def _training_command(self, runtime_root: Path) -> list[str]:
        del runtime_root
        return [
            "bash",
            str(self.project_root / "scripts" / "train_wan22_subject_motion.sh"),
        ]

    def _training_environment(
        self,
        run_dir: Path,
        dataset_dir: Path,
        metadata: Path,
    ) -> dict[str, str]:
        environment = super()._training_environment(
            run_dir,
            dataset_dir,
            metadata,
        )
        environment.pop("ST_TUBE_IOU_CONFIG_JSON", None)
        environment.pop("ST_TUBE_IOU_METRICS_PATH", None)
        environment.update({
            "SUBJECT_MOTION_CONFIG_JSON": json.dumps(
                self.config["st_tube_iou"],
                sort_keys=True,
                separators=(",", ":"),
            ),
            "SUBJECT_MOTION_METRICS_PATH": str(
                run_dir
                / "artifacts"
                / "wan22"
                / "training_subject_motion_metrics.jsonl"
            ),
            "LORA_CHECKPOINT": str(self._resolve_warm_start_checkpoint()),
            "CUDA_VISIBLE_DEVICES": self._require_four_gpu_assignment(
                self.runtime.get("cuda_visible_devices")
            ),
        })
        return environment

    def train(
        self,
        prepared_training: dict[str, Any],
        job_path: Path,
    ) -> dict[str, Any]:
        run_dir = job_path.parent
        train_ids = list(prepared_training["case_ids"])
        if self.execute and train_ids:
            cache_audit = self._seed_subject_tube_cache(
                train_ids,
                run_dir / "artifacts" / "wan22" / "dataset" / "masks",
            )
            write_json(
                run_dir / "artifacts" / "wan22" / "subject_tube_cache_seed.json",
                cache_audit,
            )
        result = super().train(prepared_training, job_path)
        return {
            **result,
            "adapter": "wan22_subject_motion",
            "warm_start_checkpoint": str(
                self._resolve_warm_start_checkpoint()
            ),
        }


__all__ = ["Wan22SubjectMotionAdapter"]
