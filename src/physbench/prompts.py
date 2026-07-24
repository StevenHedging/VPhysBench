from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .io import canonical_sha256, load_json


DEFAULT_TRAIN_PROFILE = "physics_natural"
DEFAULT_EVAL_PROFILES = ("physics_natural",)
SUPPORTED_PROMPT_PROFILES = ("generic", "physics_natural")


class PromptRegistry:
    """Load and deterministically render versioned prompt profiles."""

    def __init__(self, config_dir: str | Path):
        self.config_dir = Path(config_dir).resolve()
        if not self.config_dir.is_dir():
            raise FileNotFoundError(f"prompt config directory not found: {self.config_dir}")
        self.profiles: dict[str, dict[str, Any]] = {}
        for path in sorted(self.config_dir.glob("*.json")):
            profile = load_json(path)
            self._validate_profile(profile, path)
            profile_id = profile["profile_id"]
            if profile_id in self.profiles:
                raise ValueError(f"duplicate prompt profile {profile_id}")
            self.profiles[profile_id] = profile
        if not self.profiles:
            raise ValueError(f"no prompt profiles found in {self.config_dir}")

    @staticmethod
    def _validate_profile(profile: dict[str, Any], path: Path) -> None:
        if profile.get("schema_version") != "1.0":
            raise ValueError(f"prompt profile schema_version must be 1.0: {path}")
        profile_id = profile.get("profile_id")
        if not isinstance(profile_id, str) or not profile_id:
            raise ValueError(f"prompt profile_id is required: {path}")
        scenes = profile.get("scenes")
        if not isinstance(scenes, dict) or not scenes:
            raise ValueError(f"prompt profile scenes are required: {path}")
        for scene_id, scene in scenes.items():
            if not isinstance(scene.get("base_prompt"), str) or not scene["base_prompt"].strip():
                raise ValueError(f"prompt profile {profile_id}/{scene_id} requires base_prompt")
            clauses = scene.get("parameter_clauses", [])
            if not isinstance(clauses, list):
                raise ValueError(f"prompt profile {profile_id}/{scene_id} clauses must be a list")
            names: set[str] = set()
            for clause in clauses:
                name = clause.get("name")
                if not isinstance(name, str) or not name:
                    raise ValueError(f"prompt profile {profile_id}/{scene_id} clause requires name")
                if name in names:
                    raise ValueError(f"duplicate prompt parameter {profile_id}/{scene_id}/{name}")
                names.add(name)
                if "{value}" not in str(clause.get("template", "")):
                    raise ValueError(
                        f"prompt clause {profile_id}/{scene_id}/{name} must contain {{value}}"
                    )
                precision = clause.get("precision")
                if not isinstance(precision, int) or precision < 0:
                    raise ValueError(
                        f"prompt clause {profile_id}/{scene_id}/{name} requires non-negative precision"
                    )

    def require(self, profile_ids: Iterable[str | None]) -> None:
        unknown = sorted({
            profile_id for profile_id in profile_ids
            if profile_id is not None and profile_id not in self.profiles
        })
        if unknown:
            raise ValueError(
                f"unknown prompt profiles {unknown}; available={sorted(self.profiles)}"
            )

    def snapshot(self, profile_ids: Iterable[str | None]) -> dict[str, Any]:
        selected = sorted({value for value in profile_ids if value is not None})
        self.require(selected)
        return {
            "schema_version": "1.0",
            "config_dir": str(self.config_dir),
            "profiles": {
                profile_id: {
                    "config": self.profiles[profile_id],
                    "sha256": canonical_sha256(self.profiles[profile_id]),
                }
                for profile_id in selected
            },
        }

    def resolve(self, case: dict[str, Any], profile_id: str, *, role: str) -> dict[str, Any]:
        self.require([profile_id])
        profile = self.profiles[profile_id]
        scene_id = case["scene_id"]
        scene = profile["scenes"].get(scene_id)
        if scene is None:
            raise ValueError(f"prompt profile {profile_id} does not support scene {scene_id}")
        prompt = scene["base_prompt"].strip()
        used_parameters: dict[str, dict[str, Any]] = {}
        rendered_clauses = []
        quantities = case.get("physical_parameters", {})
        for clause in scene.get("parameter_clauses", []):
            name = clause["name"]
            quantity = quantities.get(name)
            usable = (
                isinstance(quantity, dict)
                and quantity.get("annotated", True)
                and quantity.get("value") is not None
            )
            if not usable:
                if clause.get("required", False):
                    raise ValueError(
                        f"case {case.get('case_id')} lacks required prompt parameter "
                        f"{profile_id}/{name}"
                    )
                continue
            expected_unit = clause.get("expected_unit")
            actual_unit = str(quantity.get("unit", "")).strip()
            if expected_unit is not None and actual_unit != expected_unit:
                raise ValueError(
                    f"case {case.get('case_id')} prompt parameter {name} unit "
                    f"{actual_unit!r} != expected {expected_unit!r}"
                )
            value = quantity["value"]
            if not isinstance(value, (int, float)):
                raise ValueError(
                    f"case {case.get('case_id')} prompt parameter {name} must be numeric"
                )
            rendered_value = f"{float(value):.{int(clause['precision'])}f}"
            rendered_clauses.append(clause["template"].format(value=rendered_value))
            used_parameters[name] = {
                "value": value,
                "unit": actual_unit,
                "rendered_value": rendered_value,
            }
        if rendered_clauses:
            prompt += (
                str(scene.get("parameter_intro", " Physical parameters: "))
                + str(scene.get("parameter_separator", "; ")).join(rendered_clauses)
                + str(scene.get("parameter_outro", "."))
            )
        record = {
            "schema_version": "1.0",
            "case_id": case["case_id"],
            "scene_id": scene_id,
            "role": role,
            "prompt_profile_id": profile_id,
            "profile_sha256": canonical_sha256(profile),
            "prompt": prompt,
            "used_parameters": used_parameters,
        }
        record["prompt_sha256"] = canonical_sha256(prompt)
        return record
