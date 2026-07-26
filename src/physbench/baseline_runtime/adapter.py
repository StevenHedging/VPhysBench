from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from ..baseline_api.interfaces import DataAdapter
from ..io import canonical_sha256, sha256_file
from ..prompts import PromptRegistry


RESOURCE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "baseline_plugins"
    / "resources"
)


class StandardDataAdapter(DataAdapter):
    """Declarative T2V/I2V adaptation owned by a managed Baseline."""

    PRESETS = {"standard_i2v_v1", "standard_t2v_v1"}

    def __init__(self, config: dict[str, Any]):
        self.config = copy.deepcopy(config)
        preset = self.config.get("preset")
        if preset not in self.PRESETS:
            raise ValueError(
                f"unsupported managed adapter preset {preset!r}; "
                f"supported: {sorted(self.PRESETS)}"
            )
        profile_set = self.config.get("profile_set")
        if not isinstance(profile_set, str) or not profile_set:
            raise ValueError("managed adapter requires profile_set")
        self.profile_dir = RESOURCE_ROOT / profile_set
        if not self.profile_dir.is_dir():
            raise FileNotFoundError(
                f"managed adapter profile set not found: {self.profile_dir}"
            )
        self.registry = PromptRegistry(self.profile_dir)
        if set(self.registry.profiles) != {"generic", "physics"}:
            raise ValueError(
                "managed adapter profile set must define generic and physics"
            )
        spatial = self.config.get("spatial")
        temporal = self.config.get("temporal")
        if not isinstance(spatial, dict) or not isinstance(temporal, dict):
            raise ValueError(
                "managed adapter requires spatial and temporal objects"
            )
        profiles = spatial.get("scene_profiles")
        if not isinstance(profiles, dict) or not profiles:
            raise ValueError(
                "managed adapter spatial.scene_profiles must be non-empty"
            )
        for scene_id, profile in profiles.items():
            if not isinstance(scene_id, str) or not isinstance(profile, dict):
                raise ValueError(
                    "managed adapter scene profiles must map strings to objects"
                )
        if "num_frames" in temporal:
            frames = int(temporal["num_frames"])
            if frames < 1:
                raise ValueError(
                    "managed adapter temporal.num_frames must be positive"
                )
            if (
                temporal.get("valid_frame_rule") == "4n+1"
                and (frames - 1) % 4
            ):
                raise ValueError(
                    "managed adapter temporal.num_frames must satisfy 4n+1"
                )
        if "max_frames" in temporal:
            maximum = int(temporal["max_frames"])
            minimum = int(temporal.get("min_frames", 1))
            if minimum < 1 or minimum > maximum:
                raise ValueError(
                    "managed adapter temporal frame bounds are invalid"
                )
            if temporal.get("valid_frame_rule") == "4n+1" and (
                (minimum - 1) % 4 or (maximum - 1) % 4
            ):
                raise ValueError(
                    "managed adapter temporal frame bounds must satisfy 4n+1"
                )
        if preset == "standard_i2v_v1":
            policy = self.config.get("first_frame_policy")
            if policy not in {"require_asset", "asset_or_reference_frame0"}:
                raise ValueError(
                    "managed I2V first_frame_policy must be require_asset or "
                    "asset_or_reference_frame0"
                )

    @property
    def dependency_fingerprints(self) -> dict[str, str]:
        return {
            "src/physbench/baseline_runtime/adapter.py": sha256_file(
                Path(__file__)
            ),
            (
                "src/physbench/baseline_plugins/resources/"
                f"{self.config['profile_set']}/generic.json"
            ): sha256_file(self.profile_dir / "generic.json"),
            (
                "src/physbench/baseline_plugins/resources/"
                f"{self.config['profile_set']}/physics.json"
            ): sha256_file(self.profile_dir / "physics.json"),
        }

    @property
    def stage_fingerprints(self) -> dict[str, str]:
        profiles = self.registry.snapshot(["generic", "physics"])["profiles"]
        implementation = canonical_sha256(self.dependency_fingerprints)
        return {
            "spatial": canonical_sha256({
                "config": self.config["spatial"],
                "implementation": implementation,
            }),
            "temporal": canonical_sha256({
                "config": self.config["temporal"],
                "implementation": implementation,
            }),
            "paradigm": canonical_sha256({
                "preset": self.config["preset"],
                "first_frame_policy": self.config.get("first_frame_policy"),
                "implementation": implementation,
            }),
            "text": canonical_sha256({
                "profile_set": self.config["profile_set"],
                "generic_profile": profiles["generic"],
                "implementation": implementation,
            }),
            "physics": canonical_sha256({
                "profile_set": self.config["profile_set"],
                "physics_profile": profiles["physics"],
                "implementation": implementation,
            }),
        }

    @property
    def fingerprint(self) -> str:
        return canonical_sha256({
            "type": "managed_standard_data_adapter_v1",
            "stages": self.stage_fingerprints,
        })

    @property
    def materialization_fingerprint(self) -> str:
        stages = self.stage_fingerprints
        return canonical_sha256({
            "type": "managed_media_materialization_v1",
            "spatial": stages["spatial"],
            "temporal": stages["temporal"],
            "paradigm": stages["paradigm"],
        })

    def _spatial_profile(self, scene_id: str) -> dict[str, Any]:
        profiles = self.config["spatial"]["scene_profiles"]
        try:
            return copy.deepcopy(profiles[scene_id])
        except KeyError as exc:
            raise ValueError(
                f"managed adapter has no spatial profile for {scene_id}"
            ) from exc

    def describe(self) -> dict[str, Any]:
        return {
            "type": "managed_standard_data_adapter_v1",
            "preset": self.config["preset"],
            "fingerprint": self.fingerprint,
            "materialization_fingerprint": (
                self.materialization_fingerprint
            ),
            "stage_fingerprints": self.stage_fingerprints,
            "stages": [
                "spatial",
                "temporal",
                "paradigm",
                "text",
                "physics",
            ],
            "ownership": "baseline_recipe_core_implementation",
            "source_assets_mutated": False,
            "native_inputs_are_opaque_to_benchmark": False,
            "config": self.config,
        }

    def adapt_case(
        self, case: dict[str, Any], conditioning: str, *, role: str
    ) -> dict[str, Any]:
        if conditioning not in {"generic", "physics"}:
            raise ValueError(
                f"unsupported managed conditioning {conditioning}"
            )
        generic = self.registry.resolve(
            {
                "case_id": case["case_id"],
                "scene_id": case["scene_id"],
                "physical_parameters": {},
            },
            "generic",
            role=role,
        )
        if generic["used_parameters"]:
            raise AssertionError(
                "managed generic adaptation leaked physical parameters"
            )
        record = (
            self.registry.resolve(
                {
                    "case_id": case["case_id"],
                    "scene_id": case["scene_id"],
                    "physical_parameters": case["physics"],
                },
                "physics",
                role=role,
            )
            if conditioning == "physics"
            else generic
        )
        profile = self._spatial_profile(case["scene_id"])
        temporal = copy.deepcopy(self.config["temporal"])
        preset = self.config["preset"]
        vision: dict[str, Any] = {
            "paradigm": (
                "image2video"
                if preset == "standard_i2v_v1"
                else "text2video"
            )
        }
        first_frame_source = None
        if preset == "standard_i2v_v1":
            first_frame = case["assets"].get("first_frame")
            reference = case["assets"].get(
                "physics_reference_video"
            )
            policy = self.config["first_frame_policy"]
            if not first_frame and policy == "require_asset":
                raise ValueError(
                    f"managed I2V case has no first-frame asset: "
                    f"{case['case_id']}"
                )
            if not first_frame and not reference:
                raise ValueError(
                    f"managed I2V case has neither first-frame nor physics "
                    f"reference asset: {case['case_id']}"
                )
            first_frame_source = (
                "assets.first_frame"
                if first_frame
                else "assets.physics_reference_video:frame0"
            )
            vision.update({
                "first_frame_asset": first_frame,
                "first_frame_reference_asset": (
                    reference if not first_frame else None
                ),
                "first_frame_source": first_frame_source,
                "first_frame_policy": policy,
            })
        generation_shape = {
            **profile,
            **{
                key: value
                for key, value in temporal.items()
                if key
                in {
                    "fps",
                    "num_frames",
                    "max_frames",
                    "min_frames",
                    "valid_frame_rule",
                }
            },
        }
        if "resolution" in generation_shape:
            generation_shape["resolution"] = str(
                generation_shape["resolution"]
            )
        record.update({
            "schema_version": "2.0",
            "conditioning": conditioning,
            "data_adapter_fingerprint": self.fingerprint,
            "materialization_fingerprint": (
                self.materialization_fingerprint
            ),
            "stages": {
                "spatial": {
                    "type": "managed_scene_profile_v1",
                    "target_profile": profile,
                    "materialization": "deferred_to_managed_driver",
                },
                "temporal": {
                    "type": "managed_temporal_recipe_v1",
                    **temporal,
                    "materialization": "deferred_to_managed_driver",
                },
                "paradigm": {
                    "type": preset,
                    "mode": vision["paradigm"],
                    "source": first_frame_source,
                },
                "text": {
                    "type": "managed_prompt_profile_v1",
                    "generic_description": generic["prompt"],
                    "contains_detailed_physics": False,
                },
                "physics": {
                    "type": "managed_physics_profile_v1",
                    "enabled": conditioning == "physics",
                    "strategy": (
                        "append_structured_values_to_prompt"
                        if conditioning == "physics"
                        else "disabled"
                    ),
                    "used_parameters": record["used_parameters"],
                },
            },
            "native_inputs": {
                "vision": vision,
                "text": {"prompt": record["prompt"]},
                "generation_shape": generation_shape,
            },
        })
        if conditioning == "generic" and record["used_parameters"]:
            raise AssertionError(
                "managed generic adaptation leaked physical parameters"
            )
        return record
