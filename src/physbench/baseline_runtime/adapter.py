from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from ..baseline_api.input_policy import validate_input_policy
from ..baseline_api.interfaces import DataAdapter
from ..io import canonical_sha256, load_json, sha256_file


RESOURCE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "baseline_plugins"
    / "resources"
)


class StructuredPhysicsTextRenderer:
    """Append audited physical quantities to a Dataset-owned base prompt."""

    def __init__(self, template_set: str):
        self.template_set = template_set
        self.path = RESOURCE_ROOT / f"{template_set}.json"
        if not self.path.is_file():
            raise FileNotFoundError(
                f"physics template set not found: {self.path}"
            )
        self.value = load_json(self.path)
        if self.value.get("schema_version") != "1.0":
            raise ValueError(
                "physics template set must use schema_version=1.0"
            )
        if self.value.get("template_set_id") != template_set:
            raise ValueError(
                "physics template_set_id does not match its configured name"
            )
        scenes = self.value.get("scenes")
        if not isinstance(scenes, dict) or not scenes:
            raise ValueError("physics template set requires scenes")
        for scene_id, scene in scenes.items():
            if not isinstance(scene, dict):
                raise ValueError(
                    f"physics template scene {scene_id} must be an object"
                )
            clauses = scene.get("parameter_clauses")
            if not isinstance(clauses, list) or not clauses:
                raise ValueError(
                    f"physics template scene {scene_id} requires clauses"
                )
            names: set[str] = set()
            for clause in clauses:
                name = clause.get("name")
                if not isinstance(name, str) or not name or name in names:
                    raise ValueError(
                        f"invalid or duplicate physics clause {scene_id}/{name}"
                    )
                names.add(name)
                if "{value}" not in str(clause.get("template", "")):
                    raise ValueError(
                        f"physics clause {scene_id}/{name} requires {{value}}"
                    )
                precision = clause.get("precision")
                if not isinstance(precision, int) or precision < 0:
                    raise ValueError(
                        f"physics clause {scene_id}/{name} has invalid precision"
                    )

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.value)

    def render(
        self,
        case: dict[str, Any],
        base_prompt: str,
    ) -> tuple[str, dict[str, dict[str, Any]]]:
        try:
            scene = self.value["scenes"][case["scene_id"]]
        except KeyError as exc:
            raise ValueError(
                f"physics template set does not support {case['scene_id']}"
            ) from exc
        rendered_clauses: list[str] = []
        used_parameters: dict[str, dict[str, Any]] = {}
        quantities = case["physics"]
        for clause in scene["parameter_clauses"]:
            name = clause["name"]
            quantity = quantities.get(name)
            usable = (
                isinstance(quantity, dict)
                and quantity.get("annotated") is True
                and quantity.get("value") is not None
            )
            if not usable:
                if clause.get("required", False):
                    raise ValueError(
                        f"case {case['case_id']} lacks required annotated "
                        f"physics parameter {name}"
                    )
                continue
            unit = str(quantity.get("unit", "")).strip()
            expected_unit = clause.get("expected_unit")
            if expected_unit is not None and unit != expected_unit:
                raise ValueError(
                    f"case {case['case_id']} physics.{name} unit {unit!r} "
                    f"!= expected {expected_unit!r}"
                )
            value = quantity["value"]
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
            ):
                raise ValueError(
                    f"case {case['case_id']} physics.{name} must be numeric"
                )
            rendered_value = f"{float(value):.{clause['precision']}f}"
            rendered_clauses.append(
                clause["template"].format(value=rendered_value)
            )
            used_parameters[name] = {
                "value": value,
                "unit": unit,
                "rendered_value": rendered_value,
            }
        if not rendered_clauses:
            return base_prompt, used_parameters
        prompt = (
            base_prompt.rstrip()
            + str(scene.get("parameter_intro", " Physical parameters: "))
            + str(scene.get("parameter_separator", "; ")).join(
                rendered_clauses
            )
            + str(scene.get("parameter_outro", "."))
        )
        return prompt, used_parameters


class StandardDataAdapter(DataAdapter):
    """Declarative, Baseline-owned T2V/I2V/V2V input adaptation."""

    PRESETS = {
        "standard_i2v_v1",
        "standard_t2v_v1",
        "standard_v2v_v1",
    }

    def __init__(
        self,
        config: dict[str, Any],
        input_policy: dict[str, Any],
    ):
        self.config = copy.deepcopy(config)
        self.input_policy = validate_input_policy(input_policy)
        preset = self.config.get("preset")
        if preset not in self.PRESETS:
            raise ValueError(
                f"unsupported managed adapter preset {preset!r}; "
                f"supported: {sorted(self.PRESETS)}"
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
            if policy != "require_asset":
                raise ValueError(
                    "managed I2V first_frame_policy must be require_asset"
                )
        if preset == "standard_v2v_v1":
            asset_key = self.config.get("video_asset_key")
            if not isinstance(asset_key, str) or not asset_key:
                raise ValueError(
                    "managed V2V adapter requires video_asset_key"
                )
            if asset_key in {
                "reference_video",
                "physics_reference_video",
                "source_video",
            }:
                raise ValueError(
                    "managed V2V video_asset_key must name an explicit "
                    "conditioning asset, not GT/reference/source video"
                )
        transform = self.config.get(
            "physics_transform",
            {"type": "none"},
        )
        self.physics_transform = copy.deepcopy(transform)
        transform_type = transform.get("type")
        usage = self.input_policy["physics"]["usage"]
        if usage == "ignored":
            if transform != {"type": "none"}:
                raise ValueError(
                    "physics-ignored adapter requires transform type none"
                )
            self.renderer = None
        else:
            if transform_type != "append_structured_text_v1":
                raise ValueError(
                    "standard physics adapter requires "
                    "append_structured_text_v1"
                )
            if self.input_policy["physics"]["representations"] != [
                "structured_text"
            ]:
                raise ValueError(
                    "standard physics adapter requires exactly the "
                    "structured_text representation"
                )
            self.renderer = StructuredPhysicsTextRenderer(
                transform["template_set"]
            )

    def dependency_paths(self) -> dict[str, Path]:
        if self.renderer is None:
            return {}
        return {
            (
                "src/physbench/baseline_plugins/resources/"
                f"{self.renderer.path.name}"
            ): self.renderer.path,
        }

    @property
    def dependency_fingerprints(self) -> dict[str, str]:
        return {
            "src/physbench/baseline_runtime/adapter.py": sha256_file(
                Path(__file__)
            ),
            **{
                name: sha256_file(path)
                for name, path in self.dependency_paths().items()
            },
        }

    @property
    def stage_fingerprints(self) -> dict[str, str]:
        core_implementation = self.dependency_fingerprints[
            "src/physbench/baseline_runtime/adapter.py"
        ]
        physics_implementation = canonical_sha256(
            self.dependency_fingerprints
        )
        return {
            "spatial": canonical_sha256({
                "config": self.config["spatial"],
                "implementation": core_implementation,
            }),
            "temporal": canonical_sha256({
                "config": self.config["temporal"],
                "implementation": core_implementation,
            }),
            "paradigm": canonical_sha256({
                "preset": self.config["preset"],
                "first_frame_policy": self.config.get("first_frame_policy"),
                "video_asset_key": self.config.get("video_asset_key"),
                "implementation": core_implementation,
            }),
            "text": canonical_sha256({
                "source": "case.text.prompt",
                "usage": "required",
                "implementation": core_implementation,
            }),
            "physics": canonical_sha256({
                "policy": self.input_policy["physics"],
                "transform": self.physics_transform,
                "renderer": (
                    self.renderer.fingerprint
                    if self.renderer is not None
                    else None
                ),
                "implementation": physics_implementation,
            }),
        }

    @property
    def fingerprint(self) -> str:
        return canonical_sha256({
            "type": "managed_standard_data_adapter_v2",
            "input_policy": self.input_policy,
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
        generation_mode = {
            "standard_t2v_v1": "t2v",
            "standard_i2v_v1": "i2v",
            "standard_v2v_v1": "v2v",
        }[self.config["preset"]]
        return {
            "type": "managed_standard_data_adapter_v2",
            "preset": self.config["preset"],
            "generation_mode": generation_mode,
            "input_policy": self.input_policy,
            "physics_representations": self.input_policy["physics"][
                "representations"
            ],
            "text_conditioning_required": True,
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
        self,
        case: dict[str, Any],
        *,
        role: str,
    ) -> dict[str, Any]:
        base_prompt = case["text"]["prompt"].strip()
        if not base_prompt:
            raise ValueError(
                f"case {case['case_id']} has an empty canonical prompt"
            )
        if self.renderer is None:
            prompt = base_prompt
            used_parameters: dict[str, dict[str, Any]] = {}
            transform_id = "none"
        else:
            prompt, used_parameters = self.renderer.render(
                case,
                base_prompt,
            )
            transform_id = self.physics_transform["type"]

        profile = self._spatial_profile(case["scene_id"])
        temporal = copy.deepcopy(self.config["temporal"])
        preset = self.config["preset"]
        generation_mode = {
            "standard_t2v_v1": "t2v",
            "standard_i2v_v1": "i2v",
            "standard_v2v_v1": "v2v",
        }[preset]
        vision: dict[str, Any] = {
            "paradigm": {
                "t2v": "text2video",
                "i2v": "image2video",
                "v2v": "video2video",
            }[generation_mode]
        }
        media_channels: list[dict[str, Any]] = []
        asset_access: list[str] = []
        paradigm_source = None
        if preset == "standard_i2v_v1":
            first_frame = case["assets"].get("first_frame")
            if not first_frame:
                raise ValueError(
                    f"managed I2V case has no first-frame asset: "
                    f"{case['case_id']}"
                )
            paradigm_source = "assets.first_frame"
            vision.update({
                "first_frame_asset": first_frame,
                "first_frame_source": paradigm_source,
                "first_frame_policy": "require_asset",
            })
            media_channels.append({
                "id": "initial_frame",
                "kind": "image",
                "origin": "dataset_asset",
                "asset_key": "first_frame",
                "binding": "native_inputs.vision.first_frame_asset",
            })
            asset_access.append("first_frame")
        elif preset == "standard_v2v_v1":
            asset_key = self.config["video_asset_key"]
            video = case["assets"].get(asset_key)
            if not video:
                raise ValueError(
                    f"managed V2V case has no assets.{asset_key}: "
                    f"{case['case_id']}"
                )
            paradigm_source = f"assets.{asset_key}"
            vision.update({
                "input_video_asset": video,
                "input_video_source": paradigm_source,
                "video_asset_key": asset_key,
            })
            media_channels.append({
                "id": "conditioning_video",
                "kind": "video",
                "origin": "dataset_asset",
                "asset_key": asset_key,
                "binding": "native_inputs.vision.input_video_asset",
            })
            asset_access.append(asset_key)
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
        native_inputs = {
            "vision": vision,
            "text": {"prompt": prompt},
            "generation_shape": generation_shape,
        }
        physics_channels = (
            [{
                "id": "structured_physics_text",
                "representation": "structured_text",
                "binding": "native_inputs.text.prompt",
                "transport": "inline_text",
                "used_parameters": sorted(used_parameters),
            }]
            if used_parameters
            else []
        )
        record = {
            "schema_version": "3.0",
            "case_id": case["case_id"],
            "scene_id": case["scene_id"],
            "role": role,
            "source_prompt_sha256": canonical_sha256(base_prompt),
            "prompt_sha256": canonical_sha256(prompt),
            "text_transform_id": transform_id,
            "prompt": prompt,
            "used_parameters": used_parameters,
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
                    "source": paradigm_source,
                },
                "text": {
                    "type": "dataset_case_prompt_v1",
                    "source": "case.text.prompt",
                    "source_prompt_sha256": canonical_sha256(base_prompt),
                },
                "physics": {
                    "usage": self.input_policy["physics"]["usage"],
                    "strategy": transform_id,
                    "used_parameters": used_parameters,
                },
            },
            "input_contract": {
                "schema_version": "1.0",
                "generation_mode": generation_mode,
                "text": {
                    "required": True,
                    "binding": "native_inputs.text.prompt",
                },
                "media_channels": media_channels,
                "physics_channels": physics_channels,
                "asset_access": asset_access,
            },
            "native_inputs": native_inputs,
        }
        return record
