from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from ....io import canonical_sha256
from .contracts import (
    EntitySpec,
    LifecyclePolicy,
    ReferenceCapability,
)


ENTITY_MANIFEST_SCHEMA_VERSION = "1.0"

_COLLISION_PARAMETER_PATTERN = re.compile(
    r"^ball_(\d+)_(mass|radius|initial_velocity)$"
)
_COLLISION_REQUIRED_ATTRIBUTES = (
    ("mass", "kg"),
    ("radius", "m"),
    ("initial_velocity", "m/s"),
)


def _required_text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _canonical_json_value(value: Any, *, label: str) -> Any:
    """Copy a JSON-like value while rejecting ambiguous/non-finite input."""

    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} must not contain non-finite numbers")
        return value
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for raw_key, child in value.items():
            key = _required_text(raw_key, label=f"{label} key")
            if key in output:
                raise ValueError(f"{label} contains duplicate key {key!r}")
            output[key] = _canonical_json_value(
                child,
                label=f"{label}.{key}",
            )
        return {key: output[key] for key in sorted(output)}
    if isinstance(value, (list, tuple)):
        return [
            _canonical_json_value(child, label=f"{label}[{index}]")
            for index, child in enumerate(value)
        ]
    raise ValueError(
        f"{label} must contain only JSON-compatible values, got "
        f"{type(value).__name__}"
    )


def _frozen_json_value(value: Any, *, label: str) -> Any:
    canonical = _canonical_json_value(value, label=label)

    def freeze(item: Any) -> Any:
        if isinstance(item, dict):
            return MappingProxyType(
                {key: freeze(child) for key, child in item.items()}
            )
        if isinstance(item, list):
            return tuple(freeze(child) for child in item)
        return item

    return freeze(canonical)


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _strict_fields(
    value: Mapping[str, Any],
    *,
    allowed: set[str],
    required: set[str],
    label: str,
) -> None:
    missing = sorted(required - set(value))
    unknown = sorted(set(value) - allowed)
    if missing:
        raise ValueError(f"{label} is missing fields {missing}")
    if unknown:
        raise ValueError(f"{label} has unknown fields {unknown}")


@dataclass(frozen=True)
class PhysicalAttribute:
    """One structured quantity attached to an entity or apparatus."""

    name: str
    value: float
    unit: str
    annotated: bool
    symbol: str | None = None
    source_parameter: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "name",
            _required_text(self.name, label="physical attribute name"),
        )
        if (
            isinstance(self.value, bool)
            or not isinstance(self.value, (int, float))
            or not math.isfinite(float(self.value))
        ):
            raise ValueError(
                f"physical attribute {self.name}.value must be a finite number"
            )
        normalized_value = float(self.value)
        if normalized_value == 0.0:
            normalized_value = 0.0
        object.__setattr__(self, "value", normalized_value)
        object.__setattr__(
            self,
            "unit",
            _required_text(
                self.unit,
                label=f"physical attribute {self.name}.unit",
            ),
        )
        if not isinstance(self.annotated, bool):
            raise ValueError(
                f"physical attribute {self.name}.annotated must be boolean"
            )
        if self.symbol is not None:
            object.__setattr__(
                self,
                "symbol",
                _required_text(
                    self.symbol,
                    label=f"physical attribute {self.name}.symbol",
                ),
            )
        if self.source_parameter is not None:
            object.__setattr__(
                self,
                "source_parameter",
                _required_text(
                    self.source_parameter,
                    label=(
                        f"physical attribute {self.name}.source_parameter"
                    ),
                ),
            )

    def to_canonical_dict(self) -> dict[str, Any]:
        output = {
            "value": self.value,
            "unit": self.unit,
            "annotated": self.annotated,
            "source_parameter": self.source_parameter,
        }
        if self.symbol is not None:
            output["symbol"] = self.symbol
        return output


def _attribute_from_quantity(
    name: str,
    quantity: Any,
    *,
    source_parameter: str | None,
    label: str,
) -> PhysicalAttribute:
    raw = _mapping(quantity, label=label)
    _strict_fields(
        raw,
        allowed={"value", "unit", "annotated", "symbol"},
        required={"value", "unit", "annotated"},
        label=label,
    )
    return PhysicalAttribute(
        name=name,
        value=raw["value"],
        unit=raw["unit"],
        annotated=raw["annotated"],
        symbol=raw.get("symbol"),
        source_parameter=source_parameter,
    )


def _parse_physical_attributes(
    value: Any,
    *,
    label: str,
    case_physics: Mapping[str, Any] | None = None,
    canonical_input: bool = False,
) -> tuple[PhysicalAttribute, ...]:
    raw_attributes = _mapping(value, label=label)
    attributes: list[PhysicalAttribute] = []
    for raw_name, raw_quantity in raw_attributes.items():
        name = _required_text(raw_name, label=f"{label} key")
        if isinstance(raw_quantity, str):
            if canonical_input:
                raise ValueError(
                    f"{label}.{name} canonical value must be an object"
                )
            source_parameter = _required_text(
                raw_quantity,
                label=f"{label}.{name}",
            )
            if (
                case_physics is None
                or source_parameter not in case_physics
            ):
                raise ValueError(
                    f"{label}.{name} references missing case physics "
                    f"parameter {source_parameter!r}"
                )
            attribute = _attribute_from_quantity(
                name,
                case_physics[source_parameter],
                source_parameter=source_parameter,
                label=f"case.physics.{source_parameter}",
            )
        else:
            quantity = _mapping(
                raw_quantity,
                label=f"{label}.{name}",
            )
            allowed = {
                "value",
                "unit",
                "annotated",
                "symbol",
                "source_parameter",
            }
            _strict_fields(
                quantity,
                allowed=allowed,
                required={"value", "unit", "annotated"},
                label=f"{label}.{name}",
            )
            raw_source = quantity.get("source_parameter")
            source_parameter = (
                None
                if raw_source is None
                else _required_text(
                    raw_source,
                    label=f"{label}.{name}.source_parameter",
                )
            )
            attribute = PhysicalAttribute(
                name=name,
                value=quantity["value"],
                unit=quantity["unit"],
                annotated=quantity["annotated"],
                symbol=quantity.get("symbol"),
                source_parameter=source_parameter,
            )
            if source_parameter is not None and case_physics is not None:
                if source_parameter not in case_physics:
                    raise ValueError(
                        f"{label}.{name} references missing case physics "
                        f"parameter {source_parameter!r}"
                    )
                referenced = _attribute_from_quantity(
                    name,
                    case_physics[source_parameter],
                    source_parameter=source_parameter,
                    label=f"case.physics.{source_parameter}",
                )
                if attribute != referenced:
                    raise ValueError(
                        f"{label}.{name} disagrees with "
                        f"case.physics.{source_parameter}"
                    )
        attributes.append(attribute)
    attributes.sort(key=lambda item: item.name)
    return tuple(attributes)


def _validate_attributes(
    attributes: Sequence[PhysicalAttribute],
    *,
    label: str,
) -> tuple[PhysicalAttribute, ...]:
    normalized = tuple(attributes)
    if any(
        not isinstance(attribute, PhysicalAttribute)
        for attribute in normalized
    ):
        raise ValueError(f"{label} must contain PhysicalAttribute values")
    names = [attribute.name for attribute in normalized]
    if len(names) != len(set(names)):
        raise ValueError(f"{label} contains duplicate names")
    return tuple(sorted(normalized, key=lambda item: item.name))


@dataclass(frozen=True)
class EntityDeclaration:
    """Case-level declaration of one persistent dynamic physical identity."""

    entity_id: str
    role_id: str
    entity_class: str
    physical_attributes: tuple[PhysicalAttribute, ...] = ()
    parts: tuple[str, ...] = ()
    exchangeability_group: str | None = None
    lifecycle: LifecyclePolicy = LifecyclePolicy.PERSISTENT
    condition_anchor: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "entity_id",
            _required_text(self.entity_id, label="entity.entity_id"),
        )
        object.__setattr__(
            self,
            "role_id",
            _required_text(self.role_id, label="entity.role_id"),
        )
        object.__setattr__(
            self,
            "entity_class",
            _required_text(
                self.entity_class,
                label="entity.entity_class",
            ),
        )
        lifecycle = (
            self.lifecycle
            if isinstance(self.lifecycle, LifecyclePolicy)
            else LifecyclePolicy(self.lifecycle)
        )
        object.__setattr__(self, "lifecycle", lifecycle)
        parts = tuple(
            _required_text(part, label=f"entity {self.entity_id}.parts")
            for part in self.parts
        )
        if len(parts) != len(set(parts)):
            raise ValueError(
                f"entity {self.entity_id}.parts contains duplicates"
            )
        object.__setattr__(self, "parts", parts)
        if self.exchangeability_group is not None:
            object.__setattr__(
                self,
                "exchangeability_group",
                _required_text(
                    self.exchangeability_group,
                    label=(
                        f"entity {self.entity_id}.exchangeability_group"
                    ),
                ),
            )
        object.__setattr__(
            self,
            "physical_attributes",
            _validate_attributes(
                self.physical_attributes,
                label=f"entity {self.entity_id}.physical_attributes",
            ),
        )
        anchor = _mapping(
            self.condition_anchor,
            label=f"entity {self.entity_id}.condition_anchor",
        )
        object.__setattr__(
            self,
            "condition_anchor",
            _frozen_json_value(
                anchor,
                label=f"entity {self.entity_id}.condition_anchor",
            ),
        )

    def to_entity_spec(self) -> EntitySpec:
        """Project the data declaration onto the tracking core contract."""

        return EntitySpec(
            entity_id=self.entity_id,
            role_id=self.role_id,
            entity_class=self.entity_class,
            parts=self.parts,
            exchangeability_group=self.exchangeability_group,
            lifecycle=self.lifecycle,
            anchor=self.condition_anchor,
        )

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "role_id": self.role_id,
            "entity_class": self.entity_class,
            "physical_attributes": {
                attribute.name: attribute.to_canonical_dict()
                for attribute in self.physical_attributes
            },
            "parts": list(self.parts),
            "exchangeability_group": self.exchangeability_group,
            "lifecycle": self.lifecycle.value,
            "condition_anchor": _canonical_json_value(
                self.condition_anchor,
                label=f"entity {self.entity_id}.condition_anchor",
            ),
        }


@dataclass(frozen=True)
class ApparatusDeclaration:
    """A static scene anchor; it is not counted as a dynamic object."""

    apparatus_id: str
    apparatus_class: str
    physical_attributes: tuple[PhysicalAttribute, ...] = ()
    condition_anchor: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "apparatus_id",
            _required_text(
                self.apparatus_id,
                label="apparatus.apparatus_id",
            ),
        )
        object.__setattr__(
            self,
            "apparatus_class",
            _required_text(
                self.apparatus_class,
                label=f"apparatus {self.apparatus_id}.apparatus_class",
            ),
        )
        object.__setattr__(
            self,
            "physical_attributes",
            _validate_attributes(
                self.physical_attributes,
                label=(
                    f"apparatus {self.apparatus_id}.physical_attributes"
                ),
            ),
        )
        anchor = _mapping(
            self.condition_anchor,
            label=f"apparatus {self.apparatus_id}.condition_anchor",
        )
        object.__setattr__(
            self,
            "condition_anchor",
            _frozen_json_value(
                anchor,
                label=f"apparatus {self.apparatus_id}.condition_anchor",
            ),
        )

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "apparatus_id": self.apparatus_id,
            "apparatus_class": self.apparatus_class,
            "physical_attributes": {
                attribute.name: attribute.to_canonical_dict()
                for attribute in self.physical_attributes
            },
            "condition_anchor": _canonical_json_value(
                self.condition_anchor,
                label=f"apparatus {self.apparatus_id}.condition_anchor",
            ),
        }


@dataclass(frozen=True)
class EntityManifest:
    """Frozen entity cardinality, identity, anchors, and supervision contract."""

    case_id: str
    scene_id: str
    entities: tuple[EntityDeclaration, ...]
    apparatus: tuple[ApparatusDeclaration, ...]
    reference_capability: ReferenceCapability
    materializer_id: str
    schema_version: str = ENTITY_MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ENTITY_MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                "entity manifest schema_version must be "
                f"{ENTITY_MANIFEST_SCHEMA_VERSION}"
            )
        object.__setattr__(
            self,
            "case_id",
            _required_text(self.case_id, label="manifest.case_id"),
        )
        object.__setattr__(
            self,
            "scene_id",
            _required_text(self.scene_id, label="manifest.scene_id"),
        )
        object.__setattr__(
            self,
            "materializer_id",
            _required_text(
                self.materializer_id,
                label="manifest.materializer_id",
            ),
        )
        capability = (
            self.reference_capability
            if isinstance(self.reference_capability, ReferenceCapability)
            else ReferenceCapability(self.reference_capability)
        )
        object.__setattr__(self, "reference_capability", capability)
        entities = tuple(self.entities)
        apparatus = tuple(self.apparatus)
        if not entities:
            raise ValueError("entity manifest must declare at least one entity")
        if any(
            not isinstance(entity, EntityDeclaration)
            for entity in entities
        ):
            raise ValueError(
                "manifest.entities must contain EntityDeclaration values"
            )
        if any(
            not isinstance(item, ApparatusDeclaration)
            for item in apparatus
        ):
            raise ValueError(
                "manifest.apparatus must contain ApparatusDeclaration values"
            )
        entity_ids = [entity.entity_id for entity in entities]
        role_ids = [entity.role_id for entity in entities]
        apparatus_ids = [item.apparatus_id for item in apparatus]
        if len(entity_ids) != len(set(entity_ids)):
            raise ValueError("manifest contains duplicate entity_id values")
        if len(role_ids) != len(set(role_ids)):
            raise ValueError("manifest contains duplicate role_id values")
        if len(apparatus_ids) != len(set(apparatus_ids)):
            raise ValueError("manifest contains duplicate apparatus_id values")
        object.__setattr__(self, "entities", entities)
        object.__setattr__(self, "apparatus", apparatus)

    @property
    def entity_specs(self) -> tuple[EntitySpec, ...]:
        return tuple(entity.to_entity_spec() for entity in self.entities)

    @property
    def digest(self) -> str:
        return canonical_sha256(self.to_canonical_dict())

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "case_id": self.case_id,
            "scene_id": self.scene_id,
            "reference_capability": self.reference_capability.value,
            "materializer_id": self.materializer_id,
            "entities": [
                entity.to_canonical_dict() for entity in self.entities
            ],
            "apparatus": [
                item.to_canonical_dict() for item in self.apparatus
            ],
        }

    @classmethod
    def from_canonical_dict(
        cls,
        value: Mapping[str, Any],
    ) -> EntityManifest:
        return parse_entity_manifest(value)


def _parse_entity(
    value: Any,
    *,
    label: str,
    case_physics: Mapping[str, Any] | None,
    canonical_input: bool,
) -> EntityDeclaration:
    raw = _mapping(value, label=label)
    allowed = {
        "entity_id",
        "role_id",
        "entity_class",
        "physical_attributes",
        "parts",
        "exchangeability_group",
        "lifecycle",
        "condition_anchor",
    }
    required = {"entity_id", "entity_class"}
    if canonical_input:
        required = allowed
    _strict_fields(
        raw,
        allowed=allowed,
        required=required,
        label=label,
    )
    entity_id = _required_text(
        raw["entity_id"],
        label=f"{label}.entity_id",
    )
    raw_parts = raw.get("parts", [])
    if not isinstance(raw_parts, (list, tuple)):
        raise ValueError(f"{label}.parts must be an array")
    attributes = _parse_physical_attributes(
        raw.get("physical_attributes", {}),
        label=f"{label}.physical_attributes",
        case_physics=case_physics,
        canonical_input=canonical_input,
    )
    return EntityDeclaration(
        entity_id=entity_id,
        role_id=raw.get("role_id", entity_id),
        entity_class=raw["entity_class"],
        physical_attributes=attributes,
        parts=tuple(raw_parts),
        exchangeability_group=raw.get("exchangeability_group"),
        lifecycle=raw.get(
            "lifecycle",
            LifecyclePolicy.PERSISTENT.value,
        ),
        condition_anchor=raw.get("condition_anchor", {}),
    )


def _parse_apparatus(
    value: Any,
    *,
    label: str,
    case_physics: Mapping[str, Any] | None,
    canonical_input: bool,
) -> ApparatusDeclaration:
    raw = _mapping(value, label=label)
    allowed = {
        "apparatus_id",
        "apparatus_class",
        "physical_attributes",
        "condition_anchor",
    }
    required = {"apparatus_id", "apparatus_class"}
    if canonical_input:
        required = allowed
    _strict_fields(
        raw,
        allowed=allowed,
        required=required,
        label=label,
    )
    apparatus_id = _required_text(
        raw["apparatus_id"],
        label=f"{label}.apparatus_id",
    )
    return ApparatusDeclaration(
        apparatus_id=apparatus_id,
        apparatus_class=raw["apparatus_class"],
        physical_attributes=_parse_physical_attributes(
            raw.get("physical_attributes", {}),
            label=f"{label}.physical_attributes",
            case_physics=case_physics,
            canonical_input=canonical_input,
        ),
        condition_anchor=raw.get("condition_anchor", {}),
    )


def parse_entity_manifest(value: Mapping[str, Any]) -> EntityManifest:
    """Validate and reconstruct a canonical ``EntityManifest v1`` document."""

    raw = _mapping(value, label="entity manifest")
    fields = {
        "schema_version",
        "case_id",
        "scene_id",
        "reference_capability",
        "materializer_id",
        "entities",
        "apparatus",
    }
    _strict_fields(
        raw,
        allowed=fields,
        required=fields,
        label="entity manifest",
    )
    if raw["schema_version"] != ENTITY_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            "entity manifest schema_version must be "
            f"{ENTITY_MANIFEST_SCHEMA_VERSION}"
        )
    raw_entities = raw["entities"]
    raw_apparatus = raw["apparatus"]
    if not isinstance(raw_entities, list):
        raise ValueError("entity manifest.entities must be an array")
    if not isinstance(raw_apparatus, list):
        raise ValueError("entity manifest.apparatus must be an array")
    return EntityManifest(
        schema_version=raw["schema_version"],
        case_id=raw["case_id"],
        scene_id=raw["scene_id"],
        reference_capability=raw["reference_capability"],
        materializer_id=raw["materializer_id"],
        entities=tuple(
            _parse_entity(
                entity,
                label=f"entity manifest.entities[{index}]",
                case_physics=None,
                canonical_input=True,
            )
            for index, entity in enumerate(raw_entities)
        ),
        apparatus=tuple(
            _parse_apparatus(
                item,
                label=f"entity manifest.apparatus[{index}]",
                case_physics=None,
                canonical_input=True,
            )
            for index, item in enumerate(raw_apparatus)
        ),
    )


def _case_physics(case: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(case.get("physics"), label="case.physics")


def _reference_capability(case: Mapping[str, Any]) -> ReferenceCapability:
    if case.get("has_real_reference_video") is True:
        return ReferenceCapability.SAME_CASE_GT
    provenance = case.get("provenance", {})
    if isinstance(provenance, Mapping):
        parent_case_id = provenance.get("parent_case_id")
        if isinstance(parent_case_id, str) and parent_case_id.strip():
            return ReferenceCapability.PHYSICS_PARENT
    return ReferenceCapability.ANNOTATION_ONLY


def _attributes_from_case(
    physics: Mapping[str, Any],
    bindings: Sequence[tuple[str, str]],
    *,
    label: str,
    required: bool = True,
) -> tuple[PhysicalAttribute, ...]:
    output: list[PhysicalAttribute] = []
    for local_name, source_parameter in bindings:
        if source_parameter not in physics:
            if required:
                raise ValueError(
                    f"{label} requires case.physics.{source_parameter}"
                )
            continue
        output.append(
            _attribute_from_quantity(
                local_name,
                physics[source_parameter],
                source_parameter=source_parameter,
                label=f"case.physics.{source_parameter}",
            )
        )
    return tuple(output)


def _appearance_text(
    appearance: Mapping[str, Any],
    key: str,
) -> str | None:
    value = appearance.get(key)
    if value is None:
        return None
    return _required_text(value, label=f"case.appearance.{key}")


def _collision_entities(
    case: Mapping[str, Any],
) -> tuple[EntityDeclaration, ...]:
    appearance = _mapping(
        case.get("appearance"),
        label="case.appearance",
    )
    physics = _case_physics(case)
    sequence = appearance.get("ball_sequence")
    if not isinstance(sequence, list) or not sequence:
        raise ValueError(
            "collision legacy materialization requires a non-empty "
            "case.appearance.ball_sequence array"
        )
    labels = tuple(
        _required_text(
            value,
            label=f"case.appearance.ball_sequence[{index}]",
        )
        for index, value in enumerate(sequence)
    )
    count = len(labels)

    raw_materials = appearance.get("ball_materials")
    materials: tuple[str, ...] | None = None
    if raw_materials is not None:
        if (
            not isinstance(raw_materials, list)
            or len(raw_materials) != count
        ):
            raise ValueError(
                "case.appearance.ball_materials length must match "
                "ball_sequence"
            )
        materials = tuple(
            _required_text(
                value,
                label=f"case.appearance.ball_materials[{index}]",
            )
            for index, value in enumerate(raw_materials)
        )

    discovered: dict[int, set[str]] = {}
    for parameter_name in physics:
        match = _COLLISION_PARAMETER_PATTERN.fullmatch(parameter_name)
        if match is None:
            continue
        raw_index = match.group(1)
        index = int(raw_index)
        if index < 1 or raw_index != str(index):
            raise ValueError(
                f"collision physics parameter has non-canonical ball "
                f"index: {parameter_name}"
            )
        discovered.setdefault(index, set()).add(match.group(2))
    expected_indices = set(range(1, count + 1))
    if set(discovered) != expected_indices:
        raise ValueError(
            "collision ball_sequence and indexed physics disagree: "
            f"expected indices {sorted(expected_indices)}, found "
            f"{sorted(discovered)}"
        )
    required_names = {
        attribute_name
        for attribute_name, _ in _COLLISION_REQUIRED_ATTRIBUTES
    }
    for index in sorted(discovered):
        if discovered[index] != required_names:
            raise ValueError(
                f"collision ball_{index} requires exactly "
                "mass, radius, and initial_velocity"
            )

    entities: list[EntityDeclaration] = []
    repeated_labels = {
        label for label in labels if labels.count(label) > 1
    }
    for offset, appearance_label in enumerate(labels):
        index = offset + 1
        attributes: list[PhysicalAttribute] = []
        for attribute_name, expected_unit in (
            _COLLISION_REQUIRED_ATTRIBUTES
        ):
            parameter = f"ball_{index}_{attribute_name}"
            attribute = _attribute_from_quantity(
                attribute_name,
                physics[parameter],
                source_parameter=parameter,
                label=f"case.physics.{parameter}",
            )
            if attribute.unit != expected_unit:
                raise ValueError(
                    f"case.physics.{parameter}.unit must be "
                    f"{expected_unit!r}"
                )
            if not attribute.annotated:
                raise ValueError(
                    f"case.physics.{parameter} must be annotated"
                )
            if (
                attribute_name in {"mass", "radius"}
                and attribute.value <= 0.0
            ):
                raise ValueError(
                    f"case.physics.{parameter}.value must be positive"
                )
            attributes.append(attribute)
        anchor: dict[str, Any] = {
            "source": "condition_frame",
            "initial_order_index": offset,
            "appearance_label": appearance_label,
        }
        if materials is not None:
            anchor["material"] = materials[offset]
        entities.append(
            EntityDeclaration(
                entity_id=f"ball_{index}",
                role_id=f"body_{index}",
                entity_class="ball",
                physical_attributes=tuple(attributes),
                exchangeability_group=(
                    f"appearance:{appearance_label}"
                    if appearance_label in repeated_labels
                    else None
                ),
                lifecycle=LifecyclePolicy.PERSISTENT,
                condition_anchor=anchor,
            )
        )

    striker = physics.get("striker_initial_velocity")
    if striker is not None:
        alias = _attribute_from_quantity(
            "initial_velocity",
            striker,
            source_parameter="striker_initial_velocity",
            label="case.physics.striker_initial_velocity",
        )
        striker_index = appearance.get("striker_ball_index", 1)
        if (
            not isinstance(striker_index, int)
            or isinstance(striker_index, bool)
            or striker_index < 1
            or striker_index > len(entities)
        ):
            raise ValueError(
                "case.appearance.striker_ball_index must identify one "
                "ball_sequence entry"
            )
        striker_velocity = next(
            attribute
            for attribute in entities[striker_index - 1].physical_attributes
            if attribute.name == "initial_velocity"
        )
        schema_version = case.get("schema_version")
        if schema_version == "5.0":
            disagrees = (
                abs(alias.value) != abs(striker_velocity.value)
                or alias.unit != striker_velocity.unit
                or alias.annotated
                or not striker_velocity.annotated
            )
        else:
            disagrees = (
                alias.value != striker_velocity.value
                or alias.unit != striker_velocity.unit
                or alias.annotated != striker_velocity.annotated
            )
        if disagrees:
            raise ValueError(
                "case.physics.striker_initial_velocity disagrees with "
                f"case.physics.ball_{striker_index}_initial_velocity"
            )
    return tuple(entities)


def _default_apparatus(
    scene_id: str,
    physics: Mapping[str, Any],
) -> tuple[ApparatusDeclaration, ...]:
    if scene_id == "collision_1d":
        return (
            ApparatusDeclaration(
                apparatus_id="track_axis",
                apparatus_class="linear_collision_track",
                condition_anchor={
                    "source": "condition_frame",
                    "selector": "longitudinal_track_axis",
                },
            ),
        )
    if scene_id == "parabolic_motion":
        return (
            ApparatusDeclaration(
                apparatus_id="launch_frame",
                apparatus_class="horizontal_projectile_launch_frame",
                physical_attributes=_attributes_from_case(
                    physics,
                    (
                        ("launch_height", "launch_height"),
                        (
                            "photogate_distance_before_launch",
                            "photogate_distance_before_launch",
                        ),
                    ),
                    label="parabolic launch apparatus",
                    required=False,
                ),
                condition_anchor={
                    "source": "condition_or_reference",
                    "selector": "horizontal_launch_axis_and_gravity_frame",
                },
            ),
        )
    if scene_id == "free_fall":
        return (
            ApparatusDeclaration(
                apparatus_id="gravity_frame",
                apparatus_class="vertical_axis_and_ground",
                condition_anchor={
                    "source": "condition_or_reference",
                    "selector": "vertical_axis_and_ground_or_exit",
                },
            ),
        )
    if scene_id == "inclined_plane_slide":
        return (
            ApparatusDeclaration(
                apparatus_id="incline_track",
                apparatus_class="inclined_plane",
                physical_attributes=_attributes_from_case(
                    physics,
                    (
                        ("incline_angle", "incline_angle"),
                        ("calibration_length", "calibration_length"),
                        (
                            "kinetic_friction_coefficient",
                            "kinetic_friction_coefficient",
                        ),
                        ("friction_force", "friction_force"),
                        (
                            "gravity_acceleration",
                            "gravity_acceleration",
                        ),
                        (
                            "theoretical_acceleration",
                            "theoretical_acceleration",
                        ),
                    ),
                    label="inclined-plane apparatus",
                    required=False,
                ),
                condition_anchor={
                    "source": "condition_frame",
                    "selector": "incline_axis_and_endpoints",
                },
            ),
        )
    if scene_id == "pendulum":
        return (
            ApparatusDeclaration(
                apparatus_id="pivot_support",
                apparatus_class="pendulum_pivot_and_support",
                condition_anchor={
                    "source": "condition_frame",
                    "selector": "pivot_and_support",
                },
            ),
        )
    if scene_id == "uniform_circular_motion":
        return (
            ApparatusDeclaration(
                apparatus_id="rotation_platform",
                apparatus_class="rotating_disk",
                condition_anchor={
                    "source": "condition_frame",
                    "selector": "disk_center_and_boundary",
                },
            ),
        )
    if scene_id == "decelerated_slide":
        return (
            ApparatusDeclaration(
                apparatus_id="sliding_surface",
                apparatus_class="horizontal_track",
                condition_anchor={
                    "source": "condition_frame",
                    "selector": "surface_axis_and_contact_plane",
                },
            ),
        )
    return ()


def _single_entity_defaults(
    case: Mapping[str, Any],
    *,
    entity_id: str,
    role_id: str,
    entity_class: str,
    bindings: Sequence[tuple[str, str]],
    lifecycle: LifecyclePolicy,
    parts: tuple[str, ...] = (),
    appearance_bindings: Sequence[tuple[str, str]] = (),
) -> tuple[EntityDeclaration, ...]:
    physics = _case_physics(case)
    appearance = _mapping(
        case.get("appearance", {}),
        label="case.appearance",
    )
    anchor: dict[str, Any] = {"source": "condition_frame"}
    for anchor_name, appearance_name in appearance_bindings:
        value = _appearance_text(appearance, appearance_name)
        if value is not None:
            anchor[anchor_name] = value
    return (
        EntityDeclaration(
            entity_id=entity_id,
            role_id=role_id,
            entity_class=entity_class,
            physical_attributes=_attributes_from_case(
                physics,
                bindings,
                label=f"{case.get('scene_id')} entity",
            ),
            parts=parts,
            lifecycle=lifecycle,
            condition_anchor=anchor,
        ),
    )


def _circular_entities(
    case: Mapping[str, Any],
) -> tuple[EntityDeclaration, ...]:
    physics = _case_physics(case)
    appearance = _mapping(
        case.get("appearance"),
        label="case.appearance",
    )
    object_count = appearance.get("object_count")
    if (
        isinstance(object_count, bool)
        or not isinstance(object_count, int)
        or object_count <= 0
    ):
        raise ValueError(
            "circular materialization requires a positive integer "
            "case.appearance.object_count"
        )
    moving_objects = appearance.get("moving_objects")
    if (
        not isinstance(moving_objects, list)
        or len(moving_objects) != object_count
    ):
        raise ValueError(
            "case.appearance.moving_objects length must match object_count"
        )
    labels = tuple(
        _required_text(
            value,
            label=f"case.appearance.moving_objects[{index}]",
        )
        for index, value in enumerate(moving_objects)
    )
    orbit_pattern = re.compile(r"^object_(\d+)_orbit_radius$")
    found_indices = {
        int(match.group(1))
        for name in physics
        if (match := orbit_pattern.fullmatch(name)) is not None
    }
    expected_indices = set(range(1, object_count + 1))
    if found_indices != expected_indices:
        raise ValueError(
            "circular object_count and orbit-radius physics disagree: "
            f"expected {sorted(expected_indices)}, found "
            f"{sorted(found_indices)}"
        )
    repeated = {label for label in labels if labels.count(label) > 1}
    entities: list[EntityDeclaration] = []
    for offset, appearance_label in enumerate(labels):
        index = offset + 1
        entities.append(
            EntityDeclaration(
                entity_id=f"object_{index}",
                role_id=f"orbiter_{index}",
                entity_class="orbiter",
                physical_attributes=_attributes_from_case(
                    physics,
                    (
                        ("orbit_radius", f"object_{index}_orbit_radius"),
                        ("angular_velocity", "angular_velocity"),
                        (
                            "angular_velocity_rad_s",
                            "angular_velocity_rad_s",
                        ),
                    ),
                    label=f"circular object_{index}",
                    required=False,
                ),
                exchangeability_group=(
                    f"appearance:{appearance_label}"
                    if appearance_label in repeated
                    else None
                ),
                lifecycle=LifecyclePolicy.PERSISTENT,
                condition_anchor={
                    "source": "condition_frame",
                    "initial_order_index": offset,
                    "appearance_label": appearance_label,
                },
            )
        )
    return tuple(entities)


def _legacy_scene_entities(
    case: Mapping[str, Any],
) -> tuple[EntityDeclaration, ...]:
    scene_id = str(case["scene_id"])
    if scene_id == "collision_1d":
        return _collision_entities(case)
    if scene_id == "free_fall":
        return _single_entity_defaults(
            case,
            entity_id="falling_body",
            role_id="falling_body",
            entity_class="ball",
            bindings=(
                ("mass", "ball_mass"),
                ("radius", "ball_radius"),
                ("initial_height", "initial_height"),
                ("initial_velocity", "initial_velocity"),
            ),
            lifecycle=LifecyclePolicy.MAY_EXIT,
            appearance_bindings=(
                ("material", "ball_material"),
                ("size_class", "ball_size_class"),
            ),
        )
    if scene_id == "inclined_plane_slide":
        return _single_entity_defaults(
            case,
            entity_id="sliding_block",
            role_id="sliding_block",
            entity_class="block",
            bindings=(
                ("length", "block_length"),
                ("mass", "block_mass"),
                ("initial_velocity", "initial_velocity"),
            ),
            lifecycle=LifecyclePolicy.MAY_EXIT,
            appearance_bindings=(
                ("appearance_label", "block"),
                ("release_point", "release_point"),
            ),
        )
    if scene_id == "pendulum":
        return _single_entity_defaults(
            case,
            entity_id="bob",
            role_id="pendulum_bob",
            entity_class="pendulum_bob",
            bindings=(
                ("radius", "bob_radius"),
                ("initial_angle", "initial_angle"),
                ("pendulum_length", "pendulum_length"),
                ("string_length", "string_length"),
            ),
            lifecycle=LifecyclePolicy.PERSISTENT,
            parts=("string",),
            appearance_bindings=(("material", "bob_material"),),
        )
    if scene_id == "uniform_circular_motion":
        return _circular_entities(case)
    if scene_id == "parabolic_motion":
        return _single_entity_defaults(
            case,
            entity_id="projectile_ball",
            role_id="projectile_ball",
            entity_class="ball",
            bindings=(
                ("mass", "ball_mass"),
                ("radius", "ball_radius"),
                ("initial_height", "launch_height"),
                (
                    "initial_horizontal_velocity",
                    "initial_horizontal_velocity",
                ),
            ),
            lifecycle=LifecyclePolicy.MAY_EXIT,
            appearance_bindings=(
                ("material", "ball_material"),
                ("size_class", "ball_size_class"),
            ),
        )
    if scene_id == "decelerated_slide":
        physics = _case_physics(case)
        bindings = tuple(
            (name, name)
            for name in (
                "initial_velocity",
                "friction_coefficient",
                "block_mass",
            )
            if name in physics
        )
        if not bindings:
            raise ValueError(
                "decelerated_slide requires at least one known physics "
                "parameter or explicit entities"
            )
        return _single_entity_defaults(
            case,
            entity_id="sliding_block",
            role_id="sliding_block",
            entity_class="block",
            bindings=bindings,
            lifecycle=LifecyclePolicy.PERSISTENT,
            appearance_bindings=(("appearance_label", "block"),),
        )
    raise ValueError(
        f"scene {scene_id!r} has no legacy entity materializer; "
        "declare case.entities explicitly"
    )


def materialize_entity_manifest(
    case: Mapping[str, Any],
) -> EntityManifest:
    """Build a validated Case-level entity contract.

    Explicit ``case.entities`` declarations take precedence.  Current dataset
    scenes without declarations are converted by deterministic legacy
    materializers; collision conversion deliberately fails on any cardinality
    or physical-annotation disagreement instead of guessing.
    """

    raw_case = _mapping(case, label="case")
    case_id = _required_text(raw_case.get("case_id"), label="case.case_id")
    scene_id = _required_text(
        raw_case.get("scene_id"),
        label="case.scene_id",
    )
    physics = _case_physics(raw_case)
    explicit_entities = raw_case.get("entities")
    if explicit_entities is not None:
        if not isinstance(explicit_entities, list) or not explicit_entities:
            raise ValueError("case.entities must be a non-empty array")
        entities = tuple(
            _parse_entity(
                entity,
                label=f"case.entities[{index}]",
                case_physics=physics,
                canonical_input=False,
            )
            for index, entity in enumerate(explicit_entities)
        )
        materializer_id = "explicit_case_entities_v1"
    else:
        entities = _legacy_scene_entities(raw_case)
        materializer_id = (
            "legacy_collision_entities_v1"
            if scene_id == "collision_1d"
            else "legacy_scene_defaults_v1"
        )

    explicit_apparatus = raw_case.get("apparatus")
    if explicit_apparatus is None:
        apparatus = _default_apparatus(scene_id, physics)
    else:
        if not isinstance(explicit_apparatus, list):
            raise ValueError("case.apparatus must be an array or null")
        apparatus = tuple(
            _parse_apparatus(
                item,
                label=f"case.apparatus[{index}]",
                case_physics=physics,
                canonical_input=False,
            )
            for index, item in enumerate(explicit_apparatus)
        )

    return EntityManifest(
        case_id=case_id,
        scene_id=scene_id,
        entities=entities,
        apparatus=apparatus,
        reference_capability=_reference_capability(raw_case),
        materializer_id=materializer_id,
    )


__all__ = [
    "ENTITY_MANIFEST_SCHEMA_VERSION",
    "ApparatusDeclaration",
    "EntityDeclaration",
    "EntityManifest",
    "PhysicalAttribute",
    "materialize_entity_manifest",
    "parse_entity_manifest",
]
