from __future__ import annotations

import copy
import io
import os
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import cv2

from ..contracts import EntityObservation


_SINGLE_ANCHOR_OBJECT_IDS = {
    "inclined_plane_slide": "sliding_block",
    "parabolic_motion": "projectile_ball",
    "pendulum": "bob",
    "push_bottle": "bottle",
    "vertical_spring_oscillator": "object_1",
}


def anchor_object_ids(scene_id: str, object_count: int) -> tuple[str, ...]:
    """Return the frozen first-frame NPZ identities for a scene.

    These are evaluator-facing anchor identities. They are deliberately
    independent from appearance labels such as material/spec names and from
    the physics-manifest ``object_N`` keys.
    """
    if scene_id == "collision_1d":
        if object_count not in {2, 3}:
            raise ValueError("collision_1d anchor object count must be 2 or 3")
        return tuple(f"ball_{index}" for index in range(1, object_count + 1))
    if scene_id == "uniform_circular_motion":
        if object_count not in {1, 2}:
            raise ValueError(
                "uniform_circular_motion anchor object count must be 1 or 2"
            )
        return tuple(f"object_{index}" for index in range(1, object_count + 1))
    object_id = _SINGLE_ANCHOR_OBJECT_IDS.get(scene_id)
    if object_id is None:
        raise ValueError(f"unsupported scene for anchor object IDs: {scene_id}")
    if object_count != 1:
        raise ValueError(f"{scene_id} anchor object count must be 1")
    return (object_id,)


def _deterministic_npz(arrays: Mapping[str, np.ndarray]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(
        output,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for name, array in arrays.items():
            payload = io.BytesIO()
            np.lib.format.write_array(
                payload,
                np.asarray(array),
                allow_pickle=False,
            )
            member = zipfile.ZipInfo(f"{name}.npy", (1980, 1, 1, 0, 0, 0))
            member.compress_type = zipfile.ZIP_DEFLATED
            member.external_attr = 0o600 << 16
            archive.writestr(member, payload.getvalue(), compresslevel=9)
    return output.getvalue()


def rewrite_anchor_object_id(path: str | Path, object_id: str) -> bool:
    """Atomically replace one anchor NPZ identity without changing its arrays."""
    target = Path(path)
    with np.load(target, allow_pickle=False) as payload:
        arrays = {name: np.asarray(payload[name]).copy() for name in payload.files}
    if set(arrays) != {"masks", "mask_ids", "object_ids", "frame_index"}:
        raise ValueError(f"anchor NPZ keys differ from the contract: {target}")
    current = arrays["object_ids"]
    if current.shape != (1,) or current.dtype.kind != "U":
        raise ValueError(f"anchor NPZ object_ids are invalid: {target}")
    if str(current[0]) == object_id:
        return False
    arrays["object_ids"] = np.asarray([object_id])
    encoded = _deterministic_npz(arrays)
    descriptor, temporary_value = tempfile.mkstemp(
        prefix=f".{target.name}.anchor-id-", dir=target.parent
    )
    temporary = Path(temporary_value)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return True


def anchor_geometry(mask: np.ndarray) -> dict[str, Any]:
    values = np.asarray(mask)
    if values.ndim != 2 or values.dtype != np.uint8:
        raise ValueError("anchor mask must be uint8 HW")
    if set(int(value) for value in np.unique(values)) - {0, 1}:
        raise ValueError("anchor mask values must be binary")
    ys, xs = np.nonzero(values)
    if not len(xs):
        raise ValueError("anchor mask must not be empty")
    return {
        "area_pixels": int(len(xs)),
        "centroid_xy": [float(xs.mean()), float(ys.mean())],
        "bbox_xyxy": [
            int(xs.min()),
            int(ys.min()),
            int(xs.max()) + 1,
            int(ys.max()) + 1,
        ],
    }


def entity_from_masks(
    object_id: str,
    mask_id: str,
    masks: np.ndarray,
    state: np.ndarray,
) -> EntityObservation:
    values = (np.asarray(masks) > 0).astype(np.uint8)
    states = np.asarray(state, dtype=np.uint8)
    if values.ndim != 3 or states.shape != (values.shape[0],):
        raise ValueError("candidate masks and states must use aligned THW/T layouts")
    count = values.shape[0]
    area = values.reshape(count, -1).sum(axis=1, dtype=np.int64)
    centroid = np.full((count, 2), np.nan, np.float32)
    bbox = np.full((count, 4), np.nan, np.float32)
    for index, mask in enumerate(values):
        ys, xs = np.nonzero(mask)
        if len(xs):
            centroid[index] = (float(xs.mean()), float(ys.mean()))
            bbox[index] = (xs.min(), ys.min(), xs.max(), ys.max())
    return EntityObservation(
        object_id=object_id,
        mask_id=mask_id,
        masks=values,
        centroid_xy=centroid,
        bbox_xyxy=bbox,
        area_pixels=area,
        state=states,
    )


def remap_physics_objects(
    physics: Mapping[str, Any],
    *,
    new_to_old: Mapping[str, str],
) -> dict[str, Any]:
    result = copy.deepcopy(dict(physics))
    objects = physics.get("objects")
    if not isinstance(objects, Mapping) or set(objects) != set(new_to_old):
        raise ValueError("physics remap must exactly cover all objects")
    if set(new_to_old.values()) != set(objects):
        raise ValueError("physics remap must be a permutation")
    remapped: dict[str, Any] = {}
    for new_object, old_object in new_to_old.items():
        object_index = new_object.rsplit("_", 1)[-1]
        quantities = copy.deepcopy(objects[old_object])
        for quantity in quantities.values():
            if not isinstance(quantity, dict) or not isinstance(quantity.get("symbol"), str):
                continue
            quantity["symbol"] = re.sub(
                r"_\d+$", f"_{object_index}", quantity["symbol"]
            )
        remapped[new_object] = quantities
    result["objects"] = remapped
    return result


def visual_size_order_matches_physics(
    physics: Mapping[str, Any],
    anchor_areas: Mapping[str, int],
    *,
    relative_tolerance: float = 0.08,
) -> bool:
    """Return whether distinguishable visual and physical size ranks agree.

    Near-ties are ignored because perspective and segmentation boundaries can
    perturb pixel area without implying an identity error.  A physically
    distinguishable pair whose masks are visually tied is rejected instead of
    guessed, so callers cannot silently apply a non-evidenced permutation.
    """
    objects = physics.get("objects")
    if not isinstance(objects, Mapping) or set(objects) != set(anchor_areas):
        raise ValueError("visual/physics size comparison must cover all objects")
    size_key = next(
        (
            candidate
            for candidate in ("radius", "orbit_radius")
            if all(candidate in objects[object_id] for object_id in objects)
        ),
        None,
    )
    if size_key is None:
        raise ValueError("physics objects do not expose a comparable size quantity")
    radii = {
        object_id: float(objects[object_id][size_key]["value"])
        for object_id in objects
    }
    ids = sorted(objects)
    for left_index, left in enumerate(ids):
        for right in ids[left_index + 1 :]:
            radius_scale = max(abs(radii[left]), abs(radii[right]))
            area_scale = max(abs(anchor_areas[left]), abs(anchor_areas[right]))
            radius_delta = radii[left] - radii[right]
            area_delta = anchor_areas[left] - anchor_areas[right]
            if radius_scale == 0 or abs(radius_delta) <= relative_tolerance * radius_scale:
                continue
            if area_scale == 0 or abs(area_delta) <= relative_tolerance * area_scale:
                raise ValueError(
                    "visually tied anchor areas cannot justify a physical identity remap"
                )
            if (radius_delta > 0) != (area_delta > 0):
                return False
    return True


def permute_entity_observations(
    entities: Mapping[str, EntityObservation],
    *,
    new_to_old: Mapping[str, str],
) -> dict[str, EntityObservation]:
    if set(entities) != set(new_to_old) or set(new_to_old.values()) != set(entities):
        raise ValueError("entity permutation must exactly cover all objects")
    result: dict[str, EntityObservation] = {}
    for new_object, old_object in new_to_old.items():
        source = entities[old_object]
        index = int(new_object.rsplit("_", 1)[-1])
        result[new_object] = EntityObservation(
            object_id=new_object,
            mask_id=f"{index:02d}",
            masks=np.asarray(source.masks).copy(),
            centroid_xy=np.asarray(source.centroid_xy).copy(),
            bbox_xyxy=np.asarray(source.bbox_xyxy).copy(),
            area_pixels=np.asarray(source.area_pixels).copy(),
            state=np.asarray(source.state).copy(),
        )
    return result


def write_anchor_assets(
    directory: str | Path,
    entity: EntityObservation,
    *,
    semantic_object_id: str,
    entity_class: str = "ball",
) -> dict[str, Any]:
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    mask = np.asarray(entity.masks[0], dtype=np.uint8)
    if not mask.any():
        raise ValueError(f"cannot write empty anchor for {entity.object_id}")
    np.savez_compressed(
        output / f"{entity.mask_id}.npz",
        masks=mask[None],
        mask_ids=np.asarray([entity.mask_id]),
        object_ids=np.asarray([semantic_object_id]),
        frame_index=np.asarray(0, np.int64),
    )
    if not cv2.imwrite(str(output / f"{entity.mask_id}.png"), mask * 255):
        raise RuntimeError(f"cannot write anchor visualization for {entity.object_id}")
    geometry = anchor_geometry(mask)
    return {
        "object_id": entity.object_id,
        "mask_id": entity.mask_id,
        **geometry,
        "entity_class": entity_class,
        "segmentation": {
            "model": "SAM2.1 Hiera Large",
            "review": "full-tube curation; anchor equals observation zero",
            "selection_policy": "v14_full_reference_observation_audit",
        },
    }


def remap_ordered_appearance(
    appearance: Mapping[str, Any],
    *,
    new_to_old: Mapping[str, str],
) -> dict[str, Any]:
    result = copy.deepcopy(dict(appearance))
    order = [
        int(new_to_old[f"object_{index}"].rsplit("_", 1)[-1]) - 1
        for index in range(1, len(new_to_old) + 1)
    ]
    if sorted(order) != list(range(len(order))):
        raise ValueError("appearance remap must be a contiguous object permutation")
    for key in ("ball_sequence", "ball_materials", "moving_objects"):
        value = result.get(key)
        if isinstance(value, list) and len(value) == len(order):
            result[key] = [value[index] for index in order]
    striker = result.get("striker_ball_index")
    if isinstance(striker, int) and 1 <= striker <= len(order):
        old_index = striker - 1
        result["striker_ball_index"] = order.index(old_index) + 1
    return result


__all__ = [
    "anchor_geometry",
    "anchor_object_ids",
    "entity_from_masks",
    "permute_entity_observations",
    "remap_ordered_appearance",
    "remap_physics_objects",
    "rewrite_anchor_object_id",
    "visual_size_order_matches_physics",
    "write_anchor_assets",
]
