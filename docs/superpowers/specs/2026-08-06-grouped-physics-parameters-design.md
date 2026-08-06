# Grouped Physics Parameters Design

## Scope

The current `physics.json` files become formal-parameter documents rather than
annotation/audit inventories. Every quantity marked `annotated=false` is
deleted, and the `annotated` member is removed from every retained quantity.
No media bytes, mask bytes, new physics files, or new Dataset release are
created.

Five scenes adopt grouped physics. `push_bottle` is deliberately left flat for
later scene-specific redesign, but its retained quantities also lose the
redundant `annotated` member.

## Document contract

Grouped scenes use exactly:

```json
{
  "case_id": "...",
  "scene_id": "...",
  "physics": {
    "objects": {
      "object_1": {
        "mass": {"value": 0.1, "unit": "kg", "symbol": "m"}
      }
    },
    "environment": {}
  }
}
```

Object IDs are contiguous `object_1`, `object_2`, ... and follow first-frame
mask order. Quantity leaves contain exactly `value`, `unit`, and `symbol`.
`push_bottle.physics` remains a flat map of the same three-field leaves until
its dedicated redesign.

## Classification

| Scene | Objects | Environment | Deleted |
| --- | --- | --- | --- |
| `collision_1d` | each ball: `mass`, `radius`, `initial_velocity` | empty | `striker_initial_velocity` |
| `inclined_plane_slide` | object 1: `mass`, `length` | `incline_angle`, `gravity_acceleration` | `initial_velocity`, `calibration_length`, `friction_force`, `theoretical_acceleration`, `kinetic_friction_coefficient` |
| `parabolic_motion` | object 1: `mass`, `radius`, `initial_horizontal_velocity`, `launch_height` | empty | all four photogate/ramp/release auxiliary fields |
| `pendulum` | object 1: optional `mass`, `radius`, `initial_angle` | `string_length` | `pendulum_length` |
| `uniform_circular_motion` | each object: `orbit_radius` | `angular_velocity` | `angular_velocity_rad_s` |
| `push_bottle` | deferred | deferred | no current false quantity |

The migration deletes 1,232 false quantities and 95 inclined-plane friction
coefficients. It retains 3,959 formal quantities in total.

## Runtime projection

The Dataset Loader exposes grouped `case.physics` for grouped scenes. A single
Dataset helper projects grouped leaves to the established semantic names
(`ball_1_mass`, `block_mass`, and so on) for consumers that require a flat
quantity channel. This projection is computed, not stored, and is not a second
source of truth. `push_bottle` passes through this helper unchanged.

Baseline input policy becomes `case.physics`: every retained quantity is a
formal input quantity, so filtering by `annotated=true` no longer exists.
Caption validation requires every retained symbol to appear in the caption.
The inclined-plane captions remove the deleted friction symbol.

Mask manifests use `object_1`, `object_2`, ... and canonical grouped physics
paths. Only manifest JSON metadata changes; PNG and NPZ masks do not.

## Validation

- exactly 799 `physics.json` documents;
- exactly 3,959 retained quantity leaves;
- no `annotated` key anywhere in current `physics.json` files;
- grouped scenes contain exactly `objects` and `environment`;
- `push_bottle` remains the only explicit flat-scene exception;
- all grouped object IDs are contiguous and align with available mask order;
- every retained symbol appears in its Case caption;
- no removed parameter remains in scene configs, captions, or mask physics paths;
- zero media, PNG, or NPZ changes.

