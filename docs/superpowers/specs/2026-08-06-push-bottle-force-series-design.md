# Push-Bottle Force-Series Design

## Goal

Replace the lossy mean/peak push-force labels with the complete measured force
signal from each source XLSX sheet, attach that signal to the bottle object, and
bring `push_bottle` into the same object/environment structure as every other
current Scene.

## Source facts

- All 141 accepted Cases have one exact workbook/sheet match in
  `normalized_annotations.json`.
- Each sheet records an ordered force sequence at nominal 0.1-second precision.
- Source sheets contain real gaps and a few repeated timestamps. Therefore a
  constant-period encoding would silently alter the source annotation.
- Some post-release readings are negative sensor drift. The experiment has one
  declared push direction, and the Dataset contract stores magnitudes rather
  than signed directional scalars.

## Canonical representation

Every push-bottle Case uses:

```json
{
  "physics": {
    "objects": {
      "object_1": {
        "mass": {"value": 0.40761, "unit": "kg", "symbol": "m"},
        "height": {"value": 0.22, "unit": "m", "symbol": "h"},
        "applied_force": {
          "samples": [
            {"time": 0.1, "value": 0.16},
            {"time": 0.2, "value": 0.27}
          ],
          "time_unit": "s",
          "unit": "N",
          "symbol": "F(t)"
        }
      }
    },
    "environment": {}
  }
}
```

The sample list preserves source row order and explicit source timestamps. It
does not interpolate, smooth, deduplicate, or rebuild timestamps. Formal force
values are magnitudes: `max(force_n, 0.0)`. The existing provenance keeps the
unaltered signed source value, source unit, sequence, and conversion evidence.

`mean_applied_force` and `peak_applied_force` are removed from formal physics.
They are reproducible summaries, not independent physical conditions.

## Caption and naming

Every caption becomes:

> An upright bottle of mass m and height h is pushed near its top by a
> time-varying external force F(t), then tips and falls onto its side.

No numeric sample appears in the caption. Asset directory names drop `fmax` and
`fmean`, retaining mass, height, and source image number. Stable `case_id`
values remain unchanged so View membership and prior run references do not
change.

## Runtime contract

The Dataset helper recognizes two leaf kinds:

- scalar quantity: exact `value/unit/symbol`;
- time-series quantity: exact `samples/time_unit/unit/symbol`, with non-empty
  ordered samples, finite non-negative time and value, and exact sample fields
  `time/value`.

`flat_physics_quantities` still projects canonical paths to stable Baseline
names. The new semantic name is `applied_force`; existing scalar-only text and
quantity-embedding adapters skip unsupported time-series quantities rather
than summarizing them. A future force-series Baseline may consume the full
series through a dedicated representation.

## Validation

Validation must prove:

- 141 workbook/sheet-to-Case matches and 141 non-empty force series;
- exact source sample counts and timestamps;
- each formal force value equals `max(source force_n, 0)`;
- no formal mean/peak force quantity remains;
- all push Cases are grouped with exactly one `object_1` and empty environment;
- captions contain `m`, `h`, and `F(t)`;
- mask manifests bind `object_1` to mass, height, and applied force;
- no media bytes are modified and no hash is calculated.
