# Vertical Spring Oscillator Import Design

**Date:** 2026-08-06  
**Scene ID:** `vertical_spring_oscillator`  
**Source archive:** `/root/Steven/竖直弹簧振子.zip`

## Objective

Import the vertical spring oscillator capture batch as canonical VPhysBench Cases. Each accepted Case must contain a reference video, its decoded frame zero, Case-local physics and Caption files, and a first-frame mask for the visible steel ball. The import must preserve source facts, exclude ambiguous inputs, and follow the Dataset contract in `datasets/README.md`.

## Accepted Scene Contract

### Video window

- The canonical start is the source frame at which the oscillator first returns to the release-side turning point after completing one full cycle.
- A Trial released below equilibrium starts at the first post-cycle lower turning point. A Trial released above equilibrium starts at the first post-cycle upper turning point.
- The selected frame is treated as the oscillator being at the annotated displacement and at rest.
- All source frames from the selected frame through source end-of-file are retained. The tail is not trimmed.
- The image is not cropped or scaled. Frames are not sampled, duplicated, removed after the selected start, or time-warped.
- Source variable-frame-rate timestamps are retained and shifted so the selected frame begins the canonical stream.
- Source display rotation is normalized to the displayed portrait orientation. The canonical video is encoded as broadly decodable H.264 in MP4 without changing displayed spatial resolution.

### Formal physics

`physics.json` uses SI units and non-negative scalar values only.

`physics.objects.object_1` contains:

- `mass`: value `0.5156`, unit `kg`, symbol `m`. This is the documented total moving mass of the steel ball and its following connectors.
- `radius`: value `0.025`, unit `m`, symbol `r`. This is the visible steel-ball radius.
- `initial_displacement`: the absolute value of the Trial's signed displacement converted from millimetres to metres, unit `m`, symbol `x_0`.

`physics.environment` contains:

- `spring_stiffness`: value `32.6213467096774`, unit `N/m`, symbol `k`.
- `natural_spring_length`: value `0.068`, unit `m`, symbol `L_0`.
- `gravity_acceleration`: value `9.80665`, unit `m/s^2`, symbol `g`.

The signed Trial displacement remains provenance. A positive source displacement means release below equilibrium; a negative source displacement means release above equilibrium. Its sign is expressed through Caption wording, never through a negative `physics.json` value.

The documented equilibrium spring length, static extension, theoretical period, formulas, raw workbook values, and the unexplained workbook value `42.55` remain provenance and do not become formal physics quantities. The import must not infer a meaning for `42.55`.

### Caption

Each Case uses one of the following English forms, selected from the sign of the source displacement:

> A vertically suspended oscillator of total moving mass m and ball radius r is attached to a spring with stiffness k and natural length L_0 under gravitational acceleration g. At the first frame, the oscillator is at displacement x_0 below equilibrium and starts from rest, then undergoes vertical free oscillation.

> A vertically suspended oscillator of total moving mass m and ball radius r is attached to a spring with stiffness k and natural length L_0 under gravitational acceleration g. At the first frame, the oscillator is at displacement x_0 above equilibrium and starts from rest, then undergoes vertical free oscillation.

The Caption contains every formal symbol verbatim and contains no concrete physical value or capture detail.

### Subject and mask

- The Scene has one formal object, `object_1`.
- `canonical/masks/01.png` and `01.npz` cover only visible steel-ball pixels in `first_frame.png`.
- The mask excludes the spring, suspension ring, following connectors, ruler, rig, background, shadows, hands, and release tools.
- The PNG and NPZ are full-resolution binary masks and have the same spatial dimensions as `first_frame.png`.
- `manifest.json` binds `object_1`, mask `01`, the first frame, and the object's physics fields.

## Import Architecture

The import is a staged, reproducible pipeline rather than direct bulk mutation of `datasets/assets/`:

1. **Intake normalization:** inventory the ZIP members, extract the original XLSX, record workbook rows and video metadata, and create a Trial-to-source mapping.
2. **Candidate analysis:** decode each mapped source video, track the steel-ball centre on the vertical axis, and identify a candidate post-cycle release-side turning frame.
3. **Pilot:** materialize a small set covering above-equilibrium releases, below-equilibrium releases, and multiple displacement magnitudes.
4. **Pilot review:** visually check the turning frame, tool absence, full ball visibility, mask boundary, and physics/Caption agreement. Adjust the shared detector only from documented pilot failures.
5. **Batch materialization:** create accepted Case directories using the validated detector and shared serializers.
6. **Full review and Dataset 13 publication:** review every accepted Case, write import/exclusion records and a summary, build Dataset `13.0.0`, and make it the repository's sole active runtime release.

The detector generates candidates; it never grants acceptance. Every materialized Case requires recorded visual approval.

## Dataset Release Integration

- Dataset `12.0.0` is frozen and must not be edited in place.
- Build Dataset `13.0.0` from the accepted current Cases plus the accepted `vertical_spring_oscillator` Cases.
- Add Scene metadata for the formal fields, Caption contract, mask requirement, and evaluation grouping of `vertical_spring_oscillator`.
- Assign the new Scene's accepted Cases to training and ID test sets. The ID test set contains no more than 20 Cases, and duplicate or near-duplicate source groups cannot cross the split boundary.
- Add the complete-Case evaluation grouping and Task/index entries required by the existing Dataset build contract.
- Update `physbench.data_layout.LATEST_DATASET`, CLI validation, Dataset documentation, and tests to use `13.0.0`.
- Preserve `12.0.0` in Git history rather than retaining multiple active runtime releases in `datasets/releases/`. This follows the repository's single-current-release policy.
- Do not push or publish Dataset `13.0.0` to a remote service as part of this local import.

## Per-Case Data Flow

1. Resolve one non-empty Trial row to exactly one source video.
2. Read the signed displacement and determine the release side.
3. Track ball vertical position and smooth only the analysis signal used to find extrema. Do not alter video pixels or timing.
4. Locate release, the opposite-side turning point, and the first subsequent same-side turning point. Select the exact source frame at that same-side extremum.
5. Reject the Trial if the cycle cannot be identified reliably or if the selected frame contains a release tool, an incomplete ball, or a severe occlusion.
6. Encode source frame `start_frame` through EOF as `canonical/reference.mp4`, preserving source presentation cadence and full displayed frame.
7. Decode canonical frame zero to `canonical/first_frame.png` and verify pixel equality against a fresh decode of canonical frame zero.
8. Generate and visually review the steel-ball mask, then write PNG, NPZ, and manifest.
9. Serialize `physics.json` and `caption.json`, using the source displacement sign only for `above` or `below`.
10. Append a provenance audit record containing the workbook row, source member, selected frame and timestamp, detector diagnostics, conversions, review status, and output Case identity.

Case IDs and directory names use normalized physics values and a stable source-video suffix. They do not encode background, camera, or other appearance information.

## Provenance and Storage

- Copy the unchanged ZIP to a dated directory under `datasets/provenance/source_archives/`. This directory is intentionally Git-ignored because the archive is approximately 13 GB.
- Save the original XLSX under a dated directory in `datasets/provenance/source_docs/` together with normalized annotations and mapping records.
- Save machine-readable import audit, exclusion, and summary files under `datasets/provenance/imports/`.
- Do not create asset hashes or a hash lock. Existing Dataset policy deliberately treats these as source records rather than security artifacts.
- Per-Case source media copies are optional and will not be duplicated when the unchanged source archive already preserves the media.

## Exclusions and Ambiguity Handling

A Trial is excluded, with a specific machine-readable reason, when any of the following holds:

- its Trial ID, video reference, or required displacement is missing;
- the workbook video reference is absent from the archive, including the observed `IMG_1652`, `IMG_1653`, and `IMG_1654` references;
- more than one non-identical source video maps to the same Trial;
- an identical duplicate source exists and the canonical source member cannot be selected deterministically;
- the first post-cycle release-side turning point cannot be located reliably;
- the ball is incomplete, severely occluded, or covered by a release tool at the candidate frame;
- a reliable steel-ball-only mask cannot be made;
- media decoding or encoding does not preserve the required temporal and spatial contract.

`IMG_1705.MOV` and `IMG_1705(1).MOV` must be compared by decoded/media content. If identical, only one deterministically named member is mapped and the duplicate relationship is recorded. No missing or ambiguous value is replaced with zero or guessed from a neighbouring Trial.

## Validation

### Automated validation

- Every accepted `case_id` and `scene_id` agrees across directory members and indexes.
- Every required Case member exists and parses.
- `physics.json` contains only the fixed object/environment structure and valid non-negative finite quantities with unique symbols.
- Every physics symbol occurs verbatim in the Caption, and the Caption contains no concrete physical value.
- The displacement sign and Caption direction agree.
- `first_frame.png` equals a direct decode of canonical video frame zero.
- Canonical video starts at the recorded exact source frame, retains all later source frames through EOF, has no crop or scale, and retains source presentation timing.
- Mask PNG and NPZ are binary, non-empty, dimensionally equal to the first frame, and bound to `object_1` by the manifest.
- Trial/source mapping is one-to-one after documented exclusions and duplicate collapse.
- Repository Dataset validators and relevant tests pass.

### Visual validation

For every accepted Case, review artifacts show that:

- the first frame is the first post-cycle return to the release-side turning point;
- the tool and hand have left the frame;
- the steel ball is fully visible;
- the video continues naturally to the untrimmed source tail;
- the mask contains only the steel ball;
- Caption direction matches the observed and annotated release side.

The final summary reports source Trial count, mapped count, accepted count, excluded count by reason, duplicate decisions, detector-review corrections, and validation results.

## Scope Boundaries

This import adds the new Scene and publishes it locally through Dataset `13.0.0`. It does not invent missing source annotations, redefine existing Scenes, modify Dataset `12.0.0` in place, add baseline weights, or publish/push to a remote repository.
