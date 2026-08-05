# Physics Video Benchmark Dataset 12.0.0

This is the sole active runtime Dataset, `physics_video_six_scene_v12`.
It preserves all 799 V11 Case facts, Views, prompts, symbolic quantities,
media, masks, and provenance bindings.

Each Case references one locked schema-2 `physics.json`. Its physics object is
identical to inline `case.physics`: finite non-negative magnitudes with stable
symbols and corrected independent/audit-only roles. Motion direction remains
in the Case prompt. No version-suffixed Case-local physics file is used.
