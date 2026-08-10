# WAN2.2 Physics Text + First-Entity Vector

This Baseline conditions WAN2.2-TI2V-5B on the Dataset first frame, the
canonical caption plus `seven_scene_physics_text_v1`, and one trainable MLP
embedding of `physics.objects.object_1` in the fixed SI order
`[mass_kg, size_m, initial_velocity_m_per_s]`.

The entity MLP is exactly `3 -> 256 -> 1024 -> 4096`, with SiLU after the
first two linear layers and LayerNorm after the third. Its single output
replaces one native UMT5 sentinel after the frozen text encoder and is applied
only to the positive CFG branch. The base DiT, UMT5, and VAE remain frozen;
rank-32 DiT LoRA and the MLP train jointly.

Copy `baseline.local.example.json` to the ignored `baseline.local.json` and
set the local runtime/model paths before running the Baseline elsewhere.
