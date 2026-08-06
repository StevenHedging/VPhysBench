# WAN2.2 Symbol–Value Cross-Attention Baseline Design

## Objective

Build a new, independently identified WAN2.2-TI2V-5B finetuning baseline that
conditions video generation on the first frame, the original caption, and every
registry-selected scalar physical annotation.  Each annotation is represented
as one `symbol + value` token in the same 4096-dimensional word-vector space as
WAN's frozen UMT5 encoder.  The original text sequence consumes those tokens
through explicit trainable cross-attention before it reaches the WAN DiT.

The baseline must train on every Dataset 13 View A training split and generate
every corresponding test split.  Existing scene evaluators score supported
scenes.  `vertical_spring_oscillator` retains predictions, manifests, token
audits, checkpoints, and run metadata but is reported as unscored because its
evaluator does not exist.  The same honest unsupported record is used for any
other scene that the current evaluation registry cannot score.

## Alternatives Considered

1. **Post-UMT5 bottleneck cross-attention (selected).**  Reuse frozen UMT5's
   token embedding table for symbols, construct SI-aware value vectors, add the
   two vectors, and cross-attend from the contextual UMT5 caption sequence.
   This matches the requested data flow while adding only a compact trainable
   conditioning module.
2. **Sentinel replacement followed by token self-attention.**  This is close to
   `wan22_quantity_embedding`, but replacement discards rather than preserves
   the symbol vector and does not provide explicit text-to-physics
   cross-attention.
3. **Inject symbol–value tokens directly into each DiT block.**  This offers a
   stronger intervention but changes WAN internals, increases the LoRA/control
   surface, and makes the experiment harder to compare with existing WAN
   baselines.

## Input and Adapter Contract

The model prompt is exactly normalized `case.text.prompt`; numeric values are
never appended to it.  A bundle-local registry selects independent scalar
quantities and freezes field name, expected unit, SI conversion, quantity kind,
and stable unit ID.  For every selected record the adapter emits:

- `name`, `symbol`, raw value and unit;
- rounded display value and SI-canonical value;
- seven SI dimension exponents `[L,M,T,I,Theta,N,J]`;
- stable `quantity_type_id` and `unit_id`;
- source role and symbol occurrence audit in the original prompt.

The adapter fails closed when a required value is missing, non-finite, has an
unexpected unit, has an empty symbol, or its symbol is absent from the original
caption.  Symbol matching treats ASCII identifiers as token-bound names and
Unicode mathematical symbols as literal spans.  No annotation is guessed from
free text.

## Conditioning Architecture

Let UMT5 hidden size be `D=4096`, attention bottleneck be `A=512`, and the
number of selected quantities be `m`.

1. UMT5 tokenizes and self-attention-encodes the original caption into
   `H_text in R^(1 x L x D)`.  UMT5 remains frozen.
2. For each symbol, its subword IDs are obtained with the same UMT5 tokenizer,
   without special tokens.  Their frozen token-embedding vectors are mean
   pooled and RMS-normalized to form `E_symbol in R^(m x D)`.  This avoids an
   additional UMT5-XXL forward pass while retaining its pretrained lexical
   space.
3. The value branch borrows the existing quantity baseline's eight analytic SI
   magnitude features.  It separately embeds magnitude, seven SI dimensions,
   and exact unit ID, concatenates them, and projects to `D`, producing
   `E_value in R^(m x D)` in FP32.
4. Each physical token is
   `Z_phys = LayerNorm(E_symbol + E_value)`.  Additive fusion is selected over
   an inner product because an inner product collapses the word vector to a
   scalar; elementwise multiplication is less stable when either branch has
   small coordinates.
5. A pre-normalized multi-head cross-attention layer projects text queries and
   physical keys/values to `A`, applies eight-head attention, projects the
   result back to `D`, and adds a trainable gated residual.  Padding positions
   are restored to exact zero.  The negative CFG branch uses ordinary UMT5
   encoding without physical tokens.

Only the symbol–value conditioner and WAN DiT LoRA weights are trainable.  UMT5,
VAE, and base DiT weights remain frozen.  The combined safetensors checkpoint
uses distinct prefixes and is strictly topology-checked before either the
conditioner or LoRA weights can mutate a pipeline.

## Training and Evaluation

The experiment uses all seven View A train/test scene partitions with seed 42.
Training uses eight A100 GPUs, bf16 WAN computation, an FP32 conditioner,
gradient checkpointing, LoRA rank 32 on the same 300 WAN targets as the
existing quantity baseline, AdamW, constant learning rate `1e-4`, weight decay
`0.01`, one dataset repeat, two epochs, and scene balancing to the largest
scene.  This yields a meaningful cross-scene run without silently inheriting
the old baseline's much larger ten-epoch budget.

Generation uses 50 denoising steps, CFG 5.0, 121 frames at 24 fps, the
scene-native 480x832 or 832x480 canvas, and seed 42.  All test cases are
generated.  Supported scene evaluators use a run-local protocol derived from
`scene_default_v10`; unsupported scenes are explicitly configured with the
standard `unsupported` evaluator so their absence does not invalidate the
remaining scores.

The AtomicRun directory records frozen task/baseline/dataset identities,
adapter outputs, training dataset and sampling plans, combined checkpoint and
manifest, loss/gradient audits, jobs, predictions, per-case token audits,
evaluation outputs, and a final report.  Failed or interrupted execution is
left in place with its true state and logs.

## Failure Handling and Verification

Unit tests cover registry validation, symbol presence, numeric features,
symbol token pooling, additive fusion, masked cross-attention, CFG branch
isolation, strict checkpoint topology, baseline discovery, and dry-run task
compilation.  Each behavior is introduced with a failing test before
implementation.  A one-step eight-GPU smoke train and single-case generation
must pass before the full run starts.  Completion requires fresh targeted and
full repository tests plus artifact audits that reconcile selected case IDs,
checkpoint hashes, predictions, and evaluator statuses.

## Scope Boundaries

This work does not implement a spring evaluator, alter Dataset 13, change
existing baseline identities, use temporal force curves as scalar summaries,
or claim unsupported scenes in an aggregate physics score.
