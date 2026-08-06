# Physical-Response Loss for a Physics Video World Model

## Status

The design decisions are approved and this written specification awaits final
user review before implementation planning. It defines the first
physical-response training and evaluation system built on the immutable
`physics_video_six_scene_v12` Dataset and WAN2.2-TI2V-5B.

The central objective is not merely to reconstruct one real trajectory. The
model must reproduce the real system's **signed response** when one supported
initial-state or physical parameter changes: the generated motion must change
in the same direction and by the same amount, within experimental uncertainty.

## 1. Scope and fixed decisions

The first version covers five scenes and the following response axes:

| Scene | Response axes | First-frame policy |
| --- | --- | --- |
| `pendulum` | string length; initial angle | each branch uses its corresponding real first frame |
| `collision_1d` | ball mass; signed initial velocity | branches share the same first frame for a numerical counterfactual |
| `inclined_plane_slide` | incline angle only | each branch uses its corresponding real first frame |
| `parabolic_motion` | signed initial horizontal velocity | branches share the same first frame |
| `uniform_circular_motion` | orbit radius | each branch uses its corresponding real first frame |

`push_bottle` is out of scope because bottle mass, height, identity, and force
history are not currently separable as clean single-factor responses.

The collision model is completely elastic. Restitution is neither an input nor
a response axis. Two-ball cases receive full signed-response supervision and
strong analytic constraints. Three-ball cases retain trajectory, validity,
non-interpenetration, momentum, and kinetic-energy constraints, but do not
receive a finite-difference mass or velocity response loss in this version.
The Dataset contains 264 two-ball cases and 66 three-ball cases.

The inclined-plane kinetic friction coefficient is a fixed scene constant:

```text
mu_k = 0.463
```

It is not a response axis, is not perturbed, and is not supplied as a constant
model token. It is used only by the analytic incline regularizer. Its source is
`datasets/provenance/source_docs/20260723_new_scenes/inclined_plane_annotations.xlsx`,
corroborated by the adjacent `normalized_annotations.json`. All 95 released
incline cases share this value.

No new Dataset release is created. `datasets/releases/12.0.0` and all Dataset
assets remain byte-for-byte unchanged. Response groupings, signed projections,
scene constants, splits, and derived labels live in a versioned training/task
overlay, content-addressed cache, and run artifacts.

## 2. Empirical and analytic teacher policy

Matched real experiments are the primary source of response direction and
magnitude. Analytic mechanics serves as a weak regularizer and label-quality
check. An analytic disagreement never overwrites a valid empirical target. A
disagreement larger than the response policy's pre-test audit threshold removes
that pair from the analytic term and triggers review; the empirical term remains
eligible if its tracking, calibration, and replicate-quality gates pass.

Completely elastic two-ball collision is the deliberate exception: signed
post-collision velocities, total momentum, and total kinetic energy are strong
constraints because perfect elasticity is part of the model contract. Real
collision tracks still provide the absolute state, contact timing, and
observational anchor.

The implementation must distinguish these teacher roles explicitly:

- `empirical_primary`: response derived from matched real trajectories;
- `analytic_weak`: ideal formula used as a lower-weight regularizer;
- `analytic_strong`: exact contract used for two-ball elastic collision;
- `absolute_empirical`: single-video state target, not a response target.

It must never silently replace an unavailable empirical response with an
analytic one while reporting the result as empirical.

## 3. Signed-response contract

For an ordered intervention pair with scalar response parameter `q_b > q_a`,
let `Y` be a scene-specific physical state trajectory or event summary. All
parameter values are converted to canonical SI units, and all state values are
expressed in a frozen scene coordinate system.

The empirical finite-difference response is

```text
R* = (Y_b* - Y_a*) / (q_b_tilde - q_a_tilde)
```

and the generated response is

```text
R_hat = (Y_hat_b - Y_hat_a) / (q_b_tilde - q_a_tilde).
```

`q_tilde` is the parameter after a train-split-only scale normalization. The
system uses finite differences rather than claiming an infinitesimal causal
derivative because the real Dataset contains discrete experiments with noise.
Adjacent parameter values define the primary local response. Longer-range
pairs are used to audit nonlinear amplitude and generalization, not to replace
the local target.

Direction and magnitude are separate contracts:

- direction is evaluated only when the real response exceeds twice its
  replicate-derived uncertainty;
- magnitude is measured with an uncertainty-normalized robust error;
- a target statistically indistinguishable from zero becomes a zero-response
  constraint rather than an arbitrary sign label.

The final report must retain these components separately. A weighted training
loss may not be presented as the sole scientific result.

## 4. Response overlay and pair compiler

### 4.1 Overlay identity

A versioned `physical_response_v1` policy references:

- Dataset ID `physics_video_six_scene_v12` and release `12.0.0`;
- the Dataset descriptor and resolved case set;
- the supported scene/axis registry;
- the fixed incline coefficient and its provenance;
- coordinate-system versions;
- nuisance matching rules;
- split and pair-selection policy;
- teacher and observer versions.

The compiler writes the fully resolved policy, case IDs, groups, pairs, and
input digests into the run. Nothing is written beneath `datasets/`.

### 4.2 Intervention groups

An intervention group has one response axis and a fixed nuisance signature:

```text
group_id
scene_id
response_axis
response_parameter_path
intervened_entity_id
entity_role
tracked_entity_mapping
response_component_paths
member_case_ids
ordered_parameter_values_si
nuisance_signature
first_frame_policy
teacher_roles
coordinate_frame_id
member_partitions
edge_partitions
```

`response_parameter_path` is a canonical grouped-physics path such as
`objects.object_1.initial_velocity`. `intervened_entity_id` and `entity_role`
bind that path to a frozen first-frame identity. `tracked_entity_mapping` maps
every state output to that frozen identity, and `response_component_paths`
orders the affected state vector. Collision formula indices and circular object
indices never follow instantaneous left/right position after motion begins.

For empirical groups, the nuisance signature includes every other formal
physics value and the full canonicalized `appearance` object. Only fields that
are explicitly documented as repeat identifiers may be omitted. The compiler
fails if more than one intended physical factor changes.

Cases with the same response value and nuisance signature are replicates. They
estimate the robust mean state and experimental variance at that value. They
do not form a response pair because their parameter difference is zero.

Within each eligible group, members are sorted by the signed SI parameter.
The compiler freezes the complete adjacent-value edge graph before assigning
partitions. A held-out value removes all of its incident training edges; its two
neighbors are never reconnected across the resulting gap. The ordering of every
retained edge is always stored, so a data-loader shuffle cannot reverse the
scientific sign convention.

### 4.3 Scene-specific pairing

- Pendulum length pairs hold initial angle, bob properties, and appearance
  fixed. Initial-angle pairs hold string length, bob properties, and appearance
  fixed.
- Collision velocity pairs hold ball identities, masses, radii, collision
  structure, direction roles, and other incident velocities fixed. Opposed
  incidents are eligible only when exactly one signed incident velocity varies.
- Collision mass currently has no strict mass-only empirical pair. It therefore
  uses a shared-first-frame numerical counterfactual with an
  `analytic_strong` response target and an empirical absolute-state anchor.
- Incline pairs hold block identity, mass, length, background, camera, track,
  release convention, gravity, and the fixed friction condition constant while
  varying only angle.
- Projectile pairs hold ball identity, mass, radius, launch height, appearance,
  and camera fixed while varying signed horizontal launch velocity.
- Circular pairs hold angular velocity, object identity and count, all other
  object radii, appearance, and camera fixed while varying one orbit radius.

If an exact pair cannot be built, that case remains eligible for ordinary
FlowMatch training but is excluded from response loss. Matching is never
relaxed merely to increase pair count.

### 4.4 Signed parameter projection

V12 stores velocity magnitudes while direction is represented in the prompt and
experiment structure. The overlay computes a signed training view without
changing `physics.json`:

- collision track `x` increases from image left to right; role and prompt
  semantics determine the sign of each incident velocity;
- projectile `x` uses the fixed calibrated image/world horizontal orientation,
  so leftward launch is negative;
- incline distance is positive down the slope;
- pendulum angle is signed about the downward vertical from pivot to bob;
- circular phase is unwrapped in the observed rotation direction, while radius
  remains non-negative.

Every derived sign is audited against the first frames and prompt semantics.
An ambiguous direction excludes the response label rather than defaulting to a
positive magnitude.

## 5. Response-specific holdouts

The existing View A test remains untouched and is never used for optimization,
loss selection, or response-label construction. It remains the conventional
single-video quality/generalization test; response claims come from the
additional sealed holdouts below.

Response experiments carve additional held-out cases from View A train. Those
videos are removed from both FlowMatch and response training for the applicable
run. Every experiment has explicit `response_train`, `response_validation`, and
`response_test` member and edge partitions. Validation alone selects loss
weights, checkpoints, observer tolerances, and stage progression. Response test
targets remain sealed until all configuration and checkpoint choices are
frozen. Two distinct overlay experiments avoid conflating nuisance and
parameter generalization:

1. **Group holdout:** complete intervention groups are held out. This measures
   transfer of a learned response to unseen nuisance/appearance groups.
2. **Value holdout:** selected parameter-level videos are held out while other
   levels in the same nuisance family may remain. Every training response edge
   incident to a held-out level is removed. An evaluation edge may use one
   retained anchor level and one held-out level, but its empirical response
   target is evaluation-only. This explicitly measures interpolation or
   endpoint extrapolation without claiming that both edge endpoints are unseen.

The two holdouts require separate, sealed run identities. Results may not be
pooled as if they used the same training set. When a scene lacks enough support
for a valid extrapolation split, the report states `not_estimable`; it does not
manufacture an extrapolation claim.

Split compilation is deterministic and keeps source-trial/near-duplicate
components together within the applicable member partition. It assigns
partitions on the frozen full edge graph and never recomputes adjacency after a
holdout. The compiler emits an overlap audit proving that no held-out video,
response edge, or empirical response target entered training and that no test
target entered validation-driven selection.

## 6. Real-video teacher cache

SAM2 is used offline to extract real-video state targets. The initial object
prompt comes from the Dataset's first-frame mask manifest. The two cases without
a first-frame mask remain valid for FlowMatch but are excluded from physical
loss. Any later mask for those cases is an overlay/cache artifact; it must not be
written back to `datasets/assets`. A verification gate records the Dataset tree
digest before and after cache construction and requires it to remain unchanged.

The cache stores, as applicable:

- object identity and per-frame soft/hard mask confidence;
- calibrated centroids, areas, and visibility;
- scene coordinates and their calibration source;
- velocities, accelerations, unwrapped angles, radii, and event times;
- repeat-level robust means and uncertainties;
- failure reasons and observer-quality gates.

The cache key includes the reference-video byte digest, first-frame-mask byte
digest, Dataset identity, SAM2 source commit, SAM2 configuration, checkpoint
digest, coordinate extractor version, and state extractor version. The current
SAM2 identity is:

```text
source:     /root/Nico/third_party/sam2
commit:     2b90b9f5ceec907a1c18123530e92e794ad901a4
checkpoint: /mnt/nvme1/NicoCache/checkpoints/sam2/sam2.1_hiera_tiny.pt
sha256:     7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69
```

Held-out teacher states are readable only by evaluation, not by the training
data loader.

## 7. Differentiable RGB observer

The main physical loss acts on decoded generated RGB, not on an unconstrained
latent prediction head:

```text
Wan v prediction
  -> differentiable clean-latent estimate z0_hat
  -> frozen Wan VAE decoder
  -> generated RGB frames
  -> frozen differentiable SAM2 tracking
  -> differentiable physical state extraction
  -> response loss
```

The public SAM2 video-predictor API cannot be used in this path because its
entry points are decorated with `torch.inference_mode()` and its media loader
expects files. The implementation uses `SAM2Train` or the low-level
`SAM2Base.forward_image`/`track_step` operations on tensors. SAM2 parameters are
frozen, but autograd through the input remains enabled.

Hard mask binarization, multimask `argmax`, hole filling, and non-overlap hard
post-processing are disabled. The observer consumes soft mask probabilities.
Object centroids, areas, velocities, and angular quantities are computed with
differentiable soft moments and finite differences.

Coordinate frames are obtained from the condition image or frozen real
calibration and are never re-fitted from a prediction. This prevents a generated
video from improving its score by moving the coordinate system. Generated
tracking confidence never masks its own loss: low teacher confidence can remove
an unreliable target, while low generated confidence incurs an explicit
validity penalty.

An independent, non-differentiable evaluation path processes saved RGB videos
after generation. It is a guard against adversarial textures that exploit the
differentiable observer.

## 8. Scene state registry

| Scene | Differentiable state | Primary response | Zero-response/validity terms |
| --- | --- | --- | --- |
| pendulum | signed bob angle, angular velocity, non-negative amplitude envelope, period | effective length to period; initial-angle magnitude to amplitude and nonlinear period | fixed pivot, identity, continuous swing |
| two-ball collision | track position, signed pre/post velocity, contact time/order | mass or incident velocity to both post-collision velocities | no penetration, correct separation, momentum and energy |
| three-ball collision | three track positions, signed velocities, ordered contact events, total momentum and kinetic energy | no finite-difference response in v1 | no penetration, identity, final separation, global momentum and energy |
| inclined plane | along-plane displacement, speed, acceleration, descent time, contact | angle to acceleration, displacement, and descent time | cross-track drift and loss of contact remain near zero |
| projectile | horizontal/vertical displacement, horizontal velocity, landing time | initial horizontal velocity to horizontal state | vertical response remains near zero |
| circular motion | radius, unwrapped phase, angular velocity, period, derived centripetal acceleration | orbit radius to path radius and centripetal acceleration | angular velocity and period remain near zero response |

Raw frame-index trajectories are not used when they create a false sign under a
small phase or event-time shift:

- pendulum trajectories are represented by phase-aligned curves plus amplitude
  and period summaries;
- collision velocities use pre-contact and post-separation windows aligned to
  the contact event;
- circular phase is expressed relative to frame-zero phase;
- incline and projectile time are physical time after applying
  `encoded_to_physical_speed`.

Training event alignment is differentiable: contact, turning point, and landing
summaries use soft temporal weights rather than an `argmax` frame. The hard
evaluator detects events independently. Teacher event times may choose which
frames receive the sparse 16/32-frame training loss, but they are loss-side
labels and are never passed to WAN as conditioning.

The weak analytic contracts are:

```text
pendulum:   l_eff = string_length + bob_radius
            T = 4 * sqrt(l_eff/g) * K(sin(theta0/2)^2)
incline:    a(theta) = g * (sin(theta) - 0.463 * cos(theta))
            da/dtheta = g * (cos(theta) + 0.463 * sin(theta))
projectile: dx/du = t; dv_x/du = 1; dy/du = 0; dT_land/du = 0
circular:   a_c = omega^2 * r; da_c/dr = omega^2
            domega/dr = 0; dT/dr = 0
```

`K` is the complete elliptic integral of the first kind, angles in formulas are
radians, and the pendulum response parameter `theta0` is the Dataset's
non-negative initial-angle magnitude. Initial-angle pairs never cross a signed
zero. The effective pivot-to-bob-center length is validated from the first-frame
geometry; a mismatch between `string_length + bob_radius` and calibration
removes the analytic pendulum term rather than changing the empirical target.
The response policy fixes laboratory gravity for this weak check at
`9.80665 m/s^2`.

The incline formula applies only while the block is sliding downslope with
`tan(theta) > 0.463`, continuous track contact, and before the block exits the
calibrated segment. Acceleration response is measured after motion onset in a
common physical-time window. The projectile zero-response identities apply
before landing and rely on the released fixed launch height. Circular
centripetal acceleration is derived differentiably from observed `r` and
`omega`, not from an extra Dataset label.

For a perfectly elastic two-ball collision:

```text
v1 = ((m1-m2)/(m1+m2))*u1 + (2*m2/(m1+m2))*u2
v2 = (2*m1/(m1+m2))*u1 + ((m2-m1)/(m1+m2))*u2
```

These formulas use signed track velocities. A global rule such as “larger mass
means larger velocity” is forbidden because the correct signed response depends
on the complete collision state.

For three-ball cases, global momentum and kinetic energy are computed in a
stable pre-contact window before the first contact and a stable terminal window
after the final detected contact and separation. They enter the strong analytic
term, but no closed-form per-ball response or response score is emitted. If the
terminal window is absent, the case records an event/validity failure and does
not fabricate terminal velocities.

## 9. WAN flow core and paired branches

The existing `FlowMatchSFTLoss` returns only one scalar and independently draws
noise and timestep for every call. The response trainer instead exposes the
intermediate flow values. With

```text
z_t = (1-sigma) * z0 + sigma * epsilon
v_target = epsilon - z0
```

the differentiable clean-latent estimate is

```text
z0_hat_generated = z_t_generated - sigma * v_pred
z0_hat_full = concat(stopgrad(first_frame_latent), z0_hat_generated)
```

`generated` denotes temporal latent indices after the fixed TI2V first-frame
latent. Physical loss never reconstructs, perturbs, or backpropagates through
the conditioned frame-zero latent.

Teacher availability and first-frame visibility are orthogonal contracts:

- `target_kind=empirical_pair` means both endpoints have real videos and both
  independently receive FlowMatch and absolute-state supervision;
- `target_kind=analytic_counterfactual` means only the factual endpoint has a
  real RGB target;
- `frame_kind=corresponding_real` means each response branch uses its own real
  first frame and target-derived noisy latent;
- `frame_kind=shared_anchor` means both response branches use one verified
  anchor first frame and the same generated-frame noisy latent.

Visible geometry or image-encoded initial-state axes—pendulum length/angle,
incline angle, and orbit radius—use `empirical_pair + corresponding_real`. Both
branches share timestep and the same noise realization, but each noisy latent is
constructed from its own real `z0` and its own first-frame latent.

Single-frame-hidden velocity axes use `empirical_pair + shared_anchor`. Every
real endpoint still receives its own independent factual FlowMatch/absolute
step. A separate response step chooses one endpoint as anchor and evaluates the
same anchor generated-frame `z_t` twice, once with `q_a` and once with `q_b`:

```text
z_t_generated(q_a) := z_t_generated(anchor)
z_t_generated(q_b) := z_t_generated(anchor)
first_frame(q_a)   := first_frame(anchor)
first_frame(q_b)   := first_frame(anchor)
```

The sampler alternates low- and high-value factual anchors deterministically,
while always computing `Y_hat_b - Y_hat_a` in canonical `q_b > q_a` order. This
prevents different noisy videos from explaining the hidden-parameter response.
Only the anchor's factual-condition branch receives FlowMatch in that response
step; both real endpoints receive factual losses through their independent
steps.

Collision mass uses `analytic_counterfactual + shared_anchor`: the compiler
selects an adjacent positive mass from the released collision mass support, the
factual branch receives FlowMatch/absolute loss, and the synthetic branch
receives elastic response and conservation supervision without a fake RGB
target.

The shared-anchor objective measures conditional denoising sensitivity at a
nonzero diffusion sigma; it is not claimed to be a full counterfactual rollout.
Scientific acceptance therefore evaluates complete inference trajectories with
common initial diffusion noise and each axis's declared first-frame policy;
hidden axes also share the first frame. A one-step sensitivity gain without a
corresponding five-seed full-rollout response cannot pass.

The existing SI-aware `QuantityEncoder` is reused. Its input projection consumes
the signed overlay quantities for velocity axes. `mu_k` is not injected because
it is constant and unidentifiable as a conditioning axis in v12.

FlowMatch retains its configured full timestep distribution. Every physical
stage additionally requires explicit `physical_sigma_min` and
`physical_sigma_max` bounds for response-bearing batches. Those numeric bounds
are selected on the 16-observer-frame response validation run, sealed before
later stages, and recorded with every checkpoint. A missing or invalid interval
fails configuration rather than decoding unrestricted, unusable `z0_hat`
estimates.

## 10. Loss definition

Let `O(D(z0_hat))` denote the generated RGB observer and let masks `M` denote
teacher-derived valid state/time components. The total training objective is

```text
L = L_FM
  + lambda_abs      * L_abs
  + lambda_sign     * L_sign
  + lambda_mag      * L_mag
  + lambda_zero     * L_zero
  + lambda_analytic * L_analytic
  + lambda_valid    * L_validity.
```

The components are:

- `L_FM`: the standard weighted WAN FlowMatch velocity loss on branches with a
  real video target;
- `L_abs`: uncertainty-normalized Huber error for each real branch's absolute
  state, preventing two wrong trajectories from having a correct difference;
- `L_sign`: a soft-margin loss on `sign(R*) * R_hat` for components satisfying
  `abs(R*) > 2 * sigma_R`;
- `L_mag`: uncertainty-normalized Huber error between `R_hat` and `R*`;
- `L_zero`: response magnitude for components statistically indistinguishable
  from zero;
- `L_analytic`: weak ideal-mechanics residuals, strong two-ball elastic
  response/conservation constraints, and strong three-ball global
  momentum/kinetic-energy conservation without a response term;
- `L_validity`: disappearance, identity loss, non-contact, illegal penetration,
  invalid topology, and required-event failure.

For each response component define

```text
s_R = max(2 * sigma_R, train_uncertainty_floor)
e_R = (R_hat - R*) / s_R
```

The magnitude term is unit-delta Huber loss on `e_R`. For significant targets,
the sign term is `softplus(1 - sign(R*) * R_hat / s_R)`. For non-significant
targets, the zero term is unit-delta Huber loss on `R_hat / s_R`. Absolute-state
terms use the same construction with their state uncertainty. Training validity
uses differentiable soft visibility, identity-channel, contact, and overlap
proxies; the independent evaluator applies the corresponding hard event gates.

All denominators use a train-derived uncertainty floor to avoid exploding loss
when replicate variance is spuriously small. Loss weights are required,
versioned run inputs. They ramp from zero during the curriculum and are selected
using response validation only. Per-component gradient norms on LoRA parameters
are logged. The stage policy requires numeric
`minimum_physical_gradient_norm`, `minimum_physical_to_fm_ratio`, and
`maximum_physical_to_fm_ratio`; the median over each 100 synchronized updates
must remain inside those bounds. Non-finite ratios fail immediately. Test
metrics never tune weights.

## 11. Parameter strategy and training curriculum

The main model trains WAN DiT LoRA plus `QuantityEncoder`. WAN base weights,
UMT5, VAE, and SAM2 remain frozen. The existing LoRA target family
`q,k,v,o,ffn.0,ffn.2` is retained initially. The final LoRA may be merged into
WAN weights for deployment, so inference has no adapter-management or SAM2
requirement.

Full 5B fine-tuning is not the first-line experiment: 799 videos are
insufficient to justify the overfitting and catastrophic-forgetting risk. After
the LoRA model passes all response and visual gates, a separate ablation may
unfreeze selected middle/late DiT blocks. That ablation receives a distinct run
identity and may not overwrite the LoRA result.

Training progresses through sealed stages. WAN continues to model the complete
121-frame clip in every training stage; 16 and 32 below are the ordered,
event-aware frame counts observed by the physical RGB/SAM2 path, not shortened
WAN outputs:

1. **Teacher/cache preflight:** compile groups and splits, build real-video
   states, validate coordinates, and reject invalid pairs.
2. **FlowMatch warm-up:** ordinary 121-frame SFT with response losses disabled.
3. **16-observer-frame physical stage:** enable low-weight direct-RGB absolute,
   direction, validity, and analytic losses; prove finite gradients end to end.
4. **32-observer-frame physical stage:** add full magnitude and event-aligned
   response, then ramp physical weights to their configured targets.
5. **121-observer-frame dense stage:** lower the optimizer learning rate and apply the
   observer across the complete sequence for final fine-tuning.

Moving to the next stage requires finite loss/gradients, observer coverage, a
configured minimum one-pair overfit gain, visual non-inferiority, and a passing
response-validation audit. Every numeric gate is required in the sealed stage
policy before the stage begins. Response test remains inaccessible during stage
progression. A failed stage is reported and stopped; it is not silently
relabeled as completion.

## 12. Distributed and memory design

The available machine has eight NVIDIA A100-SXM4 40GB GPUs. The current
eight-GPU WAN configuration is data parallel and therefore does not pool memory
automatically. A naive two-branch, 121-frame differentiable VAE+SAM2 graph on a
single rank is not the accepted implementation.

The paired trainer assigns intervention branches to cooperating ranks, shares
only the required low-dimensional state/response tensors, and uses an
autograd-correct distributed exchange. Frozen VAE and SAM2 operations use
activation checkpointing. SAM2 tracking is processed in temporal blocks with
checkpointed or explicitly truncated memory-state backpropagation according to
the sealed stage configuration; truncation may detach only the SAM2 memory at
declared block boundaries, never the current-frame RGB-to-state gradient. State
summaries are accumulated rather than retaining every full-resolution mask.
WAN's temporally coupled VAE must decode the full sequence or use an
overlap/caching scheme proven numerically equivalent to full decoding. Decoding
selected frames independently without the required temporal receptive field is
forbidden.

Before each temporal stage, a representative pair runs a memory and gradient
preflight. The audit records peak allocated/reserved CUDA memory, wall time,
observer frames, checkpointing mode, and whether both branch gradients reach
LoRA. If the 121-frame configuration cannot fit, the formal dense stage is
blocked. It must not silently fall back to 32 frames.

## 13. Evaluation and acceptance

Primary response evaluation uses one sealed set of exactly five generation
seeds per condition. Additional seeds, if run, are supplemental and are not
silently pooled into the primary result. The two members of a response pair
share a seed. Results are aggregated with a
group-level bootstrap 95% confidence interval so correlated pairs do not count
as independent experiments.

Every scene and response axis reports:

- `sign_accuracy`, with a near-zero prediction counted as wrong for a
  confidently non-zero target;
- `magnitude_nmae`, normalized by target scale and experimental uncertainty;
- `zero_response_leakage` for invariants;
- `absolute_state_error`;
- `validity_rate` and observer coverage;
- existing visual and object-integrity metrics.

Empirical and analytic residuals are reported separately. Results are also
split by group holdout versus value holdout and by interpolation versus
estimable extrapolation. Scene/axis macro averages accompany, but never replace,
the individual results.

The mandatory comparison arms are:

1. pretrained WAN2.2;
2. FlowMatch-only LoRA;
3. FlowMatch plus `QuantityEncoder`, without response loss;
4. the complete direct-RGB physical-response model;
5. analytic-only, empirical-only, and latent-head auxiliary ablations.

The complete model passes only if:

- both sign accuracy and magnitude error improve over arm 3 with group-level
  paired-bootstrap 95% confidence intervals lying wholly on the favorable side
  of zero for the adjacent-pair macro result;
- at least four of the five scenes have favorable point estimates on both
  primary response metrics, and no scene has a significant adverse 95%
  confidence interval;
- zero-response leakage, absolute state, validity, entity integrity, and visual
  quality satisfy frozen non-inferiority margins;
- two-ball collision satisfies response, momentum, and kinetic-energy gates
  together.

For every secondary metric, the non-inferiority margin is frozen before test as
the larger of five percent of that metric's documented valid range and one
arm-3 response-validation bootstrap standard error. For an unbounded metric,
five percent of the absolute arm-3 validation mean replaces five percent of the
range. The experiment report records the numeric margins and cannot revise them
after seeing test results.

## 14. Fail-closed behavior

- Missing or ambiguous direction, coordinate calibration, object mapping, or
  response-axis identity excludes physical supervision and records a reason.
- Multiple changed factors, split overlap, or malformed group membership fails
  response-manifest compilation.
- Low-confidence real tracking removes only the unreliable target component;
  the case may still contribute FlowMatch.
- Low-confidence generated tracking incurs validity loss and never masks itself.
- A required event that never occurs does not produce fabricated post-event
  state; it incurs event/validity failure.
- NaN/Inf in any loss or LoRA/QuantityEncoder gradient fails every distributed
  rank before optimizer update.
- Cache/config/checkpoint digest mismatch fails before training or evaluation.
- A memory preflight failure blocks the requested temporal stage rather than
  changing scientific scope.

## 15. Verification strategy

### Unit tests

- SI conversion, signed velocity projection, pair ordering, and nuisance
  signature stability;
- exact two-ball formulas, momentum/energy residuals, incline formula, and
  zero-response definitions;
- phase/event alignment and every scene coordinate transform;
- replicate uncertainty, sign-mask threshold, normalized magnitude loss, and
  zero-response loss;
- group/value split determinism and leakage rejection.

### Observer tests

- synthetic masks/videos with known line, parabola, circle, pendulum, and
  collision trajectories;
- soft-mask state extraction compared with hard-mask reference values;
- generated disappearance, identity swap, penetration, and missing-event cases;
- fixed coordinate frames proven independent of generated pixels.

### Gradient tests

- a tiny differentiable RGB path proves gradient flow from response loss through
  state extractor, SAM2, VAE, and `z0_hat` into LoRA;
- SAM2, VAE, UMT5, and WAN base parameters have no gradients;
- changing only the structured response parameter changes the condition branch;
- shuffled or detached conditions fail the sensitivity test;
- distributed two-branch gradient cosine similarity to a single-process FP32
  reference is at least `0.999`, with relative gradient-norm error at most one
  percent.

### Integration and acceptance tests

- one eligible pair overfits in the expected direction and magnitude without
  losing the object;
- 16-, 32-, and 121-frame stage preflights emit finite, complete audits;
- cache manifests are byte-identical for unchanged inputs; canonical state
  arrays are exact under deterministic kernels and otherwise agree within the
  observer policy's pre-test numeric tolerance;
- every planned held-out pair receives exactly five common-random-number seeds;
- the independent saved-RGB evaluator's state difference from the
  differentiable observer does not exceed the frozen 95th percentile measured
  on response validation.

## 16. Module boundaries

Implementation remains in this repository and does not patch SAM2 or
DiffSynth vendor sources:

```text
src/physbench/physical_response/
├── contracts.py
├── pairing.py
├── coordinates.py
├── teacher_cache.py
├── differentiable_sam2.py
├── flow_core.py
├── losses.py
├── metrics.py
├── sampling.py
└── state_extractors/
    ├── pendulum.py
    ├── collision.py
    ├── inclined_plane.py
    ├── projectile.py
    └── circular.py
```

An independent `WanPhysicalResponseTrainingModule` reuses the existing WAN
pipeline and QuantityEncoder while replacing the scalar-only flow-loss call
with `flow_core.py`. A dedicated baseline bundle, trainer entry point, response
policy, cache builder, evaluator, and tests depend on these focused modules.

The runtime interpreter is
`/mnt/nvme1/NicoCache/envs/physics_wan/bin/python`. Because that environment
inherits system site packages and SAM2 is installed editable, every run records
the resolved interpreter, package file locations, package versions, SAM2 Git
commit, `pip freeze`, CUDA/PyTorch identity, and model/checkpoint hashes.
The WAN base is resolved from
`/root/Steven/wan22_pendulum_pipeline/models/Wan-AI/Wan2.2-TI2V-5B`; its
complete resolved-file inventory and digests are sealed before training.

## 17. Explicit non-goals

This design does not:

- create or mutate a Dataset release;
- train a friction-coefficient response;
- introduce or vary a restitution coefficient;
- claim a three-ball finite-difference response model;
- train a push-bottle response;
- train WAN2.2 from scratch or start with full 5B fine-tuning;
- call observational matching a randomized physical intervention;
- use a latent-only response head as the primary scientific result;
- add SAM2 to inference or deployment.
