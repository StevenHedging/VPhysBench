# Collision V14 official-test audit — 2026-08-18

Scope: all 20 `collision_1d` jobs selected by
`six_scene_train_six_scene_eval_v1`. Every sampled source frame was reviewed,
with native-resolution inspection around every contact and terminal boundary.
Nine defective observations were rebuilt with SAM2.1 Hiera Large plus human
anchor/contact corrections; eleven already-correct observations were retained.
All final observations received a second full-video identity, mask-alignment,
trajectory, and lifecycle review.

| Case ID | Frozen QC | Human visual finding | Disposition |
|---|---|---|---|
| `collision_r2_glass_marble_glass_marble_glass_marble_v05716` | old false occlusions and contact collapse | Three transparent-ball identities remain stable; object_3 is explicitly out-of-frame after reviewed boundary exit. | repaired and approved |
| `collision_r2_large_steel_large_steel_large_steel_v02229` | partial boundary ball marked occluded/unresolved | All three tracks were rebuilt; object_3 stays visible through the last reviewed sliver and becomes out-of-frame only afterward. | repaired and approved |
| `collision_r2_large_steel_medium_steel_small_steel_v02358` | warning-only QC | Three identities and pre/contact/post ordering remain consistent. | retained and approved |
| `collision_r2_medium_steel_medium_steel_medium_steel_v01510` | warning-only QC | All three masks and trajectories remain aligned throughout the sampled sequence. | retained and approved |
| `collision_r2_small_steel_glass_marble_glass_marble_v02935` | avoidable gaps and false occlusions | Tracks were rebuilt; both exiting glass/steel boundary events now use explicit out-of-frame states. | repaired and approved |
| `collision_r2_small_steel_medium_steel_large_steel_v06332` | object_1 tracker failure | Human-corrected object_1 anchor and all three tracks are stable; terminal partial object_3 remains correctly visible. | repaired and approved |
| `collision_r2_small_steel_small_steel_small_steel_v04865` | false occlusions and post-contact identity collapse | Post-contact object_1/object_2 received an extra human correction; object_3 exit is explicit and no later false reappearance is retained. | repaired and approved |
| `collision_supp_20260729_img_0906_two_ball_single_incident` | no material defect | Both masks follow the physical balls through contact and separation. | retained and approved |
| `collision_supp_20260729_img_0942_two_ball_single_incident` | no material defect | Both identities and masks are consistent through the full video. | retained and approved |
| `collision_supp_20260729_img_0944_two_ball_single_incident` | no material defect | Both identities and masks are consistent through the full video. | retained and approved |
| `collision_supp_20260729_img_1036_two_ball_single_incident` | warning-only QC | Both masks remain centered; the warning does not indicate an identity failure. | retained and approved |
| `collision_supp_20260729_img_1057_two_ball_single_incident` | warning-only QC | Both masks remain centered across every sampled frame. | retained and approved |
| `collision_supp_20260729_img_1073_two_ball_single_incident` | anchors on ruler highlights | Both physical-ball anchors and tracks were rebuilt and remain stable through contact. | repaired and approved |
| `collision_supp_20260729_img_1085_two_ball_single_incident` | no material defect | Both masks remain centered on the physical balls. | retained and approved |
| `collision_supp_20260729_img_1086_two_ball_single_incident` | warning-only QC | Dense review confirms both tracks remain on the physical balls. | retained and approved |
| `collision_supp_20260729_img_1097_two_ball_single_incident` | no material defect | Both masks and trajectories remain aligned. | retained and approved |
| `collision_supp_20260729_img_1133_two_ball_opposed_incident` | no material defect | Opposed identities remain stable through contact without a swap. | retained and approved |
| `collision_supp_20260729_img_1173_two_ball_opposed_incident` | unresolved terminal samples | Object_2 is now explicitly out-of-frame after the last partial boundary mask. | repaired and approved |
| `collision_supp_20260729_img_1190_two_ball_single_incident` | anchor on ruler and long false occlusion | Both physical-ball anchors and 121-sample tracks were rebuilt with stable identity. | repaired and approved |
| `collision_supp_20260729_img_1234_three_ball_single_incident` | localized gaps and unresolved terminal exit | All three identities were rebuilt; object_3 terminal boundary exit is explicit. | repaired and approved |

## Acceptance rule

A repaired case is accepted only when regenerated overlays and trajectory
visualizations show correct identity at frame zero, immediately before and
after every contact, and at the terminal boundary. `OCCLUDED` is reserved for
a genuinely non-separable physical ball; `OUT_OF_FRAME` begins only after the
last reviewed partial boundary mask. No `UNRESOLVED` sample remains in the
nine rebuilt observations.
