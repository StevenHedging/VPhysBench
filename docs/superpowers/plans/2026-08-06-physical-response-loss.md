# Physical Response Loss Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Plan-authoring scope:** Until implementation starts, stage and commit only `docs/superpowers/plans/2026-08-06-physical-response-loss.md`. Every per-Task Step 5 command is deferred to implementation, must run in a clean isolated worktree, and must abort unless the index is empty before its shown `git add`.

**Goal:** Build a sealed, direct-RGB physical-response training and evaluation subsystem for WAN2.2-TI2V-5B that learns the real system's signed and magnitude response while leaving Dataset release `12.0.0` byte-for-byte unchanged.

**Architecture:** A run-local compiler projects signed SI quantities, freezes matched intervention graphs and response holdouts, and builds a content-addressed real-video teacher cache. Training replaces the scalar vendor FlowMatch call with a repository-owned flow core, decodes paired 121-frame WAN predictions through the frozen VAE and differentiable frozen SAM2 observer, exchanges only physical-state tensors between cooperating ranks, and applies separate absolute/sign/magnitude/zero/analytic/validity losses. A separate saved-RGB observer and five-common-seed evaluator make the scientific acceptance decision without trusting training-time masks or latent heads.

**Tech Stack:** Python 3.10+, PyTorch 2.6 distributed autograd, WAN2.2-TI2V-5B, pinned DiffSynth Studio, SAM2.1 Hiera Tiny, NumPy, OpenCV/FFmpeg, JSON/JSONL, `unittest`, eight NVIDIA A100-SXM4 40GB GPUs.

## Global Constraints

- Treat repository-relative `datasets/releases/12.0.0` and every referenced Dataset asset as immutable; create no Dataset release and write nothing beneath `datasets/`.
- Use only these response axes: pendulum string length and initial-angle magnitude; two-ball collision mass and signed initial velocity; incline angle; projectile signed horizontal velocity; circular orbit radius.
- Keep `push_bottle` entirely outside physical-response v1.
- Keep `mu_k = 0.463` exclusively inside the inclined-plane analytic regularizer. It is neither a response axis nor a model token.
- Treat collision as perfectly elastic with `e = 1`; introduce no restitution input. Three-ball clips receive absolute state, topology, event, momentum, and kinetic-energy constraints but no finite-difference response target.
- Preserve the scientific order `q_b > q_a`, freeze complete adjacent-value graphs before splitting, and never reconnect across a held-out value.
- Remove every response-validation or response-test video from ordinary FlowMatch training for that run; never expose response-test teachers during training, validation, stage selection, or checkpoint selection.
- WAN always models all 121 frames. Only the differentiable physical observer follows the sealed `16 -> 32 -> 121` frame curriculum.
- The primary gradient path is `v_pred -> z0_hat -> frozen full-temporal WAN VAE decode -> generated RGB -> frozen differentiable SAM2 -> physical state -> loss`; a latent head can appear only as an ablation.
- Freeze WAN base weights, UMT5, VAE, and SAM2. Train the WAN DiT LoRA target family `q,k,v,o,ffn.0,ffn.2` plus the existing SI-aware `QuantityEncoder`.
- Do not patch the locally resolved DiffSynth Studio or SAM2 checkout.
- Pin DiffSynth commit `fb337fbb90945ff829de69dbd44ded618f73e889`, SAM2 commit `2b90b9f5ceec907a1c18123530e92e794ad901a4`, SAM2 checkpoint SHA-256 `7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69`, and the complete resolved WAN base inventory in every run identity.
- Use the supplied `physics_wan` environment for this deployment and seal the resolved interpreter, package locations and versions, `pip freeze`, PyTorch/CUDA identity, code commits, and every model/config/checkpoint digest. Tracked source, scripts, portable manifests, and every `*.example.json` must contain no machine-specific absolute path below `/root/` or `/mnt/`; external locations are injected through the exact ignored `training_local.json`, `generation_local.json`, orchestration `local.json`, and `baseline.local.json` files and verified before use.
- Fail closed on ambiguous signs, changed nuisance factors, split overlap, missing calibrations, identity ambiguity, absent required events, digest mismatch, non-finite losses/gradients, or failed stage memory/gradient gates.
- Primary evaluation uses exactly five sealed common-random-number seeds per ordered pair and reports sign, magnitude, zero response, absolute state, validity/coverage, empirical residuals, analytic residuals, visual quality, and entity integrity separately.
- Final inference and deployment must not require SAM2; SAM2 is a training observer and an offline evaluator only.

## File and Ownership Map

| Path | Single responsibility |
| --- | --- |
| `src/physbench/physical_response/contracts.py` | Frozen JSON-domain contracts, strict parsing, canonical serialization, and digests. |
| `src/physbench/physical_response/integrity.py` | Byte inventories, symlink-safe Dataset tree hashing, exclusive cache finalization, and environment/model identities. |
| `src/physbench/physical_response/pairing.py` | SI projection, signed direction evidence, nuisance signatures, empirical groups, frozen edges, and elastic mass counterfactuals. |
| `src/physbench/physical_response/splits.py` | Provenance components, group/value holdouts, leakage audit, and resolved overlay serialization. |
| `src/physbench/physical_response/differentiable_sam2.py` | Tensor-native frozen SAM2 construction, soft tracking, temporal block checkpointing, and mask-moment reduction. |
| `src/physbench/physical_response/coordinates.py` | Frozen calibration records, tensor coordinate transforms, physical time, and differentiable common state helpers. |
| `src/physbench/physical_response/state.py` | Fixed-shape observation/state/target tensor contracts shared by extractors, losses, and distributed exchange. |
| `src/physbench/physical_response/state_extractors/*.py` | One scene family's differentiable states, event weights, summaries, and validity proxies per file. |
| `src/physbench/physical_response/teacher_cache.py` | Offline real-video observation, repeat aggregation, immutable cache records, and partition-scoped readers. |
| `src/physbench/physical_response/flow_core.py` | Injected-noise/timestep FlowMatch parity and clean-latent reconstruction without vendor changes. |
| `src/physbench/physical_response/vae_decode.py` | Full-temporal differentiable WAN VAE decoding and its checkpointed-equivalence gate. |
| `src/physbench/physical_response/mechanics.py` | Exact elastic-collision formulas and weak pendulum/incline/projectile/circular mechanics. |
| `src/physbench/physical_response/losses.py` | Absolute, sign, magnitude, zero, analytic, validity, weighting, eligibility counts, and gradient-ratio inputs. |
| `src/physbench/physical_response/batches.py` | Factual/response batch materialization from sealed train-only views. |
| `src/physbench/physical_response/sampling.py` | Logical update schedule, observer-frame selection, paired generation jobs, and common-noise identity. |
| `src/physbench/physical_response/distributed.py` | Fixed two-rank process groups, pair sampler state, autograd state exchange, and global gradient norms. |
| `src/physbench/physical_response/training.py` | `WanPhysicalResponseTrainingModule`, loss assembly, freeze audit, and synchronized update semantics. |
| `src/physbench/physical_response/stages.py` | Sealed stage policies, progression gates, preflight records, and test-access denial. |
| `src/physbench/physical_response/generation_runtime.py` | External generation-runtime resolution, verified checkpoint loading, and allocation boundary audits. |
| `src/physbench/physical_response/evaluation_inputs.py` | Matrix seal loading, common target construction, arm evaluation-input sealing, and fail-before-read verification. |
| `src/physbench/physical_response/saved_rgb_observer.py` | Independent hard-mask observation of encoded saved RGB and observer-agreement evidence. |
| `src/physbench/physical_response/metrics.py` | Independent saved-RGB metrics, group bootstrap, non-inferiority margins, and acceptance comparison. |
| `src/physbench/physical_response/latent_head.py` | Auxiliary latent-state ablation with explicit RGB-time observer indices. |
| `src/physbench/physical_response/run.py` | External-runtime initialization, seven-arm lifecycle, matrix sealing, status transitions, and finalization. |
| `src/physbench/baselines/wan22_physical_response.py` | Managed adapter, response artifact bindings, dependency identity, checkpoint manifest, and paired inference launch. |
| `scripts/*physical_response*` | Thin compiler, cache, training, sealing, preflight, generation, and evaluation entry points. |
| `baselines/wan22_physical_response/` | Discoverable schema-v5 Baseline bundle and operator runbook. |

---

### Task 1: Freeze policy, manifest, and byte-integrity contracts

**Files:**
- Modify: `.gitignore`
- Create: `src/physbench/physical_response/__init__.py`
- Create: `src/physbench/physical_response/contracts.py`
- Create: `src/physbench/physical_response/integrity.py`
- Create: `configs/physical_response/policies/physical_response_v1.json`
- Create: `schemas/physical_response_policy.schema.json`
- Create: `schemas/physical_response_manifest.schema.json`
- Create: `schemas/physical_response_teacher_cache.schema.json`
- Create: `tests/physical_response/__init__.py`
- Create: `tests/physical_response/_fixtures.py`
- Create: `tests/physical_response/test_contracts.py`

**Interfaces:**
- Consumes: `physbench.io.load_json(path: str | Path) -> Any`, `physbench.io.canonical_sha256(value: Any) -> str`, and `physbench.io.sha256_file(path: Path) -> str` from `src/physbench/io.py:15-71`.
- Produces: `load_response_policy(path: Path) -> ResponsePolicy`; `CandidateEdge.digest: str`; `ResponseEdge.digest: str`; `ResponseManifest.from_document(value: Mapping[str, Any]) -> ResponseManifest`; `ResponseManifest.to_document(*, include_digest: bool = True) -> dict[str, Any]`; `ResponseManifest.group_for(group_id: str) -> InterventionGroup`; `ResponseManifest.edge_for(edge_id: str) -> ResponseEdge`; `ResponseManifest.normalization_for(axis_id: str) -> ResponseNormalization`; `ResponseManifest.digest: str`; `inventory_tree(root: Path, *, symlink_policy: Literal["hash_link_text_no_follow"] = "hash_link_text_no_follow") -> TreeInventory`; `assert_same_inventory(before: TreeInventory, after: TreeInventory) -> None`; `exclusive_finalize_directory(staging: Path, destination: Path, expected_manifest_digest: str) -> Path`.
- Produces frozen dataclasses `ResponsePolicy`, `ResponseAxisSpec`, `SignedProjection`, `ResponseLevel`, `InterventionGroup`, `CandidateEdge`, `ResponseEdge`, `ResponseNormalization`, `ResponseManifest`, `OverlapAudit`, `InventoryEntry`, and `TreeInventory`; shared aliases `EdgeKind = Literal["adjacent", "long_range_audit", "analytic_counterfactual"]`, `ResponseTargetKind = Literal["empirical_pair", "analytic_counterfactual"]`, and `InterpolationKind = Literal["adjacent", "interpolation", "extrapolation", "not_estimable"]`. A `CandidateEdge` has no partition; Task 3 converts it exactly once into a partitioned `ResponseEdge`. Manifest SI scalars are canonical decimal strings; partitions are exactly `response_train`, `response_validation`, or `response_test`.

- [ ] **Step 1: Write strict contract and mutation-detection tests**

```python
class ResponseContractTests(unittest.TestCase):
    def test_manifest_digest_binds_ordered_edges_and_tracked_entities(self) -> None:
        document = minimal_response_manifest_document()
        manifest = ResponseManifest.from_document(document)
        changed = copy.deepcopy(manifest.to_document())
        changed.pop("manifest_digest")
        changed["edges"][0]["q_b_si"] = "0.75"
        self.assertNotEqual(
            manifest.digest,
            ResponseManifest.from_document(changed).digest,
        )

    def test_manifest_rejects_noncanonical_edge_order(self) -> None:
        document = minimal_response_manifest_document()
        document["edges"][0]["q_a_si"] = "1"
        document["edges"][0]["q_b_si"] = "0.5"
        with self.assertRaisesRegex(ValueError, "q_b_si must be greater"):
            ResponseManifest.from_document(document)

    def test_response_edge_digest_and_manifest_lookups_are_total(self) -> None:
        manifest = ResponseManifest.from_document(minimal_response_manifest_document())
        edge = manifest.edges[0]
        changed = dataclasses.replace(
            edge,
            partition=(
                "response_validation"
                if edge.partition == "response_test" else "response_test"
            ),
        )
        self.assertNotEqual(edge.digest, changed.digest)
        self.assertEqual(edge, manifest.edge_for(edge.edge_id))
        self.assertEqual(
            manifest.group_for(edge.group_id).axis_id,
            manifest.normalization_for(manifest.group_for(edge.group_id).axis_id).axis_id,
        )
        with self.assertRaisesRegex(KeyError, "unknown edge_id"):
            manifest.edge_for("missing")

    def test_tree_inventory_hashes_symlink_text_without_following_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "payload.bin").write_bytes(b"one")
            (root / "alias").symlink_to("payload.bin")
            before = inventory_tree(root)
            (root / "payload.bin").write_bytes(b"two")
            after = inventory_tree(root)
        self.assertNotEqual(before.digest, after.digest)
        self.assertEqual("symlink", before.entries[0].kind)

    def test_axis_registry_seals_visible_and_hidden_first_frame_policy(self) -> None:
        policy = load_response_policy(fixture_policy_path())
        actual = {axis.axis_id: axis.first_frame_policy for axis in policy.axes}
        self.assertEqual({
            "pendulum.string_length": "corresponding_real",
            "pendulum.initial_angle_magnitude": "corresponding_real",
            "collision_1d.mass": "shared_anchor",
            "collision_1d.initial_velocity_signed": "shared_anchor",
            "inclined_plane_slide.angle": "corresponding_real",
            "parabolic_motion.initial_horizontal_velocity": "shared_anchor",
            "uniform_circular_motion.orbit_radius": "corresponding_real",
        }, actual)

    def test_axis_registry_seals_exact_teacher_roles(self) -> None:
        policy = load_response_policy(fixture_policy_path())
        actual = {axis.axis_id: set(axis.teacher_roles) for axis in policy.axes}
        weak_empirical = {"empirical_primary", "analytic_weak", "absolute_empirical"}
        self.assertEqual({
            "pendulum.string_length": weak_empirical,
            "pendulum.initial_angle_magnitude": weak_empirical,
            "collision_1d.mass": {"analytic_strong", "absolute_empirical"},
            "collision_1d.initial_velocity_signed": {
                "empirical_primary", "analytic_strong", "absolute_empirical",
            },
            "inclined_plane_slide.angle": weak_empirical,
            "parabolic_motion.initial_horizontal_velocity": weak_empirical,
            "uniform_circular_motion.orbit_radius": weak_empirical,
        }, actual)

    def test_policy_requires_coordinate_pairing_and_nuisance_versions(self) -> None:
        for missing in (
            "coordinate_versions", "pair_selection_version",
            "nuisance_matching_version", "analytic_disagreement_thresholds",
        ):
            document = fixture_policy_document()
            del document[missing]
            with self.subTest(missing=missing), self.assertRaisesRegex(ValueError, missing):
                ResponsePolicy.from_document(document)
```

- [ ] **Step 2: Run the focused test and observe the missing-package failure**

Run:

```bash
PYTHONPATH=src:tests:. /mnt/nvme1/NicoCache/envs/physics_wan/bin/python \
  -m unittest tests.physical_response.test_contracts -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'physbench.physical_response'`.

- [ ] **Step 3: Implement frozen contracts and a no-follow byte inventory**

```python
Partition = Literal[
    "response_train", "response_validation", "response_test"
]
EdgeKind = Literal[
    "adjacent", "long_range_audit", "analytic_counterfactual"
]
ResponseTargetKind = Literal[
    "empirical_pair", "analytic_counterfactual"
]
InterpolationKind = Literal[
    "adjacent", "interpolation", "extrapolation", "not_estimable"
]
TeacherRole = Literal[
    "empirical_primary", "analytic_weak", "analytic_strong",
    "absolute_empirical",
]

@dataclass(frozen=True, slots=True)
class CandidateEdge:
    edge_id: str
    group_id: str
    level_a_id: str
    level_b_id: str
    q_a_si: str
    q_b_si: str
    edge_kind: EdgeKind
    target_kind: ResponseTargetKind
    frame_kind: Literal["corresponding_real", "shared_anchor"]
    endpoint_a_case_ids: tuple[str, ...]
    endpoint_b_case_ids: tuple[str, ...]
    factual_case_id: str | None
    factual_branch: Literal["a", "b"] | None
    active_quantity: ActiveQuantityBinding
    collision_topology: CollisionTopologyBinding | None
    coordinate_frame_id: str

    def __post_init__(self) -> None:
        if Decimal(self.q_b_si) <= Decimal(self.q_a_si):
            raise ValueError("q_b_si must be greater than q_a_si")

    @property
    def digest(self) -> str:
        return canonical_sha256(canonical_dataclass_document(self))

@dataclass(frozen=True, slots=True)
class ResponseEdge(CandidateEdge):
    partition: Partition
    interpolation_kind: InterpolationKind

@dataclass(frozen=True, slots=True)
class ResponseManifest:
    schema_version: str
    experiment_id: str
    policy_digest: str
    split_policy_digest: str
    dataset_id: str
    dataset_release: str
    dataset_metadata_digest: str
    dataset_tree_digest: str
    input_inventory_digest: str
    condition_index_digest: str
    provenance_digests: tuple[tuple[str, str], ...]
    normalizations: tuple[ResponseNormalization, ...]
    projections: tuple[SignedProjection, ...]
    groups: tuple[InterventionGroup, ...]
    edges: tuple[ResponseEdge, ...]
    flowmatch_train_case_ids: tuple[str, ...]
    exclusions: tuple[tuple[str, str], ...]

    @property
    def digest(self) -> str:
        return canonical_sha256(self.to_document(include_digest=False))

    def group_for(self, group_id: str) -> InterventionGroup:
        return exactly_one_by_id(self.groups, "group_id", group_id)

    def edge_for(self, edge_id: str) -> ResponseEdge:
        return exactly_one_by_id(self.edges, "edge_id", edge_id)

    def normalization_for(self, axis_id: str) -> ResponseNormalization:
        return exactly_one_by_id(self.normalizations, "axis_id", axis_id)

def inventory_tree(
    root: Path,
    *,
    symlink_policy: Literal["hash_link_text_no_follow"] =
        "hash_link_text_no_follow",
) -> TreeInventory:
    entries: list[InventoryEntry] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            target = os.readlink(path)
            entries.append(InventoryEntry(
                relative, "symlink", len(target.encode()),
                hashlib.sha256(target.encode()).hexdigest(), target,
            ))
        elif path.is_file():
            entries.append(InventoryEntry(
                relative, "file", path.stat().st_size, sha256_file(path), None,
            ))
        elif not path.is_dir():
            raise ValueError(f"unsupported Dataset tree entry: {relative}")
    frozen = tuple(entries)
    return TreeInventory(root.resolve(), frozen, canonical_sha256([
        entry.to_document() for entry in frozen
    ]))

def exclusive_finalize_directory(
    staging: Path,
    destination: Path,
    expected_manifest_digest: str,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    claim = destination.with_name(destination.name + ".finalizing")
    try:
        claim.mkdir()
    except FileExistsError as error:
        if destination.is_dir() and verified_manifest_digest(destination) == expected_manifest_digest:
            return destination
        raise CacheBuildInProgress(str(destination)) from error
    try:
        if destination.exists():
            if verified_manifest_digest(destination) == expected_manifest_digest:
                return destination
            raise FileExistsError(f"conflicting immutable cache entry: {destination}")
        os.rename(staging, destination)
        if verified_manifest_digest(destination) != expected_manifest_digest:
            raise ValueError("finalized cache manifest digest mismatch")
        return destination
    finally:
        claim.rmdir()
```

The checked-in policy must enumerate the seven axis families, their parameter-path templates, SI units, signed-projection versions, first-frame policies, teacher roles, response components, `mu_k=0.463`, `g=9.80665`, the Dataset identity, pinned SAM2 identity, and the no-vendor-patch rule. Its strict loader rejects unknown fields and rejects `mu_k` under conditioning or response axes. Manifest loading rejects duplicate group, edge, and normalization IDs before constructing the lookup methods. `.gitignore` adds exact entries for `configs/physical_response/runtime/training_local.json`, `configs/physical_response/runtime/generation_local.json`, `configs/physical_response/runtime/local.json`, and `baselines/wan22_physical_response/baseline.local.json`; no wildcard may hide the checked-in example files.

The remaining public constructors are fixed as follows:

- `ResponsePolicy(schema_version: str, dataset_id: str, dataset_release: str, axes: tuple[ResponseAxisSpec, ...], incline_mu_k: str, incline_mu_k_provenance_digest: str, gravity_m_s2: str, documented_repeat_paths: tuple[str, ...], nuisance_matching_version: str, pair_selection_version: str, analytic_disagreement_thresholds: tuple[tuple[str, str], ...], coordinate_versions: tuple[tuple[str, str], ...], teacher_versions: tuple[tuple[str, str], ...], observer_versions: tuple[tuple[str, str], ...], digest: str)`.
- `ResponseAxisSpec(axis_id: str, scene_id: str, parameter_path_template: str, canonical_si_unit: str, signed_projection_id: str, intervened_entity_role: str, response_component_paths: tuple[str, ...], first_frame_policy: Literal["corresponding_real", "shared_anchor"], teacher_roles: tuple[TeacherRole, ...])`.
- `SignedProjection(projection_id: str, case_id: str, axis_id: str, quantity_path: str, quantity_name: str, value_si: str, source_raw_literal: str, source_unit: str, prompt_value_span: tuple[int, int] | None, direction_source: str)`.
- `ActiveQuantityBinding(quantity_path: str, entity_id: str, state_object_index: int, canonical_si_unit: str, binding_digest: str)` identifies the only structured quantity an edge may intervene on; lookup by non-unique display name is forbidden.
- `CollisionTopologyBinding(ordered_entity_ids: tuple[str, str], tracked_channel_ids: tuple[str, str], mass_quantity_paths: tuple[str, str], incident_velocity_quantity_paths: tuple[str, str], intervened_entity_id: str, intervened_object_index: Literal[0, 1], coordinate_frame_id: str, digest: str)` fixes formula indices independently of left/right image position.
- `ResponseLevel(level_id: str, q_si: str, source_kind: Literal["real", "analytic_counterfactual"], member_case_ids: tuple[str, ...], partition: Partition | None)`. A real level requires at least one member. An analytic-counterfactual level has no member, is allowed only for two-ball collision mass, and inherits partition solely through its edge's factual case.
- `InterventionGroup(group_id: str, scene_id: str, axis_id: str, response_parameter_path: str, intervened_entity_id: str, entity_role: str, tracked_entity_mapping: tuple[tuple[str, str], ...], response_component_paths: tuple[str, ...], levels: tuple[ResponseLevel, ...], nuisance_signature: str, first_frame_policy: Literal["corresponding_real", "shared_anchor"], teacher_roles: tuple[TeacherRole, ...], coordinate_frame_id: str)`.
- `ResponseNormalization(axis_id: str, q_center_si: str, q_scale_si: str)`.
- `InventoryEntry(relative_path: str, kind: Literal["file", "symlink"], size_bytes: int, sha256: str, link_target: str | None)` and `TreeInventory(root: Path, entries: tuple[InventoryEntry, ...], digest: str)`.
- `OverlapAudit(manifest_digest: str, view_a_test_overlap_case_ids: tuple[str, ...], source_component_overlap_ids: tuple[str, ...], near_duplicate_overlap_ids: tuple[str, ...], passed: bool, failure_reasons: tuple[str, ...])`.

A parsed `ResponseManifest` requires every level partition to be non-null. Nuisance matching and pair selection are explicit versioned policy inputs rather than implicit code behavior. Every analytic disagreement threshold is a canonical positive Decimal string keyed by `axis_id/component`; absence is a policy error, never an implicit infinity. Prompt spans are byte offsets verified against `source_raw_literal`; they are nullable for projections that never rewrite prompt text.

- [ ] **Step 4: Run the contract tests and schema smoke check**

Run:

```bash
PYTHONPATH=src:tests:. /mnt/nvme1/NicoCache/envs/physics_wan/bin/python \
  -m unittest tests.physical_response.test_contracts -v
PYTHONPATH=src:. /mnt/nvme1/NicoCache/envs/physics_wan/bin/python -c \
  'from pathlib import Path; from physbench.physical_response.contracts import load_response_policy; p=load_response_policy(Path("configs/physical_response/policies/physical_response_v1.json")); assert p.incline_mu_k == "0.463" and len(p.axes) == 7'
```

Expected: all tests PASS and the smoke command exits zero.

- [ ] **Step 5: Commit the policy and integrity foundation**

```bash
git add -- .gitignore \
  src/physbench/physical_response/__init__.py \
  src/physbench/physical_response/contracts.py \
  src/physbench/physical_response/integrity.py \
  configs/physical_response/policies/physical_response_v1.json \
  schemas/physical_response_policy.schema.json \
  schemas/physical_response_manifest.schema.json \
  schemas/physical_response_teacher_cache.schema.json \
  tests/physical_response/__init__.py \
  tests/physical_response/_fixtures.py \
  tests/physical_response/test_contracts.py
git commit -m "feat: define physical response contracts"
```

### Task 2: Compile signed SI values, nuisance groups, and frozen local edges

**Files:**
- Create: `src/physbench/physical_response/pairing.py`
- Create: `tests/physical_response/test_pairing.py`
- Modify: `tests/physical_response/_fixtures.py`

**Interfaces:**
- Consumes: `iter_physics_quantities(case: Mapping[str, Any]) -> Iterator[tuple[str, str, Mapping[str, Any]]]` from `src/physbench/datasets/physics.py:89-163`; `ResponsePolicy`, `ResponseAxisSpec`, `SignedProjection`, `ResponseLevel`, `InterventionGroup`, and `CandidateEdge` from Task 1; frozen entity mappings materialized by `materialize_entity_manifest(case: Mapping[str, Any]) -> EntityManifest` at `src/physbench/evaluation/common/entities/manifest.py:1342-1405`.
- Produces: `resolve_quantity_path(case: Mapping[str, Any], path: str) -> Mapping[str, Any]`; `canonical_quantity_si(quantity: Mapping[str, Any], unit_scales: Mapping[str, str]) -> str`; `derive_direction_evidence(case: Mapping[str, Any], axis: ResponseAxisSpec, tracked_entity_mapping: Mapping[str, str]) -> DirectionEvidence`; `project_signed_quantity(case: Mapping[str, Any], axis: ResponseAxisSpec, evidence: DirectionEvidence, unit_scales: Mapping[str, str]) -> SignedProjection`; `nuisance_signature(case: Mapping[str, Any], *, response_parameter_path: str, documented_repeat_paths: tuple[str, ...]) -> str`; `compile_intervention_groups(cases: Sequence[Mapping[str, Any]], *, policy: ResponsePolicy, entity_mappings: Mapping[str, Mapping[str, str]], direction_evidence: Mapping[tuple[str, str], DirectionEvidence], unit_scales: Mapping[str, str]) -> GroupCompilation`; `freeze_adjacent_edges(groups: Sequence[InterventionGroup]) -> tuple[CandidateEdge, ...]`; `freeze_long_range_audit_edges(groups: Sequence[InterventionGroup], *, level_gaps: tuple[int, ...] = (2, -1)) -> tuple[CandidateEdge, ...]`; `compile_elastic_mass_counterfactuals(cases: Sequence[Mapping[str, Any]], *, axis: ResponseAxisSpec, eligible_case_ids: AbstractSet[str], released_mass_support_si: Sequence[str], entity_mappings: Mapping[str, Mapping[str, str]]) -> CounterfactualCompilation`; `merge_group_compilations(empirical: GroupCompilation, counterfactual: CounterfactualCompilation) -> GroupCompilation`.
- Invariant: quantities that are equal only after rounding remain distinct; exact canonical decimal SI values define levels and replicates. An ambiguous prompt/role/first-frame direction returns an exclusion, never a positive default.
- Data contracts: `DirectionEvidence(semantic_direction: Literal["left", "right", "stationary", "nonnegative"], sign: Literal[-1, 0, 1], source: str)`; `GroupCompilation(groups: tuple[InterventionGroup, ...], projections: tuple[SignedProjection, ...], exclusions: Mapping[str, str])`; and `CounterfactualCompilation(groups: tuple[InterventionGroup, ...], edges: tuple[CandidateEdge, ...], source_case_ids: tuple[str, ...], digest: str)`. Every counterfactual edge's `group_id`, `level_a_id`, and `level_b_id` must resolve inside its returned groups; exactly one level is real and exactly one is `analytic_counterfactual`.

- [ ] **Step 1: Write tests for exact SI identity, signed velocity, nuisance matching, adjacency, and collision scope**

```python
class PairingTests(unittest.TestCase):
    def test_projectile_leftward_velocity_is_negative_only_in_overlay(self) -> None:
        case = projectile_case(speed=0.25, prompt="launched to the left")
        original = copy.deepcopy(case["physics"])
        projection = project_signed_quantity(
            case,
            projectile_velocity_axis(),
            DirectionEvidence("left", -1, "prompt+frozen_x_axis"),
            {"m/s": "1"},
        )
        self.assertEqual("-0.25", projection.value_si)
        self.assertEqual(original, case["physics"])

    def test_grouping_rejects_a_second_changed_physics_path(self) -> None:
        a, b = incline_pair(angle_a=30, angle_b=35)
        b["physics"]["objects"]["object_1"]["mass"]["value"] = 0.2
        compiled = compile_intervention_groups(
            [a, b], policy=test_policy(), entity_mappings=test_entities(),
            direction_evidence={}, unit_scales=test_unit_scales(),
        )
        self.assertIn("multiple_changed_factors", compiled.exclusions.values())

    def test_edges_are_frozen_only_between_original_adjacent_levels(self) -> None:
        group = intervention_group_with_levels("0.1", "0.2", "0.4")
        edges = freeze_adjacent_edges([group])
        self.assertEqual(
            [("0.1", "0.2"), ("0.2", "0.4")],
            [(edge.q_a_si, edge.q_b_si) for edge in edges],
        )

    def test_long_range_edges_are_audit_only_and_never_replace_local_edges(self) -> None:
        group = intervention_group_with_levels("0.1", "0.2", "0.4", "0.8")
        audit = freeze_long_range_audit_edges([group], level_gaps=(2, -1))
        self.assertEqual({"long_range_audit"}, {edge.edge_kind for edge in audit})
        self.assertEqual(
            {("0.1", "0.4"), ("0.2", "0.8"), ("0.1", "0.8")},
            {(edge.q_a_si, edge.q_b_si) for edge in audit},
        )

    def test_three_ball_case_has_no_counterfactual_response_edge(self) -> None:
        compiled = compile_elastic_mass_counterfactuals(
            [collision_case(ball_count=3)], axis=collision_mass_axis(),
            eligible_case_ids={"collision_3"},
            released_mass_support_si=("0.014", "0.03313"),
            entity_mappings={"collision_3": collision_entities(3)},
        )
        self.assertEqual((), compiled.edges)
        self.assertEqual((), compiled.groups)

    def test_three_ball_case_never_enters_empirical_response_group(self) -> None:
        compiled = compile_intervention_groups(
            three_ball_velocity_candidates(), policy=test_policy(),
            entity_mappings=test_entities(), direction_evidence=test_directions(),
            unit_scales=test_unit_scales(),
        )
        self.assertFalse(any(group.scene_id == "collision_1d" for group in compiled.groups))
        self.assertIn("three_ball_response_out_of_scope", compiled.exclusions.values())

    def test_only_collision_mass_uses_analytic_counterfactual_target(self) -> None:
        empirical = freeze_adjacent_edges(all_empirical_fixture_groups())
        mass = compile_elastic_mass_counterfactuals(
            [collision_case(ball_count=2)], axis=collision_mass_axis(),
            eligible_case_ids={"collision_2"},
            released_mass_support_si=("0.014", "0.03313"),
            entity_mappings={"collision_2": collision_entities(2)},
        )
        self.assertEqual({"empirical_pair"}, {edge.target_kind for edge in empirical})
        self.assertEqual({"analytic_counterfactual"}, {edge.target_kind for edge in mass.edges})
        self.assertEqual({"shared_anchor"}, {edge.frame_kind for edge in mass.edges})
        levels = {level.level_id: level for group in mass.groups for level in group.levels}
        for edge in mass.edges:
            self.assertIn(edge.group_id, {group.group_id for group in mass.groups})
            self.assertIn(edge.level_a_id, levels)
            self.assertIn(edge.level_b_id, levels)
            self.assertEqual(
                {"real", "analytic_counterfactual"},
                {levels[edge.level_a_id].source_kind, levels[edge.level_b_id].source_kind},
            )
            self.assertIsNotNone(edge.collision_topology)
            self.assertIn(edge.factual_branch, {"a", "b"})

    def test_collision_formula_topology_does_not_follow_image_side_swaps(self) -> None:
        first = compile_mass_counterfactual_fixture(left_to_right=True)
        second = compile_mass_counterfactual_fixture(left_to_right=False)
        self.assertEqual(
            first.edges[0].collision_topology.intervened_object_index,
            second.edges[0].collision_topology.intervened_object_index,
        )
        self.assertEqual(
            first.edges[0].active_quantity.binding_digest,
            second.edges[0].active_quantity.binding_digest,
        )
```

- [ ] **Step 2: Run the pairing test and verify the missing-module failure**

Run:

```bash
PYTHONPATH=src:tests:. /mnt/nvme1/NicoCache/envs/physics_wan/bin/python \
  -m unittest tests.physical_response.test_pairing -v
```

Expected: FAIL because `physbench.physical_response.pairing` does not exist.

- [ ] **Step 3: Implement exact-decimal projection and one-factor grouping**

```python
@dataclass(frozen=True, slots=True)
class DirectionEvidence:
    semantic_direction: Literal["left", "right", "stationary", "nonnegative"]
    sign: Literal[-1, 0, 1]
    source: str

def canonical_quantity_si(
    quantity: Mapping[str, Any],
    unit_scales: Mapping[str, str],
) -> str:
    value = Decimal(str(quantity["value"]))
    scale = Decimal(unit_scales[str(quantity["unit"])])
    result = value * scale
    if not result.is_finite():
        raise ValueError("quantity SI value must be finite")
    return canonical_decimal(result)

def nuisance_signature(
    case: Mapping[str, Any],
    *,
    response_parameter_path: str,
    documented_repeat_paths: tuple[str, ...],
) -> str:
    payload = {
        "scene_id": case["scene_id"],
        "physics": copy.deepcopy(case["physics"]),
        "appearance": copy.deepcopy(case["appearance"]),
    }
    delete_exact_path(payload["physics"], response_parameter_path)
    for path in documented_repeat_paths:
        delete_exact_path(payload, path)
    return canonical_sha256(payload)

def freeze_adjacent_edges(
    groups: Sequence[InterventionGroup],
) -> tuple[CandidateEdge, ...]:
    edges: list[CandidateEdge] = []
    for group in sorted(groups, key=lambda item: item.group_id):
        levels = sorted(group.levels, key=lambda item: Decimal(item.q_si))
        for level_a, level_b in zip(levels, levels[1:]):
            edges.append(empirical_edge(group, level_a, level_b))
    return tuple(edges)
```

`compile_intervention_groups` must compare every grouped physics path other than the resolved intervention path plus the complete canonical `appearance` object; bind grouped object IDs to frozen entity IDs; retain zero-difference cases as replicates; forbid pendulum initial-angle pairs that cross signed zero; reject opposed collision candidates unless exactly one signed incident velocity differs; reject every three-ball collision before either empirical grouping or edge construction; and emit explicit exclusions for ambiguous direction, missing identity, multiple factors, unsupported scene/axis, zero real members, or `three_ball_response_out_of_scope`. Every edge resolves one canonical `ActiveQuantityBinding`; ambiguous or missing bindings fail rather than falling back to `quantity_name`. `freeze_long_range_audit_edges` freezes two-level-gap and endpoint edges under `edge_kind=long_range_audit`; they never enter a training sampler. `compile_elastic_mass_counterfactuals` uses only two-ball cases, picks the nearest distinct positive value from released collision mass support, creates a resolvable two-level collision-mass group containing the factual real level and one explicitly synthetic level, orders the values, records which endpoint is factual, and sets `edge_kind=analytic_counterfactual`. It freezes entity/channel/formula indices in `CollisionTopologyBinding` and verifies that the two endpoints differ only at the bound ball mass: the other mass, both signed incident velocities, radii, identities, and nuisance fields remain identical. `merge_group_compilations` rejects duplicate group/level IDs and is the only input passed to Task 3, so partitioning and manifest validation see the synthetic topology rather than dangling edge references.

- [ ] **Step 4: Run focused tests and a mutation guard**

Run:

```bash
PYTHONPATH=src:tests:. /mnt/nvme1/NicoCache/envs/physics_wan/bin/python \
  -m unittest tests.physical_response.test_contracts \
  tests.physical_response.test_pairing -v
```

Expected: all tests PASS; the fixture's source case dictionaries remain equal to their deep copies.

- [ ] **Step 5: Commit the pair compiler**

```bash
git add src/physbench/physical_response/pairing.py \
  tests/physical_response/_fixtures.py \
  tests/physical_response/test_pairing.py
git commit -m "feat: compile signed physical response pairs"
```

### Task 3: Seal group/value holdouts and the run-local overlay

**Files:**
- Create: `src/physbench/physical_response/splits.py`
- Create: `configs/physical_response/splits/group_holdout_v1.json`
- Create: `configs/physical_response/splits/value_holdout_v1.json`
- Create: `scripts/compile_physical_response_overlay.py`
- Create: `tests/physical_response/test_splits.py`
- Create: `tests/physical_response/test_v12_pairing.py`

**Interfaces:**
- Consumes: `load_dataset(path: str | Path, check_assets: bool = False, check_asset_hashes: bool = False) -> DatasetSnapshot` from `src/physbench/datasets/loader.py:871-1006`; the complete `GroupCompilation`, `freeze_adjacent_edges`, `freeze_long_range_audit_edges`, and `compile_elastic_mass_counterfactuals` contracts from Task 2; provenance files `datasets/provenance/releases/12.0.0/cases.jsonl` and `datasets/provenance/source_docs/20260730_parabolic_collision/collision/curation_audit.jsonl`; `inventory_tree` from Task 1.
- Produces: `load_provenance_index(cases_path: Path, collision_audit_path: Path, *, expected_case_ids: AbstractSet[str]) -> ProvenanceIndex`; `fit_response_normalizations(groups: Sequence[InterventionGroup]) -> tuple[ResponseNormalization, ...]`; `partition_response_graph(compilation: GroupCompilation, frozen_edges: Sequence[CandidateEdge], *, policy: ResponsePolicy, dataset_identity: DatasetResponseIdentity, condition_index_digest: str, experiment: Literal["group_holdout", "value_holdout"], split_policy: SplitPolicy, provenance: ProvenanceIndex, view_a_train_case_ids: AbstractSet[str], view_a_test_case_ids: AbstractSet[str]) -> ResponseManifest`; `classify_existing_edges_only(frozen_edges: Sequence[CandidateEdge], partitioned_groups: Sequence[InterventionGroup], *, experiment: Literal["group_holdout", "value_holdout"]) -> tuple[ResponseEdge, ...]`; `audit_response_overlap(manifest: ResponseManifest, *, view_a_test_case_ids: AbstractSet[str], provenance: ProvenanceIndex) -> OverlapAudit`; `write_response_overlay(manifest: ResponseManifest, audit: OverlapAudit, *, policy: ResponsePolicy, input_inventory: TreeInventory, condition_records: Sequence[ConditionIndexRecord], output_dir: Path) -> OverlayPaths`; CLI `compile_physical_response_overlay.py --dataset datasets/releases/12.0.0/dataset.json --policy configs/physical_response/policies/physical_response_v1.json --split configs/physical_response/splits/group_holdout_v1.json --experiment group_holdout --output-dir run/physical_response_group_holdout_v1_seed42/physical_response`.
- `compile_condition_index(snapshot: DatasetSnapshot, projections: Sequence[SignedProjection], *, entity_manifests: Mapping[str, EntityManifest], dataset_root: Path) -> tuple[ConditionIndexRecord, ...]` resolves every prompt, structured quantity, reference video, first frame, mask manifest, and initial-mask asset once; all stored paths are repository-relative and every referenced byte string is hashed. `write_response_overlay` additionally receives these records, verifies their aggregate digest equals `manifest.condition_index_digest`, and cannot infer media through a global path.
- `load_condition_index(path: Path, *, expected_digest: str) -> ConditionIndex` verifies canonical order, every record digest, and the aggregate digest before exposing `records_by_case`.
- Produces these immutable artifacts: `frozen/physical_response_policy.json`, `physical_response/manifest.json`, `physical_response/condition_index.jsonl`, `physical_response/projections.jsonl`, `physical_response/groups.jsonl`, `physical_response/edges.jsonl`, `physical_response/overlap_audit.json`, `physical_response/flowmatch_train_case_ids.json`, and `physical_response/input_inventory.json`.
- Data contracts: `SplitPolicy(experiment_id: str, seed: int, partition_fractions: tuple[tuple[Partition, str], ...], minimum_eval_groups: int, minimum_train_levels: int, minimum_eval_levels: int, digest: str)`; `ProvenanceIndex(source_trial_component_by_case: Mapping[str, str], near_duplicate_component_by_case: Mapping[str, str], input_digests: tuple[tuple[str, str], ...])`; `DatasetResponseIdentity(dataset_id: str, dataset_release: str, dataset_metadata_digest: str, dataset_tree_digest: str, input_inventory_digest: str)`; `ConditionIndexRecord(case_id: str, scene_id: str, prompt: str, quantities: tuple[dict[str, Any], ...], video_path: str, video_sha256: str, first_frame_path: str, first_frame_sha256: str, mask_manifest_path: str | None, mask_manifest_sha256: str | None, initial_mask_path: str | None, initial_mask_sha256: str | None, width: int, height: int, record_digest: str)`; `ConditionIndex(records_by_case: Mapping[str, ConditionIndexRecord], digest: str)`; and `OverlayPaths(policy: Path, manifest: Path, condition_index: Path, projections: Path, groups: Path, edges: Path, overlap_audit: Path, flowmatch_train_case_ids: Path, input_inventory: Path)`.

- [ ] **Step 1: Write split/leakage tests before the compiler**

```python
class ResponseSplitTests(unittest.TestCase):
    def test_value_holdout_does_not_reconnect_neighbors(self) -> None:
        group = intervention_group_with_levels("1", "2", "3")
        frozen = freeze_adjacent_edges([group])
        manifest = partition_response_graph(
            fixture_group_compilation(group), frozen, policy=test_policy(),
            dataset_identity=fixture_dataset_response_identity(),
            condition_index_digest=fixture_condition_index_digest(),
            experiment="value_holdout",
            split_policy=fixture_split_holding_out("2"),
            provenance=fixture_provenance(group),
            view_a_train_case_ids={
                case_id
                for level in group.levels
                for case_id in level.member_case_ids
            },
            view_a_test_case_ids=set(),
        )
        train_pairs = {
            (edge.q_a_si, edge.q_b_si)
            for edge in manifest.edges
            if edge.partition == "response_train"
        }
        self.assertNotIn(("1", "3"), train_pairs)

    def test_held_out_members_are_absent_from_flowmatch_training(self) -> None:
        manifest = fixture_partitioned_manifest()
        held_out = {
            case_id for group in manifest.groups
            for level in group.levels
            if level.partition != "response_train"
            for case_id in level.member_case_ids
        }
        self.assertTrue(held_out.isdisjoint(manifest.flowmatch_train_case_ids))

    def test_source_trial_component_cannot_cross_partitions(self) -> None:
        with self.assertRaisesRegex(ValueError, "provenance component overlap"):
            audit_response_overlap(
                manifest_with_split_source_trial(),
                view_a_test_case_ids=set(),
                provenance=linked_fixture_provenance(),
            )

    def test_q_normalization_uses_response_train_levels_only(self) -> None:
        first = fit_response_normalizations(partitioned_group(test_q="9"))
        second = fit_response_normalizations(partitioned_group(test_q="900"))
        self.assertEqual(first, second)

    def test_shared_case_and_provenance_component_has_one_partition_across_axes(self) -> None:
        manifest = partition_cross_axis_shared_member_fixture()
        by_case: dict[str, set[Partition]] = collections.defaultdict(set)
        for group in manifest.groups:
            for level in group.levels:
                for case_id in level.member_case_ids:
                    by_case[case_id].add(level.partition)
        self.assertTrue(all(len(partitions) == 1 for partitions in by_case.values()))
        self.assert_provenance_components_each_have_one_partition(manifest)

    def test_candidate_to_response_conversion_preserves_frozen_edge_identity(self) -> None:
        candidate = fixture_candidate_edge()
        response = classify_existing_edges_only(
            [candidate], fixture_partitioned_groups(), experiment="group_holdout"
        )[0]
        for field in dataclasses.fields(CandidateEdge):
            self.assertEqual(getattr(candidate, field.name), getattr(response, field.name))
        self.assertIn(response.partition, get_args(Partition))
```

- [ ] **Step 2: Run split tests and observe the missing-module failure**

Run:

```bash
PYTHONPATH=src:tests:. /mnt/nvme1/NicoCache/envs/physics_wan/bin/python \
  -m unittest tests.physical_response.test_splits -v
```

Expected: FAIL because `physbench.physical_response.splits` does not exist.

- [ ] **Step 3: Implement freeze-before-partition and a fail-closed overlay CLI**

```python
def partition_response_graph(
    compilation: GroupCompilation,
    frozen_edges: Sequence[CandidateEdge],
    *,
    policy: ResponsePolicy,
    dataset_identity: DatasetResponseIdentity,
    condition_index_digest: str,
    experiment: Literal["group_holdout", "value_holdout"],
    split_policy: SplitPolicy,
    provenance: ProvenanceIndex,
    view_a_train_case_ids: AbstractSet[str],
    view_a_test_case_ids: AbstractSet[str],
) -> ResponseManifest:
    groups = compilation.groups  # empirical + counterfactual topology merged in Task 2
    assert_all_members_in_view_a_train(groups, view_a_train_case_ids)
    blocks = provenance.atomic_assignment_blocks(
        groups, experiment=experiment,
        synthetic_partition_source="candidate_edge.factual_case_id",
    )
    assignments = deterministic_stratified_assignment(
        blocks,
        seed=split_policy.seed,
        fractions=split_policy.partition_fractions,
        strata=("scene_id", "axis_id"),
    )
    partitioned_groups = assign_members_without_splitting_blocks(
        groups, assignments
    )
    partitioned_edges = classify_existing_edges_only(
        frozen_edges, partitioned_groups, experiment=experiment
    )
    held_out = held_out_case_ids(partitioned_groups)
    flowmatch_train = tuple(sorted(view_a_train_case_ids - held_out))
    normalizations = fit_response_normalizations(partitioned_groups)
    return seal_manifest(
        experiment_id=split_policy.experiment_id,
        policy_digest=policy.digest,
        split_policy_digest=split_policy.digest,
        dataset_identity=dataset_identity,
        condition_index_digest=condition_index_digest,
        provenance_digests=provenance.input_digests,
        projections=compilation.projections,
        groups=partitioned_groups,
        edges=partitioned_edges,
        normalizations=normalizations,
        flowmatch_train_case_ids=flowmatch_train,
        exclusions=tuple(sorted(compilation.exclusions.items())),
    )
```

Both split JSON files use seed `42` and target fractions `response_train=0.60`, `response_validation=0.20`, `response_test=0.20`. Group holdout requires at least one complete validation and test group per scene/axis when at least three provenance-atomic groups exist. Value holdout requires at least two retained train levels and one validation and test level when at least four provenance-atomic levels exist; unsupported strata become `not_estimable`. For each axis, `fit_response_normalizations` stores the response-train median as `q_center_si` and the response-train interquartile range, computed by deterministic linear interpolation on sorted unique Decimal values, as positive `q_scale_si`; fewer than two train levels or zero scale is `not_estimable`. The compiler hashes and coverage-validates both provenance sources, freezes adjacent and long-range graphs before assignment, prevents all long-range edges from entering `response_train`, carries mass-counterfactual partition from its factual case, forbids output paths under `datasets/`, compares complete Dataset tree inventories before and after, writes via staging plus exclusive finalization, and never adds fields to `BaselineTaskInstance`.

- [ ] **Step 4: Run unit and V12 read-only integration tests**

Add V12 assertions for deterministic byte-identical manifests across two temporary output roots; View A train-only source membership; zero train/evaluation overlap; unchanged Dataset inventory; explicit exclusion of `incline_r1_a41deg_bgoil_img_0332` and `incline_r1_a44deg_bgblack_img_0394` from physical supervision; provenance coverage; and rejection of opposed incidents that change two velocities. Do not assert provisional total edge counts.

Run:

```bash
PYTHONPATH=src:tests:. /mnt/nvme1/NicoCache/envs/physics_wan/bin/python \
  -m unittest tests.physical_response.test_splits \
  tests.physical_response.test_v12_pairing -v
```

Expected: all tests PASS and `git status --short datasets` prints no entries.

- [ ] **Step 5: Commit the sealed overlay compiler**

```bash
git add -- src/physbench/physical_response/splits.py \
  configs/physical_response/splits/group_holdout_v1.json \
  configs/physical_response/splits/value_holdout_v1.json \
  scripts/compile_physical_response_overlay.py \
  tests/physical_response/test_splits.py \
  tests/physical_response/test_v12_pairing.py
git commit -m "feat: seal physical response holdouts"
```

### Task 4: Define differentiable state tensors and frozen coordinate frames

**Files:**
- Create: `src/physbench/physical_response/state.py`
- Create: `src/physbench/physical_response/coordinates.py`
- Create: `configs/physical_response/calibrations/v1.json`
- Create: `schemas/physical_response_coordinate_manifest.schema.json`
- Create: `scripts/build_physical_response_coordinates.py`
- Create: `tests/fixtures/physical_response/coordinates_v1.json`
- Create: `tests/physical_response/test_state_and_coordinates.py`

**Interfaces:**
- Consumes: sealed `ConditionIndex` and `ResponsePolicy` from Tasks 1/3, Dataset case metadata, real first-frame/mask bytes named by the condition index, and explicit calibration-source bytes; no generated or predicted pixel is accepted by a coordinate-construction function.
- Produces: `SoftTrackBatch`, `StateSchema`, `PhysicalStateBatch`, `CoordinateFrameRecord`, `CoordinateFrameManifest`, `CoordinateFrameBinding`, `CoordinateFrameIndex`, and `CoordinateFrameTensor`; `build_coordinate_frame_index(response_manifest: ResponseManifest, condition_index: ConditionIndex, cases_by_id: Mapping[str, Mapping[str, Any]], *, policy: ResponsePolicy, calibration_sources: Mapping[str, Path], cache_root: Path, binding_path: Path) -> CoordinateFrameBinding`; `load_coordinate_frame_index(binding_path: Path, *, cache_root: Path, expected_response_manifest_digest: str, expected_condition_index_digest: str, expected_policy_digest: str) -> CoordinateFrameIndex`; `CoordinateFrameIndex.get(case_id: str, frame_id: str, *, expected_condition_record_digest: str) -> CoordinateFrameRecord`; `reduce_soft_mask_logits(mask_logits: torch.Tensor, object_score_logits: torch.Tensor, *, output_size: tuple[int, int], frame_indices: torch.Tensor, eps: float = 1e-6) -> SoftTrackBatch`; `materialize_coordinate_frame(record: CoordinateFrameRecord, *, device: torch.device, dtype: torch.dtype = torch.float32) -> CoordinateFrameTensor`; `project_pixels(points_xy_px: torch.Tensor, frame: CoordinateFrameTensor) -> torch.Tensor`; `project_vectors(vectors_xy_px: torch.Tensor, frame: CoordinateFrameTensor) -> torch.Tensor`; `physical_times(frame_indices: torch.Tensor, frame: CoordinateFrameTensor) -> torch.Tensor`; `central_difference(values: torch.Tensor, times_s: torch.Tensor) -> torch.Tensor`; `soft_event_time(weights: torch.Tensor, times_s: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor`; `flatten_physical_state(state: PhysicalStateBatch) -> torch.Tensor`; `unflatten_physical_state(packed: torch.Tensor, schema: StateSchema, *, batch_size: int, frame_count: int) -> PhysicalStateBatch`; CLI `build_physical_response_coordinates.py --dataset datasets/releases/12.0.0/dataset.json --response-manifest run/physical_response_group_holdout_v1_seed42/physical_response/manifest.json --condition-index run/physical_response_group_holdout_v1_seed42/physical_response/condition_index.jsonl --response-policy run/physical_response_group_holdout_v1_seed42/frozen/physical_response_policy.json --calibration-sources configs/physical_response/calibrations/v1.json --cache-root cache/physical_response/coordinates_v1 --binding run/physical_response_group_holdout_v1_seed42/physical_response/coordinates/binding.json`.
- Tensor contract: state math and exchanged state tensors are FP32. `SoftTrackBatch` shapes are centroid `[B,T,O,2]`, area/visibility `[B,T,O]`, second moment `[B,T,O,2,2]`, channel overlap `[B,T,O,O]`, and frame indices `[T]`. `PhysicalStateBatch` shapes are trajectory `[B,T,C]`, trajectory-validity logits `[B,T,C]`, summaries `[B,S]`, summary-validity logits `[B,S]`, event weights `[B,T,E]`, and validity values `[B,V]`; ordered names live in `StateSchema`.
- Data contracts: `SoftTrackBatch(centroid_xy_px: torch.Tensor, area_probability_px2: torch.Tensor, visibility_logit: torch.Tensor, second_moment_px2: torch.Tensor, channel_overlap_probability: torch.Tensor, frame_indices: torch.Tensor, output_size: tuple[int, int])`; `CoordinateFrameManifest(schema_version: str, response_manifest_digest: str, condition_index_digest: str, policy_digest: str, dataset_metadata_digest: str, dataset_tree_digest: str, coordinate_versions: tuple[tuple[str, str], ...], calibration_input_digests: tuple[tuple[str, str], ...], record_digests: tuple[tuple[tuple[str, str], str], ...], frames_path: str, frames_sha256: str, digest: str)`; `CoordinateFrameBinding(manifest_digest: str, cache_entry_relative_path: str, response_manifest_digest: str, condition_index_digest: str, policy_digest: str, digest: str)`; `CoordinateFrameIndex(manifest: CoordinateFrameManifest, records_by_case_frame: Mapping[tuple[str, str], CoordinateFrameRecord], digest: str)`; `CoordinateFrameTensor(frame_id: str, scene_id: str, origin_xy_px: torch.Tensor, basis_xy: torch.Tensor, meters_per_pixel: torch.Tensor, encoded_fps: torch.Tensor, encoded_to_physical_speed: torch.Tensor, source_digest: str)`; and the `StateSchema`/`PhysicalStateBatch` constructors shown in Step 3. The builder writes `frames.jsonl` plus `manifest.json` into `cache/physical_response/coordinates_v1/sha256/{manifest.digest}/`, then atomically writes the typed run-local binding. The loader resolves `cache_entry_relative_path` only against its explicit `cache_root`, rejects absolute paths, `..`, symlink escape, or any manifest/frames path outside that root, and re-verifies both cache files before parsing a record.
- `CoordinateFrameBinding.manifest_digest` is the scientific `coordinate_manifest_digest`; `CoordinateFrameBinding.digest` authenticates only the run-local pointer document. `CoordinateFrameIndex.digest` is exactly `CoordinateFrameIndex.manifest.digest`, never the binding digest. `CoordinateFrameRecord.frame_id` and `CoordinateFrameTensor.frame_id` are internal aliases of the cross-artifact name `coordinate_frame_id` and may never diverge.

- [ ] **Step 1: Write analytic soft-mask and fixed-frame tests**

```python
class StateAndCoordinateTests(unittest.TestCase):
    def test_soft_disk_centroid_has_input_gradient(self) -> None:
        logits = synthetic_disk_logits(center=(7.0, 11.0)).requires_grad_()
        track = reduce_soft_mask_logits(
            logits, torch.full((1, 1, 1), 8.0), output_size=(24, 32),
            frame_indices=torch.tensor([0]),
        )
        self.assertTrue(torch.allclose(
            track.centroid_xy_px[0, 0, 0], torch.tensor([7.0, 11.0]),
            atol=0.15,
        ))
        track.centroid_xy_px.sum().backward()
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)

    def test_near_binary_soft_track_matches_independent_hard_mask_reference(self) -> None:
        hard_mask = asymmetric_hard_mask_fixture()
        logits = torch.where(hard_mask, 14.0, -14.0).requires_grad_()
        soft = reduce_soft_mask_logits(
            logits, torch.full((1, 1, 1), 14.0), output_size=(24, 32),
            frame_indices=torch.tensor([0]),
        )
        reference = independent_hard_mask_moments(hard_mask, output_size=(24, 32))
        self.assertTensorClose(soft.centroid_xy_px, reference.centroid_xy_px, atol=2e-4)
        self.assertTensorClose(soft.area_probability_px2, reference.area_px2, atol=2e-3)
        self.assertTensorClose(soft.second_moment_px2, reference.second_moment_px2, atol=3e-4)
        soft_state = fixture_common_state_from_track(soft)
        hard_state = independent_hard_mask_state_reference(
            hard_mask, output_size=(24, 32)
        )
        self.assertTensorClose(soft_state.trajectory, hard_state.trajectory, atol=5e-4)
        self.assertTensorClose(soft_state.summaries, hard_state.summaries, atol=5e-4)
        (soft.centroid_xy_px.sum() + soft.second_moment_px2.sum()).backward()
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)

    def test_coordinate_frame_cannot_be_fitted_from_prediction(self) -> None:
        signature = inspect.signature(materialize_coordinate_frame)
        self.assertEqual(["record", "device", "dtype"], list(signature.parameters))
        frame = materialize_coordinate_frame(frozen_frame_record(), device=torch.device("cpu"))
        before = project_pixels(torch.tensor([[[10.0, 20.0]]]), frame)
        after = project_pixels(torch.tensor([[[110.0, 220.0]]]), frame)
        self.assertTrue(torch.allclose(after - before, torch.tensor([[[1.0, 2.0]]])))

    def test_coordinate_index_is_real_only_content_addressed_and_case_bound(self) -> None:
        index = build_fixture_coordinate_index()
        record = index.get(
            "pendulum_case", "pendulum_frame_v1",
            expected_condition_record_digest="condition-record-a"
        )
        self.assertEqual("pendulum_case", record.case_id)
        self.assertNotIn("generated", index.manifest.calibration_input_digests)
        with self.assertRaisesRegex(ValueError, "condition record digest"):
            index.get(
                "pendulum_case", "pendulum_frame_v1",
                expected_condition_record_digest="tampered",
            )

    def test_coordinate_manifest_rejects_any_changed_calibration_byte(self) -> None:
        first = build_fixture_coordinate_index()
        mutate_calibration_source_byte()
        with self.assertRaisesRegex(ValueError, "calibration input digest"):
            load_coordinate_frame_index(
                fixture_coordinate_binding_path(),
                cache_root=fixture_coordinate_cache_root(),
                expected_response_manifest_digest=first.manifest.response_manifest_digest,
                expected_condition_index_digest=first.manifest.condition_index_digest,
                expected_policy_digest=first.manifest.policy_digest,
            )

    def test_binding_path_cannot_escape_explicit_coordinate_cache_root(self) -> None:
        with self.assertRaisesRegex(ValueError, "coordinate cache containment"):
            load_coordinate_frame_index(
                fixture_coordinate_binding_path(cache_entry="../foreign"),
                cache_root=fixture_coordinate_cache_root(),
                **fixture_coordinate_identity_expectations(),
            )

    def test_every_scene_coordinate_transform_and_physical_time(self) -> None:
        for scene_id, point_px, expected_si, frame_index, expected_time_s in (
            ("pendulum", (20.0, 30.0), (0.10, -0.20), 12, 0.20),
            ("uniform_circular_motion", (45.0, 25.0), (0.20, 0.00), 24, 0.40),
            ("inclined_plane_slide", (16.0, 18.0), (0.12, 0.00), 30, 0.50),
            ("parabolic_motion", (14.0, 36.0), (0.08, 0.16), 36, 0.60),
            ("collision_1d", (40.0, 22.0), (0.30, 0.00), 48, 0.80),
        ):
            with self.subTest(scene_id=scene_id):
                frame = materialize_coordinate_frame(
                    frozen_scene_frame_record(scene_id), device=torch.device("cpu")
                )
                actual_si = project_pixels(torch.tensor([[point_px]]), frame)
                actual_time = physical_times(torch.tensor([frame_index]), frame)
                self.assertTensorClose(actual_si, torch.tensor([[expected_si]]), atol=1e-6)
                self.assertTensorClose(actual_time, torch.tensor([expected_time_s]), atol=1e-6)

    def test_state_flatten_round_trip_is_exact(self) -> None:
        state = fixture_physical_state(batch_size=2, frame_count=16)
        restored = unflatten_physical_state(
            flatten_physical_state(state), state.schema,
            batch_size=2, frame_count=16,
        )
        self.assert_physical_states_equal(state, restored)
```

- [ ] **Step 2: Run the test and verify the missing types fail import**

Run:

```bash
PYTHONPATH=src:tests:. /mnt/nvme1/NicoCache/envs/physics_wan/bin/python \
  -m unittest tests.physical_response.test_state_and_coordinates -v
```

Expected: FAIL importing `physbench.physical_response.state`.

- [ ] **Step 3: Implement soft moments, immutable calibration records, and stable packing**

```python
@dataclass(frozen=True, slots=True)
class StateSchema:
    schema_id: str
    trajectory_names: tuple[str, ...]
    trajectory_units: tuple[str, ...]
    summary_names: tuple[str, ...]
    summary_units: tuple[str, ...]
    event_names: tuple[str, ...]
    validity_names: tuple[str, ...]
    object_count: int

    def __post_init__(self) -> None:
        if len(self.trajectory_names) != len(self.trajectory_units):
            raise ValueError("trajectory names and units differ")
        if len(self.summary_names) != len(self.summary_units):
            raise ValueError("summary names and units differ")

@dataclass(slots=True)
class PhysicalStateBatch:
    schema: StateSchema
    trajectory: torch.Tensor
    trajectory_validity_logit: torch.Tensor
    summaries: torch.Tensor
    summary_validity_logit: torch.Tensor
    event_weights: torch.Tensor
    validity_values: torch.Tensor

@dataclass(frozen=True, slots=True)
class CoordinateFrameRecord:
    case_id: str
    frame_id: str
    scene_id: str
    origin_xy_px: tuple[float, float]
    x_axis_xy: tuple[float, float]
    y_axis_xy: tuple[float, float]
    meters_per_pixel: float
    encoded_fps: float
    encoded_to_physical_speed: float
    first_frame_sha256: str
    initial_mask_sha256: str | None
    condition_record_digest: str
    coordinate_version: str
    calibration_input_digests: tuple[tuple[str, str], ...]
    source_digest: str
    digest: str

def reduce_soft_mask_logits(
    mask_logits: torch.Tensor,
    object_score_logits: torch.Tensor,
    *,
    output_size: tuple[int, int],
    frame_indices: torch.Tensor,
    eps: float = 1e-6,
) -> SoftTrackBatch:
    probability = mask_logits.float().sigmoid()
    height, width = probability.shape[-2:]
    yy, xx = torch.meshgrid(
        torch.arange(height, device=probability.device, dtype=torch.float32),
        torch.arange(width, device=probability.device, dtype=torch.float32),
        indexing="ij",
    )
    mass = probability.sum(dim=(-2, -1)).clamp_min(eps)
    centroid = torch.stack(
        ((probability * xx).sum((-2, -1)) / mass,
         (probability * yy).sum((-2, -1)) / mass), dim=-1,
    )
    scale = centroid.new_tensor([output_size[1] / width, output_size[0] / height])
    centroid = centroid * scale
    visibility = object_score_logits.float() + torch.log(mass / (height * width))
    return build_track_with_second_moments_and_pairwise_overlap(
        probability, centroid, visibility, output_size, frame_indices, eps
    )
```

`build_coordinate_frame_index` first verifies the response manifest's Dataset/policy/condition identities, then dispatches only to policy-versioned, scene-specific real-reference builders. Each builder may consume Dataset quantities, the encoded FPS/speed metadata, and sealed real first-frame/mask or calibration bytes, but its public signature has no prediction/generated-video argument. It rejects missing scale, basis, origin, speed, source provenance, or any disagreement with `InterventionGroup.coordinate_frame_id`, and writes canonical records only after re-verifying all bytes from the `ConditionIndex`. `CoordinateFrameRecord` validation normalizes neither basis vector nor scale silently: both basis vectors must already be unit length within `1e-6`, mutually orthogonal within `1e-6`, scale and frame rates must be finite and positive, and the record identity binds the case/frame ID, condition record, real first-frame/mask identities, every numeric value, coordinate version, and every calibration source. `project_pixels` subtracts the frozen origin, projects onto the frozen basis, and applies meters-per-pixel; `project_vectors` applies only the basis rotation so callers such as the circular extractor may apply the sealed scale exactly once. `physical_times` computes `frame_index / encoded_fps / encoded_to_physical_speed`.

- [ ] **Step 4: Run state tests in FP32 and double precision**

Run:

```bash
PYTHONPATH=src:tests:. /mnt/nvme1/NicoCache/envs/physics_wan/bin/python \
  -m unittest tests.physical_response.test_state_and_coordinates -v
PYTHONPATH=src:tests:. /mnt/nvme1/NicoCache/envs/physics_wan/bin/python \
  scripts/build_physical_response_coordinates.py \
  --fixture tests/fixtures/physical_response/coordinates_v1.json \
  --output-dir /tmp/physbench-coordinate-dry-run \
  --verify-only
```

Expected: all tests PASS, including finite-difference tolerance checks against an FP64 fixture.

- [ ] **Step 5: Commit common state and coordinate primitives**

```bash
git add src/physbench/physical_response/state.py \
  src/physbench/physical_response/coordinates.py \
  configs/physical_response/calibrations/v1.json \
  schemas/physical_response_coordinate_manifest.schema.json \
  scripts/build_physical_response_coordinates.py \
  tests/fixtures/physical_response/coordinates_v1.json \
  tests/physical_response/test_state_and_coordinates.py
git commit -m "feat: add differentiable physical state primitives"
```

### Task 5: Build the frozen differentiable SAM2 observer

**Files:**
- Create: `src/physbench/physical_response/differentiable_sam2.py`
- Create: `tests/physical_response/test_differentiable_sam2.py`
- Create: `tests/integration/physical_response_sam2_gradient_worker.py`

**Interfaces:**
- Consumes: `sam2.build_sam.build_sam2(config_file, ckpt_path=None, device="cuda", mode="eval", hydra_overrides_extra=[], apply_postprocessing=True, **kwargs)` at `/root/Nico/third_party/sam2/sam2/build_sam.py:70-102`; `SAM2Base.forward_image(img_batch: torch.Tensor)` at `sam2/modeling/sam2_base.py:467-479`; `SAM2Base._prepare_backbone_features(backbone_out)` at `:481-496`; the exact installed `SAM2Base.track_step` contract at `:814-880`; and `reduce_soft_mask_logits` from Task 4.
- Produces: `SAM2Identity.from_paths(source_root: Path, config_name: str, config_path: Path, checkpoint_path: Path) -> SAM2Identity`; `SAM2TrackConfig(image_size: int, temporal_block_frames: int, detach_after_observer_positions: tuple[int, ...], checkpoint_image_encoder: bool, checkpoint_track_step: bool, checkpoint_memory_encoder: bool, mask_temperature: float)`; `build_frozen_differentiable_sam2(identity: SAM2Identity, config: SAM2TrackConfig, *, device: torch.device, dtype: torch.dtype) -> DifferentiableSAM2Observer`; `soft_object_gate(object_score_logits: torch.Tensor, object_pointer: torch.Tensor, no_object_pointer: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]`; `DifferentiableSAM2Observer.forward(rgb: torch.Tensor, initial_mask_probability: torch.Tensor, *, frame_indices: torch.Tensor) -> SoftTrackBatch`.
- Tensor contract: `rgb` is `[B,T,3,H,W]` in `[0,1]`; the Dataset mask prompt is `[B,O,H,W]`; output follows Task 4. Parameters have `requires_grad=False`, but output remains connected to every current RGB frame. Temporal detachment may detach only packed SAM2 memory at configured block boundaries.
- Also produces `SAM2ObserverIdentity.from_runtime(model: SAM2Identity, track: SAM2TrackConfig, *, dtype: torch.dtype, overrides: Sequence[str], post_load_mutations: Sequence[str]) -> SAM2ObserverIdentity`, whose digest is the cache/run identity for the realized numerical observer.
- Data contracts: `SAM2Identity(source_root: Path, source_commit: str, config_name: str, config_path: Path, config_sha256: str, checkpoint_path: Path, checkpoint_sha256: str, checkpoint_size_bytes: int, preprocessing_id: str)` and `SAM2TrackConfig(image_size: int, temporal_block_frames: int, detach_after_observer_positions: tuple[int, ...], checkpoint_image_encoder: bool, checkpoint_track_step: bool, checkpoint_memory_encoder: bool, mask_temperature: float)`. Detach positions are zero-based positions in the observed sequence, must be exact block ends strictly before the final observed position, and an empty tuple means full recurrent autograd. `SAM2TrackConfig.digest` is a read-only canonical identity over all seven fields: image size, temporal block size, the exact detach-position tuple, all three checkpoint flags, and mask temperature.
- `SAM2ObserverIdentity(model_digest: str, track_config_digest: str, dtype: str, overrides: tuple[str, ...], post_load_mutations: tuple[str, ...], digest: str)` binds the two outer/decoder score settings separately so retaining the decoder score head cannot be confused with the disabled outer hard gate.

- [ ] **Step 1: Write construction, postprocessing, shape, and input-gradient tests**

```python
class DifferentiableSAM2Tests(unittest.TestCase):
    def test_builder_freezes_parameters_and_disables_hard_paths(self) -> None:
        observer = build_test_observer()
        self.assertTrue(all(not p.requires_grad for p in observer.sam2.parameters()))
        self.assertFalse(observer.sam2.multimask_output_in_sam)
        self.assertFalse(observer.sam2.multimask_output_for_tracking)
        self.assertFalse(observer.sam2.binarize_mask_from_pts_for_mem_enc)
        self.assertFalse(observer.sam2.non_overlap_masks_for_mem_enc)
        self.assertFalse(observer.sam2.use_mask_input_as_output_without_sam)
        self.assertFalse(observer.sam2.pred_obj_scores)
        self.assertTrue(observer.sam2.sam_mask_decoder.pred_obj_scores)
        self.assertFalse(observer.sam2.sam_mask_decoder.dynamic_multimask_via_stability)

    def test_object_presence_gate_is_soft(self) -> None:
        score = torch.tensor([-0.01, 0.01], requires_grad=True)
        pointer, presence = soft_object_gate(
            score, torch.ones(2, 4), torch.zeros(1, 4)
        )
        pointer.sum().backward()
        self.assertTrue(torch.all((presence > 0.49) & (presence < 0.51)))
        self.assertTrue(torch.all(score.grad != 0))

    @unittest.skipUnless(os.getenv("PHYSBENCH_RUN_SAM2_GPU_TESTS") == "1", "GPU gate")
    def test_three_frame_track_keeps_rgb_gradient(self) -> None:
        observer = build_real_tiny_observer(
            block_frames=2, detach_after_observer_positions=(1,)
        )
        rgb, initial_mask = three_frame_gpu_fixture(requires_grad=True)
        track = observer(rgb, initial_mask, frame_indices=torch.arange(3, device="cuda"))
        track.centroid_xy_px[:, 1:].sum().backward()
        self.assertTrue(torch.isfinite(rgb.grad).all())
        self.assertGreater(float(rgb.grad[:, 1:].abs().sum()), 0.0)
        self.assertTrue(all(parameter.grad is None for parameter in observer.sam2.parameters()))
```

- [ ] **Step 2: Run the CPU contract test and observe the missing wrapper**

Run:

```bash
PYTHONPATH=src:tests:.:/root/Nico/third_party/sam2 \
  /mnt/nvme1/NicoCache/envs/physics_wan/bin/python \
  -m unittest tests.physical_response.test_differentiable_sam2 -v
```

Expected: FAIL importing `DifferentiableSAM2Observer`.

- [ ] **Step 3: Implement low-level tensor tracking with soft reduction**

```python
SAM2_OVERRIDES = (
    "++model.multimask_output_in_sam=false",
    "++model.multimask_output_for_tracking=false",
    "++model.binarize_mask_from_pts_for_mem_enc=false",
    "++model.non_overlap_masks_for_mem_enc=false",
    "++model.use_mask_input_as_output_without_sam=false",
    "++model.compile_image_encoder=false",
    "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=false",
)

def build_frozen_differentiable_sam2(
    identity: SAM2Identity,
    config: SAM2TrackConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> DifferentiableSAM2Observer:
    identity.verify()
    model = build_sam2(
        identity.config_name,
        str(identity.checkpoint_path),
        device=str(device), mode="eval",
        hydra_overrides_extra=list(SAM2_OVERRIDES),
        apply_postprocessing=False,
    ).to(dtype=dtype)
    # build_sam2 has already strict-loaded every checkpoint key. Disable only
    # SAM2Base's outer hard score gates; retain the decoder score head and its
    # frozen weights for the repository-owned soft gate below.
    model.pred_obj_scores = False
    model.requires_grad_(False).eval()
    return DifferentiableSAM2Observer(model, identity, config)

def soft_object_gate(
    object_score_logits: torch.Tensor,
    object_pointer: torch.Tensor,
    no_object_pointer: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    presence = object_score_logits.float().sigmoid()
    pointer = (
        presence * object_pointer
        + (1.0 - presence) * no_object_pointer.expand_as(object_pointer)
    )
    return pointer, presence

class DifferentiableSAM2Observer(nn.Module):
    def forward(
        self,
        rgb: torch.Tensor,
        initial_mask_probability: torch.Tensor,
        *,
        frame_indices: torch.Tensor,
    ) -> SoftTrackBatch:
        normalized = normalize_and_resize_rgb(rgb, self.config.image_size)
        output_dict = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
        reduced_blocks: list[SoftTrackBatch] = []
        for block in temporal_blocks(normalized, self.config.temporal_block_frames):
            logits, scores, output_dict = self._soft_track_block(
                block, initial_mask_probability, output_dict
            )
            reduced_blocks.append(reduce_soft_mask_logits(
                logits / self.config.mask_temperature, scores,
                output_size=rgb.shape[-2:],
                frame_indices=frame_indices[block.start:block.stop],
            ))
            if block.stop - 1 in self.config.detach_after_observer_positions:
                output_dict = detach_only_memory_tensors(output_dict)
        return concatenate_soft_tracks(reduced_blocks)
```

`_soft_track_block` flattens `[B,O]` into SAM2's object batch, expands each frame's backbone features to the object batch without writing temporary images, and calls `track_step(..., run_mem_encoder=False)` with the resized first-frame mask only for frame zero. Because the strict-loaded outer `pred_obj_scores` flag is then false and multimask is disabled, stock tracking performs neither the hard mask replacement at `sam2_base.py:359` nor multimask argmax. The wrapper uses `sigmoid(object_score_logits)` to soft-mix `obj_ptr` with the retained frozen `no_obj_ptr`, feeds soft mask probabilities through `memory_encoder(..., skip_mask_sigmoid=True)`, and soft-mixes `no_obj_embed_spatial` by `1-presence`; it never calls stock `_encode_new_memory`, whose line 716 gate is hard.

Image, tracking, and custom-memory checkpoint functions accept and return tensors only. Before each checkpoint call, the wrapper packs an immutable snapshot of all prior memory entries; it never closes over the mutable output dictionary. A declared truncation boundary detaches `maskmem_features`, `maskmem_pos_enc`, and `obj_ptr` in every retained conditioning and non-conditioning entry, while the current block's RGB-to-moment graph remains intact. The identity record includes config name `configs/sam2.1/sam2.1_hiera_t.yaml`, config SHA-256 `f932eac1c6241e910031b2f000a81cd9f8a8d4896e2277ab5ffb721f378b188d`, checkpoint size `156008466`, every runtime override above, the post-load outer-score-gate change, dtype, block size, and exact detach-position tuple.

- [ ] **Step 4: Run the real three-frame CUDA gradient gate**

Run:

```bash
CUDA_VISIBLE_DEVICES=0 PHYSBENCH_RUN_SAM2_GPU_TESTS=1 \
PYTHONPATH=src:tests:.:/root/Nico/third_party/sam2 \
  /mnt/nvme1/NicoCache/envs/physics_wan/bin/python \
  -m unittest tests.physical_response.test_differentiable_sam2 -v
```

Expected: PASS; checkpointed and ordinary three-frame outputs/gradients meet `rtol=1e-4, atol=1e-5`; full-memory mode gives every RGB frame a finite non-zero gradient; truncated mode preserves the current block's gradients and removes only cross-boundary memory gradients; overlapping objects retain soft overlap; outputs are ordinary autograd tensors rather than inference tensors; and every SAM2 parameter gradient is `None`.

- [ ] **Step 5: Commit the differentiable observer**

```bash
git add src/physbench/physical_response/differentiable_sam2.py \
  tests/physical_response/test_differentiable_sam2.py \
  tests/integration/physical_response_sam2_gradient_worker.py
git commit -m "feat: add differentiable frozen SAM2 observer"
```

### Task 6: Extract pendulum and circular-motion states

**Files:**
- Create: `src/physbench/physical_response/state_extractors/__init__.py`
- Create: `src/physbench/physical_response/state_extractors/pendulum.py`
- Create: `src/physbench/physical_response/state_extractors/circular.py`
- Create: `tests/physical_response/test_pendulum_state.py`
- Create: `tests/physical_response/test_circular_state.py`

**Interfaces:**
- Consumes: `SoftTrackBatch`, `PhysicalStateBatch`, `StateSchema`, `CoordinateFrameTensor`, `physical_times`, and `central_difference` from Task 4.
- Produces: `PendulumCondition(string_length_m: torch.Tensor, bob_radius_m: torch.Tensor, initial_angle_magnitude_rad: torch.Tensor, pivot_xy_px: torch.Tensor)`; `CircularCondition(center_xy_px: torch.Tensor, angular_velocity_rad_s: torch.Tensor, intervened_object_index: int)`; `extract_pendulum_state(track: SoftTrackBatch, frame: CoordinateFrameTensor, condition: PendulumCondition) -> PhysicalStateBatch`; `extract_circular_state(track: SoftTrackBatch, frame: CoordinateFrameTensor, condition: CircularCondition) -> PhysicalStateBatch`; `soft_period_from_autocorrelation(signal: torch.Tensor, times_s: torch.Tensor, *, minimum_s: float, maximum_s: float, temperature: float) -> tuple[torch.Tensor, torch.Tensor]`; `unwrap_phase_straight_through(phase: torch.Tensor) -> torch.Tensor`.
- State schemas: pendulum trajectory `theta_rad, omega_rad_s, amplitude_envelope_rad` has units `rad, rad/s, rad`, summary `period_s` has unit `s`, event is `turning_point`, and validity is `pivot_drift, swing_continuity, visibility`; circular trajectory repeats `radius_m, phase_relative_rad, omega_rad_s, centripetal_acceleration_m_s2` with units `m, rad, rad/s, m/s^2` for each object, summary `period_s` has unit `s` per object, and validity is `center_drift, visibility, phase_continuity`.

- [ ] **Step 1: Write synthetic sinusoid and circular-orbit tests**

```python
class PendulumStateTests(unittest.TestCase):
    def test_signed_angle_amplitude_and_period(self) -> None:
        track, frame, condition = synthetic_pendulum(
            amplitude_rad=0.3, period_s=1.4, frames=121
        )
        state = extract_pendulum_state(track, frame, condition)
        self.assertAlmostEqual(0.3, float(state.trajectory[0, :, 2].max()), places=2)
        self.assertAlmostEqual(1.4, float(state.summaries[0, 0]), places=2)
        self.assertLess(float(state.trajectory[0, 0, 0]), 0.0)

class CircularStateTests(unittest.TestCase):
    def test_radius_response_preserves_omega_and_period(self) -> None:
        state_a = extract_circular_state(*synthetic_circle(radius_m=0.10, omega=2.0))
        state_b = extract_circular_state(*synthetic_circle(radius_m=0.20, omega=2.0))
        self.assertAlmostEqual(1.0, float((state_b.trajectory[..., 0] - state_a.trajectory[..., 0]).mean() / 0.10), places=2)
        self.assertAlmostEqual(0.0, float((state_b.trajectory[..., 2] - state_a.trajectory[..., 2]).abs().max()), places=3)

    def test_generated_disappearance_is_finite_and_cannot_mask_itself(self) -> None:
        state = extract_circular_state(*synthetic_circle_with_disappearance(start_frame=70))
        visibility_index = state.schema.validity_names.index("visibility")
        self.assertTrue(torch.isfinite(state.trajectory).all())
        self.assertTrue(torch.isfinite(state.summaries).all())
        self.assertLess(float(state.validity_values[0, visibility_index]), 0.0)
        self.assertTrue(torch.all(
            state.trajectory_validity_logit[0, 70:] < 0
        ))
```

- [ ] **Step 2: Run both scene tests and verify imports fail**

Run:

```bash
PYTHONPATH=src:tests:. /mnt/nvme1/NicoCache/envs/physics_wan/bin/python \
  -m unittest tests.physical_response.test_pendulum_state \
  tests.physical_response.test_circular_state -v
```

Expected: FAIL because the scene extractors do not exist.

- [ ] **Step 3: Implement fixed-pivot angles, phase alignment, and radius/phase states**

```python
def unwrap_phase_straight_through(phase: torch.Tensor) -> torch.Tensor:
    delta = phase[:, 1:] - phase[:, :-1]
    wrap_count = torch.round(delta.detach() / (2.0 * math.pi))
    corrected = delta - wrap_count * (2.0 * math.pi)
    return torch.cat((phase[:, :1], phase[:, :1] + corrected.cumsum(1)), dim=1)

def extract_circular_state(
    track: SoftTrackBatch,
    frame: CoordinateFrameTensor,
    condition: CircularCondition,
) -> PhysicalStateBatch:
    relative_px = track.centroid_xy_px - condition.center_xy_px[:, None, :, :]
    relative_m = project_vectors(relative_px, frame) * frame.meters_per_pixel
    radius = torch.linalg.vector_norm(relative_m, dim=-1)
    raw_phase = torch.atan2(relative_m[..., 1], relative_m[..., 0])
    phase = unwrap_phase_straight_through(raw_phase) - raw_phase[:, :1]
    times = physical_times(track.frame_indices, frame)
    omega = central_difference(phase, times)
    acceleration = omega.square() * radius
    period = 2.0 * math.pi / omega.abs().mean(1).clamp_min(1e-6)
    return circular_state_batch(radius, phase, omega, acceleration, period, track)
```

Pendulum angle uses `atan2(world_x, world_y)` about the frozen downward vertical. Its non-negative amplitude envelope is a temperature-controlled soft maximum of `abs(theta)` in a fixed odd window; turning weights combine small `abs(omega)` with non-zero amplitude; period is the softmax expectation over policy-bounded autocorrelation lags. Phase-aligned samples use differentiable linear interpolation on `phase = 2*pi*t/period`; no raw frame-index difference is used as the primary response. A generated disappearance leaves finite state sentinels and negative trajectory-validity logits; eligibility comes only from the real teacher, so generated visibility can never suppress its own validity loss.

- [ ] **Step 4: Run both tests plus finite-gradient checks**

Run:

```bash
PYTHONPATH=src:tests:. /mnt/nvme1/NicoCache/envs/physics_wan/bin/python \
  -m unittest tests.physical_response.test_pendulum_state \
  tests.physical_response.test_circular_state -v
```

Expected: all tests PASS and synthetic centroids receive finite non-zero gradients.

- [ ] **Step 5: Commit pendulum and circular extractors**

```bash
git add -- src/physbench/physical_response/state_extractors/__init__.py \
  src/physbench/physical_response/state_extractors/pendulum.py \
  src/physbench/physical_response/state_extractors/circular.py \
  tests/physical_response/test_pendulum_state.py \
  tests/physical_response/test_circular_state.py
git commit -m "feat: extract pendulum and circular states"
```

### Task 7: Extract inclined-plane and projectile states

**Files:**
- Create: `src/physbench/physical_response/state_extractors/inclined_plane.py`
- Create: `src/physbench/physical_response/state_extractors/projectile.py`
- Create: `tests/physical_response/test_inclined_plane_state.py`
- Create: `tests/physical_response/test_projectile_state.py`

**Interfaces:**
- Consumes: common state/coordinate functions from Task 4.
- Produces: `InclinedPlaneCondition(track_start_m: torch.Tensor, track_end_m: torch.Tensor, contact_cross_track_m: torch.Tensor)`; `ProjectileCondition(launch_xy_m: torch.Tensor, ground_y_m: torch.Tensor)`; `extract_inclined_plane_state(track: SoftTrackBatch, frame: CoordinateFrameTensor, condition: InclinedPlaneCondition) -> PhysicalStateBatch`; `extract_projectile_state(track: SoftTrackBatch, frame: CoordinateFrameTensor, condition: ProjectileCondition) -> PhysicalStateBatch`; `soft_first_crossing(value: torch.Tensor, threshold: torch.Tensor, *, direction: Literal["rising", "falling"], temperature: float) -> torch.Tensor`.
- State schemas: incline trajectory `along_displacement_m, cross_displacement_m, speed_m_s, acceleration_m_s2, contact_probability` has units `m, m, m/s, m/s^2, 1`, summary `descent_time_s` has unit `s`, events are `motion_onset, track_exit`, and validity is `visibility, contact, cross_track`; projectile trajectory `x_displacement_m, y_displacement_m, vx_m_s, vy_m_s` has units `m, m, m/s, m/s`, summary `landing_time_s` has unit `s`, event is `landing`, and validity is `visibility, before_landing, landing_present`.

- [ ] **Step 1: Write known-line and known-parabola tests**

```python
class InclinedPlaneStateTests(unittest.TestCase):
    def test_constant_acceleration_is_recovered_in_physical_time(self) -> None:
        state = extract_inclined_plane_state(*synthetic_incline(acceleration=1.75))
        interior = state.trajectory[0, 3:-3, 3]
        self.assertTrue(torch.allclose(interior, torch.full_like(interior, 1.75), atol=0.04))
        self.assertLess(float(state.trajectory[0, :, 1].abs().max()), 1e-4)

class ProjectileStateTests(unittest.TestCase):
    def test_horizontal_velocity_and_landing_are_differentiable(self) -> None:
        track, frame, condition = synthetic_projectile(vx=0.8, flight_time=0.6)
        track.centroid_xy_px.requires_grad_()
        state = extract_projectile_state(track, frame, condition)
        self.assertAlmostEqual(0.8, float(state.trajectory[0, 2:-2, 2].mean()), places=2)
        self.assertAlmostEqual(0.6, float(state.summaries[0, 0]), places=2)
        state.summaries.sum().backward()
        self.assertGreater(float(track.centroid_xy_px.grad.abs().sum()), 0.0)
```

- [ ] **Step 2: Run the scene tests and verify missing extractors**

Run:

```bash
PYTHONPATH=src:tests:. /mnt/nvme1/NicoCache/envs/physics_wan/bin/python \
  -m unittest tests.physical_response.test_inclined_plane_state \
  tests.physical_response.test_projectile_state -v
```

Expected: FAIL importing the two extractor modules.

- [ ] **Step 3: Implement physical-time derivatives and soft events**

```python
def soft_first_crossing(
    value: torch.Tensor,
    threshold: torch.Tensor,
    *,
    direction: Literal["rising", "falling"],
    temperature: float,
) -> torch.Tensor:
    signed = value - threshold
    if direction == "falling":
        signed = -signed
    entered = torch.sigmoid(signed / temperature)
    not_previously_entered = torch.cumprod(
        torch.cat((torch.ones_like(entered[:, :1]), 1.0 - entered[:, :-1]), 1),
        dim=1,
    )
    weights = entered * not_previously_entered
    return weights / weights.sum(1, keepdim=True).clamp_min(1e-6)
```

Incline projection uses the frozen down-slope axis and subtracts frame-zero position; speed and acceleration use physical times; motion-onset and track-exit weights define the common sliding window; contact probability penalizes visibility loss, excessive cross-track distance, and implausible area change. Projectile landing combines ground crossing with downward velocity. State values remain present when generated confidence is low, while the validity logits expose disappearance or missing landing to the loss.

- [ ] **Step 4: Run tests including encoded-speed scaling**

Run:

```bash
PYTHONPATH=src:tests:. /mnt/nvme1/NicoCache/envs/physics_wan/bin/python \
  -m unittest tests.physical_response.test_inclined_plane_state \
  tests.physical_response.test_projectile_state -v
```

Expected: all tests PASS; halving `encoded_to_physical_speed` doubles physical time and changes derivatives accordingly.

- [ ] **Step 5: Commit incline and projectile extractors**

```bash
git add src/physbench/physical_response/state_extractors/inclined_plane.py \
  src/physbench/physical_response/state_extractors/projectile.py \
  tests/physical_response/test_inclined_plane_state.py \
  tests/physical_response/test_projectile_state.py
git commit -m "feat: extract incline and projectile states"
```

### Task 8: Extract two-ball and three-ball collision states

**Files:**
- Create: `src/physbench/physical_response/state_extractors/collision.py`
- Create: `tests/physical_response/test_collision_state.py`

**Interfaces:**
- Consumes: common state/coordinate functions from Task 4 and frozen ball-channel mappings from the overlay.
- Produces: `CollisionCondition(mass_kg: torch.Tensor, radius_m: torch.Tensor, track_y_m: torch.Tensor)` with `[B,O]` fields; `extract_collision_state(track: SoftTrackBatch, frame: CoordinateFrameTensor, condition: CollisionCondition) -> PhysicalStateBatch`; `soft_pair_contact_weights(position_m: torch.Tensor, radius_m: torch.Tensor, velocity_m_s: torch.Tensor, *, gap_temperature_m: float, approach_temperature_m_s: float) -> torch.Tensor`; `soft_pre_post_windows(contact_weights: torch.Tensor, separation_probability: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]`.
- The schema is object-count-specific and ordered by frozen identity: trajectory `x_m.ball_i, vx_m_s.ball_i` has units `m, m/s` per ball; summaries `pre_v.ball_i, post_v.ball_i, contact_time_s.pair_ij, momentum_pre, momentum_post, kinetic_energy_pre, kinetic_energy_post` have units `m/s, m/s, s, kg*m/s, kg*m/s, J, J` in the same expanded order; events are ordered pair contacts; validity includes visibility, identity-channel retention, nonpenetration, contact present, terminal separation, and event ordering.

- [ ] **Step 1: Write elastic two-ball and missing-terminal three-ball tests**

```python
class CollisionStateTests(unittest.TestCase):
    def test_contact_alignment_recovers_signed_pre_and_post_velocities(self) -> None:
        state = extract_collision_state(*synthetic_elastic_two_ball(
            masses=(1.0, 2.0), incident=(1.0, 0.0)
        ))
        self.assertTensorClose(state.summaries[0, :2], torch.tensor([1.0, 0.0]), atol=0.03)
        self.assertTensorClose(state.summaries[0, 2:4], torch.tensor([-1/3, 2/3]), atol=0.04)

    def test_three_ball_without_terminal_separation_marks_invalid(self) -> None:
        state = extract_collision_state(*synthetic_three_ball(no_terminal_window=True))
        terminal_index = state.schema.validity_names.index("terminal_separation")
        self.assertLess(float(state.validity_values[0, terminal_index]), 0.0)
        terminal_summary_indices = terminal_collision_summary_indices(state.schema)
        self.assertTrue(torch.isfinite(state.summaries).all())
        self.assertTrue(torch.all(
            state.summary_validity_logit[0, terminal_summary_indices] < 0
        ))

    def test_channel_identity_never_follows_late_left_right_order(self) -> None:
        state = extract_collision_state(*crossing_two_ball_tracks())
        self.assertGreater(float(state.trajectory[0, -1, 0]), float(state.trajectory[0, -1, 2]))

    def test_mid_sequence_channel_swap_fails_identity_without_relabeling(self) -> None:
        state = extract_collision_state(*two_ball_tracks_with_injected_channel_swap(frame=61))
        identity_index = state.schema.validity_names.index("identity_channel_retention")
        self.assertTrue(torch.isfinite(state.trajectory).all())
        self.assertLess(float(state.validity_values[0, identity_index]), 0.0)
        self.assertEqual(
            fixture_frozen_channel_order(), state.schema.trajectory_names[::2]
        )

    def test_generated_interpenetration_activates_nonpenetration_failure(self) -> None:
        state = extract_collision_state(*penetrating_two_ball_tracks(overlap_m=0.03))
        index = state.schema.validity_names.index("nonpenetration")
        self.assertTrue(torch.isfinite(state.trajectory).all())
        self.assertLess(float(state.validity_values[0, index]), 0.0)
```

- [ ] **Step 2: Run collision-state tests and observe the missing module**

Run:

```bash
PYTHONPATH=src:tests:. /mnt/nvme1/NicoCache/envs/physics_wan/bin/python \
  -m unittest tests.physical_response.test_collision_state -v
```

Expected: FAIL importing `state_extractors.collision`.

- [ ] **Step 3: Implement identity-stable tracks and soft contact windows**

```python
def soft_pair_contact_weights(
    position_m: torch.Tensor,
    radius_m: torch.Tensor,
    velocity_m_s: torch.Tensor,
    *,
    gap_temperature_m: float,
    approach_temperature_m_s: float,
) -> torch.Tensor:
    pair_indices = tuple(itertools.combinations(range(position_m.shape[2]), 2))
    weights: list[torch.Tensor] = []
    for left, right in pair_indices:
        gap = (position_m[:, :, left] - position_m[:, :, right]).abs()
        gap = gap - radius_m[:, None, left] - radius_m[:, None, right]
        approaching = torch.sigmoid(
            -(velocity_m_s[:, :, right] - velocity_m_s[:, :, left])
            / approach_temperature_m_s
        )
        weights.append(torch.exp(-gap.square() / gap_temperature_m**2) * approaching)
    return torch.stack(weights, dim=-1)
```

Pre-contact weights are the normalized soft survival mass before the first contact; post weights require both contact history and positive, increasing separation. The distance and approach temperatures have distinct physical units and are required sealed event parameters. Frozen channel identities are never reassigned from generated left/right order: a mid-sequence swap keeps the original channel schema and activates identity-retention failure. Center distance below the frozen sum of radii activates nonpenetration failure rather than being repaired. Two-ball summaries are eligible for response and strong analytic terms. Three-ball summaries compute global momentum/energy only in stable pre-first-contact and post-final-contact/separation windows; absent terminal support stores a finite zero sentinel with a false summary-validity logit, never publishes it as a velocity, and keeps the missing-event validity term active. Task 11 applies `torch.where` before arithmetic, so the sentinel cannot enter response or conservation residuals.

- [ ] **Step 4: Run collision tests and identity/gradient checks**

Run:

```bash
PYTHONPATH=src:tests:. /mnt/nvme1/NicoCache/envs/physics_wan/bin/python \
  -m unittest tests.physical_response.test_collision_state -v
```

Expected: all tests PASS; both ball channels receive finite gradients in the two-ball fixture.

- [ ] **Step 5: Commit collision state extraction**

```bash
git add src/physbench/physical_response/state_extractors/collision.py \
  tests/physical_response/test_collision_state.py
git commit -m "feat: extract elastic collision states"
```

### Task 9: Build the immutable real-video teacher cache

**Files:**
- Create: `src/physbench/physical_response/teacher_cache.py`
- Create: `scripts/build_physical_response_teacher_cache.py`
- Create: `schemas/physical_response_teacher_binding.schema.json`
- Create: `tests/physical_response/test_teacher_cache.py`

**Interfaces:**
- Consumes: the overlay artifacts from Task 3; sealed `CoordinateFrameIndex` from Task 4; SAM2 identity/observer from Task 5; all scene extractors from Tasks 6-8; first-frame entity/mask manifests materialized through `src/physbench/evaluation/common/entities/manifest.py:1342-1405`; Dataset reference-video, first-frame, PNG, and NPZ bytes.
- Produces: `TeacherCacheKey.from_inputs(*, case_id: str, coordinate_frame_id: str, input_inventory: TreeInventory, dataset_metadata_digest: str, dataset_tree_digest: str, policy_digest: str, provenance_digests: Mapping[str, str], observer_identity: SAM2ObserverIdentity, response_context_digest: str, coordinate_manifest_digest: str, coordinate_digest: str, extractor_version: str) -> TeacherCacheKey`; `build_teacher_record(case: Mapping[str, Any], *, expected_key: TeacherCacheKey, edge_contexts: Sequence[ResponseEdge], observer: DifferentiableSAM2Observer, coordinate_index: CoordinateFrameIndex, coordinate_frame_id: str, condition_record_digest: str, output_root: Path) -> TeacherRecord`; `aggregate_replicate_targets(records: Sequence[TeacherRecord]) -> ReplicateTarget`; `audit_canonical_state_reproducibility(reference: TeacherRecord, candidate: TeacherRecord, *, deterministic_kernels: bool, numeric_tolerance: float) -> CanonicalStateReproducibilityAudit`; `verify_teacher_record(path: Path, expected_key: TeacherCacheKey) -> TeacherRecord`; `write_partition_bindings(records: Sequence[TeacherRecord], manifest: ResponseManifest, *, cache_root: Path, output_dir: Path) -> Mapping[Partition, Path]`; `load_teacher_cache_binding_header(binding_path: Path, *, expected_partition: Partition, expected_response_manifest_digest: str, expected_coordinate_manifest_digest: str, expected_observer_identity_digest: str) -> TeacherCacheBinding`; `open_teacher_cache_view(binding_path: Path, *, cache_root: Path, allowed_partitions: frozenset[Partition], expected_response_manifest_digest: str, expected_coordinate_manifest_digest: str, expected_observer_identity_digest: str) -> TeacherCacheView`; `TeacherCacheView.get(case_id: str, coordinate_frame_id: str) -> TeacherRecord`; CLI `build_physical_response_teacher_cache.py --manifest run/physical_response_group_holdout_v1_seed42/physical_response/manifest.json --condition-index run/physical_response_group_holdout_v1_seed42/physical_response/condition_index.jsonl --coordinate-binding run/physical_response_group_holdout_v1_seed42/physical_response/coordinates/binding.json --coordinate-cache-root cache/physical_response/coordinates_v1 --teacher-policy run/physical_response_group_holdout_v1_seed42/frozen/physical_response_policy.json --sam2-root /root/Nico/third_party/sam2 --sam2-checkpoint /mnt/nvme1/NicoCache/checkpoints/sam2/sam2.1_hiera_tiny.pt --cache-root cache/physical_response/teacher_v1 --output-dir run/physical_response_group_holdout_v1_seed42/physical_response/teacher_cache`.
- Storage contract: `cache/physical_response/teacher_v1/sha256/{teacher_cache_key_sha256}/manifest.json` plus deterministic `.npy` arrays. A run receives three disjoint content-hashed binding manifests; the training process receives only the `response_train` binding.
- Data contracts: `TeacherCacheKey(case_id: str, coordinate_frame_id: str, input_inventory_digest: str, dataset_metadata_digest: str, dataset_tree_digest: str, policy_digest: str, provenance_digests: tuple[tuple[str, str], ...], observer_identity_digest: str, response_context_digest: str, coordinate_manifest_digest: str, coordinate_digest: str, extractor_version: str, digest: str)`; `TeacherRecord(case_id: str, partition: Partition, key: TeacherCacheKey, state_schema: StateSchema, array_paths: tuple[tuple[str, str], ...], soft_mask_confidence_path: str, hard_mask_confidence_path: str, identity_confidence_path: str, visibility_path: str, observer_sigma_path: str, coordinate_frame_id: str, coordinate_digest: str, eligibility_path: str, failure_reasons: tuple[str, ...], array_digests: tuple[tuple[str, str], ...], digest: str)`; `TeacherRecordRef(case_id: str, coordinate_frame_id: str, partition: Partition, cache_entry_relative_path: str, teacher_key_digest: str, teacher_record_digest: str)`; `TeacherCacheBinding(schema_version: Literal["physical_response_teacher_binding_v1"], partition: Partition, response_manifest_digest: str, dataset_metadata_digest: str, dataset_tree_digest: str, policy_digest: str, observer_identity_digest: str, coordinate_manifest_digest: str, record_refs: tuple[TeacherRecordRef, ...], record_count: int, digest: str)`; `ReplicateTarget(mean: torch.Tensor, measured_sigma: torch.Tensor, uncertainty_available: torch.Tensor, valid: torch.Tensor)`; `CanonicalStateReproducibilityAudit(reference_record_digest: str, candidate_record_digest: str, deterministic_kernels: bool, numeric_tolerance: float, maximum_absolute_difference: float, exact: bool, passed: bool, failure_reasons: tuple[str, ...], digest: str)`; and `TeacherCacheView(binding: TeacherCacheBinding, cache_root: Path, allowed_partitions: frozenset[Partition], refs_by_case_frame: Mapping[tuple[str, str], TeacherRecordRef], opened_case_ids: tuple[str, ...])`. The header loader validates only the binding document and its upstream identities; it has no cache root and therefore cannot resolve or open a record. The view resolves a record lazily on `get`, contains every resolved path beneath its explicit `cache_root`, verifies the referenced record and arrays, and memoizes only the verified composite key.

- [ ] **Step 1: Write cache identity, determinism, conflict, and partition tests**

```python
class TeacherCacheTests(unittest.TestCase):
    def test_key_changes_when_any_referenced_mask_byte_changes(self) -> None:
        first = teacher_key_for_fixture()
        mutate_referenced_mask_byte()
        second = teacher_key_for_fixture()
        self.assertNotEqual(first.digest, second.digest)

    def test_identical_build_has_identical_manifest_and_arrays(self) -> None:
        first = build_fixture_teacher_cache()
        second = build_fixture_teacher_cache()
        self.assertEqual(first.manifest_bytes, second.manifest_bytes)
        self.assertEqual(first.array_digests, second.array_digests)

    def test_train_view_cannot_open_validation_or_test_record(self) -> None:
        view = open_teacher_cache_view(
            fixture_train_binding(), cache_root=fixture_teacher_cache_root(),
            allowed_partitions=frozenset({"response_train"}),
            **fixture_teacher_binding_identity_expectations(),
        )
        with self.assertRaisesRegex(PermissionError, "partition is not allowed"):
            view.get(
                "response_test_case",
                fixture_coordinate_frame_id("response_test_case"),
            )

    def test_single_replicate_uses_measured_observer_uncertainty(self) -> None:
        target = aggregate_replicate_targets([
            fixture_teacher_record(observer_sigma=0.03)
        ])
        self.assertTrue(bool(target.uncertainty_available.all()))
        self.assertTensorClose(target.measured_sigma, torch.tensor([0.03], dtype=torch.float64))

    def test_replicate_uncertainty_combines_observer_and_robust_between_repeat_variance(self) -> None:
        records = fixture_teacher_replicates(
            values=(1.00, 1.04, 0.98, 1.02, 4.50), observer_sigma=0.02
        )
        target = aggregate_replicate_targets(records)
        self.assertAlmostEqual(1.01, float(target.mean), delta=0.04)
        self.assertGreater(float(target.measured_sigma), 0.02 / math.sqrt(5.0))
        self.assertLess(float(target.measured_sigma), 0.20)
        self.assertTrue(bool(target.uncertainty_available))

    def test_key_changes_with_coordinate_or_observer_runtime(self) -> None:
        base = teacher_key_for_fixture()
        self.assertNotEqual(base.digest, teacher_key_for_fixture(coordinate_shift_px=0.25).digest)
        self.assertNotEqual(base.digest, teacher_key_for_fixture(mask_temperature=0.8).digest)
        self.assertNotEqual(base.digest, teacher_key_for_fixture(detach_positions=(7,)).digest)

    def test_coordinate_manifest_and_case_digest_are_verified_before_observation(self) -> None:
        with self.assertRaisesRegex(ValueError, "coordinate manifest digest"):
            build_fixture_teacher_cache(coordinate_manifest_digest="tampered")
        with self.assertRaisesRegex(ValueError, "condition record digest"):
            build_fixture_teacher_cache(condition_record_digest="tampered")

    def test_record_requires_soft_hard_identity_and_visibility_confidence(self) -> None:
        record = build_fixture_teacher_cache().record
        for name in (
            "soft_mask_confidence_path", "hard_mask_confidence_path",
            "identity_confidence_path", "visibility_path",
        ):
            self.assertTrue(verify_array_path_and_digest(record, name))

    def test_binding_header_does_not_open_records_and_view_get_is_lazy(self) -> None:
        record_reader = mock.Mock(wraps=read_verified_teacher_record)
        with mock.patch(
            "physbench.physical_response.teacher_cache.read_verified_teacher_record",
            record_reader,
        ):
            binding = load_teacher_cache_binding_header(
                fixture_train_binding(), expected_partition="response_train",
                **fixture_teacher_binding_identity_expectations(),
            )
            record_reader.assert_not_called()
            view = open_teacher_cache_view(
                fixture_train_binding(), cache_root=fixture_teacher_cache_root(),
                allowed_partitions=frozenset({"response_train"}),
                **fixture_teacher_binding_identity_expectations(),
            )
            record_reader.assert_not_called()
            ref = binding.record_refs[0]
            loaded = view.get(ref.case_id, ref.coordinate_frame_id)
            self.assertEqual(ref.teacher_record_digest, loaded.digest)
            record_reader.assert_called_once()

    def test_canonical_state_reproducibility_uses_exact_or_frozen_tolerance(self) -> None:
        deterministic_a, deterministic_b = build_fixture_teacher_cache_pair(
            deterministic_kernels=True
        )
        exact = audit_canonical_state_reproducibility(
            deterministic_a.record, deterministic_b.record,
            deterministic_kernels=True, numeric_tolerance=0.0,
        )
        self.assertTrue(exact.exact)
        self.assertTrue(exact.passed)
        nondeterministic_a, nondeterministic_b = build_fixture_teacher_cache_pair(
            deterministic_kernels=False
        )
        tolerant = audit_canonical_state_reproducibility(
            nondeterministic_a.record, nondeterministic_b.record,
            deterministic_kernels=False,
            numeric_tolerance=fixture_observer_policy().canonical_state_tolerance,
        )
        self.assertFalse(tolerant.exact)
        self.assertTrue(tolerant.passed)

    def test_three_ball_train_case_is_cached_without_response_context(self) -> None:
        record = build_three_ball_teacher_record(edge_contexts=())
        self.assertEqual("response_train", record.partition)
        self.assertIn("momentum_pre", record.state_schema.summary_names)
        self.assertIn("kinetic_energy_post", record.state_schema.summary_names)
```

- [ ] **Step 2: Run cache tests and verify the missing implementation**

Run:

```bash
PYTHONPATH=src:tests:.:/root/Nico/third_party/sam2 \
  /mnt/nvme1/NicoCache/envs/physics_wan/bin/python \
  -m unittest tests.physical_response.test_teacher_cache -v
```

Expected: FAIL importing `physbench.physical_response.teacher_cache`.

- [ ] **Step 3: Implement full input binding and deterministic robust targets**

```python
def aggregate_replicate_targets(
    records: Sequence[TeacherRecord],
) -> ReplicateTarget:
    values, observer_sigma, valid = stack_common_teacher_states(records)
    location = values.nanmedian(dim=0).values
    residual = values - location
    mad = 1.4826 * residual.abs().nanmedian(dim=0).values
    replicate_count = valid.sum(dim=0)
    for _iteration in range(5):
        scale = mad.clamp_min(torch.finfo(torch.float64).eps)
        weight = (1.345 * scale / residual.abs().clamp_min(scale)).clamp_max(1.0)
        weight = weight * valid
        location = (weight * values.nan_to_num()).sum(0) / weight.sum(0).clamp_min(1.0)
        residual = values - location
    between_standard_error = torch.where(
        replicate_count >= 2,
        mad / replicate_count.clamp_min(1).sqrt(),
        torch.zeros_like(mad),
    )
    observer_variance = torch.where(
        valid & torch.isfinite(observer_sigma), observer_sigma.square(), 0.0
    ).sum(0) / replicate_count.clamp_min(1).square()
    measured_sigma = torch.sqrt(between_standard_error.square() + observer_variance)
    uncertainty_available = (
        (replicate_count >= 2)
        | (valid & torch.isfinite(observer_sigma)).any(0)
    )
    return ReplicateTarget(
        mean=location,
        measured_sigma=torch.where(uncertainty_available, measured_sigma, 0.0),
        uncertainty_available=uncertainty_available,
        valid=valid.any(0),
    )
```

The key inventory binds the reference video, first frame, mask manifest and every referenced PNG/NPZ, the separately named Dataset metadata and Dataset-tree identities, provenance and policy identities, the complete realized `SAM2ObserverIdentity` (SAM2 commit/config/checkpoint/preprocessing, `SAM2TrackConfig`, dtype, runtime overrides, post-load mutations, and detach positions), the coordinate-manifest and per-case coordinate-record identities, extractor versions, axis/group context, and signed-projection version. `build_teacher_record` first loads the case coordinate through `CoordinateFrameIndex.get`, verifies the condition-record and manifest identities, then verifies that `expected_key` binds the exact case, response context, observer identity, and coordinate before it chooses a cache path. Cache construction snapshots the complete `datasets/` tree before observation and after finalization, records missing first-frame masks as physical-ineligible while leaving those cases available to ordinary FlowMatch, writes staging records with per-array SHA-256, claims final directories exclusively, and accepts a collision only after byte verification. Required arrays include per-frame soft and hard mask confidence, identity confidence, visibility, state, eligibility, and uncertainty. Three-ball train cases are cached and bound even though their response context contains zero edges. Each record estimates observer/calibration uncertainty by a sealed temporal block bootstrap. Replicate aggregation combines that within-record variance with robust between-replicate variance; a singleton can therefore have measured uncertainty, while a component lacking both sources remains `uncertainty_available=false`. The robust replicate fixture includes a deliberate outlier. Canonical arrays from deterministic kernels must be element-for-element equal; when the frozen observer policy explicitly permits a nondeterministic kernel, `CanonicalStateReproducibilityAudit` compares the canonical FP32 state arrays against its pre-test numeric tolerance and fails on any excess. The response-train floor in Task 11 is only a numerical normalization floor and is never reported as measured uncertainty.

For binding assignment, every case appearing in a partitioned response level uses that unique cross-axis partition; a View-A-train case with no response-group membership defaults only to `response_train`. View-A test is never bound. A missing-mask record may retain explicit failure metadata but has no physical arrays and is not returned by `build_factual_target`. `write_partition_bindings` emits one typed `TeacherCacheBinding` per partition with unique `(case_id, coordinate_frame_id)` refs. Header verification never reads a record; `TeacherCacheView.get` is the only record-opening API, rejects any ref outside the allowed partition, resolves paths only beneath the explicit cache root, and re-verifies the record's key, composite key, partition, coordinate, schema, and arrays before returning it.

- [ ] **Step 4: Run cache tests and a no-Dataset-write dry build**

Run:

```bash
PYTHONPATH=src:tests:.:/root/Nico/third_party/sam2 \
  /mnt/nvme1/NicoCache/envs/physics_wan/bin/python \
  -m unittest tests.physical_response.test_teacher_cache -v
PYTHONPATH=src:.:/root/Nico/third_party/sam2 \
  /mnt/nvme1/NicoCache/envs/physics_wan/bin/python \
  scripts/build_physical_response_teacher_cache.py --help
```

Expected: tests PASS; help exits zero; `git status --short datasets` remains empty.

- [ ] **Step 5: Commit the teacher cache**

```bash
git add src/physbench/physical_response/teacher_cache.py \
  scripts/build_physical_response_teacher_cache.py \
  schemas/physical_response_teacher_binding.schema.json \
  tests/physical_response/test_teacher_cache.py
git commit -m "feat: cache sealed real video teachers"
```

### Task 10: Reproduce FlowMatch with injected randomness and decode full videos differentiably

**Files:**
- Create: `src/physbench/physical_response/flow_core.py`
- Create: `src/physbench/physical_response/vae_decode.py`
- Create: `tests/physical_response/test_flow_core.py`
- Create: `tests/physical_response/test_vae_decode.py`

**Interfaces:**
- Consumes: vendor `FlowMatchSFTLoss` at `/root/Steven/wan22_pendulum_pipeline/vendor/DiffSynth-Studio/diffsynth/diffusion/loss.py:5-33`; scheduler equations at `diffsynth/diffusion/flow_match.py:347-362`; `WanVideoVAE.model.decode` semantics at `diffsynth/models/wan_video_vae.py:1011-1034`; TI2V first-frame fusion at `diffsynth/pipelines/wan_video.py:512-530`.
- Produces: `FlowCoreOutput`; `flow_match_core(pipe: Any, processed_inputs: Mapping[str, Any], *, timestep_id: torch.Tensor, noise: torch.Tensor) -> FlowCoreOutput`; `eligible_physical_timestep_ids(scheduler: Any, *, sigma_min: float, sigma_max: float) -> torch.Tensor`; `sample_factual_timestep_id(scheduler: Any, *, minimum_fraction: float, maximum_fraction: float, generator: torch.Generator) -> torch.Tensor`; `decode_full_video_differentiable(vae: Any, z0_hat_full: torch.Tensor, *, device: torch.device, checkpoint_full_decode: bool) -> torch.Tensor`.
- `FlowCoreOutput` fields are `fm_loss`, `timestep_id`, `timestep`, `sigma`, `noise`, `z_t_full`, `v_pred_full`, `v_pred_generated`, `v_target_generated`, `z0_hat_generated`, and `z0_hat_full`. Latents use `[B,C,T_lat,H_lat,W_lat]`; index zero is the conditioned TI2V latent and receives no physical gradient.

- [ ] **Step 1: Write fixed-noise parity, reconstruction, first-frame, and VAE temporal tests**

```python
class FlowCoreTests(unittest.TestCase):
    def test_perfect_velocity_target_recovers_generated_clean_latent(self) -> None:
        pipe, inputs, timestep_id, noise = deterministic_fake_flow_case()
        pipe.model_fn = lambda **kwargs: noise - inputs["input_latents"]
        output = flow_match_core(
            pipe, inputs, timestep_id=timestep_id, noise=noise
        )
        self.assertTrue(torch.allclose(
            output.z0_hat_generated,
            inputs["input_latents"][:, :, 1:], atol=1e-6,
        ))

    def test_conditioned_first_latent_is_detached_and_unchanged(self) -> None:
        inputs, output = run_fake_flow_core(first_frame_requires_grad=True)
        first_frame_source = inputs["first_frame_latents"]
        output.z0_hat_full.sum().backward()
        self.assertIsNone(first_frame_source.grad)
        self.assertTensorEqual(
            output.z0_hat_full[:, :, :1], first_frame_source.detach()
        )

class VaeDecodeTests(unittest.TestCase):
    def test_selected_latents_cannot_be_decoded_as_independent_frames(self) -> None:
        vae, latents = causal_fixture_vae_and_latents()
        full = decode_full_video_differentiable(
            vae, latents, device=torch.device("cpu"), checkpoint_full_decode=False
        )
        independent = forbidden_independent_frame_decode(vae, latents)
        self.assertGreater(float((full[:, 1:] - independent[:, 1:]).abs().max()), 1e-4)
```

- [ ] **Step 2: Run focused tests and observe missing flow/decode modules**

Run:

```bash
PYTHONPATH=src:tests:.:/root/Steven/wan22_pendulum_pipeline/vendor/DiffSynth-Studio \
  /mnt/nvme1/NicoCache/envs/physics_wan/bin/python -m unittest \
  tests.physical_response.test_flow_core \
  tests.physical_response.test_vae_decode -v
```

Expected: FAIL importing `flow_core` and `vae_decode`.

- [ ] **Step 3: Implement a pure flow core and functional full-causal VAE decode**

```python
@dataclass(slots=True)
class FlowCoreOutput:
    fm_loss: torch.Tensor
    timestep_id: torch.Tensor
    timestep: torch.Tensor
    sigma: torch.Tensor
    noise: torch.Tensor
    z_t_full: torch.Tensor
    v_pred_full: torch.Tensor
    v_pred_generated: torch.Tensor
    v_target_generated: torch.Tensor
    z0_hat_generated: torch.Tensor
    z0_hat_full: torch.Tensor

def flow_match_core(
    pipe: Any,
    processed_inputs: Mapping[str, Any],
    *,
    timestep_id: torch.Tensor,
    noise: torch.Tensor,
) -> FlowCoreOutput:
    inputs = clone_tensor_mapping(processed_inputs)
    timestep = pipe.scheduler.timesteps[timestep_id].to(
        dtype=pipe.torch_dtype, device=pipe.device
    )
    sigma = pipe.scheduler.sigmas[timestep_id].to(
        dtype=inputs["input_latents"].dtype,
        device=inputs["input_latents"].device,
    ).reshape(-1, 1, 1, 1, 1)
    z_t = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
    target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)
    first = inputs["first_frame_latents"].detach()
    z_t_full = torch.cat((first, z_t[:, :, 1:]), dim=2)
    inputs["latents"] = z_t_full
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    v_full = pipe.model_fn(**models, **inputs, timestep=timestep)
    v_generated = v_full[:, :, 1:]
    target_generated = target[:, :, 1:]
    fm_loss = F.mse_loss(v_generated.float(), target_generated.float())
    fm_loss = fm_loss * pipe.scheduler.training_weight(timestep)
    z0_generated = z_t_full[:, :, 1:] - sigma * v_generated
    z0_full = torch.cat((first, z0_generated), dim=2)
    return make_flow_output(
        fm_loss, timestep_id, timestep, sigma, noise, z_t_full, v_full,
        v_generated, target_generated, z0_generated, z0_full,
    )
```

`decode_full_video_differentiable` reproduces `VideoVAE_.decode`: unscale once, run `conv2` on the complete latent sequence, carry a function-local causal feature cache through every latent index, concatenate decoder outputs, clamp to `[-1,1]`, map to `[0,1]`, and return `[B,T,3,H,W]`. Full-decode checkpointing wraps that complete pure tensor function with `use_reentrant=False`; it never calls `decode_framewise` and never lets the vendor module's mutable `_feat_map` escape between calls.

- [ ] **Step 4: Prove vendor scalar/LoRA-gradient parity and VAE checkpoint parity**

Run:

```bash
PYTHONPATH=src:tests:.:/root/Steven/wan22_pendulum_pipeline/vendor/DiffSynth-Studio \
  /mnt/nvme1/NicoCache/envs/physics_wan/bin/python -m unittest \
  tests.physical_response.test_flow_core \
  tests.physical_response.test_vae_decode -v
```

Expected: fixed-randomness FM scalar and LoRA gradients match the vendor calculation within `rtol=1e-6, atol=1e-7`; ordinary and checkpointed full decode match forward outputs within `rtol=1e-5, atol=1e-6` and latent gradients within `rtol=1e-4, atol=1e-5`.

- [ ] **Step 5: Commit the flow and VAE core**

```bash
git add src/physbench/physical_response/flow_core.py \
  src/physbench/physical_response/vae_decode.py \
  tests/physical_response/test_flow_core.py \
  tests/physical_response/test_vae_decode.py
git commit -m "feat: expose physical response flow core"
```

### Task 11: Implement empirical, analytic, zero, and validity losses

**Files:**
- Create: `src/physbench/physical_response/mechanics.py`
- Create: `src/physbench/physical_response/losses.py`
- Create: `schemas/physical_response_uncertainty_floors.schema.json`
- Create: `scripts/fit_physical_response_uncertainty_floors.py`
- Create: `tests/physical_response/test_mechanics.py`
- Create: `tests/physical_response/test_losses.py`

**Interfaces:**
- Consumes: `PhysicalStateBatch` from Task 4, partition-scoped `TeacherCacheView` from Task 9, ordered `ResponseEdge` and `ResponseNormalization` from Task 1.
- Produces: `elastic_two_ball_post_velocity(mass_kg: torch.Tensor, incident_velocity_m_s: torch.Tensor) -> torch.Tensor`; `total_momentum(mass_kg: torch.Tensor, velocity_m_s: torch.Tensor) -> torch.Tensor`; `total_kinetic_energy(mass_kg: torch.Tensor, velocity_m_s: torch.Tensor) -> torch.Tensor`; `complete_elliptic_k_agm(parameter_m: torch.Tensor, *, iterations: int = 8) -> torch.Tensor`; `pendulum_period_s(string_length_m: torch.Tensor, bob_radius_m: torch.Tensor, initial_angle_rad: torch.Tensor, *, gravity_m_s2: float = 9.80665) -> torch.Tensor`; `incline_acceleration_m_s2(angle_rad: torch.Tensor, *, mu_k: float = 0.463, gravity_m_s2: float = 9.80665) -> torch.Tensor`; `incline_acceleration_derivative(angle_rad: torch.Tensor, *, mu_k: float = 0.463, gravity_m_s2: float = 9.80665) -> torch.Tensor`; `fit_train_uncertainty_floors(teacher_view: TeacherCacheView, *, record_keys: Sequence[tuple[str, str]], response_manifest_digest: str, teacher_train_binding_digest: str) -> UncertaintyFloors`; `write_uncertainty_floors(floors: UncertaintyFloors, output_path: Path) -> str`; `load_uncertainty_floors(path: Path, *, expected_response_manifest_digest: str, expected_teacher_train_binding_digest: str) -> UncertaintyFloors`; `build_factual_target(case_id: str, coordinate_frame_id: str, teacher_view: TeacherCacheView, uncertainty_floors: UncertaintyFloors) -> FactualTargetBatch`; `build_response_supervision(edge: ResponseEdge, teacher_view: TeacherCacheView, normalization: ResponseNormalization, uncertainty_floors: UncertaintyFloors, policy: ResponsePolicy) -> ResponseSupervision`; `write_analytic_eligibility_audit(records: Sequence[AnalyticEligibilityAudit], output_path: Path) -> str`; `compute_factual_physical_loss(prediction: PhysicalStateBatch, target: FactualTargetBatch, *, analytic_context: AnalyticContext, weights: LossWeights) -> PhysicalLossBreakdown`; `compute_physical_loss(prediction_a: PhysicalStateBatch, prediction_b: PhysicalStateBatch, target: ResponseTargetBatch, *, analytic_context: AnalyticContext, weights: LossWeights) -> PhysicalLossBreakdown`; CLI `fit_physical_response_uncertainty_floors.py --response-manifest run/physical_response_group_holdout_v1_seed42/physical_response/manifest.json --teacher-train-binding run/physical_response_group_holdout_v1_seed42/physical_response/teacher_train_bindings.json --teacher-cache-root cache/physical_response/teacher_v1 --output run/physical_response_group_holdout_v1_seed42/frozen/uncertainty_floors.json`.
- `PhysicalLossBreakdown` exposes scalar `total`, `absolute`, `sign`, `magnitude`, `zero`, `analytic`, `validity`, plus an eligibility count for every term. Empirical and analytic contributions remain distinct before weighting.
- Data contracts: `UncertaintyFloors(response_manifest_digest: str, teacher_train_binding_digest: str, component_units: tuple[tuple[str, str], ...], absolute_by_component: Mapping[str, float], response_by_axis_component: Mapping[str, float], source_partition: Literal["response_train"], digest: str)`; `FactualTargetBatch(case_id: str, absolute: torch.Tensor, absolute_sigma: torch.Tensor, absolute_uncertainty_floor: torch.Tensor, absolute_eligible: torch.Tensor, teacher_roles: tuple[TeacherRole, ...])`; `ResponseTargetBatch(edge_id: str, delta_q_tilde: torch.Tensor, absolute_a: torch.Tensor, absolute_b: torch.Tensor, absolute_sigma_a: torch.Tensor, absolute_sigma_b: torch.Tensor, absolute_uncertainty_floor: torch.Tensor, absolute_eligible_a: torch.Tensor, absolute_eligible_b: torch.Tensor, response_target: torch.Tensor, response_sigma: torch.Tensor, response_uncertainty_floor: torch.Tensor, uncertainty_available: torch.Tensor, response_eligible: torch.Tensor, teacher_roles: tuple[TeacherRole, ...])`; `AnalyticEligibilityAudit(edge_id: str, component: str, teacher_role: Literal["analytic_weak", "analytic_strong"], empirical_value: float | None, analytic_value: float, absolute_disagreement: float | None, threshold: float, eligible: bool, reason: str)`; `AnalyticContext(scene_id: str, teacher_roles: tuple[TeacherRole, ...], parameters: Mapping[str, torch.Tensor], eligibility: torch.Tensor, eligibility_reasons: tuple[str, ...])`; `ResponseSupervision(target: ResponseTargetBatch, analytic_context: AnalyticContext, audit_records: tuple[AnalyticEligibilityAudit, ...])`; `LossWeights(absolute: float, sign: float, magnitude: float, zero: float, analytic: float, validity: float)`; `LossTerm(value: torch.Tensor, eligible_count: torch.Tensor)`; and `PhysicalLossBreakdown(total: torch.Tensor, absolute: torch.Tensor, sign: torch.Tensor, magnitude: torch.Tensor, zero: torch.Tensor, analytic: torch.Tensor, validity: torch.Tensor, counts: Mapping[str, int])`. `L_FM` remains in `TrainingStepOutput` because only branches with RGB targets own that term.

- [ ] **Step 1: Write exact mechanics and separated-loss tests**

```python
class MechanicsTests(unittest.TestCase):
    def test_elastic_collision_conserves_signed_momentum_and_energy(self) -> None:
        mass = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
        incident = torch.tensor([[1.0, -0.25]], dtype=torch.float64)
        post = elastic_two_ball_post_velocity(mass, incident)
        self.assertTensorClose(total_momentum(mass, incident), total_momentum(mass, post), atol=1e-12)
        self.assertTensorClose(total_kinetic_energy(mass, incident), total_kinetic_energy(mass, post), atol=1e-12)

    def test_incline_derivative_contains_fixed_friction_but_no_response_token(self) -> None:
        theta = torch.tensor([0.7], dtype=torch.float64, requires_grad=True)
        acceleration = incline_acceleration_m_s2(theta)
        derivative = torch.autograd.grad(acceleration.sum(), theta)[0]
        self.assertTensorClose(derivative, incline_acceleration_derivative(theta), atol=1e-12)

class PhysicalLossTests(unittest.TestCase):
    def test_significant_and_zero_targets_use_disjoint_terms(self) -> None:
        significant = compute_fixture_loss(response=0.8, sigma=0.05, prediction=0.6)
        zero = compute_fixture_loss(response=0.02, sigma=0.05, prediction=0.3)
        self.assertEqual(1, significant.counts["sign"])
        self.assertEqual(1, significant.counts["magnitude"])
        self.assertEqual(0, significant.counts["zero"])
        self.assertEqual(0, zero.counts["sign"])
        self.assertEqual(0, zero.counts["magnitude"])
        self.assertEqual(1, zero.counts["zero"])

    def test_low_generated_visibility_never_masks_its_own_loss(self) -> None:
        good = compute_fixture_loss(generated_visibility_logit=8.0)
        missing = compute_fixture_loss(generated_visibility_logit=-8.0)
        self.assertGreater(float(missing.validity), float(good.validity))
        self.assertEqual(good.counts["magnitude"], missing.counts["magnitude"])

    def test_analytic_teacher_never_substitutes_for_missing_empirical_target(self) -> None:
        supervision = build_fixture_supervision(
            empirical_available=False, analytic_available=True
        )
        self.assertFalse(bool(supervision.target.response_eligible.any()))
        self.assertTrue(bool(supervision.analytic_context.eligibility.any()))

    def test_incline_and_pendulum_analytic_eligibility_is_built_not_injected(self) -> None:
        incline = build_incline_fixture_supervision(
            tan_theta=0.40, continuous_contact=True, sliding_downslope=True
        )
        pendulum = build_pendulum_fixture_supervision(
            effective_length_m=0.20, calibrated_pivot_to_center_m=0.30
        )
        self.assertFalse(bool(incline.analytic_context.eligibility.any()))
        self.assertFalse(bool(pendulum.analytic_context.eligibility.any()))
        self.assertIn("below_friction_threshold", incline.analytic_context.eligibility_reasons)
        self.assertIn("effective_length_mismatch", pendulum.analytic_context.eligibility_reasons)

    def test_three_ball_has_zero_response_counts_but_keeps_other_terms(self) -> None:
        loss = compute_three_ball_fixture_loss()
        self.assertEqual(0, loss.counts["sign"])
        self.assertEqual(0, loss.counts["magnitude"])
        self.assertEqual(0, loss.counts["zero"])
        self.assertGreater(loss.counts["absolute"], 0)
        self.assertGreater(loss.counts["analytic"], 0)
        self.assertGreater(loss.counts["validity"], 0)

    def test_uncertainty_floor_round_trip_is_train_only_and_identity_bound(self) -> None:
        path = fixture_output_path("uncertainty_floors.json")
        floors = fixture_train_uncertainty_floors()
        write_uncertainty_floors(floors, path)
        loaded = load_uncertainty_floors(
            path,
            expected_response_manifest_digest=floors.response_manifest_digest,
            expected_teacher_train_binding_digest=floors.teacher_train_binding_digest,
        )
        self.assertEqual("response_train", loaded.source_partition)
        self.assertEqual(floors, loaded)
        mutate_json(path, "teacher_train_binding_digest")
        with self.assertRaisesRegex(ValueError, "teacher train binding"):
            load_uncertainty_floors(
                path,
                expected_response_manifest_digest=floors.response_manifest_digest,
                expected_teacher_train_binding_digest=floors.teacher_train_binding_digest,
            )

    def test_uncertainty_floor_units_are_derived_from_verified_teacher_schemas(self) -> None:
        view, record_keys = fixture_train_teacher_view_with_units()
        floors = fit_train_uncertainty_floors(
            view, record_keys=record_keys,
            response_manifest_digest=fixture_response_manifest().digest,
            teacher_train_binding_digest=fixture_train_binding_header().digest,
        )
        self.assertEqual(fixture_expected_component_units(), dict(floors.component_units))
        with self.assertRaisesRegex(ValueError, "inconsistent teacher component unit"):
            inconsistent_view, inconsistent_keys = fixture_train_teacher_view_with_units(
                unit_override=("vx_m_s", "px/frame")
            )
            fit_train_uncertainty_floors(
                inconsistent_view, record_keys=inconsistent_keys,
                response_manifest_digest=fixture_response_manifest().digest,
                teacher_train_binding_digest=fixture_train_binding_header().digest,
            )

    def test_ineligible_nonfinite_teacher_is_sanitized_before_arithmetic(self) -> None:
        prediction = torch.tensor([0.2], requires_grad=True)
        sign, magnitude, zero = response_terms(
            prediction, torch.tensor([float("nan")]),
            torch.tensor([float("inf")]), torch.tensor([False]),
            torch.tensor([float("nan")]), torch.tensor([False]),
        )
        total = sign.value + magnitude.value + zero.value
        total.backward()
        self.assertTrue(torch.isfinite(total))
        self.assertTrue(torch.isfinite(prediction.grad).all())
```

- [ ] **Step 2: Run mechanics/loss tests and observe missing modules**

Run:

```bash
PYTHONPATH=src:tests:. /mnt/nvme1/NicoCache/envs/physics_wan/bin/python \
  -m unittest tests.physical_response.test_mechanics \
  tests.physical_response.test_losses -v
```

Expected: FAIL importing `mechanics` and `losses`.

- [ ] **Step 3: Implement exact formulas and masked robust objectives**

```python
def complete_elliptic_k_agm(
    parameter_m: torch.Tensor,
    *,
    iterations: int = 8,
) -> torch.Tensor:
    if torch.any((parameter_m < 0) | (parameter_m >= 1)):
        raise ValueError("elliptic parameter must satisfy 0 <= m < 1")
    a = torch.ones_like(parameter_m)
    b = torch.sqrt(1.0 - parameter_m)
    for _iteration in range(iterations):
        a, b = 0.5 * (a + b), torch.sqrt(a * b)
    return math.pi / (2.0 * a)

def response_terms(
    response_hat: torch.Tensor,
    response_target: torch.Tensor,
    measured_sigma: torch.Tensor,
    uncertainty_available: torch.Tensor,
    uncertainty_floor: torch.Tensor,
    teacher_eligible: torch.Tensor,
) -> tuple[LossTerm, LossTerm, LossTerm]:
    eligible = teacher_eligible & uncertainty_available
    significant = eligible & (response_target.abs() > 2.0 * measured_sigma)
    zero = eligible & ~significant
    safe_hat = torch.where(eligible, response_hat, torch.zeros_like(response_hat))
    safe_target = torch.where(
        eligible, response_target, torch.zeros_like(response_target)
    )
    safe_sigma = torch.where(
        eligible, measured_sigma, torch.ones_like(measured_sigma)
    )
    safe_floor = torch.where(
        eligible, uncertainty_floor, torch.ones_like(uncertainty_floor)
    )
    scale = torch.maximum(2.0 * safe_sigma, safe_floor)
    normalized_error = (safe_hat - safe_target) / scale
    magnitude = masked_huber(normalized_error, significant)
    signed_margin = 1.0 - safe_target.sign() * safe_hat / scale
    sign = masked_mean(F.softplus(signed_margin), significant)
    zero_loss = masked_huber(safe_hat / scale, zero)
    return sign, magnitude, zero_loss
```

Before validation or test access, `write_uncertainty_floors` creates `frozen/uncertainty_floors.json`. `fit_train_uncertainty_floors` obtains every trajectory and summary unit only from the `StateSchema` values of records reached through the verified response-train binding, rejects a component name with inconsistent units across records, and exposes no caller-supplied unit argument. Its loader re-verifies the response manifest and response-train teacher binding and rejects any unit, source-partition, or floor-value mutation; evaluators never refit floors.

`build_factual_target` resolves its record only through `teacher_view.get(case_id, coordinate_frame_id)`. `build_response_supervision` resolves every real endpoint through `teacher_view.get(case_id, edge.coordinate_frame_id)` and rejects any returned record whose `(case_id, coordinate_frame_id)` differs from that composite key.

`fit_train_uncertainty_floors` operates only on `response_train`: for each component it uses the 25th percentile of positive measured standard errors when at least five are available, otherwise the median per-record robust fit residual divided by the square root of valid frames, and clamps to the component's declared numeric epsilon. The measured sigma and floor are stored in separate target fields. The floor changes only the denominator; significance remains exactly `abs(R*) > 2*sigma_R` using measured uncertainty alone. `build_response_supervision` computes `(Y_b-Y_a)/((q_b-q_a)/q_scale)` for empirical edges and propagates measured endpoint uncertainty in quadrature; an analytic mass counterfactual instead derives its response from the factual two-ball teacher state plus the exact elastic formula and never invents a synthetic teacher record. Sign and magnitude are applied only to significant targets; zero loss is applied only to statistically non-significant targets, so no response component is double-counted. Every masked loss applies `torch.where(mask, value, 0)` before reduction, so an ineligible non-finite teacher sentinel cannot contaminate the scalar.

Teacher-role dispatch is exhaustive and tested: `empirical_primary` alone can activate empirical response fields; `absolute_empirical` alone activates real-endpoint/factual absolute fields; `analytic_weak` activates only weak mechanics; and `analytic_strong` activates elastic two-ball response/conservation or three-ball conservation. Missing empirical supervision remains missing even when an analytic value exists. `build_response_supervision` computes every analytic eligibility bit from frozen teacher/calibration/event evidence rather than accepting an arbitrary caller-supplied boolean, compares weak analytic and empirical values against the policy's per-component pre-test threshold, and emits `AnalyticEligibilityAudit` records. Excess disagreement masks only the weak analytic term and records a review reason; the empirical term remains untouched. Elastic two-ball contracts remain strong. Three-ball analytic loss contains only global momentum/energy and validity, with response eligibility forced false. The incline builder additionally requires `tan(theta) > 0.463`, observed down-slope sliding, continuous contact, and a common post-onset/pre-exit physical-time window; the pendulum builder requires agreement between `string_length + bob_radius` and frozen pivot-to-center calibration; projectile invariants stop before landing; circular zero terms cover angular velocity and period.

- [ ] **Step 4: Run all loss tests and finite-gradient assertions**

Run:

```bash
PYTHONPATH=src:tests:. /mnt/nvme1/NicoCache/envs/physics_wan/bin/python \
  -m unittest tests.physical_response.test_mechanics \
  tests.physical_response.test_losses -v
```

Expected: all tests PASS; exact mechanics pass FP64 tolerances and every active loss gives finite state gradients.

- [ ] **Step 5: Commit mechanics and loss decomposition**

```bash
git add src/physbench/physical_response/mechanics.py \
  src/physbench/physical_response/losses.py \
  schemas/physical_response_uncertainty_floors.schema.json \
  scripts/fit_physical_response_uncertainty_floors.py \
  tests/physical_response/test_mechanics.py \
  tests/physical_response/test_losses.py
git commit -m "feat: define physical response losses"
```

### Task 12: Materialize factual/paired batches and the observer-frame curriculum

**Files:**
- Create: `src/physbench/physical_response/batches.py`
- Create: `src/physbench/physical_response/sampling.py`
- Create: `schemas/physical_response_access_audit.schema.json`
- Create: `tests/physical_response/test_batches.py`
- Create: `tests/physical_response/test_sampling.py`

**Interfaces:**
- Consumes: overlay manifest, sealed `ConditionIndex`, sealed `CoordinateFrameIndex`, and train binding from Tasks 3, 4, and 9; WAN-normalized 121-frame media semantics from `src/physbench/baselines/wan22_media.py:164-343`; quantity payload contract from `src/physbench/baselines/wan22_quantity_model.py:190-216`.
- Produces: `ConditionPayload`, `FactualBatch`, `ResponseBranchBatch`, `PairScheduleConfig`, `PairScheduleState`, `LogicalUpdate`, `ResponseTrainingDataset`, `ResponseValidationDataset`, `ResponseAccessEvent`, `ResponseAccessAudit`, and `PairUpdateSchedule`; `ResponseTrainingDataset.__init__(condition_index: ConditionIndex, coordinate_index: CoordinateFrameIndex, *, manifest: ResponseManifest, policy: ResponsePolicy, teacher_view: TeacherCacheView, uncertainty_floors: UncertaintyFloors, media_loader: Callable[[str], torch.Tensor])`; `ResponseTrainingDataset.factual(case_id: str, *, update_id: str, stage_update_index: int) -> FactualBatch`; `ResponseTrainingDataset.response(edge_id: str, *, occurrence: int, update_id: str, stage_update_index: int) -> tuple[ResponseBranchBatch, ResponseBranchBatch]`; `ResponseValidationDataset.__init__(condition_index: ConditionIndex, coordinate_index: CoordinateFrameIndex, *, manifest: ResponseManifest, policy: ResponsePolicy, teacher_view: TeacherCacheView, uncertainty_floors: UncertaintyFloors, access_audit_path: Path)` permits only `response_validation` edges and has no optimizer-facing factual sampler; `create_response_access_audit(path: Path, *, partition: Partition) -> ResponseAccessAudit`; `load_response_access_audit(path: Path, *, expected_partition: Partition) -> ResponseAccessAudit`; `plan_response_access_update(current: ResponseAccessAudit, *, purpose: Literal["validation_teacher", "test_target_build"], opened_case_ids: Sequence[str], opened_edge_ids: Sequence[str], authorization_digest: str | None, target_set_digest: str | None) -> ResponseAccessAudit`; `compare_and_swap_response_access_audit(path: Path, *, expected_digest: str, replacement: ResponseAccessAudit) -> ResponseAccessAudit`; `apply_overlay_condition(prompt: str, quantities: Sequence[Mapping[str, Any]], projection: SignedProjection, *, active_quantity: ActiveQuantityBinding, counterfactual_q_si: str | None = None) -> ConditionPayload`; `ConditionPayload.from_values(prompt: str, quantities: Sequence[Mapping[str, Any]], projection_id: str) -> ConditionPayload`; `select_observer_indices(total_frames: int, observer_frames: Literal[16, 32, 121], *, teacher_event_frames: Sequence[int]) -> torch.Tensor`; `PairUpdateSchedule.__init__(config: PairScheduleConfig, *, manifest_digest: str, factual_case_ids: Sequence[str], response_edge_ids: Sequence[str])`; `PairUpdateSchedule.next() -> LogicalUpdate`; `PairUpdateSchedule.snapshot() -> PairScheduleState`; `PairUpdateSchedule.restore(state: PairScheduleState) -> None`; `PairUpdateSchedule.state_dict() -> dict[str, Any]`; `PairUpdateSchedule.load_state_dict(state: Mapping[str, Any]) -> None`.
- Branch policy: corresponding-real branches load their own 121-frame video/first frame and both carry factual FlowMatch/absolute eligibility; shared-anchor branches load the same anchor 121-frame video, generated-frame noisy latent, and first frame on both ranks, but only the branch whose quantity equals the factual anchor carries FlowMatch/absolute eligibility in that response update. The other shared-anchor condition has neither a fake RGB target nor a fake absolute target. Hidden-axis anchor choice alternates low/high by edge occurrence while output order remains canonical `q_a,q_b`; separate factual updates independently cover both real endpoints of every empirical edge. Collision-mass synthetic branches are always analytic-only.
- Data contracts: `ConditionPayload(prompt: str, quantities: tuple[dict[str, Any], ...], projection_id: str, payload_digest: str)`; `PairScheduleConfig(schema_version: Literal["pair_update_schedule_v1"], mode_sequence: tuple[Literal["factual", "response"], ...], factual_seed: int, response_seed: int, digest: str)`; `PairScheduleState(schedule_digest: str, manifest_digest: str, stage_update_index: int, factual_epoch: int, response_epoch: int, factual_cursor: int, response_cursor: int, factual_order: tuple[str, ...], response_order: tuple[str, ...], factual_generator_state_base64: str, response_generator_state_base64: str, edge_occurrences: tuple[tuple[str, int], ...], digest: str)`; `FactualBatch(update_id: str, stage_update_index: int, case_id: str, scene_id: str, coordinate: CoordinateFrameRecord | None, coordinate_manifest_digest: str | None, video_121: torch.Tensor, first_frame: torch.Tensor, initial_mask: torch.Tensor | None, prompt: str, quantities: tuple[dict[str, Any], ...], has_physical_target: bool, target: FactualTargetBatch | None, analytic_context: AnalyticContext | None, teacher_event_frames: tuple[int, ...])`; `ResponseBranchBatch(update_id: str, stage_update_index: int, edge_id: str, branch: Literal["a", "b"], q_si: str, source_case_id: str | None, anchor_case_id: str, scene_id: str, coordinate: CoordinateFrameRecord, coordinate_manifest_digest: str, video_121: torch.Tensor, first_frame: torch.Tensor, initial_mask: torch.Tensor, prompt: str, quantities: tuple[dict[str, Any], ...], has_real_rgb_target: bool, has_absolute_target: bool, response_target: ResponseTargetBatch, analytic_context: AnalyticContext, teacher_event_frames: tuple[int, ...])`; `LogicalUpdate(update_id: str, stage_update_index: int, kind: Literal["factual", "response"], case_id: str | None, edge_id: str | None, occurrence: int)`; `ResponseAccessEvent(sequence: int, purpose: Literal["validation_teacher", "test_target_build"], opened_case_ids: tuple[str, ...], opened_edge_ids: tuple[str, ...], authorization_digest: str | None, target_set_digest: str | None, digest: str)`; and `ResponseAccessAudit(schema_version: Literal["physical_response_access_audit_v1"], partition: Partition, opened_case_ids: tuple[str, ...], opened_edge_ids: tuple[str, ...], access_count: int, events: tuple[ResponseAccessEvent, ...], digest: str)`. A test-target event requires a non-null authorization and target-set identity; a validation event requires both fields null.

- [ ] **Step 1: Write branch, anchor, held-out, and frame-selection tests**

```python
class ResponseBatchTests(unittest.TestCase):
    def test_shared_anchor_alternates_without_reversing_response_order(self) -> None:
        dataset = fixture_response_dataset(frame_kind="shared_anchor")
        first = dataset.response(
            "edge_1", occurrence=0, update_id="response:0", stage_update_index=0
        )
        second = dataset.response(
            "edge_1", occurrence=1, update_id="response:1", stage_update_index=1
        )
        self.assertEqual(first[0].anchor_case_id, first[1].anchor_case_id)
        self.assertNotEqual(first[0].anchor_case_id, second[0].anchor_case_id)
        self.assertLess(Decimal(first[0].q_si), Decimal(first[1].q_si))
        self.assertLess(Decimal(second[0].q_si), Decimal(second[1].q_si))
        self.assertTensorEqual(first[0].first_frame, first[1].first_frame)
        self.assertTensorEqual(first[0].initial_mask, first[1].initial_mask)
        self.assertTensorEqual(first[0].video_121, first[1].video_121)
        self.assertEqual(1, sum(branch.has_absolute_target for branch in first))
        self.assertEqual(
            1,
            int(first[0].response_target.absolute_eligible_a.any())
            + int(first[0].response_target.absolute_eligible_b.any()),
        )

    def test_corresponding_real_branches_bind_each_case_media_and_mask(self) -> None:
        branch_a, branch_b = fixture_response_dataset(
            frame_kind="corresponding_real"
        ).response(
            "edge_1", occurrence=0, update_id="response:0", stage_update_index=0
        )
        self.assertNotEqual(branch_a.source_case_id, branch_b.source_case_id)
        self.assertEqual(branch_a.source_case_id, branch_a.anchor_case_id)
        self.assertEqual(branch_b.source_case_id, branch_b.anchor_case_id)
        self.assertNotEqual(tensor_digest(branch_a.first_frame), tensor_digest(branch_b.first_frame))
        self.assertNotEqual(tensor_digest(branch_a.initial_mask), tensor_digest(branch_b.initial_mask))
        self.assertTrue(branch_a.has_absolute_target)
        self.assertTrue(branch_b.has_absolute_target)
        self.assertTrue(bool(branch_a.response_target.absolute_eligible_a.any()))
        self.assertTrue(bool(branch_a.response_target.absolute_eligible_b.any()))

    def test_training_dataset_cannot_materialize_held_out_case(self) -> None:
        dataset = fixture_response_dataset()
        with self.assertRaisesRegex(PermissionError, "not in FlowMatch train view"):
            dataset.factual(
                "response_test_case", update_id="factual:test", stage_update_index=0
            )

    def test_out_of_scope_or_missing_mask_case_remains_flowmatch_eligible(self) -> None:
        batch = fixture_response_dataset().factual(
            "missing_first_frame_mask", update_id="factual:missing-mask",
            stage_update_index=0,
        )
        self.assertFalse(batch.has_physical_target)
        self.assertIsNone(batch.initial_mask)
        self.assertIsNone(batch.target)
        self.assertIsNone(batch.coordinate)
        self.assertEqual(121, batch.video_121.shape[0])

    def test_three_ball_has_factual_physics_but_no_response_edge(self) -> None:
        dataset = fixture_response_dataset()
        batch = dataset.factual(
            "three_ball_train_case", update_id="factual:three-ball",
            stage_update_index=0,
        )
        self.assertTrue(batch.has_physical_target)
        self.assertIsNotNone(batch.target)
        self.assertFalse(any(
            "three_ball_train_case" in edge.endpoint_a_case_ids + edge.endpoint_b_case_ids
            for edge in dataset.manifest.edges
        ))

    def test_batch_coordinate_matches_condition_and_teacher_record(self) -> None:
        batch = fixture_response_dataset().factual(
            "pendulum_train_case", update_id="factual:coordinate",
            stage_update_index=0,
        )
        self.assertEqual(batch.case_id, batch.coordinate.case_id)
        self.assertEqual(
            batch.coordinate.digest,
            fixture_teacher_view().get(
                batch.case_id, batch.coordinate.frame_id
            ).coordinate_digest,
        )

    def test_branch_coordinate_policy_is_exact(self) -> None:
        shared_a, shared_b = fixture_response_dataset(
            frame_kind="shared_anchor"
        ).response(
            "edge_1", occurrence=0, update_id="response:0", stage_update_index=0
        )
        self.assertEqual(shared_a.coordinate.digest, shared_b.coordinate.digest)
        real_a, real_b = fixture_response_dataset(
            frame_kind="corresponding_real"
        ).response(
            "edge_2", occurrence=0, update_id="response:0", stage_update_index=0
        )
        self.assertEqual(real_a.source_case_id, real_a.coordinate.case_id)
        self.assertEqual(real_b.source_case_id, real_b.coordinate.case_id)
        with self.assertRaisesRegex(ValueError, "coordinate frame id"):
            fixture_response_dataset(coordinate_frame_id="substituted")

class ObserverFrameSamplingTests(unittest.TestCase):
    def test_sparse_indices_include_endpoints_events_and_exact_count(self) -> None:
        indices = select_observer_indices(
            121, 16, teacher_event_frames=(40, 41, 89)
        )
        self.assertEqual(16, len(indices))
        self.assertEqual([0, 120], [int(indices[0]), int(indices[-1])])
        self.assertTrue({40, 41, 89}.issubset(set(indices.tolist())))

class PairScheduleTests(unittest.TestCase):
    def test_resume_replays_exact_next_100_updates(self) -> None:
        schedule = fixture_pair_schedule(
            mode_sequence=("factual", "response", "factual", "response"),
            factual_seed=101, response_seed=202,
        )
        for _index in range(37):
            schedule.next()
        state = schedule.snapshot()
        expected = tuple(schedule.next() for _index in range(100))
        restored = fixture_pair_schedule(
            mode_sequence=("factual", "response", "factual", "response"),
            factual_seed=101, response_seed=202,
        )
        restored.restore(state)
        self.assertEqual(expected, tuple(restored.next() for _index in range(100)))

    def test_restore_rejects_schedule_or_manifest_identity_change(self) -> None:
        state = fixture_pair_schedule().snapshot()
        with self.assertRaisesRegex(ValueError, "schedule digest mismatch"):
            fixture_pair_schedule(mode_sequence=("response", "factual")).restore(state)
        with self.assertRaisesRegex(ValueError, "manifest digest mismatch"):
            fixture_pair_schedule(manifest_digest="manifest-b").restore(state)

class ResponseAccessAuditTests(unittest.TestCase):
    def test_create_load_and_atomic_compare_and_swap_are_persistent(self) -> None:
        path = fixture_output_path("response_validation_access.json")
        initial = create_response_access_audit(path, partition="response_validation")
        replacement = plan_response_access_update(
            initial, purpose="validation_teacher",
            opened_case_ids=("case_a",), opened_edge_ids=("edge_a",),
            authorization_digest=None, target_set_digest=None,
        )
        committed = compare_and_swap_response_access_audit(
            path, expected_digest=initial.digest, replacement=replacement,
        )
        self.assertEqual(1, committed.access_count)
        self.assertEqual(committed, load_response_access_audit(
            path, expected_partition="response_validation"
        ))

    def test_stale_compare_and_swap_leaves_audit_byte_identical(self) -> None:
        path, current = fixture_existing_access_audit()
        before = path.read_bytes()
        replacement = fixture_next_access_audit(current)
        with self.assertRaisesRegex(RuntimeError, "access audit compare-and-swap"):
            compare_and_swap_response_access_audit(
                path, expected_digest="stale", replacement=replacement,
            )
        self.assertEqual(before, path.read_bytes())
```

- [ ] **Step 2: Run batch/sampling tests and observe missing modules**

Run:

```bash
PYTHONPATH=src:tests:. /mnt/nvme1/NicoCache/envs/physics_wan/bin/python \
  -m unittest tests.physical_response.test_batches \
  tests.physical_response.test_sampling -v
```

Expected: FAIL importing `batches` and `sampling`.

- [ ] **Step 3: Implement immutable branch copies and deterministic schedules**

```python
def select_observer_indices(
    total_frames: int,
    observer_frames: Literal[16, 32, 121],
    *,
    teacher_event_frames: Sequence[int],
) -> torch.Tensor:
    if total_frames != 121 or observer_frames not in {16, 32, 121}:
        raise ValueError("observer curriculum requires 121 source frames and 16/32/121 samples")
    if observer_frames == 121:
        return torch.arange(121, dtype=torch.long)
    chosen = {0, 120}
    chosen.update(index for index in teacher_event_frames if 0 <= index < 121)
    if len(chosen) > observer_frames:
        raise ValueError("teacher events exceed observer-frame budget")
    while len(chosen) < observer_frames:
        candidate = max(
            (index for index in range(121) if index not in chosen),
            key=lambda index: (min(abs(index - used) for used in chosen), -index),
        )
        chosen.add(candidate)
    return torch.tensor(sorted(chosen), dtype=torch.long)

def apply_overlay_condition(
    prompt: str,
    quantities: Sequence[Mapping[str, Any]],
    projection: SignedProjection,
    *,
    active_quantity: ActiveQuantityBinding,
    counterfactual_q_si: str | None = None,
) -> ConditionPayload:
    copied = tuple(copy.deepcopy(dict(item)) for item in quantities)
    target = exactly_one_quantity_at_canonical_path(
        copied, active_quantity.quantity_path, active_quantity.entity_id
    )
    target["source_raw_value"] = target["raw_value"]
    target["source_si_value"] = target["si_value"]
    target["si_value"] = float(counterfactual_q_si or projection.value_si)
    target["signed_projection_id"] = projection.projection_id
    rendered_prompt = prompt
    if counterfactual_q_si is not None:
        source_literal = target["rendered_quantity"]
        target["source_raw_unit"] = target["raw_unit"]
        target["source_rendered_quantity"] = source_literal
        replacement_literal = render_quantity_literal(
            counterfactual_q_si, projection.source_unit
        )
        active = parse_quantity_literal(
            replacement_literal, expected_unit=projection.source_unit
        )
        target["raw_value"] = active.value
        target["raw_unit"] = active.unit
        target["rendered_quantity"] = replacement_literal
        rendered_prompt = replace_verified_prompt_span(
            prompt, projection.prompt_value_span,
            expected_literal=projection.source_raw_literal,
            replacement=replacement_literal,
        )
        assert_prompt_and_structured_quantity_round_trip(
            rendered_prompt, target, projection.quantity_name, counterfactual_q_si
        )
    return ConditionPayload.from_values(
        rendered_prompt, copied, projection.projection_id
    )
```

The training dataset verifies the `ConditionIndex` and `CoordinateFrameIndex` identities against the manifest/run binding, resolves every coordinate by case plus condition-record identity, verifies it equals the teacher record's coordinate identity, resolves every `SignedProjection` from the manifest, uses `build_factual_target`/`build_response_supervision` when physical supervision exists, and refuses any teacher partition other than `response_train`. Push-bottle, missing-mask, and other physically ineligible View-A-train cases still materialize `has_physical_target=false` factual batches for ordinary FlowMatch; three-ball cases materialize `has_physical_target=true` factual batches but never response batches. The distinct validation dataset requires a validation-only binding and persists every successful teacher access through the access-audit compare-and-swap API; it cannot expose `response_test` or feed ordinary optimizer updates. Audit creation is exclusive, loading is strict, an update locks and re-reads the current document before comparing the expected identity, and a stale or failed update leaves the prior file byte-identical. The schedule uses a sealed repeating mode sequence from stage policy, permutes factual case IDs and response edge IDs with independent `torch.Generator` streams, records both generator states and cursors, and refuses restore if manifest or schedule identities differ. For an empirical shared-anchor edge it schedules both endpoint factual updates independently of the alternating response updates. Velocity projection preserves the audited magnitude text and adds signed `source_si_value/si_value` audit fields. A mass counterfactual additionally preserves `source_raw_value`, `source_raw_unit`, `source_rendered_quantity`, and `source_si_value`, then updates every active field (`raw_value`, `raw_unit`, `rendered_quantity`, `si_value`) consistently, verifies the frozen source prompt span, rerenders and reparses the exact positive mass literal, and bypasses the existing rounded-value SI derivation. Frame selection happens only after Task 10 returns all 121 RGB frames. Event indices are loss-side teacher metadata and are absent from WAN/QuantityEncoder input dictionaries.

`PairScheduleConfig.mode_sequence` is non-empty and contains only factual/response modes; its two non-negative seeds are required and distinct. Warmup permits factual only, while every physical stage requires both modes. Snapshot/restore binds the manifest, schedule, exact case/edge sets, permutations, epochs, cursors, fixed-base64 RNG states, occurrences, and `stage_update_index`; `update_id` is derived from those sealed identities rather than parsed later.

For each response occurrence, the dataset clones immutable edge supervision into a step-scoped target: corresponding-real leaves both endpoint absolute masks eligible; empirical shared-anchor leaves only the selected factual anchor endpoint eligible; analytic counterfactual leaves only its factual endpoint eligible. Both rank branches receive the same step-scoped target, and each branch's `has_absolute_target` must agree with its endpoint mask.

- [ ] **Step 4: Run deterministic epoch/recovery and mutation tests**

Run:

```bash
PYTHONPATH=src:tests:. /mnt/nvme1/NicoCache/envs/physics_wan/bin/python \
  -m unittest tests.physical_response.test_batches \
  tests.physical_response.test_sampling -v
```

Expected: all tests PASS; saved/restored schedules emit the same next 100 logical updates and source metadata remains unchanged.

- [ ] **Step 5: Commit training batches and sampling**

```bash
git add src/physbench/physical_response/batches.py \
  src/physbench/physical_response/sampling.py \
  schemas/physical_response_access_audit.schema.json \
  tests/physical_response/test_batches.py \
  tests/physical_response/test_sampling.py
git commit -m "feat: schedule factual and paired response batches"
```

### Task 13: Add autograd-correct two-rank pairing and global gradient audits

**Files:**
- Create: `src/physbench/physical_response/distributed.py`
- Create: `tests/physical_response/test_distributed_contracts.py`
- Create: `tests/integration/physical_response_distributed_worker.py`

**Interfaces:**
- Consumes: flattened FP32 state tensors from Task 4 and logical updates from Task 12; `torch.distributed.nn.functional.all_gather` from PyTorch 2.6.
- Produces: `PairRankLayout`, `PairTopology`; pure `derive_pair_rank_layout(*, world_size: int, rank: int) -> PairRankLayout`; collective `build_pair_topology(layout: PairRankLayout, *, backend: str = "nccl") -> PairTopology`; `PairRankSampler(schedule: PairUpdateSchedule, topology: PairTopology)`; `PairRankSampler.next_local() -> LocalLogicalUpdate`; `gather_pair_state(local_state: torch.Tensor, topology: PairTopology) -> tuple[torch.Tensor, torch.Tensor]`; `global_parameter_gradient(parameters: Sequence[torch.nn.Parameter], *, world_group: Any) -> torch.Tensor`; `gradient_norm_audit(update_id: str, component_gradients: Mapping[str, torch.Tensor], *, minimum_physical_gradient_norm: float, minimum_physical_to_fm_ratio: float, maximum_physical_to_fm_ratio: float) -> GradientNormAudit`.
- Topology is fixed to rank pairs `(0,1),(2,3),(4,5),(6,7)` for world size eight. World size must be positive and even; both ranks execute identical collective sequences. Each rank computes the same full pair loss after autograd gather; it does not divide that loss by two, because gather-backward duplication followed by DDP world averaging yields the mean gradient across pair groups.
- Data contracts: `PairRankLayout(world_size: int, rank: int, pair_index: int, pair_ranks: tuple[int, int], local_branch: Literal["a", "b"])`; `PairTopology(layout: PairRankLayout, pair_group: Any, world_group: Any)`; `LocalLogicalUpdate(logical: LogicalUpdate, branch: Literal["a", "b"] | None, factual_case_id: str | None)`; and `GradientNormAudit(update_id: str, component_norms: Mapping[str, float], physical_norm: float, fm_norm: float, physical_to_fm_ratio: float, finite: bool, passed: bool, failure_reasons: tuple[str, ...])`.

- [ ] **Step 1: Write topology and two-process FP32 equivalence tests**

```python
class DistributedContractTests(unittest.TestCase):
    def test_rank_pairs_are_fixed_and_exhaustive(self) -> None:
        pairs = [derive_pair_rank_layout(world_size=8, rank=rank).pair_ranks for rank in range(8)]
        self.assertEqual(
            [(0, 1), (0, 1), (2, 3), (2, 3), (4, 5), (4, 5), (6, 7), (6, 7)],
            pairs,
        )

    def test_odd_world_size_fails_before_collectives(self) -> None:
        with self.assertRaisesRegex(ValueError, "world_size must be even"):
            derive_pair_rank_layout(world_size=7, rank=0)
```

The integration worker compares `L=(y_b-y_a-target)^2` using one shared FP32 module in one process against two DDP ranks with autograd gather. It asserts gradient cosine similarity at least `0.999`, relative gradient-norm error at most `0.01`, equal logical update IDs on the pair, and identical collective counts.

- [ ] **Step 2: Run CPU contracts and observe the missing module**

Run:

```bash
PYTHONPATH=src:tests:. /mnt/nvme1/NicoCache/envs/physics_wan/bin/python \
  -m unittest tests.physical_response.test_distributed_contracts -v
```

Expected: FAIL importing `physbench.physical_response.distributed`.

- [ ] **Step 3: Implement pair groups, autograd gather, and globally averaged gradients**

```python
def gather_pair_state(
    local_state: torch.Tensor,
    topology: PairTopology,
) -> tuple[torch.Tensor, torch.Tensor]:
    if local_state.dtype != torch.float32:
        raise TypeError("pair state exchange requires FP32")
    gathered = torch.distributed.nn.functional.all_gather(
        local_state.contiguous(), group=topology.pair_group
    )
    if len(gathered) != 2 or gathered[0].shape != gathered[1].shape:
        raise RuntimeError("pair state schema mismatch")
    return gathered[0], gathered[1]

def global_parameter_gradient(
    parameters: Sequence[torch.nn.Parameter],
    *,
    world_group: Any,
) -> torch.Tensor:
    flat = torch.cat([
        torch.zeros_like(parameter).reshape(-1)
        if parameter.grad is None else parameter.grad.float().reshape(-1)
        for parameter in parameters
    ])
    torch.distributed.all_reduce(flat, group=world_group)
    flat.div_(torch.distributed.get_world_size(world_group))
    return flat
```

`derive_pair_rank_layout` is a process-free pure function used by CPU unit tests. `build_pair_topology` requires an initialized distributed process, verifies world/rank against its layout, and creates every two-rank process group in the same global order on every rank before selecting the local group, preventing `new_group` ordering deadlocks. `PairRankSampler` broadcasts one logical-update identity per pair, assigns canonical A to the lower rank and B to the higher rank for response updates, and returns `branch=None` plus deterministic disjoint factual case assignments for factual updates. Factual updates use ordinary world-DDP and never call the pair-state collective; response updates make both pair ranks execute the identical collective sequence. The sampler verifies matching tensor schema/digest before exchange and persists epoch/permutation/occurrence/cursor state. Non-finite state or mismatched update identity raises on every rank through a synchronized failure flag before any optimizer step.

- [ ] **Step 4: Run two-rank CUDA gradient equivalence**

Run:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
PYTHONPATH=src:.:/root/Steven/wan22_pendulum_pipeline/vendor/DiffSynth-Studio \
  /mnt/nvme1/NicoCache/envs/physics_wan/bin/torchrun \
  --standalone --nproc_per_node=2 \
  tests/integration/physical_response_distributed_worker.py
PYTHONPATH=src:tests:. /mnt/nvme1/NicoCache/envs/physics_wan/bin/python \
  -m unittest tests.physical_response.test_distributed_contracts -v
```

Expected: exit zero with reported cosine `>=0.999`, norm error `<=0.01`, and no collective timeout.

- [ ] **Step 5: Commit distributed pairing**

```bash
git add src/physbench/physical_response/distributed.py \
  tests/physical_response/test_distributed_contracts.py \
  tests/integration/physical_response_distributed_worker.py
git commit -m "feat: exchange paired physical states with autograd"
```

### Task 14: Assemble the WAN physical-response training module

**Files:**
- Create: `src/physbench/physical_response/training.py`
- Create: `scripts/wan22_physical_response_train.py`
- Create: `scripts/train_wan22_physical_response.sh`
- Create: `scripts/materialize_physical_response_training_runtime.py`
- Create: `configs/physical_response/runtime/training_local.example.json`
- Create: `schemas/physical_response_training_runtime.schema.json`
- Create: `tests/physical_response/test_training_module.py`
- Create: `tests/physical_response/test_training_cli.py`

**Interfaces:**
- Consumes: existing `QuantityEncoder` and quantity-pipeline installation at `src/physbench/baselines/wan22_quantity_model.py:219-344,1031-1217`; LoRA topology selectors from the same file at `:29-102`; Tasks 10-13.
- Produces: `TrainingRuntimeConfig`, `LossRamp`, `GradientGateConfig`, `PhysicalTrainingConfig`, `GradientWindow`, `GradientWindowAudit`, `StructuredConditionSensitivityAudit`, `WanPhysicalResponseTrainingModule`; `load_training_runtime_config(path: Path) -> TrainingRuntimeConfig`; `resolve_and_verify_training_runtime(config: TrainingRuntimeConfig) -> TrainingRuntimeConfig`; `LossRamp.weights_for_update(stage_update_index: int) -> LossWeights`; `GradientWindow.append_before_optimizer_step(audit: GradientNormAudit, *, stage_update_index: int, world_group: Any, device: torch.device) -> GradientWindowAudit | None`; `audit_structured_condition_sensitivity(module: WanPhysicalResponseTrainingModule, base: ConditionPayload, changed_q: ConditionPayload, *, shuffled: ConditionPayload, detach_condition: bool) -> StructuredConditionSensitivityAudit`; `WanPhysicalResponseTrainingModule.factual_step(batch: FactualBatch, *, timestep_id: torch.Tensor, noise: torch.Tensor) -> TrainingStepOutput`; `WanPhysicalResponseTrainingModule.response_step(local_branch: ResponseBranchBatch, *, topology: PairTopology, timestep_id: torch.Tensor, noise: torch.Tensor) -> TrainingStepOutput`; `audit_trainable_parameters(module: WanPhysicalResponseTrainingModule) -> TrainableParameterAudit`; `assert_finite_trainable_gradients(module: WanPhysicalResponseTrainingModule) -> None`; CLI `wan22_physical_response_train.py --condition-index run/physical_response_group_holdout_v1_seed42/physical_response/condition_index.jsonl --coordinate-binding run/physical_response_group_holdout_v1_seed42/physical_response/coordinates/binding.json --coordinate-cache-root cache/physical_response/coordinates_v1 --response-manifest run/physical_response_group_holdout_v1_seed42/physical_response/manifest.json --teacher-train-binding run/physical_response_group_holdout_v1_seed42/physical_response/teacher_train_bindings.json --teacher-cache-root cache/physical_response/teacher_v1 --uncertainty-floors run/physical_response_group_holdout_v1_seed42/frozen/uncertainty_floors.json --stage-policy run/physical_response_group_holdout_v1_seed42/frozen/stages/physical_16.json --training-runtime configs/physical_response/runtime/training_local.json --output-dir run/physical_response_group_holdout_v1_seed42/artifacts/wan22/training`.
- `TrainingStepOutput` carries `total_loss`, `fm_loss`, `physical: PhysicalLossBreakdown`, `flow: FlowCoreOutput`, observer-frame indices, branch/pair/update IDs, and teacher eligibility counts. Factual steps use unrestricted configured FlowMatch timesteps. Response steps use one pair-derived timestep/noise and require the sealed physical sigma interval.
- Data contracts: `TrainingRuntimeConfig(schema_version: Literal["physical_response_training_runtime_v1"], interpreter: Path, wan_root: Path, diffsynth_root: Path, sam2_root: Path, sam2_checkpoint: Path, source_path: Path, digest: str)`; `LossRamp(initial: LossWeights, target: LossWeights, start_update: int, end_update: int, digest: str)`; `GradientGateConfig(window_updates: Literal[100], minimum_physical_gradient_norm: float, minimum_physical_to_fm_ratio: float, maximum_physical_to_fm_ratio: float, digest: str)`; `PhysicalTrainingConfig(stage_policy_digest: str, stage_id: str, planned_updates: int, learning_rate: float, observer_frames: Literal[0, 16, 32, 121], physical_sigma_min: float | None, physical_sigma_max: float | None, physical_sigma_selection_digest: str | None, loss_ramp: LossRamp, pair_schedule: PairScheduleConfig, gradient_gates: GradientGateConfig, compute_dtype: str, checkpoint_vae: bool, sam2_track_config: SAM2TrackConfig, sam2_observer_identity_digest: str, event_parameters: Mapping[str, float], validity_scales: Mapping[str, float], optimizer_transition: Literal["continuous", "reset"], response_manifest_digest: str, coordinate_manifest_digest: str, teacher_train_binding_digest: str, teacher_validation_binding_digest: str, uncertainty_floors_digest: str, input_checkpoint_digest: str, training_runtime_digest: str, validation_selection_digest: str, digest: str)`; `GradientWindowAudit(stage_policy_digest: str, window_index: int, first_update_index: int, last_update_index: int, sample_count: Literal[100], median_physical_gradient_norm: float, median_physical_to_fm_ratio: float, maximum_physical_to_fm_ratio: float, finite: bool, passed: bool, failure_reasons: tuple[str, ...], digest: str)`; `StructuredConditionSensitivityAudit(base_payload_digest: str, changed_payload_digest: str, shuffled_payload_digest: str, condition_output_delta_norm: float, lora_gradient_norm: float, quantity_encoder_gradient_norm: float, shuffled_alignment_score: float, detach_condition: bool, passed: bool, failure_reasons: tuple[str, ...], digest: str)`; `TrainingStepOutput(total_loss: torch.Tensor, fm_loss: torch.Tensor, physical: PhysicalLossBreakdown, flow: FlowCoreOutput, observer_frame_indices: torch.Tensor, update_id: str, stage_update_index: int, stage_policy_digest: str, realized_loss_weights: LossWeights, pair_id: str | None, branch: Literal["a", "b"] | None, teacher_counts: Mapping[str, int])`; and `TrainableParameterAudit(trainable_families: tuple[str, ...], frozen_families: tuple[str, ...], trainable_names: tuple[str, ...], trainable_tensor_count_by_family: tuple[tuple[str, int], ...], total_trainable_tensor_count: int, forbidden_trainable_names: tuple[str, ...], frozen_parameter_count: int, passed: bool, failure_reasons: tuple[str, ...])`.

- [ ] **Step 1: Write freeze, signed-condition, shared-randomness, and gradient tests**

```python
class PhysicalResponseTrainingModuleTests(unittest.TestCase):
    def test_only_lora_and_quantity_encoder_are_trainable(self) -> None:
        module = fixture_training_module()
        audit = audit_trainable_parameters(module)
        self.assertEqual({"dit_lora", "quantity_encoder"}, set(audit.trainable_families))
        self.assertEqual(
            {"wan_base", "umt5", "vae", "sam2"}, set(audit.frozen_families)
        )
        self.assertTrue(audit.passed, audit.failure_reasons)
        self.assertEqual(
            {"dit_lora": 600, "quantity_encoder": 19},
            dict(audit.trainable_tensor_count_by_family),
        )
        self.assertEqual(619, audit.total_trainable_tensor_count)
        self.assertEqual((), audit.forbidden_trainable_names)

    def test_response_step_reuses_exact_timestep_and_noise_on_both_branches(self) -> None:
        outputs = run_two_branch_fixture()
        self.assertTensorEqual(outputs.rank_a.flow.timestep_id, outputs.rank_b.flow.timestep_id)
        self.assertTensorEqual(outputs.rank_a.flow.noise, outputs.rank_b.flow.noise)

    def test_physical_loss_reaches_lora_and_quantity_encoder(self) -> None:
        module, output = differentiable_training_fixture()
        torch.autograd.backward(output.physical.total)
        self.assertGreater(lora_gradient_norm(module), 0.0)
        self.assertGreater(quantity_encoder_gradient_norm(module), 0.0)
        self.assertTrue(all(p.grad is None for p in module.vae.parameters()))
        self.assertTrue(all(p.grad is None for p in module.sam2.parameters()))

    def test_changing_only_structured_q_changes_the_condition_branch(self) -> None:
        audit = run_fixture_condition_sensitivity(
            prompt_change=False, structured_q_change=True,
            shuffled=False, detach_condition=False,
        )
        self.assertGreater(audit.condition_output_delta_norm, 0.0)
        self.assertGreater(audit.lora_gradient_norm, 0.0)
        self.assertGreater(audit.quantity_encoder_gradient_norm, 0.0)
        self.assertTrue(audit.passed, audit.failure_reasons)

    def test_shuffled_or_detached_condition_fails_the_same_sensitivity_gate(self) -> None:
        for shuffled, detached, reason in (
            (True, False, "structured_quantity_alignment"),
            (False, True, "detached_condition_gradient"),
        ):
            audit = run_fixture_condition_sensitivity(
                prompt_change=False, structured_q_change=True,
                shuffled=shuffled, detach_condition=detached,
            )
            with self.subTest(shuffled=shuffled, detached=detached):
                self.assertFalse(audit.passed)
                self.assertIn(reason, audit.failure_reasons)

    def test_flowmatch_only_factual_case_never_invokes_vae_or_sam2(self) -> None:
        module, batch = flowmatch_only_factual_fixture()
        output = module.factual_step(batch, **fixture_randomness())
        self.assertGreater(float(output.fm_loss), 0.0)
        self.assertEqual(0, sum(output.physical.counts.values()))
        module.vae.decode.assert_not_called()
        module.sam2.forward.assert_not_called()

    def test_flowmatch_warmup_skips_observer_even_with_teacher(self) -> None:
        module, batch = physical_factual_fixture(observer_frames=0, all_weights=0.0)
        output = module.factual_step(batch, **fixture_randomness())
        self.assertEqual(0, output.observer_frame_indices.numel())
        self.assertEqual(0, sum(output.physical.counts.values()))
        module.vae.decode.assert_not_called()
        module.sam2.forward.assert_not_called()

    def test_three_ball_factual_step_keeps_absolute_analytic_and_validity(self) -> None:
        output = run_three_ball_factual_fixture()
        self.assertGreater(output.physical.counts["absolute"], 0)
        self.assertGreater(output.physical.counts["analytic"], 0)
        self.assertGreater(output.physical.counts["validity"], 0)
        self.assertEqual(0, output.physical.counts["sign"])
        self.assertEqual(0, output.physical.counts["magnitude"])
        self.assertEqual(0, output.physical.counts["zero"])

    def test_loss_ramp_has_exact_boundaries_and_midpoint(self) -> None:
        ramp = fixture_loss_ramp(start_update=10, end_update=20)
        self.assertEqual(ramp.initial, ramp.weights_for_update(9))
        self.assertEqual(ramp.initial, ramp.weights_for_update(10))
        self.assertEqual(fixture_midpoint_weights(), ramp.weights_for_update(15))
        self.assertEqual(ramp.target, ramp.weights_for_update(20))
        self.assertEqual(ramp.target, ramp.weights_for_update(21))

    def test_nonfinite_ratio_prevents_optimizer_step(self) -> None:
        optimizer = mock.Mock()
        with self.assertRaisesRegex(
            SynchronizedGradientGateError,
            "nonfinite_gradient_ratio_or_norm_before_optimizer_step",
        ):
            run_fixture_optimizer_update(
                optimizer=optimizer, ratio=float("inf"), stage_update_index=3
            )
        optimizer.step.assert_not_called()
```

- [ ] **Step 2: Run training tests and observe missing module/entry point**

Run:

```bash
PYTHONPATH=src:tests:.:/root/Steven/wan22_pendulum_pipeline/vendor/DiffSynth-Studio \
  /mnt/nvme1/NicoCache/envs/physics_wan/bin/python -m unittest \
  tests.physical_response.test_training_module \
  tests.physical_response.test_training_cli -v
```

Expected: FAIL importing `WanPhysicalResponseTrainingModule`.

- [ ] **Step 3: Implement logical updates without altering the quantity trainer**

```python
class WanPhysicalResponseTrainingModule(nn.Module):
    def factual_step(
        self,
        batch: FactualBatch,
        *,
        timestep_id: torch.Tensor,
        noise: torch.Tensor,
    ) -> TrainingStepOutput:
        flow = self._flow_prediction(batch, timestep_id=timestep_id, noise=noise)
        weights = self.runtime.loss_ramp.weights_for_update(batch.stage_update_index)
        physical_enabled = (
            batch.has_physical_target
            and self.runtime.observer_frames != 0
            and weights.any_nonzero()
        )
        if not physical_enabled:
            return TrainingStepOutput.flowmatch_only(
                flow, batch.update_id,
                stage_update_index=batch.stage_update_index,
                stage_policy_digest=self.runtime.stage_policy_digest,
                realized_loss_weights=weights,
            )
        if batch.initial_mask is None or batch.target is None or batch.analytic_context is None:
            raise RuntimeError("physical factual batch is incomplete")
        rgb_121, indices, state = self._observe_clean_latent(flow, batch)
        physical = compute_factual_physical_loss(
            state, batch.target, analytic_context=batch.analytic_context,
            weights=weights,
        )
        return TrainingStepOutput.from_factual(
            flow.fm_loss, physical, flow, indices, batch
        )

    def response_step(
        self,
        local_branch: ResponseBranchBatch,
        *,
        topology: PairTopology,
        timestep_id: torch.Tensor,
        noise: torch.Tensor,
    ) -> TrainingStepOutput:
        flow = flow_match_core(
            self.pipe, self.prepare_branch_inputs(local_branch),
            timestep_id=timestep_id, noise=noise,
        )
        weights = self.runtime.loss_ramp.weights_for_update(
            local_branch.stage_update_index
        )
        rgb_121 = decode_full_video_differentiable(
            self.vae, flow.z0_hat_full, device=self.device,
            checkpoint_full_decode=self.runtime.checkpoint_vae,
        )
        indices = select_observer_indices(
            121, self.runtime.observer_frames,
            teacher_event_frames=local_branch.teacher_event_frames,
        )
        track = self.sam2(
            rgb_121.index_select(1, indices), local_branch.initial_mask,
            frame_indices=indices,
        )
        frame = materialize_coordinate_frame(
            local_branch.coordinate, device=self.device, dtype=torch.float32
        )
        local_state = self.extract_state(track, frame, local_branch)
        packed_a, packed_b = gather_pair_state(
            flatten_physical_state(local_state), topology
        )
        state_a = unflatten_physical_state_from_branch(packed_a, local_branch)
        state_b = unflatten_physical_state_from_branch(packed_b, local_branch)
        physical = compute_physical_loss(
            state_a, state_b, local_branch.response_target,
            analytic_context=local_branch.analytic_context,
            weights=weights,
        )
        fm = flow.fm_loss if local_branch.has_real_rgb_target else flow.fm_loss.new_zeros(())
        return TrainingStepOutput.from_parts(fm, physical, flow, indices, local_branch)
```

`LossRamp.weights_for_update` linearly interpolates every one of the six weights from `initial` to `target`, clamps before/after the inclusive boundaries, handles an equal start/end as an explicit step, and rejects negative indices or reversed boundaries. Both branches use the same schedule-provided `stage_update_index`; outputs and checkpoints record the realized weights.

The driver calls the gradient gate in this exact order:

```python
optimizer.zero_grad(set_to_none=True)
output.total_loss.backward()
assert_finite_trainable_gradients(module)
audit = gradient_norm_audit(
    output.update_id, component_gradients,
    minimum_physical_gradient_norm=config.gradient_gates.minimum_physical_gradient_norm,
    minimum_physical_to_fm_ratio=config.gradient_gates.minimum_physical_to_fm_ratio,
    maximum_physical_to_fm_ratio=config.gradient_gates.maximum_physical_to_fm_ratio,
)
completed_window = gradient_window.append_before_optimizer_step(
    audit, stage_update_index=output.stage_update_index,
    world_group=world_group, device=output.total_loss.device,
)
optimizer.step()
```

`append_before_optimizer_step` first all-reduces a fatal flag for any non-finite component norm, physical/FM norm, or ratio and raises the same `SynchronizedGradientGateError` on every rank before the current optimizer step. It then requires contiguous indices, emits one audit for each exact 100-update window, recomputes both configured norm/ratio bounds from the samples, all-reduces window failure, and blocks before stepping if the completed window fails. Physical stages require `planned_updates >= 100` and a multiple of 100; checkpoint state contains the partial window and all completed window digests, so resume cannot skip or duplicate a sample.

The stage loader converts a strict stage document into `PhysicalTrainingConfig`; Task 14 therefore has no forward import of Task 15. It transfers compute dtype, exact SAM2 detach positions, event parameters, validity scales, and the uncertainty-floor identity without dropping fields, and verifies the physical sigma interval on every response update. Before allocating a model, the CLI loads the coordinate binding and frozen uncertainty floors, verifies their response/condition/policy and response-train teacher identities, requires `source_partition="response_train"`, and requires every physical batch coordinate identity to equal its teacher record. Training and validation never refit, replace, or default the floors. State extraction receives only `materialize_coordinate_frame(batch.coordinate, ...)`; no training function fits a coordinate from generated RGB. Both input preparers pass the batch's prompt, signed quantities, first frame, and 121-frame target explicitly. The tracked shell entry point uses repository-relative scientific paths plus the Task-14-owned ignored training-runtime JSON for the interpreter, WAN root, DiffSynth root, SAM2 root, and checkpoint; neither Task 14 nor Task 15 imports the orchestration runtime from Task 18. `configs/physical_response/runtime/training_local.example.json` contains the strict training-runtime schema and only portable paths below `../external/`; `--validate-config-only` checks schema and portability without resolving them. The Task-14 materializer creates the ignored concrete file from explicit paths. The CLI tests separately synthesize a temporary file with real resolved fixture paths and verify pinned identities before model allocation. It verifies the pinned DiffSynth commit, 121 frames, LoRA rank 32/targets, and required response-manifest/cache/stage/floor identities after local resolution. The structured-condition gate holds prompt, first frame, noise, timestep, and every nuisance quantity fixed while changing only the active signed SI value; it requires a non-zero condition-output delta and gradients in both LoRA and `QuantityEncoder`. The identical gate must fail for an edge-mismatched shuffled quantity or any detached condition branch. It builds explicit pair process groups before loading the schedule, disallows Accelerate DataLoader sharding for response updates, all-reduces a failure flag before backward/step, checks every trainable gradient for NaN/Inf, and logs per-component LoRA gradients at the sealed cadence. Checkpoints contain only the 600 expected LoRA tensors, 19 `QuantityEncoder` tensors, optimizer/scheduler/sampler state, and sealed artifact identities.

`TrainingStepOutput.flowmatch_only` returns an empty observer-index tensor and an all-zero `PhysicalLossBreakdown` with zero counts; it does not decode VAE RGB, invoke SAM2, or call a state extractor.

- [ ] **Step 4: Run training tests and a no-GPU CLI validation**

Run:

```bash
PYTHONPATH=src:tests:.:/root/Steven/wan22_pendulum_pipeline/vendor/DiffSynth-Studio \
  /mnt/nvme1/NicoCache/envs/physics_wan/bin/python -m unittest \
  tests.physical_response.test_training_module \
  tests.physical_response.test_training_cli -v
bash scripts/train_wan22_physical_response.sh \
  --local-config configs/physical_response/runtime/training_local.example.json \
  --validate-config-only
```

Expected: all tests PASS; the example validation reports schema/portability success without claiming resolved identities, and the resolved-runtime test prints pinned dependency identities before model allocation.

- [ ] **Step 5: Commit the independent trainer**

```bash
git add src/physbench/physical_response/training.py \
  scripts/wan22_physical_response_train.py \
  scripts/train_wan22_physical_response.sh \
  scripts/materialize_physical_response_training_runtime.py \
  configs/physical_response/runtime/training_local.example.json \
  schemas/physical_response_training_runtime.schema.json \
  tests/physical_response/test_training_module.py \
  tests/physical_response/test_training_cli.py
git commit -m "feat: train WAN with physical response loss"
```

### Task 15: Seal stages and enforce memory, gradient, overfit, and validation gates

**Files:**
- Create: `src/physbench/physical_response/stages.py`
- Create: `schemas/physical_response_stage_policy.schema.json`
- Create: `scripts/seal_physical_response_stage.py`
- Create: `scripts/preflight_physical_response.py`
- Create: `tests/physical_response/test_stages.py`
- Create: `tests/integration/physical_response_end_to_end_gradient_worker.py`

**Interfaces:**
- Consumes: the training module, `TrainingRuntimeConfig`, and `GradientWindowAudit` from Task 14; typed teacher-binding headers from Task 9; `ResponseValidationDataset`, `ResponseAccessAudit`, and validation-only teacher artifacts from Tasks 9 and 12; the frozen uncertainty floors from Task 11; and the existing `SceneEvaluatorRegistry`. Task 15 has no dependency on Tasks 16-18 and cannot open a response-test binding. No caller-supplied gate boolean or metric scalar is trusted.
- Produces: `StagePolicy`, `TeacherCachePreflightAudit`, `StageCheckpointArtifact`, `StagePreflight`, `PairOverfitAudit`, `VisualNoninferiorityAudit`, `ResponseValidationAudit`, `OptimizerTransitionAudit`, and `StageGateAudit`; `StagePolicy.to_training_config() -> PhysicalTrainingConfig`; `seal_stage_policy(candidate_path: Path, *, previous_stage_policy_path: Path | None, response_manifest_path: Path, coordinate_binding_path: Path, coordinate_cache_root: Path, teacher_train_binding_path: Path, teacher_validation_binding_path: Path, teacher_cache_root: Path, uncertainty_floors_path: Path, input_checkpoint_path: Path | None, validation_selection_path: Path, response_test_access_audit_path: Path, training_runtime_config_path: Path, output_path: Path) -> StagePolicy`; `verify_stage_policy_inputs(policy: StagePolicy, *, response_manifest_path: Path, coordinate_binding_path: Path, coordinate_cache_root: Path, teacher_train_binding_path: Path, teacher_validation_binding_path: Path, teacher_cache_root: Path, uncertainty_floors_path: Path, input_checkpoint_path: Path | None, validation_selection_path: Path, response_test_access_audit_path: Path, training_runtime_config_path: Path) -> None`; `run_teacher_cache_preflight(*, response_manifest: ResponseManifest, coordinate_index: CoordinateFrameIndex, train_binding: TeacherCacheBinding, validation_binding: TeacherCacheBinding, reproducibility_audits: Sequence[CanonicalStateReproducibilityAudit], output_path: Path) -> TeacherCachePreflightAudit`; `run_stage_preflight(module: WanPhysicalResponseTrainingModule, batch: FactualBatch | tuple[ResponseBranchBatch, ResponseBranchBatch], policy: StagePolicy) -> StagePreflight`; `run_pair_overfit_probe(module_factory: Callable[[StageCheckpointArtifact], WanPhysicalResponseTrainingModule], validation: ResponseValidationDataset, checkpoint: StageCheckpointArtifact, policy: StagePolicy, *, edge_id: str, seed: int, probe_updates: int) -> PairOverfitAudit`; `run_visual_noninferiority_audit(candidate: StageCheckpointArtifact, reference: StageCheckpointArtifact, policy: StagePolicy, registry: SceneEvaluatorRegistry, *, validation_case_ids: Sequence[str], output_dir: Path) -> VisualNoninferiorityAudit`; `run_response_validation_audit(module: WanPhysicalResponseTrainingModule, validation: ResponseValidationDataset, checkpoint: StageCheckpointArtifact, policy: StagePolicy, *, response_test_access: ResponseAccessAudit) -> ResponseValidationAudit`; `evaluate_stage_transition(previous: StagePolicy | None, candidate: StagePolicy, checkpoint: StageCheckpointArtifact | None, teacher_cache_preflight: TeacherCachePreflightAudit | None, preflight: StagePreflight | None, gradient_windows: Sequence[GradientWindowAudit], pair_overfit: PairOverfitAudit | None, visual: VisualNoninferiorityAudit | None, response_validation: ResponseValidationAudit | None, optimizer_transition: OptimizerTransitionAudit | None, response_test_access: ResponseAccessAudit) -> StageGateAudit`; CLI `seal_physical_response_stage.py --candidate run/physical_response_group_holdout_v1_seed42/validation/stage_candidates/physical_16.json --previous-stage-policy run/physical_response_group_holdout_v1_seed42/frozen/stages/flowmatch_warmup.json --response-manifest run/physical_response_group_holdout_v1_seed42/physical_response/manifest.json --coordinate-binding run/physical_response_group_holdout_v1_seed42/physical_response/coordinates/binding.json --coordinate-cache-root cache/physical_response/coordinates_v1 --teacher-train-binding run/physical_response_group_holdout_v1_seed42/physical_response/teacher_train_bindings.json --teacher-validation-binding run/physical_response_group_holdout_v1_seed42/physical_response/teacher_validation_bindings.json --teacher-cache-root cache/physical_response/teacher_v1 --uncertainty-floors run/physical_response_group_holdout_v1_seed42/frozen/uncertainty_floors.json --input-checkpoint run/physical_response_group_holdout_v1_seed42/artifacts/wan22/checkpoints/flowmatch_warmup.safetensors --validation-selection run/physical_response_group_holdout_v1_seed42/validation/stage_selection_physical_16.json --response-test-access-audit run/physical_response_group_holdout_v1_seed42/access/response_test.json --training-runtime configs/physical_response/runtime/training_local.json --output run/physical_response_group_holdout_v1_seed42/frozen/stages/physical_16.json`; CLI `preflight_physical_response.py --stage-policy run/physical_response_group_holdout_v1_seed42/frozen/stages/physical_16.json --representative-edge collision_velocity_edge_0001 --output run/physical_response_group_holdout_v1_seed42/preflight/physical_16.json`.
- Stages are exactly `teacher_cache_preflight`, `flowmatch_warmup`, `physical_16`, `physical_32`, and `physical_121`. Each sealed policy binds its previous policy, planned updates, complete Task-5 `SAM2TrackConfig` (image size, block size, exact detach positions, all three checkpoint flags, and mask temperature), pair schedule/seeds, loss ramp, gradient gates, overlay/coordinate/two teacher bindings, frozen response-train uncertainty floors, input checkpoint, runtime, validation selection, and zero-access audit. The three physical policies additionally require numeric sigma bounds, observer coverage, typed one-pair overfit, non-empty visual and response-validation gate specifications, and checkpoint selection before execution.
- Data contracts: `StagePolicy(schema_version: Literal["physical_response_stage_policy_v1"], experiment_id: str, stage_id: Literal["teacher_cache_preflight", "flowmatch_warmup", "physical_16", "physical_32", "physical_121"], previous_stage_policy_digest: str | None, planned_updates: int, observer_frames: Literal[0, 16, 32, 121], learning_rate: float, physical_sigma_min: float | None, physical_sigma_max: float | None, physical_sigma_selection_digest: str | None, loss_ramp: LossRamp, pair_schedule: PairScheduleConfig, gradient_gates: GradientGateConfig, compute_dtype: str, checkpoint_vae: bool, sam2_track_config: SAM2TrackConfig, sam2_observer_identity_digest: str, event_parameters: Mapping[str, float], validity_scales: Mapping[str, float], minimum_observer_coverage: float, minimum_pair_overfit_gain: float, visual_noninferiority_gates: tuple[NonInferiorityGateSpec, ...], response_validation_gates: tuple[ResponseValidationGateSpec, ...], optimizer_transition: Literal["continuous", "reset"], checkpoint_selection_rule: str, response_manifest_digest: str, coordinate_manifest_digest: str, teacher_train_binding_digest: str, teacher_validation_binding_digest: str, uncertainty_floors_digest: str, input_checkpoint_digest: str | None, training_runtime_digest: str, validation_selection_digest: str, response_test_access_audit_digest: str, digest: str)`; `TeacherCachePreflightAudit(stage_policy_digest: str, response_manifest_digest: str, coordinate_manifest_digest: str, teacher_train_binding_digest: str, teacher_validation_binding_digest: str, observer_identity_digest: str, expected_record_count: int, verified_record_count: int, eligible_pair_count: int, rejected_pair_ids: tuple[str, ...], reproducibility_audit_digests: tuple[str, ...], passed: bool, failure_reasons: tuple[str, ...], digest: str)`; `StageCheckpointArtifact(stage_policy_digest: str, input_checkpoint_digest: str, checkpoint_path: Path, checkpoint_sha256: str, optimizer_state_digest: str, scheduler_state_digest: str, sampler_state_digest: str, completed_updates: int, artifact_digest: str)`; `StagePreflight(stage_policy_digest: str, training_config_digest: str, response_manifest_digest: str, coordinate_manifest_digest: str, teacher_train_binding_digest: str, uncertainty_floors_digest: str, input_checkpoint_digest: str, sam2_observer_identity_digest: str, sam2_track_config_digest: str, decoded_frame_count: Literal[0, 121], observer_frames: Literal[0, 16, 32, 121], observer_coverage: float, peak_allocated_bytes: int, peak_reserved_bytes: int, wall_time_s: float, finite_loss_and_gradients: bool, both_branch_lora_gradient: bool, both_branch_quantity_gradient: bool, memory_fit: bool, component_gradient_norms: Mapping[str, float], physical_gradient_norm: float, fm_gradient_norm: float, physical_to_fm_ratio: float, artifact_path: Path, digest: str)`; `PairOverfitAudit(stage_policy_digest: str, checkpoint_artifact_digest: str, teacher_validation_binding_digest: str, edge_id: str, seed: int, probe_updates: int, metric_name: str, initial_error: float, final_error: float, relative_gain: float, object_retained: bool, finite: bool, passed: bool, failure_reasons: tuple[str, ...], digest: str)`; `VisualNoninferiorityAudit(stage_policy_digest: str, checkpoint_artifact_digest: str, evaluator_manifest_digest: str, metrics: tuple[VisualMetricAudit, ...], passed: bool, failure_reasons: tuple[str, ...], digest: str)`; `ResponseValidationAudit(stage_policy_digest: str, checkpoint_artifact_digest: str, teacher_validation_binding_digest: str, source_partition: Literal["response_validation"], validation_selection_digest: str, response_test_access_audit_digest: str, metrics: tuple[ResponseValidationMetricAudit, ...], passed: bool, failure_reasons: tuple[str, ...], digest: str)`; `OptimizerTransitionAudit(previous_stage_policy_digest: str | None, candidate_stage_policy_digest: str, declared_transition: Literal["continuous", "reset"], realized_transition: Literal["continuous", "reset"], previous_optimizer_state_digest: str | None, candidate_input_optimizer_state_digest: str, passed: bool, failure_reasons: tuple[str, ...], digest: str)`; and `StageGateAudit(stage_id: str, stage_policy_digest: str, checkpoint_artifact_digest: str | None, requested_observer_frames: int, passed: bool, failure_reasons: tuple[str, ...], teacher_cache_preflight_digest: str | None, preflight_digest: str | None, gradient_window_digests: tuple[str, ...], pair_overfit_audit_digest: str | None, visual_noninferiority_audit_digest: str | None, response_validation_audit_digest: str | None, optimizer_transition_audit_digest: str | None, response_test_access_audit_digest: str, digest: str)`. `NonInferiorityGateSpec` and `ResponseValidationGateSpec` give typed metric name/direction/comparison/margin or threshold; their required name sets are non-empty and exact.

- Gate-spec contracts: `NonInferiorityGateSpec(metric_name: str, direction: Literal["higher_is_better", "lower_is_better"], margin: float)`; `ResponseValidationGateSpec(metric_name: str, comparison: Literal["ge", "le"], threshold: float)`; `VisualMetricAudit(metric_name: str, direction: Literal["higher_is_better", "lower_is_better"], reference_value: float, candidate_value: float, margin: float, passed: bool)`; and `ResponseValidationMetricAudit(metric_name: str, estimate: float, comparison: Literal["ge", "le"], threshold: float, passed: bool)`.

- [ ] **Step 1: Write policy completeness and fail-closed transition tests**

```python
class StagePolicyTests(unittest.TestCase):
    def test_physical_stage_rejects_missing_numeric_gradient_gate(self) -> None:
        candidate = physical_stage_document(observer_frames=16)
        del candidate["gradient_gates"]["minimum_physical_to_fm_ratio"]
        with self.assertRaisesRegex(ValueError, "minimum_physical_to_fm_ratio"):
            StagePolicy.from_document(candidate)

    def test_every_sam2_schedule_ramp_and_identity_field_round_trips(self) -> None:
        policy = seal_fixture_stage_policy()
        config = policy.to_training_config()
        self.assertEqual(policy.sam2_track_config, config.sam2_track_config)
        self.assertEqual(policy.loss_ramp, config.loss_ramp)
        self.assertEqual(policy.pair_schedule, config.pair_schedule)
        self.assertEqual(policy.response_manifest_digest, config.response_manifest_digest)
        self.assertEqual(policy.uncertainty_floors_digest, config.uncertainty_floors_digest)
        for missing in (
            "image_size", "temporal_block_frames",
            "detach_after_observer_positions", "checkpoint_image_encoder",
            "checkpoint_track_step", "checkpoint_memory_encoder",
            "mask_temperature",
        ):
            with self.subTest(missing=missing), self.assertRaisesRegex(ValueError, missing):
                seal_fixture_stage_policy(missing_sam2_field=missing)

    def test_observer_identity_must_be_rebuilt_from_every_runtime_field(self) -> None:
        for changed in (
            "image_size", "temporal_block_frames", "detach_positions",
            "checkpoint_image_encoder", "checkpoint_track_step",
            "checkpoint_memory_encoder", "mask_temperature", "dtype",
            "override", "post_load_mutation",
        ):
            with self.subTest(changed=changed), self.assertRaisesRegex(
                ValueError, "observer identity",
            ):
                verify_fixture_stage_policy(changed_observer_field=changed)

    def test_121_memory_failure_cannot_fall_back_to_32(self) -> None:
        audit = evaluate_fixture_transition(
            previous=physical_32_policy(), candidate=physical_121_policy(),
            preflight=failed_memory_preflight(requested_frames=121),
            gradient_windows=passing_gradient_windows("physical_121"),
        )
        self.assertFalse(audit.passed)
        self.assertEqual(121, audit.requested_observer_frames)

    def test_any_response_test_access_blocks_progression(self) -> None:
        audit = evaluate_fixture_transition(
            response_test_access=fixture_response_test_access(count=1)
        )
        self.assertFalse(audit.passed)
        self.assertIn("response_test_access", audit.failure_reasons)

    def test_coverage_quantity_gradient_and_complete_window_are_mandatory(self) -> None:
        for preflight, windows, reason in (
            (fixture_preflight(observer_coverage=0.2), passing_gradient_windows(), "observer_coverage"),
            (fixture_preflight(both_branch_quantity_gradient=False), passing_gradient_windows(), "missing_quantity_gradient"),
            (fixture_preflight(), (fixture_gradient_window(sample_count=99),), "gradient_window_sample_count"),
            (fixture_preflight(), (fixture_gradient_window(median_ratio=float("nan")),), "gradient_window_nonfinite"),
            (fixture_preflight(), discontinuous_gradient_windows(), "gradient_window_coverage"),
        ):
            audit = evaluate_fixture_transition(
                preflight=preflight, gradient_windows=windows
            )
            self.assertFalse(audit.passed)
            self.assertIn(reason, audit.failure_reasons)

    def test_gradient_window_presence_matches_stage_kind(self) -> None:
        missing = evaluate_fixture_transition(gradient_windows=())
        self.assertFalse(missing.passed)
        self.assertIn("missing_gradient_window", missing.failure_reasons)
        warmup = evaluate_warmup_fixture_transition(gradient_windows=())
        self.assertEqual((), warmup.gradient_window_digests)
        unexpected = evaluate_warmup_fixture_transition(
            gradient_windows=(passing_gradient_window_audit("flowmatch_warmup"),)
        )
        self.assertFalse(unexpected.passed)
        self.assertIn("unexpected_gradient_window", unexpected.failure_reasons)

    def test_teacher_cache_preflight_has_no_model_checkpoint_or_cuda_preflight(self) -> None:
        audit = evaluate_teacher_cache_fixture_transition(
            checkpoint=None,
            teacher_cache_preflight=passing_teacher_cache_preflight(),
            preflight=None,
            optimizer_transition=None,
            pair_overfit=None,
            visual=None,
            response_validation=None,
        )
        self.assertTrue(audit.passed, audit.failure_reasons)
        self.assertIsNone(audit.checkpoint_artifact_digest)
        self.assertIsNone(audit.preflight_digest)
        self.assertIsNotNone(audit.teacher_cache_preflight_digest)

    def test_warmup_requires_checkpoint_and_zero_decode_but_no_physical_audits(self) -> None:
        audit = evaluate_warmup_fixture_transition(
            preflight=fixture_preflight(decoded_frame_count=0, observer_frames=0),
            pair_overfit=None, visual=None, response_validation=None,
        )
        self.assertTrue(audit.passed, audit.failure_reasons)
        self.assertIsNotNone(audit.checkpoint_artifact_digest)

    def test_nonphysical_stage_rejects_even_passing_physical_audit_objects(self) -> None:
        audit = evaluate_warmup_fixture_transition(
            pair_overfit=passing_pair_overfit_audit(),
            visual=passing_visual_audit(),
            response_validation=passing_response_validation_audit(),
        )
        self.assertFalse(audit.passed)
        self.assertIn("unexpected_physical_stage_audit", audit.failure_reasons)

    def test_sigma_selection_is_identical_from_16_through_121(self) -> None:
        audit = evaluate_fixture_transition(
            previous=physical_16_policy(sigma=(0.2, 0.7), selection_digest="sigma-a"),
            candidate=physical_32_policy(sigma=(0.2, 0.8), selection_digest="sigma-b"),
        )
        self.assertFalse(audit.passed)
        self.assertIn("physical_sigma_selection_changed", audit.failure_reasons)

    def test_mutated_inputs_or_foreign_typed_audits_fail_closed(self) -> None:
        for changed, reason in (
            ("response_manifest", "response_manifest_digest"),
            ("coordinate", "coordinate_manifest_digest"),
            ("teacher_train", "teacher_train_binding_digest"),
            ("teacher_validation", "teacher_validation_binding_digest"),
            ("uncertainty_floors", "uncertainty_floors_digest"),
            ("checkpoint", "input_checkpoint_digest"),
            ("runtime", "training_runtime_digest"),
            ("validation_selection", "validation_selection_digest"),
            ("pair_overfit_policy", "pair_overfit_policy_digest"),
            ("visual_checkpoint", "visual_checkpoint_digest"),
            ("response_validation_partition", "response_validation_partition"),
            ("optimizer_transition", "optimizer_transition"),
        ):
            with self.subTest(changed=changed):
                audit = evaluate_fixture_transition(changed=changed)
                self.assertFalse(audit.passed)
                self.assertIn(reason, audit.failure_reasons)

    def test_visual_and_validation_metric_sets_are_exact_and_nonempty(self) -> None:
        for changed in ("empty_visual", "missing_visual", "extra_visual", "empty_validation"):
            audit = evaluate_fixture_transition(changed=changed)
            self.assertFalse(audit.passed)
            self.assertIn("required_metric_set", audit.failure_reasons)
```

- [ ] **Step 2: Run stage tests and observe the missing stage module**

Run:

```bash
PYTHONPATH=src:tests:. /mnt/nvme1/NicoCache/envs/physics_wan/bin/python \
  -m unittest tests.physical_response.test_stages -v
```

Expected: FAIL importing `physbench.physical_response.stages`.

- [ ] **Step 3: Implement sealed policies, rolling medians, and transition audits**

```python
def evaluate_stage_transition(
    previous: StagePolicy | None,
    candidate: StagePolicy,
    checkpoint: StageCheckpointArtifact | None,
    teacher_cache_preflight: TeacherCachePreflightAudit | None,
    preflight: StagePreflight | None,
    gradient_windows: Sequence[GradientWindowAudit],
    pair_overfit: PairOverfitAudit | None,
    visual: VisualNoninferiorityAudit | None,
    response_validation: ResponseValidationAudit | None,
    optimizer_transition: OptimizerTransitionAudit | None,
    response_test_access: ResponseAccessAudit,
) -> StageGateAudit:
    failures: list[str] = []
    expected_previous = {
        "flowmatch_warmup": "teacher_cache_preflight",
        "physical_16": "flowmatch_warmup",
        "physical_32": "physical_16",
        "physical_121": "physical_32",
    }.get(candidate.stage_id)
    if expected_previous is not None and (
        previous is None or previous.stage_id != expected_previous
    ):
        failures.append("stage_sequence")
    if candidate.stage_id == "teacher_cache_preflight" and previous is not None:
        failures.append("stage_sequence")
    if candidate.previous_stage_policy_digest != (
        None if previous is None else previous.digest
    ):
        failures.append("previous_stage_policy_digest")
    if (
        response_test_access.partition != "response_test"
        or response_test_access.access_count != 0
        or response_test_access.digest != candidate.response_test_access_audit_digest
    ):
        failures.append("response_test_access")
    is_teacher_cache = candidate.stage_id == "teacher_cache_preflight"
    is_physical = candidate.stage_id.startswith("physical_")
    physical_audits = (pair_overfit, visual, response_validation)
    if is_teacher_cache:
        if any(item is not None for item in (
            checkpoint, preflight, optimizer_transition, *physical_audits
        )):
            failures.append("unexpected_teacher_cache_stage_artifact")
        if gradient_windows:
            failures.append("unexpected_gradient_window")
        if teacher_cache_preflight is None:
            failures.append("missing_teacher_cache_preflight")
        else:
            teacher_checks = (
                (teacher_cache_preflight.stage_policy_digest, candidate.digest, "teacher_cache_policy_digest"),
                (teacher_cache_preflight.response_manifest_digest, candidate.response_manifest_digest, "response_manifest_digest"),
                (teacher_cache_preflight.coordinate_manifest_digest, candidate.coordinate_manifest_digest, "coordinate_manifest_digest"),
                (teacher_cache_preflight.teacher_train_binding_digest, candidate.teacher_train_binding_digest, "teacher_train_binding_digest"),
                (teacher_cache_preflight.teacher_validation_binding_digest, candidate.teacher_validation_binding_digest, "teacher_validation_binding_digest"),
                (teacher_cache_preflight.observer_identity_digest, candidate.sam2_observer_identity_digest, "sam2_observer_identity_digest"),
            )
            failures.extend(
                reason for actual, expected, reason in teacher_checks
                if actual != expected
            )
            if not teacher_cache_preflight.passed:
                failures.extend(teacher_cache_preflight.failure_reasons)
    else:
        if teacher_cache_preflight is not None:
            failures.append("unexpected_teacher_cache_preflight")
        if checkpoint is None:
            failures.append("missing_stage_checkpoint")
        if preflight is None:
            failures.append("missing_stage_preflight")
        if optimizer_transition is None:
            failures.append("missing_optimizer_transition")
        if checkpoint is not None and preflight is not None:
            identity_checks = (
                (checkpoint.stage_policy_digest, candidate.digest, "checkpoint_policy_digest"),
                (checkpoint.input_checkpoint_digest, candidate.input_checkpoint_digest, "input_checkpoint_digest"),
                (preflight.stage_policy_digest, candidate.digest, "preflight_policy_digest"),
                (preflight.training_config_digest, candidate.to_training_config().digest, "training_config_digest"),
                (preflight.response_manifest_digest, candidate.response_manifest_digest, "response_manifest_digest"),
                (preflight.coordinate_manifest_digest, candidate.coordinate_manifest_digest, "coordinate_manifest_digest"),
                (preflight.teacher_train_binding_digest, candidate.teacher_train_binding_digest, "teacher_train_binding_digest"),
                (preflight.uncertainty_floors_digest, candidate.uncertainty_floors_digest, "uncertainty_floors_digest"),
                (preflight.input_checkpoint_digest, candidate.input_checkpoint_digest, "preflight_checkpoint_digest"),
                (preflight.sam2_observer_identity_digest, candidate.sam2_observer_identity_digest, "sam2_observer_identity_digest"),
                (preflight.sam2_track_config_digest, candidate.sam2_track_config.digest, "sam2_track_config_digest"),
            )
            failures.extend(
                reason for actual, expected, reason in identity_checks
                if actual != expected
            )
            if preflight.observer_frames != candidate.observer_frames:
                failures.append("observer_frame_mismatch")
            if not preflight.finite_loss_and_gradients:
                failures.append("nonfinite_loss_or_gradient")
            if not preflight.memory_fit:
                failures.append("memory_preflight")

    if is_physical and checkpoint is not None and preflight is not None:
        if preflight.decoded_frame_count != 121:
            failures.append("decoded_frame_count")
        if preflight.observer_coverage < candidate.minimum_observer_coverage:
            failures.append("observer_coverage")
        if not preflight.both_branch_lora_gradient:
            failures.append("missing_branch_gradient")
        if not preflight.both_branch_quantity_gradient:
            failures.append("missing_quantity_gradient")
        if any(item is None for item in physical_audits):
            failures.append("missing_physical_stage_audit")
        else:
            assert pair_overfit is not None
            assert visual is not None
            assert response_validation is not None
            physical_identity_checks = (
                (pair_overfit.stage_policy_digest, candidate.digest, "pair_overfit_policy_digest"),
                (pair_overfit.checkpoint_artifact_digest, checkpoint.artifact_digest, "pair_overfit_checkpoint_digest"),
                (visual.stage_policy_digest, candidate.digest, "visual_policy_digest"),
                (visual.checkpoint_artifact_digest, checkpoint.artifact_digest, "visual_checkpoint_digest"),
                (response_validation.stage_policy_digest, candidate.digest, "response_validation_policy_digest"),
                (response_validation.checkpoint_artifact_digest, checkpoint.artifact_digest, "response_validation_checkpoint_digest"),
                (response_validation.teacher_validation_binding_digest, candidate.teacher_validation_binding_digest, "response_validation_teacher_digest"),
                (response_validation.validation_selection_digest, candidate.validation_selection_digest, "response_validation_selection_digest"),
            )
            failures.extend(
                reason for actual, expected, reason in physical_identity_checks
                if actual != expected
            )
            recomputed_gain = (
                (pair_overfit.initial_error - pair_overfit.final_error)
                / pair_overfit.initial_error
                if pair_overfit.initial_error > 0.0 else float("-inf")
            )
            if (
                not pair_overfit.finite
                or not math.isfinite(recomputed_gain)
                or recomputed_gain != pair_overfit.relative_gain
                or recomputed_gain < candidate.minimum_pair_overfit_gain
                or not pair_overfit.object_retained
            ):
                failures.append("pair_overfit_gain")
            required_visual = {
                item.metric_name for item in candidate.visual_noninferiority_gates
            }
            actual_visual = {item.metric_name for item in visual.metrics}
            required_validation = {
                item.metric_name for item in candidate.response_validation_gates
            }
            actual_validation = {
                item.metric_name for item in response_validation.metrics
            }
            if not required_visual or required_visual != actual_visual:
                failures.append("visual_required_metric_set")
            if not required_validation or required_validation != actual_validation:
                failures.append("response_validation_required_metric_set")
            if not visual.passed or any(not item.passed for item in visual.metrics):
                failures.append("visual_noninferiority")
            if (
                response_validation.source_partition != "response_validation"
                or not response_validation.passed
                or any(not item.passed for item in response_validation.metrics)
            ):
                failures.append("response_validation")

        expected_windows = candidate.planned_updates // 100
        ordered = sorted(gradient_windows, key=lambda item: item.window_index)
        if not ordered:
            failures.append("missing_gradient_window")
        if len(ordered) != expected_windows:
            failures.append("gradient_window_coverage")
        for index, window in enumerate(ordered):
            if window.stage_policy_digest != candidate.digest:
                failures.append("gradient_window_policy_digest")
            if (
                window.sample_count != 100
                or window.first_update_index != 100 * index
                or window.last_update_index != 100 * index + 99
            ):
                failures.append("gradient_window_sample_count")
            if not window.finite:
                failures.append("gradient_window_nonfinite")
            if not window.passed:
                failures.extend(window.failure_reasons)
    elif not is_teacher_cache and preflight is not None:
        if preflight.decoded_frame_count != 0 or preflight.observer_frames != 0:
            failures.append("unexpected_physical_decode")
        if gradient_windows:
            failures.append("unexpected_gradient_window")
        if any(item is not None for item in physical_audits):
            failures.append("unexpected_physical_stage_audit")
        if candidate.visual_noninferiority_gates or candidate.response_validation_gates:
            failures.append("unexpected_physical_gate_spec")

    if not is_teacher_cache and optimizer_transition is not None:
        if (
            optimizer_transition.candidate_stage_policy_digest != candidate.digest
            or optimizer_transition.declared_transition != candidate.optimizer_transition
            or optimizer_transition.realized_transition != candidate.optimizer_transition
            or not optimizer_transition.passed
        ):
            failures.append("optimizer_transition")
    if previous is not None and previous.stage_id.startswith("physical_"):
        if (
            candidate.physical_sigma_min != previous.physical_sigma_min
            or candidate.physical_sigma_max != previous.physical_sigma_max
            or candidate.physical_sigma_selection_digest
            != previous.physical_sigma_selection_digest
        ):
            failures.append("physical_sigma_selection_changed")
    if candidate.stage_id == "physical_121" and previous is not None:
        if candidate.learning_rate >= previous.learning_rate:
            failures.append("dense_learning_rate_not_lower")
    return StageGateAudit.from_verified_artifacts(
        candidate=candidate, checkpoint=checkpoint,
        teacher_cache_preflight=teacher_cache_preflight, preflight=preflight,
        gradient_windows=tuple(gradient_windows), pair_overfit=pair_overfit,
        visual=visual, response_validation=response_validation,
        optimizer_transition=optimizer_transition,
        response_test_access=response_test_access,
        failure_reasons=tuple(sorted(set(failures))),
    )
```

The teacher/cache stage is a zero-update artifact stage: its policy requires `previous=None`, `planned_updates=0`, `observer_frames=0`, and `input_checkpoint_digest=None`; it produces only `TeacherCachePreflightAudit`, with no model allocation, training config, CUDA `StagePreflight`, optimizer transition, or checkpoint. `StagePolicy.to_training_config` rejects this stage. FlowMatch warmup requires a positive update count, produces a normal checkpoint and zero-decode `StagePreflight` from a factual batch, and has physical weights, observer frames, physical audits, and gradient windows all absent. Each physical `StagePreflight` records allocated/reserved peak CUDA bytes, wall time, exactly 121 decoded frames, requested observer count and coverage, the full SAM2 configuration/identity, frozen uncertainty floors, both-branch LoRA/QuantityEncoder gradients, component/global norms, realized physical/FM ratio, and every upstream artifact identity. For warmup and physical stages, `StagePolicy.to_training_config` is field-for-field round-trip tested for schedule, ramp, gradient gates, full SAM2 config, event parameters, validity scales, uncertainty floors, and identities. Every physical policy requires a positive `planned_updates >= 100` with `planned_updates % 100 == 0`, so its complete set of contiguous 100-update gradient windows is unambiguous. The transition recomputes typed overfit gain, object retention, each visual margin, each response-validation threshold, optimizer continuity/reset, and exact window coverage; missing, empty, extra, foreign, non-finite, or caller-claimed-only evidence fails. Warmup fixes every physical weight to zero. `physical_16` enables ramped absolute/sign/validity/analytic terms; `physical_32` adds magnitude, zero, and event-aligned terms; `physical_121` retains all terms and must use a lower learning rate. The sigma interval and its validation-selection identity are selected once at 16 frames and inherited exactly through 32/121.

The sealer reconstructs `SAM2ObserverIdentity` from the verified Task-14 `TrainingRuntimeConfig`, the exact candidate `SAM2TrackConfig`, fixed overrides, dtype, and post-load mutations. It requires the train and validation teacher-binding headers to contain that same observer identity and rejects unless `observer_identity.track_config_digest == sam2_track_config.digest`; header checks do not open records and no caller-supplied observer identity is trusted. It calls `load_uncertainty_floors` with the verified response-manifest and teacher-train-binding identities, requires `source_partition="response_train"`, and records the resulting identity in the stage policy, training config, and preflight. The candidate schema has no identity fields and no numeric defaults.

- [ ] **Step 4: Run unit tests, all three CUDA preflights, and the synchronized fatal gate**

Run:

```bash
PYTHONPATH=src:tests:. /mnt/nvme1/NicoCache/envs/physics_wan/bin/python \
  -m unittest tests.physical_response.test_stages -v
CUDA_VISIBLE_DEVICES=0,1 \
PYTHONPATH=src:.:/root/Nico/third_party/sam2:/root/Steven/wan22_pendulum_pipeline/vendor/DiffSynth-Studio \
  /mnt/nvme1/NicoCache/envs/physics_wan/bin/torchrun \
  --standalone --nproc_per_node=2 \
  tests/integration/physical_response_end_to_end_gradient_worker.py \
  --observer-frames 16 \
  --run-pair-overfit-probe \
  --run-visual-and-response-validation-audits \
  --audit-output /tmp/physical_response_preflight_16.json
CUDA_VISIBLE_DEVICES=0,1 \
PYTHONPATH=src:.:/root/Nico/third_party/sam2:/root/Steven/wan22_pendulum_pipeline/vendor/DiffSynth-Studio \
  /mnt/nvme1/NicoCache/envs/physics_wan/bin/torchrun \
  --standalone --nproc_per_node=2 \
  tests/integration/physical_response_end_to_end_gradient_worker.py \
  --observer-frames 32 \
  --run-pair-overfit-probe \
  --run-visual-and-response-validation-audits \
  --audit-output /tmp/physical_response_preflight_32.json
CUDA_VISIBLE_DEVICES=0,1 \
PYTHONPATH=src:.:/root/Nico/third_party/sam2:/root/Steven/wan22_pendulum_pipeline/vendor/DiffSynth-Studio \
  /mnt/nvme1/NicoCache/envs/physics_wan/bin/torchrun \
  --standalone --nproc_per_node=2 \
  tests/integration/physical_response_end_to_end_gradient_worker.py \
  --observer-frames 121 \
  --run-pair-overfit-probe \
  --run-visual-and-response-validation-audits \
  --allow-memory-blocked \
  --audit-output /tmp/physical_response_preflight_121.json
CUDA_VISIBLE_DEVICES=0,1 \
PYTHONPATH=src:.:/root/Nico/third_party/sam2:/root/Steven/wan22_pendulum_pipeline/vendor/DiffSynth-Studio \
  /mnt/nvme1/NicoCache/envs/physics_wan/bin/torchrun \
  --standalone --nproc_per_node=2 \
  tests/integration/physical_response_end_to_end_gradient_worker.py \
  --observer-frames 16 \
  --inject-nonfinite-ratio-at-update 3 \
  --expect-blocked-reason nonfinite_gradient_ratio_or_norm_before_optimizer_step \
  --audit-output /tmp/physical_response_nonfinite_gate.json
```

Expected: unit tests PASS; 16 and 32 write complete finite preflight/gate artifacts, including direction-and-magnitude pair-overfit plus object-retention evidence; 121 either passes at 121 or records only the explicit memory-blocked outcome; the injected non-finite run exits zero only when both ranks skip optimizer update 3 with the same failure reason.

- [ ] **Step 5: Commit stage sealing and preflights**

```bash
git add src/physbench/physical_response/stages.py \
  schemas/physical_response_stage_policy.schema.json \
  scripts/seal_physical_response_stage.py \
  scripts/preflight_physical_response.py \
  tests/physical_response/test_stages.py \
  tests/integration/physical_response_end_to_end_gradient_worker.py
git commit -m "feat: seal physical response training stages"
```

### Task 16: Compile exactly five common-noise pair jobs and run pair-affine inference

**Files:**
- Modify: `src/physbench/physical_response/sampling.py`
- Create: `src/physbench/physical_response/generation_runtime.py`
- Create: `schemas/physical_response_generation_runtime.schema.json`
- Create: `configs/physical_response/runtime/generation_smoke.example.json`
- Create: `scripts/materialize_physical_response_generation_runtime.py`
- Create: `scripts/compile_physical_response_jobs.py`
- Create: `scripts/wan22_physical_response_generate_batch.py`
- Modify: `tests/physical_response/_fixtures.py`
- Create: `tests/physical_response/test_generation_runtime.py`
- Create: `tests/physical_response/test_common_seed_generation.py`
- Create: `tests/integration/physical_response_generation_worker.py`

**Interfaces:**
- Consumes: sealed response-test edges, `ConditionIndex`, and `CoordinateFrameIndex` from Tasks 3/4; stage-selected model artifacts from Task 15; existing WAN noise initialization and authenticated combined-checkpoint loader at `src/physbench/baselines/wan22_quantity.py:646-702`. Common scientific jobs are compiled exactly once per holdout and contain no arm, checkpoint, runtime, worker, or output identity.
- Produces: `GenerationRuntimeConfig`, `ResolvedGenerationRuntime`, `ModelSelectionRef`, `ArmGenerationBinding`, `PipelineLoadAudit`, `CommonPairJob`, `CommonJobManifest`, `EndpointGenerationResult`, `PairGenerationResult`, and `ArmGenerationResultsManifest`; `load_generation_runtime_config(path: Path) -> GenerationRuntimeConfig`; `resolve_and_verify_generation_runtime(config: GenerationRuntimeConfig) -> ResolvedGenerationRuntime`; `seal_arm_generation_binding(*, run_id: str, experiment_id: str, arm_id: str, arm_spec_digest: str, arm_run_artifact_digest: str, model_selection: ModelSelectionRef, common: CommonJobManifest, runtime: ResolvedGenerationRuntime, run_dir: Path, output_path: Path) -> ArmGenerationBinding`; `load_verified_generation_pipeline(runtime: ResolvedGenerationRuntime, binding: ArmGenerationBinding, *, run_dir: Path, device: torch.device, allocator: Callable[..., Any]) -> tuple[Any, PipelineLoadAudit]`; `derive_primary_seed_set(*, master_seed: int) -> tuple[int, int, int, int, int]`; `scientific_job_set_digest(jobs: Sequence[CommonPairJob]) -> str`; `compile_common_response_jobs(manifest: ResponseManifest, condition_index: ConditionIndex, coordinate_index: CoordinateFrameIndex, *, master_seed: int, output_dir: Path) -> CommonJobManifest`; `load_common_pair_jobs(manifest_path: Path) -> tuple[CommonJobManifest, tuple[CommonPairJob, ...]]`; `initial_noise_digest(*, seed: int, latent_shape: tuple[int, ...], dtype: torch.dtype) -> str`; `run_pair_generation_job(pipe: Any, job: CommonPairJob, binding: ArmGenerationBinding, runtime: ResolvedGenerationRuntime, *, output_dir: Path) -> PairGenerationResult`; `verify_pair_generation_result(result: PairGenerationResult, *, job: CommonPairJob, binding: ArmGenerationBinding, runtime: ResolvedGenerationRuntime) -> None`; and `write_arm_generation_results(results: Sequence[PairGenerationResult], *, common: CommonJobManifest, binding: ArmGenerationBinding, output_path: Path) -> ArmGenerationResultsManifest`.
- CLI contracts: `compile_physical_response_jobs.py --manifest run/physical_response_group_holdout_v1_seed42/physical_response/manifest.json --condition-index run/physical_response_group_holdout_v1_seed42/physical_response/condition_index.jsonl --coordinate-binding run/physical_response_group_holdout_v1_seed42/physical_response/coordinates/binding.json --coordinate-cache-root cache/physical_response/coordinates_v1 --master-seed 42 --output-dir run/physical_response_group_holdout_v1_seed42/physical_response/generation/common`; pair-affine worker `wan22_physical_response_generate_batch.py --run-dir run/physical_response_group_holdout_v1_seed42 --generation-runtime configs/physical_response/runtime/generation_local.json --common-jobs run/physical_response_group_holdout_v1_seed42/physical_response/generation/common/manifest.json --arm-binding run/physical_response_group_holdout_v1_seed42/arms/complete_direct_rgb/generation_binding.json --output-dir run/physical_response_group_holdout_v1_seed42/physical_response/generation/complete_direct_rgb --results-manifest run/physical_response_group_holdout_v1_seed42/physical_response/generation/complete_direct_rgb/manifest.json --gpu 0`.
- Runtime contracts: `GenerationRuntimeConfig(schema_version: Literal["physical_response_generation_runtime_v1"], interpreter: Path, wan_root: Path, diffsynth_root: Path, baseline_bundle: Path, quantity_registry: Path, torch_dtype: Literal["bfloat16"], pipeline_loader_version: str, source_path: Path, digest: str)`; `ResolvedGenerationRuntime(config_digest: str, interpreter_path: str, wan_inventory_digest: str, diffsynth_commit: str, diffsynth_inventory_digest: str, baseline_digest: str, quantity_registry_digest: str, torch_cuda_identity_digest: str, digest: str)`; `ModelSelectionRef(load_mode: Literal["base_only", "lora_only", "lora_quantity"], checkpoint_relative_path: str | None, checkpoint_manifest_relative_path: str | None, checkpoint_sha256: str | None, checkpoint_inventory_digest: str, base_model_inventory_digest: str, digest: str)`; `ArmGenerationBinding(run_id: str, experiment_id: str, arm_id: str, arm_spec_digest: str, arm_run_artifact_digest: str, model_selection: ModelSelectionRef, common_job_manifest_digest: str, runtime_identity_digest: str, digest: str)`; and `PipelineLoadAudit(arm_id: str, model_selection_digest: str, runtime_identity_digest: str, checkpoint_sha256: str | None, checkpoint_load_mode: str, same_bytes_load_boundary_verified: bool, passed: bool, digest: str)`.
- Scientific contracts: `PairEndpointCondition(branch: Literal["a", "b"], q_si: str, source_case_id: str | None, anchor_case_id: str, condition_payload_digest: str, prompt: str, quantities: tuple[dict[str, Any], ...], active_quantity: ActiveQuantityBinding, first_frame_path: Path, first_frame_sha256: str, mask_manifest_path: Path, mask_manifest_sha256: str, initial_mask_path: Path, initial_mask_sha256: str, coordinate_frame_id: str, coordinate_record_digest: str)`; `CommonPairJob(job_id: str, experiment_id: str, group_id: str, edge_id: str, edge_record_digest: str, edge_kind: EdgeKind, target_kind: ResponseTargetKind, interpolation_kind: InterpolationKind, scene_id: str, axis_id: str, seed: int, frame_kind: Literal["corresponding_real", "shared_anchor"], factual_branch: Literal["a", "b"] | None, latent_shape: tuple[int, int, int, int], expected_initial_noise_sha256: str, condition_index_digest: str, coordinate_manifest_digest: str, response_manifest_digest: str, policy_digest: str, collision_topology: CollisionTopologyBinding | None, endpoint_a: PairEndpointCondition, endpoint_b: PairEndpointCondition, digest: str)`; and `CommonJobManifest(schema_version: Literal["physical_response_common_jobs_v1"], experiment_id: str, response_manifest_digest: str, condition_index_digest: str, coordinate_manifest_digest: str, policy_digest: str, master_seed: int, seed_set: tuple[int, int, int, int, int], jobs_relative_path: str, jobs_sha256: str, job_count: int, scientific_job_set_digest: str, edge_seed_keys_digest: str, digest: str)`. Because `CommonPairJob` is already arm-independent, `scientific_job_set_digest` is the canonical identity of the complete ordered job documents, not a projection that drops fields.
- Result contracts: `EndpointGenerationResult(branch: Literal["a", "b"], anchor_case_id: str, condition_payload_digest: str, video_relative_path: str, video_sha256: str, first_frame_sha256: str, mask_manifest_sha256: str, initial_mask_sha256: str, initial_noise_sha256: str, worker_id: str, endpoint_digest: str)`; `PairGenerationResult(job_id: str, common_job_digest: str, arm_id: str, arm_run_artifact_digest: str, runtime_identity_digest: str, endpoint_a: EndpointGenerationResult, endpoint_b: EndpointGenerationResult, result_digest: str)`; and `ArmGenerationResultsManifest(arm_id: str, arm_run_artifact_digest: str, common_job_manifest_digest: str, runtime_identity_digest: str, results_relative_path: str, results_sha256: str, expected_job_count: int, completed_job_count: int, completed_edge_seed_keys_digest: str, result_digests: tuple[str, ...], digest: str)`. `PairGenerationResult.from_endpoints(*, job: CommonPairJob, binding: ArmGenerationBinding, runtime: ResolvedGenerationRuntime, endpoint_a: EndpointGenerationResult, endpoint_b: EndpointGenerationResult) -> PairGenerationResult` is the only public constructor.

- [ ] **Step 1: Write runtime-boundary, exact-five, endpoint-affinity, and result-factory tests**

```python
class GenerationRuntimeTests(unittest.TestCase):
    def test_worker_rejects_every_identity_change_before_pipe_allocation(self) -> None:
        for changed in (
            "runtime", "wan_inventory", "checkpoint_manifest", "checkpoint_bytes",
            "load_mode", "base_inventory", "path_escape",
        ):
            allocator = mock.Mock()
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                load_fixture_generation_pipeline(changed=changed, allocator=allocator)
            allocator.assert_not_called()

    def test_combined_checkpoint_uses_authenticated_same_bytes_loader(self) -> None:
        loader = mock.Mock(return_value=fixture_loaded_pipeline())
        _pipe, audit = load_fixture_generation_pipeline(
            changed=None, load_mode="lora_quantity", combined_loader=loader,
        )
        loader.assert_called_once()
        self.assertTrue(audit.same_bytes_load_boundary_verified)
        self.assertTrue(audit.passed)

    def test_arm_binding_is_constructed_only_from_verified_model_jobs_and_runtime(self) -> None:
        binding = seal_fixture_arm_generation_binding()
        self.assertEqual(fixture_common_jobs().digest, binding.common_job_manifest_digest)
        self.assertEqual(fixture_resolved_generation_runtime().digest, binding.runtime_identity_digest)
        for changed in ("arm", "model_selection", "common_jobs", "runtime", "path_escape"):
            output = fixture_output_path(f"binding-{changed}.json")
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                seal_fixture_arm_generation_binding(changed=changed, output_path=output)
            self.assertFalse(output.exists())

class CommonSeedGenerationTests(unittest.TestCase):
    def test_every_edge_has_exactly_five_shared_seeds(self) -> None:
        manifest, jobs = compile_fixture_common_jobs(edge_count=3, master_seed=42)
        by_edge = group_jobs_by_edge(jobs)
        self.assertEqual({5}, {len(edge_jobs) for edge_jobs in by_edge.values()})
        for edge_jobs in by_edge.values():
            self.assertEqual(5, len({job.seed for job in edge_jobs}))
        self.assertEqual(len(jobs), manifest.job_count)

    def test_common_jobs_are_compiled_once_and_have_no_arm_fields(self) -> None:
        first = compile_fixture_common_jobs(master_seed=42)
        second = compile_fixture_common_jobs(master_seed=42)
        self.assertEqual(first.manifest_bytes, second.manifest_bytes)
        for forbidden in ("arm_id", "checkpoint", "output_path"):
            self.assertNotIn(forbidden, CommonPairJob.__dataclass_fields__)
        self.assertEqual(
            scientific_job_set_digest(first.jobs),
            first.manifest.scientific_job_set_digest,
        )

    def test_pair_worker_keeps_both_endpoints_noise_and_coordinate_together(self) -> None:
        result = run_fixture_pair_generation(frame_kind="shared_anchor")
        self.assertEqual(result.endpoint_a.worker_id, result.endpoint_b.worker_id)
        self.assertEqual(result.endpoint_a.initial_noise_sha256, result.endpoint_b.initial_noise_sha256)
        self.assertEqual(result.endpoint_a.first_frame_sha256, result.endpoint_b.first_frame_sha256)
        self.assertEqual(result.endpoint_a.initial_mask_sha256, result.endpoint_b.initial_mask_sha256)
        self.assertEqual(result.endpoint_a.mask_manifest_sha256, result.endpoint_b.mask_manifest_sha256)
        self.assertEqual(result.endpoint_a.anchor_case_id, result.endpoint_b.anchor_case_id)
        job = fixture_common_job(frame_kind="shared_anchor")
        self.assertEqual(
            job.endpoint_a.coordinate_record_digest,
            job.endpoint_b.coordinate_record_digest,
        )

    def test_corresponding_real_pair_shares_noise_but_not_real_inputs(self) -> None:
        result = run_fixture_pair_generation(frame_kind="corresponding_real")
        self.assertEqual(result.endpoint_a.initial_noise_sha256, result.endpoint_b.initial_noise_sha256)
        self.assertNotEqual(result.endpoint_a.first_frame_sha256, result.endpoint_b.first_frame_sha256)
        self.assertNotEqual(result.endpoint_a.initial_mask_sha256, result.endpoint_b.initial_mask_sha256)
        self.assertNotEqual(result.endpoint_a.anchor_case_id, result.endpoint_b.anchor_case_id)
        job = fixture_common_job(frame_kind="corresponding_real")
        self.assertNotEqual(
            job.endpoint_a.coordinate_record_digest,
            job.endpoint_b.coordinate_record_digest,
        )

    def test_jobs_carry_all_scientific_strata_and_input_identities(self) -> None:
        _, jobs = compile_fixture_common_jobs(edge_count=1, master_seed=42)
        job = jobs[0]
        self.assertEqual("value_holdout_v1", job.experiment_id)
        self.assertEqual("interpolation", job.interpolation_kind)
        self.assertEqual(fixture_condition_index().digest, job.condition_index_digest)
        self.assertEqual(fixture_coordinate_index().digest, job.coordinate_manifest_digest)
        self.assertTrue(job.endpoint_a.initial_mask_path.is_file())

    def test_worker_rejects_runtime_or_checkpoint_before_pipe_allocation(self) -> None:
        allocator = mock.Mock()
        with self.assertRaisesRegex(ValueError, "runtime identity mismatch"):
            load_fixture_generation_pipeline(changed="runtime", allocator=allocator)
        allocator.assert_not_called()

    def test_result_factory_populates_and_verifies_result_identity(self) -> None:
        result = PairGenerationResult.from_endpoints(**fixture_result_parts())
        verify_pair_generation_result(result, **fixture_result_expectations())
        for changed in ("video", "noise", "mask", "condition", "arm", "runtime"):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                verify_tampered_pair_generation_result(result, changed)

    def test_results_cover_exactly_the_common_edge_seed_keys(self) -> None:
        common, _jobs = compile_fixture_common_jobs(edge_count=3)
        results = build_fixture_generation_results(common)
        self.assertEqual(common.job_count, results.completed_job_count)
        self.assertEqual(
            common.edge_seed_keys_digest, results.completed_edge_seed_keys_digest
        )
```

- [ ] **Step 2: Run runtime and common-seed tests and observe missing generation contracts**

Run:

```bash
PYTHONPATH=src:tests:.:/root/Steven/wan22_pendulum_pipeline/vendor/DiffSynth-Studio \
  /mnt/nvme1/NicoCache/envs/physics_wan/bin/python -m unittest \
  tests.physical_response.test_generation_runtime \
  tests.physical_response.test_common_seed_generation -v
```

Expected: FAIL importing the new job/compiler symbols.

- [ ] **Step 3: Implement deterministic common jobs, verify-before-allocation loading, and the result factory**

```python
PRIMARY_SEED_OFFSETS = (0, 1009, 2027, 3041, 4051)

def derive_primary_seed_set(
    *,
    master_seed: int,
) -> tuple[int, int, int, int, int]:
    if master_seed < 0 or master_seed + PRIMARY_SEED_OFFSETS[-1] >= 2**31:
        raise ValueError("master seed is outside the supported range")
    return cast(
        tuple[int, int, int, int, int],
        tuple(master_seed + offset for offset in PRIMARY_SEED_OFFSETS),
    )

def load_verified_generation_pipeline(
    runtime: ResolvedGenerationRuntime,
    binding: ArmGenerationBinding,
    *,
    run_dir: Path,
    device: torch.device,
    allocator: Callable[..., Any],
) -> tuple[Any, PipelineLoadAudit]:
    selection = binding.model_selection
    verify_binding_runtime_and_common_jobs(binding, runtime)
    resolved = resolve_selection_paths_without_escape(run_dir, selection)
    verify_base_model_inventory(runtime, selection.base_model_inventory_digest)
    verified_checkpoint = verify_checkpoint_manifest_and_tensor_inventory(
        resolved, selection
    )
    pipe, same_bytes = allocator(
        runtime=runtime,
        selection=selection,
        verified_checkpoint=verified_checkpoint,
        device=device,
    )
    audit = build_pipeline_load_audit(
        binding=binding,
        runtime=runtime,
        selection=selection,
        same_bytes_load_boundary_verified=same_bytes,
    )
    if not audit.passed:
        raise RuntimeError("pipeline load audit failed")
    return pipe, audit

def seal_arm_generation_binding(
    *,
    run_id: str,
    experiment_id: str,
    arm_id: str,
    arm_spec_digest: str,
    arm_run_artifact_digest: str,
    model_selection: ModelSelectionRef,
    common: CommonJobManifest,
    runtime: ResolvedGenerationRuntime,
    run_dir: Path,
    output_path: Path,
) -> ArmGenerationBinding:
    verify_model_selection_paths_and_inventory(
        model_selection, run_dir=run_dir, runtime=runtime
    )
    verify_common_job_manifest(common)
    binding = ArmGenerationBinding.from_verified_fields(
        run_id=run_id, experiment_id=experiment_id, arm_id=arm_id,
        arm_spec_digest=arm_spec_digest,
        arm_run_artifact_digest=arm_run_artifact_digest,
        model_selection=model_selection,
        common_job_manifest_digest=common.digest,
        runtime_identity_digest=runtime.digest,
    )
    write_new_typed_document(output_path, binding)
    return binding

@dataclass(frozen=True, slots=True)
class PairGenerationResult:
    job_id: str
    common_job_digest: str
    arm_id: str
    arm_run_artifact_digest: str
    runtime_identity_digest: str
    endpoint_a: EndpointGenerationResult
    endpoint_b: EndpointGenerationResult
    result_digest: str

    @classmethod
    def from_endpoints(
        cls,
        *,
        job: CommonPairJob,
        binding: ArmGenerationBinding,
        runtime: ResolvedGenerationRuntime,
        endpoint_a: EndpointGenerationResult,
        endpoint_b: EndpointGenerationResult,
    ) -> "PairGenerationResult":
        verify_endpoint_result_against_job(endpoint_a, job.endpoint_a)
        verify_endpoint_result_against_job(endpoint_b, job.endpoint_b)
        if endpoint_a.initial_noise_sha256 != endpoint_b.initial_noise_sha256:
            raise ValueError("pair endpoint initial noise differs")
        fields = dict(
            job_id=job.job_id,
            common_job_digest=job.digest,
            arm_id=binding.arm_id,
            arm_run_artifact_digest=binding.arm_run_artifact_digest,
            runtime_identity_digest=runtime.digest,
            endpoint_a=endpoint_a,
            endpoint_b=endpoint_b,
        )
        return cls(**fields, result_digest=canonical_sha256(fields))

def run_pair_generation_job(
    pipe: Any,
    job: CommonPairJob,
    binding: ArmGenerationBinding,
    runtime: ResolvedGenerationRuntime,
    *,
    output_dir: Path,
) -> PairGenerationResult:
    verify_endpoint_condition_bytes(job.endpoint_a)
    verify_endpoint_condition_bytes(job.endpoint_b)
    expected_noise = initial_noise_digest(
        seed=job.seed, latent_shape=job.latent_shape, dtype=pipe.torch_dtype
    )
    if expected_noise != job.expected_initial_noise_sha256:
        raise ValueError("initial noise digest mismatch")
    endpoint_a = generate_and_hash_endpoint(pipe, job.endpoint_a, job.seed, output_dir)
    endpoint_b = generate_and_hash_endpoint(pipe, job.endpoint_b, job.seed, output_dir)
    if endpoint_a.initial_noise_sha256 != endpoint_b.initial_noise_sha256:
        raise RuntimeError("pair endpoints did not share initial noise")
    for expected, actual in (
        (job.endpoint_a.initial_mask_sha256, endpoint_a.initial_mask_sha256),
        (job.endpoint_b.initial_mask_sha256, endpoint_b.initial_mask_sha256),
        (job.endpoint_a.condition_payload_digest, endpoint_a.condition_payload_digest),
        (job.endpoint_b.condition_payload_digest, endpoint_b.condition_payload_digest),
    ):
        if expected != actual:
            raise RuntimeError("endpoint identity digest mismatch")
    return PairGenerationResult.from_endpoints(
        job=job,
        binding=binding,
        runtime=runtime,
        endpoint_a=endpoint_a,
        endpoint_b=endpoint_b,
    )
```

The checked-in generation smoke config is portable and contains no workstation-specific absolute path:

```json
{
  "schema_version": "physical_response_generation_runtime_v1",
  "interpreter": "../external/physics_wan/bin/python",
  "wan_root": "../external/Wan2.2-TI2V-5B",
  "diffsynth_root": "../external/DiffSynth-Studio",
  "baseline_bundle": "../../../baselines/wan22_quantity_embedding/baseline.json",
  "quantity_registry": "../../../baselines/wan22_quantity_embedding/quantity_registry_v2.json",
  "torch_dtype": "bfloat16",
  "pipeline_loader_version": "wan22_quantity_authenticated_v1"
}
```

The compiler verifies the response-manifest, condition-index, and coordinate-binding identities before constructing an endpoint. It fails if the primary seed set has any size other than five, an edge lacks either endpoint, any required first-frame/mask/coordinate byte is absent or mismatched, pair canvases or latent shapes differ, a shared-anchor image/mask/coordinate identity differs, or a corresponding-real endpoint uses anything other than its own case's image, mask, and coordinate record. It copies scene, axis, holdout classification, edge kind, target kind, and interpolation kind directly from the immutable `ResponseEdge`; a `not_estimable` edge may be generated for coverage but can never yield a numeric scientific target. It emits one arm-independent JSONL record per pair-seed and proves exactly `5 * edge_count` complete keys. The collision compiler additionally re-verifies that a counterfactual changes only the bound ball mass and that formula object indices do not follow left/right image swaps.

Task 16 owns and parses only `GenerationRuntimeConfig`; it has no dependency on the Task-18 orchestration runtime. `seal_arm_generation_binding` is the sole binding constructor: it verifies the model-selection inventory and run-relative containment plus the exact common-job and resolved-runtime identities before exclusively writing a typed binding. Task 18 must call this constructor for each verified arm rather than synthesizing binding documents. The batch worker then loads and validates runtime config, common manifest/jobs, arm binding, runtime identity, path containment, base-model inventory, checkpoint manifest, checkpoint bytes, and tensor inventory before calling the allocator. `base_only` validates the complete WAN inventory and has no adapter path; `lora_only` and `lora_quantity` use verified loaders, with `lora_quantity` reusing `load_verified_combined_quantity_checkpoint` so the authenticated bytes are the bytes loaded. Only after this boundary may GPU model allocation occur. Endpoint output paths are derived as `{output_dir}/{job_id}/{branch}.mp4`, never read from the scientific job. `write_arm_generation_results` rejects duplicates, missing jobs, extra jobs, or any edge-seed key set unequal to the common manifest. Extra exploratory seeds use a separately named supplemental manifest and never enter the primary aggregate.

- [ ] **Step 4: Run unit tests and a real two-job generation worker**

Run:

```bash
PYTHONPATH=src:tests:.:/root/Steven/wan22_pendulum_pipeline/vendor/DiffSynth-Studio \
  /mnt/nvme1/NicoCache/envs/physics_wan/bin/python -m unittest \
  tests.physical_response.test_generation_runtime \
  tests.physical_response.test_common_seed_generation -v
PYTHONPATH=src:. /mnt/nvme1/NicoCache/envs/physics_wan/bin/python \
  scripts/materialize_physical_response_generation_runtime.py \
  --interpreter /mnt/nvme1/NicoCache/envs/physics_wan/bin/python \
  --wan-root /root/Steven/wan22_pendulum_pipeline/models/Wan-AI/Wan2.2-TI2V-5B \
  --diffsynth-root /root/Steven/wan22_pendulum_pipeline/vendor/DiffSynth-Studio \
  --baseline-bundle /root/Steven/VPhysBench/baselines/wan22_quantity_embedding/baseline.json \
  --quantity-registry /root/Steven/VPhysBench/baselines/wan22_quantity_embedding/quantity_registry_v2.json \
  --output /tmp/physical_response_generation_runtime.json
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=src:tests:.:/root/Steven/wan22_pendulum_pipeline/vendor/DiffSynth-Studio \
  /mnt/nvme1/NicoCache/envs/physics_wan/bin/python \
  tests/integration/physical_response_generation_worker.py \
  --dataset datasets/releases/12.0.0/dataset.json \
  --runtime-config /tmp/physical_response_generation_runtime.json \
  --job-count 2 \
  --output-root /tmp \
  --audit-output /tmp/physical_response_generation_two_job_audit.json
```

Expected: unit tests PASS. The integration worker uses `load_mode="base_only"`, allocates the real WAN pipeline only after all negative pre-allocation probes pass, generates and reopens four 121-frame endpoint videos for two pair jobs, verifies shared noise within each pair and complete result-factory identities, writes an audit with `completed_job_count == 2`, and exits zero. A missing model, runtime mismatch, allocation failure, non-121-frame output, or endpoint verification failure is a hard failure rather than a skipped smoke test.

- [ ] **Step 5: Commit common-random-number generation**

```bash
git add src/physbench/physical_response/sampling.py \
  src/physbench/physical_response/generation_runtime.py \
  schemas/physical_response_generation_runtime.schema.json \
  configs/physical_response/runtime/generation_smoke.example.json \
  scripts/materialize_physical_response_generation_runtime.py \
  scripts/compile_physical_response_jobs.py \
  scripts/wan22_physical_response_generate_batch.py \
  tests/physical_response/_fixtures.py \
  tests/physical_response/test_generation_runtime.py \
  tests/physical_response/test_common_seed_generation.py \
  tests/integration/physical_response_generation_worker.py
git commit -m "feat: generate response pairs with common noise"
```

### Task 17: Evaluate saved RGB independently and make the acceptance decision

**Files:**
- Create: `src/physbench/physical_response/evaluation_inputs.py`
- Create: `src/physbench/physical_response/saved_rgb_observer.py`
- Create: `src/physbench/physical_response/metrics.py`
- Create: `scripts/build_physical_response_evaluation_targets.py`
- Create: `scripts/seal_physical_response_evaluation_input.py`
- Create: `scripts/audit_physical_response_observer_validation.py`
- Create: `scripts/evaluate_physical_response.py`
- Create: `scripts/seal_physical_response_evaluation_policy.py`
- Create: `schemas/physical_response_evaluation_policy.schema.json`
- Create: `schemas/physical_response_evaluation_input.schema.json`
- Create: `schemas/physical_response_evaluation_artifact.schema.json`
- Create: `schemas/physical_response_evaluation_authorization.schema.json`
- Create: `schemas/physical_response_observer_agreement.schema.json`
- Create: `schemas/physical_response_observer_validation.schema.json`
- Create: `schemas/physical_response_matrix_test_seal.schema.json`
- Modify: `tests/physical_response/_fixtures.py`
- Create: `tests/physical_response/test_evaluation_inputs.py`
- Create: `tests/physical_response/test_saved_rgb_observer.py`
- Create: `tests/physical_response/test_metrics.py`
- Create: `tests/physical_response/test_acceptance.py`
- Create: `tests/integration/physical_response_saved_rgb_evaluator_worker.py`

**Interfaces:**
- Consumes: the one `CommonJobManifest`, common jobs, arm binding, and complete `ArmGenerationResultsManifest` from Task 16; the response policy/manifest and `ConditionIndex`; the Task-4 coordinate binding; the response-test teacher binding from Task 9; the Task-11 frozen train-only uncertainty floors; a validation-sealed `EvaluationPolicy`; a zero-access `MatrixTestSeal`; and existing visual/entity evaluators through `physbench.evaluation.registry.SceneEvaluatorRegistry`.
- Produces: `ObserverValidationAudit`, `EvaluationPolicy`, `MatrixTestSeal`, `EvaluationAuthorization`, `EvaluationArtifactRole`, `EvaluationArtifactRef`, `EvaluationResponseTarget`, `EvaluationTargetManifest`, `EvaluationInputManifest`, `EvaluationInputBundle`, `SavedRGBObserverIdentity`, `SavedRGBObservation`, `ObserverAgreementAudit`, `ResponseStratumKey`, `ResponseMetricRecord`, `EdgeMetricAggregate`, `BootstrapInterval`, `AxisMetricReport`, `ResponseReport`, `AcceptanceDecision`, and `EvaluationArtifactManifest`.
- Evaluation-input APIs: `load_evaluation_authorization(path: Path, *, run_dir: Path, matrix_seal: MatrixTestSeal, response_test_access: ResponseAccessAudit) -> EvaluationAuthorization`; `build_evaluation_response_targets(response_manifest: ResponseManifest, condition_index: ConditionIndex, coordinate_index: CoordinateFrameIndex, uncertainty_floors: UncertaintyFloors, policy: ResponsePolicy, matrix_seal: MatrixTestSeal, *, run_dir: Path, authorization_path: Path, common_jobs_path: Path, teacher_test_binding_path: Path, teacher_cache_root: Path, response_test_access_path: Path, output_path: Path) -> EvaluationTargetManifest`; `seal_evaluation_input(*, run_dir: Path, response_policy_path: Path, response_manifest_path: Path, condition_index_path: Path, coordinate_binding_path: Path, coordinate_cache_root: Path, teacher_test_binding_path: Path, uncertainty_floors_path: Path, evaluation_policy_path: Path, matrix_seal_path: Path, common_jobs_path: Path, target_manifest_path: Path, arm_binding_path: Path, generation_results_path: Path, sam2_source_root: Path, sam2_checkpoint_path: Path, output_path: Path) -> EvaluationInputManifest`; `open_evaluation_input(path: Path, *, run_dir: Path, coordinate_cache_root: Path, expected_arm_id: str | None = None) -> EvaluationInputBundle`; and `verify_scientific_identity_before_media_or_metrics(candidate: EvaluationInputManifest, reference: EvaluationInputManifest) -> None`.
- Observation and scoring APIs: `SavedRGBObserverIdentity.from_runtime(model: SAM2Identity, track: SAM2TrackConfig, *, decode_backend: str, hard_mask_threshold: float, event_parameters: Mapping[str, float], validity_thresholds: Mapping[str, float]) -> SavedRGBObserverIdentity`; `observe_saved_rgb(video_path: Path, *, initial_mask_path: Path, scene_id: str, coordinate: CoordinateFrameRecord, policy: EvaluationPolicy, observer: Any) -> SavedRGBObservation`; `audit_saved_rgb_observer_agreement(hard: Sequence[SavedRGBObservation], differentiable: Sequence[PhysicalStateBatch], *, evaluation_input: EvaluationInputManifest, policy: EvaluationPolicy) -> ObserverAgreementAudit`; `score_response_pair(job: CommonPairJob, result: PairGenerationResult, observation_a: SavedRGBObservation, observation_b: SavedRGBObservation, target: EvaluationResponseTarget, policy: EvaluationPolicy, evaluation_input: EvaluationInputManifest) -> tuple[ResponseMetricRecord, ...]`; `average_five_seed_edge_metrics(records: Sequence[ResponseMetricRecord], *, expected_seed_set: tuple[int, int, int, int, int]) -> tuple[EdgeMetricAggregate, ...]`; `paired_group_bootstrap(edges: Sequence[EdgeMetricAggregate], *, metric_name: str, resamples: int = 10000, seed: int = 20260728) -> BootstrapInterval`; `paired_arm_group_bootstrap(complete: Sequence[ResponseMetricRecord], arm3: Sequence[ResponseMetricRecord], *, metric_name: str, resamples: int = 10000, seed: int = 20260728) -> BootstrapInterval`; `aggregate_response_metrics(records: Sequence[ResponseMetricRecord], *, expected_seed_set: tuple[int, int, int, int, int], resamples: int = 10000, seed: int = 20260728) -> ResponseReport`; `secondary_noninferiority_margin(*, validation_mean: float, validation_bootstrap_se: float, valid_range: tuple[float, float] | None) -> float`; and `compare_acceptance(complete: ResponseReport, arm3: ResponseReport, *, complete_records: Sequence[ResponseMetricRecord], arm3_records: Sequence[ResponseMetricRecord], policy: EvaluationPolicy) -> AcceptanceDecision`.
- Policy and CLI contracts: `seal_evaluation_policy(validation_selection_path: Path, observer_validation_audit_path: Path, *, experiment_id: str, response_test_access_count: int, output_path: Path) -> EvaluationPolicy`; validation audit CLI `audit_physical_response_observer_validation.py --run-dir run/physical_response_group_holdout_v1_seed42 --teacher-validation-binding run/physical_response_group_holdout_v1_seed42/physical_response/teacher_validation_bindings.json --teacher-cache-root cache/physical_response/teacher_v1 --coordinate-binding run/physical_response_group_holdout_v1_seed42/physical_response/coordinates/binding.json --coordinate-cache-root cache/physical_response/coordinates_v1 --sam2-root /root/Nico/third_party/sam2 --sam2-checkpoint /mnt/nvme1/NicoCache/checkpoints/sam2/sam2.1_hiera_tiny.pt --output run/physical_response_group_holdout_v1_seed42/validation/observer_validation_audit.json`; target CLI `build_physical_response_evaluation_targets.py --run-dir run/physical_response_group_holdout_v1_seed42 --authorization run/physical_response_group_holdout_v1_seed42/evaluation/authorization.json --matrix-seal run/physical_response_group_holdout_v1_seed42/frozen/matrix_test_seal.json --teacher-cache-root cache/physical_response/teacher_v1 --coordinate-cache-root cache/physical_response/coordinates_v1 --output run/physical_response_group_holdout_v1_seed42/evaluation/common/targets.json`; input CLI `seal_physical_response_evaluation_input.py --run-dir run/physical_response_group_holdout_v1_seed42 --arm-id complete_direct_rgb --target-manifest run/physical_response_group_holdout_v1_seed42/evaluation/common/targets.json --coordinate-cache-root cache/physical_response/coordinates_v1 --sam2-root /root/Nico/third_party/sam2 --sam2-checkpoint /mnt/nvme1/NicoCache/checkpoints/sam2/sam2.1_hiera_tiny.pt --output run/physical_response_group_holdout_v1_seed42/evaluation/inputs/complete_direct_rgb.json`; evaluator CLI `evaluate_physical_response.py --run-dir run/physical_response_group_holdout_v1_seed42 --evaluation-input run/physical_response_group_holdout_v1_seed42/evaluation/inputs/complete_direct_rgb.json --arm3-evaluation-input run/physical_response_group_holdout_v1_seed42/evaluation/inputs/flowmatch_quantity.json --arm3-evaluation-manifest run/physical_response_group_holdout_v1_seed42/evaluation/flowmatch_quantity/manifest.json --coordinate-cache-root cache/physical_response/coordinates_v1 --sam2-root /root/Nico/third_party/sam2 --sam2-checkpoint /mnt/nvme1/NicoCache/checkpoints/sam2/sam2.1_hiera_tiny.pt --output-dir run/physical_response_group_holdout_v1_seed42/evaluation/complete_direct_rgb`.
- Independent-observer rule: primary metrics read encoded RGB bytes from disk, run inference-only SAM2 with hard masks and separately coded hard event detection, and never consume training-time generated masks, moments, states, latents, or gradients. A second pass through the frozen differentiable observer may consume only those same decoded saved-RGB frames and the same frozen initial mask to form `ObserverAgreementAudit`; it is a concordance gate and never supplies a scientific metric. Reusing frozen real-video targets and coordinate records is required; fitting a coordinate transform from generated pixels is forbidden.

Data contracts are:

- `EvaluationPolicy(schema_version: Literal["physical_response_evaluation_policy_v1"], experiment_id: str, prediction_zero_tolerance: float, observer_discrepancy_p95_by_component: tuple[tuple[str, float], ...], observer_validation_audit_digest: str, noninferiority_margins: tuple[tuple[str, float], ...], hard_event_parameters: tuple[tuple[str, float], ...], validity_thresholds: tuple[tuple[str, float], ...], bootstrap_resamples: Literal[10000], bootstrap_seed: Literal[20260728], saved_rgb_observer_identity_digest: str, differentiable_observer_identity_digest: str, validation_selection_digest: str, sealed_before_response_test_access_count: Literal[0], digest: str)`.
- `ObserverValidationAudit(schema_version: Literal["physical_response_observer_validation_v1"], experiment_id: str, source_partition: Literal["response_validation"], validation_selection_digest: str, teacher_validation_binding_digest: str, coordinate_manifest_digest: str, saved_rgb_observer_identity_digest: str, differentiable_observer_identity_digest: str, sample_keys_digest: str, expected_observation_count: int, observed_observation_count: int, component_observation_counts: tuple[tuple[str, int], ...], component_p95_absolute_discrepancy: tuple[tuple[str, float], ...], event_index_disagreement_p95: float, validity_agreement_rate: float, passed: bool, failure_reasons: tuple[str, ...], digest: str)`. It is built only from response validation before policy sealing; Task 17 test-time agreement may consume its frozen limits but cannot modify it.
- `MatrixTestSeal(schema_version: Literal["physical_response_matrix_test_seal_v1"], run_id: str, experiment_id: str, ablation_matrix_digest: str, frozen_inputs_digest: str, response_policy_digest: str, response_manifest_digest: str, condition_index_digest: str, coordinate_binding_digest: str, coordinate_manifest_digest: str, teacher_test_binding_digest: str, teacher_train_binding_digest: str, uncertainty_floors_digest: str, common_job_manifest_digest: str, scientific_job_set_digest: str, generation_runtime_identity_digest: str, common_seed_set: tuple[int, int, int, int, int], edge_seed_keys_digest: str, arm_spec_digests: tuple[tuple[str, str], ...], arm_run_artifact_digests: tuple[tuple[str, str], ...], arm_generation_binding_digests: tuple[tuple[str, str], ...], evaluation_policy_digest: str, saved_rgb_observer_identity_digest: str, differentiable_observer_identity_digest: str, response_test_access_audit_digest_at_seal: str, response_test_access_count_at_seal: Literal[0], digest: str)`. Task 17 owns the strict contract/loader; Task 18 is the only producer.
- `EvaluationAuthorization(schema_version: Literal["physical_response_evaluation_authorization_v1"], run_id: str, experiment_id: str, run_relative_path: str, transition_relative_path: str, previous_run_digest: str, evaluation_transition_digest: str, matrix_test_seal_digest: str, response_test_access_audit_digest_at_authorization: str, response_test_access_count_at_authorization: Literal[0], purpose: Literal["build_response_test_targets"], digest: str)`. Task 17 owns its strict loader; Task 18 is the only producer. The referenced run and transition documents must resolve beneath `run_dir`; the run must be `status="evaluating"` and bind the authorization, seal, and transition, while the transition must be an exact `evaluation_ready -> evaluating` link from `previous_run_digest`.
- `EvaluationArtifactRole = Literal["response_policy", "response_manifest", "condition_index", "coordinate_binding", "teacher_test_binding", "uncertainty_floors", "evaluation_policy", "matrix_test_seal", "common_jobs", "evaluation_targets", "arm_generation_binding", "generation_results"]`; `EvaluationArtifactRef(role: EvaluationArtifactRole, relative_path: str, digest: str)`; roles are unique and exact, and paths are normalized run-relative POSIX paths that may not escape `run_dir`.
- `TeacherMetricKind = Literal["empirical_primary", "analytic_weak", "analytic_strong", "absolute_empirical", "validity", "visual", "not_estimable"]`; `ResponseStratumKey(scene_id: str, axis_id: str, holdout_type: Literal["group_holdout", "value_holdout"], edge_kind: EdgeKind, interpolation_kind: InterpolationKind, teacher_kind: TeacherMetricKind, digest: str)`.
- `EvaluationResponseTarget(target_id: str, edge_id: str, edge_record_digest: str, group_id: str, stratum: ResponseStratumKey, q_a_si: str, q_b_si: str, normalized_delta_q: str, target_kind: ResponseTargetKind, teacher_roles: tuple[TeacherRole, ...], coordinate_frame_ids: tuple[str, str], coordinate_record_digests: tuple[str, str], response_supervision_relative_path: str | None, response_supervision_sha256: str | None, eligible_components: tuple[str, ...], ineligible_reasons: tuple[tuple[str, str], ...], uncertainty_floors_digest: str, digest: str)`. Estimability is represented only by `stratum.interpolation_kind`, eligible components, ineligible reasons, and metric status; it never changes the edge's target kind.
- `EvaluationTargetManifest(schema_version: Literal["physical_response_evaluation_targets_v1"], experiment_id: str, matrix_test_seal_digest: str, evaluation_authorization_digest: str, response_policy_digest: str, response_manifest_digest: str, condition_index_digest: str, coordinate_binding_digest: str, coordinate_manifest_digest: str, teacher_test_binding_digest: str, teacher_train_binding_digest: str, uncertainty_floors_digest: str, common_job_manifest_digest: str, response_test_access_audit_digest: str, response_test_access_count_after_build: Literal[1], targets_relative_path: str, targets_sha256: str, edge_ids: tuple[str, ...], target_digests: tuple[str, ...], target_set_digest: str, digest: str)`.
- `EvaluationInputManifest(schema_version: Literal["physical_response_evaluation_input_v1"], run_id: str, experiment_id: str, arm_id: str, artifact_refs: tuple[EvaluationArtifactRef, ...], response_policy_digest: str, response_manifest_digest: str, condition_index_digest: str, coordinate_binding_digest: str, coordinate_manifest_digest: str, teacher_test_binding_digest: str, teacher_train_binding_digest: str, uncertainty_floors_digest: str, response_test_access_audit_digest: str, evaluation_policy_digest: str, matrix_test_seal_digest: str, saved_rgb_observer_identity_digest: str, differentiable_observer_identity_digest: str, common_job_manifest_digest: str, scientific_job_set_digest: str, evaluation_target_manifest_digest: str, evaluation_target_set_digest: str, arm_generation_binding_digest: str, arm_run_artifact_digest: str, generation_results_manifest_digest: str, scientific_identity_digest: str, digest: str)`.
- `EvaluationInputBundle(manifest: EvaluationInputManifest, run_root: Path, verified_paths: Mapping[EvaluationArtifactRole, Path], common_job_manifest: CommonJobManifest, jobs_by_id: Mapping[str, CommonPairJob], generation_results_by_job_id: Mapping[str, PairGenerationResult], targets_by_edge_id: Mapping[str, EvaluationResponseTarget], coordinate_index: CoordinateFrameIndex)`. Its constructor is private to `open_evaluation_input`; every map is total and exact against the sealed manifests, and evaluator code may obtain a path, job, result, target, or coordinate only from this bundle.
- `SavedRGBObserverIdentity(sam2_identity_digest: str, track_config_digest: str, decode_backend: str, hard_mask_threshold: float, event_parameters: tuple[tuple[str, float], ...], validity_thresholds: tuple[tuple[str, float], ...], observer_version: str, digest: str)` and `SavedRGBObservation(video_sha256: str, initial_mask_sha256: str, coordinate_frame_id: str, coordinate_record_digest: str, state: PhysicalStateBatch, hard_event_indices: Mapping[str, int | None], hard_validity: Mapping[str, bool], observer_identity_digest: str, digest: str)`.
- `ObserverAgreementAudit(schema_version: Literal["physical_response_observer_agreement_v1"], run_id: str, experiment_id: str, arm_id: str, source_partition: Literal["response_test"], evaluation_input_digest: str, evaluation_policy_digest: str, observer_validation_audit_digest: str, common_job_manifest_digest: str, generation_results_manifest_digest: str, coordinate_manifest_digest: str, saved_rgb_observer_identity_digest: str, differentiable_observer_identity_digest: str, sample_keys_digest: str, expected_observation_count: int, observed_observation_count: int, component_observation_counts: tuple[tuple[str, int], ...], component_discrepancy_p95: tuple[tuple[str, float], ...], frozen_component_limits: tuple[tuple[str, float], ...], exceeded_components: tuple[str, ...], event_index_disagreement_p95: float, validity_agreement_rate: float, passed: bool, failure_reasons: tuple[str, ...], digest: str)`.
- `ResponseMetricRecord(arm_id: str, evaluation_input_digest: str, scientific_identity_digest: str, matrix_test_seal_digest: str, evaluation_policy_digest: str, common_job_digest: str, generation_result_digest: str, target_digest: str, endpoint_digests: tuple[str, str], initial_noise_sha256: str, coordinate_record_digests: tuple[str, str], stratum: ResponseStratumKey, group_id: str, edge_id: str, edge_kind: EdgeKind, seed: int, component: str, metric_name: str, value: float | None, eligible: bool, status: Literal["ok", "invalid", "not_estimable"], digest: str)`.
- `EdgeMetricAggregate(arm_id: str, scientific_identity_digest: str, stratum: ResponseStratumKey, group_id: str, edge_id: str, edge_kind: EdgeKind, component: str, metric_name: str, value: float | None, status: Literal["ok", "invalid", "not_estimable"], seed_count: Literal[5], digest: str)`; `BootstrapInterval(estimate: float, lower: float, upper: float, sampling_units: int, resamples: int, seed: int)`; `AxisMetricReport(stratum: ResponseStratumKey, primary_metrics: Mapping[str, BootstrapInterval], secondary_metrics: Mapping[str, BootstrapInterval], eligibility_counts: Mapping[str, int], status: Literal["ok", "not_estimable"])`; `ResponseReport(arm_id: str, scientific_identity_digest: str, strata: tuple[AxisMetricReport, ...], adjacent_macro: Mapping[str, BootstrapInterval], scene_metrics: Mapping[str, Mapping[str, BootstrapInterval]], coverage_counts: Mapping[str, int], digest: str)`; and `AcceptanceDecision(passed: bool, primary_intervals: Mapping[str, BootstrapInterval], scene_decisions: Mapping[str, bool], secondary_decisions: Mapping[str, bool], collision_decisions: Mapping[str, bool], failure_reasons: tuple[str, ...], digest: str)`.
- `EvaluationArtifactManifest(schema_version: Literal["physical_response_evaluation_artifact_v1"], run_id: str, experiment_id: str, arm_id: str, status: Literal["evaluated", "failed"], evaluation_input_digest: str, scientific_identity_digest: str, matrix_test_seal_digest: str, common_job_manifest_digest: str, generation_results_manifest_digest: str, evaluation_target_manifest_digest: str, evaluation_policy_digest: str, response_test_access_audit_digest: str, observer_agreement_audit_digest: str, pair_results_relative_path: str | None, pair_results_sha256: str | None, group_results_relative_path: str | None, group_results_sha256: str | None, summary_relative_path: str | None, summary_sha256: str | None, expected_job_count: int, evaluated_job_count: int, evaluated_edge_seed_keys_digest: str, reference_arm_id: str | None, reference_evaluation_input_digest: str | None, reference_evaluation_artifact_manifest_digest: str | None, reference_pair_results_sha256: str | None, acceptance_decision_digest: str | None, failure_reasons: tuple[str, ...], digest: str)`.

- [ ] **Step 1: Write fail-before-read input, target reconstruction, observer independence, strata, bootstrap, and acceptance tests**

```python
class EvaluationInputTests(unittest.TestCase):
    def test_complete_and_arm3_identity_mismatch_fails_before_any_read(self) -> None:
        video_reader = mock.Mock()
        metric_reader = mock.Mock()
        with self.assertRaisesRegex(ValueError, "scientific identity mismatch"):
            evaluate_fixture_complete_against_arm3(
                changed="coordinate_manifest",
                video_reader=video_reader,
                reference_metric_reader=metric_reader,
            )
        video_reader.assert_not_called()
        metric_reader.assert_not_called()

    def test_coordinate_binding_is_verified_pointer_not_scientific_identity(self) -> None:
        first, relocated = build_equivalent_coordinate_binding_inputs()
        self.assertNotEqual(first.coordinate_binding_digest, relocated.coordinate_binding_digest)
        self.assertEqual(first.coordinate_manifest_digest, relocated.coordinate_manifest_digest)
        self.assertEqual(first.evaluation_target_set_digest, relocated.evaluation_target_set_digest)
        self.assertEqual(first.scientific_identity_digest, relocated.scientific_identity_digest)
        self.assertNotEqual(first.digest, relocated.digest)

    def test_every_bound_artifact_mutation_fails_before_video_decode(self) -> None:
        for changed in (
            "policy", "manifest", "condition_index", "coordinate_binding",
            "coordinate_record", "teacher_test", "teacher_train", "floors",
            "evaluation_policy", "matrix_seal", "common_job", "target",
            "generation_result", "noise", "first_frame", "mask", "video",
        ):
            decoder = mock.Mock()
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                open_tampered_evaluation_input(changed, decoder=decoder)
            decoder.assert_not_called()

    def test_typed_evaluation_authorization_precedes_teacher_view_open(self) -> None:
        teacher_opener = mock.Mock()
        with mock.patch(
            "physbench.physical_response.evaluation_inputs.open_teacher_cache_view",
            teacher_opener,
        ), self.assertRaisesRegex(PermissionError, "run is not evaluating"):
            build_fixture_evaluation_targets(
                authorization_path=fixture_forged_authorization_path(
                    run_status="evaluation_ready"
                )
            )
        teacher_opener.assert_not_called()

    def test_empty_callback_cannot_substitute_for_evaluation_authorization(self) -> None:
        self.assertNotIn(
            "evaluation_phase_guard",
            inspect.signature(build_evaluation_response_targets).parameters,
        )

    def test_failed_target_build_leaves_test_access_audit_byte_identical(self) -> None:
        access_path = fixture_response_test_access_path(count=0)
        before = access_path.read_bytes()
        with self.assertRaisesRegex(ValueError, "target coverage"):
            build_fixture_evaluation_targets(
                authorization_path=fixture_valid_evaluation_authorization_path(),
                changed="missing_common_edge",
            )
        self.assertEqual(before, access_path.read_bytes())

    def test_successful_target_build_commits_exactly_one_authorized_access(self) -> None:
        target_manifest = build_fixture_evaluation_targets(
            authorization_path=fixture_valid_evaluation_authorization_path()
        )
        access = load_response_access_audit(
            fixture_response_test_access_path(), expected_partition="response_test"
        )
        self.assertEqual(1, access.access_count)
        self.assertEqual(target_manifest.target_set_digest, access.events[0].target_set_digest)
        self.assertEqual(
            target_manifest.evaluation_authorization_digest,
            access.events[0].authorization_digest,
        )

    def test_target_is_rebuilt_from_test_edge_roles_and_train_only_floors(self) -> None:
        target = build_fixture_evaluation_target(partition="response_test")
        self.assertEqual(fixture_test_edge().digest, target.edge_record_digest)
        self.assertEqual(fixture_normalized_delta_q(), target.normalized_delta_q)
        group = fixture_response_manifest().group_for(fixture_test_edge().group_id)
        self.assertEqual(group.teacher_roles, target.teacher_roles)
        self.assertEqual(fixture_train_floors().digest, target.uncertainty_floors_digest)
        self.assertEqual("response_train", fixture_train_floors().source_partition)

    def test_missing_empirical_target_is_not_replaced_by_analytic_target(self) -> None:
        target = build_fixture_evaluation_target(
            target_kind="empirical_pair", empirical_available=False,
        )
        self.assertEqual((), target.eligible_components)
        self.assertIn("empirical_target_unavailable", dict(target.ineligible_reasons).values())

    def test_all_arms_share_the_same_edge_targets(self) -> None:
        complete, arm3 = build_fixture_evaluation_inputs(
            "complete_direct_rgb", "flowmatch_quantity"
        )
        self.assertEqual(
            complete.evaluation_target_manifest_digest,
            arm3.evaluation_target_manifest_digest,
        )
        self.assertEqual(
            complete.evaluation_target_set_digest,
            arm3.evaluation_target_set_digest,
        )
        self.assertEqual(target_digests(complete), target_digests(arm3))

    def test_open_input_returns_only_verified_total_indices(self) -> None:
        bundle = open_fixture_evaluation_input_bundle()
        self.assertEqual(set(required_evaluation_artifact_roles()), set(bundle.verified_paths))
        self.assertEqual(bundle.common_job_manifest.job_count, len(bundle.jobs_by_id))
        self.assertEqual(set(bundle.jobs_by_id), set(bundle.generation_results_by_job_id))
        self.assertEqual(
            {job.edge_id for job in bundle.jobs_by_id.values()},
            set(bundle.targets_by_edge_id),
        )
        self.assertEqual(fixture_run_dir().resolve(), bundle.run_root)

    def test_target_set_covers_every_common_edge_kind(self) -> None:
        common, targets = build_fixture_common_targets(
            edge_kinds=("adjacent", "long_range_audit", "analytic_counterfactual")
        )
        self.assertEqual(common_edge_ids(common), targets.edge_ids)
        self.assertEqual(
            {"adjacent", "long_range_audit", "analytic_counterfactual"},
            {target.stratum.edge_kind for target in load_targets(targets)},
        )

class SavedRGBObserverTests(unittest.TestCase):
    def test_observer_requires_saved_video_and_has_no_training_state_argument(self) -> None:
        self.assertEqual(
            ["video_path", "initial_mask_path", "scene_id", "coordinate", "policy", "observer"],
            list(inspect.signature(observe_saved_rgb).parameters),
        )
        with self.assertRaises(FileNotFoundError):
            observe_saved_rgb(Path("missing.mp4"), **fixture_saved_rgb_kwargs())

    def test_differentiable_observer_is_only_an_agreement_gate(self) -> None:
        hard = fixture_saved_rgb_observations()
        differentiable = fixture_differentiable_observations(changed=True)
        audit = audit_saved_rgb_observer_agreement(
            hard, differentiable,
            evaluation_input=fixture_evaluation_input(),
            policy=fixture_evaluation_policy(),
        )
        self.assertFalse(audit.passed)
        self.assertEqual(hard, scientific_metric_observations())

    def test_runtime_observer_identity_mismatch_precedes_decode(self) -> None:
        decoder = mock.Mock()
        with self.assertRaisesRegex(ValueError, "saved RGB observer identity mismatch"):
            open_fixture_evaluation_input(runtime_observer="changed", decoder=decoder)
        decoder.assert_not_called()

    def test_observer_agreement_fails_closed_for_every_invalid_evidence(self) -> None:
        for changed in (
            "missing_component", "low_coverage", "nonfinite",
            "above_validation_p95", "event_disagreement", "validity_disagreement",
        ):
            audit = build_fixture_observer_agreement(changed=changed)
            with self.subTest(changed=changed):
                self.assertFalse(audit.passed)
                self.assertTrue(audit.failure_reasons)

class ResponseMetricTests(unittest.TestCase):
    def test_sign_magnitude_zero_validity_remain_separate(self) -> None:
        report = aggregate_response_metrics(
            fixture_metric_records(), expected_seed_set=fixture_five_seeds(),
            resamples=2000, seed=20260728,
        )
        axis = find_stratum(
            report,
            scene_id="parabolic_motion",
            axis_id="initial_horizontal_velocity",
            holdout_type="value_holdout",
            edge_kind="adjacent",
            interpolation_kind="interpolation",
            teacher_kind="empirical_primary",
        )
        self.assertEqual(
            {"sign_accuracy", "magnitude_nmae"},
            set(axis.primary_metrics),
        )
        self.assertTrue({
            "zero_response_leakage", "absolute_state_error",
            "validity_rate", "observer_coverage",
            "empirical_residual", "analytic_residual",
            "visual_quality", "entity_integrity",
        }.issubset(axis.secondary_metrics))

    def test_group_bootstrap_resamples_groups_not_edges(self) -> None:
        edges = average_five_seed_edge_metrics(
            correlated_five_seed_records(), expected_seed_set=fixture_five_seeds(),
        )
        interval = paired_group_bootstrap(
            edges, metric_name="sign_accuracy",
            resamples=2000, seed=20260728,
        )
        self.assertEqual(number_of_unique_groups(edges), interval.sampling_units)

    def test_seed_average_precedes_group_bootstrap(self) -> None:
        edges = average_five_seed_edge_metrics(
            two_edges_with_five_seeds_each(), expected_seed_set=fixture_five_seeds(),
        )
        self.assertEqual(2, len(edges))
        self.assertEqual({5}, {edge.seed_count for edge in edges})

    def test_records_and_aggregates_retain_the_complete_stratum_key(self) -> None:
        records = fixture_metric_records()
        aggregates = average_five_seed_edge_metrics(
            records, expected_seed_set=fixture_five_seeds(),
        )
        self.assertEqual(
            {(r.stratum.scene_id, r.stratum.axis_id, r.stratum.holdout_type,
              r.stratum.edge_kind, r.stratum.interpolation_kind,
              r.stratum.teacher_kind) for r in records},
            {(r.stratum.scene_id, r.stratum.axis_id, r.stratum.holdout_type,
              r.stratum.edge_kind, r.stratum.interpolation_kind,
              r.stratum.teacher_kind) for r in aggregates},
        )

    def test_paired_arm_bootstrap_requires_matched_edge_and_seed_keys(self) -> None:
        with self.assertRaisesRegex(ValueError, "matched edge/seed keys"):
            paired_arm_group_bootstrap(
                complete=five_seed_arm(drop_key=None),
                arm3=five_seed_arm(drop_key=("edge_1", 202)),
                metric_name="magnitude_nmae",
            )

    def test_not_estimable_stratum_never_emits_numeric_metric(self) -> None:
        edge = average_five_seed_edge_metrics(
            five_seed_records(interpolation_kind="not_estimable", value=None),
            expected_seed_set=fixture_five_seeds(),
        )[0]
        self.assertEqual("not_estimable", edge.status)
        self.assertIsNone(edge.value)

class EvaluationPolicyTests(unittest.TestCase):
    def test_policy_must_be_sealed_before_any_response_test_access(self) -> None:
        with self.assertRaisesRegex(PermissionError, "response test already accessed"):
            seal_evaluation_policy(
                fixture_validation_selection(), fixture_observer_validation_audit(),
                experiment_id="group_holdout_v1", response_test_access_count=1,
                output_path=fixture_output_path(),
            )

    def test_policy_requires_passing_observer_audit_and_all_margins(self) -> None:
        for missing in (
            "prediction_zero_tolerance",
            "observer_discrepancy_p95_by_component",
            "noninferiority_margins",
        ):
            with self.subTest(missing=missing), self.assertRaisesRegex(ValueError, missing):
                seal_fixture_evaluation_policy(missing=missing)

class AcceptanceTests(unittest.TestCase):
    def test_gain_on_only_three_scenes_cannot_pass(self) -> None:
        decision = compare_acceptance(
            complete_report(favorable_scenes=3), arm3_report(),
            complete_records=complete_raw_records(),
            arm3_records=arm3_raw_records(),
            policy=fixture_evaluation_policy(),
        )
        self.assertFalse(decision.passed)
        self.assertIn("fewer_than_four_favorable_scenes", decision.failure_reasons)

    def test_collision_must_pass_response_momentum_and_energy_together(self) -> None:
        decision = compare_acceptance(
            complete_report(collision_energy_passed=False), arm3_report(),
            complete_records=complete_raw_records(),
            arm3_records=arm3_raw_records(),
            policy=fixture_evaluation_policy(),
        )
        self.assertFalse(decision.passed)
        self.assertIn("two_ball_joint_gate_failed", decision.failure_reasons)

    def test_arm3_summary_cannot_replace_bound_raw_pair_results(self) -> None:
        with self.assertRaisesRegex(ValueError, "reference pair results"):
            compare_fixture_acceptance(
                arm3_summary_unchanged=True,
                arm3_pair_results_changed=True,
            )
```

- [ ] **Step 2: Run evaluator tests and observe missing modules**

Run:

```bash
PYTHONPATH=src:tests:.:/root/Nico/third_party/sam2 \
  /mnt/nvme1/NicoCache/envs/physics_wan/bin/python -m unittest \
  tests.physical_response.test_evaluation_inputs \
  tests.physical_response.test_saved_rgb_observer \
  tests.physical_response.test_metrics \
  tests.physical_response.test_acceptance -v
```

Expected: FAIL importing `evaluation_inputs`, `saved_rgb_observer`, and `metrics`.

- [ ] **Step 3: Implement the sealed evaluation-input boundary, hard saved-RGB scoring, observer agreement, and group-level reporting**

```python
SCIENTIFIC_IDENTITY_FIELDS = (
    "response_policy_digest", "response_manifest_digest",
    "condition_index_digest", "coordinate_manifest_digest",
    "teacher_test_binding_digest",
    "teacher_train_binding_digest", "uncertainty_floors_digest",
    "response_test_access_audit_digest",
    "evaluation_policy_digest", "matrix_test_seal_digest",
    "saved_rgb_observer_identity_digest",
    "differentiable_observer_identity_digest",
    "common_job_manifest_digest", "scientific_job_set_digest",
    "evaluation_target_set_digest",
)

def scientific_identity_digest(document: Mapping[str, Any]) -> str:
    return canonical_sha256({name: document[name] for name in SCIENTIFIC_IDENTITY_FIELDS})

def verify_scientific_identity_before_media_or_metrics(
    candidate: EvaluationInputManifest,
    reference: EvaluationInputManifest,
) -> None:
    if candidate.scientific_identity_digest != reference.scientific_identity_digest:
        raise ValueError("scientific identity mismatch")
    if candidate.common_job_manifest_digest != reference.common_job_manifest_digest:
        raise ValueError("common job manifest mismatch")
    if candidate.scientific_job_set_digest != reference.scientific_job_set_digest:
        raise ValueError("scientific job set mismatch")

def load_evaluation_authorization(
    path: Path,
    *,
    run_dir: Path,
    matrix_seal: MatrixTestSeal,
    response_test_access: ResponseAccessAudit,
) -> EvaluationAuthorization:
    authorization = EvaluationAuthorization.from_document(load_json(path))
    if (
        authorization.matrix_test_seal_digest != matrix_seal.digest
        or authorization.run_id != matrix_seal.run_id
        or authorization.experiment_id != matrix_seal.experiment_id
        or authorization.response_test_access_count_at_authorization != 0
        or authorization.response_test_access_audit_digest_at_authorization
        != response_test_access.digest
        or response_test_access.access_count != 0
    ):
        raise PermissionError("evaluation authorization does not match sealed zero-access run")
    run_path = resolve_normalized_path_beneath(
        run_dir, authorization.run_relative_path
    )
    transition_path = resolve_normalized_path_beneath(
        run_dir, authorization.transition_relative_path
    )
    run_document = load_strict_physical_response_run_header(run_path)
    transition = load_strict_status_transition(transition_path)
    if (
        run_document.status != "evaluating"
        or run_document.matrix_test_seal_digest != matrix_seal.digest
        or run_document.evaluation_authorization_digest != authorization.digest
        or run_document.latest_status_transition_digest
        != authorization.evaluation_transition_digest
        or transition.digest != authorization.evaluation_transition_digest
        or transition.previous_run_digest != authorization.previous_run_digest
        or transition.previous_status != "evaluation_ready"
        or transition.next_status != "evaluating"
    ):
        raise PermissionError("run is not evaluating under this authorization")
    return authorization

def build_evaluation_response_targets(
    response_manifest: ResponseManifest,
    condition_index: ConditionIndex,
    coordinate_index: CoordinateFrameIndex,
    uncertainty_floors: UncertaintyFloors,
    policy: ResponsePolicy,
    matrix_seal: MatrixTestSeal,
    *,
    run_dir: Path,
    authorization_path: Path,
    common_jobs_path: Path,
    teacher_test_binding_path: Path,
    teacher_cache_root: Path,
    response_test_access_path: Path,
    output_path: Path,
) -> EvaluationTargetManifest:
    common, jobs = load_common_pair_jobs(common_jobs_path)
    teacher_binding = load_teacher_cache_binding_header(
        teacher_test_binding_path,
        expected_partition="response_test",
        expected_response_manifest_digest=matrix_seal.response_manifest_digest,
        expected_coordinate_manifest_digest=matrix_seal.coordinate_manifest_digest,
        expected_observer_identity_digest=matrix_seal.differentiable_observer_identity_digest,
    )
    current_access = load_response_access_audit(
        response_test_access_path, expected_partition="response_test"
    )
    authorization = load_evaluation_authorization(
        authorization_path, run_dir=run_dir, matrix_seal=matrix_seal,
        response_test_access=current_access,
    )
    verify_target_builder_identities(
        response_manifest, condition_index, coordinate_index,
        uncertainty_floors, policy, matrix_seal, common, teacher_binding,
    )
    teacher_test_view = open_teacher_cache_view(
        teacher_test_binding_path, cache_root=teacher_cache_root,
        allowed_partitions=frozenset({"response_test"}),
        expected_response_manifest_digest=matrix_seal.response_manifest_digest,
        expected_coordinate_manifest_digest=matrix_seal.coordinate_manifest_digest,
        expected_observer_identity_digest=matrix_seal.differentiable_observer_identity_digest,
    )
    expected_edge_ids = tuple(sorted({job.edge_id for job in jobs}))
    edges_by_id = {edge.edge_id: edge for edge in response_manifest.edges}
    if set(edges_by_id).intersection(expected_edge_ids) != set(expected_edge_ids):
        raise ValueError("common job edge is absent from response manifest")
    targets: list[EvaluationResponseTarget] = []
    for edge_id in expected_edge_ids:
        edge = edges_by_id[edge_id]
        if edge.partition != "response_test":
            raise ValueError("common target edge is not response_test")
        group = response_manifest.group_for(edge.group_id)
        normalization = response_manifest.normalization_for(group.axis_id)
        supervision = build_response_supervision(
            edge, teacher_test_view, normalization, uncertainty_floors, policy
        )
        if supervision.target.teacher_roles != group.teacher_roles:
            raise ValueError("group and supervision teacher roles differ")
        targets.append(materialize_evaluation_target(
            edge=edge,
            group=group,
            supervision=supervision,
            condition_index=condition_index,
            coordinate_index=coordinate_index,
            output_path=output_path,
        ))
    target_set_digest = compute_evaluation_target_set_digest(
        targets, coordinate_manifest_digest=coordinate_index.digest
    )
    updated_access = plan_response_access_update(
        current_access, purpose="test_target_build",
        opened_case_ids=teacher_test_view.opened_case_ids,
        opened_edge_ids=expected_edge_ids,
        authorization_digest=authorization.digest,
        target_set_digest=target_set_digest,
    )
    staged = stage_evaluation_target_artifacts(
        targets=targets, target_set_digest=target_set_digest,
        authorization=authorization, matrix_seal=matrix_seal,
        common=common, committed_access=updated_access,
        output_path=output_path,
    )
    publish_immutable_evaluation_target_artifacts(staged)
    compare_and_swap_response_access_audit(
        response_test_access_path,
        expected_digest=current_access.digest,
        replacement=updated_access,
    )
    return staged.manifest

def secondary_noninferiority_margin(
    *,
    validation_mean: float,
    validation_bootstrap_se: float,
    valid_range: tuple[float, float] | None,
) -> float:
    range_term = (
        0.05 * (valid_range[1] - valid_range[0])
        if valid_range is not None
        else 0.05 * abs(validation_mean)
    )
    return max(range_term, validation_bootstrap_se)

def paired_group_bootstrap(
    edges: Sequence[EdgeMetricAggregate],
    *,
    metric_name: str,
    resamples: int = 10000,
    seed: int = 20260728,
) -> BootstrapInterval:
    by_group = edges_by_group(edges)
    keys = tuple(sorted(by_group))
    generator = np.random.default_rng(seed)
    samples = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        draw = generator.choice(keys, size=len(keys), replace=True)
        samples[index] = macro_metric(
            [edge for key in draw for edge in by_group[key]], metric_name
        )
    return BootstrapInterval(
        estimate=macro_metric(edges, metric_name),
        lower=float(np.quantile(samples, 0.025)),
        upper=float(np.quantile(samples, 0.975)), sampling_units=len(keys),
        resamples=resamples, seed=seed,
    )
```

`seal_evaluation_policy` runs before opening the response-test binding and fails unless access count is zero and the validation-only hard-versus-differentiable observer audit passed. Validation selection freezes prediction-zero tolerance, hard event/validity thresholds, per-component 95th-percentile observer discrepancy, every secondary margin, both realized observer identities, and bootstrap settings; no field has a test-time default. `build_evaluation_response_targets` is permitted only by a valid matrix seal and an access audit still identical to the seal's zero-access record. Its teacher-binding header reader authenticates the typed document but has no cache root and cannot resolve or read a teacher record. The strict authorization loader independently parses the Task-18 run and transition documents and requires the exact `evaluation_ready -> evaluating` transition, authorization identity, matrix seal, and zero-access state; there is no injectable callback or boolean. Only after all those checks pass does the builder construct the `response_test`-only lazy `TeacherCacheView`. It loads uncertainty floors only when their response-manifest and response-train teacher-binding identities match and calls the same Task-11 supervision builder used by training. Missing empirical targets remain ineligible; they are never replaced by an analytic target. The target set covers every unique edge in the sealed common jobs, including separately stratified `long_range_audit` edges, and rejects any missing or extra edge. `target_set_digest` binds sorted target content, source record identities, tensor dtype/shape/bytes, and coordinate-manifest identity but excludes run-local pointer paths. The builder computes the prospective access event, stages and publishes immutable target bytes, and then performs one compare-and-swap of the access audit as the final commit pointer. Any failure before that final update leaves the audit unchanged; a stale compare-and-swap leaves only an unreferenced immutable target candidate. A successful build records exactly one event binding the authorization, complete opened case/edge sets, and target-set identity. The arm-independent target manifest is built once and referenced byte-for-byte by all seven evaluation inputs; arm evaluators never reopen teacher records.

`seal_evaluation_input` first resolves every `EvaluationArtifactRef` under `run_dir`, verifies the seal and all common scientific identities, requires the current response-test audit to equal the target manifest's one committed access event (including authorization and target-set identities), verifies that the named arm artifact is in the seal, reconstructs both observer identities from the explicit SAM2 source/checkpoint paths, and proves exact common-job/result/target edge-seed coverage. An orphan target candidate whose access compare-and-swap never committed is therefore unusable. Task 17 owns no general local-runtime schema. It derives `scientific_identity_digest` only from `SCIENTIFIC_IDENTITY_FIELDS`. `coordinate_binding_digest` and `evaluation_target_manifest_digest` still bind the exact run-local input document, but scientific comparability uses `coordinate_manifest_digest` and `evaluation_target_set_digest`; equivalent relocated pointers therefore do not redefine the experiment. Arm/checkpoint/result/output fields are likewise excluded from the equality key but remain bound by the input's own digest. `open_evaluation_input` returns the fully indexed `EvaluationInputBundle`; after construction, evaluator code may read artifact paths, jobs, results, targets, or coordinates only through that bundle. For the complete-versus-arm-3 path, `verify_scientific_identity_before_media_or_metrics` runs immediately after strict input parsing and before either a video decoder or a reference `pair_results.jsonl` reader is constructed. After that check, each encoded video, initial noise, first frame, mask manifest, initial mask, condition payload, endpoint coordinate, job, target, and generation result is verified before observation.

The evaluator sequence is fixed:

1. Strictly parse input schemas, run-relative path boundaries, and document identities without allocating SAM2 or opening teacher/media/metric files.
2. Verify the matrix seal, seven arm/binding entries, the named arm, and the seal-time zero-access audit.
3. Verify policy/manifest, condition index, coordinate binding-to-manifest relation, train-only floors, evaluation policy, and validation observer calibration.
4. Verify common jobs, arm binding, generation results, runtime, job count, and exact edge-seed keys.
5. In complete comparison mode, verify complete/arm-3 scientific identity and the arm-3 artifact header before reading either arm's video or any pair-result row.
6. Strictly validate `EvaluationAuthorization` against the on-disk evaluating run, transition, seal, and zero-access audit; then build the common response-test targets once and use the access audit as the final atomic commit pointer. Task 17 does not import Task 18.
7. Verify each job/result endpoint, then decode saved RGB and run the hard observer; rerun the differentiable observer only from those same saved frames for agreement.
8. Fail the arm if agreement is incomplete, non-finite, under-covered, or above a frozen component limit; do not emit numeric scientific metrics on failure.
9. Write verified per-pair metrics, average the sealed five seeds within each edge, and bootstrap complete `group_id` clusters without collapsing edge kind or any other stratum dimension.
10. Atomically write pair/group/summary/agreement artifacts, then write `EvaluationArtifactManifest`; the complete manifest additionally binds the typed arm-3 manifest and raw arm-3 pair results.

`score_response_pair` counts `abs(R_hat) <= prediction_zero_tolerance * max(2*sigma_R, uncertainty_floor)` as near zero and therefore wrong for a confidently non-zero target. For significant targets, `magnitude_nmae = abs(R_hat-R*) / max(abs(R*), 2*sigma_R, uncertainty_floor)`; statistically non-significant targets contribute only `zero_response_leakage`. A `not_estimable` target always emits `value=None`. Every record retains its typed `(scene, axis, holdout, edge kind, interpolation, teacher kind)` stratum plus the seal, input, job, result, target, noise, endpoint, and coordinate identities. Primary metrics contain only sign accuracy and magnitude error. Zero leakage, absolute state, validity/coverage, empirical and analytic residuals, visual quality, and entity integrity remain separate secondary/non-inferiority records. The hard observer alone supplies scientific state. The differentiable observer is rerun from saved RGB only to produce `ObserverAgreementAudit`; any missing component/count, non-finite value, coverage deficit, frozen validation-p95 exceedance, event disagreement beyond the sealed threshold, or validity disagreement blocks the arm artifact.

`average_five_seed_edge_metrics` rejects an estimable edge unless it has exactly the five sealed seed IDs, rejects mixed strata within an aggregate, averages those five records within the edge, and only then supplies edge aggregates to whole-`group_id` cluster resampling. `long_range_audit` is reported separately and never enters `adjacent_macro` acceptance. `paired_arm_group_bootstrap` consumes raw complete and arm-3 `pair_results.jsonl`, joins on `(scientific_identity_digest, stratum.digest, group_id, edge_id, seed, component, metric_name, target_digest)`, verifies the two arm IDs differ but all scientific keys match, computes signed per-seed arm differences, averages five seeds per edge, and resamples complete groups. A summary alone is never accepted as arm-3 input. `compare_acceptance` requires adjacent-pair macro sign improvement and magnitude-error reduction over arm 3 with paired-bootstrap intervals wholly favorable, favorable point estimates for both metrics in at least four scenes, no significantly adverse scene interval, all frozen secondary non-inferiority gates, and simultaneous two-ball response/momentum/energy gates. Each arm evaluator writes exactly `manifest.json`, `pair_results.jsonl`, `group_results.jsonl`, and `summary.json`; the complete arm's manifest also binds the raw arm-3 pair-results digest and acceptance decision.

- [ ] **Step 4: Run evaluator tests and a real two-job saved-RGB integration worker**

Run:

```bash
PYTHONPATH=src:tests:.:/root/Nico/third_party/sam2 \
  /mnt/nvme1/NicoCache/envs/physics_wan/bin/python -m unittest \
  tests.physical_response.test_evaluation_inputs \
  tests.physical_response.test_saved_rgb_observer \
  tests.physical_response.test_metrics \
  tests.physical_response.test_acceptance -v
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=src:tests:.:/root/Nico/third_party/sam2 \
  /mnt/nvme1/NicoCache/envs/physics_wan/bin/python \
  tests/integration/physical_response_saved_rgb_evaluator_worker.py \
  --dataset datasets/releases/12.0.0/dataset.json \
  --sam2-root /root/Nico/third_party/sam2 \
  --sam2-checkpoint /mnt/nvme1/NicoCache/checkpoints/sam2/sam2.1_hiera_tiny.pt \
  --job-count 2 \
  --output-root /tmp \
  --audit-output /tmp/physical_response_saved_rgb_two_job_audit.json
```

Expected: all unit tests PASS. The integration worker materializes two sealed pair jobs from real encoded Dataset RGB into a temporary run, builds one common target manifest and two arm inputs, proves their scientific identities equal before decoding, runs the hard and differentiable SAM2 paths from disk, scores both pairs, and reopens the written artifacts. It exits zero only if the observer agreement gate passes, both arm artifacts cover the exact two-job edge-seed fixture, and every record-to-report layer retains scene, axis, holdout, edge kind, interpolation/extrapolation, teacher kind, and eligibility. Missing SAM2 assets, an identity mismatch, decode failure, a metric read before identity verification, or a failed agreement gate is a hard failure.

- [ ] **Step 5: Commit independent response evaluation**

```bash
git add src/physbench/physical_response/evaluation_inputs.py \
  src/physbench/physical_response/saved_rgb_observer.py \
  src/physbench/physical_response/metrics.py \
  scripts/build_physical_response_evaluation_targets.py \
  scripts/seal_physical_response_evaluation_input.py \
  scripts/audit_physical_response_observer_validation.py \
  scripts/evaluate_physical_response.py \
  scripts/seal_physical_response_evaluation_policy.py \
  schemas/physical_response_evaluation_policy.schema.json \
  schemas/physical_response_evaluation_input.schema.json \
  schemas/physical_response_evaluation_artifact.schema.json \
  schemas/physical_response_evaluation_authorization.schema.json \
  schemas/physical_response_observer_agreement.schema.json \
  schemas/physical_response_observer_validation.schema.json \
  schemas/physical_response_matrix_test_seal.schema.json \
  tests/physical_response/_fixtures.py \
  tests/physical_response/test_evaluation_inputs.py \
  tests/physical_response/test_saved_rgb_observer.py \
  tests/physical_response/test_metrics.py \
  tests/physical_response/test_acceptance.py \
  tests/integration/physical_response_saved_rgb_evaluator_worker.py
git commit -m "feat: evaluate saved RGB physical responses"
```

### Task 18: Package the model, seal run provenance, and document the complete experiment

**Files:**
- Create: `src/physbench/physical_response/run.py`
- Create: `src/physbench/physical_response/latent_head.py`
- Create: `src/physbench/baselines/wan22_physical_response.py`
- Create: `src/physbench/baseline_plugins/wan22_physical_response.py`
- Create: `src/physbench/baseline_runtime/drivers/wan22_physical_response.py`
- Create: `scripts/run_wan22_physical_response.py`
- Create: `scripts/materialize_physical_response_runtime.py`
- Create: `configs/physical_response/experiments/ablation_matrix_v1.json`
- Create: `configs/physical_response/runtime/local.example.json`
- Create: `schemas/physical_response_local_runtime.schema.json`
- Create: `schemas/physical_response_arm_run.schema.json`
- Create: `schemas/physical_response_arm_evaluation.schema.json`
- Create: `schemas/physical_response_ablation_matrix.schema.json`
- Create: `schemas/physical_response_run.schema.json`
- Create: `baselines/wan22_physical_response/__init__.py`
- Create: `baselines/wan22_physical_response/adapter.py`
- Create: `baselines/wan22_physical_response/driver.py`
- Create: `baselines/wan22_physical_response/baseline.json`
- Create: `baselines/wan22_physical_response/baseline.local.example.json`
- Create: `baselines/wan22_physical_response/quantity_registry_v2.json`
- Create: `baselines/wan22_physical_response/README.md`
- Create: `docs/experiments/PHYSICAL_RESPONSE_LOSS_RUNBOOK.md`
- Create: `tests/physical_response/test_run_manifest.py`
- Create: `tests/physical_response/test_baseline_bundle.py`
- Create: `tests/physical_response/test_ablation_arms.py`
- Create: `tests/physical_response/test_latent_head.py`
- Modify: `scripts/build_physical_response_evaluation_targets.py`
- Modify: `scripts/evaluate_physical_response.py`
- Modify: `tests/test_integrated_baselines.py:31-119`

**Interfaces:**
- Consumes: all Tasks 1-17; combined LoRA/QuantityEncoder checkpoint validation at `src/physbench/baselines/wan22_quantity.py:646-702`; dynamic Baseline discovery at `src/physbench/baseline_api/registry.py:476-549`.
- Produces: `PhysicalResponseRuntimeConfig`, `FrozenRunInputs`, `ArmSpec`, `ArmRunArtifact`, `ArmEvaluationArtifact`, `PhysicalResponseRun`, `LatentResponseAuxiliaryHead`, and `PhysicalResponseRunManager`; `load_physical_response_runtime_config(path: Path) -> PhysicalResponseRuntimeConfig`; `compile_ablation_matrix(path: Path) -> tuple[ArmSpec, ...]`; `seal_matrix_for_response_test(run: PhysicalResponseRun, arm_specs: Sequence[ArmSpec], arms: Sequence[ArmRunArtifact], bindings: Sequence[ArmGenerationBinding], common_jobs: CommonJobManifest, *, run_dir: Path, frozen_inputs: FrozenRunInputs, evaluation_policy: EvaluationPolicy, observer_validation_audit_path: Path, teacher_train_binding_path: Path, teacher_test_binding_path: Path, uncertainty_floors_path: Path, response_test_access_path: Path, seal_output_path: Path, transition_output_path: Path, run_output_path: Path) -> tuple[MatrixTestSeal, PhysicalResponseRun]`; `seal_arm_evaluation_artifact(arm: ArmRunArtifact, binding: ArmGenerationBinding, generation_results: ArmGenerationResultsManifest, evaluation_input: EvaluationInputManifest, evaluation_artifact: EvaluationArtifactManifest, observer_agreement: ObserverAgreementAudit, *, matrix_seal: MatrixTestSeal, common_jobs: CommonJobManifest, evaluation_policy: EvaluationPolicy, response_test_access: ResponseAccessAudit, output_path: Path) -> ArmEvaluationArtifact`; `begin_response_evaluation(run: PhysicalResponseRun, *, matrix_seal: MatrixTestSeal, response_test_access_path: Path, authorization_output_path: Path, transition_output_path: Path, run_output_path: Path) -> tuple[PhysicalResponseRun, EvaluationAuthorization]`; `finalize_physical_response_run(run: PhysicalResponseRun, *, matrix_seal: MatrixTestSeal, common_jobs: CommonJobManifest, arm_evaluations: Sequence[ArmEvaluationArtifact], response_test_access: ResponseAccessAudit, output_path: Path) -> PhysicalResponseRun`; `PhysicalResponseRunManager.freeze_inputs(*, dataset_path: Path, policy_path: Path, split_path: Path, baseline_path: Path, runtime_config_path: Path, run_dir: Path) -> FrozenRunInputs`; `PhysicalResponseRunManager.execute(*, run_dir: Path, runtime_config_path: Path, stages: Sequence[StagePolicy], execute: bool) -> PhysicalResponseRun`; `verify_physical_response_run(run_dir: Path, *, runtime_config_path: Path | None = None) -> PhysicalResponseRun`; `Wan22PhysicalResponseAdapter`; `Wan22PhysicalResponseExecutionEngine`; and `Wan22PhysicalResponseManagedDriver`.
- CLI subcommands: `run_wan22_physical_response.py init --dataset datasets/releases/12.0.0/dataset.json --policy configs/physical_response/policies/physical_response_v1.json --split configs/physical_response/splits/group_holdout_v1.json --baseline wan22_ti2v_5b_lora_r32_physical_response_v1 --runtime-config configs/physical_response/runtime/local.json --run-id physical_response_group_holdout_v1_seed42 --output-root run --execute`; `run_wan22_physical_response.py seal-matrix --run-dir run/physical_response_group_holdout_v1_seed42 --common-jobs run/physical_response_group_holdout_v1_seed42/physical_response/generation/common/manifest.json --evaluation-policy run/physical_response_group_holdout_v1_seed42/frozen/evaluation_policy.json --arms-dir run/physical_response_group_holdout_v1_seed42/arms`; `run_wan22_physical_response.py finalize --run-dir run/physical_response_group_holdout_v1_seed42 --matrix-seal run/physical_response_group_holdout_v1_seed42/frozen/matrix_test_seal.json --arm-evaluations-dir run/physical_response_group_holdout_v1_seed42/evaluation/arm_artifacts`; and `run_wan22_physical_response.py verify --run-dir run/physical_response_group_holdout_v1_seed42`. `materialize_physical_response_runtime.py` writes the ignored external local config from explicit paths and validates it against the local-runtime schema. The init `--runtime-config` is mandatory, must exist before the target run, and must be outside both the Dataset tree and target run directory.
- `LatentResponseAuxiliaryHead.predict_full_rgb_timeline(z0_hat_full: torch.Tensor) -> torch.Tensor`; `LatentResponseAuxiliaryHead.forward(z0_hat_full: torch.Tensor, *, observer_indices: torch.Tensor) -> torch.Tensor` emits the Task-4 flattened state layout at the requested RGB-frame indices; `latent_auxiliary_loss(predicted_packed_state: torch.Tensor, teacher: FactualTargetBatch | ResponseTargetBatch, *, schema: StateSchema) -> torch.Tensor` is enabled only by the latent-head arm and never supplies evaluation state.
- Run contract: this is a separate `physical_response_run_v1`, because the current Task planner cannot encode response member/edge partitions. It never adds unrecognized fields to `BaselineTaskInstance`. `component_fingerprints.json` and `run.json` bind Dataset metadata/tree, policy, split, response manifest, condition index, coordinate binding, teacher bindings, uncertainty floors, stage policies/audits, external orchestration runtime and its task-specific derived runtimes, WAN/SAM2/DiffSynth identities, one common job manifest, seven frozen arm/model-selection artifacts, seven generation-result manifests, seven evaluation artifacts, evaluator policy, matrix seal, and ablation matrix. The pretrained arm is base-only; the other six frozen arms are checkpoint-bound.
Data contracts are:

- `PhysicalResponseRuntimeConfig(schema_version: Literal["physical_response_local_runtime_v1"], interpreter: Path, wan_root: Path, diffsynth_root: Path, sam2_root: Path, sam2_checkpoint: Path, baseline_bundle: Path, quantity_registry: Path, torch_dtype: Literal["bfloat16"], pipeline_loader_version: str, source_path: Path, digest: str)`. It is orchestration-only and deterministically materializes Task-14 `TrainingRuntimeConfig` and Task-16 `GenerationRuntimeConfig`; Task 17 receives the derived SAM2 source/checkpoint paths explicitly. No downstream loader imports this type or adds a default.
- `LatentHeadConfig(input_channels: int, hidden_channels: int, output_schema: StateSchema, auxiliary_weight: float, digest: str)`.
- `ArmSpec(arm_id: Literal["pretrained_wan22", "flowmatch_lora", "flowmatch_quantity", "complete_direct_rgb", "analytic_only", "empirical_only", "latent_head_auxiliary"], trainable_families: tuple[str, ...], enabled_loss_terms: tuple[str, ...], enabled_teacher_roles: tuple[TeacherRole, ...], observer_path: Literal["none", "direct_rgb", "latent_auxiliary"], primary_scientific_result: bool, initialization_digest: str, checkpoint_selection_rule: str, digest: str)`.
- `FrozenRunInputs(schema_version: Literal["physical_response_frozen_run_inputs_v1"], run_id: str, experiment_id: str, dataset_id: str, dataset_release: Literal["12.0.0"], dataset_metadata_digest: str, dataset_tree_digest: str, input_inventory_digest: str, baseline_digest: str, policy_digest: str, split_policy_digest: str, response_manifest_digest: str, condition_index_digest: str, coordinate_binding_digest: str, coordinate_manifest_digest: str, common_job_manifest_digest: str, scientific_job_set_digest: str, runtime_config_relative_path: str, runtime_config_digest: str, training_runtime_config_relative_path: str, training_runtime_config_digest: str, training_runtime_identity_digest: str, generation_runtime_config_relative_path: str, generation_runtime_config_digest: str, generation_runtime_identity_digest: str, environment_identity_digest: str, component_fingerprints_relative_path: str, digest: str)`. Every locator is a normalized run-relative string; absolute paths, `..`, and symlink escape are rejected. The binding digest authenticates the exact run-local pointer, while the manifest digest is the coordinate system's scientific identity.
- `ArmRunArtifact(schema_version: Literal["physical_response_arm_run_v1"], run_id: str, experiment_id: str, arm_id: str, status: Literal["model_frozen", "blocked"], arm_spec_digest: str, stage_checkpoint_artifact_digest: str | None, model_selection: ModelSelectionRef | None, validation_report_digest: str, common_job_manifest_digest: str, scientific_job_set_digest: str, generation_runtime_identity_digest: str, coordinate_manifest_digest: str, blocked_reasons: tuple[str, ...], artifact_digest: str)`. `pretrained_wan22` uses `ModelSelectionRef(load_mode="base_only")` with no adapter locator and `stage_checkpoint_artifact_digest=None`; each of the other six successful arms binds a `StageCheckpointArtifact`. A blocked arm has `model_selection=None`, at least one reason, and cannot be sealed. This artifact is never mutated into a generated/evaluated state.
- `MatrixTestSeal` is the Task-17 contract. Its seed set and edge-seed key identity are copied only from `CommonJobManifest`; callers cannot pass an independent seed tuple. It contains exactly one artifact identity for each of the seven distinct arm IDs.
- `ArmEvaluationArtifact(schema_version: Literal["physical_response_arm_evaluation_v1"], run_id: str, experiment_id: str, arm_id: str, status: Literal["evaluated", "failed"], arm_run_artifact_digest: str, arm_generation_binding_digest: str, matrix_test_seal_digest: str, common_job_manifest_digest: str, scientific_job_set_digest: str, generation_runtime_identity_digest: str, generation_results_manifest_digest: str | None, evaluation_input_digest: str | None, evaluation_policy_digest: str, evaluation_artifact_manifest_digest: str | None, pair_results_sha256: str | None, group_results_sha256: str | None, summary_sha256: str | None, observer_agreement_audit_digest: str | None, response_test_access_audit_digest: str | None, acceptance_decision_digest: str | None, expected_job_count: int, evaluated_job_count: int, evaluated_edge_seed_keys_digest: str | None, failure_reasons: tuple[str, ...], artifact_digest: str)`. An evaluated artifact has every success field, exact common-job coverage, and a passing observer audit. Only `complete_direct_rgb` may bind an acceptance decision and raw arm-3 comparison; a negative acceptance decision remains a valid completed scientific result.
- `RunStatusTransition(schema_version: Literal["physical_response_status_transition_v1"], sequence: int, run_id: str, experiment_id: str, previous_status: str, next_status: str, previous_run_digest: str, matrix_test_seal_digest: str | None, previous_transition_digest: str, digest: str)`; every transition is an immutable run-relative document, and `run.json` is the sole mutable commit pointer.
- `PhysicalResponseRun(schema_version: Literal["physical_response_run_v1"], run_id: str, experiment_id: str, status: Literal["frozen", "training", "blocked", "evaluation_ready", "evaluating", "complete", "failed"], frozen_inputs: FrozenRunInputs, stage_audit_digests: tuple[str, ...], ablation_matrix_digest: str, common_job_manifest_digest: str | None, evaluation_policy_digest: str | None, matrix_test_seal_digest: str | None, evaluation_authorization_digest: str | None, arm_run_artifact_digests: tuple[tuple[str, str], ...], arm_generation_binding_digests: tuple[tuple[str, str], ...], arm_generation_results_manifest_digests: tuple[tuple[str, str], ...], arm_evaluation_artifact_digests: tuple[tuple[str, str], ...], response_test_access_audit_digest: str, response_test_access_count: int, scientific_acceptance_decision_digest: str | None, scientific_acceptance_passed: bool | None, latest_status_transition_relative_path: str, latest_status_transition_digest: str, status_transition_count: int, failure_reasons: tuple[str, ...], run_digest: str)`.

- [ ] **Step 1: Write first-run runtime, seven-arm seal/finalize, latent-head, Dataset immutability, and deployment tests**

```python
class PhysicalResponseRunManifestTests(unittest.TestCase):
    def test_run_verification_detects_overlay_mutation(self) -> None:
        run_dir = build_fixture_physical_response_run()
        mutate_json(run_dir / "physical_response" / "manifest.json", "policy_digest")
        with self.assertRaisesRegex(ValueError, "response manifest digest mismatch"):
            verify_physical_response_run(run_dir)

    def test_run_has_no_response_test_access_before_final_evaluation(self) -> None:
        run = verify_physical_response_run(build_fixture_pretest_run())
        self.assertEqual(0, run.response_test_access_count)

    def test_first_run_accepts_external_runtime_before_run_dir_exists(self) -> None:
        run_dir = fixture_nonexistent_run_dir()
        frozen = PhysicalResponseRunManager().freeze_inputs(
            dataset_path=fixture_dataset_path(),
            policy_path=fixture_policy_path(),
            split_path=fixture_split_path(),
            baseline_path=fixture_baseline_path(),
            runtime_config_path=fixture_external_runtime_path(),
            run_dir=run_dir,
        )
        self.assertTrue(run_dir.is_dir())
        self.assertEqual(fixture_resolved_runtime().digest, frozen.generation_runtime_identity_digest)

    def test_runtime_inside_dataset_or_target_run_is_rejected(self) -> None:
        for location in ("dataset", "target_run", "symlink_escape"):
            with self.subTest(location=location), self.assertRaisesRegex(
                ValueError, "runtime config must be external",
            ):
                freeze_fixture_inputs(runtime_location=location)

    def test_existing_run_is_rejected_without_writing_or_allocating(self) -> None:
        allocator = mock.Mock()
        before = inventory_tree(fixture_dataset_root())
        with self.assertRaisesRegex(FileExistsError, "run already exists"):
            freeze_fixture_inputs(existing_run=True, allocator=allocator)
        allocator.assert_not_called()
        self.assertEqual(before, inventory_tree(fixture_dataset_root()))

class PhysicalResponseBaselineBundleTests(unittest.TestCase):
    def test_bundle_is_discovered_and_deployment_has_no_sam2_dependency(self) -> None:
        bundle = load_baseline_bundle("wan22_ti2v_5b_lora_r32_physical_response_v1")
        plugin = load_baseline_plugin(bundle)
        dependencies = plugin.driver.dependency_paths()
        self.assertNotIn("sam2", " ".join(map(str, dependencies.values())).lower())
        self.assertEqual(
            ["pendulum", "collision_1d", "inclined_plane_slide", "uniform_circular_motion", "parabolic_motion"],
            bundle.value["supported_scenes"],
        )

    def test_deployment_import_and_checkpoint_smoke_without_sam2(self) -> None:
        result = run_deployment_subprocess(
            block_import_prefixes=("sam2", "physbench.physical_response.saved_rgb_observer"),
            remove_sam2_from_pythonpath=True,
        )
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("checkpoint_validated", result.stdout)

class AblationArmTests(unittest.TestCase):
    def test_matrix_has_exactly_seven_executable_arms(self) -> None:
        arms = compile_ablation_matrix(fixture_ablation_matrix_path())
        self.assertEqual({
            "pretrained_wan22", "flowmatch_lora", "flowmatch_quantity",
            "complete_direct_rgb", "analytic_only", "empirical_only",
            "latent_head_auxiliary",
        }, {arm.arm_id for arm in arms})
        self.assertEqual(7, len(arms))
        latent = arm_by_id(arms, "latent_head_auxiliary")
        self.assertEqual("latent_auxiliary", latent.observer_path)
        self.assertFalse(latent.primary_scientific_result)

    def test_analytic_and_empirical_arms_have_disjoint_teacher_modes(self) -> None:
        arms = compile_ablation_matrix(fixture_ablation_matrix_path())
        self.assertEqual(
            {"analytic_weak", "analytic_strong", "absolute_empirical"},
            set(arm_by_id(arms, "analytic_only").enabled_teacher_roles),
        )
        self.assertEqual(
            {"empirical_primary", "absolute_empirical"},
            set(arm_by_id(arms, "empirical_only").enabled_teacher_roles),
        )

    def test_response_test_seal_requires_all_frozen_arm_models_and_common_jobs(self) -> None:
        with self.assertRaisesRegex(ValueError, "seven frozen arm artifacts"):
            seal_matrix_for_response_test(
                fixture_training_run(), fixture_arm_specs(), six_frozen_arm_artifacts(),
                six_arm_generation_bindings(), fixture_common_jobs(),
                run_dir=fixture_run_dir(),
                frozen_inputs=fixture_frozen_inputs(),
                evaluation_policy=fixture_evaluation_policy(),
                observer_validation_audit_path=fixture_observer_validation_audit_path(),
                teacher_train_binding_path=fixture_teacher_train_binding_path(),
                teacher_test_binding_path=fixture_teacher_test_binding_path(),
                uncertainty_floors_path=fixture_uncertainty_floors_path(),
                response_test_access_path=fixture_response_test_access_path(count=0),
                seal_output_path=fixture_matrix_seal_path(),
                transition_output_path=fixture_status_transition_path("evaluation_ready"),
                run_output_path=fixture_run_json_path(),
            )

    def test_one_blocked_arm_prevents_matrix_seal_and_test_access(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "blocked arm"):
            seal_matrix_for_response_test(
                fixture_training_run(), fixture_arm_specs(),
                seven_arm_artifacts(blocked_arm_id="complete_direct_rgb"),
                fixture_seven_arm_generation_bindings(), fixture_common_jobs(),
                run_dir=fixture_run_dir(),
                frozen_inputs=fixture_frozen_inputs(),
                evaluation_policy=fixture_evaluation_policy(),
                observer_validation_audit_path=fixture_observer_validation_audit_path(),
                teacher_train_binding_path=fixture_teacher_train_binding_path(),
                teacher_test_binding_path=fixture_teacher_test_binding_path(),
                uncertainty_floors_path=fixture_uncertainty_floors_path(),
                response_test_access_path=fixture_response_test_access_path(count=0),
                seal_output_path=fixture_matrix_seal_path(),
                transition_output_path=fixture_status_transition_path("evaluation_ready"),
                run_output_path=fixture_run_json_path(),
            )
        self.assertFalse(fixture_matrix_seal_path().exists())
        self.assertEqual(0, response_test_access_count())

    def test_successful_matrix_seal_atomically_enters_evaluation_ready(self) -> None:
        seal, ready = seal_fixture_matrix(return_run=True)
        self.assertEqual("evaluation_ready", ready.status)
        self.assertEqual(seal.digest, ready.matrix_test_seal_digest)
        self.assertEqual(0, ready.response_test_access_count)

    def test_latent_head_is_auxiliary_to_generated_rgb_not_a_primary_evaluator(self) -> None:
        head = LatentResponseAuxiliaryHead(fixture_latent_head_config())
        prediction = head(
            fixture_wan_latents(), observer_indices=fixture_observer_indices()
        )
        self.assertEqual(fixture_auxiliary_state_shape(), prediction.shape)
        self.assertNotIn("latent_head", independent_saved_rgb_metric_inputs())

    def test_latent_head_requires_valid_explicit_observer_indices(self) -> None:
        head = LatentResponseAuxiliaryHead(fixture_latent_head_config())
        with self.assertRaises(TypeError):
            head(fixture_wan_latents())
        for indices in (
            torch.tensor([4, 4]), torch.tensor([7, 3]),
            torch.tensor([-1, 4]), torch.tensor([0, 121]),
        ):
            with self.subTest(indices=indices.tolist()), self.assertRaises(ValueError):
                head(fixture_wan_latents(), observer_indices=indices)

    def test_latent_head_builds_rgb_timeline_before_selecting_indices(self) -> None:
        head = LatentResponseAuxiliaryHead(fixture_latent_head_config())
        z0_hat_full = fixture_wan_latents()
        indices = torch.tensor([0, 16, 32, 120])
        full = head.predict_full_rgb_timeline(z0_hat_full)
        selected = head(z0_hat_full, observer_indices=indices)
        self.assertEqual(121, full.shape[1])
        self.assertTensorEqual(full.index_select(1, indices), selected)

    def test_latent_head_index_zero_uses_detached_conditioned_latent(self) -> None:
        conditioned, generated, z0_hat_full = fixture_split_full_latents()
        output = LatentResponseAuxiliaryHead(fixture_latent_head_config())(
            z0_hat_full, observer_indices=torch.tensor([0, 120])
        )
        output.sum().backward()
        self.assertIsNone(conditioned.grad)
        self.assertGreater(float(generated.grad.abs().sum()), 0.0)

    def test_matrix_seal_rejects_wrong_jobs_duplicate_or_missing_arm(self) -> None:
        for changed in ("common_jobs", "duplicate_arm", "missing_arm", "runtime"):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                seal_fixture_matrix(changed=changed)

class ArmFinalizationTests(unittest.TestCase):
    def test_response_access_requires_evaluating_transition_first(self) -> None:
        run = fixture_evaluation_ready_run()
        with self.assertRaisesRegex(PermissionError, "run is not evaluating"):
            open_fixture_response_test_teacher(run)
        evaluating, authorization = begin_response_evaluation(
            run, matrix_seal=fixture_matrix_seal(),
            response_test_access_path=fixture_response_test_access_path(count=0),
            authorization_output_path=fixture_evaluation_authorization_path(),
            transition_output_path=fixture_status_transition_path("evaluating"),
            run_output_path=fixture_run_json_path(),
        )
        self.assertEqual("evaluating", evaluating.status)
        self.assertEqual(authorization.digest, evaluating.evaluation_authorization_digest)
        self.assertEqual(
            authorization,
            load_evaluation_authorization(
                fixture_evaluation_authorization_path(),
                run_dir=fixture_run_dir(), matrix_seal=fixture_matrix_seal(),
                response_test_access=fixture_response_test_access(count=0),
            ),
        )

    def test_failed_run_commit_pointer_does_not_activate_orphan_authorization(self) -> None:
        run = fixture_evaluation_ready_run()
        with self.assertRaisesRegex(RuntimeError, "run compare-and-swap"):
            begin_response_evaluation(
                run, matrix_seal=fixture_matrix_seal(),
                response_test_access_path=fixture_response_test_access_path(count=0),
                authorization_output_path=fixture_evaluation_authorization_path(),
                transition_output_path=fixture_status_transition_path("evaluating"),
                run_output_path=fixture_run_json_path(changed_after_read=True),
            )
        self.assertEqual("evaluation_ready", load_fixture_run_json().status)
        with self.assertRaisesRegex(PermissionError, "run is not evaluating"):
            load_evaluation_authorization(
                fixture_evaluation_authorization_path(), run_dir=fixture_run_dir(),
                matrix_seal=fixture_matrix_seal(),
                response_test_access=fixture_response_test_access(count=0),
            )

    def test_complete_requires_exactly_seven_successful_evaluations(self) -> None:
        for changed in (
            "six_arms", "failed_arm", "wrong_seal", "wrong_jobs",
            "missing_edge_seed", "extra_edge_seed", "observer_gate_failed",
        ):
            with self.subTest(changed=changed):
                failed = finalize_fixture_run(changed=changed)
                self.assertEqual("failed", failed.status)
                self.assertTrue(failed.failure_reasons)

    def test_exactly_seven_complete_evaluations_finish_the_run(self) -> None:
        completed = finalize_physical_response_run(
            fixture_evaluating_run(),
            matrix_seal=fixture_matrix_seal(),
            common_jobs=fixture_common_jobs(),
            arm_evaluations=fixture_seven_evaluated_arms(),
            response_test_access=fixture_response_test_access(count=1),
            output_path=fixture_run_json_path(),
        )
        self.assertEqual("complete", completed.status)

    def test_negative_scientific_acceptance_is_still_a_complete_experiment(self) -> None:
        completed = finalize_fixture_run(scientific_acceptance_passed=False)
        self.assertEqual("complete", completed.status)
        self.assertFalse(completed.scientific_acceptance_passed)

    def test_failure_after_matrix_seal_marks_run_failed(self) -> None:
        failed = finalize_fixture_run(changed="post_seal_generation_failure", raise_error=False)
        self.assertEqual("failed", failed.status)
```

- [ ] **Step 2: Run run/bundle tests and observe missing package components**

Run:

```bash
PYTHONPATH=src:tests:. /mnt/nvme1/NicoCache/envs/physics_wan/bin/python \
  -m unittest tests.physical_response.test_run_manifest \
  tests.physical_response.test_baseline_bundle \
  tests.physical_response.test_ablation_arms \
  tests.physical_response.test_latent_head -v
```

Expected: FAIL because the run manager and Baseline bundle do not exist.

- [ ] **Step 3: Implement external-runtime freezing, the seven-arm state machine, and a SAM2-free deployment bundle**

```python
class PhysicalResponseRunManager:
    def freeze_inputs(
        self,
        *,
        dataset_path: Path,
        policy_path: Path,
        split_path: Path,
        baseline_path: Path,
        runtime_config_path: Path,
        run_dir: Path,
    ) -> FrozenRunInputs:
        reject_existing_or_dataset_contained_run_dir(run_dir)
        reject_runtime_config_inside_dataset_or_run(
            runtime_config_path, dataset_path.parents[2], run_dir
        )
        dataset_before = inventory_tree(dataset_path.parents[2])
        runtime_config = load_physical_response_runtime_config(runtime_config_path)
        training_runtime = runtime_config.to_training_runtime_config()
        generation_runtime = runtime_config.to_generation_runtime_config()
        resolved_training_runtime = resolve_and_verify_training_runtime(training_runtime)
        resolved_runtime = resolve_and_verify_generation_runtime(generation_runtime)
        staging = create_run_staging_sibling(run_dir)
        overlay, condition_index = compile_overlay_in_run(
            dataset_path, policy_path, split_path, staging
        )
        coordinate_binding = build_coordinates_in_run(staging, overlay, condition_index)
        common_jobs = compile_common_jobs_in_run(
            staging, overlay, condition_index, coordinate_binding
        )
        create_response_access_audit(
            staging / "access" / "response_validation.json",
            partition="response_validation",
        )
        create_response_access_audit(
            staging / "access" / "response_test.json",
            partition="response_test",
        )
        environment = capture_runtime_identity(resolved_runtime)
        dataset_after = inventory_tree(dataset_path.parents[2])
        assert_same_inventory(dataset_before, dataset_after)
        frozen = write_frozen_run_inputs(
            staging, overlay, condition_index, coordinate_binding, common_jobs,
            runtime_config, training_runtime, generation_runtime,
            resolved_training_runtime, resolved_runtime, environment,
            load_baseline_bundle(baseline_path),
        )
        finalize_staging_run(staging, run_dir)
        return frozen

def begin_response_evaluation(
    run: PhysicalResponseRun,
    *,
    matrix_seal: MatrixTestSeal,
    response_test_access_path: Path,
    authorization_output_path: Path,
    transition_output_path: Path,
    run_output_path: Path,
) -> tuple[PhysicalResponseRun, EvaluationAuthorization]:
    if run.status != "evaluation_ready":
        raise ValueError("response evaluation requires evaluation_ready")
    if run.matrix_test_seal_digest != matrix_seal.digest:
        raise ValueError("matrix seal mismatch")
    access = load_response_access_audit(
        response_test_access_path, expected_partition="response_test"
    )
    if (
        access.access_count != 0
        or access.digest != matrix_seal.response_test_access_audit_digest_at_seal
    ):
        raise PermissionError("response-test access changed after matrix seal")
    transition = RunStatusTransition.from_verified_link(
        run=run, next_status="evaluating", matrix_seal_digest=matrix_seal.digest,
    )
    run_root = run_output_path.parent.resolve()
    authorization = EvaluationAuthorization.from_verified_transition(
        run_id=run.run_id, experiment_id=run.experiment_id,
        run_relative_path=normalized_relative_path(run_output_path, run_root),
        transition_relative_path=normalized_relative_path(
            transition_output_path, run_root
        ),
        previous_run_digest=run.run_digest,
        evaluation_transition_digest=transition.digest,
        matrix_test_seal_digest=matrix_seal.digest,
        response_test_access_audit_digest_at_authorization=access.digest,
        response_test_access_count_at_authorization=0,
        purpose="build_response_test_targets",
    )
    evaluating = dataclasses.replace(
        run, status="evaluating",
        evaluation_authorization_digest=authorization.digest,
        latest_status_transition_relative_path=authorization.transition_relative_path,
        latest_status_transition_digest=transition.digest,
        status_transition_count=run.status_transition_count + 1,
    ).with_recomputed_digest()
    write_new_typed_document(transition_output_path, transition)
    write_new_typed_document(authorization_output_path, authorization)
    compare_and_swap_run_commit_pointer(
        run_output_path, expected_run_digest=run.run_digest,
        replacement=evaluating,
    )
    return evaluating, authorization

def finalize_physical_response_run(
    run: PhysicalResponseRun,
    *,
    matrix_seal: MatrixTestSeal,
    common_jobs: CommonJobManifest,
    arm_evaluations: Sequence[ArmEvaluationArtifact],
    response_test_access: ResponseAccessAudit,
    output_path: Path,
) -> PhysicalResponseRun:
    if run.status != "evaluating":
        raise ValueError("finalization requires evaluating state")
    expected_ids = required_ablation_arm_ids()
    by_id, identity_failures = audit_unique_records_by_arm(arm_evaluations)
    failures = list(identity_failures)
    if set(by_id) != expected_ids:
        failures.append("exactly_seven_arm_evaluations_required")
    for arm_id, artifact in by_id.items():
        failures.extend(audit_arm_evaluation_against_seal_and_common_jobs(
            arm_id, artifact, matrix_seal, common_jobs,
            response_test_access, run.frozen_inputs,
        ))
        if artifact.status != "evaluated":
            failures.append(f"arm_evaluation_failed:{arm_id}")
    if failures:
        failed = dataclasses.replace(
            run,
            response_test_access_audit_digest=response_test_access.digest,
            response_test_access_count=response_test_access.access_count,
            failure_reasons=tuple(sorted(set(failures))),
        )
        return append_verified_status_transition(
            failed, next_status="failed", output_path=output_path
        )
    completed = dataclasses.replace(
        run,
        arm_evaluation_artifact_digests=tuple(
            sorted((arm_id, artifact.artifact_digest) for arm_id, artifact in by_id.items())
        ),
        response_test_access_audit_digest=response_test_access.digest,
        response_test_access_count=response_test_access.access_count,
        scientific_acceptance_decision_digest=complete_arm_acceptance_digest(by_id),
        scientific_acceptance_passed=complete_arm_acceptance_passed(by_id),
        failure_reasons=(),
    )
    return append_verified_status_transition(
        completed, next_status="complete", output_path=output_path
    )
```

The checked-in orchestration example is complete but portable; the materializer injects the declared workstation paths into an ignored external file:

```json
{
  "schema_version": "physical_response_local_runtime_v1",
  "interpreter": "../external/physics_wan/bin/python",
  "wan_root": "../external/Wan2.2-TI2V-5B",
  "diffsynth_root": "../external/DiffSynth-Studio",
  "sam2_root": "../external/sam2",
  "sam2_checkpoint": "../external/sam2.1_hiera_tiny.pt",
  "baseline_bundle": "../../../baselines/wan22_physical_response/baseline.json",
  "quantity_registry": "../../../baselines/wan22_physical_response/quantity_registry_v2.json",
  "torch_dtype": "bfloat16",
  "pipeline_loader_version": "wan22_physical_response_authenticated_v1"
}
```

`freeze_inputs` verifies the external orchestration runtime before creating the target run, builds derived static artifacts in a sibling staging directory, and writes the unified document plus strict Task-14 training and Task-16 generation documents into `run/local/` only as frozen evidence. The copied unified file can never bootstrap a first run. Every later training or generation process receives only its task-specific derived document, re-resolves the current environment, and must equal the corresponding `FrozenRunInputs.training_runtime_identity_digest` or `generation_runtime_identity_digest` before model allocation. Task 17 receives the verified SAM2 source/checkpoint paths explicitly and reconstructs its observer identities; it never parses the orchestration document. The single common job manifest is built once and its identity is reused by all arms.

The manager executes cache preflight, warmup, 16/32/121 gated stages, seven model selections (one base-only plus six checkpoint-bound), seven calls to Task-16 `seal_arm_generation_binding`, matrix sealing, seven paired generations, seven saved-RGB evaluations, and final verification in that order. Its only legal status transitions are `frozen -> training`, `training -> blocked | evaluation_ready | failed`, `evaluation_ready -> evaluating`, and `evaluating -> complete | failed`; `blocked`, `failed`, and `complete` are terminal. Every transition is a numbered immutable record binding the prior run and transition. `run.json` is updated by compare-and-swap last and is the only commit pointer. Task 18 calls `begin_response_evaluation`, then passes the resulting authorization path to Task 17's target builder before any response-test teacher or video access.

The ablation matrix contains exactly the seven typed arms in `ArmSpec`. Pretrained, FlowMatch-LoRA, and FlowMatch-plus-QuantityEncoder have physical losses disabled; complete uses the direct-RGB empirical/analytic path; analytic-only and empirical-only have disjoint sealed teacher-role/loss masks; latent-head auxiliary instantiates `LatentResponseAuxiliaryHead` and adds its auxiliary state loss while saved RGB remains the only scientific evaluator and `primary_scientific_result=false`. The head consumes `z0_hat_full`, predicts a state sequence aligned to all 121 RGB frames, and only then applies `index_select` with strictly increasing `observer_indices`; it never applies RGB-frame indices directly to the shorter latent-time dimension. RGB index zero is legal and corresponds to the detached conditioned first-frame latent. Every arm must end validation as either `model_frozen` or `blocked`; missing arms are not silently dropped. A blocked arm leaves the pre-test run `blocked`, cannot satisfy `MatrixTestSeal`, cannot be omitted, and cannot open response test. The optional middle/late DiT-block unfreezing study is explicitly deferred to a future independent plan and run identity; it cannot overwrite or join this seven-arm LoRA experiment.

Before any response-test teacher binding opens, `seal_matrix_for_response_test` requires a `training` run, exactly the seven distinct required `ArmSpec` IDs, seven matching `ArmRunArtifact` records with `status="model_frozen"`, and seven matching bindings created through Task-16's verified constructor. It requires the pretrained selection to be base-only with no checkpoint and each other arm to be checkpoint-bound. It loads the explicitly named observer-validation audit, train/test teacher-binding headers, and frozen uncertainty floors without opening teacher records, then recomputes rather than trusts each model-selection, validation, coordinate-manifest, runtime, common-job, scientific-job-set, response policy/manifest, condition-index, teacher-binding, uncertainty-floor, evaluation-policy, and persistent zero-access identity. Seed IDs and edge-seed keys come only from `CommonJobManifest`. After all checks pass, it publishes the immutable matrix seal and immutable `training -> evaluation_ready` transition, then compare-and-swaps `run.json` last; only that final pointer activates the seal. A failed precondition publishes nothing. A failed final pointer update may leave immutable unreferenced candidates, but the prior run remains authoritative and response-test access remains unchanged. Group-holdout and value-holdout use different run IDs, manifests, policies, common jobs, and matrix seals.

After sealing, no checkpoint, binding, threshold, target policy, coordinate, job, or arm list may change. Each arm generation must produce an `ArmGenerationResultsManifest` covering exactly the common jobs. Each evaluator must produce a passing `EvaluationArtifactManifest`, which is wrapped in an `ArmEvaluationArtifact` without rewriting any earlier lifecycle artifact. `finalize_physical_response_run` recomputes all identities, requires exactly one response-test access event with purpose `test_target_build` and no later teacher access, and permits `complete` only for exactly seven `status="evaluated"` artifacts whose expected/evaluated job counts and edge-seed keys equal the common manifest. A pre-seal blocked arm makes the run `blocked`; any generation, observer-agreement, scoring, coverage, or artifact failure after the matrix seal makes the run `failed`, never a partial success. `complete` means the seven-arm experiment and audits are complete; an `AcceptanceDecision.passed=false` is a complete negative scientific conclusion and is stored separately rather than turning the run into `failed`. The deployment engine reuses the existing quantity-conditioned WAN inference/checkpoint path and contains no import or runtime reference to SAM2, teacher cache, response loss, evaluator, or latent auxiliary head; a subprocess import blocker proves this rather than relying only on dependency-path strings.

- [ ] **Step 4: Run first-run dry-build and the complete verification matrix**

Run a real first-run dry build from an external runtime document:

```bash
PYTHONPATH=src:. /mnt/nvme1/NicoCache/envs/physics_wan/bin/python \
  scripts/materialize_physical_response_runtime.py \
  --interpreter /mnt/nvme1/NicoCache/envs/physics_wan/bin/python \
  --wan-root /root/Steven/wan22_pendulum_pipeline/models/Wan-AI/Wan2.2-TI2V-5B \
  --diffsynth-root /root/Steven/wan22_pendulum_pipeline/vendor/DiffSynth-Studio \
  --sam2-root /root/Nico/third_party/sam2 \
  --sam2-checkpoint /mnt/nvme1/NicoCache/checkpoints/sam2/sam2.1_hiera_tiny.pt \
  --baseline-bundle /root/Steven/VPhysBench/baselines/wan22_physical_response/baseline.json \
  --quantity-registry /root/Steven/VPhysBench/baselines/wan22_physical_response/quantity_registry_v2.json \
  --output /tmp/physical_response_local_runtime.json
PYTHONPATH=src:. /mnt/nvme1/NicoCache/envs/physics_wan/bin/python \
  scripts/run_wan22_physical_response.py init \
  --dataset datasets/releases/12.0.0/dataset.json \
  --policy configs/physical_response/policies/physical_response_v1.json \
  --split configs/physical_response/splits/group_holdout_v1.json \
  --baseline wan22_ti2v_5b_lora_r32_physical_response_v1 \
  --runtime-config /tmp/physical_response_local_runtime.json \
  --run-id physical_response_first_run_dry_build \
  --output-root /tmp \
  --dry-build
```

Expected: the target run does not exist at entry; the external orchestration runtime is accepted, task-specific runtime documents are derived and verified, the Dataset remains unchanged, and the response manifest, coordinate binding, and one common-job manifest are written in a staging sibling and verified after finalization; no model is allocated and response-test access remains zero.

Validate all lifecycle entry points without opening response test:

```bash
PYTHONPATH=src:. /mnt/nvme1/NicoCache/envs/physics_wan/bin/python scripts/run_wan22_physical_response.py init --help
PYTHONPATH=src:. /mnt/nvme1/NicoCache/envs/physics_wan/bin/python scripts/run_wan22_physical_response.py seal-matrix --help
PYTHONPATH=src:. /mnt/nvme1/NicoCache/envs/physics_wan/bin/python scripts/run_wan22_physical_response.py finalize --help
PYTHONPATH=src:. /mnt/nvme1/NicoCache/envs/physics_wan/bin/python scripts/run_wan22_physical_response.py verify --help
```

Expected: all four help commands exit zero without model allocation, artifact mutation, or response-test access.

Run lightweight physical-response tests:

```bash
PYTHONPATH=src:tests:.:/root/Nico/third_party/sam2:/root/Steven/wan22_pendulum_pipeline/vendor/DiffSynth-Studio \
  /mnt/nvme1/NicoCache/envs/physics_wan/bin/python \
  -m unittest discover -s tests/physical_response -p 'test_*.py' -v
```

Run Dataset and Baseline regressions:

```bash
PYTHONPATH=src:tests:. /mnt/nvme1/NicoCache/envs/physics_wan/bin/python -m unittest \
  tests.test_current_dataset \
  tests.test_single_current_physics_v12 \
  tests.test_entity_manifest \
  tests.test_wan22_adapter \
  tests.test_wan22_quantity_embedding \
  tests.test_wan22_quantity_dependency_lock \
  tests.test_baseline_bundle_v5 \
  tests.test_integrated_baselines \
  tests.test_managed_baselines -v
```

Run Dataset validation and confirm no Dataset diff:

```bash
PYTHONPATH=src:tests:. /mnt/nvme1/NicoCache/envs/physics_wan/bin/python \
  scripts/validate_dataset_v12.py
git status --short datasets
```

Run the required GPU gates from Tasks 5, 13, and 15, then execute representative stage preflights for 16, 32, and 121 observer frames. Expected: every lightweight/regression test passes; Dataset validation passes; the Dataset status is empty; distributed cosine/norm gates pass; each feasible stage emits a complete finite audit, while an infeasible 121-frame stage remains explicitly blocked.

- [ ] **Step 5: Write the operator runbook and commit the complete bundle**

The runbook gives exact commands for both holdout experiments, all seven comparison arms, cache construction, validation-only stage sealing, training, five-seed paired generation, independent evaluation, run verification, checkpoint merge, and SAM2-free inference. It labels incomplete/blocked stages and `not_estimable` strata without imputing scores.

```bash
git add src/physbench/physical_response/run.py \
  src/physbench/physical_response/latent_head.py \
  src/physbench/baselines/wan22_physical_response.py \
  src/physbench/baseline_plugins/wan22_physical_response.py \
  src/physbench/baseline_runtime/drivers/wan22_physical_response.py \
  scripts/run_wan22_physical_response.py \
  scripts/materialize_physical_response_runtime.py \
  scripts/build_physical_response_evaluation_targets.py \
  scripts/evaluate_physical_response.py \
  configs/physical_response/experiments/ablation_matrix_v1.json \
  configs/physical_response/runtime/local.example.json \
  schemas/physical_response_local_runtime.schema.json \
  schemas/physical_response_arm_run.schema.json \
  schemas/physical_response_arm_evaluation.schema.json \
  schemas/physical_response_ablation_matrix.schema.json \
  schemas/physical_response_run.schema.json \
  baselines/wan22_physical_response/__init__.py \
  baselines/wan22_physical_response/adapter.py \
  baselines/wan22_physical_response/driver.py \
  baselines/wan22_physical_response/baseline.json \
  baselines/wan22_physical_response/baseline.local.example.json \
  baselines/wan22_physical_response/quantity_registry_v2.json \
  baselines/wan22_physical_response/README.md \
  docs/experiments/PHYSICAL_RESPONSE_LOSS_RUNBOOK.md \
  tests/physical_response/test_run_manifest.py \
  tests/physical_response/test_baseline_bundle.py \
  tests/physical_response/test_ablation_arms.py \
  tests/physical_response/test_latent_head.py \
  tests/test_integrated_baselines.py
git commit -m "feat: package physical response world model"
```

## Final Evidence Checklist

- [ ] Scope evidence proves exactly seven response axes: pendulum length/initial-angle magnitude, two-ball mass/signed initial velocity, incline angle, projectile signed horizontal velocity, and circular radius; `mu_k=0.463` appears only in the incline analytic term, no restitution parameter exists, and no Dataset release is created or modified.
- [ ] The resolved overlay is deterministic, contains no View A test member, proves source-trial/near-duplicate isolation, freezes adjacency before partitioning, and leaves `datasets/` byte-identical.
- [ ] The manifest-to-`ConditionIndex` chain verifies every video, first frame, mask manifest, and initial-mask digest before cache construction, batching, generation, or scoring.
- [ ] Teacher bindings hash every referenced media/mask byte and SAM2/code/config/checkpoint identity; the trainer can enumerate only `response_train` records.
- [ ] Fixed-noise `flow_match_core` matches vendor FlowMatch scalar and gradients; conditioned latent index zero is detached; all 121 frames are decoded causally before observer subsampling.
- [ ] A real three-frame SAM2 test and an end-to-end pair test prove response gradients reach both LoRA and `QuantityEncoder`, while WAN base, UMT5, VAE, and SAM2 remain frozen.
- [ ] Near-binary soft masks match independent hard-mask moment/state references; disappearance, identity-swap, penetration, and missing-event fixtures all fail closed without generated self-masking or relabeling.
- [ ] Changing only structured signed `q` changes the condition branch; the same sensitivity gate rejects shuffled quantities and a detached condition branch.
- [ ] Two-rank FP32 gradients meet cosine `>=0.999` and relative norm error `<=1%`; rank pairs and logical-update recovery never drift.
- [ ] Every loss and metric reports eligibility/counts separately for absolute, sign, magnitude, zero, analytic, and validity components; three-ball response counts are exactly zero.
- [ ] Stage audits preserve 121-frame WAN modeling, record requested 16/32/121 observer counts, and block rather than downshift on memory or gradient failure.
- [ ] Every primary response-test edge has exactly five pair-affine common-noise jobs and an independently re-observed saved-RGB result.
- [ ] Both holdout experiments freeze one `EvaluationPolicy` with zero prior test accesses before any response-test binding opens.
- [ ] All seven arms have frozen model-selection, validation, and common-job identities in one `MatrixTestSeal`; pretrained is base-only, the other six are checkpoint-bound, and a missing or blocked arm prevents test access.
- [ ] Acceptance compares the complete model against arm 3 with group-level paired-bootstrap intervals, scene-level safeguards, frozen non-inferiority margins, and joint elastic-collision gates.
- [ ] The arm-3 comparison is reconstructed from its raw `pair_results.jsonl` with matched edge/seed keys; no summary artifact substitutes for paired records.
- [ ] The final merged deployment loads with no SAM2 dependency, and both group/value runs verify against all frozen component fingerprints.
