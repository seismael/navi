# PLAN.md - Imported Architecture Adoption Program

## 1. Purpose

This plan replaces the previous mixed benchmark diary and roadmap with one
coherent adoption program for the imported architecture ideas.

The goal is not to copy the imported documents verbatim. The goal is to:

- adopt the strongest architectural ideas from `imported/`
- correct the mathematical and logical issues found in those documents
- integrate the adopted design into Navi without violating `AGENTS.md`
- prove correctness, stability, and performance end to end before promotion

This plan is the canonical implementation roadmap for the large-scale refactor.

## 2. Target Outcome

At completion, Navi will have one production architecture with the following
properties:

Continu

The final system must be mathematically specified, operationally complete,
performance-tested, and regression-protected.

## 3. Non-Negotiable Constraints

This adoption must remain aligned with `AGENTS.md`.

The plan therefore preserves these repository rules:

1. the actor cognitive pipeline remains sacred and unchanged in topology
2. canonical production training remains one in-process trainer
3. no service-to-service hot-loop transport is reintroduced
4. the actor-facing public contract remains `DistanceMatrix`
5. canonical training remains CUDA-only on the compiled `sdfdag` path
6. all runtime evolution stays benchmark-gated on the real trainer
7. no alternate production trainer or equal-status runtime surface is created

## 4. Imported Architecture Adoption Decisions

### 4.1 Adopted Directly

These imported ideas are accepted as core direction:

- strict tensor boundary validation between Python and CUDA
- offline dataset normalization and compile-first corpus preparation
- batched compiled-scene execution over reusable GPU buffers
- passive dashboard and dataset-auditor direction
- explicit multi-layer verification language
- stronger documentation of low-level runtime contracts

### 4.2 Adopted With Correction

These imported ideas are accepted only after mathematical or logical
correction:

- sphere tracing claims must use bounded-complexity language, not `O(1)` claims
- DAG deduplication must use explicit canonical hash rules plus mandatory
  structural equality fallback
- leaf payload precision claims must reflect real numeric error bounds
- tensor normalization contracts must use explicit tolerances, not exact float
  equality
- dataset coordinate transforms must be specified as explicit matrices and
  validated by golden fixtures
- episodic-memory reward claims must be reduced to explicit equations and
  thresholds
- observer transport must clearly distinguish ZMQ ingestion from any browser
  proxy surface

### 4.3 Rejected As Production Architecture

These imported directions are not adopted into the canonical production path:

- distributed hot-loop actor/environment stepping
- multiple equal-status training topologies
- swarm cognition as the production default
- unqualified `>10,000 SPS` headline claims as architecture truth
- undefined overlap or double-buffering schemes that widen the production
  architecture without trainer-proof wins

### 4.4 Benchmark-Gated Research Only

These imported ideas remain candidates for later phases only:

- truncation-aware storage metadata
- alternate leaf distance payload precision policies
- Morton or related layout redesigns
- overlap strategies inside the runtime or trainer
- shared global episodic memory and coordination research

## 5. Canonical Design Answers To Imported Open Questions

This section records the best current architectural answers so the refactor can
proceed without unresolved ambiguity.

### 5.1 Sphere-Tracing Complexity

Canonical statement:

- sphere tracing is not specified as `O(1)`
- the runtime is specified as bounded by `MAX_STEPS` and traversal depth
- the important production property is bounded execution with high practical
  throughput on batched CUDA workloads

Implementation consequence:

- all docs, tests, and benchmarks must use bounded-complexity language
- all kernels must expose explicit `MAX_STEPS`, hit epsilon, and horizon rules

### 5.2 Canonical Deduplication Rule

Canonical statement:

- structural equality is the correctness authority
- hashing is an acceleration filter only
- canonical compiler behavior must use one deterministic hash algorithm with a
  fixed seed and a mandatory structural equality fallback before deduplication

Planned canonical choice:

- deterministic MurmurHash3 with fixed seed `0` for candidate grouping
- structural equality of node payload and child layout before merge

Implementation consequence:

- documentation, compiler code, and tests must use one consistent story
- any old FNV references are removed during migration

### 5.3 Canonical Leaf Payload Rule

Canonical statement:

- leaf payload precision is a design parameter, not a slogan
- any distance payload choice must be documented with total error budget:
  discretization error plus quantization error plus runtime decode error

Planned canonical policy:

- keep current production payload conservative until a lower-precision leaf
  format wins end-to-end benchmarks and passes correctness thresholds
- if fp16 leaf storage is explored, the docs must state centimeter-scale error
  bounds honestly rather than claiming sub-millimeter precision

### 5.4 Canonical Tensor Normalization Rule

Canonical statement:

- direction vectors are valid when `abs(norm - 1.0) <= eps`
- exact floating-point equality is forbidden as a contract rule

Planned canonical policy:

- validate contiguous CUDA `float32` tensors shaped `[batch, rays, 3]`
- enforce normalization tolerance with explicit epsilon in bindings and tests

### 5.5 Canonical Domain Boundary Rule

Canonical statement:

- outside the compiled domain, rays are treated by the environment contract as
  horizon-saturated invalid observations, not as a promise of globally correct
  analytic SDF continuation
- inside-solid queries must preserve negative-distance semantics where the
  runtime supports them, or be explicitly converted into collision semantics at
  the environment boundary with no ambiguity

Implementation consequence:

- runtime docs and environment docs must distinguish field math from published
  observation semantics

### 5.6 Canonical Coordinate Transform Rule

Canonical statement:

- no dataset transform may be described only in prose
- each supported dataset adapter must define an explicit rigid transform matrix,
  handedness convention, and semantic mapping table

Planned canonical policy:

- implement dataset-specific golden tests using known poses and rendered ground
  truth to verify orientation, scale, and forward axis semantics

### 5.7 Canonical Episodic Memory Reward Rule

Canonical statement:

- loop-foraging logic is valid only if its reward math is explicit
- the architecture will not claim that memory "eliminates" failure modes unless
  a precise reward definition is documented and tested

Planned canonical policy:

- define loop similarity metric
- define threshold for loop activation
- define reward or penalty term and weighting relative to environment reward
- define insertion cadence, capacity, and eviction semantics explicitly

### 5.8 Canonical Observer Transport Rule

Canonical statement:

- the production observer surface is ZMQ-based and passive
- browser or web transport, if present, is an observer-side proxy layer only
- no browser transport claim may be used in hot-path reasoning

## 6. Program Structure

The adoption program is organized into eight workstreams that execute in a
controlled sequence.

### 6.1 Workstream A - Architecture And Contract Lock

Scope:

- rewrite docs so imported claims become corrected repo-local contracts
- remove contradictions and undefined terms before large code movement

Deliverables:

- updated `docs/ARCHITECTURE.md`
- updated `docs/CONTRACTS.md`
- updated `docs/SDFDAG_RUNTIME.md`
- updated `docs/COMPILER.md`
- updated `docs/SIMULATION.md`
- updated `docs/AUDITOR.md`
- updated `docs/VERIFICATION.md`
- updated `AGENTS.md` only if a new permanent repository rule is required

Exit criteria:

- one consistent story for tensor contracts, runtime complexity, DAG layout,
  semantic mapping, and observer transport

### 6.2 Workstream B - Compiler Correctness And File-Format Lock

Scope:

- harden `projects/voxel-dag` as a first-class compiler domain
- formalize `.gmdag` invariants and deduplication behavior

Tasks:

1. lock canonical file header, payload, and versioning rules
2. implement canonical hash plus structural equality fallback
3. document and test node ordering, child mask semantics, and leaf payload
4. add artifact-level invariant tests and corruption tests
5. add golden compilation fixtures from representative scenes

Exit criteria:

- `.gmdag` generation is deterministic for fixed inputs
- corrupted files fail fast with actionable errors
- deduplication correctness is proven independently of compression ratio

### 6.3 Workstream C - CUDA Runtime Contract Hardening

Scope:

- harden `projects/torch-sdf` around explicit tensor and kernel rules

Tasks:

1. validate device, dtype, rank, shape, and contiguity at the binding boundary
2. validate normalization tolerance for direction tensors
3. define and expose `MAX_STEPS`, hit epsilon, horizon clamp, and negative-hit
   semantics explicitly
4. prove GIL-release scope and make docs precise about what remains in Python
5. add unit and integration tests for failure traps and edge conditions

Exit criteria:

- all low-level runtime assumptions are explicit, tested, and documented
- edge-case behavior is deterministic and identical across docs and code

### 6.4 Workstream D - Environment Integration Refactor

Scope:

- integrate corrected imported runtime ideas into `projects/environment`
- preserve actor contract while maximizing tensor-native behavior internally

Tasks:

1. unify live asset loading, metadata validation, and runtime preflight
2. keep DAG, ray buffers, and outputs GPU-resident across steps
3. keep observation adaptation vectorized and selective
4. formalize outside-domain and horizon-saturation semantics at the environment
   boundary
5. move all dataset transforms and semantic remaps behind adapters only
6. formalize compiled corpus refresh, promotion, overwrite, and resolution
   replacement behavior
7. document and test coordinate transforms per dataset source

Exit criteria:

- environment service and direct trainer path both consume the same corrected
  runtime contracts
- public `DistanceMatrix` semantics are preserved and precisely documented

### 6.5 Workstream E - Actor And Trainer Integration

Scope:

- adopt imported tensor-native and dataflow ideas without violating the sacred
  actor topology or canonical in-process trainer rule

Tasks:

1. keep the actor cognitive stack unchanged in topology
2. keep canonical training fully in-process with no hot-loop ZMQ dependency
3. remove any remaining avoidable host staging in rollout and learner paths
4. migrate canonical rollout storage toward preallocated batched GPU slabs with
  masked tensor bookkeeping for hidden state, auxiliary state, and episode
  accumulators
5. add explicit attribution timers inside PPO update sub-stages
6. formalize episodic-memory reward math and update cadence
7. collapse episodic-memory query/add work for shared rollout embedding batches
  so the canonical trainer normalizes each batch once before reuse
8. preserve tensor-native observation, action, reward, and rollout seams
9. batch sparse telemetry host extraction for action and done flags so
  selected-actor publication avoids per-actor device synchronization on
  telemetry ticks
10. batch completed-episode host extraction so actor id, episode return, and
  episode length cross the device boundary in one packed transfer before
  sparse episode publication
11. keep running reward accumulation on-device and skip unused scalar host
   mirrors when step telemetry is disabled
12. batch PPO epoch metric materialization so learner loss, KL, entropy, clip,
  and RND summaries cross to Python in one packed transfer at update end
13. expose PPO update sub-stage attribution on the canonical learner path for
  minibatch prep, policy evaluation, backward, gradient clip, optimizer step,
  and RND step timing
14. require sequence-native minibatches on the canonical `seq_len > 0` PPO path
   and remove learner-side flat-to-sequence reshaping fallback
15. skip hidden-state minibatch reconstruction on the canonical PPO path because
  the active `Mamba2TemporalCore` ignores hidden state in both step and
  sequence evaluation
16. normalize advantages once per finalized rollout and reuse the cached
  normalized tensor across canonical PPO minibatch sampling instead of
  re-normalizing per epoch or leaving the batched path unnormalized
17. remove canonical trainer-side hidden-state carry, reset, and PPO bootstrap
  plumbing because the active `Mamba2TemporalCore` ignores hidden state during
  both rollout stepping and sequence evaluation
18. remove canonical rollout-buffer hidden-state allocation, caching, and
  minibatch emission on the batched PPO path because the active
  `Mamba2TemporalCore` does not consume hidden starts
19. remove per-actor `TrajectoryBuffer` wrapper allocation from the canonical
  `MultiTrajectoryBuffer(capacity=...)` path so production rollout storage is a
  real batched slab rather than a dual-surface container
20. gate completed-episode host extraction behind real sparse episode
  publication need so canonical training does not batch-copy done actor
  returns/lengths to the CPU when episode telemetry is disabled or filtered out
21. gate initial and live observation materialization behind real dashboard
  publication need so canonical tensor resets and runtime steps do not build
  `DistanceMatrix` observations when the observation stream is disabled
22. keep only one canonical trainer surface and frame any ablation as diagnostic
   only

Exit criteria:

- actor-side runtime and learner paths are explicit, measurable, and aligned
  with corrected docs

### 6.6 Workstream F - Auditor And Dataset QA Integration

Scope:

- adopt the imported passive-observer and dataset-auditor ideas safely

Tasks:

1. keep dashboard passive and actor-stream-first during training
2. support optional dataset-auditor rendering through the real runtime, not
   geometry export
3. formalize any ZMQ-to-web proxy as observer-only infrastructure
4. harden `WAITING` and partial-system modes
5. cap ingestion and rendering cost to protect UI responsiveness

Exit criteria:

- observer tooling remains fully non-blocking and operationally independent

### 6.7 Workstream G - Verification And Qualification

Scope:

- turn the corrected architecture into a complete proof surface

Tasks:

1. unit tests for tensor contracts, math utilities, and semantic transforms
2. compiler invariant tests and corruption tests
3. runtime seam tests for CUDA boundary failures and edge behavior
4. live corpus tests against promoted `.gmdag` fixtures
5. long-run allocation-stability tests
6. trainer-seam tests for tensor-native stepping and publication selectivity
7. end-to-end stack tests for training, resume, dashboard attach, and replay

Exit criteria:

- each architecture claim has a corresponding proof layer
- correctness failures are attributable to a single domain quickly

### 6.8 Workstream H - Performance Qualification

Scope:

- prove that the adoption is not only correct but performance-safe on the real
  trainer

Tasks:

1. maintain environment-layer `bench-sdfdag` qualification
2. add compiler-side artifact size and load-time tracking
3. add runtime microbenchmarks for cast-rays and observation adaptation
4. add trainer attribution for rollout, memory, host extraction, PPO sub-stages,
   and telemetry publication
5. define canonical benchmark profiles and artifact naming rules
6. run repeat sets for any major runtime-format or trainer-hot-path change

Exit criteria:

- no architecture promotion without end-to-end trainer proof
- attribution is rich enough to explain regressions, not just detect them

## 7. Implementation Phases

### Phase 0 - Documentation And Decision Lock

Objective:

- convert imported ideas and review findings into corrected canonical docs

Required outputs:

- corrected complexity language
- corrected precision language
- corrected hash and dedup language
- corrected observer transport language
- explicit reward and boundary semantics

Validation:

- doc cross-check pass across all touched files
- no unresolved contradictions between docs

### Phase 1 - Low-Level Contract Hardening

Objective:

- lock compiler and runtime contracts before broad integration changes

Required outputs:

- `.gmdag` invariant suite
- tensor-boundary validation suite
- explicit edge-behavior tests
- deterministic compile fixtures

Validation:

- focused `ruff`
- focused `mypy --strict`
- compiler and runtime unit suites

### Phase 2 - Environment Refactor And Corpus Pipeline

Objective:

- integrate corrected imported environment ideas behind one stable boundary

Required outputs:

- hardened corpus refresh and promotion flow
- corrected dataset adapters
- corrected horizon and out-of-bounds semantics
- stable tensor-native environment seam

Validation:

- environment unit tests
- live compiled-corpus tests
- `check-sdfdag`
- `bench-sdfdag`

### Phase 3 - Actor And Trainer Alignment

Objective:

- align actor and trainer with the corrected runtime without changing sacred
  actor topology

Required outputs:

- explicit PPO sub-stage attribution
- explicit episodic-memory reward math
- end-to-end tensor-native learner seams
- no alternate production trainer path

Validation:

- actor unit tests
- trainer state tests
- targeted end-to-end training smoke runs

### Phase 4 - Auditor And Dataset QA Completion

Objective:

- complete passive observability and dataset-auditor surfaces safely

Required outputs:

- passive training-safe dashboard behavior
- runtime-backed dataset auditor
- clarified proxy transport if browser clients are supported

Validation:

- auditor unit and smoke tests
- observer attach during canonical training without regressions

### Phase 5 - End-To-End Qualification

Objective:

- prove the integrated system is complete and robust

Required outputs:

- canonical train-run qualification
- checkpoint resume qualification
- dashboard attach qualification
- corpus refresh to training qualification
- replay qualification

Validation:

- end-to-end scripted scenarios
- failure injection for missing assets, invalid tensors, and bad ports

### Phase 6 - Performance Promotion Review

Objective:

- decide whether any benchmark-gated runtime experiments should be promoted

Required outputs:

- repeat-set benchmark comparisons against current canonical trainer
- decision record for any proposed payload/layout/overlap change

Validation:

- promotion only if correctness stays green and the real trainer wins

## 8. Validation Matrix

### 8.1 Documentation Validation

Checks:

- no contradictory claims across docs
- every complexity claim uses correct bounded or approximate language
- every contract uses explicit dimensions, dtype, and tolerances

### 8.2 Compiler Validation

Checks:

- deterministic compile output for golden fixtures
- hash collision defense tests
- structural equality fallback tests
- corrupted header and payload rejection tests
- child-mask and node-ordering invariant tests

### 8.3 Runtime Validation

Checks:

- CUDA device mismatch traps
- dtype mismatch traps
- rank and shape traps
- non-contiguous tensor traps
- normalization tolerance traps
- inside-solid, out-of-bounds, and horizon-saturation behavior tests
- `MAX_STEPS` termination tests

### 8.4 Dataset And Adapter Validation

Checks:

- coordinate transform golden fixtures
- semantic remap fixtures
- scale and orientation sanity tests
- promoted manifest correctness tests

### 8.5 Environment Validation

Checks:

- reset and step against real compiled assets
- public `DistanceMatrix` shape and semantic correctness
- selective materialization correctness
- reward and truncation contract correctness

### 8.6 Actor And Trainer Validation

Checks:

- sacred actor interface preservation
- tensor-native minibatch and sequence path tests
- episodic-memory reward math tests
- PPO learner attribution stability
- canonical trainer hot-path no-stall tests

### 8.7 Auditor Validation

Checks:

- passive attach with missing producers
- actor-only training mode attach
- capped ingestion and frame-drop safety
- dataset auditor runtime render correctness

### 8.8 End-To-End Validation

Checks:

- corpus refresh to successful training
- successful continuous training run
- checkpoint save and resume
- dashboard attach during training
- recorder and replay surfaces
- fail-fast startup and restart verification

## 9. Performance Qualification Plan

### 9.1 Canonical Benchmark Profiles

Every major adoption step must be measured on stable named profiles.

Required profiles:

1. environment microbench profile
2. runtime cast-rays microbench profile
3. direct trainer bounded profile
4. direct trainer repeat-set profile
5. optional long-run stability profile

### 9.2 Required Metrics

Environment metrics:

- load time
- step latency
- ray throughput
- observation adaptation time

Trainer metrics:

- `sps`
- `env_ms`
- `mem_ms`
- `trans_ms`
- `host_extract_ms`
- `telemetry_publish_ms`
- `ppo_update_ms`
- PPO sub-stage attribution for minibatch prep, policy evaluation, backward,
  gradient clip, optimizer step, and RND step

### 9.3 Performance Gates

A change is promotable only if:

1. quality gates are green
2. contract behavior stays correct
3. the change improves or materially advances end-to-end trainer throughput on
   canonical profiles
4. any regression is explained and accepted explicitly, not silently carried
5. no observer or compatibility cost is pushed into the hot path

### 9.4 Artifact Discipline

Every benchmark set must record:

- command profile
- scene or corpus scope
- actor count
- resolution
- run duration or step bound
- artifact log path
- summary path
- source of final optimizer timing
- comparison baseline

## 10. End-To-End Completion Checklist

The program is complete only when all items below are true.

1. docs are internally consistent and mathematically corrected
2. `.gmdag` compile and load contracts are formalized and tested
3. CUDA boundary contracts are explicit and enforced
4. environment adaptation is tensor-native internally and contract-correct
   externally
5. actor and trainer remain sacred in topology and canonical in topology
6. episodic-memory reward and boundary semantics are explicit and tested
7. passive observer tooling is complete and non-blocking
8. dataset-auditor path works through the real runtime
9. end-to-end training, resume, and replay flows are operational
10. performance qualification is recorded and reproducible

## 11. Risk Register And Controls

### 11.1 Major Risks

Risk:

- large documentation-driven refactor creates temporary contract drift

Control:

- Phase 0 completes before major code movement

Risk:

- runtime-format work creates correctness regressions hidden by benchmark noise

Control:

- correctness gates must pass before performance promotion

Risk:

- imported claims are adopted too literally and reintroduce invalid assumptions

Control:

- this plan's corrected canonical answers override imported prose wherever they
  conflict

Risk:

- observer or web tooling leaks back into hot-path reasoning

Control:

- observer transport remains explicitly non-canonical for throughput reasoning

Risk:

- parallel research paths widen production architecture surface

Control:

- only one production trainer and one production runtime remain canonical at any
  time

## 12. Execution Order

The implementation order for this refactor is:

1. Phase 0 - documentation and decision lock
2. Phase 1 - compiler and runtime contract hardening
3. Phase 2 - environment and corpus pipeline refactor
4. Phase 3 - actor and trainer alignment
5. Phase 4 - auditor and dataset QA completion
6. Phase 5 - end-to-end qualification
7. Phase 6 - performance promotion review

## 13. Immediate Next Actions

The first implementation sprint under this new plan is:

1. rewrite the affected docs to replace imported overclaims with corrected
   canonical contracts
2. formalize `.gmdag` invariants and canonical deduplication rules
3. formalize CUDA tensor boundary validation including normalization tolerance
4. add the first missing correctness tests for hash fallback, out-of-bounds
   semantics, and dataset transform fixtures
5. add PPO sub-stage attribution so future trainer work is evidence-rich

## 14. Decision Record

This plan makes the following explicit architectural decisions for the adoption:

1. imported architecture ideas are adopted selectively, not wholesale
2. corrected mathematical contracts override imported prose where they differ
3. the imported runtime vision is integrated under Navi's in-process canonical
   trainer rule
4. correctness and validation are first-class workstreams, not cleanup after the
   refactor
5. benchmark-gated runtime experiments remain later-phase work until the core
   adoption is complete and proven
