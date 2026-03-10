# ARCHITECTURE.md - Canonical Ghost-Matrix Architecture

## 1. Executive Summary

Navi is a throughput-first geometric navigation stack built around one
production mathematical runtime:

- source scenes are staged and compiled into `.gmdag` assets
- `projects/torch-sdf` executes batched CUDA sphere tracing against those assets
- `projects/environment` owns observation shaping, reward seams, corpus
  orchestration, and service-mode exposure
- `projects/actor` owns the sacred policy, rollout loop, PPO update path, and
  coarse training telemetry
- `projects/auditor` remains a passive observer layer for dashboards, replay,
  and recording

The repository no longer treats multiple training architectures as equal peers.
There is one canonical production trainer and one canonical compiled-scene
runtime. That decision is not stylistic. It follows directly from the current
codebase and from the current bottleneck map: once world queries are fast and
batched, the dominant risks shift upward into rollout dataflow, host staging,
and PPO update cost.

## 2. System Objective

The system objective is not generic rendering. It is high-throughput training of
an agent that navigates from spherical distance observations while preserving a
stable external actor contract.

The active architectural constraints are:

- preserve `DistanceMatrix` as the external wire and diagnostic contract
- keep the actor cognitive pipeline unchanged
- keep canonical rollout on the compiled `sdfdag` path
- fail fast when CUDA or the compiled runtime is unavailable
- avoid reopening legacy mesh, habitat, voxel, or distributed-hot-loop paths as
  production alternatives

## 3. System Taxonomy

| Domain | Project | Current Role |
| --- | --- | --- |
| Contracts | `projects/contracts` | canonical wire models, serialization, logging |
| Compiler | `projects/voxel-dag` | offline mesh-to-`.gmdag` compilation |
| Runtime | `projects/torch-sdf` | CUDA sphere tracing over compact DAG memory |
| Environment | `projects/environment` | corpus prep, stepping, reward seams, contract adaptation |
| Actor | `projects/actor` | sacred policy, tensor-native rollout, PPO, intrinsic signals |
| Auditor | `projects/auditor` | passive dashboard, recorder, rewinder |

This split is preserved because it keeps the production hot path narrow while
allowing service-mode and diagnostic tooling to remain decoupled.

## 4. Canonical Runtime Boundary

The production data path is:

```text
transient source staging
  -> compiler normalization and `.gmdag` generation
  -> compiled corpus promotion
  -> CUDA DAG loading and batched ray execution
  -> environment reward and observation adaptation
  -> actor rollout and PPO
  -> passive telemetry, record, and replay surfaces
```

The canonical training integration point is at the CLI and orchestration
boundary. The actor is allowed to instantiate or drive the environment backend
only inside the single production training entrypoint. That does not widen the
service import graph and does not create a new cross-package dependency surface.

## 5. Responsibilities By Domain

### 5.1 Contracts Domain

The contracts domain defines the only stable wire models allowed between
services:

- `DistanceMatrix`
- `Action`
- `StepRequest`
- `StepResult`
- `BatchStepRequest`
- `BatchStepResult`
- `TelemetryEvent`

The contracts domain also owns canonical logging initialization. This matters
architecturally because performance work can change internal seams without
reopening public wire compatibility.

### 5.2 Compiler Domain

The compiler domain is responsible for converting source geometry into a compact
runtime artifact that the CUDA kernel can consume directly. The active repo has
both:

- a native C++ compiler executable used by environment orchestration
- a Python verification/compiler surface used by tests and local validation

The environment project owns orchestration of compiler invocation, but the
compiler project remains the sovereign implementation of `.gmdag` generation.

### 5.3 Runtime Domain

The runtime domain exposes the lowest-level mathematical execution path in the
repository:

- load compact DAG payloads
- keep them contiguous and CUDA-resident
- accept batched origin and direction tensors
- write batched distance and semantic outputs into preallocated tensors
- release the Python GIL during long kernel execution

This domain does not own training logic, reward logic, or dashboard concerns.

### 5.4 Environment Domain

The environment domain is the spatial truth layer of Navi. It owns:

- scene discovery and corpus preparation
- `.gmdag` asset validation and loading
- batched actor stepping and kinematic integration
- environment-side shaping signals such as clearance, starvation, and proximity
- contract-preserving conversion from runtime tensors to `DistanceMatrix`
- service-mode PUB/REP exposure

In the canonical trainer, the environment may expose tensor-native seams so the
actor can avoid rebuilding Python wire objects every step.

### 5.5 Actor Domain

The actor domain owns:

- the fixed sacred cognitive stack
- batched inference and action production
- rollout buffer management
- PPO and RND optimization
- episodic memory query and insertion
- coarse training-time telemetry publication

The current architectural reality is that actor-side dataflow and PPO update
cost are now at least as important as raw environment query speed.

### 5.6 Auditor Domain

The auditor domain is passive by policy. It may:

- subscribe to actor and environment streams
- display a selected actor view
- record and replay telemetry and matrices
- render diagnostics from received contracts

It may not become a mandatory dependency of the training path.

## 6. Mathematical Foundations

### 6.1 Signed-Distance Execution

The active environment runtime is built around signed-distance queries against a
compiled volumetric representation. For a ray `p + t d`, the kernel repeatedly
samples local free-space distance and advances by that clearance until it either
contacts geometry or exceeds the configured environment horizon.

The production contract is:

- one run-wide horizon from `EnvironmentConfig.max_distance`
- depth normalization tied to that same horizon
- horizon-saturated rays marked invalid
- no dynamic per-step trace radius in the canonical runtime

### 6.2 Spherical Observation Model

The actor's observation boundary remains spherical and stable.

- canonical matrix shape is `(Az, El)` with default `(256, 48)`
- external arrays remain `(1, Az, El)` per environment
- the internal trainer seam may use `(B, 3, Az, El)` CUDA tensors
- azimuth bin `0` is forward along local `-Z`

This is the key architectural bridge between old and new runtime work: the
runtime beneath the contract may change, but the actor's observation semantics
must not drift.

### 6.3 Persistence-First Collision Policy

Canonical training is not reset-first. It is persistence-first.

- collisions do not end training episodes
- invalid motion is reverted and penalized
- actors remain in scene and are encouraged to recover
- scene rotation is coarse and throughput-aware, not trigger-happy

This is one of the most important behavioral architecture choices in the repo.
It keeps local recovery learning inside the actual scene geometry instead of
turning collisions into cheap reset churn.

## 7. Adopted Imported Ideas

The imported documentation was valuable in several areas and those ideas are now
explicitly part of the Navi docs.

### 7.1 Strict Low-Level Tensor Contracts

The Python-to-CUDA seam is now documented as a systems contract, not an informal
helper surface.

Adopted principles:

- explicit device, dtype, rank, shape, and contiguity requirements
- preallocated output tensors on the hot path
- GIL release during long CUDA execution
- explicit separation of public wire contracts from internal tensor-native seams

### 7.2 Offline DataOps, Narrow Runtime

The imported OmniSense material correctly emphasized that external data mess does
not belong in the PPO hot loop.

Adopted principles:

- coordinate normalization belongs in corpus prep or adapters
- semantic translation belongs in corpus prep or adapters
- compiled `.gmdag` assets are the only canonical runtime input
- sensor adaptation is a boundary concern, not a justification for alternate
  production runtimes

### 7.3 Passive Observer Discipline

The imported dashboard material correctly emphasized a non-blocking observer
model.

Adopted principles:

- dashboards are droppable and passive
- training mode detection should use low-volume actor telemetry
- human-facing refresh rates must never regulate rollout cadence
- dataset QA should prefer rendering through the real mathematical runtime over
  geometry-export shims

### 7.4 Verification Structure

The imported test and verification docs were stronger than the previous local
docs in how they separated:

- mathematical and file-format invariants
- low-level runtime seam validation
- live corpus checks
- benchmark proof

That structure now exists in Navi documentation as well.

## 8. Rejected Or Demoted Imported Ideas

The imported project also contained ideas that do not match the active repo.
They are preserved only as rejected or benchmark-gated concepts.

### 8.1 Distributed REQ/REP Training As Canonical

Rejected as production architecture.

Why:

- current canonical trainer is direct in-process `sdfdag`
- reintroducing service transport into the hot loop would add avoidable
  serialization and staging cost
- the main repo bottlenecks are now above that transport seam anyway

### 8.2 Headline `>10,000 SPS` Claims

Rejected as current repo documentation language.

Why:

- Navi documentation must state measured repo baselines and acceptance floors
- headline numbers without repo-local proof are not architecture facts

### 8.3 Premature Format Lock-In

Demoted to benchmark-gated candidate status:

- truncation metadata policies
- alternate distance payload precision policies
- Morton or other layout redesigns
- overlap strategies that complicate the single canonical trainer

## 9. Current Bottleneck Map

The architecture should now be read through the current performance bottlenecks,
not through older assumptions.

Environment acceleration has already shifted the dominant risks upward. The most
important current pressure points are:

- remaining device-to-host extraction inside the actor rollout loop
- optional but still costly Python `DistanceMatrix` and `Action` materialization
- episodic-memory and reward-shaping attribution cost on the trainer surface
- PPO minibatch assembly, optimizer coordination, and update wall time

The architectural rule that follows is simple:

`If tensors are already on CUDA, do not bounce them through Python wire objects unless a passive observer truly needs them.`

## 10. Canonical Runtime Modes

### 10.1 Production Training Mode

Production training means:

- one in-process trainer
- one batched `sdfdag` backend
- full discovered corpus by default
- continuous training until stopped unless the user requests a bound

### 10.2 Service Mode

Service mode remains valid and supported for:

- environment serving over REP/PUB
- actor serving over SUB/PUB
- manual diagnostics and teleoperation
- record/replay tooling

Service mode is a valid architectural surface, but it is not the canonical
throughput path.

### 10.3 Passive Observability Mode

The dashboard and recorder must tolerate missing producers and idle sockets.
Observer tooling should display `WAITING` or equivalent non-crashing states when
upstream streams are absent.

## 11. Launch Surfaces

| Scope | Command | Notes |
| --- | --- | --- |
| Full stack training | `./scripts/run-ghost-stack.ps1 -Train` | canonical continuous training surface |
| Full stack inference | `./scripts/run-ghost-stack.ps1 -GmDagFile <asset>` | service-mode stack |
| Actor trainer | `uv run --project projects/actor navi-actor train` | full-corpus default |
| Environment service | `uv run --project projects/environment environment` | canonical environment shortcut |
| Dashboard | `uv run --project projects/auditor dashboard` | passive viewer |

## 12. Document Map

- `docs/COMPILER.md`: compiler stages, `.gmdag` artifact generation, and CLI orchestration
- `docs/SDFDAG_RUNTIME.md`: low-level DAG layout, tensor contracts, and CUDA boundary details
- `docs/SIMULATION.md`: environment runtime, reward seams, corpus rules, and service surfaces
- `docs/DATAFLOW.md`: production training and service-mode execution sequences
- `docs/PARALLEL.md`: canonical multi-actor topology and research boundaries
- `docs/ACTOR.md`: sacred policy stack and actor-side runtime details
- `docs/PERFORMANCE.md`: benchmark gates, bottleneck interpretation, and hot-path rules
- `docs/AUDITOR.md`: dashboard, recorder, rewinder, and passive observer policy
- `docs/CONTRACTS.md`: public wire models and internal tensor seam notes
- `docs/VERIFICATION.md`: test layers and benchmark proof requirements
