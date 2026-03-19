# DATAFLOW.md - Canonical Execution Sequences

## 1. Purpose

This document captures the execution lifecycle of Navi in the forms that matter
most today:

- canonical in-process training
- service-mode stepping
- passive observer ingestion

The imported data-flow document assumed a distributed hot loop. Navi now keeps
the same level of detail while grounding it in the live canonical trainer.

## 2. Canonical In-Process Training Loop

### 2.1 Sequence

```text
trainer observation tensor
  -> batched policy forward pass
  -> batched action tensor
  -> direct backend step
  -> CUDA ray execution
  -> environment reward and observation adaptation
  -> rollout append
  -> periodic PPO update
  -> optional coarse dashboard heartbeat and telemetry
```

### 2.2 Step Order

1. the actor holds a batched observation tensor on CUDA
2. the policy produces batched action outputs
3. the trainer prefers tensor-native action stepping when the runtime supports it
4. the environment backend performs batched kinematic integration and ray
   execution
5. the backend returns batched observation tensors plus step results
6. the trainer computes shaping, memory, transition, and telemetry work
7. once rollout capacity is reached, PPO update runs inline on the canonical path
8. the rollout policy is resynchronized to learner weights

## 3. Canonical Service-Mode Step Sequence

Service mode remains valid but is not the production throughput path.

```text
actor service
  -> BatchStepRequest over REP
  -> environment service step
  -> `torch_sdf.cast_rays()`
  -> BatchStepResult reply
  -> optional PUB telemetry and matrices
```

Service mode is important for:

- manual stepping
- integration tests
- external consumers of the wire contracts
- diagnostics that intentionally exercise the real transport boundary

## 4. Passive Observer Sequence

```text
actor and environment PUB streams
  -> auditor StreamEngine poll
  -> selected actor state update
  -> dashboard render at capped UI cadence
  -> optional record or replay surfaces
```

The critical rule is that observer consumption is droppable. If the dashboard is
behind, it must drop intermediate frames rather than backpressuring producers.
Observer-side reshaping of the published sphere is allowed; changing the core
observation contract to satisfy a view is not.

## 5. Synchronization Rules

### 5.1 What Must Synchronize

- policy forward pass must consume a coherent observation batch
- backend step must finish before its outputs are read
- PPO update must consume a complete rollout buffer
- publish-time materialization must only occur after observation state is valid

### 5.2 What Must Not Become Global Barriers

- dashboard rendering
- recorder disk I/O
- observer frame rate
- optional telemetry consumers

## 6. Current Concurrency Guidance

The imported project proposed double buffering and explicit multi-stream rules.
Those remain useful ideas, but in Navi they are currently classified as follows:

- useful for reasoning about future overlap work
- not part of the current production contract
- benchmark-gated if they complicate the single canonical trainer

The current production flow is intentionally simpler: direct trainer stepping,
coarse observer heartbeat, and one production update loop.

## 7. Hot-Path No-Go List

The following are architectural regressions on the canonical path:

- introducing new ZMQ or serialization hops into direct trainer stepping
- rebuilding Python wire objects for every actor every step when not needed for
  passive publication
- adding UI or observer synchronization barriers to rollout progress
- multiplying host extraction points when one coarse batched extraction would do
- making observation shape, axis ordering, normalization, or step semantics conditional on dashboard or replay requirements
- embedding view-specific crop, half-sphere, or palette logic into environment or actor code

## 8. Observability Hooks That Remain Valid

The current trainer exposes enough observability to support bottleneck
attribution without becoming a second runtime:

- selective actor publication via `publish_actor_ids`
- dashboard heartbeat publication during optimizer windows
- host extraction timing
- telemetry publication timing
- coarse environment perf snapshots

## 9. Related Docs

- `docs/ARCHITECTURE.md`
- `docs/SIMULATION.md`
- `docs/ACTOR.md`
- `docs/PERFORMANCE.md`
- `docs/AUDITOR.md`
