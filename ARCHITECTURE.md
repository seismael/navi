# ARCHITECTURE.md — Ghost-Matrix (Canonical)

This document is the active architecture source of truth for Navi.

## 1) System Objective

Navi is optimized for high-throughput reinforcement learning by separating:
- Simulation/physics execution,
- Observation transformation into distance matrices,
- Policy inference and learning,
- Asynchronous visualization/replay.

The runtime never depends on rendering for training progress.

## 2) Layer Model

Ghost-Matrix uses three logical layers connected by message contracts.

1. Simulation Layer (Engine Room)
   - Headless step execution.
   - Produces DistanceMatrix v2 observations.
   - Consumes Action v2 commands.

2. Brain Layer (Policy)
   - Consumes DistanceMatrix v2.
   - Produces Action v2.
   - In step mode, exchanges StepRequest/StepResult with Simulation Layer.

3. Gallery Layer (Visualization & Replay)
   - Passive subscriber and recorder.
   - Replays state logs asynchronously.
   - Does not gate simulation throughput.

## 3) Canonical Contracts

The only canonical wire models are:
- `RobotPose`
- `DistanceMatrix`
- `Action`
- `StepRequest`
- `StepResult`
- `TelemetryEvent`

Legacy contracts are removed from the target architecture.

### 3.1 DistanceMatrix v2

`DistanceMatrix` fields:
- `episode_id: int`
- `env_ids: NDArray[int32]`
- `matrix_shape: tuple[int, int]` (azimuth, elevation)
- `depth: NDArray[float32]` normalized to [0, 1]
- `delta_depth: NDArray[float32]`
- `semantic: NDArray[int32]`
- `valid_mask: NDArray[bool]`
- `robot_pose: RobotPose`
- `step_id: int`
- `timestamp: float`

## 4) Transport Topology (ZMQ)

PUB/SUB topics:
- `distance_matrix_v2`
- `action_v2`
- `telemetry_event_v2`

REQ/REP topics:
- `step_request_v2`
- `step_result_v2`

Flow:
- Simulation PUB `distance_matrix_v2` -> Brain SUB
- Brain PUB `action_v2` -> Simulation SUB (async mode)
- Brain REQ `step_request_v2` <-> Simulation REP `step_result_v2` (step mode)
- Gallery SUB records all v2 streams

## 5) Runtime Principles

- Headless-first: no rendering in the critical training loop.
- Matrix-first: observations are geometric distance matrices, not image frames.
- Non-blocking telemetry: logs are asynchronous and float-only.
- Stateless transport: all inter-process integration uses message contracts.

## 6) Current Refactor Target

Refactor completion requires:
- All services using v2 contracts only.
- No legacy topics in runtime paths.
- No deprecated compatibility branches.
- Updated tests, docs, and launch scripts aligned to v2 flow.

## 7) Performance Orientation

Primary throughput metrics:
- Steps per second (aggregate across environments)
- Policy inference latency per batch
- Logging overhead budget relative to step loop

Rendering is diagnostic only and must not reduce training throughput.
