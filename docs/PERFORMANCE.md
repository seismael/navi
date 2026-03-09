# PERFORMANCE.md — Throughput And Runtime Gates

This document defines the performance-first acceptance criteria for Navi.
Correctness is mandatory, but performance decides which runtime path remains
canonical for training.

## 1. Priority Order

1. Rollout throughput
2. No-stall execution
3. GPU residency
4. Contract preservation

## 2. Canonical Runtime Path

The production training path is:

1. `projects/voxel-dag` compiles source meshes to `.gmdag`.
2. `projects/torch-sdf` executes batched CUDA sphere tracing against the DAG.
3. `projects/environment` adapts outputs back into canonical `(1, Az, El)` `DistanceMatrix` tensors.
4. `projects/actor` consumes the same observation contract.

Canonical training keeps one runtime only: direct in-process backend stepping over `sdfdag`.

## 3. Runtime Rules

- canonical high-performance runtime must not silently fall back to CPU
- batched stepping is mandatory for the SDF/DAG path
- per-step allocation churn in the hot path is forbidden when reusable buffers are practical
- observation adaptation must be vectorized and preserve actor contracts
- actor-side rollout code must be optimized once environment stepping stops dominating SPS
- attribution tooling must stay on the canonical trainer surface, not create alternate production modes

## 4. Benchmark Gates

- standard fleet: `4` actors
- acceptance floor: `>= 60 SPS` for canonical training runtime
- environment changes must preserve or improve end-to-end training throughput
- canonical runtime must fail fast when CUDA or compiled dependencies are unavailable

## 5. Active Optimization Direction

Current work should focus on:

- removing remaining device -> host -> device bounces in the actor rollout path
- minimizing Python object construction inside per-step training loops
- reducing optimizer-side device copies and allocator churn on CUDA
- keeping telemetry coarse and non-blocking

No runtime path becomes canonical because it is easy to explain. It stays canonical only while it is contract-correct and measurably faster in end-to-end training.
