# PARALLEL.md - Canonical Multi-Actor Execution And Research Boundaries

## 1. Purpose

This document explains how Navi uses multiple actors today and how that differs
from broader multi-agent or swarm ideas found in the imported material.

The short version is:

- Navi absolutely depends on batched multi-actor execution
- but it does so for distributed experience accumulation under one global
  learner, not for production swarm cognition

## 2. Physics Of Shared Compiled Worlds

The compiled `sdfdag` runtime makes same-scene multi-actor execution efficient:

- one `.gmdag` asset can be shared across many actors
- one DAG tensor can stay resident on CUDA
- batched ray queries amortize launch and memory overhead across actors
- actors are mathematically independent unless explicit interaction terms are
  added above the runtime

This is one of the strongest performance arguments for the current architecture.

## 3. Canonical Production Topology

The active production topology is Paradigm A from the imported notes,
reframed in current repo language:

- many actors step in parallel
- one global policy generates actions
- one global learner updates shared weights
- per-actor trajectories remain logically distinct for PPO and BPTT correctness
- episodic memory remains scoped for robust on-policy training behavior

This is not merely a convenient default. It is the architecture that best fits:

- current PPO training
- current tensor-native rollout structure
- current benchmark and attribution surfaces
- the requirement to keep one production trainer only

## 4. Current Multi-Actor Dataflow

At a high level the canonical trainer does the following:

1. keep one batched observation tensor for all active actors
2. run one batched policy forward pass
3. step the environment backend for all actors together
4. append one rollout step for all actors
5. run PPO on the filled rollout buffer

This exploits the compiled runtime without reopening service transport in the
hot loop.

## 5. Why Swarm Cognition Is Not Canonical

The imported swarm material was conceptually interesting, but it is not the
current production direction.

Reasons:

- shared global episodic memory would change the optimization problem, not just
  the implementation details
- cross-actor reward sharing would complicate credit assignment and PPO stability
- truly shared memory would create new hot-path synchronization costs
- the current repo bottlenecks are already in rollout dataflow and PPO update
  cost, so adding swarm synchronization now would move in the wrong direction

## 6. Research-Only Extension Path

Swarm-style ideas remain valid research directions only after the canonical
single-policy parallel trainer is fully optimized.

Research-only candidates include:

- global episodic memory across actors
- cooperative reward propagation
- cross-actor latent communication
- proximity-conditioned coordination layers

If explored, they should be framed as explicit research branches, not as silent
extensions of the canonical trainer.

## 7. Parallelism Rules That Remain Canonical

Regardless of research direction, the following stay true:

- batch the environment path wherever possible
- keep actor trajectories isolated for current PPO correctness
- keep passive observability decoupled from rollout cadence
- do not duplicate DAG memory per actor when shared residency is available
- prefer tensor-native batched storage over per-actor Python transition objects

## 8. Relationship To Service Mode

Service mode may still run multiple actors, but it is not the canonical
throughput path. It exists for diagnostics, integration, teleop, and replay.
The production parallel architecture is the direct in-process trainer.

## 9. Related Docs

- `docs/ARCHITECTURE.md`
- `docs/DATAFLOW.md`
- `docs/ACTOR.md`
- `docs/PERFORMANCE.md`
