# VERIFICATION.md - Runtime And Benchmark Validation

## 1. Purpose

This document organizes Navi verification around the boundaries that actually
matter for the canonical runtime.

The imported docs were correct that a serious system needs more than generic
unit tests. Navi now documents verification in layers that match the active
codebase rather than assumed external infrastructure.

## 2. Mandatory Repository Gates

All changes still pass through the repository quality gates:

- `ruff`
- `mypy --strict`
- `pytest`

For canonical runtime work, that is the minimum, not the whole proof.

## 3. Verification Layers

### 3.1 Compiler And Artifact Invariants

Compiler validation should cover:

- successful `.gmdag` generation from supported source assets
- header validity and payload-size consistency
- reproducible load behavior in the environment integration layer
- resolution and bounding-box metadata correctness

Some imported ideas such as analytical sphere proofs and deep compiler
invariant tests remain good ideas, but they should be implemented against the
actual in-repo compiler rather than copied as abstract requirements.

### 3.2 Runtime Seam Validation

These tests assert the behavior of the current CUDA and environment seam.

Important covered examples already present in the repo include:

- horizon propagation from environment config into batched ray casting
- clamp and validity behavior for rays beyond the fixed horizon
- tensor-step preference tests when runtime tensor seams are available

### 3.3 Live Corpus Validation

These tests assert that promoted compiled assets are actually usable on the real
runtime.

Current live checks include:

- compiled corpus manifests point at live compiled assets
- `SdfDagBackend` can reset and step against a real `.gmdag`
- saturated rays from the live backend obey the fixed-horizon depth and validity contract

### 3.4 CUDA Boundary Validation

The imported project correctly emphasized that the Python-to-CUDA seam must be
validated aggressively.

For Navi this means tests should continue to cover:

- device mismatch rejection
- dimensionality rejection
- contiguity assumptions
- CUDA-only fail-fast behavior when the backend is unavailable

### 3.5 Trainer-Seam Validation

Because the canonical training runtime now depends on tensor-native environment
surfaces, tests should also protect:

- observation-tensor preference over object rebuilding when available
- action-tensor stepping preference when available
- safe publication of only selected actor observations for passive viewers
- perf-only telemetry configurations that do not crash the trainer

### 3.6 Benchmark Proof

Performance claims are valid only when measured on the current canonical path.
Proof should distinguish between:

- environment-layer attribution via `bench-sdfdag`
- service and integration correctness checks
- full trainer impact on the production rollout loop

## 4. Current Implemented Proof Surfaces

Current repo-local proof surfaces include:

- runtime readiness checks through `navi-environment check-sdfdag`
- direct environment throughput attribution through `navi-environment bench-sdfdag`
- unit tests for fixed-horizon semantics and tensor-step preference
- live integration tests against promoted compiled assets

## 5. Current Gaps Worth Closing

The imported docs highlighted useful missing checks. The following remain strong
next additions:

1. long-run allocation-stability tests around repeated CUDA stepping
2. explicit dtype and contiguity traps at the extension boundary
3. deeper compiler invariant tests that reflect the actual in-repo file format
4. passive dataset-auditor validation if that surface is implemented
5. longer trainer attribution regression tests for host extraction and telemetry cost

## 6. Benchmark Language Policy

Documentation must not claim throughput numbers that are not currently backed by
Navi measurements.

Use these rules instead:

- state current measured baselines as baselines
- state acceptance floors as floors
- state candidate ideas as candidates
- avoid importing external headline numbers as if they were repo facts

## 7. Failure Interpretation Rule

When a benchmark or training run regresses, interpret it using the current
layered model:

- if `check-sdfdag` fails, the issue is in compiler/runtime readiness
- if `bench-sdfdag` regresses but trainer SPS does not, the issue may be local to
  environment attribution and not production-critical
- if trainer SPS regresses while environment timing stays flat, the issue is
  likely actor-side dataflow or PPO update cost
- if passive observers trigger regressions, the failure is architectural because
  observability must remain droppable

## 8. Related Docs

- `docs/ARCHITECTURE.md`
- `docs/PERFORMANCE.md`
- `docs/SDFDAG_RUNTIME.md`
- `docs/SIMULATION.md`
- `docs/DATAFLOW.md`
- `PLAN.md`
