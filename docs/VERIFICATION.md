# VERIFICATION.md — Runtime And Benchmark Validation

## 1. Purpose

This document organizes Navi verification around the boundaries that actually
matter for the canonical runtime.

The imported docs were right to separate mathematical correctness, low-level
memory contracts, and performance validation. Navi now keeps that structure,
but ties it to the current codebase rather than assumed external tooling.

## 2. Mandatory Repository Gates

All changes still pass through the repository quality gates:

- `ruff`
- `mypy --strict`
- `pytest`

For canonical runtime work, that is the minimum, not the whole story.

## 3. Verification Layers

### 3.1 Contract Tests

These assert the behavior of the current runtime seam.

Examples already present in the repo include:

- horizon propagation from environment config into batched ray casting
- clamp and validity behavior for rays beyond the fixed horizon
- tensor-step preference tests on the actor trainer when runtime tensor seams are available

### 3.2 Live Corpus Validation

These assert that promoted compiled assets are actually usable on the real
runtime.

Current live checks include:

- compiled corpus manifests point at live compiled assets
- compiled corpus manifest metadata (`scene_count`, `gmdag_root`, `requested_resolution`, `compiled_resolutions`) matches the live promoted assets
- `SdfDagBackend` can reset and step against a real `.gmdag`
- saturated rays from the live backend obey the fixed-horizon depth and validity contract
- `check-sdfdag --json` produces a parseable runtime-plus-corpus summary suitable for qualification artifacts and scripted preflight checks
- `bench-sdfdag --json` produces a parseable benchmark summary suitable for live promoted-corpus throughput smoke checks
- `navi-auditor dataset-audit --json` merges those runtime-backed checks into one passive observer-side artifact
- auditor integration smoke covers live `dataset-audit --json` execution against the promoted corpus when the runtime is available
- `navi-auditor dashboard-attach-check --json` provides a headless passive proof that the dashboard-visible actor stream is observable during live training or replay
- `check-sdfdag` validates the promoted corpus manifest metadata against live `.gmdag` assets and the canonical compile resolution when run without `--gmdag-file`

### 3.2.1 End-To-End Qualification

The first scripted canonical end-to-end qualification surface is:

- `./scripts/qualify-canonical-stack.ps1`

It runs one bounded canonical train through `run-ghost-stack.ps1 -Train`,
captures a passive observer recording, proves live passive attach on the actor
stream, proves checkpoint resume from a produced periodic checkpoint, replays
the captured session, proves passive attach on the replay PUB, and emits one
JSON artifact under `artifacts/qualification/canonical_stack/`.

Qualification is also expected to archive explicit SDF/DAG validation evidence
for the active training corpus:

- one `check-sdfdag --json` corpus-level proof for the active compiled root
- one `check-sdfdag --json` representative-asset proof for the training corpus
- one `bench-sdfdag --json` representative benchmark artifact at the canonical observation contract

When invoked with `-EnableCorpusRefreshQualification`, the same script first
runs `scripts/refresh-scene-corpus.ps1` into a sandboxed compiled corpus root
under the run artifact and then trains against that refreshed corpus without
overwriting the user's live promoted corpus.

### 3.3 CUDA Boundary Validation

The imported project correctly emphasized that the Python-to-CUDA seam must be
validated aggressively.

For Navi this means tests should continue to cover:

- device mismatch rejection
- dimensionality rejection
- contiguity assumptions
- normalization tolerance enforcement for direction tensors
- CUDA-only fail-fast behavior when the backend is unavailable

### 3.4 Compiler And File-Format Validation

Compiler and artifact correctness should continue to expand around the real
`.gmdag` implementation.

Important coverage includes:

- deterministic compile output for fixed fixtures
- hash-collision defense and structural-equality fallback tests
- corrupted-header and corrupted-payload rejection tests
- child-mask and node-ordering invariants
- documented payload precision checks against the actual stored format

Required acceptance style for the canonical compiler surface:

- analytic plane and axis-aligned box fixtures must assert distance error against explicit voxel-scale tolerances rather than only checking for nonzero output
- repeated fixture compilation must assert byte-identical `.gmdag` output and stable node counts
- malformed `.gmdag` assets must fail before runtime traversal with errors attributable to header, bounds, or pointer-layout corruption
- integrity tests must be able to localize a failure to compiler generation versus loader validation instead of returning one generic corruption failure

Runtime load policy for canonical `.gmdag` assets:

- full DAG pointer-layout traversal remains mandatory on explicit validation surfaces such as loader integrity tests, `check-sdfdag`, corpus validation, qualification, and nightly proofs
- ordinary environment runtime startup and `bench-sdfdag` must not re-run that deep traversal on every backend construction once the asset is already on the validated canonical corpus path
- runtime startup still validates header fields, payload length, bounds finiteness, and trailing-byte integrity during load; the skipped work is the repeated full DAG graph walk, not all binary checks

### 3.5 Dataset And Transform Validation

Dataset-adapter behavior must be verified explicitly rather than described only
in prose.

Important coverage includes:

- coordinate-transform golden fixtures
- semantic remap fixtures
- forward-axis and orientation sanity checks
- adapter-backed fixture-sequence checks that prove episode resets clear delta-depth state before `DistanceMatrix` materialization
- dataset-preset checks that prove Habitat's explicit camera transform preserves Navi's canonical `+X` right, `+Y` up, `-Z` forward axes

### 3.6 Performance Proof

Performance claims are valid only when measured on the current canonical path.

Required proof style:

- use the real trainer for hot-path changes
- use `bench-sdfdag` for environment-layer attribution and preflight benchmarking
- compare against current baselines and current bottleneck interpretation
- report end-to-end impact, not just isolated kernel numbers
- prefer structured `bench-sdfdag --json` summaries when capturing benchmark artifacts for comparison
- environment-layer benchmarks must measure backend stepping rather than spending most of the wall time in repeated pre-step binary integrity traversal of the same already-qualified asset

### 3.6.1 Validation Matrix

Canonical SDF/DAG validation should be tracked in five explicit lanes:

1. compiler correctness: analytic fixtures, determinism, hash/dedup invariants
2. binary integrity: header validation, non-finite bounds rejection, pointer-layout rejection, trailing/truncated payload rejection
3. runtime correctness: hit/miss/epsilon/max-step/origin-boundary behavior against analytic or independent reference expectations
4. promoted-corpus proof: manifest parity, real-dataset-only validation, representative load and bench viability
5. regression automation: qualification and nightly artifacts that preserve machine-readable evidence for the prior four lanes

Each lane must emit pass/fail evidence that can be archived independently. The objective is not merely a green test run, but failure localization.

Fixture-level oracle policy:

- one small reusable fixture family should carry the known-result geometry truth across compiler, environment, actor, and auditor tests
- fixture-level proofs do not replace promoted-corpus validation; they localize math and contract failures while real-corpus checks prove canonical operational readiness
- expected spherical views, movement deltas, and projection-region checks should be defined once and reused across test layers so failures stay attributable

Nightly automation should therefore preserve at least one artifact per lane when
the canonical overnight flow runs.

## 4. Current Gaps Worth Closing

The imported docs also highlighted areas where more verification would be
valuable. The following are reasonable next additions:

1. long-run allocation-stability tests around repeated CUDA stepping
2. more explicit tests for tensor contiguity and dtype traps at the extension boundary
3. explicit `MAX_STEPS`, hit-epsilon, inside-solid, and out-of-domain behavior tests
4. native compiler/kernel invariant tests where practical, provided they reflect the real in-repo implementation
5. broader long-run end-to-end qualification beyond the first bounded scripted surface

The first passive dataset-auditor validation surface now exists as
`navi-auditor dataset-audit --json`, so future work can extend it toward richer
runtime-native rendering instead of building a second geometry-export proof path.

## 5. Benchmark Language Policy

The documentation must not claim throughput numbers that are not currently
backed by Navi measurements.

Use these rules instead:

- state current measured baselines as baselines
- state acceptance floors as floors
- state experiments as experiments
- avoid importing external headline numbers as if they were repo facts

## 6. Related Docs

- `docs/PERFORMANCE.md`
- `docs/SDFDAG_RUNTIME.md`
- `docs/SIMULATION.md`
