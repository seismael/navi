# COMPILER.md - `.gmdag` Compiler And Corpus Build Architecture

## 1. Executive Summary

The compiler domain bridges human-authored 3D assets and the CUDA runtime used
by Navi training. Its job is to transform source geometry into compact compiled
artifacts that the runtime can load and traverse efficiently.

In the current repo this function is split across:

- `projects/voxel-dag` as the sovereign compiler implementation
- `projects/environment` as the orchestration layer that invokes compilation,
  prepares corpora, and validates live artifacts

## 2. Current Compiler Surfaces

### 2.1 Native C++ Compiler

The native compiler project includes:

- `src/main.cpp` for CLI entry
- `src/compiler.hpp` and `src/compiler.cpp` for pipeline stages
- CMake-based build orchestration with `assimp` fetched via `FetchContent`

The native CLI accepts:

- `--input`
- `--output`
- `--resolution`

### 2.2 Python Verification Surface

The repo also includes a Python compiler surface used for tests and local
artifact generation. This is useful because it lets the monorepo validate file
format and runtime integration without always shelling out to the native CLI.

## 3. Compilation Pipeline

The imported compiler document had the right broad structure. The current repo
can be documented in the same staged style.

### 3.1 Mesh Ingest

The compiler ingests source meshes and computes source bounds.
The native implementation uses `assimp` for multi-format support.

### 3.2 Spatial Normalization

The imported compiler emphasized cubic normalization and power-of-two alignment.
Those remain useful architectural principles because they simplify downstream
voxel spacing and traversal assumptions.

### 3.3 Distance-Field Computation

The native compiler code documents an Eikonal / Fast Sweeping stage for dense
SDF construction before compression.

### 3.4 DAG Compression

The compiled representation is compressed into a compact DAG-friendly payload.
The native code currently documents a `compressToDAG()` stage and uses hash-based
structural folding in the compiler pipeline.

### 3.5 Binary Serialization

The final stage writes the `.gmdag` binary with:

- a fixed header
- a contiguous `uint64` node payload

That artifact becomes the source of truth for the runtime engine.

## 4. Current Environment-Orchestrated Surfaces

The environment CLI exposes the compiler and corpus lifecycle through:

- `compile-gmdag`
- `prepare-corpus`
- `check-sdfdag`
- `bench-sdfdag`

This is the production-facing operational surface that users actually rely on,
so environment orchestration must remain well documented even though the
compiler project is sovereign.

## 5. Corpus Preparation Rules

Canonical corpus preparation rules are:

- default resolution `512`
- full discovered corpus by default when the user does not narrow the dataset
- overwrite-first rebuild when explicitly requested
- promotion only after a successful staged refresh
- replacement of stale compiled assets when stored resolution mismatches the
  requested canonical resolution

## 6. File Format And Runtime Relationship

The compiler does not exist in isolation. Its artifact contract must match the
runtime loader and kernel assumptions.

The important repo-local relationship is:

- compiler writes `.gmdag`
- environment integration validates and loads `.gmdag`
- runtime traverses that payload directly on CUDA

For this reason, compiler changes are benchmark-gated and integration-gated.
They are not accepted on elegance alone.

## 7. Benchmark-Gated Compiler Ideas

The imported docs proposed several low-level storage ideas that are still worth
keeping as candidates:

- truncation-aware storage
- alternate distance payload precision policies
- deeper layout redesigns for locality
- more aggressive DAG folding strategies

These remain benchmark-gated because the real acceptance criterion is end-to-end
trainer impact.

## 8. Validation Surfaces

Compiler and corpus validation currently includes:

- `prepare-corpus` for discovery and build orchestration
- `check-sdfdag` for runtime and artifact readiness
- live asset loading through environment integration helpers
- runtime stepping against promoted compiled assets

## 9. Related Docs

- `docs/ARCHITECTURE.md`
- `docs/SDFDAG_RUNTIME.md`
- `docs/SIMULATION.md`
- `docs/VERIFICATION.md`
