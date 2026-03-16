# TODO — Performance Compliance Implementation Plan

## Status Legend
- `[ ]` Not started
- `[~]` In progress
- `[x]` Complete

---

## Phase A: torch.compile / JIT Enablement — COMPLETE

### A1. torch.compile Investigation `[x]`
- Confirmed: Triton requires SM >= 7.0. MX150 (SM 6.1) cannot use torch.compile/inductor.
- Installed Triton is for Python 3.10 (wrong version for active 3.12 environment).

### A2. fullgraph Fix `[x]`
- Changed `fullgraph=False` to `fullgraph=True` in `_maybe_compile_callable` (sdfdag_backend.py).
- Changed `fullgraph=False` to `fullgraph=True` in reward_shaping.py.

### A3. torch.jit.script Fallback `[x]`
- Implemented fallback chain: `torch.compile(fullgraph=True)` → `torch.jit.script` → eager.
- Applied to sdfdag_backend.py (`_maybe_compile_callable`) and reward_shaping.py.
- Type hints changed `Any` → `torch.Tensor` for jit.script compatibility.
- Module constants `_PROGRESS_REWARD_SCALE` and `_COLLISION_PENALTY` passed as function params.
- Measured speedups on SM 6.1: 1.44x (kinematics), 3.25x (reward computation).

---

## Phase B: Legacy Code Elimination — COMPLETE

### B1. Dead Code Removal `[x]`
- Removed `_reward_components_impl` (~60 lines, zero callers).
- Removed `_step_kinematics_impl` (~35 lines, zero callers).
- Removed `_postprocess_cast_outputs_impl` (~30 lines, test updated to `_postprocess_cast_outputs`).
- Updated test: `test_sdfdag_conventions.py` to use instance method with required attributes.

### B2. bench-sdfdag Tensor Conversion `[x]`
- Converted CLI benchmark from legacy `Action` objects to tensor-native stepping.
- `_benchmark_actions_tensor` creates `(actors, 4)` tensor: `[fwd, vert, lat, yaw]`.
- `_run_bench_iteration` uses `batch_step_tensor_actions()` directly.
- Removed `numpy` and `Action` imports from cli.py.

### B3. Legacy Methods Retained (Server Path) `[x]`
- Audit confirmed: `step()`, `batch_step()`, `reset()` actively used by `server.py`.
- These serve the ZMQ distributed inference/dashboard path — cannot be removed.
- Internal helpers (`_build_observation`, `_compute_reward`, `_step_kinematics_batch`)
  are called only from legacy `batch_step()` — retained as part of server path.

---

## Phase C: Configuration Alignment — COMPLETE

### C1. config.py Defaults `[x]`
Updated `projects/actor/src/navi_actor/config.py` to match CLI production defaults:

| Parameter | Old | New |
|-----------|-----|-----|
| `minibatch_size` | 32 | 64 |
| `bptt_len` | 16 | 8 |
| `value_coeff` | 0.005 | 0.5 |
| `existential_tax` | -0.01 | -0.02 |
| `velocity_weight` | 0.0 | 0.1 |
| `intrinsic_coeff_init` | 0.2 | 1.0 |
| `loop_penalty_coeff` | 0.5 | 2.0 |

---

## Phase D: Encoder Compilation — COMPLETE

### D1. RayViTEncoder torch.compile `[x]`
- Added `torch.compile(fullgraph=False, dynamic=False, mode="reduce-overhead")` wrapping.
- C++ compiler availability check (`torch._inductor.cpp_builder.get_cpp_compiler()`)
  prevents lazy-compilation failures on machines without MSVC.
- Falls back to eager mode with log message on unsupported hardware.
- Positional encoding caching verified as already implemented.

---

## Phase E: Documentation — COMPLETE

### E1. NOTES.md Rewrite `[x]`
- Restored `NOTES.md` to a verified status document after later drift reintroduced
  aspirational large-scale claims and stale execution details.
- Notes now document the active canonical path, hardware-gated compile behavior,
  durable long-run launch guidance, and the remaining optional perf work only.

---

## Phase F: Verification — COMPLETE

### F1. Ruff `[x]`
- Final close-out validation is scoped to files changed in this work rather than
  unrelated repository-wide lint debt.
- Repo-wide ruff still reports pre-existing issues outside this task surface.

### F2. Tests `[x]`
- Runtime validation covered the canonical training surface, passive dashboard
  attachment behavior, compile gating on SM 6.1, and selector startup behavior.
- Targeted actor and environment unit validation remains the regression gate for
  the touched code paths.

### F3. Type Checking `[x]`
- Strict `mypy` remains part of the required validation surface.
- Use `uv run --project .\projects\actor --with mypy mypy ...` and
  `uv run --project .\projects\environment --with mypy mypy ...` when the dev
  extra is not already installed in the local workspace environment.

---

## Close-Out Status

All mandatory implementation, validation, and documentation work from this plan
is complete.

## Deferred Performance R&D

These items are intentionally not tracked as active TODOs for the current
close-out because they are future optimization experiments, not unresolved
correctness, architecture, or validation gaps.

### CUDA Macro-Cell Spatial Caching
- `projects/torch-sdf/cpp_src/kernel.cu` void-distance cache in ray marching loop.
- Targets `env_ms` (44-55ms), which is the dominant bottleneck on MX150.
- Requires C++ CUDA kernel changes and benchmark proof before promotion.

### Asynchronous Double-Buffering
- `torch.cuda.Stream` ping-pong in `ppo_trainer.py`.
- Not beneficial at 4 actors on MX150 (batch too small to saturate GPU).
- Relevant only for future high-end GPU and larger fleet sizes.
