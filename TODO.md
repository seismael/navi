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

## Phase B: Legacy Code Elimination — PARTIAL

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
- Replaced 664 lines of aspirational design document with concise 80-line status document.
- Documents what's implemented vs remaining opportunities.
- No stale code examples or contradictory architecture descriptions.

---

## Phase F: Verification — COMPLETE

### F1. Ruff `[x]`
- All modified files pass ruff: sdfdag_backend.py, cli.py, cognitive_policy.py,
  config.py, reward_shaping.py, test_sdfdag_conventions.py.
- Pre-existing RET504 in sdfdag_backend.py (not from our changes).

### F2. Tests `[x]`
- Environment: 30/30 passed (test_sdfdag_conventions.py).
- Actor: 49/50 passed, 1 pre-existing failure (missing mamba_ssm module), 3 skipped.
- No regressions from our changes.

---

## Remaining Work (Not Addressed)

### CUDA Macro-Cell Spatial Caching `[ ]`
- `projects/torch-sdf/cpp_src/kernel.cu` void-distance cache in ray marching loop.
- Targets `env_ms` (44-55ms), which is the dominant bottleneck on MX150.
- Requires C++ CUDA kernel changes — highest potential impact.

### Asynchronous Double-Buffering `[ ]`
- `torch.cuda.Stream` ping-pong in `ppo_trainer.py`.
- Not beneficial at 4 actors on MX150 (batch too small to saturate GPU).
- Relevant for future high-end GPU with large fleet (10,000+ actors).
