# Future-Hardware Performance Roadmap

> **Status:** Specification — implementation-ready when target hardware is available.
> **Audience:** Engineers preparing the next jump in training/inference throughput
> on hardware that exceeds the active MX150 (`sm_61`) machine.

This document specifies the optimization work that is **already validated as
correct in principle** but is **blocked on the active hardware**. Each item
below includes the exact code anchor, the gating capability check that already
exists, the concrete implementation steps, the expected gain, and the
benchmark gate that must accept the change before it becomes canonical.

The intent is that on the day a `sm_70+` (Volta/Turing/Ampere/Ada/Blackwell)
machine becomes the target, an engineer can execute these specs in order
without re-investigating feasibility.

For the validated reasons each item is currently blocked, see
[PERFORMANCE.md §9](PERFORMANCE.md#9-optimization-roadmap-and-blocked-proposals).
For permanently-blocked proposals (CUDA Graphs of the rollout loop, alternate
trainer modes, host-staged rollout), see PERFORMANCE.md §9.2 — those are
**not** in scope here regardless of hardware.

---

## 1. Hardware Tiers

| Tier | Min Capability | Unlocks | Representative GPU |
|------|----------------|---------|--------------------|
| **T1** | `sm_70` (Volta) | `torch.compile` Triton path on RayViT, reward shaping, PPO learner; first-gen tensor cores | RTX 2060, RTX 2070/2080, Titan V |
| **T2** | `sm_80` (Ampere) | Native `bfloat16` math; high-throughput tensor cores | RTX 3060, RTX 3070/3080/3090, A100 |
| **T3** | `sm_89` (Ada) / `sm_90` (Hopper) / `sm_100` (Blackwell) | FP8, Transformer Engine, larger SM count for `rollout_overlap_groups > 1` | RTX 4090, RTX 5090, H100 |
| **T-OS** | Linux + `sm_70+` | `mamba-ssm` fused Triton SSD kernels (Windows wheel unavailable) | Any Linux box with Volta+ |

Each implementation item below cites its required tier.

---

## 2. T1 — `torch.compile` for the PPO Learner

**Required hardware:** `sm_70+` (Volta or newer)
**Expected gain:** Eliminates the dispatcher-gap component of the
~1020 ms PPO update. On Mamba2 specifically, fuses the surrounding
element-wise math (KL, clip, value loss, entropy) but does **not** replace
the Mamba2 SSD scan itself — that requires Item 5 (`mamba-ssm`).
**Risk class:** Medium — must not change numeric results vs. eager.

### 2.1 Existing capability gate (already implemented elsewhere)

The pattern to follow is already established for `RayViTEncoder` and reward
shaping. Reuse the same gate verbatim:

```python
# projects/actor/src/navi_actor/cognitive_policy.py:144-152
capability = torch.cuda.get_device_capability()
if tuple(int(part) for part in capability) < (7, 0):
    _compile_reason = (
        f"CUDA capability {capability[0]}.{capability[1]} "
        "is below the Triton minimum 7.0"
    )
else:
    # ... torch.compile attach ...
```

### 2.2 Where to attach

- **File:** [projects/actor/src/navi_actor/learner_ppo.py](../projects/actor/src/navi_actor/learner_ppo.py)
- **Function:** `train_ppo_epoch` (around line 271)
- **Attach surface:** wrap the inner per-minibatch loss-and-step closure, **not**
  the top-level method. Wrapping the top-level method breaks the dynamic
  minibatch-shuffle / shape control flow.

### 2.3 Implementation steps

1. Extract the per-minibatch forward + loss computation (everything between
   the minibatch fetch and `optimizer.zero_grad()`) into a private function
   `_minibatch_loss_step(self, batch, ...) -> tuple[Tensor, dict[str, Tensor]]`.
2. Add a one-time capability-gated compilation in the learner constructor:
   ```python
   if _compile_supported():
       self._minibatch_loss_step = torch.compile(
           self._minibatch_loss_step,
           fullgraph=False,    # PPO has Python-side control flow per batch
           dynamic=False,      # minibatch shape is fixed per epoch
           mode="reduce-overhead",
       )
   ```
3. **Do not** use `mode="max-autotune"` — it causes long warm-up stalls that
   regress the first epoch wall time and AGENTS.md §3.3.1 forbids hot-path
   regressions for theoretical wins.
4. Add a `compile_attribution: dict[str, str]` field to `PpoMetrics` recording
   whether the compiled or eager path was taken and why (matches the existing
   `_compile_reason` pattern).

### 2.4 Validation

- Run [scripts/run-temporal-bakeoff.ps1](../scripts/run-temporal-bakeoff.ps1)
  with and without compile — quality must be statistically indistinguishable.
- `tests/` must be extended with a numerical-parity test:
  `eager_loss == compiled_loss` within `1e-5` for a fixed seed minibatch.
- **Benchmark gate:** end-to-end SPS must improve on the canonical 4-actor
  trainer without regressing PPO update wall time on epoch 1.

### 2.5 Rejection conditions

- Numeric drift > `1e-4` on any loss term
- First-epoch warm-up cost > +500 ms on the canonical trainer
- Any per-minibatch shape change forcing recompilation more than once

---

## 3. T1 — Promote Existing `torch.compile` Paths from Soft-Disabled to Required

**Required hardware:** `sm_70+`
**Expected gain:** Eliminates the soft-fallback path on the new hardware tier;
removes the `_compile_reason` warning chain.

The capability gate in `RayViTEncoder` and `reward_shaping` currently
**warns and falls back** on `sm_61`. On `sm_70+` the fallback path becomes
dead code we no longer need to maintain.

### 3.1 Steps

1. Keep the capability gate (it remains correct for older devices an external
   user might run on).
2. Add a startup hard-warning when `_compile_reason` is non-empty on a machine
   that AGENTS.md §3.1 declares the production target — this prevents silent
   fallback on a deployed-canonical box.
3. Promote the `eager` reward-shaping path in
   [projects/actor/src/navi_actor/reward_shaping.py:122](../projects/actor/src/navi_actor/reward_shaping.py)
   from `jit.script` fallback to `torch.compile` on the new tier (jit.script
   stays as the second-tier fallback).

### 3.2 Validation

Existing reward-shaping unit tests cover numeric parity; just re-run them
with `NAVI_ACTOR_FORCE_COMPILE=1` (env flag to be added) on the new hardware.

---

## 4. T2 — `bfloat16` Distance Output from `cast_rays`

**Required hardware:** `sm_80+` (Ampere or newer) for native `bfloat16`.
On `sm_70` (Volta) prefer `float16` instead — see §4.5.
**Expected gain:** Halves the bandwidth of the
`[B, R] = [4, 12_288]` distance/semantic tensors flowing into RayViT.
On `sm_80+`, the ingesting `RayViT` MatMul becomes tensor-core-native.
**Risk class:** Medium — touches the C++ kernel ABI and the actor encoder
input dtype.

### 4.1 Current state

- **Storage:** GMDAG already stores SDF leaf payloads as `float16`
  ([projects/voxel-dag/voxel_dag/compiler.py:505](../projects/voxel-dag/voxel_dag/compiler.py))
- **Decode:** Kernel decodes `float16 → float32` via `__half2float`
  ([projects/torch-sdf/cpp_src/kernel.cu:45](../projects/torch-sdf/cpp_src/kernel.cu))
- **Output:** Pinned to `float32`
  ([projects/torch-sdf/cpp_src/bindings.cpp:70](../projects/torch-sdf/cpp_src/bindings.cpp)):
  ```cpp
  TORCH_CHECK(out_distances.scalar_type() == torch::kFloat32,
              "out_distances must be float32");
  ```
- **Python contract:**
  [projects/torch-sdf/torch_sdf/backend.py:18](../projects/torch-sdf/torch_sdf/backend.py)
  hard-codes `_DISTANCE_DTYPE = torch.float32`.

### 4.2 Implementation steps

1. **Kernel side** — extend `bindings.cpp` with a templated dispatch:
   ```cpp
   TORCH_CHECK(
     out_distances.scalar_type() == torch::kFloat32 ||
     out_distances.scalar_type() == torch::kBFloat16,
     "out_distances must be float32 or bfloat16");
   ```
   The kernel writes a `__nv_bfloat16` cast of its internal `float` accumulator
   when the output tensor is `bfloat16`. Internal accumulation **must remain
   `float32`** — sphere-tracing accumulates many small steps and `bfloat16`
   accumulation diverges from oracle.
2. **Python side** — promote `_DISTANCE_DTYPE` to a function that resolves from
   `EnvironmentConfig.observation_dtype: Literal["float32", "bfloat16"]`
   (new field, default `"float32"`). Reject any other value with a fail-fast
   error per AGENTS.md.
3. **Backend allocation** — in
   [projects/environment/src/navi_environment/backends/sdfdag_backend.py](../projects/environment/src/navi_environment/backends/sdfdag_backend.py),
   the preallocated `out_distances` buffer must use the configured dtype.
4. **Adapter** — the `(B, R) → (B, Az, El) → (B, 3, Az, El)` reshape chain
   already runs in PyTorch and is dtype-agnostic. Verify the `valid_mask`
   channel (channel 2) and `semantic` channel (channel 1) handling — they are
   `int32`/`bool`, **not** `bfloat16`.
5. **Encoder side** — RayViT's first `nn.Linear` patch projection must accept
   `bfloat16` input. On `sm_80+` PyTorch handles this automatically; on `sm_70`
   force `float16` and add an `autocast` block.
6. **Oracle invariant** — keep the existing oracle test
   (`tests/.../test_ray_direction_oracle.py`) running in `float32` mode. Add a
   parallel test in `bfloat16` mode that allows tolerance `1e-2` (the
   bfloat16 epsilon on the canonical max distance).

### 4.3 Validation

- **Numerical drift:** `||cast_rays_fp32 - cast_rays_bf16||_∞ < 0.5%` of
  configured `max_distance`.
- **End-to-end:** Bake-off training run with `bfloat16` observations must
  reach learning quality within `0.1` reward_ema of the `float32` baseline at
  25K steps.
- **Bandwidth verification:** Nsight Compute / `torch.profiler` confirming
  `cast_rays` global-write bandwidth halves.

### 4.4 Rejection conditions

- Sphere-trace divergence — geometry hits move by more than 1 voxel
- Reward EMA regression > `0.1` on the canonical bake-off
- Any unexpected `bfloat16 → float32` upcast in the encoder forward path

### 4.5 Tier-T1 fallback

If only `sm_70` is available, emit `float16` instead. The kernel ABI extension
above already supports both. Mathematical risk is higher (`float16` has
narrower dynamic range than `bfloat16`); the tolerance gate must be tightened
to `1e-3` and clamping the max-distance horizon to `≤ 30 m` is required.

---

## 5. T-OS — Hardware-Fused `mamba-ssm` Triton Kernels

**Required hardware:** Linux + `sm_70+`. The PyPI `mamba-ssm` wheel is
**not built for Windows** (confirmed
[projects/actor/pyproject.toml:36](../projects/actor/pyproject.toml)).
**Expected gain:** Collapses Mamba2's 55–60 dispatcher launches per forward
into a single fused SSD scan kernel. This is the largest single backward-pass
optimization available, dwarfing items 2–4.

### 5.1 Steps

1. Switch deployment OS to Linux (or use WSL2 with CUDA passthrough — verify
   wheel install first).
2. The conditional dependency in `projects/actor/pyproject.toml` is
   already in place:
   ```toml
   "mamba-ssm>=2.2.2; platform_system != 'Windows'"
   ```
   It will install automatically on the new platform.
3. Add a new temporal-core selector value `"mamba2_fused"` to
   [projects/actor/src/navi_actor/config.py:12-13](../projects/actor/src/navi_actor/config.py):
   ```python
   TemporalCoreName = Literal["gru", "mambapy", "mamba2", "mamba2_fused"]
   ```
4. Implement `Mamba2FusedTemporalCore` in
   [projects/actor/src/navi_actor/cognitive_policy.py](../projects/actor/src/navi_actor/cognitive_policy.py)
   wrapping `mamba_ssm.modules.mamba2.Mamba2`. Match the existing
   `Mamba2SSDTemporalCore` parameter signature 1:1.
5. Add the selector branch in `_build_temporal_core`
   ([cognitive_policy.py:76-101](../projects/actor/src/navi_actor/cognitive_policy.py)).

### 5.2 Validation — Promotion bake-off

Per AGENTS.md §2.7 Promotion Rule, the fused core may replace `mamba2` as the
canonical default **only when** a 25K-step bake-off proves equal-or-better
learning quality **and** equal-or-better throughput. Run via
[scripts/run-temporal-bakeoff.ps1](../scripts/run-temporal-bakeoff.ps1) with
matrix `mamba2 vs mamba2_fused`.

### 5.3 Rejection conditions

- Any numerical divergence from `mamba2` reference in deterministic mode
- Throughput regression vs. `mamba2` SSD on the same machine
- Backward-pass instability (NaN gradients, exploding loss) at any seed

---

## 6. T3 — Multi-Group Rollout Overlap (`rollout_overlap_groups > 1`)

**Required hardware:** GPU with ≥ 30 SMs (typical for RTX 30/40/50 series and
data-center cards). The MX150 has only 3 SMs, which is why
`rollout_overlap_groups=2` causes a documented ~47% throughput regression on
this machine ([PERFORMANCE.md §9.1](PERFORMANCE.md#91-blocked-on-active-hardware-future-hardware-roadmap)).

### 6.1 Steps

1. Set `NAVI_ACTOR_ROLLOUT_OVERLAP_GROUPS=2` (or higher) on the new machine
   and run the canonical trainer. The infrastructure already exists at
   [projects/actor/src/navi_actor/config.py:126-129](../projects/actor/src/navi_actor/config.py).
2. Sweep `{2, 3, 4, 6, 8}` against fixed PPO update cadence and pick the
   value that maximizes SPS without regressing learning quality.
3. Update [docs/PERFORMANCE.md §9](PERFORMANCE.md#9-optimization-roadmap-and-blocked-proposals)
   recording the new optimum on the new tier.

### 6.2 Hard rule (already in AGENTS.md §3.3.1)

> *"Grouped rollout-overlap rewrites MUST preserve actor-local
> `observation -> action -> next observation` ordering. No group may consume
> stale observations from another phase just to increase overlap."*

If the SPS win comes from breaking this rule, it is **rejected regardless of
benchmark numbers**.

---

## 7. Cross-Cutting Hard Rules (All Items)

These are non-negotiable per AGENTS.md and apply to every item above:

1. **Sacred Brain (AGENTS.md §2.5)** — the cognitive pipeline
   `RayViTEncoder → TemporalCore → EpisodicMemory → ActorCriticHeads` is
   immutable. Optimizations may change *implementation* (compile, dtype, fused
   kernel) but **not the architectural composition**.
2. **One Canonical Trainer (AGENTS.md §2.7)** — no alternate trainer mode for
   any of these items. All work lands on the existing `navi-actor train`
   surface, capability-gated.
3. **Benchmark Gate (AGENTS.md §3.1)** — every item ships only after a
   numbered bake-off shows the gain on the canonical 4-actor trainer.
4. **No Compatibility Wrappers (AGENTS.md §2.6)** — when an item is enabled by
   default on the new tier, the eager / fp32 / Windows-Mamba2 paths remain
   only as capability fallbacks, not as deprecated aliases.
5. **Hard Dependency Policy** — none of these items may introduce optional
   fallbacks that duplicate canonical behavior. Capability gates are the only
   acceptable branching mechanism.

---

## 8. Recommended Implementation Order on New Hardware

When a `sm_80+` machine becomes the production target, execute in this order
to minimize integration risk:

1. **Item 3** (promote existing compile paths) — pure warning-removal, no
   code rewrite. Validates the new toolchain works.
2. **Item 5** (`mamba-ssm` fused SSD) if on Linux — biggest single win and
   independent of dtype changes.
3. **Item 2** (`torch.compile` PPO learner) — depends on Item 1's toolchain
   validation but independent of dtype.
4. **Item 4** (`bfloat16` output) — touches both kernel and actor encoder;
   land it last so the previous wins are stable baselines for the bandwidth
   gate.
5. **Item 6** (rollout overlap) — sweep last; the optimal value depends on
   how much PPO time the previous items have already saved.

After all five, re-run the full nightly validation
([docs/NIGHTLY_VALIDATION.md](NIGHTLY_VALIDATION.md)) and update the canonical
benchmark baselines in [PERFORMANCE.md §6](PERFORMANCE.md#6-current-reference-baselines).

---

## 9. Related Docs

- [PERFORMANCE.md](PERFORMANCE.md) — current measured baselines and §9
  validated-but-blocked roadmap
- [SDFDAG_RUNTIME.md](SDFDAG_RUNTIME.md) — CUDA boundary contract that Item 4
  must respect
- [ACTOR.md](ACTOR.md) — sacred cognitive pipeline composition that Items 2,
  3, 5 must preserve
- [SIM_TO_REAL_PARITY.md](SIM_TO_REAL_PARITY.md) — the Pipeline-A invariants
  that all items must preserve
- [AGENTS.md](../AGENTS.md) — non-negotiable architectural rules

---

*This document is a forward-looking implementation specification. None of the
items below are active on the canonical `sm_61` trainer. They become active
work items the moment the hardware tier is upgraded.*
