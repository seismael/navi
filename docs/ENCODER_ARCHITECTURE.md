# ENCODER_ARCHITECTURE.md — Encoder Selector Standard

Navi's cognitive pipeline supports selectable encoder backends that transform
the canonical spherical `DistanceMatrix` observation `(B, 3, Az=256, El=48)`
into a 128-dimensional spatial embedding for the temporal core.  This document
defines the selector contract, the two available encoder architectures, their
performance characteristics, and the promotion criteria for changing the
canonical default.

---

## 1. Encoder Contract

All encoders MUST satisfy:

| Contract | Detail |
|----------|--------|
| Input | `(B, 3, Az, El)` float tensor — channels are `(depth, semantic, valid)` |
| Output | `(B, embedding_dim)` float tensor — default `embedding_dim=128` |
| Gradient flow | Full autograd from policy loss through encoder parameters |
| Critical stop-gradient | Value-loss gradients from `ActorCriticHeads` must NOT flow to encoder |
| Checkpoint | Weights stored in `policy_state_dict` under `encoder.*` keys |
| Environment | No dependency on environment backend — must work with any `(B, 3, Az, El)` tensor |
| torch.compile | Attempted at policy construction on `sm_70+`; falls back to eager on older hardware |

---

## 2. Available Encoders

### 2.1 RayViTEncoder (`rayvit`) — Canonical Default

A Vision Transformer with fixed spherical positional encodings.

| Property | Value |
|----------|-------|
| Architecture | Patch projection (Conv2d, `patch_size=8`) → [CLS] token → 2-layer TransformerEncoder (4 heads, d=128, d_ff=256) → LayerNorm → Linear projection |
| Tokens at 256×48 | 192 (32 azimuth × 6 elevation) + 1 CLS |
| Self-attention complexity | O(T²) where T=193 tokens |
| Parameters | ~306,560 |
| FLOPs (forward, 4×256×48) | ~150M |
| Forward latency (MX150, eager) | 5.0 ms (median of 200 trials) |
| Forward+backward latency | 11.1 ms |
| Strengths | Global attention catches long-range spatial patterns; cached positional encodings; cuDNN-fused multi-head attention |
| Weaknesses | Quadratic token cost limits high-resolution scaling; OOM at 768×144 |

### 2.2 SphericalCNN (`spherical_cnn`) — Experimental

A CNN that processes the spherical distance matrix as a 2D image with circular
azimuth topology.

| Property | Value |
|----------|-------|
| Architecture | Stem (3→32, k=3, stride=(1,2)) → 2× depthwise-separable blocks (32→64→128) → 1× standard conv refinement → Global Average Pool → small MLP projection |
| Parameters | ~226,752 |
| FLOPs (forward, 4×256×48) | ~29M (5.1× fewer than RayViT) |
| Forward latency (MX150, eager) | 4.3 ms |
| Forward+backward latency | 10.8 ms |
| Strengths | O(N) linear complexity; fewer parameters; GAP eliminates massive FC layer; train-mode BN distinguishes elevation |
| Weaknesses | Depthwise-separable convs dispatch many small CUDA kernels on eager PyTorch; requires `torch.compile` on `sm_70+` for the FLOP advantage to translate to wall-time speedup |

**Circular Azimuth Padding:** PyTorch's native `padding_mode='circular'` wraps
both spatial dimensions.  `CircularAzimuthConv2d` applies circular padding only
to the azimuth (width) dimension — ray 0 is adjacent to ray 255 on the 360°
sphere.  Elevation (height) is zero-padded because the celestial poles are not
adjacent.

```
F.pad(x, (pw, pw, 0, 0), mode='circular')   ← azimuth wrap
F.pad(x, (0, 0, ph, ph), mode='constant')    ← elevation zero-pad
```

---

## 3. Training Comparison (MX150, sm_61, no torch.compile)

| Metric | RayViT | SphericalCNN | Note |
|--------|--------|-------------|------|
| Encoder forward (4×256×48) | 5.0 ms | 4.3 ms | 1.2× faster |
| Encoder fwd+bwd | 11.1 ms | 10.8 ms | ~1.0× (dispatcher-bound) |
| Full-policy forward | 17.0 ms | 15.8 ms | 1.1× |
| PPO minibatch BPTT (8×8) | 46.9 ms | 44.8 ms | 1.0× |
| Training SPS (4096 steps) | **52.8** | 46.4 | -14% |
| Early reward_ema (2K steps) | **−0.68** | −2.59 | RayViT learns faster |

**Interpretation:** On `sm_61` without `torch.compile`, the SphericalCNN's
depthwise-separable conv dispatches consume the FLOP advantage through eager
PyTorch kernel-launch overhead (~10–100μs per kernel).  RayViT remains the
throughput leader, and shows superior early learning quality.

---

## 4. Expected Gains on sm_70+ (with torch.compile)

When `torch.compile` fuses the CNN's depthwise and pointwise conv operations
into single kernels, the dispatcher overhead disappears and the 5.1× FLOP
advantage translates to wall-time speedup:

| Metric | Current (sm_61) | Predicted (sm_70+) |
|--------|-----------------|-------------------|
| Encoder forward | 4.3 ms | ~0.5 ms |
| Full-policy forward | 15.8 ms | ~12 ms |
| Training SPS | 46.4 | ~60–70 |
| Ratio vs RayViT | 0.9× | ~1.2–1.4× |

**Gate:** A bounded 25K-step bake-off on `sm_70+` hardware is required to prove
both throughput AND converged reward_ema before `spherical_cnn` can be promoted
to canonical default per `AGENTS.md §2.9` Promotion Rule.

---

## 5. 3D Gaussian Splatting Investigation — Not Applicable

A rigorous analysis determined that 3DGS is not a productive investment for
Navi:

1. **Category error:** NAVI uses an explicit SDF/DAG representation, not an
   implicit neural field like NeRF. The 3DGS-vs-NeRF comparison does not apply.

2. **Output mismatch:** NAVI produces spherical depth maps, not RGB images.
   3DGS tile-based rasterization is optimized for single pinhole camera
   viewpoints and does not map to 12,288 spherical ray directions.

3. **Wrong bottleneck:** The environment backend (SDF/DAG sphere tracing) is
   **not the training bottleneck**. PPO update cost (~1020 ms) dominates the
   environment step (~36 ms) by ~28:1 on MX150.

4. **SDF/DAG advantages preserved:** Exact deterministic distances, zero-
   probabilistic-uncertainty collision detection, continuous surface gradients,
   zero-query-cost void caching, and byte-identical compilation.

**Verdict:** The proven path to better throughput is hardware upgrade
(`sm_70+` for `torch.compile`) and PPO/rollout double-buffer overlap.
3DGS would require massive engineering investment for negligible benefit.

---

## 6. Integration Surface

The `encoder_backend` selector is integrated across the full stack:

| Component | Integration |
|-----------|------------|
| `ActorConfig` | `encoder_backend` field (default `"rayvit"`, env `NAVI_ACTOR_ENCODER_BACKEND`) |
| `CognitiveMambaPolicy` | `encoder_backend` parameter in `__init__` |
| CLI — `serve` | `--encoder-backend` flag |
| CLI — `train` | `--encoder-backend` flag |
| CLI — `infer` | `--encoder-backend` flag |
| CLI — `profile` | `--encoder-backend` flag |
| CLI — `brain` | `--encoder-backend` flag (passes to sub-commands) |
| CLI — `bc-pretrain` | `--encoder-backend` flag |
| CLI — `evaluate` | `--encoder-backend` flag |
| CLI — `compare` | `--encoder-backend` flag |
| v3 checkpoints | `encoder_backend` field in metadata |
| `load_training_state` | Cross-encoder rejection: mismatch raises `RuntimeError` |
| `ModelRegistry` / `ModelEntry` | `encoder_backend` field stored in `registry.json` |
| Old checkpoints | No `encoder_backend` field → defaults to `"rayvit"` (backward compatible) |

---

## 7. Usage

```powershell
# Canonical training (RayViT — default, no flag needed)
uv run brain train --actors 4

# Experimental SphericalCNN
uv run brain train --actors 4 --encoder-backend spherical_cnn

# Inference with SphericalCNN checkpoint
uv run brain infer --checkpoint ./model.pt --encoder-backend spherical_cnn

# Compare encoders in evaluate
uv run brain evaluate ./model.pt --encoder-backend spherical_cnn
```

---

## 8. Related Documents

- [AGENTS.md §2.9](../AGENTS.md) — Encoder Selector Standard
- [PERFORMANCE.md §4.2](PERFORMANCE.md) — SphericalCNN encoder baselines
- [ACTOR.md](ACTOR.md) — Cognitive engine architecture
- [ARCHITECTURE.md](ARCHITECTURE.md) — System architecture and runtime boundaries
