# RESOLUTION_BENCHMARKS.md — Observation-Resolution Benchmark Appendix

This appendix collects the key March 17, 2026 observation-resolution evidence
in one place so the canonical docs can cite one compact summary instead of
repeating the same artifact roots.

Use this file for quick review only. Policy and interpretation rules still live
in `AGENTS.md`, `docs/PERFORMANCE.md`, `docs/ACTOR.md`, `docs/TRAINING.md`, and
`docs/VERIFICATION.md`.

## 1. Active Machine

- GPU: NVIDIA GeForce MX150
- CUDA capability: `sm_61`
- VRAM: `2 GB`
- canonical trainer temporal core: `gru`
- canonical production observation contract: `256x48`

## 2. Canonical Trainer Sweep

Primary artifact root:

- `artifacts/benchmarks/resolution-compare/resolution-compare-20260317-003916/`

Focused GPU-sampled `512x96` artifact root:

- `artifacts/benchmarks/resolution-compare/resolution-compare-20260317-004714/`
- `artifacts/benchmarks/resolution-compare/gpu-sample-512x96-20260317.csv`

High-resolution failure artifact:

- `artifacts/benchmarks/resolution-compare/resolution-compare-20260317-002948/768x144/repeat-01/train.log`

| Profile | Rays / Actor | Patch Tokens | Steady SPS | `env_ms` | `ppo_update_ms` | Outcome |
| --- | --- | --- | --- | --- | --- | --- |
| `256x48` | `12,288` | `192` | `49.6` | `36.50` | `1,019.68` | canonical bounded baseline |
| `384x72` | `27,648` | `432` | `49.34` | `39.96` | `1,249.87` | still healthy |
| `512x96` | `49,152` | `768` | `43.96` | `50.64` | `17,731.88` | runnable, PPO-update dominated |
| `768x144` | `110,592` | `1728` | n/a | n/a | n/a | full trainer OOM in actor attention |

Interpretation:

- `env_ms` grows with the larger sphere, but `ppo_update_ms` grows far faster
- the current scaling wall is actor-side RayViT attention and PPO update memory
- on the active MX150 surface, `512x96` is a diagnostic comparison profile, not
  a production default

## 3. GPU Sample Notes For `512x96`

Focused `512x96` run from `resolution-compare-20260317-004714`:

- `steady_sps_mean = 25.1`
- `env_ms_mean = 76.42`
- `trans_ms_mean = 19.44`
- `ppo_update_ms_mean = 21,549.10`
- `backward_ms_mean = 1,586.31`

Sampled GPU behavior from `gpu-sample-512x96-20260317.csv`:

- GPU utilization reaches `100%`
- VRAM peaks at about `1963 MiB` out of `2048 MiB`
- the PPO update window is the dominant pressure period

Interpretation:

- the machine is already near its memory ceiling at `512x96`
- end-to-end scaling is constrained before the environment runtime itself stops
  being benchmark-viable

## 4. Environment-Only Interpretation Rule

Environment-only `bench-sdfdag` runs remain a separate proof surface.

What they prove:

- whether the CUDA runtime still steps efficiently as ray count rises
- whether compiler or runtime changes regress environment throughput
- whether a profile is still runtime-viable before the actor is involved

What they do not prove:

- that the full trainer will sustain the same profile
- that actor-side attention or PPO update memory will remain healthy

## 5. Canonical Conclusion

The active production default stays at `256x48`.

Current upgrade implications are:

- better hardware may move the ceiling outward
- a future fused temporal runtime may reduce part of PPO update cost
- neither of those changes alone removes RayViT token-attention scaling
- true high-resolution production promotion will require fresh end-to-end proof
  on the canonical trainer surface