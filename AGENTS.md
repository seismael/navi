# AGENTS.md — Ghost-Matrix Implementation Blueprint

This file is the implementation policy for [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## 1) Scope

Navi is a Ghost-Matrix system focused on throughput RL with strict separation of:
- Simulation Layer (headless stepping + sensing via pluggable backends)
- Brain Layer (policy + training — sacred, immutable engine)
- Gallery Layer (record/replay/visualization — passive only)

## 2) Non-Negotiables

1. Canonical wire contracts are v2 only:
   - `RobotPose`
   - `DistanceMatrix`
   - `Action`
   - `StepRequest`
   - `StepResult`
   - `BatchStepRequest`
   - `BatchStepResult`
   - `TelemetryEvent`
2. No other models may be added to the contracts package without explicit approval.
   Visualization types (RGB frames, camera images) are never canonical contracts.
3. Legacy wire contracts/topics are not allowed in new code.
4. Inter-process communication is ZMQ only (PUB/SUB + REQ/REP).
5. Training runtime is headless; rendering is optional and asynchronous.
6. Services are sovereign packages; no service imports another service package.
   **Known exception:** The actor’s `train-sequential` CLI command imports from
   `navi_environment` to enable in-process training without ZMQ. This is an
   acknowledged sovereignty violation scoped to a single CLI command; the core
   actor engine never imports environment types.
7. Code quality gates remain mandatory: `ruff`, `mypy --strict`, `pytest`.
8. **The training engine is sacred.** The actor's cognitive pipeline
   (FoveatedEncoder → Mamba2 → EpisodicMemory → ActorCriticHeads → PPO)
   is never modified to accommodate a new data source. External data always
   connects through a `DatasetAdapter` that transforms *to* the engine's
   canonical `(1, Az, El)` DistanceMatrix format.
9. All backends must produce arrays in canonical shape `(n_envs, Az, El)` with
   `matrix_shape = (azimuth_bins, elevation_bins)`. Depth normalised to `[0, 1]`.

## 3) Repository Structure

```text
navi/
├── AGENTS.md
├── README.md
├── TODO.md
├── Makefile
├── docs/
│   ├── ARCHITECTURE.md          # system layers, SDF theory, design decisions
│   ├── ACTOR.md                 # cognitive engine specification (sacred)
│   ├── SIMULATION.md            # simulation layer + backends + raycasting
│   ├── PERFORMANCE.md           # theoretical baselines & throughput targets
│   └── CONTRACTS.md             # canonical wire format specification
├── data/
│   └── scenes/                  # scene assets + manifest
│       └── sample_episodes.json
├── artifacts/
│   └── checkpoints/             # training checkpoints (gitignored)
├── scripts/
│   ├── run-ghost-stack.ps1      # one-command full stack launch
│   ├── train-habitat-sequential.ps1  # sequential multi-scene training
│   ├── download-habitat-data.ps1     # download HSSD/ReplicaCAD scenes
│   ├── generate_sample_scene.py      # generate sample PointNav episodes
│   └── bench_raycast.py              # raycast engine benchmark
└── projects/
    ├── contracts/
    │   └── src/navi_contracts/
    │       ├── models.py
    │       ├── topics.py
    │       ├── serialization.py
    │       └── types.py
    ├── environment/
    │   └── src/navi_environment/
    │       ├── server.py            # thin ZMQ shell
    │       ├── cli.py
    │       ├── config.py
    │       ├── raycast.py           # RaycastEngine (scatter-reduce)
    │       ├── distance_matrix_v2.py
    │       ├── mjx_env.py           # MjxEnvironment (kinematics)
    │       ├── matrix.py            # SparseVoxelGrid
    │       ├── sliding_window.py    # SlidingWindow
    │       ├── frustum.py           # FrustumLoader
    │       ├── lookahead.py         # LookAheadBuffer
    │       ├── pruning.py           # chunk pruning utilities
    │       ├── backends/
    │       │   ├── base.py              # SimulatorBackend ABC
    │       │   ├── adapter.py           # DatasetAdapter Protocol
    │       │   ├── voxel.py             # VoxelBackend (procedural)
    │       │   ├── mesh_backend.py      # MeshSceneBackend (trimesh)
    │       │   ├── habitat_backend.py   # HabitatBackend (habitat-sim)
    │       │   ├── habitat_adapter.py   # HabitatAdapter (raw→canonical)
    │       │   └── habitat_semantic_lut.py
    │       ├── generators/
    │       │   ├── base.py              # AbstractWorldGenerator ABC
    │       │   ├── arena.py
    │       │   ├── city.py
    │       │   ├── maze.py
    │       │   ├── rooms.py
    │       │   ├── open3d_voxel.py
    │       │   └── file_loader.py
    │       └── transformers/
    │           └── compiler.py          # WorldModelCompiler (PLY/OBJ/STL→Zarr)
    ├── actor/
    │   └── src/navi_actor/
    │       ├── server.py
    │       ├── cli.py
    │       ├── config.py
    │       ├── spherical_features.py    # 17-dim feature extraction
    │       ├── cognitive_policy.py      # CognitiveMambaPolicy (sacred)
    │       ├── perception.py            # FoveatedEncoder CNN
    │       ├── mamba_core.py            # Mamba2TemporalCore
    │       ├── actor_critic.py          # ActorCriticHeads
    │       ├── rnd.py                   # RND curiosity
    │       ├── reward_shaping.py
    │       ├── learner_ppo.py           # PpoLearner
    │       ├── rollout_buffer.py        # TrajectoryBuffer (BPTT)
    │       ├── memory/
    │       │   └── episodic.py           # EpisodicMemory (CPU FAISS KNN)
    │       └── training/
    │           └── ppo_trainer.py        # PpoTrainer (PPO + RND + GAE)
    └── auditor/
        └── src/navi_auditor/
            ├── recorder.py
            ├── rewinder.py
            ├── stream_engine.py      # StreamEngine (ZMQ subscriber)
            ├── matrix_viewer.py
            ├── cli.py
            ├── config.py
            ├── storage/
            │   ├── base.py
            │   └── zarr_backend.py
            └── dashboard/
                ├── app.py            # GhostMatrixDashboard (PyQtGraph)
                ├── panels.py
                ├── renderers.py
                ├── scene_view.py
                └── occupancy_view.py
```

## 4) Active Runtime Topics

- `distance_matrix_v2`
- `action_v2`
- `step_request_v2`
- `step_result_v2`
- `telemetry_event_v2`

No other topics may be added without explicit approval.

## 5) Implementation Rules

- Every module must use `from __future__ import annotations`.
- Public modules and all package `__init__.py` must define `__all__`.
- All functions/methods require full type annotations.
- Keep module sizes focused and single-responsibility.
- Remove dead code during migration; do not leave deprecated branches.
- No visualization types in canonical contracts — ever.

## 6) Migration Policy

- This repository is in hard-cut migration mode.
- Allowed projects for final architecture: `contracts`, `environment`, `actor`, `auditor`.
- `ingress` and `cartographer` are removed from active architecture.
- Any new code must target v2 contracts and Ghost-Matrix flow only.

## 7) Adapter Isolation Boundary

External data sources (Habitat, Isaac, real-robot feeds, etc.) connect through a
formal `DatasetAdapter` Protocol:

```
DatasetAdapter Protocol
  .adapt(raw_obs, step_id) → dict[str, NDArray]     # canonical arrays
  .reset()                                            # clear frame-diff state
  .metadata → AdapterMetadata                         # az/el bins, LUT info
```

The adapter is the **only** place where axis transposes, depth normalisation,
semantic class remapping, delta-depth computation, and env-dimension insertion
are performed. The result is always `(1, Az, El)` arrays that slot directly into
a `DistanceMatrix` without any changes to the engine.

Adapters live in `environment/backends/` alongside their backend. They never
import from `actor` or `auditor`.

## 8) Validation Commands

Per project:

```bash
uv sync
uv run ruff check .
uv run mypy src/
uv run pytest tests/
```

Repository goal:

```bash
make check-all
```
