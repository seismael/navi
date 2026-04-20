# Architecture Notes

## Distance Matrix Parity — Dual-Pipeline Architecture

The key architectural insight: **Deployment Mathematics** (physical sensor noise)
and **Training Mathematics** (absolute simulation throughput) must be decoupled.
The Actor's brain is completely blind to *how* the distance was calculated; it
only knows the mathematical shape of the world around it.

This dual-pipeline strategy is formally documented in
[docs/SIM_TO_REAL_PARITY.md](docs/SIM_TO_REAL_PARITY.md).

**Summary:**
- **Pipeline A (Training):** `.gmdag` CUDA sphere tracing → `DistanceMatrix(B, 256, 48)` — deterministic, ~50 SPS (Mamba-2) on MX150.
- **Pipeline B (Deployment):** Sensor fusion → Bayesian log-odds grid → JFA SDF synthesis → `DistanceMatrix(1, 256, 48)` — planned, not yet implemented.
- Both pipelines converge on the identical `DistanceMatrix` tensor contract, making the actor's sacred cognitive pipeline fully pipeline-agnostic.