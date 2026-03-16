> Historical imported reference only. This file preserves external design material and is not canonical Navi policy. For current repository behavior, use `docs/ARCHITECTURE.md`, `docs/TRAINING.md`, `docs/PERFORMANCE.md`, and `AGENTS.md`.

> Historical imported reference only. This file preserves external design material and is not canonical Navi policy. For current repository behavior, use `docs/ARCHITECTURE.md`, `docs/TRAINING.md`, `docs/PERFORMANCE.md`, and `AGENTS.md`.

# DATA_FLOW_AND_SEQUENCE.md — Execution Lifecycle

## 1. Executive Summary

Executing a Reinforcement Learning loop at $>10,000$ SPS requires orchestrating multiple asynchronous event loops, Python's Global Interpreter Lock (GIL), and parallel GPU streams. This document defines the exact order of execution and the synchronization barriers required to prevent race conditions.

---

## 2. The Synchronous Physics Step (REQ/REP)

The primary interaction between the neural network and the C++ physics engine operates in strictly synchronized lock-step to satisfy Markov Decision Process (MDP) constraints.

### 2.1. Execution Sequence

1. **Network Inference (Python):** `toponav-actor` executes the Mamba-2 forward pass and samples continuous actions.
2. **IPC Transmission (ZMQ):** `toponav-actor` packs a `BatchStepRequest` and blocks on the ZeroMQ `REQ` socket.
3. **Physics Integration (Python):** `toponav-env` unmarshals the request, applies velocity vectors to the current actor coordinates, and updates the `origins` and `dirs` PyTorch tensors.
4. **CUDA Kernel Dispatch (C++):** `toponav-env` calls `torch_sdf.cast_rays()`.
* *Critical Mechanism:* The PyBind11 wrapper invokes `pybind11::gil_scoped_release`. This explicitly releases the Python GIL, allowing background Python threads (like the ZMQ listener or telemetry publisher) to operate while the GPU is locked in computation.


5. **Hardware Synchronization (CUDA):** `torch.cuda.synchronize()` is called *immediately* after the C++ function returns. This forces the CPU to wait until the GPU has completely finished writing the distances to VRAM.
6. **Observation Packaging (Python):** The PyTorch tensors are cast to NumPy, packed into a `BatchStepResult`, and transmitted back to `toponav-actor` via the `REP` socket.

---

## 3. Concurrency & Synchronization Rules

### 3.1. The Double-Buffering Contract

The optimization phase (PPO backpropagation) is computationally massive and cannot be allowed to stall the physics engine.

* **Rule:** `toponav-actor` utilizes two isolated Trajectory Buffers (`Buffer A` and `Buffer B`).
* **Execution:** While the background thread executes `.backward()` on `Buffer A`, the main ZMQ thread instantly swaps to `Buffer B` and continues requesting physics steps from `toponav-env`.
* **Barrier:** A `threading.Event` barrier ensures the main thread pauses only if `Buffer B` fills up before `Buffer A` finishes optimization.

### 3.2. PyTorch GPU Streams

If the environment and the neural network reside on the same GPU, they must not overwrite each other's L2 cache allocations concurrently.

* **Rule:** The Mamba-2 forward/backward passes execute on `torch.cuda.default_stream()`.
* **Rule:** The `TorchSDF` raycasting engine must execute on an isolated secondary stream: `torch.cuda.Stream()`.
* **Barrier:** Stream synchronization (`stream.wait_stream()`) must occur precisely before reading the resulting `DistanceMatrix` into the actor's vision transformer.

---

## 4. Asynchronous Telemetry Loop (PUB/SUB)

Human observation (rendering heatmaps or monitoring loss curves) inherently operates at $60$ FPS, while the engine operates at $10,000$ SPS.

### 4.1. Execution Rule (The "Drop" Protocol)

* The `toponav-env` publisher broadcasts `TelemetryEvent` payloads via a ZMQ `PUB` socket using `zmq.NOBLOCK`.
* The `toponav-auditor` UI thread runs independently. If the dashboard is currently rendering frame $N$, and the engine broadcasts frames $N+1$ through $N+50$, the ZMQ buffer silently drops the intermediate frames.
* **Constraint:** The physics engine MUST NEVER await a UI rendering cycle. The simulation throughput remains physically decoupled from human visual tracking.