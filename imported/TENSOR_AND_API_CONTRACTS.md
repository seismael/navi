> Historical imported reference only. This file preserves external design material and is not canonical Navi policy. For current repository behavior, use `docs/CONTRACTS.md`, `docs/SDFDAG_RUNTIME.md`, `docs/VERIFICATION.md`, and `AGENTS.md`.

# TENSOR_AND_API_CONTRACTS.md — Interface Boundaries

## 1. Executive Summary

This document defines the immutable memory contracts required to safely cross the Python-to-C++ and ZeroMQ boundaries. Because PyTorch C++ extensions (`libtorch`) execute raw memory operations on the GPU, violating these contracts will result in instant hardware-level segmentation faults or silent memory corruption.

---

## 2. Domain I: PyTorch C++ Extension Contracts (`TorchSDF`)

The `torch_sdf.cast_rays()` function bypasses the Python interpreter to execute raw CUDA shaders. The incoming PyTorch tensors must strictly adhere to the following memory and dimensional constraints.

### 2.1. Input Tensors

**`origins` Tensor**

* **Purpose:** Defines the absolute $(x, y, z)$ spatial origin of every ray cast by every actor.
* **Shape:** `[Batch, Rays, 3]` (e.g., `[64, 3072, 3]`).
* **Data Type:** `torch.float32`.
* **Device:** `cuda` (must reside on the same GPU index as the DAG memory).
* **Contiguity:** MUST be `.is_contiguous()`. Sliced or transposed tensors must call `.contiguous()` before being passed to C++ to ensure linear memory addressing.

**`dirs` Tensor**

* **Purpose:** Defines the normalized spherical direction vector for every ray.
* **Shape:** `[Batch, Rays, 3]`.
* **Data Type:** `torch.float32`.
* **Constraint:** The Euclidean norm of the inner dimension must exactly equal $1.0$. Un-normalized vectors will mathematically corrupt the Signed Distance evaluation.

### 2.2. Output Tensors (In-Place / Pre-Allocated)

To maintain a zero-copy architecture, output tensors are allocated once in Python during environment initialization and passed by reference to C++.

**`out_distances` Tensor**

* **Shape:** `[Batch, Rays]`.
* **Data Type:** `torch.float32`.
* **Execution:** The CUDA kernel writes the final intersected distance directly to this pointer.

**`out_semantics` Tensor**

* **Shape:** `[Batch, Rays]`.
* **Data Type:** `torch.int32` (Habitat standard).
* **Execution:** The CUDA kernel writes the `uint16_t` semantic payload from the DAG leaf node into this 32-bit array.

### 2.3. C++ Failsafe Protocol (`TORCH_CHECK`)

The PyBind11 layer must enforce these contracts before launching the CUDA grid.

```cpp
TORCH_CHECK(origins.is_cuda(), "origins tensor must reside on CUDA device.");
TORCH_CHECK(origins.is_contiguous(), "origins tensor must be contiguous.");
TORCH_CHECK(origins.dim() == 3 && origins.size(2) == 3, "origins must be [B, R, 3].");

```

---

## 3. Domain II: IPC Network Contracts (`toponav-contracts`)

Data flowing over the ZeroMQ loopback between the simulation server and the RL neural network is strictly typed using standard Python DataClasses and serialized via `msgpack`.

### 3.1. `BatchStepRequest`

The synchronous payload sent from the neural network to the environment server.

```python
@dataclass(frozen=True)
class Action:
    linear_velocity: np.ndarray  # Shape: (1, 3), dtype: float32, Range: [-1.0, 1.0]
    angular_velocity: np.ndarray # Shape: (1, 3), dtype: float32, Range: [-1.0, 1.0]

@dataclass(frozen=True)
class BatchStepRequest:
    step_id: int
    actions: tuple[Action, ...]  # Length exactly equals config.n_actors

```

### 3.2. `BatchStepResult`

The synchronous response returned from the environment server.

```python
@dataclass(frozen=True)
class DistanceMatrix:
    depth: np.ndarray        # Shape: (128, 24), dtype: float32, Range: [0.0, 1.0]
    semantic: np.ndarray     # Shape: (128, 24), dtype: int32
    valid_mask: np.ndarray   # Shape: (128, 24), dtype: bool

@dataclass(frozen=True)
class BatchStepResult:
    step_id: int
    matrices: tuple[DistanceMatrix, ...]
    rewards: tuple[float, ...]
    dones: tuple[bool, ...]

```

---