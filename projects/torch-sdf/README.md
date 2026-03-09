# torch-sdf: Domain I

The CUDA Sphere-Tracing Extension (Compute Engine). This module exposes $O(1)$ stackless sphere-tracing directly to PyTorch via zero-copy memory bindings using PyBind11 and LibTorch.

---

## 1. Executive Summary

`torch-sdf` is the high-performance runtime core of the TopoNav ecosystem. It bypasses standard graphics pipelines by executing purely mathematical sphere-tracing against a **Ghost-Matrix Directed Acyclic Graph (.gmdag)**. 

By mutating PyTorch VRAM pointers in-place, it achieves theoretical maximum simulation throughput, specifically designed for large-scale multi-agent reinforcement learning.

---

## 2. Key Features

*   **O(1) Stackless Traversal**: Iterative descent through Sparse Voxel Octrees (SVO) without recursive overhead or thread divergence.
*   **Zero-Copy Execution**: Directly reads/writes from PyTorch tensors using raw pointers, eliminating PCIe overhead.
*   **Strict CUDA Backend**: Optimized kernels for NVIDIA GPUs with fail-fast validation when CUDA/backend requirements are not met.
*   **Agnostic Spatial Logic**: Automatically handles arbitrary bounding boxes and resolutions.

---

## 3. Performance Benchmarks (Verified)

| Backend | Rays Per Step | Steps-Per-Second (SPS) | Latency |
| :--- | :--- | :--- | :--- |
| **Native CUDA** | 196,608 | **>10,000** (Target) | <1.0 ms |

---

## 4. Installation & Setup

### Local Deployment
The project relies on being installed into the primary `toponav-env` virtual environment.

```bash
cd torch-sdf
# Ensure your environment variables point to your CUDA installation
# Windows example: $env:CUDA_HOME="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
pip install -e .
```

PowerShell (explicit):
```powershell
$env:CUDA_HOME="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4"
$env:CUDA_PATH=$env:CUDA_HOME
pip install -e .
```

### Build Requirements
*   **Python**: 3.10+
*   **PyTorch**: 2.0+ (CUDA 12.1+ recommended for GPU path)
*   **CUDA Toolkit**: 11.8+ (Required for Native GPU acceleration)

---

## 5. Enabling Native CUDA Acceleration

If your system reports a CUDA-compatible GPU but runtime startup fails, ensure the **NVIDIA CUDA Toolkit** is installed and the `torch_sdf_backend` extension is built correctly.

**Windows Specific Notes:**
1. Python 3.8+ does not automatically search the system `PATH` for DLLs.
2. The package will attempt to read the `CUDA_HOME` or `CUDA_PATH` environment variable to automatically locate and load CUDA dependencies at runtime via `os.add_dll_directory()`.
3. If you encounter an `ImportError: DLL load failed` when importing the backend, ensure `CUDA_PATH` is correctly set to your toolkit installation (e.g., `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4`).

---

## 6. Testing & Verification

The integrity of the kernel is validated via `tests/test_full_pipeline.py`.

*   **Hallway Integrity**: Confirms accurate distance detection in complex long-range environments.
*   **Throughput Benchmark**: Stress-tests the engine to verify SPS performance.

```bash
# Run the full pipeline test
python tests/test_full_pipeline.py
```

---

## 7. Architectural Compliance

This component strictly follows the schemas and memory layouts defined in the TopoNav [**Docs**](../docs/). Any modifications to the bit-packing logic must be mirrored in the `voxel-dag` compiler.
