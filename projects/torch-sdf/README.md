# torch-sdf: Domain I

The CUDA sphere-tracing extension for Navi's canonical `.gmdag` runtime. This
module exposes bounded batched sphere tracing directly to PyTorch through
zero-copy bindings implemented with PyBind11 and LibTorch.

---

## 1. Executive Summary

`torch-sdf` is the low-level compute core for Navi's canonical compiled-scene
path. It bypasses graphics pipelines by executing mathematical sphere tracing
against `.gmdag` assets and writing results directly into preallocated PyTorch
CUDA tensors.

The important production guarantee is bounded batched execution with explicit
runtime limits, not a claim of literal constant-time cost per ray.

---

## 2. Key Features

*   **Bounded Stackless Traversal**: Iterative DAG descent with explicit
	`max_steps`, horizon, and hit-epsilon semantics.
*   **Zero-Copy Execution**: Directly reads/writes PyTorch CUDA tensors using
	raw pointers, avoiding avoidable CPU staging.
*   **Strict CUDA Backend**: Fail-fast validation for CUDA placement, dtype,
	shape, contiguity, and runtime parameters.
*   **Explicit Spatial Bounds**: Bounding boxes and resolution are validated at
	the binding boundary before kernel launch.

---

## 3. Runtime Contract

Canonical inputs and outputs:

* `origins`: contiguous CUDA `float32`, shape `[batch, rays, 3]`
* `dirs`: contiguous CUDA `float32`, shape `[batch, rays, 3]`
* `out_distances`: contiguous CUDA `float32`, shape `[batch, rays]`
* `out_semantics`: contiguous CUDA `int32`, shape `[batch, rays]`

Explicit runtime rules:

* `max_steps` must be a positive integer
* `max_distance` must be finite and positive
* `resolution` must be a positive integer
* `bbox_min` and `bbox_max` must each contain three finite floats
* every `bbox_min[i]` must be strictly less than `bbox_max[i]`

Current kernel semantics:

* a hit is accepted when local clearance falls below the internal hit epsilon
* rays that leave the compiled domain or exceed the configured horizon return a
	miss semantic (`0`)
* misses are bounded by the configured iteration limit and horizon rather than a
	promise of globally correct analytic continuation outside the compiled asset

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
# Run the focused wrapper and full-pipeline tests
python tests/test_full_pipeline.py
```

Focused wrapper validation lives in `tests/test_sphere_tracing.py`.

---

## 7. Architectural Compliance

This component follows the repository-level contracts in `docs/`. Any changes to
bit packing, hit semantics, or tensor-boundary validation must be mirrored in
the `voxel-dag` compiler, the environment integration layer, and their tests in
the same change.
