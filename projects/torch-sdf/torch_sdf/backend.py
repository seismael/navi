from typing import Any
import math

import torch

torch_sdf_backend: Any | None = None
try:
    import torch_sdf_backend as _torch_sdf_backend  # type: ignore[import-not-found]

    torch_sdf_backend = _torch_sdf_backend
    HAS_CUDA_BACKEND = True
except ImportError:
    HAS_CUDA_BACKEND = False

# --- PYTHON API ---

_DAG_DTYPE = torch.int64
_DISTANCE_DTYPE = torch.float32
_SEMANTIC_DTYPE = torch.int32
_HIT_EPSILON = 0.01
_DIRECTION_NORM_EPS = 1e-4


def _shape_tuple(tensor: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in tensor.shape)


def _validate_cuda_tensor(
    tensor: Any,
    *,
    name: str,
    dtype: torch.dtype,
    shape: tuple[int, ...] | None = None,
) -> None:
    if not tensor.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor. CPU tensors are not supported.")
    if tensor.dtype != dtype:
        raise RuntimeError(f"{name} must have dtype {dtype}; got {tensor.dtype}.")
    if not tensor.is_contiguous():
        raise RuntimeError(f"{name} must be contiguous.")
    if shape is not None and _shape_tuple(tensor) != shape:
        raise RuntimeError(f"{name} must have shape {shape}; got {_shape_tuple(tensor)}.")


def _validate_cast_rays_inputs(
    dag_tensor: Any,
    origins: Any,
    dirs: Any,
    out_distances: Any,
    out_semantics: Any,
) -> None:
    if not dag_tensor.is_cuda:
        raise RuntimeError("dag_tensor must be on CUDA. CPU tensors are not supported.")
    if dag_tensor.dtype != _DAG_DTYPE:
        raise RuntimeError(f"dag_tensor must have dtype {_DAG_DTYPE}; got {dag_tensor.dtype}.")
    if not dag_tensor.is_contiguous():
        raise RuntimeError("dag_tensor must be contiguous.")

    if origins.dim() != 3 or origins.size(2) != 3:
        raise RuntimeError(f"origins must have shape [batch, rays, 3]; got {_shape_tuple(origins)}.")
    if dirs.dim() != 3 or dirs.size(2) != 3:
        raise RuntimeError(f"dirs must have shape [batch, rays, 3]; got {_shape_tuple(dirs)}.")

    batch = int(origins.size(0))
    rays = int(origins.size(1))
    if _shape_tuple(dirs) != (batch, rays, 3):
        raise RuntimeError(
            f"dirs must match origins shape {(batch, rays, 3)}; got {_shape_tuple(dirs)}."
        )

    _validate_cuda_tensor(origins, name="origins", dtype=_DISTANCE_DTYPE)
    _validate_cuda_tensor(dirs, name="dirs", dtype=_DISTANCE_DTYPE)
    _validate_cuda_tensor(
        out_distances,
        name="out_distances",
        dtype=_DISTANCE_DTYPE,
        shape=(batch, rays),
    )
    _validate_cuda_tensor(
        out_semantics,
        name="out_semantics",
        dtype=_SEMANTIC_DTYPE,
        shape=(batch, rays),
    )
    if torch.is_tensor(dirs):
        _validate_direction_norms(dirs)


def _validate_direction_norms(dirs: torch.Tensor, *, eps: float = _DIRECTION_NORM_EPS) -> None:
    norms = torch.linalg.vector_norm(dirs, dim=-1)
    if bool((~torch.isfinite(norms)).any().item()):
        raise RuntimeError("dirs must contain only finite direction vectors.")
    max_error = float(torch.abs(norms - 1.0).amax().detach().cpu().item())
    if max_error > eps:
        raise RuntimeError(
            f"dirs must be normalized within tolerance {eps}; max error was {max_error:.6g}."
        )


def _validate_bounds(bbox_min: list[float], bbox_max: list[float]) -> None:
    if len(bbox_min) != 3 or len(bbox_max) != 3:
        raise RuntimeError("bbox_min and bbox_max must each contain exactly 3 floats.")
    for index, (lower, upper) in enumerate(zip(bbox_min, bbox_max, strict=False)):
        if not math.isfinite(lower) or not math.isfinite(upper):
            raise RuntimeError("bbox_min and bbox_max must contain only finite floats.")
        if lower >= upper:
            raise RuntimeError(
                f"bbox_min[{index}] must be strictly less than bbox_max[{index}]; got {lower} >= {upper}."
            )


def _validate_runtime_parameters(
    *,
    max_steps: int,
    max_distance: float,
    bbox_min: list[float],
    bbox_max: list[float],
    resolution: int,
) -> None:
    if max_steps <= 0:
        raise RuntimeError(f"max_steps must be a positive integer; got {max_steps}.")
    if not math.isfinite(max_distance) or max_distance <= 0.0:
        raise RuntimeError(f"max_distance must be a finite positive float; got {max_distance}.")
    if resolution <= 0:
        raise RuntimeError(f"resolution must be a positive integer; got {resolution}.")
    _validate_bounds(bbox_min, bbox_max)


def cast_rays(
    dag_tensor: Any,
    origins: Any,
    dirs: Any,
    out_distances: Any,
    out_semantics: Any,
    max_steps: int,
    max_distance: float,
    bbox_min: list[float],
    bbox_max: list[float],
    resolution: int,
) -> None:
    """
    Top-level API for TopoNav Sphere Tracing.
    Strict CUDA-only path. CPU fallback is intentionally disabled.
    """
    if not HAS_CUDA_BACKEND:
        raise RuntimeError(
            "torch_sdf_backend is not available. Build/install the CUDA extension in torch-sdf before running TopoNav."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. TopoNav requires CUDA and does not support CPU fallback.")
    _validate_cast_rays_inputs(dag_tensor, origins, dirs, out_distances, out_semantics)
    _validate_runtime_parameters(
        max_steps=max_steps,
        max_distance=max_distance,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        resolution=resolution,
    )
    backend = torch_sdf_backend
    if backend is None:
        raise RuntimeError("torch_sdf_backend import unexpectedly resolved to None.")

    backend.cast_rays(
        dag_tensor,
        origins,
        dirs,
        out_distances,
        out_semantics,
        max_steps,
        max_distance,
        bbox_min,
        bbox_max,
        resolution,
    )
