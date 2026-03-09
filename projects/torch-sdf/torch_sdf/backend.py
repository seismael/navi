import torch

try:
    import torch_sdf_backend
    HAS_CUDA_BACKEND = True
except ImportError:
    HAS_CUDA_BACKEND = False

# --- PYTHON API ---

def cast_rays(dag_tensor, origins, dirs, out_distances, out_semantics, max_steps, bbox_min, bbox_max, resolution):
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
    if not dag_tensor.is_cuda:
        raise RuntimeError("dag_tensor must be on CUDA. CPU tensors are not supported.")
    if not origins.is_cuda or not dirs.is_cuda or not out_distances.is_cuda or not out_semantics.is_cuda:
        raise RuntimeError("All ray tensors must be CUDA tensors. CPU tensors are not supported.")

    torch_sdf_backend.cast_rays(
        dag_tensor,
        origins.contiguous(),
        dirs.contiguous(),
        out_distances.contiguous(),
        out_semantics.contiguous(),
        max_steps,
        bbox_min,
        bbox_max,
        resolution
    )
