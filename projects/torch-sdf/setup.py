import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

cpp_src_dir = "cpp_src"


def _ensure_cuda_home() -> None:
    if os.environ.get("CUDA_HOME"):
        os.environ.setdefault("CUDA_PATH", os.environ["CUDA_HOME"])
        return

    if os.environ.get("CUDA_PATH"):
        os.environ["CUDA_HOME"] = os.environ["CUDA_PATH"]
        return

    candidates = [
        Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"),
        Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"),
        Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3"),
        Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1"),
        Path("/usr/local/cuda"),
    ]
    for candidate in candidates:
        if candidate.exists():
            os.environ["CUDA_HOME"] = str(candidate)
            os.environ.setdefault("CUDA_PATH", str(candidate))
            break


_ensure_cuda_home()

setup(
    name='torch_sdf',
    version='0.1.0',
    ext_modules=[
        CUDAExtension(
            name='torch_sdf_backend',
            sources=[
                os.path.join(cpp_src_dir, 'bindings.cpp'),
                os.path.join(cpp_src_dir, 'kernel.cu'),
            ],
            extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3', '-lineinfo']}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
