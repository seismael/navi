import os
import sys

# Windows-specific CUDA DLL loading for Python 3.8+
if sys.platform == "win32":
    cuda_path = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")
    if cuda_path:
        bin_dir = os.path.join(cuda_path, "bin")
        if os.path.isdir(bin_dir):
            try:
                os.add_dll_directory(bin_dir)
            except Exception:
                pass

from .backend import cast_rays

__all__ = ["cast_rays"]
