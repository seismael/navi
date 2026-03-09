import torch
import sys
import platform

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"Platform: {platform.platform()}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    print(f"Device Name: {device_name}")
    print(f"Device Capability: sm_{capability[0]}{capability[1]}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"cuDNN Version: {torch.backends.cudnn.version()}")

    # Validate practical kernel execution, not just CUDA runtime visibility.
    try:
        x = torch.randn(16, 16, device="cuda")
        y = torch.mm(x, x)
        torch.cuda.synchronize()
        print(f"CUDA Kernel Check: OK ({y.shape[0]}x{y.shape[1]})")
    except Exception as exc:
        print(f"CUDA Kernel Check: FAILED ({exc})")
        print("WARNING: CUDA runtime is present, but this GPU/torch build cannot execute kernels.")
        print("Use a PyTorch build that supports your GPU compute capability, or upgrade GPU.")
        sys.exit(2)
else:
    print("WARNING: CUDA is NOT available. Training will be slow.")
    print("Suggested fix (Windows first):")
    print("  powershell -ExecutionPolicy Bypass -File ./scripts/setup-actor-cuda.ps1")
    print("Suggested fix (Linux/WSL2 later):")
    print("  bash ./scripts/setup-actor-cuda.sh")
    sys.exit(1)
