"""Isolated temporal core benchmark — measures ONLY the temporal core forward+backward.

This removes the RayViT encoder and actor-critic heads from the measurement
to give a true apples-to-apples comparison of the sequence modeling backends.
"""
import time
import statistics
import torch
print("Device:", torch.cuda.get_device_name(0))

from navi_actor.gru_core import GRUTemporalCore
from navi_actor.mambapy_core import MambapyTemporalCore
from navi_actor.mamba2_core import Mamba2SSDTemporalCore

DEVICE = "cuda"
WARMUP = 10
REPEATS = 50
D = 128

configs = [
    ("B=4, T=64 (PPO minibatch)", 4, 64),
    ("B=4, T=128", 4, 128),
    ("B=4, T=256 (full rollout)", 4, 256),
    ("B=1, T=256", 1, 256),
]

def bench_core(core, name, B, T):
    z = torch.randn(B, T, D, device=DEVICE, requires_grad=False)
    aux = torch.randn(B, T, 3, device=DEVICE, requires_grad=False)
    
    # Warmup
    for _ in range(WARMUP):
        core.zero_grad()
        out, _ = core(z, aux_tensor=aux)
        out.sum().backward()
    torch.cuda.synchronize()
    
    fwd_times = []
    bwd_times = []
    
    for _ in range(REPEATS):
        core.zero_grad()
        torch.cuda.synchronize()
        
        t0 = time.perf_counter()
        out, _ = core(z, aux_tensor=aux)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        
        out.sum().backward()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        
        fwd_times.append((t1 - t0) * 1000)
        bwd_times.append((t2 - t1) * 1000)
    
    fwd = statistics.median(fwd_times)
    bwd = statistics.median(bwd_times)
    return fwd, bwd, fwd + bwd

cores = {
    "gru": lambda: GRUTemporalCore(d_model=D).to(DEVICE),
    "mambapy": lambda: MambapyTemporalCore(d_model=D).to(DEVICE),
    "mamba2": lambda: Mamba2SSDTemporalCore(d_model=D).to(DEVICE),
    "mamba2_c32": lambda: Mamba2SSDTemporalCore(d_model=D, chunk_size=32).to(DEVICE),
    "mamba2_c128": lambda: Mamba2SSDTemporalCore(d_model=D, chunk_size=128).to(DEVICE),
}

for cfg_name, B, T in configs:
    print(f"\n{'='*60}")
    print(f"Config: {cfg_name}")
    print(f"{'='*60}")
    
    results = {}
    for name, factory in cores.items():
        core = factory()
        fwd, bwd, total = bench_core(core, name, B, T)
        results[name] = (fwd, bwd, total)
        del core
        torch.cuda.empty_cache()
    
    baseline = results["gru"][2]
    for name, (fwd, bwd, total) in results.items():
        speedup = baseline / total
        print(f"  {name:12s}: total={total:7.2f}ms  (fwd={fwd:.2f}, bwd={bwd:.2f})  [{speedup:.2f}x vs GRU]")

# Memory usage comparison
print(f"\n{'='*60}")
print("Memory comparison (B=4, T=256)")
print(f"{'='*60}")
for name, factory in [("gru", cores["gru"]), ("mamba2", cores["mamba2"])]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    core = factory()
    z = torch.randn(4, 256, D, device=DEVICE)
    aux = torch.randn(4, 256, 3, device=DEVICE)
    core.zero_grad()
    out, _ = core(z, aux_tensor=aux)
    out.sum().backward()
    torch.cuda.synchronize()
    peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    print(f"  {name:12s}: peak CUDA memory = {peak_mb:.1f} MB")
    del core, z, aux, out
    torch.cuda.empty_cache()
