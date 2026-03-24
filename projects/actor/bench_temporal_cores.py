"""Benchmark: Mamba2 SSD vs GRU vs Mambapy temporal cores.

Measures forward + backward wall time for the full evaluate_sequence path
(the PPO training hot path) across all three temporal cores.
"""
import time
import torch
print("Benchmarking temporal cores on:", torch.cuda.get_device_name(0))
print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}")

from navi_actor.cognitive_policy import CognitiveMambaPolicy

DEVICE = "cuda"
WARMUP = 5
REPEATS = 20
B, T = 4, 64  # Matches PPO minibatch: 4 actors, 64-step sequences
AZ, EL = 256, 48

results = {}
for core_name in ("gru", "mambapy", "mamba2"):
    print(f"\n--- {core_name.upper()} ---")
    policy = CognitiveMambaPolicy(
        embedding_dim=128,
        temporal_core=core_name,
        azimuth_bins=AZ,
        elevation_bins=EL,
    ).to(DEVICE)
    
    params = sum(p.numel() for p in policy.parameters())
    core_params = sum(p.numel() for p in policy.temporal_core.parameters())
    print(f"  Params: total={params:,}, temporal_core={core_params:,}")
    
    obs_seq = torch.randn(B, T, 3, AZ, EL, device=DEVICE)
    actions_seq = torch.randn(B, T, 4, device=DEVICE)
    
    # Warmup
    for _ in range(WARMUP):
        policy.zero_grad()
        lp, v, ent, _, _ = policy.evaluate_sequence(obs_seq, actions_seq)
        loss = -lp.mean() + 0.5 * v.mean()
        loss.backward()
    torch.cuda.synchronize()
    
    # Benchmark
    fwd_times = []
    bwd_times = []
    total_times = []
    
    for _ in range(REPEATS):
        policy.zero_grad()
        torch.cuda.synchronize()
        
        t0 = time.perf_counter()
        lp, v, ent, _, _ = policy.evaluate_sequence(obs_seq, actions_seq)
        loss = -lp.mean() + 0.5 * v.mean()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        
        loss.backward()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        
        fwd_times.append((t1 - t0) * 1000)
        bwd_times.append((t2 - t1) * 1000)
        total_times.append((t2 - t0) * 1000)
    
    import statistics
    fwd_med = statistics.median(fwd_times)
    bwd_med = statistics.median(bwd_times)
    total_med = statistics.median(total_times)
    
    results[core_name] = {
        "fwd_ms": fwd_med,
        "bwd_ms": bwd_med,
        "total_ms": total_med,
    }
    
    print(f"  Forward:  {fwd_med:7.2f} ms (median of {REPEATS})")
    print(f"  Backward: {bwd_med:7.2f} ms (median of {REPEATS})")
    print(f"  Total:    {total_med:7.2f} ms (median of {REPEATS})")
    
    del policy
    torch.cuda.empty_cache()

print("\n" + "=" * 60)
print("SUMMARY (median total ms, lower is better)")
print("=" * 60)
baseline = results["gru"]["total_ms"]
for name, r in results.items():
    speedup = baseline / r["total_ms"]
    print(f"  {name:8s}: {r['total_ms']:7.2f} ms  (fwd={r['fwd_ms']:.1f}, bwd={r['bwd_ms']:.1f})  [{speedup:.2f}x vs GRU]")
