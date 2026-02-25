"""Quick benchmark: measure raycast + step time at different resolutions.

Usage:
    uv run --project projects/environment scripts/bench_raycast.py
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from navi_contracts import Action
from navi_environment.backends.mesh_backend import MeshSceneBackend
from navi_environment.config import EnvironmentConfig

# Use relative path for robustness
SCENE = str(Path(__file__).parent.parent / "data" / "scenes" / "sample_apartment.glb")

RESOLUTIONS = [
    (64, 12),
    (128, 24),   # Standard
    (256, 48),
]

print(f"Benchmarking scene: {SCENE}")
print(f"{'Resolution':<15} | {'Actors':<6} | {'Rays':<8} | {'Step (ms)':<10} | {'SPS':<8} | {'Raycast (ms)':<12} | {'Overhead (ms)':<12}")
print("-" * 90)

for az, el in RESOLUTIONS:
    for n_actors in [1, 4]:
        cfg = EnvironmentConfig(
            backend="mesh",
            habitat_scene=SCENE,
            max_distance=15.0,
            azimuth_bins=az,
            elevation_bins=el,
            n_actors=n_actors,
        )
        
        try:
            b = MeshSceneBackend(cfg)
            # Reset all actors
            for i in range(n_actors):
                b.reset(0, actor_id=i)

            # Create batched actions
            actions = []
            for i in range(n_actors):
                actions.append(Action(
                    env_ids=np.array([i], dtype=np.int32),
                    linear_velocity=np.array([[0.5, 0, 0]], dtype=np.float32),
                    angular_velocity=np.array([[0, 0, 0.1]], dtype=np.float32),
                    policy_id="bench",
                    step_id=0,
                    timestamp=0,
                ))
            actions_tuple = tuple(actions)

            N = 20
            
            # Warmup
            for _ in range(5):
                b.batch_step(actions_tuple, 0)

            # Full step (Batched)
            t0 = time.perf_counter()
            for i in range(N):
                b.batch_step(actions_tuple, i)
            full = (time.perf_counter() - t0) / N * 1000

            # Raycast only (Batched logic manual simulation for timing)
            # MeshSceneBackend doesn't expose a public batched_raycast method easily separable
            # without duplicating logic, but we can infer it from single actor timing x N or
            # better, just trust the full step time for now as 'SPS' is the metric.
            # But let's try to be consistent with previous metric:
            
            # For raycast-only benchmark, we'll just run the internal logic if possible, 
            # or just report N/A if it's too complex to extract.
            # actually let's just approximate by running the single-actor _cast_rays loop N times
            # THIS IS NOT ACCURATE for batching gain measurement.
            # So let's skip "Raycast Only" column for batched, or mark it.
            
            ray_only = 0.0 
            
            # Calculate Total Rays per step
            rays = az * el * n_actors
            sps = 1000 / full if full > 0 else 0

            print(f"{az}x{el:<11} | {n_actors:<6} | {rays:<8} | {full:<10.2f} | {sps:<8.1f} | {'N/A':<12} | {'N/A':<12}")

            b.close()
            
        except Exception as e:
            print(f"{az}x{el:<11} | {n_actors:<6} | FAILED: {e}")
