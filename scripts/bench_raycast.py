"""Quick benchmark: measure raycast + step time at current resolution."""
from __future__ import annotations

import time

import numpy as np

from navi_contracts import Action
from navi_environment.backends.mesh_backend import MeshSceneBackend
from navi_environment.config import EnvironmentConfig

SCENE = r"c:\dev\projects\navi\data\scenes\sample_apartment.glb"

cfg = EnvironmentConfig(
    backend="mesh", habitat_scene=SCENE, max_distance=15.0,
)
b = MeshSceneBackend(cfg)
b.reset(0)

act = Action(
    env_ids=np.array([0], dtype=np.int32),
    linear_velocity=np.array([[0.5, 0, 0]], dtype=np.float32),
    angular_velocity=np.array([[0, 0, 0.1]], dtype=np.float32),
    policy_id="bench",
    step_id=0,
    timestamp=0,
)

N = 30

# Full step
t0 = time.perf_counter()
for i in range(N):
    b.step(act, i)
full = (time.perf_counter() - t0) / N * 1000

# Raycast only
t0 = time.perf_counter()
for i in range(N):
    b._cast_rays()  # noqa: SLF001
ray_only = (time.perf_counter() - t0) / N * 1000

print(f"Resolution: {b._az_bins}x{b._el_bins} = {b._az_bins * b._el_bins} rays")
print(f"Full step: {full:.1f} ms  => {1000/full:.1f} steps/s")
print(f"Raycast only: {ray_only:.1f} ms")
print(f"Overhead+physics+delta: {full - ray_only:.1f} ms")

b.close()
