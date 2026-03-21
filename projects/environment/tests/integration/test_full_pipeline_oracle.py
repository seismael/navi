"""Phase 9 — Full Pipeline Oracle Integration Tests.

End-to-end validation from compiled unit box through the entire observation
pipeline, checking correctness at every boundary using analytical expectations.

Requires CUDA.  Skipped gracefully when unavailable.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch

from navi_contracts.testing.oracle_box import (
    analytical_ray_box_distance,
    write_unit_box_obj,
)
from navi_environment.backends.sdfdag_backend import (
    SdfDagBackend,
    build_spherical_ray_directions,
)
from navi_environment.config import EnvironmentConfig
from navi_environment.integration.voxel_dag import probe_sdfdag_runtime

# ── Helpers ──────────────────────────────────────────────────────────

_AZ = 64
_EL = 16
_MAX_DISTANCE = 10.0
_RESOLUTION = 64


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "AGENTS.md").exists() and (parent / "projects").exists():
            return parent
    msg = "Could not resolve repo root"
    raise RuntimeError(msg)


def _load_voxel_dag_compiler() -> Any:
    compiler_path = _repo_root() / "projects" / "voxel-dag" / "voxel_dag" / "compiler.py"
    spec = importlib.util.spec_from_file_location("navi_test_voxel_dag_compiler", compiler_path)
    if spec is None or spec.loader is None:
        msg = f"Failed to load voxel-dag compiler from {compiler_path}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _compile_unit_box_gmdag(output_dir: Path) -> Path:
    """Write unit box .obj → compile to .gmdag at resolution 64."""
    source = output_dir / "unit_box.obj"
    output = output_dir / "unit_box.gmdag"
    write_unit_box_obj(source)

    compiler = _load_voxel_dag_compiler()
    mesh_ingestor = cast("Any", compiler.MeshIngestor)
    compute_dense_sdf = cast("Any", compiler.compute_dense_sdf)
    compress_to_dag = cast("Any", compiler.compress_to_dag)
    write_gmdag = cast("Any", compiler.write_gmdag)

    vertices, indices, bbox_min, bbox_max = mesh_ingestor.load_obj(str(source))
    grid, voxel_size, cube_min = compute_dense_sdf(
        vertices, indices, bbox_min, bbox_max, _RESOLUTION, padding=0.0,
    )
    dag = compress_to_dag(grid, _RESOLUTION)
    write_gmdag(output, dag, _RESOLUTION, cube_min, voxel_size)
    return output


def _create_backend(gmdag_path: Path) -> SdfDagBackend:
    return SdfDagBackend(
        EnvironmentConfig(
            backend="sdfdag",
            gmdag_file=str(gmdag_path),
            n_actors=1,
            azimuth_bins=_AZ,
            elevation_bins=_EL,
            max_distance=_MAX_DISTANCE,
            training_mode=True,
            sdfdag_torch_compile=False,
        )
    )


def _skip_unless_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


# ── Phase 9a: Compile → Load → Observe ──────────────────────────────


class TestEndToEndObservation:
    """Actor placed at box center, verify the full observation at every boundary."""

    @pytest.fixture(autouse=True)
    def _build(self, tmp_path: Path) -> None:
        _skip_unless_cuda()
        gmdag = _compile_unit_box_gmdag(tmp_path)
        status = probe_sdfdag_runtime(gmdag)
        if status.issues:
            pytest.skip("sdfdag runtime: " + "; ".join(status.issues))

        self.backend = _create_backend(gmdag)
        self.device = self.backend._device

        # Place actor at box center (0, 1, 0), yaw=0
        self.backend._spawn_positions[0].copy_(
            torch.tensor([0.0, 1.0, 0.0], device=self.device, dtype=torch.float32)
        )

    def teardown_method(self) -> None:
        if hasattr(self, "backend"):
            self.backend.close()

    def test_observation_shape(self) -> None:
        obs, _dm = self.backend.reset_tensor(episode_id=1, actor_id=0, materialize=True)
        assert tuple(obs.shape) == (3, _AZ, _EL), f"Expected (3,{_AZ},{_EL}), got {obs.shape}"

    def test_depth_in_unit_range(self) -> None:
        obs, _ = self.backend.reset_tensor(episode_id=1, actor_id=0, materialize=True)
        depth = obs[0].cpu().numpy()
        assert np.all(depth >= 0.0), "Depth contains negative values"
        assert np.all(depth <= 1.0), "Depth exceeds 1.0"

    def test_materialized_distance_matrix(self) -> None:
        _, dm = self.backend.reset_tensor(episode_id=1, actor_id=0, materialize=True)
        assert dm is not None
        assert dm.matrix_shape == (_AZ, _EL)
        assert dm.depth.shape == (1, _AZ, _EL)
        assert np.all(np.isfinite(dm.depth))

    def test_center_actor_distances_match_analytical(self) -> None:
        """The raw observation at box center should match analytical box distances
        within the tolerance budget.

        For each azimuth/elevation bin, we compute the expected analytical distance
        from (0,1,0) along the ray direction and compare.
        """
        obs, _ = self.backend.reset_tensor(episode_id=1, actor_id=0, materialize=True)
        depth_normalized = obs[0].cpu().numpy()  # (AZ, EL), normalized [0,1]
        depth_metric = depth_normalized * _MAX_DISTANCE

        dirs = build_spherical_ray_directions(_AZ, _EL)  # (AZ*EL, 3)
        origin = (0.0, 1.0, 0.0)

        voxel_size = 2.0 / _RESOLUTION  # box extent (2m) / resolution
        tol = 5 * voxel_size + 0.01 + 0.05  # 5*voxel + kHitEpsilon + margin

        n_checked = 0
        n_close = 0
        for az in range(_AZ):
            for el in range(_EL):
                d_np = dirs[az * _EL + el]
                d_tuple = (float(d_np[0]), float(d_np[1]), float(d_np[2]))
                expected = analytical_ray_box_distance(origin, d_tuple)
                if expected is None:
                    continue  # ray misses box (shouldn't happen from center, but be safe)
                if expected > _MAX_DISTANCE:
                    continue  # beyond max distance

                actual = float(depth_metric[az, el])
                n_checked += 1
                if abs(actual - expected) < tol:
                    n_close += 1

        assert n_checked > 0, "No rays checked — something is wrong with the oracle"
        match_ratio = n_close / n_checked
        assert match_ratio > 0.7, (
            f"Only {match_ratio:.1%} of {n_checked} rays within tol={tol:.3f}. "
            f"Expected > 70% (accounting for surface voxelization artifacts)."
        )


# ── Phase 9b: Yaw rotation shifts the observation ───────────────────


class TestYawRotationShiftsObservation:
    """Rotating the actor by 90° should shift the depth image by ~N/4 azimuth bins."""

    @pytest.fixture(autouse=True)
    def _build(self, tmp_path: Path) -> None:
        _skip_unless_cuda()
        gmdag = _compile_unit_box_gmdag(tmp_path)
        status = probe_sdfdag_runtime(gmdag)
        if status.issues:
            pytest.skip("sdfdag runtime: " + "; ".join(status.issues))
        self.gmdag = gmdag

    def test_ninety_degree_shift(self) -> None:
        """Verify yaw rotation shifts the observation by the expected azimuth offset.

        Uses an off-center position (0.5, 1.0, 0.0) so +X wall (0.5m away)
        and -X wall (1.5m away) produce different depths, breaking the cube's
        rotational symmetry.

        Rotation convention (from backend):
          world_x = base_x * cos(yaw) + base_z * sin(yaw)
          world_z = -base_x * sin(yaw) + base_z * cos(yaw)

        At yaw=0:  az=N/4 → +X (distance 0.5),  az=3N/4 → -X (distance 1.5)
        At yaw=π/2: az=N/2 → +X (distance 0.5), az=0     → -X (distance 1.5)
        """
        off_center = torch.tensor(
            [0.5, 1.0, 0.0], dtype=torch.float32
        )

        # Observation at yaw=0
        backend_0 = _create_backend(self.gmdag)
        try:
            backend_0._spawn_positions[0].copy_(off_center.to(backend_0._device))
            backend_0._spawn_yaws[0] = 0.0
            obs_0, _ = backend_0.reset_tensor(episode_id=1, actor_id=0, materialize=False)
            depth_0 = obs_0[0].cpu().numpy()
        finally:
            backend_0.close()

        # Observation at yaw=π/2
        backend_90 = _create_backend(self.gmdag)
        try:
            backend_90._spawn_positions[0].copy_(off_center.to(backend_90._device))
            backend_90._spawn_yaws[0] = math.pi / 2
            obs_90, _ = backend_90.reset_tensor(episode_id=2, actor_id=0, materialize=False)
            depth_90 = obs_90[0].cpu().numpy()
        finally:
            backend_90.close()

        mid_el = _EL // 2
        quarter = _AZ // 4  # N/4

        # --- Directional bin checks (off-center breaks symmetry) ---
        # At yaw=0 the +X bin (az=N/4) should be shorter than -X bin (az=3N/4)
        assert depth_0[quarter, mid_el] < depth_0[3 * quarter, mid_el], (
            f"yaw=0: +X depth ({depth_0[quarter, mid_el]:.4f}) "
            f"should be < -X depth ({depth_0[3 * quarter, mid_el]:.4f})"
        )

        # At yaw=π/2 the +X bin moves to az=N/2, -X to az=0
        assert depth_90[2 * quarter, mid_el] < depth_90[0, mid_el], (
            f"yaw=π/2: +X depth ({depth_90[2 * quarter, mid_el]:.4f}) "
            f"should be < -X depth ({depth_90[0, mid_el]:.4f})"
        )

        # Cross-check: same world direction (+X) should produce similar depth
        # regardless of yaw.  yaw=0 az=N/4 and yaw=π/2 az=N/2 both face +X.
        tol = 0.05  # voxelization noise
        d_plus_x_yaw0 = depth_0[quarter, mid_el]
        d_plus_x_yaw90 = depth_90[2 * quarter, mid_el]
        assert abs(d_plus_x_yaw0 - d_plus_x_yaw90) < tol, (
            f"+X depth mismatch across yaws: {d_plus_x_yaw0:.4f} vs {d_plus_x_yaw90:.4f}"
        )


# ── Phase 9c: Forward motion reduces forward distance ───────────────


class TestForwardMotionReducesDistance:
    """Moving toward +X wall should reduce the observed distance in that direction."""

    @pytest.fixture(autouse=True)
    def _build(self, tmp_path: Path) -> None:
        _skip_unless_cuda()
        gmdag = _compile_unit_box_gmdag(tmp_path)
        status = probe_sdfdag_runtime(gmdag)
        if status.issues:
            pytest.skip("sdfdag runtime: " + "; ".join(status.issues))
        self.gmdag = gmdag

    def test_closer_means_smaller_depth(self) -> None:
        # Position 1: at center (0, 1, 0)
        backend = _create_backend(self.gmdag)
        try:
            backend._spawn_positions[0].copy_(
                torch.tensor([0.0, 1.0, 0.0], device=backend._device, dtype=torch.float32)
            )
            backend._spawn_yaws[0] = 0.0
            obs_center, _ = backend.reset_tensor(episode_id=1, actor_id=0, materialize=False)
            depth_center = obs_center[0].cpu().numpy()
        finally:
            backend.close()

        # Position 2: shifted toward +X wall (0.5, 1, 0) — closer to +X wall
        backend2 = _create_backend(self.gmdag)
        try:
            backend2._spawn_positions[0].copy_(
                torch.tensor([0.5, 1.0, 0.0], device=backend2._device, dtype=torch.float32)
            )
            backend2._spawn_yaws[0] = 0.0
            obs_closer, _ = backend2.reset_tensor(episode_id=2, actor_id=0, materialize=False)
            depth_closer = obs_closer[0].cpu().numpy()
        finally:
            backend2.close()

        # At yaw=0, the +X direction corresponds to az=N/4 (quarter turn)
        # Check that the mean depth in that quadrant is lower when closer
        quarter = _AZ // 4
        mid_el = _EL // 2
        # Sample a few bins around the +X direction
        az_range = range(quarter - 2, quarter + 3)
        el_range = range(mid_el - 1, mid_el + 2)

        mean_center = np.mean([depth_center[az % _AZ, el] for az in az_range for el in el_range])
        mean_closer = np.mean([depth_closer[az % _AZ, el] for az in az_range for el in el_range])

        assert mean_closer < mean_center, (
            f"Depth toward +X wall should decrease when closer: "
            f"center={mean_center:.3f}, closer={mean_closer:.3f}"
        )
