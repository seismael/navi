"""Phase 7 — Post-Processing Oracle Tests.

Validates the transformation from raw CUDA sphere-trace output to the
normalized observation tensor and DistanceMatrix, using synthetic data
with known expected values.

No CUDA needed — all tensors are CPU for these unit tests.
"""

from __future__ import annotations

import math

import numpy as np
import torch

# ── Inline post-processing reproducing the backend logic ─────────────
# (Avoids importing the full backend which requires CUDA at import time)

def _postprocess(
    out_distances: torch.Tensor,
    out_semantics: torch.Tensor,
    az_bins: int,
    el_bins: int,
    max_distance: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Minimal post-processing matching sdfdag_backend._postprocess_cast_outputs_tensor."""
    actor_count = int(out_distances.shape[0])
    metric = out_distances.reshape(actor_count, az_bins, el_bins)
    semantic = out_semantics.reshape(actor_count, az_bins, el_bins).to(torch.int32)
    valid = torch.isfinite(metric) & (metric <= max_distance)
    clamped = torch.where(valid, metric, torch.full_like(metric, max_distance))
    log_denom = math.log1p(max_distance)
    depth = (torch.log1p(clamped) / log_denom).clamp(0.0, 1.0)
    return depth, valid, semantic


# ── Tests ────────────────────────────────────────────────────────────

class TestNormalization:
    """Synthetic distances → expected normalized depth values."""

    MAX_DIST = 100.0

    def test_known_values(self) -> None:
        raw_distances = torch.tensor([[0.5, 1.0, 5.0, 100.0, float("inf"), float("nan")]])
        raw_semantics = torch.tensor([[1, 1, 1, 1, 0, 0]], dtype=torch.int32)

        depth, _valid, _sem = _postprocess(raw_distances, raw_semantics, 6, 1, self.MAX_DIST)

        log_denom = math.log1p(100.0)
        expected_depth = [
            math.log1p(0.5) / log_denom,
            math.log1p(1.0) / log_denom,
            math.log1p(5.0) / log_denom,
            1.0,  # log1p(100) / log1p(100)
            1.0,  # invalid → clamped to max_distance
            1.0,  # invalid → clamped to max_distance
        ]
        for i, exp in enumerate(expected_depth):
            assert abs(float(depth[0, i, 0]) - exp) < 1e-6, (
                f"depth[{i}]: expected {exp:.6f}, got {float(depth[0, i, 0]):.6f}"
            )

    def test_zero_distance(self) -> None:
        raw = torch.tensor([[0.0]])
        sem = torch.tensor([[1]], dtype=torch.int32)
        depth, _valid, _ = _postprocess(raw, sem, 1, 1, 100.0)
        assert float(depth[0, 0, 0]) == 0.0


class TestValidMask:
    """Valid mask must be True for finite distances ≤ max_distance."""

    def test_known_mask(self) -> None:
        raw = torch.tensor([[0.5, 1.0, 5.0, 30.0, float("inf"), float("nan")]])
        sem = torch.tensor([[1, 1, 1, 1, 0, 0]], dtype=torch.int32)
        _, valid, _ = _postprocess(raw, sem, 6, 1, 30.0)

        expected = [True, True, True, True, False, False]
        for i, exp in enumerate(expected):
            assert bool(valid[0, i, 0]) == exp, f"valid[{i}]: expected {exp}"

    def test_just_over_max_is_invalid(self) -> None:
        raw = torch.tensor([[100.001]])
        sem = torch.tensor([[1]], dtype=torch.int32)
        _, valid, _ = _postprocess(raw, sem, 1, 1, 100.0)
        assert not bool(valid[0, 0, 0])

    def test_exactly_max_is_valid(self) -> None:
        raw = torch.tensor([[100.0]])
        sem = torch.tensor([[1]], dtype=torch.int32)
        _, valid, _ = _postprocess(raw, sem, 1, 1, 100.0)
        assert bool(valid[0, 0, 0])

    def test_finite_distance_with_no_hit_is_still_valid(self) -> None:
        """Non-converged ray (semantic=0) with finite distance in range is valid.

        The SDF-guided march accumulates a useful approximate depth even
        when the tracer does not fully converge."""
        raw = torch.tensor([[15.0]])
        sem = torch.tensor([[0]], dtype=torch.int32)
        _, valid, _ = _postprocess(raw, sem, 1, 1, 100.0)
        assert bool(valid[0, 0, 0])


class TestReshapeFlatToGrid:
    """Unique per-ray values must survive flat → grid reshape correctly."""

    AZ = 8
    EL = 4

    def test_unique_values_roundtrip(self) -> None:
        n = self.AZ * self.EL
        max_dist = float(n + 1)
        raw = torch.arange(n, dtype=torch.float32).unsqueeze(0)  # [1, 32]
        sem = torch.ones((1, n), dtype=torch.int32)
        depth, _, _ = _postprocess(raw, sem, self.AZ, self.EL, max_dist)

        log_denom = math.log1p(max_dist)
        grid = depth[0]  # [AZ, EL]
        for az in range(self.AZ):
            for el in range(self.EL):
                flat_idx = az * self.EL + el
                expected = math.log1p(flat_idx) / log_denom
                actual = float(grid[az, el])
                assert abs(actual - expected) < 1e-6, (
                    f"grid[{az},{el}] = {actual:.6f}, expected {expected:.6f} (flat_idx={flat_idx})"
                )


class TestDeltaDepthComputation:
    """Delta-depth = obs_new - obs_prev."""

    def test_delta_is_difference(self) -> None:
        prev = torch.tensor([[[0.2, 0.3], [0.4, 0.5]]])  # [1, 2, 2]
        curr = torch.tensor([[[0.3, 0.3], [0.5, 0.4]]])
        delta = curr - prev
        expected = torch.tensor([[[0.1, 0.0], [0.1, -0.1]]])
        np.testing.assert_allclose(delta.numpy(), expected.numpy(), atol=1e-6)

    def test_first_frame_delta_is_zero(self) -> None:
        """When prev is all zeros (uninitialized), delta should be zeroed."""
        prev = torch.zeros(1, 4, 4)
        curr = torch.full((1, 4, 4), 0.5)
        # Backend logic: delta = curr - prev, then overridden to zero if prev was uninitialized
        uninitialized = prev.abs().sum(dim=(1, 2)) < 1e-6
        delta = curr - prev
        delta = torch.where(uninitialized.view(-1, 1, 1), torch.zeros_like(delta), delta)
        assert float(delta.abs().sum()) == 0.0
