"""Tests for EpisodicMemory (KNN loop detection)."""

from __future__ import annotations

import numpy as np
import torch

from navi_actor.memory.episodic import EpisodicMemory


def _make_memory(
    dim: int = 128,
    capacity: int = 100,
    exclusion_window: int = 5,
    threshold: float = 0.85,
) -> EpisodicMemory:
    return EpisodicMemory(
        embedding_dim=dim,
        capacity=capacity,
        exclusion_window=exclusion_window,
        similarity_threshold=threshold,
    )


def test_empty_query() -> None:
    """Querying empty memory should return similarity=0, no match."""
    mem = _make_memory()
    vec = np.random.randn(128).astype(np.float32)
    sim, _matched, is_loop = mem.query(vec)
    assert sim == 0.0
    assert _matched is None
    assert is_loop is False


def test_add_and_query_identical() -> None:
    """Adding a vector then querying with the same vector (outside window) → loop."""
    mem = _make_memory(exclusion_window=0, threshold=0.9)
    vec = np.random.randn(128).astype(np.float32)
    mem.add(vec)
    sim, _matched, is_loop = mem.query(vec)
    assert sim > 0.99
    assert is_loop is True
    assert _matched is not None


def test_exclusion_window() -> None:
    """Recent embeddings within the exclusion window should not match."""
    mem = _make_memory(exclusion_window=3, threshold=0.5)
    vec = np.random.randn(128).astype(np.float32)
    # Add 3 vectors — all within exclusion window
    for _ in range(3):
        mem.add(vec)
    sim, _matched, is_loop = mem.query(vec)
    # All stored vectors are within exclusion window → no match
    assert sim == 0.0
    assert is_loop is False


def test_outside_exclusion_window() -> None:
    """Vectors outside the exclusion window should be query-able."""
    mem = _make_memory(exclusion_window=2, threshold=0.5)
    target = np.random.randn(128).astype(np.float32)
    mem.add(target)
    # Add 2 more random vectors to push target outside window
    for _ in range(2):
        mem.add(np.random.randn(128).astype(np.float32))
    sim, _matched, is_loop = mem.query(target)
    assert sim > 0.9
    assert is_loop is True


def test_reset_clears_memory() -> None:
    """reset() should clear all stored embeddings."""
    mem = _make_memory()
    for _ in range(10):
        mem.add(np.random.randn(128).astype(np.float32))
    mem.reset()
    vec = np.random.randn(128).astype(np.float32)
    sim, _matched, is_loop = mem.query(vec)
    assert sim == 0.0
    assert is_loop is False


def test_capacity_eviction() -> None:
    """When capacity is exceeded, oldest entries should be evicted."""
    mem = _make_memory(dim=8, capacity=5, exclusion_window=0, threshold=0.9)
    vecs = [np.eye(8, dtype=np.float32)[i] for i in range(6)]
    for v in vecs:
        mem.add(v)

    assert mem.size == 5

    sim, matched, is_loop = mem.query(vecs[0])
    assert sim < 0.1
    assert matched is not None
    assert is_loop is False


def test_capacity_eviction_overwrites_oldest_entries() -> None:
    """Eviction should retain the newest entries in the fixed-capacity ring."""
    mem = _make_memory(dim=8, capacity=4, exclusion_window=0, threshold=0.9)

    for i in range(8):
        mem.add(np.eye(8, dtype=np.float32)[i])

    assert mem.size == 4

    sim, matched, is_loop = mem.query(np.eye(8, dtype=np.float32)[0])
    assert sim < 0.1
    assert matched is not None
    assert is_loop is False


def test_tensor_batch_query_and_add_stay_vectorized() -> None:
    mem = _make_memory(dim=4, capacity=6, exclusion_window=1, threshold=0.8)
    basis = torch.eye(4, dtype=torch.float32)

    mem.add_batch_tensor(basis)
    similarities, loop_flags = mem.query_batch_tensor(basis)

    assert similarities.shape == (4,)
    assert loop_flags.shape == (4,)
    assert similarities[-1].item() < 0.1
    assert torch.all(loop_flags[:-1])


def test_orthogonal_vectors_no_loop() -> None:
    """Orthogonal vectors should produce low similarity → no loop."""
    mem = _make_memory(exclusion_window=0, threshold=0.85)
    # Create two nearly orthogonal vectors
    v1 = np.zeros(128, dtype=np.float32)
    v1[0] = 1.0
    v2 = np.zeros(128, dtype=np.float32)
    v2[1] = 1.0
    mem.add(v1)
    sim, _, is_loop = mem.query(v2)
    assert sim < 0.1
    assert is_loop is False
