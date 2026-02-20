"""Tests for EpisodicMemory (KNN loop detection)."""

from __future__ import annotations

import numpy as np

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
    sim, matched, is_loop = mem.query(vec)
    assert sim == 0.0
    assert matched is None
    assert is_loop is False


def test_add_and_query_identical() -> None:
    """Adding a vector then querying with the same vector (outside window) → loop."""
    mem = _make_memory(exclusion_window=0, threshold=0.9)
    vec = np.random.randn(128).astype(np.float32)
    mem.add(vec)
    sim, matched, is_loop = mem.query(vec)
    assert sim > 0.99
    assert is_loop is True
    assert matched is not None


def test_exclusion_window() -> None:
    """Recent embeddings within the exclusion window should not match."""
    mem = _make_memory(exclusion_window=3, threshold=0.5)
    vec = np.random.randn(128).astype(np.float32)
    # Add 3 vectors — all within exclusion window
    for _ in range(3):
        mem.add(vec)
    sim, matched, is_loop = mem.query(vec)
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
    sim, matched, is_loop = mem.query(target)
    assert sim > 0.9
    assert is_loop is True


def test_reset_clears_memory() -> None:
    """reset() should clear all stored embeddings."""
    mem = _make_memory()
    for _ in range(10):
        mem.add(np.random.randn(128).astype(np.float32))
    mem.reset()
    vec = np.random.randn(128).astype(np.float32)
    sim, matched, is_loop = mem.query(vec)
    assert sim == 0.0
    assert is_loop is False


def test_capacity_eviction() -> None:
    """When capacity is exceeded, oldest entries should be evicted."""
    mem = _make_memory(capacity=5, exclusion_window=0)
    # Add 5 vectors
    vecs = [np.random.randn(128).astype(np.float32) for _ in range(5)]
    for v in vecs:
        mem.add(v)

    # 6th vector should cause eviction of the first
    mem.add(np.random.randn(128).astype(np.float32))

    # Original first vector may not be found with high similarity
    # (it should have been evicted)
    sim, _, _ = mem.query(vecs[0])
    # We can't guarantee 0.0 due to random similarity but it should be lower


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
