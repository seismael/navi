from __future__ import annotations
import numpy as np
import pytest

from voxel_dag.compiler import canonical_node_hash, compress_to_dag, deduplicate_signatures


def _decode_leaf_distance(node_word: np.uint64) -> float:
    bits = np.array([int(node_word) & 0xFFFF], dtype=np.uint16)
    return float(bits.view(np.float16)[0])


def test_canonical_node_hash_is_deterministic() -> None:
    payload = b"ghost-matrix-node"

    first = canonical_node_hash(payload, seed=0)
    second = canonical_node_hash(payload, seed=0)

    assert first == second


def test_canonical_node_hash_changes_with_payload() -> None:
    first = canonical_node_hash(b"node-a", seed=0)
    second = canonical_node_hash(b"node-b", seed=0)

    assert first != second


def test_deduplicate_signatures_reuses_exact_duplicates() -> None:
    unique, remap = deduplicate_signatures([b"leaf-a", b"leaf-a", b"leaf-b"], seed=0)

    assert unique == [b"leaf-a", b"leaf-b"]
    assert remap == [0, 0, 1]


def test_deduplicate_signatures_uses_structural_fallback_on_hash_collision() -> None:
    def constant_hash(_payload: bytes, _seed: int) -> int:
        return 7

    unique, remap = deduplicate_signatures(
        [b"leaf-a", b"leaf-b", b"leaf-a"],
        seed=0,
        hash_fn=constant_hash,
    )

    assert unique == [b"leaf-a", b"leaf-b"]
    assert remap == [0, 1, 0]


def test_compress_to_dag_emits_deduplicated_leaf_pool() -> None:
    grid = np.ones((4, 4, 4), dtype=np.float32)
    grid[0:2, :, :] = 2.0

    dag = compress_to_dag(grid, 4)

    assert dag.shape == (11,)
    assert int(dag[0] >> 63) == 0
    child_indices = {int(value) for value in dag[1:9].tolist()}
    assert len(child_indices) == 2


def test_compress_to_dag_preserves_octant_child_order() -> None:
    grid = np.zeros((2, 2, 2), dtype=np.float32)
    expected_distances: list[float] = []
    for octant in range(8):
        distance = 0.5 + octant
        z_index = 1 if octant & 4 else 0
        y_index = 1 if octant & 2 else 0
        x_index = 1 if octant & 1 else 0
        grid[z_index, y_index, x_index] = distance
        expected_distances.append(distance)

    dag = compress_to_dag(grid, 2)

    assert int(dag[0] >> 63) == 0
    child_indices = [int(value) for value in dag[1:9].tolist()]
    decoded_distances = [_decode_leaf_distance(dag[child_index]) for child_index in child_indices]

    assert decoded_distances == pytest.approx(expected_distances, rel=0.0, abs=1e-3)