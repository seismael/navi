from __future__ import annotations
import numpy as np

from voxel_dag.compiler import canonical_node_hash, compress_to_dag, deduplicate_signatures


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