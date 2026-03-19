"""Python surface for the voxel-dag compiler domain.

This package provides the minimal compiler/test API used by the environment's
native integration tests and by `projects/torch-sdf` pipeline tests.
"""

from voxel_dag.compiler import (
    MeshIngestor,
    canonical_node_hash,
    compress_to_dag,
    compute_dense_sdf,
    deduplicate_signatures,
    main,
    write_gmdag,
)

__all__ = [
    "MeshIngestor",
    "canonical_node_hash",
    "compress_to_dag",
    "compute_dense_sdf",
    "deduplicate_signatures",
    "main",
    "write_gmdag",
]
