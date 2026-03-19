"""Non-parametric episodic memory using tensor-native cosine similarity."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from torch import Tensor

__all__: list[str] = ["EpisodicMemory"]

_LOGGER = logging.getLogger(__name__)


class EpisodicMemory:
    """KNN-like episodic memory that detects revisited states.

    Canonical training interacts with this store through tensor-native batch
    query/add methods so embeddings stay on the same device as the policy.
    Thin NumPy wrappers remain for tests and non-hot-path diagnostics.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        capacity: int = 10000,
        exclusion_window: int = 50,
        similarity_threshold: float = 0.85,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.capacity = capacity
        self.exclusion_window = exclusion_window
        self.similarity_threshold = similarity_threshold

        self._storage: Tensor | None = None
        self._size = 0
        self._next_index = 0
        _LOGGER.info("EpisodicMemory: using tensor-native cosine store")

    def reset(self) -> None:
        """Clear all stored embeddings."""
        self._size = 0
        self._next_index = 0

    def add(self, embedding: np.ndarray[Any, Any]) -> None:
        """Add one embedding via the non-hot-path NumPy wrapper."""
        tensor = torch.from_numpy(np.asarray(embedding, dtype=np.float32)).reshape(1, -1)
        self.add_batch_tensor(tensor)

    def query(
        self,
        embedding: np.ndarray[Any, Any],
    ) -> tuple[float, np.ndarray[Any, Any] | None, bool]:
        """Query one embedding via the non-hot-path NumPy wrapper."""
        tensor = torch.from_numpy(np.asarray(embedding, dtype=np.float32)).reshape(1, -1)
        similarities, indices = self._query_batch_indices(tensor)
        return self._format_query_result(
            float(similarities[0].detach().to(device="cpu")),
            int(indices[0].detach().to(device="cpu")),
        )

    def add_batch(self, embeddings: np.ndarray[Any, Any]) -> None:
        """Add a batch of embeddings via the non-hot-path NumPy wrapper."""
        tensor = torch.from_numpy(np.asarray(embeddings, dtype=np.float32))
        self.add_batch_tensor(tensor)

    def query_batch(
        self,
        embeddings: np.ndarray[Any, Any],
    ) -> list[tuple[float, np.ndarray[Any, Any] | None, bool]]:
        """Query a batch of embeddings via the non-hot-path NumPy wrapper."""
        tensor = torch.from_numpy(np.asarray(embeddings, dtype=np.float32))
        similarities, indices = self._query_batch_indices(tensor)
        results: list[tuple[float, np.ndarray[Any, Any] | None, bool]] = []
        sims_cpu = similarities.detach().to(device="cpu")
        idx_cpu = indices.detach().to(device="cpu")
        for row in range(int(sims_cpu.shape[0])):
            results.append(self._format_query_result(float(sims_cpu[row]), int(idx_cpu[row])))
        return results

    def add_batch_tensor(self, embeddings: Tensor) -> None:
        """Add normalized embeddings on the caller's device."""
        vecs = self.normalize_batch_tensor(embeddings)
        self.add_normalized_batch_tensor(vecs)

    def add_normalized_batch_tensor(self, normalized_embeddings: Tensor) -> None:
        """Add a pre-normalized embedding batch without recomputing norms."""
        vecs = self._coerce_prepared_tensor(normalized_embeddings)
        if vecs.numel() == 0:
            return

        storage = self._ensure_storage(vecs.device)
        vecs = vecs.to(device=storage.device, dtype=torch.float32)
        batch_size = int(vecs.shape[0])

        if batch_size >= self.capacity:
            storage.copy_(vecs[-self.capacity :])
            self._size = self.capacity
            self._next_index = 0
            return

        first = min(batch_size, self.capacity - self._next_index)
        storage[self._next_index : self._next_index + first].copy_(vecs[:first])
        remaining = batch_size - first
        if remaining > 0:
            storage[:remaining].copy_(vecs[first:])

        self._size = min(self.capacity, self._size + batch_size)
        self._next_index = (self._next_index + batch_size) % self.capacity

    def query_batch_tensor(self, embeddings: Tensor) -> tuple[Tensor, Tensor]:
        """Query a batch of embeddings on-device.

        Returns:
            similarities: best cosine similarity per query.
            is_loops: thresholded loop-detection flags.
        """
        vecs = self.normalize_batch_tensor(embeddings)
        similarities, indices = self._query_prepared_batch_indices(vecs)
        valid = indices >= 0
        loops = valid & (similarities >= self.similarity_threshold)
        return similarities, loops

    def query_normalized_batch_tensor(
        self, normalized_embeddings: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Query a pre-normalized embedding batch without recomputing norms."""
        similarities, indices = self._query_prepared_batch_indices(normalized_embeddings)
        valid = indices >= 0
        loops = valid & (similarities >= self.similarity_threshold)
        return similarities, loops

    def normalize_batch_tensor(self, embeddings: Tensor) -> Tensor:
        """Normalize one embedding batch for reuse across query and add."""
        return self._normalize_tensor(embeddings)

    def _query_batch_indices(self, embeddings: Tensor) -> tuple[Tensor, Tensor]:
        vecs = self.normalize_batch_tensor(embeddings)
        return self._query_prepared_batch_indices(vecs)

    def _query_prepared_batch_indices(
        self, normalized_embeddings: Tensor
    ) -> tuple[Tensor, Tensor]:
        vecs = self._coerce_prepared_tensor(normalized_embeddings)
        if vecs.numel() == 0:
            empty = torch.zeros((0,), dtype=torch.float32, device=vecs.device)
            return empty, empty.to(dtype=torch.long)

        if self._size <= self.exclusion_window:
            return (
                torch.zeros((vecs.shape[0],), dtype=torch.float32, device=vecs.device),
                torch.full((vecs.shape[0],), -1, dtype=torch.long, device=vecs.device),
            )

        storage = self._ensure_storage(vecs.device)
        queries = vecs.to(device=storage.device, dtype=torch.float32)
        searchable_size = self._size - self.exclusion_window
        best_sims = torch.full(
            (queries.shape[0],), -1.0, dtype=torch.float32, device=storage.device
        )
        best_indices = torch.full((queries.shape[0],), -1, dtype=torch.long, device=storage.device)

        for start, end in self._searchable_segments(searchable_size):
            if end <= start:
                continue
            segment = storage[start:end]
            sims = queries @ segment.T
            seg_sims, seg_pos = sims.max(dim=1)
            physical_indices = seg_pos + start
            better = seg_sims > best_sims
            best_sims = torch.where(better, seg_sims, best_sims)
            best_indices = torch.where(better, physical_indices, best_indices)

        best_sims = best_sims.clamp_min(0.0)
        return best_sims, best_indices

    def _normalize_tensor(self, embeddings: Tensor) -> Tensor:
        vecs = embeddings.detach().to(dtype=torch.float32)
        if vecs.ndim == 1:
            vecs = vecs.unsqueeze(0)
        if vecs.shape[-1] != self.embedding_dim:
            raise ValueError(f"Expected embedding dim {self.embedding_dim}, got {vecs.shape[-1]}")
        norms = torch.linalg.norm(vecs, dim=1, keepdim=True)
        return vecs / torch.clamp(norms, min=1e-8)

    def _coerce_prepared_tensor(self, normalized_embeddings: Tensor) -> Tensor:
        vecs = normalized_embeddings.detach().to(dtype=torch.float32)
        if vecs.ndim == 1:
            vecs = vecs.unsqueeze(0)
        if vecs.shape[-1] != self.embedding_dim:
            raise ValueError(f"Expected embedding dim {self.embedding_dim}, got {vecs.shape[-1]}")
        return vecs

    def _ensure_storage(self, device: torch.device) -> Tensor:
        if self._storage is None:
            self._storage = torch.empty(
                (self.capacity, self.embedding_dim),
                dtype=torch.float32,
                device=device,
            )
            return self._storage
        if self._storage.device != device:
            self._storage = self._storage.to(device=device)
        return self._storage

    def _searchable_segments(self, searchable_size: int) -> list[tuple[int, int]]:
        if searchable_size <= 0:
            return []
        if self._size < self.capacity:
            return [(0, searchable_size)]

        start = self._next_index
        end = start + searchable_size
        if end <= self.capacity:
            return [(start, end)]
        return [(start, self.capacity), (0, end - self.capacity)]

    def _format_query_result(
        self,
        similarity: float,
        index: int,
    ) -> tuple[float, np.ndarray[Any, Any] | None, bool]:
        if index < 0 or self._storage is None:
            return 0.0, None, False
        matched = self._storage[index].detach().to(device="cpu").numpy().copy()
        return similarity, matched, similarity >= self.similarity_threshold

    @property
    def size(self) -> int:
        """Number of embeddings currently stored."""
        return self._size
