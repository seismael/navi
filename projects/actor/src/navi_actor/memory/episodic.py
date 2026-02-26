"""Non-parametric episodic memory using KNN similarity for loop detection."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

__all__: list[str] = ["EpisodicMemory"]

_LOGGER = logging.getLogger(__name__)

_HAS_FAISS = False
try:
    import faiss  # type: ignore[import-untyped]

    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False


class EpisodicMemory:
    """KNN-based episodic memory that detects revisited states.

    Stores historical spatial embeddings and returns the maximum cosine
    similarity of a query against the memory buffer (excluding the
    most recent ``exclusion_window`` entries to avoid trivial matches).

    Uses FAISS when available, falls back to brute-force numpy cosine
    similarity otherwise.

    Args:
        embedding_dim: dimensionality of stored embeddings.
        capacity: maximum number of embeddings to retain.
        exclusion_window: number of most recent entries to skip when querying.
        similarity_threshold: cosine similarity above which a loop is detected.

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

        self._embeddings: list[np.ndarray[Any, Any]] = []
        self._faiss_index: Any | None = None
        self._use_faiss = _HAS_FAISS

        if self._use_faiss:
            self._faiss_index = faiss.IndexFlatIP(embedding_dim)
            _LOGGER.info("EpisodicMemory: using FAISS IndexFlatIP")
        else:
            _LOGGER.info("EpisodicMemory: using numpy fallback (faiss not found)")

    def reset(self) -> None:
        """Clear all stored embeddings (call at episode start)."""
        self._embeddings.clear()
        if self._use_faiss and self._faiss_index is not None:
            self._faiss_index.reset()

    def add(self, embedding: np.ndarray[Any, Any]) -> None:
        """Add an embedding to the memory buffer.

        Args:
            embedding: (D,) float32 vector — must be L2-normalized.

        """
        vec = np.asarray(embedding, dtype=np.float32).ravel()
        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            vec = vec / norm

        self._embeddings.append(vec)

        if self._use_faiss and self._faiss_index is not None:
            self._faiss_index.add(vec.reshape(1, -1))

        # Enforce capacity
        if len(self._embeddings) > self.capacity:
            self._embeddings.pop(0)
            if self._use_faiss:
                # Rebuild FAISS index from scratch (no incremental delete)
                self._rebuild_faiss_index()

    def query(
        self, embedding: np.ndarray[Any, Any],
    ) -> tuple[float, np.ndarray[Any, Any] | None, bool]:
        """Query the memory for the most similar historical embedding.

        Excludes the most recent ``exclusion_window`` entries.

        Args:
            embedding: (D,) float32 query vector.

        Returns:
            similarity: maximum cosine similarity (0.0 if memory too small).
            matched_embedding: the closest historical embedding, or None.
            is_loop: True if similarity >= similarity_threshold.

        """
        vec = np.asarray(embedding, dtype=np.float32).ravel()
        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            vec = vec / norm

        n = len(self._embeddings)
        searchable = n - self.exclusion_window
        if searchable <= 0:
            return 0.0, None, False

        if self._use_faiss and self._faiss_index is not None:
            return self._query_faiss(vec, searchable)

        return self._query_numpy(vec, searchable)

    def add_batch(self, embeddings: np.ndarray[Any, Any]) -> None:
        """Add a batch of embeddings to the memory buffer.

        Args:
            embeddings: (B, D) float32 vectors.
        """
        # Ensure normalized
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        vecs = embeddings / np.maximum(norms, 1e-8)

        for i in range(vecs.shape[0]):
            self._embeddings.append(vecs[i])

        if self._use_faiss and self._faiss_index is not None:
            self._faiss_index.add(vecs.astype(np.float32))

        # Enforce capacity (simple pop for now, could be optimized)
        while len(self._embeddings) > self.capacity:
            self._embeddings.pop(0)
            if self._use_faiss:
                self._rebuild_faiss_index()

    def query_batch(
        self, embeddings: np.ndarray[Any, Any],
    ) -> list[tuple[float, np.ndarray[Any, Any] | None, bool]]:
        """Query the memory for a batch of embeddings.

        Args:
            embeddings: (B, D) float32 query vectors.

        Returns:
            List of (similarity, matched_embedding, is_loop) tuples.
        """
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        vecs = (embeddings / np.maximum(norms, 1e-8)).astype(np.float32)

        n = len(self._embeddings)
        searchable = n - self.exclusion_window
        if searchable <= 0:
            return [(0.0, None, False)] * vecs.shape[0]

        if self._use_faiss and self._faiss_index is not None:
            # Batched FAISS search
            k = min(searchable, 10)
            distances, indices = self._faiss_index.search(vecs, k)
            
            results = []
            for b in range(vecs.shape[0]):
                best_sim = -1.0
                best_vec = None
                for i in range(k):
                    idx = int(indices[b, i])
                    if 0 <= idx < searchable:
                        sim = float(distances[b, i])
                        if sim > best_sim:
                            best_sim = sim
                            best_vec = self._embeddings[idx]
                
                if best_sim < 0:
                    results.append((0.0, None, False))
                else:
                    results.append((best_sim, best_vec, best_sim >= self.similarity_threshold))
            return results

        # Fallback to loop for numpy
        results = []
        for i in range(vecs.shape[0]):
            results.append(self._query_numpy(vecs[i], searchable))
        return results

    def _query_faiss(
        self,
        vec: np.ndarray[Any, Any],
        searchable: int,
    ) -> tuple[float, np.ndarray[Any, Any] | None, bool]:
        """FAISS inner-product search over the searchable window."""
        # We need to search only the first `searchable` entries.
        # FAISS doesn't support range queries easily, so we search top-K
        # and filter by index.
        k = min(searchable, 10)
        assert self._faiss_index is not None
        distances, indices = self._faiss_index.search(vec.reshape(1, -1), k)

        best_sim = -1.0
        best_vec: np.ndarray[Any, Any] | None = None
        for i in range(k):
            idx = int(indices[0, i])
            if idx < 0 or idx >= searchable:
                continue
            sim = float(distances[0, i])
            if sim > best_sim:
                best_sim = sim
                best_vec = self._embeddings[idx]

        if best_sim < 0:
            return 0.0, None, False

        return best_sim, best_vec, best_sim >= self.similarity_threshold

    def _query_numpy(
        self,
        vec: np.ndarray[Any, Any],
        searchable: int,
    ) -> tuple[float, np.ndarray[Any, Any] | None, bool]:
        """Brute-force cosine similarity search over numpy arrays."""
        history = np.array(self._embeddings[:searchable], dtype=np.float32)
        sims: np.ndarray[Any, Any] = history @ vec  # cosine sim (both L2-normed)
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        return best_sim, self._embeddings[best_idx], best_sim >= self.similarity_threshold

    def _rebuild_faiss_index(self) -> None:
        """Rebuild the FAISS index from the current embeddings list."""
        if not self._use_faiss:
            return
        self._faiss_index = faiss.IndexFlatIP(self.embedding_dim)
        if self._embeddings:
            matrix = np.array(self._embeddings, dtype=np.float32)
            self._faiss_index.add(matrix)

    @property
    def size(self) -> int:
        """Number of embeddings currently stored."""
        return len(self._embeddings)
