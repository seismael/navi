"""Zarr storage backend."""

from __future__ import annotations

import numpy as np
import zarr

from navi_auditor.storage.base import AbstractStorageBackend

__all__: list[str] = ["ZarrBackend"]


class ZarrBackend(AbstractStorageBackend):
    """Zarr-based storage backend for recording and replaying sessions."""

    def __init__(self) -> None:
        self._store: zarr.Group | None = None
        self._index: int = 0
        self._path: str = ""
        self._mode: str = "w"

    def open(self, path: str, mode: str = "w") -> None:
        """Open a Zarr store at the given path."""
        self._path = path
        self._mode = mode
        store_mode = "w" if mode == "w" else "r"
        self._store = zarr.open_group(path, mode=store_mode)
        if mode == "w":
            self._index = 0
        else:
            # Count existing entries
            self._index = len([k for k in self._store if k.startswith("msg_")])

    def write(self, topic: str, data: bytes, timestamp: float) -> None:
        """Write a message to the Zarr store."""
        if self._store is None:
            msg = "Storage not opened. Call open() first."
            raise RuntimeError(msg)

        key = f"msg_{self._index:08d}"
        arr = np.frombuffer(data, dtype=np.uint8)
        dataset = self._store.create_array(
            key,
            shape=arr.shape,
            dtype=arr.dtype,
            overwrite=True,
        )
        dataset[:] = arr
        # Store metadata as attributes
        dataset.attrs["topic"] = topic
        dataset.attrs["timestamp"] = timestamp
        self._index += 1

    def read_all(self) -> list[tuple[str, bytes, float]]:
        """Read all recorded messages from the Zarr store."""
        if self._store is None:
            msg = "Storage not opened. Call open() first."
            raise RuntimeError(msg)

        results: list[tuple[str, bytes, float]] = []
        keys = sorted([k for k in self._store if k.startswith("msg_")])
        for key in keys:
            dataset = self._store[key]
            topic = str(dataset.attrs["topic"])
            timestamp = float(dataset.attrs["timestamp"])
            data = bytes(dataset[:])
            results.append((topic, data, timestamp))
        return results

    def close(self) -> None:
        """Close the Zarr store."""
        self._store = None

    def __len__(self) -> int:
        """Return the number of recorded messages."""
        return self._index
