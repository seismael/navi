"""Abstract base for storage backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

__all__: list[str] = ["AbstractStorageBackend"]


class AbstractStorageBackend(ABC):
    """Base class for auditor storage backends.

    Provides a uniform interface for writing and reading recorded sessions.
    """

    @abstractmethod
    def open(self, path: str, mode: str = "w") -> None:
        """Open the storage at the given path.

        Args:
            path: File or directory path.
            mode: 'w' for write, 'r' for read.
        """

    @abstractmethod
    def write(self, topic: str, data: bytes, timestamp: float) -> None:
        """Write a serialized message to storage.

        Args:
            topic: ZMQ topic string.
            data: Serialized message bytes.
            timestamp: Recording timestamp.
        """

    @abstractmethod
    def read_all(self) -> list[tuple[str, bytes, float]]:
        """Read all recorded messages.

        Returns:
            List of (topic, data, timestamp) tuples.
        """

    @abstractmethod
    def close(self) -> None:
        """Close the storage and flush any buffers."""

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of recorded messages."""
