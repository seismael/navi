"""Foveated CNN encoder for spherical distance matrices."""

from __future__ import annotations

from torch import Tensor, nn

__all__: list[str] = ["FoveatedEncoder"]


class FoveatedEncoder(nn.Module):  # type: ignore[misc]
    """4-layer CNN that encodes a (B, 2, Az, El) spherical image into z_t in R^D.

    Channel 0 = depth, channel 1 = semantic class id (one-hot or raw).
    """

    def __init__(self, embedding_dim: int = 128) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

        self.convs = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Encode spherical observation.

        Args:
            x: (B, 2, Az, El) float tensor.

        Returns:
            z_t: (B, embedding_dim) float tensor.

        """
        h: Tensor = self.convs(x)
        h = self.pool(h).squeeze(-1).squeeze(-1)
        z_t: Tensor = self.fc(h)
        return z_t
