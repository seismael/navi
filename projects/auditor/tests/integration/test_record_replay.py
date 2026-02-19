"""Integration test for record and replay flow."""

from __future__ import annotations

import pytest


@pytest.mark.integration
class TestRecordReplay:
    """Integration tests for the full record → replay pipeline."""

    def test_placeholder(self) -> None:
        """Placeholder — full pipeline test requires active publishers."""
        # TODO: Implement with in-process publisher mock
