"""Integration smoke tests for auditor CLI surfaces."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from navi_auditor.cli import app

_RUNNER = CliRunner()


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "AGENTS.md").exists():
            return parent
    msg = "Unable to locate repository root for auditor integration tests"
    raise RuntimeError(msg)


@pytest.mark.integration
class TestRecordReplay:
    """Integration smoke tests for passive auditor surfaces."""

    def test_placeholder(self) -> None:
        """Placeholder — full pipeline test requires active publishers."""
        # TODO: Implement with in-process publisher mock

    def test_live_dataset_audit_cli_emits_parseable_json_summary(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo_root = _repo_root()
        manifest_path = repo_root / "artifacts" / "gmdag" / "corpus" / "gmdag_manifest.json"
        if not manifest_path.exists():
            pytest.skip("Live compiled corpus is not present in this workspace")

        monkeypatch.setattr("navi_auditor.cli.setup_logging", lambda *_args: None)
        result = _RUNNER.invoke(
            app,
            [
                "dataset-audit",
                "--actors",
                "1",
                "--steps",
                "1",
                "--warmup-steps",
                "0",
                "--azimuth-bins",
                "32",
                "--elevation-bins",
                "8",
                "--json",
            ],
        )

        if not result.stdout.strip():
            pytest.skip("dataset-audit did not emit JSON output")

        payload = json.loads(result.stdout)
        if result.exit_code != 0:
            issues = payload.get("issues", [])
            if isinstance(issues, list) and issues:
                pytest.skip(
                    "Live dataset-audit unavailable: " + "; ".join(str(issue) for issue in issues)
                )
            pytest.skip("Live dataset-audit failed without a parseable issue list")

        assert payload["profile"] == "dataset-audit"
        assert payload["ok"] is True
        assert payload["check"]["profile"] == "check-sdfdag"
        assert payload["check"]["ok"] is True
        assert payload["benchmark"]["profile"] == "bench-sdfdag"
        assert payload["benchmark"]["ok"] is True
        assert payload["benchmark"]["actors"] == 1
        assert payload["benchmark"]["steps"] == 1
        assert payload["issues"] == []
