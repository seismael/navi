"""Canonical temporal profiling harness.

This script benchmarks the active actor temporal core under the actor interface
shape: (B, T, D) -> (B, T, D) and single-step (B, D) -> (B, D).

It is diagnostic tooling only and does not alter production runtime selection.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Protocol

import torch
from torch import Tensor, nn


class TemporalCoreLike(Protocol):
    """Protocol for candidate temporal cores used by the bake-off harness."""

    def eval(self) -> TemporalCoreLike: ...

    def forward(self, z_seq: Tensor) -> tuple[Tensor, Tensor | None]: ...

    def forward_step(
        self,
        z_t: Tensor,
        hidden: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]: ...


@dataclass
class BenchmarkResult:
    """Single candidate benchmark summary."""

    candidate: str
    available: bool
    device: str
    batch: int
    seq_len: int
    d_model: int
    repeats: int
    warmup: int
    forward_ms: float | None
    step_ms: float | None
    tokens_per_second: float | None
    error: str | None


def _time_call(fn: Callable[[], object], repeats: int) -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total = time.perf_counter() - start
    return (total / max(1, repeats)) * 1000.0


def _make_candidate(name: str, d_model: int, device: torch.device) -> TemporalCoreLike:
    if name == "mambapy":
        from navi_actor.mambapy_core import MambapyTemporalCore

        return MambapyTemporalCore(d_model=d_model).to(device)
    if name == "gru":
        from navi_actor.gru_core import GRUTemporalCore

        return GRUTemporalCore(d_model=d_model).to(device)
    raise ValueError(f"Unknown candidate: {name}")


def benchmark_candidate(
    candidate: str,
    *,
    batch: int,
    seq_len: int,
    d_model: int,
    repeats: int,
    warmup: int,
    device: torch.device,
) -> BenchmarkResult:
    try:
        core = _make_candidate(candidate, d_model, device)
    except Exception as exc:
        return BenchmarkResult(
            candidate=candidate,
            available=False,
            device=str(device),
            batch=batch,
            seq_len=seq_len,
            d_model=d_model,
            repeats=repeats,
            warmup=warmup,
            forward_ms=None,
            step_ms=None,
            tokens_per_second=None,
            error=f"candidate unavailable: {exc}",
        )

    if isinstance(core, nn.Module):
        core.eval()
    z_seq = torch.randn(batch, seq_len, d_model, device=device)
    z_t = torch.randn(batch, d_model, device=device)

    with torch.no_grad():
        for _ in range(warmup):
            core.forward(z_seq)
            core.forward_step(z_t)

        forward_ms = _time_call(lambda: core.forward(z_seq), repeats)
        step_ms = _time_call(lambda: core.forward_step(z_t), repeats)

    tokens = float(batch * seq_len)
    tokens_per_second = (tokens / (forward_ms / 1000.0)) if forward_ms > 0 else None

    return BenchmarkResult(
        candidate=candidate,
        available=True,
        device=str(device),
        batch=batch,
        seq_len=seq_len,
        d_model=d_model,
        repeats=repeats,
        warmup=warmup,
        forward_ms=forward_ms,
        step_ms=step_ms,
        tokens_per_second=tokens_per_second,
        error=None,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the canonical actor temporal runtimes")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument(
        "--candidates",
        nargs="+",
        default=["mamba2"],
        choices=["gru", "mambapy", "mamba2"],
        help="Temporal-core candidates to benchmark. Mamba2 SSD is the canonical default.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON only",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    names = list(args.candidates)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    device = torch.device(args.device)
    results: list[BenchmarkResult] = []
    for name in names:
        # Keep JSON output deterministic even if third-party libs print banners.
        with contextlib.redirect_stdout(io.StringIO()):
            result = benchmark_candidate(
                name,
                batch=args.batch,
                seq_len=args.seq_len,
                d_model=args.d_model,
                repeats=args.repeats,
                warmup=args.warmup,
                device=device,
            )
        results.append(result)

    if args.json:
        print(json.dumps([asdict(x) for x in results], indent=2))  # noqa: T201
        return 0

    print("Canonical temporal profile results")  # noqa: T201
    print(f"  device={device} batch={args.batch} seq_len={args.seq_len} d_model={args.d_model}")  # noqa: T201
    for result in results:
        if not result.available:
            print(f"  {result.candidate:<8} unavailable: {result.error}")  # noqa: T201
            continue

        print(  # noqa: T201
            "  "
            f"{result.candidate:<8} "
            f"forward_ms={result.forward_ms:.3f} "
            f"step_ms={result.step_ms:.3f} "
            f"tokens_per_second={result.tokens_per_second:.1f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
