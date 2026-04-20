"""PPO learner with clipped surrogate objective and GAE for Ghost-Matrix actor."""

from __future__ import annotations

import contextlib
import logging
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

if TYPE_CHECKING:
    from navi_actor.cognitive_policy import CognitiveMambaPolicy, PolicyEvalStageMetrics
    from navi_actor.rnd import RNDModule
    from navi_actor.rollout_buffer import MultiTrajectoryBuffer, TrajectoryBuffer

__all__: list[str] = ["PpoLearner", "PpoMetrics"]

_LOGGER = logging.getLogger(__name__)


@dataclass
class _StageTiming:
    wall_ms: float = 0.0
    device_ms: float = 0.0


@contextlib.contextmanager
def _stage_timer(*, device: torch.device, use_cuda_events: bool) -> Iterator[_StageTiming]:
    """Measure a learner stage with host wall-clock and optional synchronized CUDA events."""
    elapsed = _StageTiming()
    if use_cuda_events and device.type == "cuda":
        start_event: Any = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
        end_event: Any = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
        t_start = time.perf_counter()
        torch.cuda.synchronize(device)
        start_event.record()
        try:
            yield elapsed
        finally:
            end_event.record()
            torch.cuda.synchronize(device)
            elapsed.wall_ms = (time.perf_counter() - t_start) * 1000
            elapsed.device_ms = start_event.elapsed_time(end_event)
        return

    t_start = time.perf_counter()
    try:
        yield elapsed
    finally:
        elapsed.wall_ms = (time.perf_counter() - t_start) * 1000


def _prepare_minibatch_tensor(tensor: Tensor, device: torch.device) -> Tensor:
    """Reuse CUDA-resident minibatch tensors instead of re-copying them."""
    if tensor.device == device:
        return tensor
    return tensor.to(device=device, non_blocking=True)


def _uses_sequence_minibatch_surface(
    *, seq_len: int, sequence_observations: Tensor | None, sequence_actions: Tensor | None
) -> bool:
    """Canonical PPO BPTT path requires sequence-native minibatch views."""
    if seq_len <= 0:
        return False
    if sequence_observations is None or sequence_actions is None:
        raise RuntimeError(
            "Canonical PPO sequence training requires sequence-native minibatches when seq_len > 0.",
        )
    return True


@dataclass(frozen=True)
class PpoMetrics:
    """Metrics from a single PPO training epoch."""

    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clip_fraction: float
    total_loss: float
    rnd_loss: float
    learning_rate: float
    rnd_learning_rate: float
    n_updates: int = 0
    epoch_setup_ms: float = 0.0
    policy_optimizer_init_ms: float = 0.0
    rnd_optimizer_init_ms: float = 0.0
    policy_train_mode_ms: float = 0.0
    policy_param_cache_ms: float = 0.0
    setup_overhead_ms: float = 0.0
    minibatch_fetch_ms: float = 0.0
    minibatch_prep_ms: float = 0.0
    policy_eval_ms: float = 0.0
    loss_build_ms: float = 0.0
    backward_ms: float = 0.0
    zero_grad_ms: float = 0.0
    grad_clip_ms: float = 0.0
    optimizer_step_ms: float = 0.0
    rnd_step_ms: float = 0.0
    post_update_stats_ms: float = 0.0
    update_loop_overhead_ms: float = 0.0
    progress_callback_ms: float = 0.0
    iterator_setup_ms_total: float = 0.0
    minibatch_fetch_ms_total: float = 0.0
    minibatch_prep_ms_total: float = 0.0
    policy_eval_ms_total: float = 0.0
    loss_build_ms_total: float = 0.0
    backward_ms_total: float = 0.0
    zero_grad_ms_total: float = 0.0
    grad_clip_ms_total: float = 0.0
    optimizer_step_ms_total: float = 0.0
    rnd_step_ms_total: float = 0.0
    post_update_stats_ms_total: float = 0.0
    update_loop_overhead_ms_total: float = 0.0
    progress_callback_ms_total: float = 0.0
    epoch_finalize_ms: float = 0.0
    epoch_total_ms: float = 0.0
    policy_eval_device_ms_total: float = 0.0
    policy_eval_encode_ms_total: float = 0.0
    policy_eval_temporal_ms_total: float = 0.0
    policy_eval_heads_ms_total: float = 0.0
    policy_eval_encode_device_ms_total: float = 0.0
    policy_eval_temporal_device_ms_total: float = 0.0
    policy_eval_heads_device_ms_total: float = 0.0
    loss_build_device_ms_total: float = 0.0
    backward_device_ms_total: float = 0.0
    zero_grad_device_ms_total: float = 0.0
    grad_clip_device_ms_total: float = 0.0
    optimizer_step_device_ms_total: float = 0.0
    rnd_step_device_ms_total: float = 0.0
    post_update_stats_device_ms_total: float = 0.0


def _materialize_metric_means(
    *,
    running_policy_loss: Tensor,
    running_value_loss: Tensor,
    running_entropy: Tensor,
    running_kl: Tensor,
    running_clip: Tensor,
    running_total: Tensor,
    running_rnd_loss: Tensor,
    n_updates: int,
) -> Tensor:
    """Pack PPO epoch mean metrics into one host transfer for logging and return values."""
    if n_updates <= 0:
        return torch.zeros((7,), dtype=torch.float32, device="cpu")
    metric_tensor = torch.stack(
        (
            running_policy_loss,
            running_value_loss,
            running_entropy,
            running_kl,
            running_clip,
            running_total,
            running_rnd_loss,
        ),
        dim=0,
    ) / float(n_updates)
    return metric_tensor.detach().to(device="cpu", dtype=torch.float32)


class PpoLearner:
    """PPO learner with clipped surrogate, value clipping, and entropy bonus."""

    def __init__(
        self,
        gamma: float = 0.99,
        clip_ratio: float = 0.2,
        entropy_coeff: float = 0.01,
        value_coeff: float = 0.5,
        max_grad_norm: float = 0.5,
        learning_rate: float = 3e-4,
        rnd_learning_rate: float = 3e-5,
        profile_cuda_events: bool = False,
    ) -> None:
        self._gamma = gamma
        self._clip_ratio = clip_ratio
        self._entropy_coeff = entropy_coeff
        self._value_coeff = value_coeff
        self._max_grad_norm = max_grad_norm
        self._learning_rate = learning_rate
        self._rnd_learning_rate = rnd_learning_rate
        self._profile_cuda_events = profile_cuda_events
        self._optimizer: torch.optim.Adam | None = None
        self._rnd_optimizer: torch.optim.Adam | None = None
        self._policy_param_cache: tuple[Tensor, ...] | None = None

    def _create_adam_optimizer(
        self,
        params: Iterator[Tensor] | tuple[Tensor, ...],
        *,
        learning_rate: float,
        use_cuda: bool,
        label: str,
    ) -> torch.optim.Adam:
        """Create the fastest supported Adam variant for the active device."""
        if not use_cuda:
            return torch.optim.Adam(params, lr=learning_rate)

        try:
            optimizer = torch.optim.Adam(params, lr=learning_rate, fused=True)
            _LOGGER.info("PpoLearner: fused Adam enabled for %s", label)
            return optimizer
        except (TypeError, RuntimeError) as exc:
            _LOGGER.warning(
                "PpoLearner: fused Adam unavailable for %s (%s); falling back to foreach Adam",
                label,
                exc,
            )
            return torch.optim.Adam(params, lr=learning_rate, foreach=True)

    def set_learning_rate(self, lr: float, rnd_lr: float | None = None) -> None:
        """Update the learning rate for all optimizers (annealing)."""
        self._learning_rate = lr
        if rnd_lr is not None:
            self._rnd_learning_rate = rnd_lr

        if self._optimizer is not None:
            for param_group in self._optimizer.param_groups:
                param_group["lr"] = self._learning_rate
        if self._rnd_optimizer is not None:
            for param_group in self._rnd_optimizer.param_groups:
                param_group["lr"] = self._rnd_learning_rate

    def _get_optimizer(self, policy: CognitiveMambaPolicy) -> torch.optim.Adam:
        """Lazily create or return the unified policy optimizer.

        Covers the entire model: encoder + temporal core + actor head + critic head.
        """
        if self._optimizer is None:
            use_cuda = policy.device.type == "cuda"
            self._optimizer = self._create_adam_optimizer(
                policy.parameters(),
                learning_rate=self._learning_rate,
                use_cuda=use_cuda,
                label="policy",
            )
        return self._optimizer

    def _get_rnd_optimizer(self, rnd: RNDModule) -> torch.optim.Adam:
        """Lazily create or return the RND predictor optimizer."""
        if self._rnd_optimizer is None:
            use_foreach = next(rnd.predictor.parameters()).device.type == "cuda"
            self._rnd_optimizer = self._create_adam_optimizer(
                rnd.predictor.parameters(),
                learning_rate=self._rnd_learning_rate,
                use_cuda=use_foreach,
                label="rnd",
            )
        return self._rnd_optimizer

    def _policy_params(self, policy: CognitiveMambaPolicy) -> tuple[Tensor, ...]:
        """Cache the policy parameter tuple used for gradient clipping."""
        if self._policy_param_cache is None:
            self._policy_param_cache = tuple(policy.parameters())
        return self._policy_param_cache

    def prime_update_runtime(
        self,
        policy: CognitiveMambaPolicy,
        *,
        rnd: RNDModule | None = None,
    ) -> None:
        """Eagerly create optimizer and parameter-cache state before the first PPO update."""
        self._get_optimizer(policy)
        if rnd is not None:
            self._get_rnd_optimizer(rnd)
        self._policy_params(policy)

    def train_ppo_epoch(
        self,
        policy: CognitiveMambaPolicy,
        buffer: TrajectoryBuffer | MultiTrajectoryBuffer,
        *,
        ppo_epochs: int = 4,
        minibatch_size: int = 64,
        seq_len: int = 32,
        actor_ids: Tensor | None = None,
        rnd: RNDModule | None = None,
        progress_callback: Callable[[], None] | None = None,
        materialize_summary_scalars: bool = True,
    ) -> PpoMetrics:
        """Run multiple PPO mini-batch updates on a filled trajectory buffer.

        Args:
            actor_ids: Optional 1-D int64 tensor of actor indices to train on.
                When provided with a MultiTrajectoryBuffer, only the given actors'
                data is sampled (staggered per-actor PPO mode).
        """
        epoch_total_start = time.perf_counter()
        setup_start = time.perf_counter()
        optimizer_start = time.perf_counter()
        optimizer = self._get_optimizer(policy)
        policy_optimizer_init_ms = (time.perf_counter() - optimizer_start) * 1000
        rnd_optimizer_init_ms = 0.0
        if rnd is not None:
            rnd_optimizer_start = time.perf_counter()
            rnd_optimizer = self._get_rnd_optimizer(rnd)
            rnd_optimizer_init_ms = (time.perf_counter() - rnd_optimizer_start) * 1000
        else:
            rnd_optimizer = None
        device = policy.device
        train_mode_start = time.perf_counter()
        policy.train()
        policy_train_mode_ms = (time.perf_counter() - train_mode_start) * 1000

        _use_actor_subset = (
            actor_ids is not None
            and hasattr(buffer, "sample_minibatches_for_actors")
        )
        _sample_count = (
            buffer.actor_data_len(actor_ids)  # type: ignore[union-attr]
            if _use_actor_subset and actor_ids is not None
            else len(buffer)
        )
        _LOGGER.debug(
            "Starting PPO epoch with %d samples (epochs=%d, batch=%d)",
            _sample_count,
            ppo_epochs,
            minibatch_size,
        )

        param_cache_start = time.perf_counter()
        params = self._policy_params(policy)
        policy_param_cache_ms = (time.perf_counter() - param_cache_start) * 1000

        track_summary_scalars = materialize_summary_scalars
        metric_device = device
        running_policy_loss = (
            torch.zeros((), device=metric_device) if track_summary_scalars else None
        )
        running_value_loss = (
            torch.zeros((), device=metric_device) if track_summary_scalars else None
        )
        running_entropy = torch.zeros((), device=metric_device) if track_summary_scalars else None
        running_kl = torch.zeros((), device=metric_device) if track_summary_scalars else None
        running_clip = torch.zeros((), device=metric_device) if track_summary_scalars else None
        running_total = torch.zeros((), device=metric_device) if track_summary_scalars else None
        running_rnd_loss = torch.zeros((), device=metric_device) if track_summary_scalars else None
        n_updates = 0
        fetch_ms_acc = 0.0
        prep_ms_acc = 0.0
        eval_ms_acc = 0.0
        eval_encode_ms_acc = 0.0
        eval_temporal_ms_acc = 0.0
        eval_heads_ms_acc = 0.0
        loss_build_ms_acc = 0.0
        backward_ms_acc = 0.0
        zero_grad_ms_acc = 0.0
        grad_clip_ms_acc = 0.0
        optimizer_step_ms_acc = 0.0
        rnd_step_ms_acc = 0.0
        post_update_stats_ms_acc = 0.0
        progress_callback_ms_acc = 0.0
        eval_device_ms_acc = 0.0
        eval_encode_device_ms_acc = 0.0
        eval_temporal_device_ms_acc = 0.0
        eval_heads_device_ms_acc = 0.0
        loss_build_device_ms_acc = 0.0
        backward_device_ms_acc = 0.0
        zero_grad_device_ms_acc = 0.0
        grad_clip_device_ms_acc = 0.0
        optimizer_device_ms_acc = 0.0
        rnd_device_ms_acc = 0.0
        stats_device_ms_acc = 0.0
        iterator_setup_ms_acc = 0.0
        update_loop_overhead_ms_acc = 0.0
        epoch_setup_ms = (time.perf_counter() - setup_start) * 1000
        setup_overhead_ms = max(
            0.0,
            epoch_setup_ms
            - policy_optimizer_init_ms
            - rnd_optimizer_init_ms
            - policy_train_mode_ms
            - policy_param_cache_ms,
        )

        for _epoch in range(ppo_epochs):
            iterator_setup_start = time.perf_counter()
            if _use_actor_subset and actor_ids is not None:
                minibatch_iter = iter(
                    buffer.sample_minibatches_for_actors(  # type: ignore[union-attr]
                        actor_ids, minibatch_size, seq_len
                    )
                )
            else:
                minibatch_iter = iter(buffer.sample_minibatches(minibatch_size, seq_len))
            iterator_setup_ms_acc += (time.perf_counter() - iterator_setup_start) * 1000
            _kl_early_stop = False
            while True:
                update_total_start = time.perf_counter()
                t_fetch_start = time.perf_counter()
                try:
                    mb = next(minibatch_iter)
                except StopIteration:
                    break
                fetch_ms = (time.perf_counter() - t_fetch_start) * 1000
                fetch_ms_acc += fetch_ms
                t_prep_start = time.perf_counter()
                old_lp = _prepare_minibatch_tensor(mb.old_log_probs, device)
                adv = _prepare_minibatch_tensor(mb.advantages, device)
                ret = _prepare_minibatch_tensor(mb.returns, device)
                old_vals = _prepare_minibatch_tensor(mb.old_values, device)
                use_sequence_path = _uses_sequence_minibatch_surface(
                    seq_len=seq_len,
                    sequence_observations=mb.sequence_observations,
                    sequence_actions=mb.sequence_actions,
                )
                obs_seq_tensor = None
                acts_seq_tensor = None
                aux_seq_tensor = None
                obs: Tensor | None = None
                acts: Tensor | None = None
                aux_tensor: Tensor | None = None
                if use_sequence_path:
                    assert mb.sequence_observations is not None
                    assert mb.sequence_actions is not None
                    obs_seq_tensor = _prepare_minibatch_tensor(mb.sequence_observations, device)
                    acts_seq_tensor = _prepare_minibatch_tensor(mb.sequence_actions, device)
                    if mb.sequence_aux_tensors is not None:
                        aux_seq_tensor = _prepare_minibatch_tensor(mb.sequence_aux_tensors, device)
                else:
                    obs = _prepare_minibatch_tensor(mb.observations, device)
                    acts = _prepare_minibatch_tensor(mb.actions, device)
                    aux_tensor = (
                        _prepare_minibatch_tensor(mb.aux_tensors, device)
                        if mb.aux_tensors is not None
                        else None
                    )
                h0: Tensor | None = None
                prep_ms = (time.perf_counter() - t_prep_start) * 1000
                prep_ms_acc += prep_ms

                # Forward pass
                with _stage_timer(
                    device=device, use_cuda_events=self._profile_cuda_events
                ) as eval_ms:
                    eval_stage_metrics: PolicyEvalStageMetrics | None = None
                    if use_sequence_path:
                        assert obs_seq_tensor is not None
                        assert acts_seq_tensor is not None
                        if self._profile_cuda_events:
                            new_lp, new_vals, ent, _, z_mb, eval_stage_metrics = (
                                policy.evaluate_sequence_profiled(
                                    obs_seq_tensor,
                                    acts_seq_tensor,
                                    hidden=h0,
                                    aux_seq=aux_seq_tensor,
                                    use_cuda_events=True,
                                )
                            )
                        else:
                            new_lp, new_vals, ent, _, z_mb = policy.evaluate_sequence(
                                obs_seq_tensor,
                                acts_seq_tensor,
                                hidden=h0,
                                aux_seq=aux_seq_tensor,
                            )
                    else:
                        assert obs is not None
                        assert acts is not None
                        if self._profile_cuda_events:
                            new_lp, new_vals, ent, _, z_mb, eval_stage_metrics = (
                                policy.evaluate_profiled(
                                    obs,
                                    acts,
                                    aux_tensor=aux_tensor,
                                    use_cuda_events=True,
                                )
                            )
                        else:
                            new_lp, new_vals, ent, _, z_mb = policy.evaluate(
                                obs, acts, aux_tensor=aux_tensor
                            )

                    # H1 sub-attribution: nested timer isolates the loss-build math
                    # (clipped surrogate, value clipping, total composition). The outer
                    # eval_ms timer continues to cover both forward and loss build to
                    # preserve back-compat semantics for existing consumers.
                    with _stage_timer(
                        device=device, use_cuda_events=self._profile_cuda_events
                    ) as loss_build_ms:
                        new_vals = new_vals.squeeze(-1)
                        log_ratio = new_lp - old_lp
                        ratio = log_ratio.exp()
                        surr1 = ratio * adv
                        surr2 = (
                            torch.clamp(ratio, 1.0 - self._clip_ratio, 1.0 + self._clip_ratio)
                            * adv
                        )
                        policy_loss = -torch.min(surr1, surr2).mean()

                        value_pred_clipped = old_vals + torch.clamp(
                            new_vals - old_vals, -self._clip_ratio, self._clip_ratio
                        )
                        vf_loss = (
                            0.5
                            * torch.max(
                                (new_vals - ret) ** 2, (value_pred_clipped - ret) ** 2
                            ).mean()
                        )

                        total_loss = (
                            policy_loss + self._value_coeff * vf_loss - self._entropy_coeff * ent
                        )
                eval_ms_acc += eval_ms.wall_ms
                loss_build_ms_acc += loss_build_ms.wall_ms
                loss_build_device_ms_acc += loss_build_ms.device_ms
                eval_device_ms_acc += eval_ms.device_ms
                if eval_stage_metrics is not None:
                    eval_encode_ms_acc += eval_stage_metrics.encode_ms
                    eval_temporal_ms_acc += eval_stage_metrics.temporal_ms
                    eval_heads_ms_acc += eval_stage_metrics.heads_ms
                    eval_encode_device_ms_acc += eval_stage_metrics.encode_device_ms
                    eval_temporal_device_ms_acc += eval_stage_metrics.temporal_device_ms
                    eval_heads_device_ms_acc += eval_stage_metrics.heads_device_ms

                # --- Divergence guards ---
                if not torch.isfinite(total_loss):
                    _LOGGER.critical(
                        "Non-finite loss detected, skipping optimizer step"
                    )
                    n_updates += 1
                    if progress_callback is not None:
                        progress_callback()
                    continue

                with torch.no_grad():
                    approx_kl = (ratio - 1.0 - log_ratio).mean()
                    _approx_kl = float(approx_kl.detach().to(device="cpu").item())
                if _approx_kl > 0.03:
                    _LOGGER.warning(
                        "KL divergence %.4f exceeds threshold 0.03, stopping epoch early",
                        _approx_kl,
                    )
                    _kl_early_stop = True
                    break

                # Optimization step
                with _stage_timer(
                    device=device, use_cuda_events=self._profile_cuda_events
                ) as backward_ms:
                    # H1 sub-attribution: nested timer isolates zero_grad cost from
                    # the autograd traversal. backward_ms continues to report the
                    # combined block for back-compat with existing consumers.
                    with _stage_timer(
                        device=device, use_cuda_events=self._profile_cuda_events
                    ) as zero_grad_ms:
                        optimizer.zero_grad(set_to_none=True)
                    total_loss.backward()
                backward_ms_acc += backward_ms.wall_ms
                backward_device_ms_acc += backward_ms.device_ms
                zero_grad_ms_acc += zero_grad_ms.wall_ms
                zero_grad_device_ms_acc += zero_grad_ms.device_ms

                with _stage_timer(
                    device=device, use_cuda_events=self._profile_cuda_events
                ) as clip_ms:
                    torch.nn.utils.clip_grad_norm_(
                        params, self._max_grad_norm, foreach=(device.type == "cuda")
                    )
                grad_clip_ms_acc += clip_ms.wall_ms
                grad_clip_device_ms_acc += clip_ms.device_ms

                with _stage_timer(
                    device=device, use_cuda_events=self._profile_cuda_events
                ) as optimizer_ms:
                    optimizer.step()
                optimizer_step_ms_acc += optimizer_ms.wall_ms
                optimizer_device_ms_acc += optimizer_ms.device_ms

                # RND distillation
                rnd_loss_metric: Tensor | None = None
                rnd_wall_ms = 0.0
                if rnd is not None and rnd_optimizer is not None:
                    with _stage_timer(
                        device=device, use_cuda_events=self._profile_cuda_events
                    ) as rnd_ms:
                        z_rnd = z_mb.detach()
                        rnd_loss = rnd.distillation_loss(z_rnd)
                        rnd_optimizer.zero_grad(set_to_none=True)
                        rnd_loss.backward()  # type: ignore[no-untyped-call]
                        rnd_optimizer.step()
                        if track_summary_scalars:
                            rnd_loss_metric = rnd_loss.detach()
                    rnd_wall_ms = rnd_ms.wall_ms
                    rnd_step_ms_acc += rnd_ms.wall_ms
                    rnd_device_ms_acc += rnd_ms.device_ms

                stats_wall_ms = 0.0
                stats_device_ms = 0.0
                if track_summary_scalars:
                    with _stage_timer(
                        device=device, use_cuda_events=self._profile_cuda_events
                    ) as stats_ms:
                        with torch.no_grad():
                            approx_kl = (ratio - 1.0 - log_ratio).mean()
                            clip_frac = ((ratio - 1.0).abs() > self._clip_ratio).float().mean()

                        assert running_policy_loss is not None
                        assert running_value_loss is not None
                        assert running_entropy is not None
                        assert running_kl is not None
                        assert running_clip is not None
                        assert running_total is not None
                        assert running_rnd_loss is not None
                        running_policy_loss += policy_loss.detach()
                        running_value_loss += vf_loss.detach()
                        running_entropy += ent.detach()
                        running_kl += approx_kl.detach()
                        running_clip += clip_frac.detach()
                        running_total += total_loss.detach()
                        if rnd_loss_metric is not None:
                            running_rnd_loss += rnd_loss_metric
                    stats_wall_ms = stats_ms.wall_ms
                    stats_device_ms = stats_ms.device_ms
                    n_updates += 1
                else:
                    n_updates += 1
                post_update_stats_ms_acc += stats_wall_ms
                stats_device_ms_acc += stats_device_ms

                if progress_callback is not None:
                    t_callback_start = time.perf_counter()
                    progress_callback()
                    callback_ms = (time.perf_counter() - t_callback_start) * 1000
                    progress_callback_ms_acc += callback_ms
                else:
                    callback_ms = 0.0

                timed_update_ms = (
                    fetch_ms
                    + prep_ms
                    + eval_ms.wall_ms
                    + backward_ms.wall_ms
                    + clip_ms.wall_ms
                    + optimizer_ms.wall_ms
                    + rnd_wall_ms
                )
                timed_update_ms += stats_wall_ms + callback_ms
                update_total_ms = (time.perf_counter() - update_total_start) * 1000
                update_loop_overhead_ms_acc += max(0.0, update_total_ms - timed_update_ms)
            if _kl_early_stop:
                break

        if n_updates == 0:
            return PpoMetrics(
                policy_loss=0.0,
                value_loss=0.0,
                entropy=0.0,
                approx_kl=0.0,
                clip_fraction=0.0,
                total_loss=0.0,
                rnd_loss=0.0,
                learning_rate=self._learning_rate,
                rnd_learning_rate=self._rnd_learning_rate,
                n_updates=0,
            )

        mean_metric_values = [0.0] * 7
        if track_summary_scalars:
            assert running_policy_loss is not None
            assert running_value_loss is not None
            assert running_entropy is not None
            assert running_kl is not None
            assert running_clip is not None
            assert running_total is not None
            assert running_rnd_loss is not None
            finalize_start = time.perf_counter()
            mean_metrics = _materialize_metric_means(
                running_policy_loss=running_policy_loss,
                running_value_loss=running_value_loss,
                running_entropy=running_entropy,
                running_kl=running_kl,
                running_clip=running_clip,
                running_total=running_total,
                running_rnd_loss=running_rnd_loss,
                n_updates=n_updates,
            )
            mean_metric_values = mean_metrics.tolist()
            epoch_finalize_ms = (time.perf_counter() - finalize_start) * 1000

            _LOGGER.debug(
                "PPO epoch completed: loss=%0.4f, kl=%0.4f, entropy=%0.4f",
                mean_metric_values[5],
                mean_metric_values[3],
                mean_metric_values[2],
            )
        else:
            epoch_finalize_ms = 0.0

        epoch_total_ms = (time.perf_counter() - epoch_total_start) * 1000

        return PpoMetrics(
            policy_loss=mean_metric_values[0],
            value_loss=mean_metric_values[1],
            entropy=mean_metric_values[2],
            approx_kl=mean_metric_values[3],
            clip_fraction=mean_metric_values[4],
            total_loss=mean_metric_values[5],
            rnd_loss=mean_metric_values[6],
            learning_rate=self._learning_rate,
            rnd_learning_rate=self._rnd_learning_rate,
            n_updates=n_updates,
            epoch_setup_ms=epoch_setup_ms,
            policy_optimizer_init_ms=policy_optimizer_init_ms,
            rnd_optimizer_init_ms=rnd_optimizer_init_ms,
            policy_train_mode_ms=policy_train_mode_ms,
            policy_param_cache_ms=policy_param_cache_ms,
            setup_overhead_ms=setup_overhead_ms,
            minibatch_fetch_ms=(fetch_ms_acc / max(1, n_updates)),
            minibatch_prep_ms=(prep_ms_acc / max(1, n_updates)),
            policy_eval_ms=(eval_ms_acc / max(1, n_updates)),
            loss_build_ms=(loss_build_ms_acc / max(1, n_updates)),
            backward_ms=(backward_ms_acc / max(1, n_updates)),
            zero_grad_ms=(zero_grad_ms_acc / max(1, n_updates)),
            grad_clip_ms=(grad_clip_ms_acc / max(1, n_updates)),
            optimizer_step_ms=(optimizer_step_ms_acc / max(1, n_updates)),
            rnd_step_ms=(rnd_step_ms_acc / max(1, n_updates)),
            post_update_stats_ms=(post_update_stats_ms_acc / max(1, n_updates)),
            update_loop_overhead_ms=(update_loop_overhead_ms_acc / max(1, n_updates)),
            progress_callback_ms=(progress_callback_ms_acc / max(1, n_updates)),
            iterator_setup_ms_total=iterator_setup_ms_acc,
            minibatch_fetch_ms_total=fetch_ms_acc,
            minibatch_prep_ms_total=prep_ms_acc,
            policy_eval_ms_total=eval_ms_acc,
            policy_eval_encode_ms_total=eval_encode_ms_acc,
            policy_eval_temporal_ms_total=eval_temporal_ms_acc,
            policy_eval_heads_ms_total=eval_heads_ms_acc,
            backward_ms_total=backward_ms_acc,
            zero_grad_ms_total=zero_grad_ms_acc,
            loss_build_ms_total=loss_build_ms_acc,
            grad_clip_ms_total=grad_clip_ms_acc,
            optimizer_step_ms_total=optimizer_step_ms_acc,
            rnd_step_ms_total=rnd_step_ms_acc,
            post_update_stats_ms_total=post_update_stats_ms_acc,
            update_loop_overhead_ms_total=update_loop_overhead_ms_acc,
            progress_callback_ms_total=progress_callback_ms_acc,
            epoch_finalize_ms=epoch_finalize_ms,
            epoch_total_ms=epoch_total_ms,
            policy_eval_device_ms_total=eval_device_ms_acc,
            policy_eval_encode_device_ms_total=eval_encode_device_ms_acc,
            policy_eval_temporal_device_ms_total=eval_temporal_device_ms_acc,
            policy_eval_heads_device_ms_total=eval_heads_device_ms_acc,
            backward_device_ms_total=backward_device_ms_acc,
            zero_grad_device_ms_total=zero_grad_device_ms_acc,
            loss_build_device_ms_total=loss_build_device_ms_acc,
            grad_clip_device_ms_total=grad_clip_device_ms_acc,
            optimizer_step_device_ms_total=optimizer_device_ms_acc,
            rnd_step_device_ms_total=rnd_device_ms_acc,
            post_update_stats_device_ms_total=stats_device_ms_acc,
        )
