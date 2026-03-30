"""Behavioral Cloning (BC) pre-training for the cognitive policy.

Loads demonstration recordings (numpy ``.npz`` archives) and trains the
full ``CognitiveMambaPolicy`` pipeline via supervised maximum-likelihood
on human action choices.  Produces a v2 checkpoint compatible with the
canonical ``PpoTrainer.load_training_state()`` for seamless RL fine-tuning.

Algorithm
---------
1. Load all ``.npz`` demonstration files from the specified directory.
2. Chunk into BPTT sequences of length ``bptt_len``.
3. For each minibatch of sequences, forward through the full pipeline
   via ``evaluate_sequence()`` and maximise :math:`\\log \\pi(a|o)` plus an
   entropy bonus to prevent premature collapse.
4. Save a v2 checkpoint with a fresh RND module for PPO compatibility.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from navi_actor.cognitive_policy import CognitiveMambaPolicy
from navi_actor.config import TemporalCoreName
from navi_actor.rnd import RNDModule

__all__: list[str] = ["BehavioralCloningTrainer"]

_LOG = logging.getLogger(__name__)


def _load_demonstrations(demo_dir: Path) -> tuple[Tensor, Tensor]:
    """Load all ``.npz`` demo files and return concatenated tensors.

    Returns
    -------
    observations : (N, 3, Az, El) float32
    actions : (N, 4) float32
    """
    obs_parts: list[np.ndarray] = []
    act_parts: list[np.ndarray] = []
    npz_files = sorted(demo_dir.glob("*.npz"))
    if not npz_files:
        msg = f"No .npz demonstration files found in {demo_dir}"
        raise FileNotFoundError(msg)

    for f in npz_files:
        try:
            data = np.load(f)
        except Exception:
            _LOG.warning("Skipping corrupt demo file: %s", f.name)
            continue
        obs_parts.append(data["observations"])  # (N_i, 3, Az, El)
        act_parts.append(data["actions"])        # (N_i, 4)
        _LOG.info("Loaded %s: %d steps", f.name, len(data["observations"]))

    if not obs_parts:
        msg = f"No valid .npz demonstration files found in {demo_dir}"
        raise FileNotFoundError(msg)

    all_obs = np.concatenate(obs_parts, axis=0)
    all_act = np.concatenate(act_parts, axis=0)
    _LOG.info(
        "Total demonstrations: %d steps, obs=%s, actions=%s",
        len(all_obs),
        all_obs.shape,
        all_act.shape,
    )
    return torch.from_numpy(all_obs), torch.from_numpy(all_act)


def _chunk_sequences(
    observations: Tensor,
    actions: Tensor,
    bptt_len: int,
) -> tuple[Tensor, Tensor]:
    """Chunk a flat demonstration buffer into BPTT-length sequences.

    Returns
    -------
    obs_seqs : (num_seqs, bptt_len, 3, Az, El)
    act_seqs : (num_seqs, bptt_len, 4)
    """
    n = observations.shape[0]
    # Trim to exact multiple of bptt_len
    usable = (n // bptt_len) * bptt_len
    if usable == 0:
        msg = f"Not enough steps ({n}) for bptt_len={bptt_len}"
        raise ValueError(msg)
    if usable < n:
        _LOG.info("Trimming %d trailing steps (not a full sequence)", n - usable)

    obs_seqs = observations[:usable].reshape(-1, bptt_len, *observations.shape[1:])
    act_seqs = actions[:usable].reshape(-1, bptt_len, actions.shape[-1])
    return obs_seqs, act_seqs


class BehavioralCloningTrainer:
    """Supervised BC trainer producing PPO-compatible checkpoints.

    Parameters
    ----------
    demo_dir:
        Directory containing ``.npz`` demonstration files.
    output_path:
        Path for the output v2 checkpoint file.
    temporal_core:
        Temporal core variant to train.
    embedding_dim:
        Encoder embedding dimensionality.
    azimuth_bins:
        Observation azimuth resolution.
    elevation_bins:
        Observation elevation resolution.
    epochs:
        Number of full passes through the demonstration data.
    learning_rate:
        Adam learning rate.
    bptt_len:
        Sequence length for BPTT training.
    minibatch_size:
        Number of sequences per minibatch.
    entropy_coeff:
        Entropy regularization coefficient.
    max_grad_norm:
        Gradient clipping threshold.
    freeze_log_std:
        If True, the policy log-std parameter is frozen during BC to
        preserve exploration capacity for subsequent PPO training.
    """

    def __init__(
        self,
        *,
        demo_dir: Path,
        output_path: Path,
        checkpoint_path: Path | None = None,
        temporal_core: TemporalCoreName = "mamba2",
        embedding_dim: int = 128,
        azimuth_bins: int = 256,
        elevation_bins: int = 48,
        epochs: int = 50,
        learning_rate: float = 1e-3,
        bptt_len: int = 8,
        minibatch_size: int = 32,
        entropy_coeff: float = 0.01,
        max_grad_norm: float = 0.5,
        freeze_log_std: bool = True,
    ) -> None:
        self._demo_dir = Path(demo_dir)
        self._output_path = Path(output_path)
        self._checkpoint_path = checkpoint_path
        self._temporal_core_name: TemporalCoreName = temporal_core
        self._embedding_dim = embedding_dim
        self._azimuth_bins = azimuth_bins
        self._elevation_bins = elevation_bins
        self._epochs = epochs
        self._lr = learning_rate
        self._bptt_len = bptt_len
        self._minibatch_size = minibatch_size
        self._entropy_coeff = entropy_coeff
        self._max_grad_norm = max_grad_norm
        self._freeze_log_std = freeze_log_std

    def run(self) -> Path:
        """Execute the full BC training loop.

        Returns
        -------
        Path to the saved checkpoint.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _LOG.info("BC training device: %s", device)

        # ── 1. Load demonstrations ───────────────────────────────
        observations, actions = _load_demonstrations(self._demo_dir)

        # ── 2. Chunk into BPTT sequences ─────────────────────────
        obs_seqs, act_seqs = _chunk_sequences(
            observations, actions, self._bptt_len,
        )
        n_seqs = obs_seqs.shape[0]
        _LOG.info(
            "Training data: %d sequences of length %d",
            n_seqs,
            self._bptt_len,
        )

        # ── 3. Build policy ──────────────────────────────────────
        policy = CognitiveMambaPolicy(
            embedding_dim=self._embedding_dim,
            temporal_core=self._temporal_core_name,
            azimuth_bins=self._azimuth_bins,
            elevation_bins=self._elevation_bins,
        ).to(device)

        # Resume from existing checkpoint if provided
        if self._checkpoint_path is not None:
            ckpt = torch.load(self._checkpoint_path, map_location=device, weights_only=True)
            policy_sd = ckpt.get("policy_state_dict", {})
            policy.load_state_dict(policy_sd)
            _LOG.info("Resumed policy weights from %s", self._checkpoint_path)

        policy.train()

        # Optionally freeze log_std to preserve exploration capacity
        if self._freeze_log_std:
            policy.heads.log_std.requires_grad_(False)
            _LOG.info(
                "Froze log_std at %s (std=%s)",
                policy.heads.log_std.data.tolist(),
                policy.heads.log_std.data.exp().tolist(),
            )

        # ── 4. Optimizer ─────────────────────────────────────────
        trainable_params = [p for p in policy.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable_params, lr=self._lr)

        # ── 5. Training loop ─────────────────────────────────────
        total_start = time.perf_counter()
        for epoch in range(1, self._epochs + 1):
            epoch_start = time.perf_counter()

            # Shuffle sequences each epoch
            perm = torch.randperm(n_seqs)
            obs_seqs = obs_seqs[perm]
            act_seqs = act_seqs[perm]

            epoch_nll_sum = 0.0
            epoch_entropy_sum = 0.0
            epoch_batches = 0

            for batch_start in range(0, n_seqs, self._minibatch_size):
                batch_end = min(batch_start + self._minibatch_size, n_seqs)
                batch_obs = obs_seqs[batch_start:batch_end].to(device)   # (B, T, 3, Az, El)
                batch_act = act_seqs[batch_start:batch_end].to(device)   # (B, T, 4)

                # Forward through full pipeline
                log_probs, _values, entropy, _hidden, _z_t = policy.evaluate_sequence(
                    batch_obs,
                    batch_act,
                    hidden=None,
                )

                # BC loss = negative log-likelihood + entropy regularisation
                nll_loss = -log_probs.mean()
                entropy_loss = -self._entropy_coeff * entropy
                loss = nll_loss + entropy_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(trainable_params, self._max_grad_norm)
                optimizer.step()

                epoch_nll_sum += nll_loss.item()
                epoch_entropy_sum += entropy.item()
                epoch_batches += 1

            epoch_ms = (time.perf_counter() - epoch_start) * 1000
            avg_nll = epoch_nll_sum / max(epoch_batches, 1)
            avg_entropy = epoch_entropy_sum / max(epoch_batches, 1)
            _LOG.info(
                "Epoch %3d/%d — NLL=%.4f  entropy=%.4f  (%.0fms)",
                epoch,
                self._epochs,
                avg_nll,
                avg_entropy,
                epoch_ms,
            )

        total_s = time.perf_counter() - total_start
        _LOG.info("BC training complete in %.1fs", total_s)

        # ── 6. Save v2 checkpoint ────────────────────────────────
        return self._save_checkpoint(policy, device)

    def _save_checkpoint(self, policy: CognitiveMambaPolicy, device: torch.device) -> Path:
        """Save a PpoTrainer-compatible v2 checkpoint."""
        # Unfreeze log_std before saving so PPO can train it
        policy.heads.log_std.requires_grad_(True)

        # Fresh RND module for checkpoint compatibility
        rnd = RNDModule(
            input_dim=self._embedding_dim,
            hidden_dim=self._embedding_dim,
            output_dim=64,
        )

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        state = {
            "version": 2,
            "run_id": f"bc-pretrain-{timestamp}",
            "policy_state_dict": {
                k: v.cpu() for k, v in policy.state_dict().items()
            },
            "rnd_state_dict": {k: v.cpu() for k, v in rnd.state_dict().items()},
            "reward_shaper_step": 0,
        }

        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, self._output_path)
        _LOG.info("BC checkpoint saved to %s", self._output_path)
        return self._output_path
