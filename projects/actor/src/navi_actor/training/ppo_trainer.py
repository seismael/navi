"""ZMQ-based PPO training loop for the Cognitive Mamba Policy."""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import zmq

from navi_actor.cognitive_policy import CognitiveMambaPolicy
from navi_actor.config import ActorConfig
from navi_actor.learner_ppo import PpoLearner, PpoMetrics
from navi_actor.memory.episodic import EpisodicMemory
from navi_actor.reward_shaping import RewardShaper
from navi_actor.rnd import RNDModule
from navi_actor.rollout_buffer import PPOTransition, TrajectoryBuffer
from navi_contracts import (
    TOPIC_ACTION,
    TOPIC_DISTANCE_MATRIX,
    TOPIC_TELEMETRY_EVENT,
    Action,
    BatchStepRequest,
    BatchStepResult,
    DistanceMatrix,
    StepRequest,
    StepResult,
    TelemetryEvent,
    deserialize,
    serialize,
)

__all__: list[str] = ["PpoTrainer", "PpoTrainingMetrics"]

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PpoTrainingMetrics:
    """Summary metrics from a PPO training run."""

    total_steps: int
    episodes: int
    reward_mean: float
    reward_ema: float
    policy_loss: float
    value_loss: float
    entropy: float
    rnd_loss: float
    intrinsic_reward_mean: float
    loop_detections: int
    beta_final: float
    zero_wait_ratio: float = 0.0
    # Performance instrumentation
    sps_mean: float = 0.0
    forward_pass_ms_mean: float = 0.0
    batch_step_ms_mean: float = 0.0
    memory_query_ms_mean: float = 0.0
    transition_ms_mean: float = 0.0
    tick_total_ms_mean: float = 0.0
    ppo_update_ms_mean: float = 0.0
    wall_clock_seconds: float = 0.0


class PpoTrainer:
    """Train a CognitiveMambaPolicy using PPO with ZMQ step-mode environment.

    Collects rollouts of configurable length, computes GAE advantages,
    and runs multi-epoch minibatch PPO updates.
    """

    def __init__(self, config: ActorConfig) -> None:
        self._config = config
        self._ctx: zmq.Context[zmq.Socket[bytes]] = zmq.Context()
        self._sub_socket: zmq.Socket[bytes] | None = None
        self._step_socket: zmq.Socket[bytes] | None = None
        self._pub_socket: zmq.Socket[bytes] | None = None

        self._n_actors = config.n_actors

        # Compute device
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Shared policy
        self._policy = CognitiveMambaPolicy(
            embedding_dim=config.embedding_dim,
            azimuth_bins=config.azimuth_bins,
            elevation_bins=config.elevation_bins,
            max_forward=config.max_forward,
            max_vertical=config.max_vertical,
            max_lateral=config.max_lateral,
            max_yaw=config.max_yaw,
        ).to(self._device)

        # Shared RND curiosity module
        self._rnd = RNDModule(
            input_dim=config.embedding_dim,
        ).to(self._device)

        # Per-actor episodic memories
        self._memories: dict[int, EpisodicMemory] = {
            i: EpisodicMemory(
                embedding_dim=config.embedding_dim,
                capacity=config.memory_capacity,
                exclusion_window=config.memory_exclusion_window,
                similarity_threshold=config.loop_threshold,
            )
            for i in range(self._n_actors)
        }

        # Shared reward shaper with annealing
        self._reward_shaper = RewardShaper(
            collision_penalty=config.collision_penalty,
            existential_tax=config.existential_tax,
            velocity_weight=config.velocity_weight,
            intrinsic_coeff_init=config.intrinsic_coeff_init,
            intrinsic_coeff_final=config.intrinsic_coeff_final,
            intrinsic_anneal_steps=config.intrinsic_anneal_steps,
            loop_penalty_coeff=config.loop_penalty_coeff,
            loop_threshold=config.loop_threshold,
        )

        # Shared learner
        self._learner = PpoLearner(
            gamma=config.gamma,
            clip_ratio=config.clip_ratio,
            entropy_coeff=config.entropy_coeff,
            value_coeff=config.value_coeff,
            learning_rate=config.learning_rate,
        )

        # Trajectory buffers
        self._buffers_a: dict[int, TrajectoryBuffer] = {
            i: TrajectoryBuffer(gamma=config.gamma, gae_lambda=config.gae_lambda)
            for i in range(self._n_actors)
        }
        self._buffers_b: dict[int, TrajectoryBuffer] = {
            i: TrajectoryBuffer(gamma=config.gamma, gae_lambda=config.gae_lambda)
            for i in range(self._n_actors)
        }
        self._buffers: dict[int, TrajectoryBuffer] = self._buffers_a

        # ── Async optimisation thread ──
        self._opt_event = threading.Event()
        self._opt_done = threading.Event()
        self._opt_done.set()
        self._opt_lock = threading.Lock()
        self._opt_thread: threading.Thread | None = None
        self._opt_stop = threading.Event()
        
        self._opt_buffers: dict[int, TrajectoryBuffer] | None = None
        self._opt_obs: dict[int, DistanceMatrix] | None = None
        self._opt_hiddens: dict[int, torch.Tensor | None] | None = None
        self._opt_aux: dict[int, torch.Tensor] | None = None
        
        self._sim_steps_during_opt: int = 0
        self._total_sim_steps: int = 0
        self._last_opt_duration_ms: float = 0.0
        self._opt_duration_acc: float = 0.0
        self._opt_duration_count: int = 0
        self._last_opt_metrics: PpoMetrics | None = None
        self._pending_obs: dict[int, DistanceMatrix] = {}

        self._mem_pool: ThreadPoolExecutor | None = (
            ThreadPoolExecutor(max_workers=self._n_actors)
            if self._n_actors > 1 else None
        )

    def start(self) -> None:
        """Open ZMQ sockets."""
        sub = self._ctx.socket(zmq.SUB)
        sub.connect(self._config.sub_address)
        sub.setsockopt(zmq.SUBSCRIBE, TOPIC_DISTANCE_MATRIX.encode("utf-8"))
        self._sub_socket = sub

        req = self._ctx.socket(zmq.REQ)
        req.connect(self._config.step_endpoint)
        self._step_socket = req

        pub = self._ctx.socket(zmq.PUB)
        pub.bind(self._config.pub_address)
        self._pub_socket = pub

        _LOGGER.info("PPO Trainer started: SUB=%s, REQ=%s, PUB=%s", 
                     self._config.sub_address, self._config.step_endpoint, self._config.pub_address)

    def stop(self) -> None:
        """Close ZMQ context."""
        import contextlib
        for sock in (self._sub_socket, self._step_socket, self._pub_socket):
            if sock: 
                with contextlib.suppress(Exception): 
                    sock.close()
        self._sub_socket = self._step_socket = self._pub_socket = None
        with contextlib.suppress(Exception): 
            self._ctx.term()

    def save_training_state(self, path: str | Path) -> None:
        """Save complete training state."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "version": 2,
            "policy_state_dict": self._policy.state_dict(),
            "rnd_state_dict": self._rnd.state_dict(),
            "reward_shaper_step": self._reward_shaper._global_step,
        }
        if self._learner._optimizer: state["optimizer_state_dict"] = self._learner._optimizer.state_dict()
        if self._learner._value_optimizer: state["value_optimizer_state_dict"] = self._learner._value_optimizer.state_dict()
        if self._learner._rnd_optimizer: state["rnd_optimizer_state_dict"] = self._learner._rnd_optimizer.state_dict()
        torch.save(state, save_path)

    def load_training_state(self, path: str | Path) -> None:
        """Restore training state."""
        data = torch.load(path, weights_only=False, map_location="cpu")
        if isinstance(data, dict) and data.get("version") == 2:
            self._policy.load_state_dict(data["policy_state_dict"])
            self._rnd.load_state_dict(data["rnd_state_dict"])
            self._reward_shaper._global_step = int(data.get("reward_shaper_step", 0))
            if "optimizer_state_dict" in data:
                self._learner._get_optimizer(self._policy).load_state_dict(data["optimizer_state_dict"])
            if "value_optimizer_state_dict" in data:
                self._learner._get_value_optimizer(self._policy).load_state_dict(data["value_optimizer_state_dict"])
            if "rnd_optimizer_state_dict" in data:
                self._learner._get_rnd_optimizer(self._rnd).load_state_dict(data["rnd_optimizer_state_dict"])
        else:
            self._policy.load_state_dict(data)

    def _recv_matrix(self, timeout_ms: int = 3000, *, expected_actor_id: int | None = None) -> DistanceMatrix | None:
        if expected_actor_id in self._pending_obs: return self._pending_obs.pop(expected_actor_id)
        if not self._sub_socket: return None
        deadline = time.monotonic() + timeout_ms / 1000.0
        while True:
            rem = int((deadline - time.monotonic()) * 1000)
            if rem <= 0: return None
            self._sub_socket.setsockopt(zmq.RCVTIMEO, rem)
            try:
                parts = self._sub_socket.recv_multipart()
                msg = deserialize(parts[1])
                aid = int(msg.env_ids[0]) if len(msg.env_ids) > 0 else 0
                if expected_actor_id is None or aid == expected_actor_id: return msg
                self._pending_obs[aid] = msg
            except zmq.Again: return None

    def _request_batch_step(self, actions: tuple[Action, ...], step_id: int) -> BatchStepResult:
        req = BatchStepRequest(actions=actions, step_id=step_id, timestamp=time.time())
        self._step_socket.send(serialize(req))
        return deserialize(self._step_socket.recv())

    def _publish_action(self, action: Action) -> None:
        if self._pub_socket: self._pub_socket.send_multipart([TOPIC_ACTION.encode("utf-8"), serialize(action)])

    def _publish_step_telemetry(self, *, step_id: int, episode_id: int, actor_id: int, **kwargs: Any) -> None:
        if not self._pub_socket: return
        p = np.array([kwargs.get(k, 0.0) for k in ["raw_reward", "shaped_reward", "intrinsic_reward", 
                     "loop_similarity", "is_loop", "beta", "done", "forward_vel", "yaw_vel"]], dtype=np.float32)
        event = TelemetryEvent(event_type="actor.training.ppo.step", episode_id=episode_id, env_id=actor_id, 
                               step_id=step_id, payload=p, timestamp=time.time())
        self._pub_socket.send_multipart([TOPIC_TELEMETRY_EVENT.encode("utf-8"), serialize(event)])

    def _publish_update_telemetry(self, step_id: int, reward_ema: float, metrics: PpoMetrics) -> None:
        if not self._pub_socket: return
        p = np.array([reward_ema, metrics.policy_loss, metrics.value_loss, metrics.entropy, metrics.approx_kl, 
                     metrics.clip_fraction, metrics.total_loss, metrics.rnd_loss, self._reward_shaper.beta], dtype=np.float32)
        event = TelemetryEvent(event_type="actor.training.ppo.update", episode_id=0, env_id=0, step_id=step_id, payload=p, timestamp=time.time())
        self._pub_socket.send_multipart([TOPIC_TELEMETRY_EVENT.encode("utf-8"), serialize(event)])

    def _publish_episode_telemetry(self, *, step_id: int, episode_id: int, actor_id: int, **kwargs: Any) -> None:
        if not self._pub_socket: return
        p = np.array([kwargs["episode_return"], float(kwargs["episode_length"])], dtype=np.float32)
        event = TelemetryEvent(event_type="actor.training.ppo.episode", episode_id=episode_id, env_id=actor_id, 
                               step_id=step_id, payload=p, timestamp=time.time())
        self._pub_socket.send_multipart([TOPIC_TELEMETRY_EVENT.encode("utf-8"), serialize(event)])

    def _publish_perf_telemetry(self, *, step_id: int, **kwargs: Any) -> None:
        if not self._pub_socket: return
        p = np.array([kwargs[k] for k in ["sps", "forward_pass_ms", "batch_step_ms", "memory_query_ms", 
                     "transition_ms", "tick_total_ms", "zero_wait_ratio", "ppo_update_ms"]], dtype=np.float32)
        event = TelemetryEvent(event_type="actor.training.ppo.perf", episode_id=0, env_id=0, step_id=step_id, 
                               payload=p, timestamp=time.time())
        self._pub_socket.send_multipart([TOPIC_TELEMETRY_EVENT.encode("utf-8"), serialize(event)])

    def _obs_to_tensor(self, obs: DistanceMatrix) -> torch.Tensor:
        d = np.asarray(obs.depth, dtype=np.float32)
        s = np.asarray(obs.semantic, dtype=np.float32)
        v = np.asarray(obs.valid_mask, dtype=np.float32)
        if d.ndim == 3: d, s, v = d[0], s[0], v[0]
        return torch.from_numpy(np.stack([d, s, v]))

    def _stack_hiddens(self, hiddens: dict[int, torch.Tensor | None], n: int) -> torch.Tensor | None:
        if all(h is None for h in hiddens.values()): return None
        dim = self._policy.temporal_core.d_model
        parts = []
        for i in range(n):
            h_state = hiddens[i]
            if h_state is not None: parts.append(h_state.to(self._device))
            else: parts.append(torch.zeros(1, 1, dim, device=self._device))
        return torch.cat(parts, dim=1)

    def _extract_hidden(self, batched: torch.Tensor | None, actor_id: int) -> torch.Tensor | None:
        return batched[:, actor_id : actor_id + 1, :].clone() if batched is not None else None

    def _optimisation_worker(self, ppo_epochs: int, minibatch_size: int, seq_len: int, step_ref: list[int], reward_ema_ref: list[float]) -> None:
        while not self._opt_stop.is_set():
            self._opt_event.wait(timeout=0.5)
            if self._opt_stop.is_set(): break
            if not self._opt_event.is_set(): continue
            self._opt_event.clear()
            buffers, obs_dict, hiddens, aux_dict = self._opt_buffers, self._opt_obs, self._opt_hiddens, self._opt_aux
            if not all([buffers, obs_dict, hiddens, aux_dict]): self._opt_done.set(); continue
            try:
                t_opt = time.perf_counter()
                self._policy.eval()
                active = [aid for aid in range(self._n_actors) if len(buffers[aid]) > 0]
                if active:
                    with torch.no_grad():
                        b_obs = torch.stack([self._obs_to_tensor(obs_dict[aid]) for aid in active]).to(self._device)
                        b_hid = self._stack_hiddens({i: hiddens[active[i]] for i in range(len(active))}, len(active))
                        b_aux = torch.stack([aux_dict[active[i]] for i in range(len(active))]).to(self._device)
                        _, _, b_val, _, _ = self._policy.forward(b_obs, b_hid, aux_tensor=b_aux)
                    for k, aid in enumerate(active): buffers[aid].compute_returns_and_advantages(last_value=b_val[k].item())
                merged = None
                for aid in range(self._n_actors):
                    if len(buffers[aid]) == 0: continue
                    if merged is None: merged = buffers[aid]
                    else: merged.extend_from(buffers[aid])
                if merged and len(merged) > 0:
                    with self._opt_lock:
                        self._policy.train()
                        self._last_opt_metrics = self._learner.train_ppo_epoch(self._policy, merged, ppo_epochs=ppo_epochs, minibatch_size=minibatch_size, seq_len=seq_len, rnd=self._rnd)
                ms = (time.perf_counter() - t_opt) * 1000
                self._last_opt_duration_ms = ms
                self._opt_duration_acc += ms
                self._opt_duration_count += 1
                for buf in buffers.values(): buf.clear()
            except Exception: _LOGGER.exception("Opt error")
            finally: self._opt_done.set()

    def train(self, total_steps: int, *, log_every: int = 100, checkpoint_every: int = 0, checkpoint_dir: str = "checkpoints") -> PpoTrainingMetrics:
        n = self._n_actors
        rl, ep, mb, sl = self._config.rollout_length, self._config.ppo_epochs, self._config.minibatch_size, self._config.bptt_len
        r_sum, r_ema, ep_cnt, step_id = 0.0, 0.0, 0, 0
        r_acc, s_acc, hids = {i: 0.0 for i in range(n)}, {i: 0 for i in range(n)}, {i: None for i in range(n)}
        aux_states = {i: torch.zeros(3, dtype=torch.float32, device=self._device) for i in range(n)}
        obs_dict = {i: self._recv_matrix(expected_actor_id=i) for i in range(n)}
        s_ref, r_ema_ref = [0], [0.0]
        self._opt_stop.clear(); self._opt_done.set()
        self._opt_thread = threading.Thread(target=self._optimisation_worker, args=(ep, mb, sl, s_ref, r_ema_ref), daemon=True)
        self._opt_thread.start()
        t_start = time.perf_counter()
        
        acc_fwd, acc_step, acc_mem, acc_trans, acc_tick, t_cnt = 0.0, 0.0, 0.0, 0.0, 0.0, 0
        
        while step_id < total_steps:
            self._policy.eval()
            for _ in range(rl):
                if step_id >= total_steps: break
                t_tick = time.perf_counter()
                o_tens = [self._obs_to_tensor(obs_dict[i]) for i in range(n)]
                o_batch = torch.stack(o_tens).to(self._device)
                h_batch = self._stack_hiddens(hids, n)
                aux_batch = torch.stack([aux_states[i] for i in range(n)])
                
                t_fwd_start = time.perf_counter()
                with torch.no_grad():
                    a_t, lp_t, v_t, n_h_batch, z_t = self._policy.forward(o_batch, h_batch, aux_tensor=aux_batch)
                    intr_r = self._rnd.intrinsic_reward(z_t)
                fwd_ms = (time.perf_counter() - t_fwd_start) * 1000
                
                a_np, z_np, now = a_t.cpu().numpy(), z_t.cpu().numpy(), time.time()
                actions = [Action(env_ids=np.array([i], dtype=np.int32), linear_velocity=np.array([[a[0], a[1], a[2]]], dtype=np.float32), 
                           angular_velocity=np.array([[0.0, 0.0, a[3]]], dtype=np.float32), policy_id="ppo", step_id=step_id, timestamp=now) for i, a in enumerate(a_np)]
                for a in actions: self._publish_action(a)
                
                t_mem_start = time.perf_counter()
                sims = [self._memories[i].query(z_np[i])[0] for i in range(n)]
                mem_ms = (time.perf_counter() - t_mem_start) * 1000
                
                t_step_start = time.perf_counter()
                res = self._request_batch_step(tuple(actions), step_id)
                step_ms = (time.perf_counter() - t_step_start) * 1000
                
                t_trans_start = time.perf_counter()
                for i in range(n):
                    r, d, tr = float(res.results[i].reward), bool(res.results[i].done), bool(res.results[i].truncated)
                    sh = self._reward_shaper.shape(raw_reward=r, done=d, forward_velocity=float(a_np[i, 0]), angular_velocity=float(a_np[i, 3]), 
                                                  intrinsic_reward=intr_r[i].item(), loop_similarity=sims[i])
                    self._reward_shaper.step()
                    self._publish_step_telemetry(step_id=step_id, episode_id=obs_dict[i].episode_id, actor_id=i, raw_reward=r, shaped_reward=sh.total, 
                                                 intrinsic_reward=intr_r[i].item(), loop_similarity=sims[i], is_loop=(sims[i] > self._config.loop_threshold), 
                                                 beta=self._reward_shaper.beta, done=d or tr, forward_vel=float(a_np[i, 0]), yaw_vel=float(a_np[i, 3]))
                    self._memories[i].add(z_np[i])
                    self._buffers[i].append(PPOTransition(observation=o_tens[i], action=a_t[i].cpu(), log_prob=lp_t[i].item(), 
                                                          value=v_t[i].item(), reward=sh.total, done=d, truncated=tr, hidden_state=self._extract_hidden(h_batch, i), aux_tensor=aux_batch[i].cpu()))
                    r_sum += r; r_ema = 0.01 * sh.total + 0.99 * r_ema
                    r_acc[i] += sh.total; s_acc[i] += 1
                    aux_states[i] = torch.tensor([sh.total, sims[i], intr_r[i].item()], dtype=torch.float32, device=self._device)
                    if d or tr:
                        ep_cnt += 1
                        self._publish_episode_telemetry(step_id=step_id, episode_id=obs_dict[i].episode_id, actor_id=i, episode_return=r_acc[i], episode_length=s_acc[i])
                        r_acc[i] = 0.0; s_acc[i] = 0; hids[i] = None; self._memories[i].reset()
                        aux_states[i] = torch.zeros(3, dtype=torch.float32, device=self._device)
                    else: hids[i] = self._extract_hidden(n_h_batch, i)
                    obs_dict[i] = res.observations[i]
                trans_ms = (time.perf_counter() - t_trans_start) * 1000
                
                step_id += n; s_ref[0] = step_id; r_ema_ref[0] = r_ema
                if not self._opt_done.is_set(): self._sim_steps_during_opt += n
                self._total_sim_steps += n
                
                tick_ms = (time.perf_counter() - t_tick) * 1000
                acc_fwd += fwd_ms; acc_step += step_ms; acc_mem += mem_ms; acc_trans += trans_ms; acc_tick += tick_ms; t_cnt += 1
                
                if log_every > 0 and step_id % log_every < n:
                    sps = step_id / max(0.001, time.perf_counter() - t_start)
                    zw = self._sim_steps_during_opt / max(1, self._total_sim_steps)
                    _LOGGER.info("[step %d] reward_ema=%.4f episodes=%d | sps=%.1f zero_wait=%.1f%%", step_id, r_ema, ep_cnt, sps, zw * 100)
                    self._publish_perf_telemetry(step_id=step_id, sps=sps, forward_pass_ms=fwd_ms, batch_step_ms=step_ms, memory_query_ms=mem_ms, 
                                                 transition_ms=trans_ms, tick_total_ms=tick_ms, zero_wait_ratio=zw, ppo_update_ms=self._last_opt_duration_ms)
            self._opt_done.wait()
            if self._last_opt_metrics: self._publish_update_telemetry(step_id, r_ema, self._last_opt_metrics)
            f = self._buffers; self._buffers = self._buffers_b if f is self._buffers_a else self._buffers_a
            self._opt_buffers, self._opt_obs, self._opt_hiddens, self._opt_aux = f, dict(obs_dict), dict(hids), {k: v.clone() for k, v in aux_states.items()}
            self._opt_done.clear(); self._opt_event.set()
            if checkpoint_every > 0 and step_id % checkpoint_every < n:
                self._opt_done.wait()
                self.save_training_state(Path(checkpoint_dir) / f"policy_step_{step_id:07d}.pt")
        self._opt_stop.set(); self._opt_event.set()
        return PpoTrainingMetrics(total_steps=step_id, episodes=ep_cnt, reward_mean=r_sum/max(1, step_id), reward_ema=r_ema, 
                                  policy_loss=0.0, value_loss=0.0, entropy=0.0, rnd_loss=0.0, intrinsic_reward_mean=0.0, 
                                  loop_detections=0, beta_final=self._reward_shaper.beta)
