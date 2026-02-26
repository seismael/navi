A rigorous architectural audit of your provided `ppo_trainer.py`, `learner_ppo.py`, `server.py`, and `rollout_buffer.py` reveals exactly why your pipeline is suffering from both poor hardware utilization and a failure to learn.

While you successfully implemented GPU tensors and batched environments, there are **fatal mathematical flaws in how the data is shaped for PPO**, and **massive Python-level bottlenecks in your networking and memory queries**.

Here is the step-by-step breakdown of the critical issues and the exact code required to fix them.

---

### Part 1: Critical Mathematical Defects (Why the model fails to learn)

#### 1. The "Cross-Actor Bleed" in BPTT (FATAL)

Your `TrajectoryBuffer.extend_from()` concatenates all actors' trajectories into a single, massive 1D list.

* **The Bug:** When `sample_minibatches` extracts `seq_len=32` chunks for the Mamba-2 Backpropagation Through Time (BPTT), it slices blindly across this 1D list.
* **The Mathematical Result:** A chunk might contain frames 95-99 from Actor 0, and frames 0-26 from Actor 1. You are feeding Mamba-2 a sequence where the drone is instantly teleported from one environment to a completely different one. This permanently poisons the hidden state gradients, making temporal learning impossible.
* **The Fix:** You must *never* flatten multiple actors into a 1D list for sequence models. You must stack them into a `(Batch, Time, ...)` tensor.

#### 2. The Value Loss Broadcast Bug (FATAL)

In `learner_ppo.py` inside the value loss calculation:

```python
vf_loss = 0.5 * torch.max((new_vals - ret)**2, (value_pred_clipped - ret)**2).mean()

```

* **The Bug:** `new_vals` from your policy typically has shape `(B, 1)`. The returns tensor `ret` has shape `(B,)`.
* **The Mathematical Result:** In PyTorch, subtracting a `(B,)` tensor from a `(B, 1)` tensor triggers broadcasting, resulting in a **`(B, B)` matrix**. You are accidentally calculating the loss of every actor's value against every *other* actor's return. The Critic network is learning pure garbage, which in turn destroys the Actor network via the advantage calculation.
* **The Fix:** You must explicitly squeeze the values.
**In `learner_ppo.py`:**

```python
# Squeeze the outputs from the policy to ensure they are 1D (B,)
new_vals = new_vals.squeeze(-1)
# ...
vf_loss = 0.5 * torch.max((new_vals - ret)**2, (value_pred_clipped - ret)**2).mean()

```

#### 3. GAE Bootstrapping Flaw

In `rollout_buffer.py` inside `compute_returns_and_advantages`:

```python
if tr.truncated and not tr.done:
    delta = tr.reward + self.gamma * tr.value - tr.value # <--- BUG

```

* **The Bug:** When an episode truncates, you are bootstrapping the future return using `tr.value` (the value of the *current* state $t$).
* **The Fix:** You must bootstrap using the value of the *next* state ($t+1$).

```python
if tr.truncated and not tr.done:
    delta = tr.reward + self.gamma * prev_value - tr.value

```

---

### Part 2: Critical Performance Bottlenecks (Why utilization is < 80%)

#### 1. The ZMQ "Publish Storm" (CPU Maxed, GPU Starved)

In `server.py` inside your `_run_step_loop`, when a batch finishes, you do this:

```python
for obs, action in zip(observations, msg.actions, strict=True):
    self._publish(obs)
    self._publish_telemetry(...)

```

And similarly in `ppo_trainer.py`, you publish 64 individual actions and telemetry events per step.

* **The Bottleneck:** If you have 64 actors, your Python loop is serializing and sending **256 individual ZMQ messages per physical tick**. At 50 steps per second, your CPU is trying to serialize 12,800 JSON/MsgPack objects per second. The CPU hits 100% trying to manage the networking, leaving the GPU sitting idle waiting for the next PyTorch forward pass.
* **The Fix:** During high-speed training, **disable the PUB/SUB telemetry** in the inner loop. The `DistanceMatrix` and actions are already passing efficiently over the `REQ/REP` socket (`BatchStepResult`).
**In `ppo_trainer.py`, comment out these lines in the rollout loop:**

```python
# for a in actions:
#     self._publish_action(a)

# ONLY publish telemetry once every 100 steps, not every step!
if step_id % 100 == 0:
    self._publish_step_telemetry(...)

```

#### 2. The Sequential Memory Query (GPU Context Switching)

In `ppo_trainer.py`, right after the forward pass:

```python
a_np, z_np = a_t.cpu().numpy(), z_t.cpu().numpy()
sims = [self._memories[i].query(z_np[i])[0] for i in range(n)]

```

* **The Bottleneck:** You are moving GPU tensors to the CPU, then running a Python `for` loop 64 times to query 64 separate `EpisodicMemory` objects. This stalls the CUDA stream and prevents proper parallelization.
* **The Fix:** Your `EpisodicMemory` must be refactored to hold a single batched tensor for all actors, or at the very least, you must query it without transferring the latent vectors back to the CPU memory space if possible.

---

### Part 3: The Architectural Solution (How to fix `TrajectoryBuffer`)

To fix the fatal **Cross-Actor Bleed**, you must rewrite how your optimization worker consumes the buffers. Instead of `extend_from()`, you must pass the dictionary of buffers to a specialized BPTT sampler.

**Update `TrajectoryBuffer` sampling logic:**
Do not merge the buffers. Keep them as `dict[int, TrajectoryBuffer]`. Inside your `_optimisation_worker` in `ppo_trainer.py`, you should process them like this:

```python
# In ppo_trainer.py -> _optimisation_worker()

# 1. Collect all active actor IDs
active = [aid for aid in range(self._n_actors) if len(buffers[aid]) == seq_len_config] 

# 2. Extract tensors, preserving the [Actor, Time, ...] shape
all_obs = torch.stack([buffers[aid]._t_obs for aid in active]) # Shape: (B, T, 3, Az, El)
all_acts = torch.stack([buffers[aid]._t_actions for aid in active]) # Shape: (B, T, 4)
all_advs = torch.stack([buffers[aid]._advantages for aid in active]) # Shape: (B, T)
all_rets = torch.stack([buffers[aid]._returns for aid in active]) # Shape: (B, T)

# Flatten ONLY the Batch and Time dimensions for standard PPO math, 
# BUT pass them into Mamba keeping the (B, T) structure intact!
flat_obs = all_obs.view(-1, *all_obs.shape[2:]) 
flat_acts = all_acts.view(-1, all_acts.shape[-1])
# ...

```

### Summary of Next Steps

1. **Fix the Value Loss:** Add `.squeeze(-1)` to `new_vals` in `learner_ppo.py`. This alone will likely fix your agent's inability to learn.
2. **Fix the Bootstrapping:** Correct the `tr.truncated` logic in `rollout_buffer.py`.
3. **Mute the Network Spam:** Stop publishing individual telemetry and observation arrays in the tight training loop. Rely strictly on the `REQ/REP` batch result.
4. **Stop Merging Buffers:** Never use `extend_from()` to combine different actors' histories. Feed them to the network as distinct sequences.