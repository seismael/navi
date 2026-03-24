"""Integration test: Mamba2 SSD through CognitiveMambaPolicy."""
import torch
print("Imports...")
from navi_actor.cognitive_policy import CognitiveMambaPolicy
print("CognitiveMambaPolicy imported")

device = "cuda"

# Create policy with mamba2 temporal core
policy = CognitiveMambaPolicy(
    embedding_dim=128,
    temporal_core="mamba2",
    azimuth_bins=256,
    elevation_bins=48,
)
policy = policy.to(device)
print(f"Policy created with mamba2 core: {policy.temporal_core_name}")
print(f"Total params: {sum(p.numel() for p in policy.parameters()):,}")

# Test single-step forward (inference)
obs = torch.randn(1, 3, 256, 48, device=device)
with torch.no_grad():
    actions, log_probs, values, hidden, z_t = policy(obs)
print(f"Inference: actions={actions.shape}, values={values.shape}, z_t={z_t.shape}")

# Test evaluate_sequence (PPO training path)
B, T = 4, 64
obs_seq = torch.randn(B, T, 3, 256, 48, device=device)
actions_seq = torch.randn(B, T, 4, device=device)
log_probs, values, entropy, hidden, z_flat = policy.evaluate_sequence(obs_seq, actions_seq)
print(f"Evaluate sequence: log_probs={log_probs.shape}, values={values.shape}, entropy={entropy.item():.4f}")

# Backward
loss = -log_probs.mean() + 0.5 * values.mean()
loss.backward()
print(f"Backward OK, loss={loss.item():.4f}")

print("\nFull integration test passed!")
