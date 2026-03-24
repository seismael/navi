"""Quick smoke test for Mamba2SSDTemporalCore."""
import torch
print("torch imported")

from navi_actor.mamba2_core import Mamba2SSDTemporalCore
print("Mamba2SSDTemporalCore imported")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

core = Mamba2SSDTemporalCore(d_model=128)
core = core.to(device)
params = sum(p.numel() for p in core.parameters())
print(f"Core created: {params} params")

# Forward test
B, T, D = 4, 64, 128
z_seq = torch.randn(B, T, D, device=device)
aux = torch.randn(B, T, 3, device=device)
out, hidden = core(z_seq, aux_tensor=aux)
print(f"Forward: {z_seq.shape} -> {out.shape}, hidden={hidden}")

# Forward step test
z_t = torch.randn(B, D, device=device)
aux_t = torch.randn(B, 3, device=device)
out_t, _ = core.forward_step(z_t, aux_tensor=aux_t)
print(f"Forward step: {z_t.shape} -> {out_t.shape}")

# Backward test
core.zero_grad()
loss = out.sum()
loss.backward()
grad_ok = all(p.grad is not None for p in core.parameters() if p.requires_grad)
print(f"Backward: grad_ok={grad_ok}")

# Non-chunk-aligned sequence length
z2 = torch.randn(2, 37, D, device=device)  # 37 is not divisible by 64
out2, _ = core(z2)
print(f"Non-aligned seq: {z2.shape} -> {out2.shape}")

print("\nAll tests passed!")
