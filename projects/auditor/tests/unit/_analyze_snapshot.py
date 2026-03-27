"""Analyze the dashboard snapshot for the 'bad floor' diagnosis."""
import math
import numpy as np

d = np.load("../../artifacts/dashboard-captures/snapshot_20260327_115018/distance_matrix_arrays.npz")
depth = d["depth"]       # (256, 48) log-normalized
valid = d["valid_mask"]  # (256, 48) bool
semantic = d["semantic"]  # (256, 48) int32

print("=== SHAPES ===")
print(f"depth: {depth.shape}, valid: {valid.shape}, semantic: {semantic.shape}")
print(f"depth dtype: {depth.dtype}, range: [{depth.min():.4f}, {depth.max():.4f}]")
print(f"valid: {valid.sum()}/{valid.size} = {valid.mean()*100:.1f}%")

# Invert log normalization to get metric distance
max_distance = 100.0
metric = np.expm1(depth * math.log1p(max_distance))
print(f"\n=== METRIC DISTANCES (meters) ===")
vm = metric[valid]
print(f"valid range: [{vm.min():.2f}m, {vm.max():.2f}m]")
print(f"valid mean:  {vm.mean():.2f}m, median: {np.median(vm):.2f}m, std: {vm.std():.2f}m")

# Elevation profile
el_bins = 48
elevations_deg = np.linspace(90, -90, el_bins)
print(f"\n=== ELEVATION PROFILE (avg metric dist per elevation) ===")
for ei in range(0, el_bins, 4):
    col = metric[:, ei]
    vcol = valid[:, ei]
    avg = col[vcol].mean() if vcol.any() else float("nan")
    pct_valid = vcol.mean() * 100
    elev = elevations_deg[ei]
    direction = "UP" if elev > 10 else "DOWN" if elev < -10 else "HORIZON"
    print(f"  el={ei:2d} ({elev:+6.1f}deg {direction:>7s}): avg={avg:6.2f}m  valid={pct_valid:.0f}%")

# Azimuth profile
print(f"\n=== AZIMUTH PROFILE (avg metric dist per 45-deg sector) ===")
az_bins = 256
labels = ["FORWARD(-Z)", "FWD-RIGHT", "RIGHT(+X)", "BACK-RIGHT",
          "BACK(+Z)", "BACK-LEFT", "LEFT(-X)", "FWD-LEFT"]
for sector_i in range(8):
    start = sector_i * 32
    end = start + 32
    sect = metric[start:end, :]
    vsect = valid[start:end, :]
    avg = sect[vsect].mean() if vsect.any() else float("nan")
    heading_deg = sector_i * 45
    print(f"  az={heading_deg:3d}deg ({labels[sector_i]:>12s}): avg={avg:6.2f}m")

# Floor analysis
print(f"\n=== FLOOR (bottom 12 el bins = looking down) ===")
floor_depth = metric[:, 36:]
floor_valid = valid[:, 36:]
if floor_valid.any():
    print(f"floor valid: {floor_valid.mean()*100:.1f}%")
    print(f"floor avg: {floor_depth[floor_valid].mean():.2f}m")
    print(f"floor min: {floor_depth[floor_valid].min():.2f}m")

# Ceiling
print(f"\n=== CEILING (top 12 el bins = looking up) ===")
ceil_depth = metric[:, :12]
ceil_valid = valid[:, :12]
if ceil_valid.any():
    print(f"ceiling valid: {ceil_valid.mean()*100:.1f}%")
    print(f"ceiling avg: {ceil_depth[ceil_valid].mean():.2f}m")

# Horizon band
print(f"\n=== HORIZON (el bins 20-28) ===")
horiz_depth = metric[:, 20:28]
horiz_valid = valid[:, 20:28]
if horiz_valid.any():
    print(f"horizon valid: {horiz_valid.mean()*100:.1f}%")
    print(f"horizon avg: {horiz_depth[horiz_valid].mean():.2f}m")

# Distance histogram
print(f"\n=== DISTANCE HISTOGRAM ===")
edges = [0, 0.5, 1, 2, 3, 5, 10, 20, 50, 100]
for i in range(len(edges) - 1):
    cnt = ((metric >= edges[i]) & (metric < edges[i + 1]) & valid).sum()
    pct = cnt / valid.sum() * 100
    print(f"  [{edges[i]:5.1f}m, {edges[i+1]:5.1f}m): {cnt:5d} ({pct:5.1f}%)")

# Semantic analysis
print(f"\n=== SEMANTIC CLASSES ===")
for cls_id in np.unique(semantic):
    cnt = (semantic == cls_id).sum()
    print(f"  class {cls_id}: {cnt} ({cnt/semantic.size*100:.1f}%)")

# Per-elevation min/max to understand the floor shape
print(f"\n=== PER-ELEVATION RAW DEPTH (log-normalized) ===")
for ei in range(el_bins):
    col = depth[:, ei]
    vcol = valid[:, ei]
    if vcol.any():
        elev = elevations_deg[ei]
        print(f"  el={ei:2d} ({elev:+6.1f}deg): depth_norm min={col[vcol].min():.4f} max={col[vcol].max():.4f} mean={col[vcol].mean():.4f}")
