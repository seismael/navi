#!/usr/bin/env python3
"""End-to-End Throughput Proof for the Canonical Ghost-Matrix Stack (Phase 15)."""

import subprocess
import sys
import re
from pathlib import Path

def main():
    print("--- Navi Canonical Stack Benchmark ---")
    
    # Try to find a gmdag file
    gmdag_path = None
    search_paths = [
        Path("artifacts/scenes/scene_0.gmdag"),
        Path("projects/environment/artifacts/scenes/scene_0.gmdag"),
    ]
    
    # Also search in the corpus
    corpus_dir = Path("artifacts/gmdag/corpus")
    if corpus_dir.exists():
        gmdag_files = list(corpus_dir.glob("**/*.gmdag"))
        if gmdag_files:
            # Prefer larger/canonical scenes if possible, or just the first one
            search_paths.extend(gmdag_files)
            
    for p in search_paths:
        if p.exists() and p.is_file():
            gmdag_path = p
            break
            
    if not gmdag_path:
        print("Warning: No compiled .gmdag found. Benchmark may fail unless --scene is provided.")
        scene_arg = ""
    else:
        scene_arg = str(gmdag_path)
        print(f"Using scene: {scene_arg}")

    cmd = [
        "uv", "run", "brain", "profile",
        "--actors", "4",
        "--steps", "1024",
        "--azimuth-bins", "64",
        "--elevation-bins", "32"
    ]
    if scene_arg:
        cmd.extend(["--scene", scene_arg])

    print(f"Running command: {' '.join(cmd)}")
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    max_sps = 0.0
    
    for line in process.stdout:
        print(line, end="")
        # [step 100] ... | sps=123.4 ...
        match = re.search(r"sps=([\d.]+)", line)
        if match:
            sps = float(match.group(1))
            max_sps = max(max_sps, sps)

    process.wait()
    
    print("\n--- Benchmark Results ---")
    print(f"Peak Throughput: {max_sps:.1f} SPS")

    # Target 60+ SPS for Phase 15 on standard hardware (MX150/CPU-equivalent)
    if max_sps >= 60.0:
        print("STATUS: PASSED (60+ SPS TARGET ACHIEVED)")
        sys.exit(0)
    else:
        print("STATUS: FAILED (DID NOT REACH 60 SPS)")
        sys.exit(1)

if __name__ == "__main__":
    main()
