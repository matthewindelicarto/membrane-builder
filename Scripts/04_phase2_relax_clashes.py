#!/usr/bin/env python3
"""
Phase-2 runner template.

Behavior:
- Reads Phase-1 outputs from: cfg["output"]["out_dir"] (e.g., Outputs/phase1_popc)
- Writes Phase-2 outputs to: same path but with "phase1" -> "phase2" (e.g., Outputs/phase2_popc)
- Never overwrites Phase-1 files.

Drop this into your Phase-2 script (e.g., Scripts/04_phase2_*.py) as a complete file,
OR copy/paste the main() into your existing Phase-2 script.
"""

from __future__ import annotations

import os
import argparse
import numpy as np
import yaml
import MDAnalysis as mda


def load_cfg(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", help="Inputs/build_phase1.yaml")
    ap.add_argument("--seed", type=int, default=2025)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)  # available if you need randomness in Phase 2
    cfg = load_cfg(args.config)

    # ---------------------------
    # Phase-1 input directory
    # ---------------------------
    phase1_dir = cfg["output"]["out_dir"]
    if not os.path.isdir(phase1_dir):
        raise FileNotFoundError(f"Phase-1 output directory not found: {phase1_dir}")

    # ---------------------------
    # Phase-2 output directory
    # ---------------------------
    # Make a sibling folder by replacing "phase1" with "phase2"
    # Examples:
    #   Outputs/phase1_popc -> Outputs/phase2_popc
    #   /abs/path/.../phase1_popc -> /abs/path/.../phase2_popc
    phase2_dir = phase1_dir.replace("phase1", "phase2")
    if phase2_dir == phase1_dir:
        # If the user didn't name their phase1 folder with "phase1", fall back to appending
        phase2_dir = phase1_dir.rstrip("/").rstrip("\\") + "_phase2"

    os.makedirs(phase2_dir, exist_ok=True)

    # ---------------------------
    # Phase-1 input files
    # ---------------------------
    in_pdb = os.path.join(phase1_dir, "bilayer_placed.pdb")
    in_gro = os.path.join(phase1_dir, "bilayer_placed.gro")

    if os.path.exists(in_pdb):
        in_path = in_pdb
    elif os.path.exists(in_gro):
        in_path = in_gro
    else:
        raise FileNotFoundError(
            "Could not find Phase-1 bilayer file. Expected one of:\n"
            f"  {in_pdb}\n"
            f"  {in_gro}"
        )

    # ---------------------------
    # Phase-2 output files
    # ---------------------------
    out_pdb = os.path.join(phase2_dir, "bilayer_phase2.pdb")
    out_gro = os.path.join(phase2_dir, "bilayer_phase2.gro")

    # ---------------------------
    # LOAD Phase-1 structure
    # ---------------------------
    u = mda.Universe(in_path)

    # ============================================================
    # TODO: PUT YOUR PHASE-2 LOGIC HERE
    # - e.g., repack lipids, relax clashes, randomize tails, etc.
    # - whatever you do, operate on u.atoms.positions in-place.
    # ============================================================
    # Example no-op (keeps positions unchanged):
    _ = u.atoms.positions

    # ---------------------------
    # WRITE Phase-2 outputs
    # ---------------------------
    # PDB
    u.atoms.write(out_pdb)

    # GRO (needs dimensions set; try to preserve if present)
    # MDAnalysis stores box as u.dimensions = [lx, ly, lz, alpha, beta, gamma]
    # If not present, we write anyway but GRO box may be zeros.
    u.atoms.write(out_gro)

    print("Phase 2 complete.")
    print(f"Read from:   {in_path}")
    print(f"Wrote PDB:   {out_pdb}")
    print(f"Wrote GRO:   {out_gro}")


if __name__ == "__main__":
    main()

