#!/usr/bin/env python3
"""
05_phase3_solvate.py

Phase 3: Solvate the phase2 bilayer using GROMACS.

- Reads:  Outputs/phase2_popc/bilayer_phase2.gro   (or from YAML)
- Writes: Outputs/phase3_popc/bilayer_solv.gro
- Optionally updates topology (topol.top) if provided.

Run:
  python Scripts/05_phase3_solvate.py Inputs/build_phase1.yaml

Notes:
- Requires `gmx` on PATH (your Homebrew install is fine).
- If you provide a topology file (cfg["gromacs"]["topol_top"]), the script will copy it into phase3
  and let `gmx solvate -p` update the SOL/W count.
"""

from __future__ import annotations
import os
import sys
import shutil
import argparse
import subprocess
from pathlib import Path

import yaml


def load_cfg(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run(cmd: list[str], cwd: str | None = None) -> None:
    print("\n>>", " ".join(cmd))
    p = subprocess.run(cmd, cwd=cwd, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {p.returncode}: {' '.join(cmd)}")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", help="Inputs/build_phase1.yaml")
    ap.add_argument(
        "--gmx",
        default="gmx",
        help="GROMACS executable name/path (default: gmx)",
    )
    ap.add_argument(
        "--phase2_name",
        default="phase2_popc",
        help="Folder name under Outputs/ for phase2 (default: phase2_popc)",
    )
    ap.add_argument(
        "--phase3_name",
        default="phase3_popc",
        help="Folder name under Outputs/ for phase3 (default: phase3_popc)",
    )
    ap.add_argument(
        "--in_gro",
        default="bilayer_phase2.gro",
        help="Input GRO inside phase2 folder (default: bilayer_phase2.gro)",
    )
    ap.add_argument(
        "--out_gro",
        default="bilayer_solv.gro",
        help="Output GRO inside phase3 folder (default: bilayer_solv.gro)",
    )
    ap.add_argument(
        "--water_model",
        default="spc216.gro",
        help="GROMACS solvent configuration (default: spc216.gro)",
    )
    ap.add_argument(
        "--no_topology_update",
        action="store_true",
        help="Do not attempt to use/update a topology (runs gmx solvate without -p).",
    )
    args = ap.parse_args()

    cfg = load_cfg(args.config)

    # Base outputs directory
    outputs_root = cfg.get("output", {}).get("outputs_root", "Outputs")
    outputs_root = str(outputs_root)

    phase2_dir = os.path.join(outputs_root, args.phase2_name)
    phase3_dir = os.path.join(outputs_root, args.phase3_name)
    ensure_dir(phase3_dir)

    in_gro = os.path.join(phase2_dir, args.in_gro)
    if not os.path.isfile(in_gro):
        raise FileNotFoundError(
            f"Could not find input GRO:\n  {in_gro}\n"
            f"Check that Phase 2 finished and wrote bilayer_phase2.gro in {phase2_dir}."
        )

    out_gro = os.path.join(phase3_dir, args.out_gro)

    # Topology handling (optional)
    # You can set in YAML:
    # gromacs:
    #   topol_top: "topol.top"  # path relative to project root OR absolute
    topol_src = None
    topol_dst = os.path.join(phase3_dir, "topol.top")

    if not args.no_topology_update:
        topol_src = cfg.get("gromacs", {}).get("topol_top", None)

    # If they supplied a topology path, copy it into phase3 and let gmx update it.
    use_top = False
    if topol_src:
        topol_src = str(topol_src)
        if not os.path.isabs(topol_src):
            # treat as path relative to project root (where you're running from)
            topol_src = os.path.abspath(topol_src)

        if not os.path.isfile(topol_src):
            raise FileNotFoundError(
                f"YAML specifies gromacs.topol_top but file not found:\n  {topol_src}"
            )

        shutil.copyfile(topol_src, topol_dst)
        use_top = True
        print(f"Copied topology to: {topol_dst}")
    else:
        print("No topology specified (or --no_topology_update set). Solvating without -p.")

    # Build gmx solvate command
    cmd = [
        args.gmx,
        "solvate",
        "-cp",
        in_gro,
        "-cs",
        args.water_model,
        "-o",
        out_gro,
    ]
    if use_top:
        cmd += ["-p", topol_dst]

    run(cmd)

    print("\nPhase 3 complete.")
    print(f"Read GRO:   {in_gro}")
    print(f"Wrote GRO:  {out_gro}")
    if use_top:
        print(f"Updated TOP: {topol_dst}")
    else:
        print("Topology was not updated (no -p).")


if __name__ == "__main__":
    main()

