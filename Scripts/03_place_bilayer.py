#!/usr/bin/env python3
"""
03_place_bilayer.py

Place lipid conformers onto pre-generated XY sites and build a bilayer.

Phase-1 assumptions:
- single lipid type (POPC)
- anchor atom = P
- clash checking = anchor-anchor only
"""

from __future__ import annotations
import os
import sys
import glob
import argparse
import numpy as np
import yaml
import MDAnalysis as mda
from scipy.spatial.transform import Rotation as R


def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_sites(path):
    return np.load(path)


def random_z_rotation(rng):
    theta = rng.uniform(0, 2 * np.pi)
    return R.from_rotvec([0.0, 0.0, theta])


def flip_z():
    # 180° rotation about X flips Z
    return R.from_rotvec([np.pi, 0.0, 0.0])


def pbc_xy(d, Lx, Ly):
    d[0] -= Lx * np.round(d[0] / Lx)
    d[1] -= Ly * np.round(d[1] / Ly)
    return d


def anchor_clash(anchor, anchors, min_dist, box):
    for a in anchors:
        d = anchor - a
        d = pbc_xy(d, box["Lx"], box["Ly"])
        if np.linalg.norm(d[:2]) < min_dist:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", help="Inputs/build_phase1.yaml")
    ap.add_argument("--seed", type=int, default=2025)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    cfg = load_cfg(args.config)

    out_dir = cfg["output"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    box = cfg["box"]
    z_top = cfg["leaflets"]["z_top"]
    z_bot = cfg["leaflets"]["z_bottom"]
    min_dist = cfg["packing"]["min_anchor_dist"]

    # Load sites
    sites_top = load_sites(os.path.join(out_dir, "sites_top.npy"))
    sites_bot = load_sites(os.path.join(out_dir, "sites_bottom.npy"))

    # Load lipid conformers
    lipid_name, lipid_cfg = next(iter(cfg["lipids"].items()))
    conf_path = lipid_cfg["path"]
    conf_files = sorted(glob.glob(os.path.join(conf_path, lipid_cfg["file_glob"])))
    if len(conf_files) == 0:
        raise RuntimeError(f"No lipid conformers found at: {conf_path} with glob: {lipid_cfg['file_glob']}")

    # --- Placement state ---
    placed_atoms = []          # list of AtomGroup snapshots (one per lipid)
    placed_anchors_top = []    # list of np.array([x,y,z]) for top leaflet only
    placed_anchors_bot = []    # list of np.array([x,y,z]) for bottom leaflet only

    def place_leaflet(sites, z, flip, placed_anchors_this_leaflet):
        """
        Place lipids on a given leaflet.

        sites: (N,2) array of XY anchor positions
        z: target z-plane for the anchor (P)
        flip: whether to flip lipid orientation (True for bottom leaflet)
        placed_anchors_this_leaflet: list of already-placed anchors for THIS leaflet only
        """
        for i, (x, y) in enumerate(sites):
            success = False

            for _ in range(200):  # attempts per site
                u = mda.Universe(rng.choice(conf_files))
                atoms = u.atoms

                # Anchor atom (phosphate)
                sel = atoms.select_atoms("name P")
                if len(sel) == 0:
                    raise RuntimeError("No atom named 'P' found in conformer/template.")
                anchor = sel[0]

                # Center on anchor
                atoms.positions -= anchor.position

                # Random rotation about Z
                atoms.positions = random_z_rotation(rng).apply(atoms.positions)

                # Flip for bottom leaflet (invert Z)
                if flip:
                    atoms.positions = flip_z().apply(atoms.positions)

                # Translate to target site
                atoms.positions += np.array([x, y, z], dtype=float)

                # Clash check: ONLY within this leaflet (XY only)
                anchor_pos = atoms.select_atoms("name P")[0].position
                if anchor_clash(anchor_pos, placed_anchors_this_leaflet, min_dist, box):
                    continue

                # Accept placement
                placed_anchors_this_leaflet.append(anchor_pos.copy())
                placed_atoms.append(atoms.copy())  # snapshot of coordinates
                success = True
                break

            if not success:
                raise RuntimeError(f"Failed to place lipid at site {i}")

    # --- Place top and bottom leaflets (separate clash pools) ---
    place_leaflet(sites_top, z_top, flip=False, placed_anchors_this_leaflet=placed_anchors_top)
    place_leaflet(sites_bot, z_bot, flip=True,  placed_anchors_this_leaflet=placed_anchors_bot)

    # --- Combine placed lipids into one Universe and write outputs ---
    if len(placed_atoms) == 0:
        raise RuntimeError("No lipids were placed (placed_atoms is empty).")

    # Merge all AtomGroups into one Universe
    merged = mda.Merge(*placed_atoms)

    # Set periodic box (MDAnalysis expects [lx, ly, lz, alpha, beta, gamma])
    merged.dimensions = np.array([box["Lx"], box["Ly"], box["Lz"], 90.0, 90.0, 90.0], dtype=float)

    out_pdb = os.path.join(out_dir, "bilayer_placed.pdb")
    merged.atoms.write(out_pdb)

    # Optional GRO output (works if writer is available)
    out_gro = os.path.join(out_dir, "bilayer_placed.gro")
    try:
        merged.atoms.write(out_gro)
    except Exception as e:
        print(f"[warn] Could not write GRO ({e}). PDB written to: {out_pdb}")

    print(f"Wrote: {out_pdb}")
    if os.path.exists(out_gro):
        print(f"Wrote: {out_gro}")

if __name__ == "__main__":
    main()

