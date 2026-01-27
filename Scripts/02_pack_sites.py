#!/usr/bin/env python3
"""
02_pack_sites.py

Generate XY anchor sites for top and bottom leaflets.

Method: jittered grid + rejection enforcing a minimum pairwise distance.
This is a robust Phase-1 packing baseline (fast, simple, reproducible).

Outputs:
  - sites_top.npy, sites_bottom.npy (shape: (N, 2))
  - sites_top.txt, sites_bottom.txt (x y per line)
"""

from __future__ import annotations
import os
import sys
import math
import argparse
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import yaml


@dataclass
class Box:
    Lx: float
    Ly: float
    Lz: float


def load_cfg(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)


def pbc_delta_xy(a: np.ndarray, b: np.ndarray, box: Box) -> np.ndarray:
    """Minimum-image delta in XY under periodic boundary conditions."""
    d = a - b
    d[0] -= box.Lx * np.round(d[0] / box.Lx)
    d[1] -= box.Ly * np.round(d[1] / box.Ly)
    return d


def min_dist_to_set(xy: np.ndarray, sites: np.ndarray, box: Box) -> float:
    """Return min distance from xy to any site (PBC in XY)."""
    if sites.size == 0:
        return np.inf
    # vectorized min-image in xy
    d = xy[None, :] - sites
    d[:, 0] -= box.Lx * np.round(d[:, 0] / box.Lx)
    d[:, 1] -= box.Ly * np.round(d[:, 1] / box.Ly)
    r = np.sqrt(np.sum(d * d, axis=1))
    return float(np.min(r))


def jittered_grid_sites(
    N: int,
    box: Box,
    min_dist: float,
    jitter: float,
    seed: int,
    max_tries_per_site: int = 2000,
) -> np.ndarray:
    """
    Build N XY points in [-Lx/2,Lx/2) x [-Ly/2,Ly/2) with min_dist separation.
    Start from a grid then jitter; reject if too close.
    """
    rng = np.random.default_rng(seed)

    if N <= 0:
        return np.zeros((0, 2), dtype=float)

    # Choose a grid dimension close to square
    nx = int(math.ceil(math.sqrt(N * box.Lx / box.Ly)))
    ny = int(math.ceil(N / nx))

    dx = box.Lx / nx
    dy = box.Ly / ny

    # Base grid centers in box-centered coordinates
    xs = (np.arange(nx) + 0.5) * dx - box.Lx / 2.0
    ys = (np.arange(ny) + 0.5) * dy - box.Ly / 2.0
    grid = np.array([(x, y) for y in ys for x in xs], dtype=float)

    rng.shuffle(grid)
    grid = grid[:N]

    placed = np.zeros((0, 2), dtype=float)

    for i in range(N):
        base = grid[i]
        ok = False

        for _ in range(max_tries_per_site):
            trial = base + rng.uniform(-jitter, jitter, size=2)

            # wrap into [-L/2, L/2)
            trial[0] = ((trial[0] + box.Lx / 2) % box.Lx) - box.Lx / 2
            trial[1] = ((trial[1] + box.Ly / 2) % box.Ly) - box.Ly / 2

            if min_dist_to_set(trial, placed, box) >= min_dist:
                placed = np.vstack([placed, trial[None, :]])
                ok = True
                break

        if not ok:
            raise RuntimeError(
                f"Failed to place site {i+1}/{N}. "
                f"Try reducing min_anchor_dist or jitter, or increase box size."
            )

    return placed


def write_sites_txt(path: str, sites: np.ndarray) -> None:
    with open(path, "w") as f:
        for x, y in sites:
            f.write(f"{x: .6f} {y: .6f}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", help="Path to Inputs/build_phase1.yaml")
    ap.add_argument("--seed", type=int, default=12345, help="RNG seed (base)")
    args = ap.parse_args()

    cfg = load_cfg(args.config)

    box = Box(**cfg["box"])
    out_dir = cfg["output"]["out_dir"]
    ensure_dir(out_dir)

    # For Phase 1 we assume a single lipid entry. This supports multiple too.
    # We’ll pack using the first lipid's counts.
    lipids = cfg["lipids"]
    if len(lipids) == 0:
        raise ValueError("No lipids specified in config.")

    # If you later do mixed lipids, you'll pack total counts then assign types in placement.
    first = next(iter(lipids.values()))
    N_top = int(first["count_top"])
    N_bot = int(first["count_bottom"])

    min_dist = float(cfg["packing"]["min_anchor_dist"])
    jitter = float(cfg["packing"].get("jitter", 0.0))

    sites_top = jittered_grid_sites(
        N=N_top, box=box, min_dist=min_dist, jitter=jitter, seed=args.seed + 1
    )
    sites_bot = jittered_grid_sites(
        N=N_bot, box=box, min_dist=min_dist, jitter=jitter, seed=args.seed + 2
    )

    np.save(os.path.join(out_dir, "sites_top.npy"), sites_top)
    np.save(os.path.join(out_dir, "sites_bottom.npy"), sites_bot)
    write_sites_txt(os.path.join(out_dir, "sites_top.txt"), sites_top)
    write_sites_txt(os.path.join(out_dir, "sites_bottom.txt"), sites_bot)

    print(f"Wrote {len(sites_top)} top sites and {len(sites_bot)} bottom sites to: {out_dir}")
    print("Files: sites_top.npy, sites_bottom.npy, sites_top.txt, sites_bottom.txt")


if __name__ == "__main__":
    main()

