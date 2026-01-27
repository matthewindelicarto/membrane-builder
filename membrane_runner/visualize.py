"""
Visualization tools for membrane permeability.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import LinearSegmentedColormap
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class PermeabilityVisualizer:
    """Visualize membrane permeability results."""

    # Membrane colors
    WATER_COLOR = "#a8d5e5"
    HEADGROUP_COLOR = "#f4a460"
    CORE_COLOR = "#ffe4b5"

    def __init__(self, figsize: Tuple[float, float] = (12, 8)):
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required for visualization. Install with: pip install matplotlib")
        self.figsize = figsize

    def plot_energy_profile(
        self,
        result,
        title: str = "Transfer Energy Profile",
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> None:
        """
        Plot the transfer energy profile through the membrane.

        Args:
            result: PermeabilityResults object
            title: Plot title
            save_path: Path to save figure (optional)
            show: Whether to display the plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        z = result.energy_profile["z"]
        energy = result.energy_profile["energy_kJ_mol"]

        # Draw membrane regions
        self._draw_membrane_background(ax, z.min(), z.max())

        # Plot energy profile
        ax.plot(z, energy, "b-", linewidth=2.5, label="Transfer Energy")

        # Mark minimum energy (binding site)
        min_idx = np.argmin(energy)
        ax.scatter([z[min_idx]], [energy[min_idx]], color="red", s=100, zorder=5,
                   label=f"Binding site ({z[min_idx]:.1f} Å)")

        # Mark water reference (z=0 energy normalized)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5, label="Water reference")

        ax.set_xlabel("Z Position (Å from membrane center)", fontsize=12)
        ax.set_ylabel("Transfer Energy (kJ/mol)", fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        # Add permeability annotation
        textstr = f"log P = {result.log_p:.2f} cm/s\nP = {result.permeability_cm_s:.2e} cm/s"
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment="top", bbox=props)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_partition_coefficient(
        self,
        result,
        title: str = "Partition Coefficient Profile",
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> None:
        """Plot the partition coefficient K(z) through the membrane."""
        fig, ax = plt.subplots(figsize=(10, 6))

        z = result.partition_coefficients["z"]
        K = result.partition_coefficients["K"]

        self._draw_membrane_background(ax, z.min(), z.max())

        ax.semilogy(z, K, "g-", linewidth=2.5, label="K(z) = exp(-ΔG/RT)")

        ax.set_xlabel("Z Position (Å from membrane center)", fontsize=12)
        ax.set_ylabel("Partition Coefficient K(z)", fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3, which="both")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_membrane_cross_section(
        self,
        result,
        molecule_name: str = "Molecule",
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> None:
        """
        Plot a cross-section view of molecule permeating through membrane.
        Shows energy as color intensity.
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        z = result.energy_profile["z"]
        energy = result.energy_profile["energy_kJ_mol"]

        # Normalize energy for color mapping
        energy_norm = (energy - energy.min()) / (energy.max() - energy.min() + 1e-10)

        # Draw membrane layers
        # Water (top)
        ax.fill_between([-30, 30], [40, 40], [25, 25], color=self.WATER_COLOR, alpha=0.7)
        # Headgroup (top)
        ax.fill_between([-30, 30], [25, 25], [15, 15], color=self.HEADGROUP_COLOR, alpha=0.7)
        # Hydrocarbon core
        ax.fill_between([-30, 30], [15, 15], [-15, -15], color=self.CORE_COLOR, alpha=0.7)
        # Headgroup (bottom)
        ax.fill_between([-30, 30], [-15, -15], [-25, -25], color=self.HEADGROUP_COLOR, alpha=0.7)
        # Water (bottom)
        ax.fill_between([-30, 30], [-25, -25], [-40, -40], color=self.WATER_COLOR, alpha=0.7)

        # Draw molecule trajectory
        cmap = plt.cm.RdYlGn_r  # Red = high energy, Green = low energy
        for i, (zi, ei) in enumerate(zip(z, energy_norm)):
            color = cmap(ei)
            circle = plt.Circle((0, zi), 3, color=color, alpha=0.8)
            ax.add_patch(circle)

        # Mark binding position
        binding_z = result.binding_position
        ax.annotate(
            f"Binding site\n({binding_z:.1f} Å)",
            xy=(0, binding_z),
            xytext=(15, binding_z),
            fontsize=10,
            arrowprops=dict(arrowstyle="->", color="black"),
        )

        # Add labels
        ax.text(-25, 32, "WATER", fontsize=12, fontweight="bold")
        ax.text(-25, 20, "Headgroups", fontsize=10)
        ax.text(-25, 0, "Hydrocarbon\nCore", fontsize=10, ha="left", va="center")
        ax.text(-25, -20, "Headgroups", fontsize=10)
        ax.text(-25, -32, "WATER", fontsize=12, fontweight="bold")

        ax.set_xlim(-30, 30)
        ax.set_ylim(-40, 40)
        ax.set_aspect("equal")
        ax.set_xlabel("X (Å)", fontsize=12)
        ax.set_ylabel("Z (Å) - Membrane Normal", fontsize=12)
        ax.set_title(f"{molecule_name} Permeation Through Membrane", fontsize=14)

        # Add colorbar for energy
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(energy.min(), energy.max()))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label="Transfer Energy (kJ/mol)", shrink=0.8)

        # Add permeability info
        textstr = f"log P = {result.log_p:.2f} cm/s"
        props = dict(boxstyle="round", facecolor="white", alpha=0.9)
        ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=11,
                verticalalignment="bottom", horizontalalignment="right", bbox=props)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_comparison(
        self,
        results: Dict[str, "PermeabilityResults"],
        title: str = "Permeability Comparison",
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> None:
        """
        Compare energy profiles for multiple molecules.

        Args:
            results: Dict mapping molecule names to PermeabilityResults
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

        # Left plot: Energy profiles
        for (name, result), color in zip(results.items(), colors):
            z = result.energy_profile["z"]
            energy = result.energy_profile["energy_kJ_mol"]
            ax1.plot(z, energy, linewidth=2, label=f"{name} (log P={result.log_p:.1f})", color=color)

        self._draw_membrane_background(ax1, -15, 15)
        ax1.set_xlabel("Z Position (Å)", fontsize=12)
        ax1.set_ylabel("Transfer Energy (kJ/mol)", fontsize=12)
        ax1.set_title("Energy Profiles", fontsize=12)
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

        # Right plot: Bar chart of log P values
        names = list(results.keys())
        log_ps = [r.log_p for r in results.values()]

        bars = ax2.barh(names, log_ps, color=colors)
        ax2.set_xlabel("log P (cm/s)", fontsize=12)
        ax2.set_title("Permeability Comparison", fontsize=12)
        ax2.axvline(x=0, color="gray", linestyle="--", alpha=0.5)

        # Color bars by permeability (green=high, red=low)
        for bar, log_p in zip(bars, log_ps):
            if log_p > -6:
                bar.set_color("green")
            elif log_p > -8:
                bar.set_color("orange")
            else:
                bar.set_color("red")

        # Add value labels
        for i, (name, log_p) in enumerate(zip(names, log_ps)):
            ax2.text(log_p + 0.1, i, f"{log_p:.2f}", va="center", fontsize=10)

        plt.suptitle(title, fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close()

    def _draw_membrane_background(self, ax, z_min: float, z_max: float) -> None:
        """Draw membrane region background on axis."""
        # Hydrocarbon core (|z| < 10)
        ax.axvspan(-10, 10, alpha=0.2, color="yellow", label="Hydrocarbon core")
        # Headgroup regions
        ax.axvspan(-15, -10, alpha=0.2, color="orange")
        ax.axvspan(10, 15, alpha=0.2, color="orange", label="Headgroups")

    def create_summary_figure(
        self,
        result,
        molecule_name: str = "Molecule",
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> None:
        """Create a comprehensive summary figure with multiple panels."""
        fig = plt.figure(figsize=(14, 10))

        # Panel 1: Energy profile (top left)
        ax1 = fig.add_subplot(2, 2, 1)
        z = result.energy_profile["z"]
        energy = result.energy_profile["energy_kJ_mol"]
        self._draw_membrane_background(ax1, z.min(), z.max())
        ax1.plot(z, energy, "b-", linewidth=2)
        min_idx = np.argmin(energy)
        ax1.scatter([z[min_idx]], [energy[min_idx]], color="red", s=80, zorder=5)
        ax1.set_xlabel("Z (Å)")
        ax1.set_ylabel("ΔG (kJ/mol)")
        ax1.set_title("Transfer Energy Profile")
        ax1.grid(True, alpha=0.3)

        # Panel 2: Partition coefficient (top right)
        ax2 = fig.add_subplot(2, 2, 2)
        K = result.partition_coefficients["K"]
        self._draw_membrane_background(ax2, z.min(), z.max())
        ax2.semilogy(z, K, "g-", linewidth=2)
        ax2.set_xlabel("Z (Å)")
        ax2.set_ylabel("K(z)")
        ax2.set_title("Partition Coefficient")
        ax2.grid(True, alpha=0.3, which="both")

        # Panel 3: Cross-section view (bottom left)
        ax3 = fig.add_subplot(2, 2, 3)
        energy_norm = (energy - energy.min()) / (energy.max() - energy.min() + 1e-10)
        cmap = plt.cm.RdYlGn_r

        # Draw simplified membrane
        ax3.fill_between([-20, 20], [20, 20], [15, 15], color=self.WATER_COLOR, alpha=0.7)
        ax3.fill_between([-20, 20], [15, 15], [-15, -15], color=self.CORE_COLOR, alpha=0.5)
        ax3.fill_between([-20, 20], [-15, -15], [-20, -20], color=self.WATER_COLOR, alpha=0.7)

        # Draw molecule positions
        for zi, ei in zip(z, energy_norm):
            circle = plt.Circle((0, zi), 2, color=cmap(ei), alpha=0.7)
            ax3.add_patch(circle)

        ax3.set_xlim(-20, 20)
        ax3.set_ylim(-20, 20)
        ax3.set_aspect("equal")
        ax3.set_xlabel("X (Å)")
        ax3.set_ylabel("Z (Å)")
        ax3.set_title("Membrane Cross-Section")

        # Panel 4: Summary text (bottom right)
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.axis("off")

        summary_text = f"""
Permeability Summary
{'-'*40}

Molecule: {molecule_name}
Membrane type: {result.membrane_type}

Results:
  log P = {result.log_p:.2f}
  P = {result.permeability_cm_s:.2e} cm/s

Membrane interaction:
  Binding energy = {result.membrane_bound_energy:.1f} kJ/mol
  Binding position = {result.binding_position:.1f} Å

Classification:
  {"High permeability (> 10⁻⁶ cm/s)" if result.log_p > -6 else
   "Moderate permeability (10⁻⁸ to 10⁻⁶ cm/s)" if result.log_p > -8 else
   "Low permeability (< 10⁻⁸ cm/s)"}
"""
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=11,
                 verticalalignment="top", fontfamily="monospace",
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        plt.suptitle(f"Membrane Permeability Analysis: {molecule_name}", fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved summary figure: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()
