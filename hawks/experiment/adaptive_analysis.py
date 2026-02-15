"""Adaptive AI analysis — coupled trust–accuracy dynamics and stability regimes."""

from __future__ import annotations

import enum
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure

from hawks.detection.adaptive import AdaptiveAIDetector
from hawks.operators.population import ARCHETYPES
from hawks.operators.trust import TrustModel
from hawks.seed import SeedManager


class Regime(enum.Enum):
    HIGH_TRUST = "high_trust_stable"
    LOW_TRUST = "low_trust_stable"
    OSCILLATORY = "oscillatory"
    UNSTABLE = "unstable"


class StabilityAnalyzer:
    """Coupled trust–accuracy dynamics: simulation, sweep, figures, and reporting."""

    def __init__(
        self,
        archetype_name: str = "calibrated_professional",
        archetypes: dict[str, dict[str, float]] | None = None,
        master_seed: int = 42,
    ) -> None:
        all_archetypes = archetypes if archetypes is not None else ARCHETYPES
        self._params = all_archetypes[archetype_name]
        self._master_seed = master_seed

    # --- Coupled simulation ---

    def simulate_trajectory(
        self,
        p_base: float,
        gamma: float,
        n_steps: int = 1000,
    ) -> tuple[list[float], list[float]]:
        """Run coupled trust-accuracy loop.

        Returns (trust_history, accuracy_history).
        Uses SeedManager for reproducibility — fresh instance per call
        so different (p_base, gamma) pairs don't share RNG state.
        """
        seed_mgr = SeedManager(self._master_seed)
        rng = seed_mgr.spawn("adaptive")
        detector = AdaptiveAIDetector(p_base, gamma, rng)
        trust_model = TrustModel(
            initial_trust=self._params["initial_trust"],
            alpha=self._params["alpha"],
            beta=self._params["beta"],
        )

        trust_history = [trust_model.get_trust()]
        accuracy_history = [detector.effective_accuracy(trust_model.get_trust())]

        for _ in range(n_steps):
            correct = detector.sample_correctness(trust_model.get_trust())
            trust_model.update(correct)
            t = trust_model.get_trust()
            trust_history.append(t)
            accuracy_history.append(detector.effective_accuracy(t))

        return trust_history, accuracy_history

    def classify_regime(
        self,
        trust_history: list[float],
        tail_length: int = 200,
    ) -> Regime:
        """Classify a trajectory's dynamical regime from its tail."""
        tail = trust_history[-tail_length:]
        mean = float(np.mean(tail))
        variance = float(np.var(tail))

        if variance > 0.15:
            return Regime.UNSTABLE
        if variance > 0.05:
            return Regime.OSCILLATORY
        if mean > 0.7:
            return Regime.HIGH_TRUST
        if mean < 0.3:
            return Regime.LOW_TRUST
        return Regime.OSCILLATORY

    # --- Parameter sweep ---

    def sweep(
        self,
        p_base_range: tuple[float, float] = (0.4, 0.95),
        gamma_range: tuple[float, float] = (0.0, 1.5),
        resolution: int = 30,
        n_steps: int = 1000,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sweep p_base x gamma, classify each.

        Returns (p_base_values, gamma_values, regime_grid).
        regime_grid is 2D int array (Regime enum index mapped as:
        HIGH_TRUST=0, LOW_TRUST=1, OSCILLATORY=2, UNSTABLE=3).
        """
        regime_order = [Regime.HIGH_TRUST, Regime.LOW_TRUST, Regime.OSCILLATORY, Regime.UNSTABLE]
        regime_to_int = {r: i for i, r in enumerate(regime_order)}

        p_base_values = np.linspace(p_base_range[0], p_base_range[1], resolution)
        gamma_values = np.linspace(gamma_range[0], gamma_range[1], resolution)
        regime_grid = np.zeros((resolution, resolution), dtype=int)

        for i, gamma in enumerate(gamma_values):
            for j, p_base in enumerate(p_base_values):
                trust_hist, _ = self.simulate_trajectory(p_base, gamma, n_steps)
                regime = self.classify_regime(trust_hist)
                regime_grid[i, j] = regime_to_int[regime]

        return p_base_values, gamma_values, regime_grid

    # --- Figures ---

    def plot_bifurcation_heatmap(
        self,
        p_base_range: tuple[float, float] = (0.4, 0.95),
        gamma_range: tuple[float, float] = (0.0, 1.5),
        resolution: int = 30,
    ) -> Figure:
        """Bifurcation diagram: X = p_base, Y = gamma, Color = regime."""
        p_base_values, gamma_values, regime_grid = self.sweep(
            p_base_range, gamma_range, resolution
        )

        cmap = ListedColormap(["blue", "red", "orange", "black"])
        regime_labels = ["High Trust", "Low Trust", "Oscillatory", "Unstable"]

        fig, ax = plt.subplots(figsize=(8, 6))
        mesh = ax.pcolormesh(
            p_base_values,
            gamma_values,
            regime_grid,
            cmap=cmap,
            vmin=0,
            vmax=3,
            shading="auto",
        )

        cbar = fig.colorbar(mesh, ax=ax, ticks=[0, 1, 2, 3])
        cbar.ax.set_yticklabels(regime_labels)

        ax.set_xlabel("Base Accuracy (p_base)", fontsize=12)
        ax.set_ylabel("Coupling Strength (γ)", fontsize=12)
        ax.set_title("Bifurcation Diagram: Trust–Accuracy Coupling", fontsize=14)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.tight_layout()
        return fig

    def plot_trajectory_examples(self, n_steps: int = 1000) -> Figure:
        """Three T_t trajectories: high-trust stable, low-trust stable, oscillatory."""
        examples = [
            ("High-Trust Stable", 0.85, 0.3),
            ("Low-Trust Stable", 0.45, 0.5),
            ("Oscillatory", 0.60, 1.2),
        ]

        fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

        for ax, (title, p_base, gamma) in zip(axes, examples):
            trust_hist, acc_hist = self.simulate_trajectory(p_base, gamma, n_steps)
            steps = range(len(trust_hist))

            ax.plot(steps, trust_hist, "-", color="steelblue", label="Trust (T)")
            ax.plot(steps, acc_hist, "--", color="coral", label="Accuracy (p)")
            ax.set_ylabel("Value")
            ax.set_title(f"{title} (p_base={p_base}, γ={gamma})")
            ax.set_ylim(-0.05, 1.05)
            ax.legend(loc="upper right", fontsize=8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        axes[-1].set_xlabel("Timestep")
        fig.tight_layout()
        return fig

    # --- Mathematical reporting ---

    def report(
        self,
        p_base_range: tuple[float, float] = (0.4, 0.95),
        gamma_range: tuple[float, float] = (0.0, 1.5),
        resolution: int = 30,
    ) -> dict[str, Any]:
        """Compute and return analytical summary."""
        p_base_values, gamma_values, regime_grid = self.sweep(
            p_base_range, gamma_range, resolution
        )

        # Regime int mapping: 0=HIGH_TRUST, 1=LOW_TRUST, 2=OSCILLATORY, 3=UNSTABLE

        # critical_gamma: for each p_base column, smallest gamma where regime
        # transitions from stable (HIGH_TRUST or LOW_TRUST) to OSCILLATORY/UNSTABLE
        critical_gamma: dict[float, float | None] = {}
        for j, p_base in enumerate(p_base_values):
            col = regime_grid[:, j]
            found = None
            for i, gamma in enumerate(gamma_values):
                if col[i] in (2, 3):  # OSCILLATORY or UNSTABLE
                    found = float(gamma)
                    break
            critical_gamma[float(p_base)] = found

        # stabilizing_regions: gamma=0 row is LOW_TRUST but cell is HIGH_TRUST
        baseline_row = regime_grid[0, :]  # gamma=0
        stabilizing: list[tuple[float, float]] = []
        for i in range(regime_grid.shape[0]):
            for j in range(regime_grid.shape[1]):
                if baseline_row[j] == 1 and regime_grid[i, j] == 0:
                    stabilizing.append((float(p_base_values[j]), float(gamma_values[i])))

        # destabilizing_regions: gamma=0 row is HIGH_TRUST but cell is OSCILLATORY/UNSTABLE
        destabilizing: list[tuple[float, float]] = []
        for i in range(regime_grid.shape[0]):
            for j in range(regime_grid.shape[1]):
                if baseline_row[j] == 0 and regime_grid[i, j] in (2, 3):
                    destabilizing.append((float(p_base_values[j]), float(gamma_values[i])))

        result = {
            "critical_gamma": critical_gamma,
            "stabilizing_regions": stabilizing,
            "destabilizing_regions": destabilizing,
        }

        # Print formatted report
        print("=" * 60)
        print("Adaptive AI Stability Report")
        print("=" * 60)

        print("\nCritical Gamma (onset of oscillation/instability):")
        for p_base, cg in critical_gamma.items():
            label = f"  p_base={p_base:.3f}: "
            if cg is not None:
                print(f"{label}γ_crit = {cg:.3f}")
            else:
                print(f"{label}stable for all γ")

        print(f"\nStabilizing regions (LOW→HIGH): {len(stabilizing)} cells")
        print(f"Destabilizing regions (HIGH→OSC/UNSTABLE): {len(destabilizing)} cells")
        print("=" * 60)

        return result
