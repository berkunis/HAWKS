"""Analytical trust-dynamics module — closed-form equilibrium, tipping points, scrap rates."""

from __future__ import annotations

import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from hawks.operators.population import ARCHETYPES

# Consistent 4-color palette for archetypes
_ARCHETYPE_COLORS: dict[str, str] = {
    "conservative_skeptic": "#d62728",
    "calibrated_professional": "#1f77b4",
    "automation_biased": "#2ca02c",
    "algorithm_averse": "#ff7f0e",
}


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


class TrustAnalyzer:
    """Analytical computations and publication figures for trust dynamics.

    Derives closed-form equilibrium trust, critical tipping points, and
    expected scrap rates from the trust update equations without simulation.
    """

    def __init__(self, archetypes: dict[str, dict[str, float]] | None = None) -> None:
        self._archetypes = archetypes if archetypes is not None else dict(ARCHETYPES)

    # --- Pure analytical computations ---

    def equilibrium_trust(self, alpha: float, beta: float, p: float) -> float:
        """Equilibrium trust T*(p) = p*alpha / (p*alpha + (1-p)*beta).

        Returns 0.0 when the denominator is zero.
        """
        denom = p * alpha + (1.0 - p) * beta
        if denom == 0.0:
            return 0.0
        return (p * alpha) / denom

    def critical_accuracy(self, alpha: float, beta: float) -> float:
        """Critical AI accuracy where T* = 0.5.

        p_critical = beta / (alpha + beta).
        """
        denom = alpha + beta
        if denom == 0.0:
            return 0.0
        return beta / denom

    def expected_scrap_rate(
        self,
        p: float,
        archetype_weights: dict[str, float] | None = None,
        structural_risk: float = 0.3,
    ) -> float:
        """Population-average scrap rate at AI accuracy p.

        Uses the sigmoid decision model at equilibrium trust.
        Confidence is approximated as c = p (well-calibrated AI).
        Structural risk defaults to 0.3 (moderate).
        archetype_weights defaults to equal weights across all archetypes.
        """
        if archetype_weights is None:
            n = len(self._archetypes)
            archetype_weights = {name: 1.0 / n for name in self._archetypes}

        total_scrap = 0.0
        for name, weight in archetype_weights.items():
            params = self._archetypes[name]
            t_eq = self.equilibrium_trust(params["alpha"], params["beta"], p)
            logit = (
                params["trust_weight"] * t_eq
                + params["confidence_weight"] * p
                + params["risk_tolerance"] * structural_risk
            )
            p_accept = _sigmoid(logit)
            total_scrap += weight * (1.0 - p_accept)

        return total_scrap

    # --- Figure generators ---

    def _style_axes(self, ax: plt.Axes) -> None:
        """Apply clean publication style to axes."""
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.3, color="lightgray")
        ax.tick_params(labelsize=10)

    def plot_phase_diagram(self) -> Figure:
        """Equilibrium trust T* vs AI accuracy p, one curve per archetype.

        Vertical dashed lines at each archetype's p_critical.
        Horizontal dashed line at T* = 0.5.
        """
        p_vals = np.linspace(0.0, 1.0, 200)
        fig, ax = plt.subplots(figsize=(8, 5))

        # Horizontal reference at T* = 0.5
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.4, linewidth=1.0)

        for name, params in self._archetypes.items():
            alpha, beta = params["alpha"], params["beta"]
            color = _ARCHETYPE_COLORS.get(name, None)
            t_star = [self.equilibrium_trust(alpha, beta, float(p)) for p in p_vals]
            label = name.replace("_", " ").title()
            ax.plot(p_vals, t_star, linewidth=2.0, color=color, label=label)

            # Vertical dashed line at p_critical
            p_crit = self.critical_accuracy(alpha, beta)
            ax.axvline(
                p_crit, color=color, linestyle="--", alpha=0.5, linewidth=1.0
            )
            ax.annotate(
                f"{p_crit:.2f}",
                xy=(p_crit, 0.05),
                fontsize=8,
                color=color,
                ha="center",
            )

        ax.set_xlabel("AI Accuracy (p)", fontsize=12)
        ax.set_ylabel("Equilibrium Trust (T*)", fontsize=12)
        ax.set_title("Trust Phase Diagram", fontsize=13)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=9, loc="upper left")
        self._style_axes(ax)
        fig.tight_layout()
        return fig

    def plot_scrap_rate(
        self, archetype_weights: dict[str, float] | None = None
    ) -> Figure:
        """Expected scrap rate vs AI accuracy for a heterogeneous population.

        Shows per-archetype dashed curves behind the population average,
        with a shaded band between min and max archetype scrap rates.
        """
        p_vals = np.linspace(0.0, 1.0, 200)

        # Compute per-archetype scrap rates
        per_archetype: dict[str, list[float]] = {}
        for name in self._archetypes:
            single_weight = {name: 1.0}
            per_archetype[name] = [
                self.expected_scrap_rate(float(p), archetype_weights=single_weight)
                for p in p_vals
            ]

        # Population average
        pop_avg = [
            self.expected_scrap_rate(float(p), archetype_weights=archetype_weights)
            for p in p_vals
        ]

        fig, ax = plt.subplots(figsize=(8, 5))

        # Per-archetype thin dashed
        for name, rates in per_archetype.items():
            color = _ARCHETYPE_COLORS.get(name, None)
            label = name.replace("_", " ").title()
            ax.plot(p_vals, rates, linewidth=1.0, linestyle="--", color=color,
                    alpha=0.3, label=label)

        # Shaded band
        all_rates = np.array(list(per_archetype.values()))
        ax.fill_between(
            p_vals, all_rates.min(axis=0), all_rates.max(axis=0),
            alpha=0.12, color="steelblue",
        )

        # Population average
        ax.plot(p_vals, pop_avg, linewidth=2.0, color="steelblue",
                label="Population Average")

        ax.set_xlabel("AI Accuracy (p)", fontsize=12)
        ax.set_ylabel("Expected Scrap Rate", fontsize=12)
        ax.set_title("Scrap Rate vs AI Accuracy", fontsize=13)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=9, loc="upper right")
        self._style_axes(ax)
        fig.tight_layout()
        return fig

    def plot_beta_sensitivity(
        self,
        archetype_name: str = "calibrated_professional",
        beta_values: list[float] | None = None,
    ) -> Figure:
        """How increasing beta shifts the tipping threshold.

        Shows multiple T*(p) curves for different beta values on the same axes,
        with a color gradient from light to dark as beta increases.
        """
        if beta_values is None:
            beta_values = [0.05, 0.10, 0.20, 0.40]

        params = self._archetypes[archetype_name]
        alpha = params["alpha"]
        p_vals = np.linspace(0.0, 1.0, 200)

        fig, ax = plt.subplots(figsize=(8, 5))

        # Horizontal reference
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.4, linewidth=1.0)

        cmap = plt.cm.Blues  # type: ignore[attr-defined]
        n = len(beta_values)
        for i, beta in enumerate(beta_values):
            shade = 0.3 + 0.7 * (i / max(n - 1, 1))
            color = cmap(shade)
            t_star = [self.equilibrium_trust(alpha, beta, float(p)) for p in p_vals]
            ax.plot(p_vals, t_star, linewidth=2.0, color=color,
                    label=f"\u03b2 = {beta:.2f}")

            p_crit = self.critical_accuracy(alpha, beta)
            ax.axvline(p_crit, color=color, linestyle="--", alpha=0.5, linewidth=1.0)

        title_name = archetype_name.replace("_", " ").title()
        ax.set_xlabel("AI Accuracy (p)", fontsize=12)
        ax.set_ylabel("Equilibrium Trust (T*)", fontsize=12)
        ax.set_title(f"Beta Sensitivity — {title_name}", fontsize=13)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=9, loc="upper left")
        self._style_axes(ax)
        fig.tight_layout()
        return fig
