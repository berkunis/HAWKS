"""Compare surrogate MLP steady-state to analytical equilibrium trust.

Rolls out the trained TrustSurrogate for 500 steps (expected value over
correctness) and checks that the resulting steady-state matches the
closed-form T*(p) = pα / (pα + (1-p)β) for every archetype × accuracy pair.

Outputs
-------
- Summary table to stdout with per-pair absolute error and overall MAE.
- Two-panel figure saved to docs/figures/surrogate_vs_equilibrium.png.
"""

from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from hawks.analysis.trust_analysis import TrustAnalyzer, _ARCHETYPE_COLORS
from hawks.models.surrogate import TrustSurrogate
from hawks.operators.population import ARCHETYPES

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "surrogate.pt"
FIGURE_PATH = PROJECT_ROOT / "docs" / "figures" / "surrogate_vs_equilibrium.png"

STRUCTURAL_RISK = 0.3
N_STEPS = 500
P_VALUES = [round(0.1 * i, 1) for i in range(1, 10)]  # 0.1 … 0.9
TRAJECTORY_P = 0.7  # p value used for the rollout panel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_model() -> TrustSurrogate:
    model = TrustSurrogate()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()
    return model


@torch.no_grad()
def _rollout(
    model: TrustSurrogate,
    p: float,
    initial_trust: float,
    n_steps: int = N_STEPS,
) -> list[float]:
    """Roll out the surrogate for *n_steps*, returning the full trajectory.

    At each step the expected next trust is:
        T_{t+1} = p · model(sr, p, 1, T_t) + (1-p) · model(sr, p, 0, T_t)
    """
    trajectory = [initial_trust]
    T = initial_trust
    for _ in range(n_steps):
        feat_correct = torch.tensor(
            [STRUCTURAL_RISK, p, 1.0, T], dtype=torch.float32,
        )
        feat_incorrect = torch.tensor(
            [STRUCTURAL_RISK, p, 0.0, T], dtype=torch.float32,
        )
        T = float(
            p * model(feat_correct.unsqueeze(0))
            + (1 - p) * model(feat_incorrect.unsqueeze(0))
        )
        trajectory.append(T)
    return trajectory


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    model = _load_model()
    analyzer = TrustAnalyzer()

    # ---- Collect results for all (archetype, p) pairs ----
    rows: list[dict] = []
    trajectories_at_p07: dict[str, list[float]] = {}

    for name, params in ARCHETYPES.items():
        alpha, beta = params["alpha"], params["beta"]
        initial_trust = params["initial_trust"]

        for p in P_VALUES:
            traj = _rollout(model, p, initial_trust)
            t_star_analytical = analyzer.equilibrium_trust(alpha, beta, p)
            t_star_surrogate = traj[-1]
            error = abs(t_star_surrogate - t_star_analytical)

            rows.append({
                "archetype": name,
                "p": p,
                "T*_analytical": t_star_analytical,
                "T*_surrogate": t_star_surrogate,
                "error": error,
            })

            if p == TRAJECTORY_P:
                trajectories_at_p07[name] = traj

    # ---- Print summary table ----
    header = f"{'Archetype':<28s} {'p':>4s}  {'T*_anal':>8s}  {'T*_surr':>8s}  {'|err|':>8s}"
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['archetype']:<28s} {r['p']:4.1f}  "
            f"{r['T*_analytical']:8.4f}  {r['T*_surrogate']:8.4f}  "
            f"{r['error']:8.4f}"
        )
    mae = sum(r["error"] for r in rows) / len(rows)
    print("-" * len(header))
    print(f"Mean Absolute Error (MAE): {mae:.6f}")
    print("=" * len(header))

    # ---- Two-panel figure ----
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 5))

    # Left panel — rollout trajectories at p = TRAJECTORY_P
    for name, traj in trajectories_at_p07.items():
        alpha, beta = ARCHETYPES[name]["alpha"], ARCHETYPES[name]["beta"]
        t_star = analyzer.equilibrium_trust(alpha, beta, TRAJECTORY_P)
        color = _ARCHETYPE_COLORS[name]
        label = name.replace("_", " ").title()

        ax_left.plot(range(len(traj)), traj, color=color, linewidth=1.5, label=label)
        ax_left.axhline(t_star, color=color, linestyle="--", alpha=0.5, linewidth=1.0)

    ax_left.set_xlabel("Rollout Step", fontsize=12)
    ax_left.set_ylabel("Trust (T)", fontsize=12)
    ax_left.set_title(f"Surrogate Rollout to Steady State (p = {TRAJECTORY_P})", fontsize=13)
    ax_left.set_xlim(0, N_STEPS)
    ax_left.set_ylim(0, 1)
    ax_left.legend(fontsize=9, loc="center right")
    ax_left.spines["top"].set_visible(False)
    ax_left.spines["right"].set_visible(False)
    ax_left.grid(True, alpha=0.3, color="lightgray")
    ax_left.tick_params(labelsize=10)

    # Right panel — |error| vs accuracy
    for name in ARCHETYPES:
        errors = [r["error"] for r in rows if r["archetype"] == name]
        ps = [r["p"] for r in rows if r["archetype"] == name]
        color = _ARCHETYPE_COLORS[name]
        label = name.replace("_", " ").title()
        ax_right.plot(ps, errors, "o-", color=color, linewidth=1.5, markersize=5, label=label)

    ax_right.set_xlabel("AI Accuracy (p)", fontsize=12)
    ax_right.set_ylabel("|T*_surrogate \u2212 T*_analytical|", fontsize=12)
    ax_right.set_title("Surrogate vs Analytical Equilibrium Error", fontsize=13)
    ax_right.set_xlim(0, 1)
    ax_right.legend(fontsize=9, loc="upper right")
    ax_right.spines["top"].set_visible(False)
    ax_right.spines["right"].set_visible(False)
    ax_right.grid(True, alpha=0.3, color="lightgray")
    ax_right.tick_params(labelsize=10)

    fig.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=150)
    plt.close(fig)
    print(f"\nFigure saved to {FIGURE_PATH}")


if __name__ == "__main__":
    main()
