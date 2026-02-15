"""Tests for adaptive AI detector and stability analysis."""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.figure import Figure

from hawks.detection.adaptive import AdaptiveAIDetector
from hawks.analysis.adaptive_analysis import Regime, StabilityAnalyzer


# ---------------------------------------------------------------------------
# TestAdaptiveAIDetector
# ---------------------------------------------------------------------------


class TestAdaptiveAIDetector:
    def test_effective_accuracy_at_trust_half(self) -> None:
        """gamma*(0.5-0.5)=0, so p_t = p_base."""
        rng = np.random.default_rng(0)
        det = AdaptiveAIDetector(p_base=0.7, gamma=0.5, rng=rng)
        assert det.effective_accuracy(0.5) == pytest.approx(0.7)

    def test_high_trust_increases_accuracy(self) -> None:
        """trust=0.9 -> p_t > p_base when gamma > 0."""
        rng = np.random.default_rng(0)
        det = AdaptiveAIDetector(p_base=0.7, gamma=0.5, rng=rng)
        assert det.effective_accuracy(0.9) > 0.7

    def test_low_trust_decreases_accuracy(self) -> None:
        """trust=0.1 -> p_t < p_base when gamma > 0."""
        rng = np.random.default_rng(0)
        det = AdaptiveAIDetector(p_base=0.7, gamma=0.5, rng=rng)
        assert det.effective_accuracy(0.1) < 0.7

    def test_clamping_respects_bounds(self) -> None:
        """Extreme gamma doesn't exceed [min_p, max_p]."""
        rng = np.random.default_rng(0)
        det = AdaptiveAIDetector(p_base=0.5, gamma=10.0, rng=rng, min_p=0.05, max_p=0.95)
        # trust=1.0 -> p_base + 10*(1-0.5) = 5.5, clamped to 0.95
        assert det.effective_accuracy(1.0) == pytest.approx(0.95)
        # trust=0.0 -> p_base + 10*(0-0.5) = -4.5, clamped to 0.05
        assert det.effective_accuracy(0.0) == pytest.approx(0.05)

    def test_zero_gamma_is_constant(self) -> None:
        """gamma=0 -> p_t = p_base for any trust."""
        rng = np.random.default_rng(0)
        det = AdaptiveAIDetector(p_base=0.6, gamma=0.0, rng=rng)
        for trust in [0.0, 0.25, 0.5, 0.75, 1.0]:
            assert det.effective_accuracy(trust) == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# TestSimulateTrajectory
# ---------------------------------------------------------------------------


class TestSimulateTrajectory:
    def test_trajectory_length(self) -> None:
        """Returns n_steps + 1 entries (includes initial)."""
        sa = StabilityAnalyzer(master_seed=0)
        n_steps = 100
        trust_hist, acc_hist = sa.simulate_trajectory(0.7, 0.3, n_steps=n_steps)
        assert len(trust_hist) == n_steps + 1
        assert len(acc_hist) == n_steps + 1

    def test_trajectory_values_bounded(self) -> None:
        """All trust values in [0, 1], all p_t in [min_p, max_p]."""
        sa = StabilityAnalyzer(master_seed=0)
        trust_hist, acc_hist = sa.simulate_trajectory(0.7, 0.5, n_steps=200)
        for t in trust_hist:
            assert 0.0 <= t <= 1.0
        for a in acc_hist:
            assert 0.01 <= a <= 0.99

    def test_reproducibility(self) -> None:
        """Same seed -> identical trajectory."""
        sa1 = StabilityAnalyzer(master_seed=42)
        sa2 = StabilityAnalyzer(master_seed=42)
        t1, a1 = sa1.simulate_trajectory(0.7, 0.5, n_steps=50)
        t2, a2 = sa2.simulate_trajectory(0.7, 0.5, n_steps=50)
        assert t1 == t2
        assert a1 == a2

    def test_different_seeds_differ(self) -> None:
        """Different master_seed -> different trajectory."""
        sa1 = StabilityAnalyzer(master_seed=42)
        sa2 = StabilityAnalyzer(master_seed=99)
        t1, _ = sa1.simulate_trajectory(0.7, 0.5, n_steps=50)
        t2, _ = sa2.simulate_trajectory(0.7, 0.5, n_steps=50)
        assert t1 != t2


# ---------------------------------------------------------------------------
# TestClassifyRegime
# ---------------------------------------------------------------------------


class TestClassifyRegime:
    def test_constant_high_trust(self) -> None:
        """[0.9] * 300 -> HIGH_TRUST."""
        sa = StabilityAnalyzer()
        assert sa.classify_regime([0.9] * 300) == Regime.HIGH_TRUST

    def test_constant_low_trust(self) -> None:
        """[0.1] * 300 -> LOW_TRUST."""
        sa = StabilityAnalyzer()
        assert sa.classify_regime([0.1] * 300) == Regime.LOW_TRUST

    def test_oscillatory_variance(self) -> None:
        """Alternating 0.2/0.8 -> OSCILLATORY or UNSTABLE."""
        sa = StabilityAnalyzer()
        alternating = [0.2, 0.8] * 150
        regime = sa.classify_regime(alternating)
        assert regime in (Regime.OSCILLATORY, Regime.UNSTABLE)

    def test_classification_from_simulation(self) -> None:
        """Run a trajectory and verify classification is a valid Regime."""
        sa = StabilityAnalyzer(master_seed=0)
        trust_hist, _ = sa.simulate_trajectory(0.7, 0.3, n_steps=1000)
        regime = sa.classify_regime(trust_hist)
        assert isinstance(regime, Regime)


# ---------------------------------------------------------------------------
# TestSweep
# ---------------------------------------------------------------------------


class TestSweep:
    def test_sweep_returns_correct_shapes(self) -> None:
        """Grid dimensions match resolution."""
        sa = StabilityAnalyzer(master_seed=0)
        res = 5
        p_vals, g_vals, grid = sa.sweep(resolution=res, n_steps=100)
        assert len(p_vals) == res
        assert len(g_vals) == res
        assert grid.shape == (res, res)

    def test_sweep_all_regimes_valid(self) -> None:
        """All values in grid are valid Regime enum indices (0-3)."""
        sa = StabilityAnalyzer(master_seed=0)
        _, _, grid = sa.sweep(resolution=5, n_steps=100)
        assert np.all(grid >= 0)
        assert np.all(grid <= 3)


# ---------------------------------------------------------------------------
# TestPlots
# ---------------------------------------------------------------------------


class TestPlots:
    def test_bifurcation_heatmap_returns_figure(self) -> None:
        """Returns Figure (small resolution for speed)."""
        sa = StabilityAnalyzer(master_seed=0)
        fig = sa.plot_bifurcation_heatmap(resolution=5)
        assert isinstance(fig, Figure)
        plt_cleanup(fig)

    def test_trajectory_examples_returns_figure(self) -> None:
        """Returns Figure."""
        sa = StabilityAnalyzer(master_seed=0)
        fig = sa.plot_trajectory_examples(n_steps=100)
        assert isinstance(fig, Figure)
        plt_cleanup(fig)


# ---------------------------------------------------------------------------
# TestReport
# ---------------------------------------------------------------------------


class TestReport:
    def test_report_returns_dict_with_keys(self) -> None:
        """Has 'critical_gamma', 'stabilizing_regions', 'destabilizing_regions'."""
        sa = StabilityAnalyzer(master_seed=0)
        result = sa.report(resolution=5)
        assert "critical_gamma" in result
        assert "stabilizing_regions" in result
        assert "destabilizing_regions" in result

    def test_report_critical_gamma_values(self) -> None:
        """critical_gamma values are non-negative floats or None."""
        sa = StabilityAnalyzer(master_seed=0)
        result = sa.report(resolution=5)
        for p_base, cg in result["critical_gamma"].items():
            assert isinstance(p_base, float)
            if cg is not None:
                assert isinstance(cg, float)
                assert cg >= 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

import matplotlib.pyplot as plt


def plt_cleanup(fig: Figure) -> None:
    """Close a figure to avoid resource warnings."""
    plt.close(fig)
