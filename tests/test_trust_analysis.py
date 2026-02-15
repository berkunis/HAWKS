"""Tests for the analytical trust-dynamics module."""

from __future__ import annotations

import pytest
from matplotlib.figure import Figure

from hawks.analysis.trust_analysis import TrustAnalyzer
from hawks.operators.population import ARCHETYPES


@pytest.fixture
def analyzer() -> TrustAnalyzer:
    return TrustAnalyzer()


# ---------------------------------------------------------------------------
# TestEquilibriumTrust
# ---------------------------------------------------------------------------


class TestEquilibriumTrust:
    def test_perfect_ai_trust_approaches_one(self, analyzer: TrustAnalyzer) -> None:
        for params in ARCHETYPES.values():
            t_star = analyzer.equilibrium_trust(params["alpha"], params["beta"], 1.0)
            assert t_star == pytest.approx(1.0)

    def test_zero_accuracy_trust_is_zero(self, analyzer: TrustAnalyzer) -> None:
        for params in ARCHETYPES.values():
            t_star = analyzer.equilibrium_trust(params["alpha"], params["beta"], 0.0)
            assert t_star == pytest.approx(0.0)

    def test_symmetric_rates_at_half(self, analyzer: TrustAnalyzer) -> None:
        t_star = analyzer.equilibrium_trust(alpha=0.1, beta=0.1, p=0.5)
        assert t_star == pytest.approx(0.5)

    def test_equilibrium_in_valid_range(self, analyzer: TrustAnalyzer) -> None:
        for params in ARCHETYPES.values():
            for p in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
                t_star = analyzer.equilibrium_trust(params["alpha"], params["beta"], p)
                assert 0.0 <= t_star <= 1.0

    def test_monotonically_increasing(self, analyzer: TrustAnalyzer) -> None:
        for params in ARCHETYPES.values():
            prev = analyzer.equilibrium_trust(params["alpha"], params["beta"], 0.0)
            for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                curr = analyzer.equilibrium_trust(params["alpha"], params["beta"], p)
                assert curr >= prev
                prev = curr


# ---------------------------------------------------------------------------
# TestCriticalAccuracy
# ---------------------------------------------------------------------------


class TestCriticalAccuracy:
    def test_symmetric_critical_is_half(self, analyzer: TrustAnalyzer) -> None:
        p_crit = analyzer.critical_accuracy(alpha=0.1, beta=0.1)
        assert p_crit == pytest.approx(0.5)

    def test_high_beta_shifts_right(self, analyzer: TrustAnalyzer) -> None:
        p_low = analyzer.critical_accuracy(alpha=0.1, beta=0.1)
        p_high = analyzer.critical_accuracy(alpha=0.1, beta=0.3)
        assert p_high > p_low

    def test_all_archetypes_critical_in_range(self, analyzer: TrustAnalyzer) -> None:
        for params in ARCHETYPES.values():
            p_crit = analyzer.critical_accuracy(params["alpha"], params["beta"])
            assert 0.0 < p_crit < 1.0

    def test_known_critical_values(self, analyzer: TrustAnalyzer) -> None:
        """Verify p_critical matches hand-calculated values from the plan."""
        expected = {
            "conservative_skeptic": 0.20 / (0.05 + 0.20),      # 0.80
            "calibrated_professional": 0.10 / (0.10 + 0.10),    # 0.50
            "automation_biased": 0.03 / (0.15 + 0.03),          # 0.1667
            "algorithm_averse": 0.25 / (0.03 + 0.25),           # 0.8929
        }
        for name, expected_val in expected.items():
            params = ARCHETYPES[name]
            p_crit = analyzer.critical_accuracy(params["alpha"], params["beta"])
            assert p_crit == pytest.approx(expected_val, abs=1e-4)


# ---------------------------------------------------------------------------
# TestScrapRate
# ---------------------------------------------------------------------------


class TestScrapRate:
    def test_perfect_ai_low_scrap(self, analyzer: TrustAnalyzer) -> None:
        scrap = analyzer.expected_scrap_rate(1.0)
        assert scrap < 0.3

    def test_scrap_rate_in_range(self, analyzer: TrustAnalyzer) -> None:
        for p in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
            scrap = analyzer.expected_scrap_rate(p)
            assert 0.0 <= scrap <= 1.0

    def test_scrap_rate_decreases_with_accuracy(self, analyzer: TrustAnalyzer) -> None:
        prev = analyzer.expected_scrap_rate(0.0)
        for p in [0.2, 0.4, 0.6, 0.8, 1.0]:
            curr = analyzer.expected_scrap_rate(p)
            assert curr <= prev + 1e-9  # allow tiny float tolerance
            prev = curr


# ---------------------------------------------------------------------------
# TestPlots
# ---------------------------------------------------------------------------


class TestPlots:
    def test_phase_diagram_returns_figure(self, analyzer: TrustAnalyzer) -> None:
        fig = analyzer.plot_phase_diagram()
        assert isinstance(fig, Figure)

    def test_scrap_rate_returns_figure(self, analyzer: TrustAnalyzer) -> None:
        fig = analyzer.plot_scrap_rate()
        assert isinstance(fig, Figure)

    def test_beta_sensitivity_returns_figure(self, analyzer: TrustAnalyzer) -> None:
        fig = analyzer.plot_beta_sensitivity()
        assert isinstance(fig, Figure)
