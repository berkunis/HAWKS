"""Experiment runner — batch execution and parameter sweeps."""

from __future__ import annotations

import copy
import itertools
from typing import Any

import pandas as pd

from hawks.config import HAWKSConfig
from hawks.engine.metrics import MetricsCollector
from hawks.engine.simulation import SimulationEngine


class ExperimentRunner:
    """Batch execution and parameter sweeps over simulation configurations."""

    def run_single(self, config: HAWKSConfig) -> MetricsCollector:
        """Run a single simulation and return the metrics collector."""
        engine = SimulationEngine(config)
        return engine.run()

    def run_sweep(
        self, base_config: HAWKSConfig, param_grid: dict[str, list[Any]]
    ) -> pd.DataFrame:
        """Run Cartesian product over parameter grid, aggregate results.

        param_grid maps dot-separated config paths to lists of values.
        Example: {"detection.true_positive_rate": [0.7, 0.85, 0.95]}
        """
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        all_results = []

        for combo in itertools.product(*values):
            config = copy.deepcopy(base_config)
            params = dict(zip(keys, combo))

            # Apply parameter overrides
            for path, value in params.items():
                _set_nested_attr(config, path, value)

            metrics = self.run_single(config)
            summary = metrics.summary()
            summary.update(params)
            all_results.append(summary)

        return pd.DataFrame(all_results)

    def run_replications(
        self, config: HAWKSConfig, n_reps: int
    ) -> pd.DataFrame:
        """Run same config with different seeds, collect results."""
        all_results = []

        for rep in range(n_reps):
            cfg = copy.deepcopy(config)
            cfg.master_seed = config.master_seed + rep
            metrics = self.run_single(cfg)
            summary = metrics.summary()
            summary["replication"] = rep
            summary["seed"] = cfg.master_seed
            all_results.append(summary)

        return pd.DataFrame(all_results)


def _set_nested_attr(obj: Any, path: str, value: Any) -> None:
    """Set a nested attribute using dot-separated path."""
    parts = path.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)
