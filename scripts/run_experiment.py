"""CLI entry point for running HAWKS experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hawks.config import HAWKSConfig
from hawks.analysis.runner import ExperimentRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HAWKS simulation experiment")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--replications",
        type=int,
        default=1,
        help="Number of replications with different seeds",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file path for results",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        sys.exit(1)

    config = HAWKSConfig.from_yaml(config_path)
    runner = ExperimentRunner()

    if args.replications > 1:
        print(f"Running {args.replications} replications...")
        results_df = runner.run_replications(config, args.replications)
    else:
        print("Running single simulation...")
        metrics = runner.run_single(config)
        results_df = metrics.to_dataframe()
        summary = metrics.summary()
        print("\n--- Simulation Summary ---")
        for key, value in summary.items():
            print(f"  {key}: {value}")

    if args.output:
        results_df.to_csv(args.output, index=False)
        print(f"\nResults saved to {args.output}")
    else:
        print(f"\nResults shape: {results_df.shape}")
        print(results_df.head())


if __name__ == "__main__":
    main()
