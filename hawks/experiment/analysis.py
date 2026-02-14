"""Results analyzer — post-hoc visualization and statistical analysis."""

from __future__ import annotations

from typing import Any

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure


class ResultsAnalyzer:
    """Post-hoc visualization and statistical analysis of simulation results."""

    def plot_trust_evolution(self, df: pd.DataFrame) -> Figure:
        """Plot trust trajectories over time, per operator."""
        fig, ax = plt.subplots(figsize=(10, 6))
        decided = df.dropna(subset=["trust_level"])

        for op_id, group in decided.groupby("operator_id"):
            group_sorted = group.sort_values("step")
            ax.plot(group_sorted["step"], group_sorted["trust_level"], alpha=0.5, label=op_id)

        ax.set_xlabel("Simulation Step")
        ax.set_ylabel("Trust Level")
        ax.set_title("Operator Trust Evolution Over Time")
        ax.set_ylim(0, 1)
        if len(decided["operator_id"].unique()) <= 10:
            ax.legend(fontsize="small")
        plt.tight_layout()
        return fig

    def plot_detection_performance(self, df: pd.DataFrame) -> Figure:
        """Plot detection performance metrics over time."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Per-step detection rate
        step_stats = df.groupby("step").agg(
            detections=("detections_count", "sum"),
            missed=("missed_count", "sum"),
        ).reset_index()

        if not step_stats.empty and (step_stats["detections"] + step_stats["missed"]).sum() > 0:
            step_stats["recall"] = step_stats["detections"] / (
                step_stats["detections"] + step_stats["missed"]
            ).replace(0, np.nan)

            axes[0].plot(step_stats["step"], step_stats["recall"])
            axes[0].set_xlabel("Step")
            axes[0].set_ylabel("Detection Recall")
            axes[0].set_title("Detection Recall Over Time")
            axes[0].set_ylim(0, 1)

        # Confidence distribution for TP vs FP
        decided = df.dropna(subset=["confidence"])
        if not decided.empty:
            tp = decided[decided["is_true_positive"] == True]["confidence"]
            fp = decided[decided["is_true_positive"] == False]["confidence"]
            if len(tp) > 0:
                axes[1].hist(tp, bins=20, alpha=0.6, label="True Positive", density=True)
            if len(fp) > 0:
                axes[1].hist(fp, bins=20, alpha=0.6, label="False Positive", density=True)
            axes[1].set_xlabel("Confidence")
            axes[1].set_ylabel("Density")
            axes[1].set_title("Confidence Distribution: TP vs FP")
            axes[1].legend()

        plt.tight_layout()
        return fig

    def plot_operator_comparison(self, df: pd.DataFrame) -> Figure:
        """Plot cross-operator performance comparison."""
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        decided = df.dropna(subset=["operator_id"])

        if decided.empty:
            return fig

        op_stats = decided.groupby("operator_id").agg(
            accuracy=("correct", "mean"),
            mean_trust=("trust_level", "mean"),
            mean_fatigue=("fatigue_level", "mean"),
            mean_response_time=("response_time", "mean"),
        ).reset_index()

        # Accuracy
        axes[0].bar(range(len(op_stats)), op_stats["accuracy"])
        axes[0].set_xlabel("Operator")
        axes[0].set_ylabel("Accuracy")
        axes[0].set_title("Operator Accuracy")
        axes[0].set_ylim(0, 1)

        # Trust vs Fatigue scatter
        axes[1].scatter(op_stats["mean_trust"], op_stats["mean_fatigue"], alpha=0.7)
        axes[1].set_xlabel("Mean Trust")
        axes[1].set_ylabel("Mean Fatigue")
        axes[1].set_title("Trust vs Fatigue by Operator")

        # Response time
        axes[2].bar(range(len(op_stats)), op_stats["mean_response_time"])
        axes[2].set_xlabel("Operator")
        axes[2].set_ylabel("Response Time (s)")
        axes[2].set_title("Mean Response Time")

        plt.tight_layout()
        return fig

    def compute_statistics(self, df: pd.DataFrame) -> dict[str, Any]:
        """Compute summary statistics with confidence intervals."""
        decided = df.dropna(subset=["action"])
        result: dict[str, Any] = {}

        if decided.empty:
            return result

        # Accuracy with 95% CI (normal approximation)
        n = len(decided)
        acc = float(decided["correct"].mean())
        se = (acc * (1 - acc) / n) ** 0.5 if n > 0 else 0
        result["accuracy"] = {"mean": acc, "ci_95": (acc - 1.96 * se, acc + 1.96 * se), "n": n}

        # Trust statistics
        trust = decided["trust_level"]
        result["trust"] = {
            "mean": float(trust.mean()),
            "std": float(trust.std()),
            "min": float(trust.min()),
            "max": float(trust.max()),
        }

        # Fatigue statistics
        fatigue = decided["fatigue_level"]
        result["fatigue"] = {
            "mean": float(fatigue.mean()),
            "std": float(fatigue.std()),
            "min": float(fatigue.min()),
            "max": float(fatigue.max()),
        }

        # Response time statistics
        rt = decided["response_time"]
        result["response_time"] = {
            "mean": float(rt.mean()),
            "std": float(rt.std()),
            "median": float(rt.median()),
        }

        # Action distribution
        result["action_distribution"] = decided["action"].value_counts(normalize=True).to_dict()

        return result
