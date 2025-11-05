# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Multi-run PNG exporter for comparison plots.

Generates static PNG images comparing multiple profiling runs.
"""

from pathlib import Path

import pandas as pd

from aiperf.plot.constants import DEFAULT_PERCENTILE, NON_METRIC_KEYS
from aiperf.plot.core.data_loader import RunData
from aiperf.plot.exporters.png.base import BasePNGExporter


class MultiRunPNGExporter(BasePNGExporter):
    """
    PNG exporter for multi-run comparison plots.

    Generates static PNG images comparing multiple profiling runs:
    1. Pareto curve (latency vs throughput)
    2. TTFT vs Throughput
    3. Throughput per User vs Concurrency
    4. Token Throughput per GPU vs Latency (conditional on telemetry)
    5. Token Throughput per GPU vs Interactivity (conditional on telemetry)
    """

    def export(self, runs: list[RunData], available_metrics: dict) -> list[Path]:
        """
        Export multi-run comparison plots as PNG files.

        Args:
            runs: List of RunData objects with aggregated metrics
            available_metrics: Dictionary with display_names and units for metrics

        Returns:
            List of Path objects for generated PNG files
        """
        self.info(f"Generating multi-run comparison plots for {len(runs)} runs")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        df = self._runs_to_dataframe(runs, available_metrics)

        generated_files = []

        # Plot 1: Pareto Curve
        generated_files.extend(self._generate_pareto_curve(df, available_metrics))

        # Plot 2: TTFT vs Throughput
        generated_files.extend(self._generate_ttft_vs_throughput(df, available_metrics))

        # Plot 3: Throughput per User vs Concurrency
        generated_files.extend(
            self._generate_throughput_per_user_vs_concurrency(df, available_metrics)
        )

        # Plot 4: Token Throughput per GPU vs Latency (conditional)
        if "output_token_throughput_per_gpu" in df.columns:
            generated_files.extend(
                self._generate_throughput_per_gpu_vs_latency(df, available_metrics)
            )

        # Plot 5: Token Throughput per GPU vs Interactivity (conditional)
        if "output_token_throughput_per_gpu" in df.columns:
            generated_files.extend(
                self._generate_throughput_per_gpu_vs_interactivity(
                    df, available_metrics
                )
            )

        self._create_summary_file(generated_files)

        return generated_files

    def _generate_pareto_curve(
        self, df: pd.DataFrame, available_metrics: dict
    ) -> list[Path]:
        """Generate pareto curve plot."""
        try:
            fig = self.plot_generator.create_pareto_plot(
                df=df,
                x_metric="request_latency",
                y_metric="request_throughput",
                label_by="concurrency",
                group_by="model",
                title="Pareto Curve: Throughput vs Latency",
                x_label=self._get_metric_label(
                    "request_latency", DEFAULT_PERCENTILE, available_metrics
                ),
                y_label=self._get_metric_label(
                    "request_throughput", "avg", available_metrics
                ),
            )
            path = self.output_dir / "pareto_curve.png"
            self._export_figure(fig, path)
            self.info("✓ Generated pareto_curve.png")
            return [path]
        except Exception as e:
            self.error(f"Failed to generate pareto curve: {e}")
            return []

    def _generate_ttft_vs_throughput(
        self, df: pd.DataFrame, available_metrics: dict
    ) -> list[Path]:
        """Generate TTFT vs Throughput plot."""
        try:
            fig = self.plot_generator.create_scatter_line_plot(
                df=df,
                x_metric="time_to_first_token",
                y_metric="request_throughput",
                label_by="concurrency",
                group_by="model",
                title="TTFT vs Throughput",
                x_label=self._get_metric_label(
                    "time_to_first_token", DEFAULT_PERCENTILE, available_metrics
                ),
                y_label=self._get_metric_label(
                    "request_throughput", "avg", available_metrics
                ),
            )
            path = self.output_dir / "ttft_vs_throughput.png"
            self._export_figure(fig, path)
            self.info("✓ Generated ttft_vs_throughput.png")
            return [path]
        except Exception as e:
            self.error(f"Failed to generate TTFT vs throughput: {e}")
            return []

    def _generate_throughput_per_user_vs_concurrency(
        self, df: pd.DataFrame, available_metrics: dict
    ) -> list[Path]:
        """Generate Throughput per User vs Concurrency plot."""
        try:
            fig = self.plot_generator.create_scatter_line_plot(
                df=df,
                x_metric="concurrency",
                y_metric="output_token_throughput_per_user",
                label_by="concurrency",
                group_by="model",
                title="Output Token Throughput per User",
                x_label="Concurrency Level",
                y_label=self._get_metric_label(
                    "output_token_throughput_per_user", "avg", available_metrics
                ),
            )
            path = self.output_dir / "throughput_per_user_vs_concurrency.png"
            self._export_figure(fig, path)
            self.info("✓ Generated throughput_per_user_vs_concurrency.png")
            return [path]
        except Exception as e:
            self.error(f"Failed to generate throughput per user plot: {e}")
            return []

    def _generate_throughput_per_gpu_vs_latency(
        self, df: pd.DataFrame, available_metrics: dict
    ) -> list[Path]:
        """Generate Token Throughput per GPU vs Latency plot."""
        try:
            fig = self.plot_generator.create_pareto_plot(
                df=df,
                x_metric="request_latency",
                y_metric="output_token_throughput_per_gpu",
                label_by="concurrency",
                group_by="model",
                title="Token Throughput per GPU vs End-to-end Latency",
                x_label=self._get_metric_label(
                    "request_latency", DEFAULT_PERCENTILE, available_metrics
                ),
                y_label=self._get_metric_label(
                    "output_token_throughput_per_gpu", "avg", available_metrics
                ),
            )
            path = self.output_dir / "throughput_per_gpu_vs_latency.png"
            self._export_figure(fig, path)
            self.info("✓ Generated throughput_per_gpu_vs_latency.png")
            return [path]
        except Exception as e:
            self.error(f"Failed to generate throughput per GPU vs latency plot: {e}")
            return []

    def _generate_throughput_per_gpu_vs_interactivity(
        self, df: pd.DataFrame, available_metrics: dict
    ) -> list[Path]:
        """Generate Token Throughput per GPU vs Interactivity plot."""
        try:
            fig = self.plot_generator.create_scatter_line_plot(
                df=df,
                x_metric="output_token_throughput_per_gpu",
                y_metric="output_token_throughput_per_user",
                label_by="concurrency",
                group_by="model",
                title="Token Throughput per GPU vs Interactivity",
                x_label=self._get_metric_label(
                    "output_token_throughput_per_gpu", "avg", available_metrics
                ),
                y_label=self._get_metric_label(
                    "output_token_throughput_per_user", "avg", available_metrics
                ),
            )
            path = self.output_dir / "throughput_per_gpu_vs_interactivity.png"
            self._export_figure(fig, path)
            self.info("✓ Generated throughput_per_gpu_vs_interactivity.png")
            return [path]
        except Exception as e:
            self.error(
                f"Failed to generate throughput per GPU vs interactivity plot: {e}"
            )
            return []

    def _runs_to_dataframe(
        self, runs: list[RunData], available_metrics: dict
    ) -> pd.DataFrame:
        """
        Convert list of run data into a DataFrame for plotting.

        Args:
            runs: List of RunData objects
            available_metrics: Dictionary with display_names and units

        Returns:
            DataFrame with columns for metrics and metadata
        """
        from aiperf.common.models.record_models import MetricResult

        rows = []
        for run in runs:
            row = {}

            row["model"] = run.metadata.model or "Unknown"
            row["concurrency"] = run.metadata.concurrency or 1

            for key, value in run.aggregated.items():
                if key in NON_METRIC_KEYS:
                    continue

                if isinstance(value, MetricResult):
                    if (
                        hasattr(value, DEFAULT_PERCENTILE)
                        and getattr(value, DEFAULT_PERCENTILE) is not None
                    ):
                        row[key] = getattr(value, DEFAULT_PERCENTILE)
                    elif value.avg is not None:
                        row[key] = value.avg
                elif isinstance(value, dict) and "unit" in value and value is not None:
                    if DEFAULT_PERCENTILE in value:
                        row[key] = value[DEFAULT_PERCENTILE]
                    elif "avg" in value:
                        row[key] = value["avg"]
                    elif "value" in value:
                        row[key] = value["value"]

            rows.append(row)

        return pd.DataFrame(rows)
