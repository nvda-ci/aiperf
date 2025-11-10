# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Multi-run PNG exporter for comparison plots.

Generates static PNG images comparing multiple profiling runs.
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from aiperf.common.models.record_models import MetricResult
from aiperf.plot.constants import DEFAULT_PERCENTILE, NON_METRIC_KEYS
from aiperf.plot.core.data_loader import RunData
from aiperf.plot.core.plot_specs import MULTI_RUN_PLOT_SPECS, PlotSpec, PlotType
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

        # Generate all configured plots
        for spec in MULTI_RUN_PLOT_SPECS:
            try:
                # Check if we can generate this plot
                if not self._can_generate_plot(spec, df):
                    self.debug(f"Skipping {spec.name} - required columns not available")
                    continue

                # Create the plot from spec
                fig = self._create_plot_from_spec(spec, df, available_metrics)

                # Export to PNG
                path = self.output_dir / spec.filename
                self._export_figure(fig, path)
                self.info(f"✓ Generated {spec.filename}")
                generated_files.append(path)

            except Exception as e:
                self.error(f"Failed to generate {spec.name}: {e}")

        self._create_summary_file(generated_files)

        return generated_files

    def _can_generate_plot(self, spec: PlotSpec, df: pd.DataFrame) -> bool:
        """
        Check if a plot can be generated based on column availability.

        Args:
            spec: Plot specification
            df: DataFrame with aggregated metrics

        Returns:
            True if the plot can be generated, False otherwise
        """
        # Check that all required metric columns exist in the DataFrame
        for metric in spec.metrics:
            if metric.name not in df.columns and metric.name != "concurrency":
                return False
        return True

    def _create_plot_from_spec(
        self, spec: PlotSpec, df: pd.DataFrame, available_metrics: dict
    ) -> go.Figure:
        """
        Create a plot figure from a plot specification.

        Args:
            spec: Plot specification
            df: DataFrame with aggregated metrics
            available_metrics: Dictionary with display_names and units for metrics

        Returns:
            Plotly figure object
        """
        x_metric = next(m for m in spec.metrics if m.axis == "x")
        y_metric = next(m for m in spec.metrics if m.axis == "y")

        # Determine x and y labels
        if x_metric.name == "concurrency":
            x_label = "Concurrency Level"
        else:
            x_label = self._get_metric_label(
                x_metric.name, x_metric.stat or DEFAULT_PERCENTILE, available_metrics
            )

        y_label = self._get_metric_label(
            y_metric.name, y_metric.stat or "avg", available_metrics
        )

        if spec.plot_type == PlotType.PARETO:
            return self.plot_generator.create_pareto_plot(
                df=df,
                x_metric=x_metric.name,
                y_metric=y_metric.name,
                label_by=spec.label_by,
                group_by=spec.group_by,
                title=spec.title,
                x_label=x_label,
                y_label=y_label,
            )
        elif spec.plot_type == PlotType.SCATTER_LINE:
            return self.plot_generator.create_scatter_line_plot(
                df=df,
                x_metric=x_metric.name,
                y_metric=y_metric.name,
                label_by=spec.label_by,
                group_by=spec.group_by,
                title=spec.title,
                x_label=x_label,
                y_label=y_label,
            )
        else:
            raise ValueError(f"Unsupported plot type for multi-run: {spec.plot_type}")

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
