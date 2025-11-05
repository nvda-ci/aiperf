# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Single-run PNG exporter for time series plots.

Generates static PNG images for analyzing a single profiling run.
"""

from pathlib import Path

import pandas as pd

from aiperf.plot.core.data_loader import RunData
from aiperf.plot.exporters.png.base import BasePNGExporter


class SingleRunPNGExporter(BasePNGExporter):
    """
    PNG exporter for single-run time series plots.

    Generates static PNG images for analyzing a single profiling run:
    1. TTFT over time (scatter)
    2. ITL over time (scatter)
    3. Latency over time (area)
    """

    def export(self, run: RunData, available_metrics: dict) -> list[Path]:
        """
        Export single-run time series plots as PNG files.

        Args:
            run: RunData object with per-request data
            available_metrics: Dictionary with display_names and units for metrics

        Returns:
            List of Path objects for generated PNG files
        """
        self.info("Generating single-run analysis plots")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        df = self._per_request_to_dataframe(run)

        if df.empty:
            self.warning("No per-request data available for plotting")
            return []

        generated_files = []

        # Plot 1: TTFT over time
        generated_files.extend(self._generate_ttft_over_time(df, available_metrics))

        # Plot 2: ITL over time
        generated_files.extend(self._generate_itl_over_time(df, available_metrics))

        # Plot 3: Latency over time
        generated_files.extend(self._generate_latency_over_time(df, available_metrics))

        self._create_summary_file(generated_files)

        return generated_files

    def _generate_ttft_over_time(
        self, df: pd.DataFrame, available_metrics: dict
    ) -> list[Path]:
        """Generate TTFT over time plot."""
        try:
            fig = self.plot_generator.create_time_series_scatter(
                df=df,
                x_col="request_number",
                y_metric="time_to_first_token",
                title="TTFT Over Time",
                x_label="Request Number",
                y_label=self._get_metric_label(
                    "time_to_first_token", None, available_metrics
                ),
            )
            path = self.output_dir / "ttft_over_time.png"
            self._export_figure(fig, path)
            self.info("✓ Generated ttft_over_time.png")
            return [path]
        except Exception as e:
            self.error(f"Failed to generate TTFT over time: {e}")
            return []

    def _generate_itl_over_time(
        self, df: pd.DataFrame, available_metrics: dict
    ) -> list[Path]:
        """Generate ITL over time plot."""
        try:
            fig = self.plot_generator.create_time_series_scatter(
                df=df,
                x_col="request_number",
                y_metric="inter_token_latency",
                title="Inter-Token Latency Over Time",
                x_label="Request Number",
                y_label=self._get_metric_label(
                    "inter_token_latency", None, available_metrics
                ),
            )
            path = self.output_dir / "itl_over_time.png"
            self._export_figure(fig, path)
            self.info("✓ Generated itl_over_time.png")
            return [path]
        except Exception as e:
            self.error(f"Failed to generate ITL over time: {e}")
            return []

    def _generate_latency_over_time(
        self, df: pd.DataFrame, available_metrics: dict
    ) -> list[Path]:
        """Generate latency over time plot."""
        try:
            fig = self.plot_generator.create_time_series_area(
                df=df,
                x_col="timestamp",
                y_metric="request_latency",
                title="Request Latency Over Time",
                x_label="Time (seconds)",
                y_label=self._get_metric_label(
                    "request_latency", None, available_metrics
                ),
            )
            path = self.output_dir / "latency_over_time.png"
            self._export_figure(fig, path)
            self.info("✓ Generated latency_over_time.png")
            return [path]
        except Exception as e:
            self.error(f"Failed to generate latency over time: {e}")
            return []

    def _per_request_to_dataframe(self, run: RunData) -> pd.DataFrame:
        """
        Convert per-request data into a DataFrame for time series plotting.

        Args:
            run: RunData object with requests DataFrame

        Returns:
            DataFrame with per-request metrics
        """
        if run.requests is None or run.requests.empty:
            return pd.DataFrame()

        df = run.requests.copy()

        df["request_number"] = range(len(df))

        if "request_end_ns" in df.columns:
            df["timestamp"] = df["request_end_ns"] / 1_000_000_000

            if len(df) > 0:
                df["timestamp"] = df["timestamp"] - df["timestamp"].min()

        return df
