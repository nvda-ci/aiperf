# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Single-run HTML exporter for time series visualization.

Generates self-contained interactive HTML files for analyzing a single
profiling run with per-request data, timeslices, and aggregated metrics.
"""

from pathlib import Path

import pandas as pd

import aiperf.plot.handlers.single_run_handlers  # noqa: F401
from aiperf.plot.constants import ALL_STAT_KEYS, STAT_LABELS
from aiperf.plot.core.data_loader import RunData
from aiperf.plot.core.plot_generator import PlotGenerator
from aiperf.plot.core.plot_specs import DataSource, PlotSpec
from aiperf.plot.core.plot_type_handlers import PlotTypeHandlerFactory
from aiperf.plot.dashboard.utils import prepare_timeseries_dataframe
from aiperf.plot.exporters.html.base import BaseHTMLExporter
from aiperf.plot.exporters.html.serializers import HTMLDataSerializer


class SingleRunHTMLExporter(BaseHTMLExporter):
    """
    HTML exporter for single-run time series visualization.

    Generates interactive HTML with:
    - Per-request scatter plots
    - Timeslice histograms
    - GPU telemetry plots (if available)
    - Dynamic metric/stat selection
    """

    def _can_generate_plot(self, spec: PlotSpec, run: RunData) -> bool:
        """
        Check if a plot can be generated based on data availability.

        Args:
            spec: Plot specification
            run: RunData object

        Returns:
            True if the plot can be generated, False otherwise
        """
        for metric in spec.metrics:
            if metric.source == DataSource.REQUESTS and (
                run.requests is None or run.requests.empty
            ):
                return False
            if metric.source == DataSource.TIMESLICES and (
                run.timeslices is None or run.timeslices.empty
            ):
                return False
            if metric.source == DataSource.GPU_TELEMETRY and (
                run.gpu_telemetry is None or run.gpu_telemetry.empty
            ):
                return False
        return True

    def export(
        self,
        run: RunData,
        available_metrics: dict,
        plot_specs: list[PlotSpec],
    ) -> list[Path]:
        """
        Export single-run data as interactive HTML.

        Args:
            run: RunData object with per-request data
            available_metrics: Dictionary with display_names and units
            plot_specs: List of plot specifications

        Returns:
            List containing path to generated HTML file
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        data = HTMLDataSerializer.serialize_single_run(run, available_metrics)

        initial_plots = self._create_initial_plots(plot_specs, run)

        figures_html = self._generate_figures(
            run, initial_plots, plot_specs, available_metrics
        )

        config = {
            "theme": self.theme.value,
            "mode": "single",
        }

        serialized_metrics = HTMLDataSerializer.serialize_available_metrics(
            available_metrics
        )

        run_name = run.metadata.run_name
        model = run.metadata.model or "Unknown"
        concurrency = run.metadata.concurrency or "?"
        request_count = run.metadata.request_count or 0
        header_info = (
            f"{run_name} | {model} | C{concurrency} | {request_count} requests"
        )

        run_metadata = {
            "runName": run.metadata.run_name,
            "model": run.metadata.model,
            "concurrency": run.metadata.concurrency,
            "requestCount": run.metadata.request_count,
            "durationSeconds": run.metadata.duration_seconds,
        }

        context = {
            "title": f"Single Run: {run_name}",
            "theme": self.theme.value,
            "header_info": header_info,
            "run_metadata": run_metadata,
            "available_metrics": serialized_metrics,
            "stat_keys": ALL_STAT_KEYS,
            "stat_labels": STAT_LABELS,
            "swept_parameters": [],
            "initial_plots": initial_plots,
            "classification": False,
            "plotly_js": self._get_plotly_js(),
            "css_bundle": self._get_css_bundle(),
            "js_bundle": self._get_js_bundle(),
            "data_json": HTMLDataSerializer.to_json_string(data),
            "initial_plots_json": HTMLDataSerializer.to_json_string(initial_plots),
            "config_json": HTMLDataSerializer.to_json_string(config),
            "figures_html": figures_html,
        }

        html_content = self._render_template("content.html.j2", context)

        output_path = self._write_html_file(html_content, "dashboard.html")

        return [output_path]

    def _create_initial_plots(
        self,
        plot_specs: list[PlotSpec],
        run: RunData,
    ) -> list[dict]:
        """
        Create initial plot configurations from plot specs.

        Args:
            plot_specs: List of plot specifications
            run: RunData object (to check metric availability)

        Returns:
            List of plot config dicts for JavaScript consumption
        """
        plots = []

        default_plots = [
            {
                "id": "ttft-over-time",
                "config": {
                    "title": "Time to First Token Over Time",
                    "xMetric": "request_number",
                    "yMetric": "time_to_first_token",
                    "plotType": "scatter",
                    "logScaleX": False,
                    "logScaleY": False,
                },
                "sizeClass": "half",
                "visible": True,
            },
            {
                "id": "itl-over-time",
                "config": {
                    "title": "Inter-Token Latency Over Time",
                    "xMetric": "request_number",
                    "yMetric": "inter_token_latency_avg",
                    "plotType": "scatter",
                    "logScaleX": False,
                    "logScaleY": False,
                },
                "sizeClass": "half",
                "visible": True,
            },
            {
                "id": "latency-over-time",
                "config": {
                    "title": "Request Latency Over Time",
                    "xMetric": "request_number",
                    "yMetric": "request_latency",
                    "plotType": "scatter",
                    "logScaleX": False,
                    "logScaleY": False,
                },
                "sizeClass": "half",
                "visible": True,
            },
            {
                "id": "osl-over-time",
                "config": {
                    "title": "Output Sequence Length Over Time",
                    "xMetric": "request_number",
                    "yMetric": "output_sequence_length",
                    "plotType": "scatter",
                    "logScaleX": False,
                    "logScaleY": False,
                },
                "sizeClass": "half",
                "visible": True,
            },
        ]

        if plot_specs:
            for spec in plot_specs:
                if len(spec.metrics) >= 1:
                    y_metric = spec.metrics[-1]
                    x_metric = spec.metrics[0] if len(spec.metrics) >= 2 else None

                    plots.append(
                        {
                            "id": spec.name,
                            "config": {
                                "title": spec.title or spec.name,
                                "xMetric": x_metric.name
                                if x_metric
                                else "request_number",
                                "yMetric": y_metric.name,
                                "plotType": spec.plot_type.value,
                                "logScaleX": False,
                                "logScaleY": False,
                            },
                            "sizeClass": "half",
                            "visible": True,
                        }
                    )
        else:
            plots = default_plots

        return plots

    def _generate_figures(
        self,
        run: RunData,
        plot_configs: list[dict],
        plot_specs: list[PlotSpec] | None,
        available_metrics: dict,
    ) -> dict[str, str]:
        """
        Generate Plotly figures as HTML for each plot configuration.

        Uses plot type handlers for specs that define them (e.g., dual-axis GPU plots),
        and falls back to simple scatter plots for default configurations.

        Args:
            run: RunData object with per-request data
            plot_configs: List of plot configuration dicts
            plot_specs: Original PlotSpec objects (for handler-based generation)
            available_metrics: Dictionary with display_names and units

        Returns:
            Dict mapping plot ID to pre-rendered HTML div string
        """
        plot_gen = PlotGenerator(theme=self.theme)
        figures = {}

        spec_lookup = {s.name: s for s in (plot_specs or [])}

        df = None
        x_col = None
        if run.requests is not None and not run.requests.empty:
            df, x_col = prepare_timeseries_dataframe(run.requests)

        for plot_config in plot_configs:
            plot_id = plot_config["id"]
            config = plot_config["config"]

            try:
                spec = spec_lookup.get(plot_id)

                if spec and self._can_generate_plot(spec, run):
                    handler = PlotTypeHandlerFactory.create_instance(
                        spec.plot_type,
                        plot_generator=plot_gen,
                    )
                    fig = handler.create_plot(spec, run, available_metrics)
                else:
                    if df is None:
                        self.warning(
                            f"No per-request data available for plot {plot_id}"
                        )
                        continue

                    y_metric = config["yMetric"]
                    x_metric = config.get("xMetric", "request_number")
                    plot_type = config.get("plotType", "scatter")

                    if y_metric not in df.columns:
                        self.warning(
                            f"Metric {y_metric} not found in data for plot {plot_id}"
                        )
                        continue

                    if plot_type == "request_timeline":
                        timeline_df = self._prepare_timeline_data(df, y_metric)
                        if timeline_df is None:
                            self.warning(
                                f"Could not prepare timeline data for plot {plot_id}"
                            )
                            continue

                        fig = plot_gen.create_request_timeline(
                            df=timeline_df,
                            y_metric=y_metric,
                            title=config.get("title", plot_id),
                        )
                    else:
                        actual_x_col = (
                            x_col if x_metric == "request_number" else x_metric
                        )
                        fig = plot_gen.create_time_series_scatter(
                            df=df,
                            x_col=actual_x_col,
                            y_metric=y_metric,
                            title=config.get("title", plot_id),
                        )

                figures[plot_id] = fig.to_html(
                    full_html=False,
                    include_plotlyjs=False,
                    div_id=f"plot-{plot_id}",
                )

            except Exception as e:
                self.warning(f"Failed to generate figure for {plot_id}: {e}")
                continue

        return figures

    def _prepare_timeline_data(
        self,
        df: pd.DataFrame,
        y_metric: str,
    ) -> pd.DataFrame | None:
        """
        Prepare timeline data with phase calculations.

        Args:
            df: DataFrame with per-request data
            y_metric: Name of the metric to plot on Y-axis

        Returns:
            DataFrame with columns: request_id, y_value, start_s, ttft_end_s, end_s
            or None if required columns are missing
        """
        required_cols = [
            "request_start_ns",
            "request_end_ns",
            "time_to_first_token",
            y_metric,
        ]

        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            self.warning(f"Timeline plot missing required columns: {missing_cols}")
            return None

        timeline_df = df.dropna(subset=required_cols).copy()

        if timeline_df.empty:
            return None

        start_min = timeline_df["request_start_ns"].min()
        timeline_df["start_s"] = (timeline_df["request_start_ns"] - start_min) / 1e9
        timeline_df["end_s"] = (timeline_df["request_end_ns"] - start_min) / 1e9

        timeline_df["ttft_s"] = timeline_df["time_to_first_token"] / 1000.0
        timeline_df["ttft_end_s"] = timeline_df["start_s"] + timeline_df["ttft_s"]

        timeline_df["duration_s"] = timeline_df["end_s"] - timeline_df["start_s"]
        timeline_df["has_valid_phases"] = (
            timeline_df["ttft_s"] <= timeline_df["duration_s"]
        )

        invalid_count = (~timeline_df["has_valid_phases"]).sum()
        if invalid_count > 0:
            self.warning(
                f"Filtered {invalid_count} requests where TTFT exceeds total duration"
            )

        timeline_df = timeline_df[timeline_df["has_valid_phases"]]

        if timeline_df.empty:
            return None

        timeline_df["request_id"] = range(len(timeline_df))
        timeline_df["y_value"] = timeline_df[y_metric]

        return timeline_df[["request_id", "y_value", "start_s", "ttft_end_s", "end_s"]]
