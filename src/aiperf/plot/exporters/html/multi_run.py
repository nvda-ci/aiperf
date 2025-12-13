# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Multi-run HTML exporter for comparison plots.

Generates self-contained interactive HTML files for comparing multiple
profiling runs with embedded data and client-side JavaScript interactivity.
"""

from pathlib import Path

from aiperf.plot.constants import ALL_STAT_KEYS, STAT_LABELS
from aiperf.plot.core.data_loader import RunData
from aiperf.plot.core.plot_generator import PlotGenerator
from aiperf.plot.core.plot_specs import ExperimentClassificationConfig, PlotSpec
from aiperf.plot.dashboard.utils import runs_to_dataframe
from aiperf.plot.exporters.html.base import BaseHTMLExporter
from aiperf.plot.exporters.html.serializers import HTMLDataSerializer
from aiperf.plot.metric_names import get_metric_display_name


class MultiRunHTMLExporter(BaseHTMLExporter):
    """
    HTML exporter for multi-run comparison visualization.

    Generates interactive HTML with:
    - Pareto curves
    - Scatter/line comparison plots
    - Run selection controls
    - Dynamic metric/stat selection
    """

    def export(
        self,
        runs: list[RunData],
        available_metrics: dict,
        plot_specs: list[PlotSpec],
        classification_config: ExperimentClassificationConfig | None = None,
    ) -> list[Path]:
        """
        Export multi-run comparison data as interactive HTML.

        Args:
            runs: List of RunData objects with aggregated metrics
            available_metrics: Dictionary with display_names and units
            plot_specs: List of plot specifications
            classification_config: Optional experiment classification

        Returns:
            List containing path to generated HTML file
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        data = HTMLDataSerializer.serialize_multi_run(
            runs, available_metrics, classification_config
        )

        initial_plots = self._create_initial_plots(plot_specs, runs)

        figures_html = self._generate_figures(
            runs, initial_plots, classification_config
        )

        config = {
            "theme": self.theme.value,
            "mode": "multi",
        }

        serialized_metrics = HTMLDataSerializer.serialize_available_metrics(
            available_metrics
        )

        run_count = len(runs)
        first_model = runs[0].metadata.model if runs else "Unknown"
        header_info = f"{run_count} runs | {first_model}"

        swept_parameters = HTMLDataSerializer._detect_swept_parameters(runs)

        context = {
            "title": "Multi-Run Comparison",
            "theme": self.theme.value,
            "header_info": header_info,
            "runs": data["runs"],
            "available_metrics": serialized_metrics,
            "stat_keys": ALL_STAT_KEYS,
            "stat_labels": STAT_LABELS,
            "swept_parameters": swept_parameters,
            "initial_plots": initial_plots,
            "classification": classification_config is not None,
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
        runs: list[RunData],
    ) -> list[dict]:
        """
        Create initial plot configurations from plot specs.

        Args:
            plot_specs: List of plot specifications
            runs: List of RunData objects (to check metric availability)

        Returns:
            List of plot config dicts for JavaScript consumption
        """
        plots = []

        default_plots = [
            {
                "id": "pareto",
                "config": {
                    "title": "Pareto Curve: Throughput vs Latency",
                    "xMetric": "request_latency",
                    "yMetric": "output_token_throughput",
                    "xStat": "p50",
                    "yStat": "avg",
                    "plotType": "scatter_line",
                    "logScaleX": False,
                    "logScaleY": False,
                },
                "sizeClass": "half",
                "visible": True,
            },
            {
                "id": "ttft-vs-throughput",
                "config": {
                    "title": "TTFT vs Throughput",
                    "xMetric": "time_to_first_token",
                    "yMetric": "output_token_throughput",
                    "xStat": "p50",
                    "yStat": "avg",
                    "plotType": "scatter_line",
                    "logScaleX": False,
                    "logScaleY": False,
                },
                "sizeClass": "half",
                "visible": True,
            },
            {
                "id": "throughput-vs-concurrency",
                "config": {
                    "title": "Throughput vs Concurrency",
                    "xMetric": "concurrency",
                    "yMetric": "output_token_throughput",
                    "xStat": "value",
                    "yStat": "avg",
                    "plotType": "scatter_line",
                    "logScaleX": False,
                    "logScaleY": False,
                },
                "sizeClass": "half",
                "visible": True,
            },
            {
                "id": "latency-vs-concurrency",
                "config": {
                    "title": "Latency vs Concurrency",
                    "xMetric": "concurrency",
                    "yMetric": "request_latency",
                    "xStat": "value",
                    "yStat": "p50",
                    "plotType": "scatter_line",
                    "logScaleX": False,
                    "logScaleY": False,
                },
                "sizeClass": "half",
                "visible": True,
            },
        ]

        if plot_specs:
            for spec in plot_specs:
                if len(spec.metrics) >= 2:
                    x_metric = spec.metrics[0]
                    y_metric = spec.metrics[1]

                    plots.append(
                        {
                            "id": spec.name,
                            "config": {
                                "title": spec.title or spec.name,
                                "xMetric": x_metric.name,
                                "yMetric": y_metric.name,
                                "xStat": x_metric.stat or "p50",
                                "yStat": y_metric.stat or "avg",
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
        runs: list[RunData],
        plot_configs: list[dict],
        classification_config: ExperimentClassificationConfig | None = None,
    ) -> dict[str, str]:
        """
        Generate Plotly figures as HTML for each plot configuration.

        Uses the same PlotGenerator as the interactive dashboard to ensure
        visual parity. Returns pre-rendered HTML divs that can be embedded
        directly in the template.

        Args:
            runs: List of RunData objects
            plot_configs: List of plot configuration dicts
            classification_config: Optional experiment classification config

        Returns:
            Dict mapping plot ID to pre-rendered HTML div string
        """
        plot_gen = PlotGenerator(theme=self.theme)
        figures = {}

        # Determine grouping strategy based on classification
        group_by = "experiment_group" if classification_config is not None else "model"

        for plot_config in plot_configs:
            plot_id = plot_config["id"]
            config = plot_config["config"]

            try:
                x_metric = config["xMetric"]
                y_metric = config["yMetric"]
                x_stat = config.get("xStat", "p50")
                y_stat = config.get("yStat", "avg")

                result = runs_to_dataframe(
                    runs,
                    x_metric=x_metric,
                    x_stat=x_stat,
                    y_metric=y_metric,
                    y_stat=y_stat,
                )
                df = result["df"]

                if df.empty:
                    self.warning(f"No data for plot {plot_id}")
                    continue

                # Extract experiment_types mapping (like dashboard callbacks)
                experiment_types = None
                if "experiment_type" in df.columns and group_by in df.columns:
                    experiment_types = {
                        g: df[df[group_by] == g]["experiment_type"].iloc[0]
                        for g in df[group_by].unique()
                    }

                # Build axis labels with stat (matching dashboard behavior)
                x_label = f"{get_metric_display_name(x_metric)} ({x_stat})"
                y_label = f"{get_metric_display_name(y_metric)} ({y_stat})"

                fig = plot_gen.create_scatter_line_plot(
                    df=df,
                    x_metric=x_metric,
                    y_metric=y_metric,
                    title=config.get("title", plot_id),
                    x_label=x_label,
                    y_label=y_label,
                    label_by="concurrency",
                    group_by=group_by,
                    experiment_types=experiment_types,
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
