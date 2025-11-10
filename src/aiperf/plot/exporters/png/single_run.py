# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Single-run PNG exporter for time series plots.

Generates static PNG images for analyzing a single profiling run.
"""

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from aiperf.plot.core.data_loader import RunData
from aiperf.plot.core.plot_specs import (
    GPU_PLOT_SPECS,
    SINGLE_RUN_PLOT_SPECS,
    TIMESLICE_PLOT_SPECS,
    DataSource,
    PlotSpec,
    PlotType,
    TimeSlicePlotSpec,
)
from aiperf.plot.exporters.png.base import BasePNGExporter


class SingleRunPNGExporter(BasePNGExporter):
    """
    PNG exporter for single-run time series plots.

    Generates static PNG images for analyzing a single profiling run:
    1. TTFT over time (scatter)
    2. ITL over time (scatter)
    3. Latency over time (area)
    4. Timeslice plots (histograms) - if timeslice data available:
       - TTFT across time slices
       - ITL across time slices
       - Throughput across time slices
       - Latency across time slices
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

        generated_files = []

        # Generate all configured plots
        all_specs = SINGLE_RUN_PLOT_SPECS + TIMESLICE_PLOT_SPECS + GPU_PLOT_SPECS

        for spec in all_specs:
            try:
                # Check if we can generate this plot based on data availability
                if not self._can_generate_plot(spec, run):
                    self.debug(f"Skipping {spec.name} - required data not available")
                    continue

                # Create the plot from spec
                fig = self._create_plot_from_spec(spec, run, available_metrics)

                # Export to PNG
                path = self.output_dir / spec.filename
                self._export_figure(fig, path)
                self.info(f"✓ Generated {spec.filename}")
                generated_files.append(path)

            except Exception as e:
                self.error(f"Failed to generate {spec.name}: {e}")

        self._create_summary_file(generated_files)

        return generated_files

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
            if (
                metric.source == DataSource.REQUESTS
                and (run.requests is None or run.requests.empty)
                or metric.source == DataSource.TIMESLICES
                and (run.timeslices is None or run.timeslices.empty)
                or metric.source == DataSource.GPU_TELEMETRY
                and (run.gpu_telemetry is None or run.gpu_telemetry.empty)
            ):
                return False

        # Special validation for dispersed throughput plot
        if spec.name == "dispersed_throughput_over_time":
            if run.requests is None or run.requests.empty:
                return False
            required_cols = [
                "request_start_ns",
                "request_end_ns",
                "output_sequence_length",
            ]
            if not all(col in run.requests.columns for col in required_cols):
                return False

        # Special validation for GPU utilization with throughput
        if spec.name == "gpu_utilization_and_throughput_over_time":
            if run.requests is None or run.requests.empty:
                return False
            required_cols = [
                "request_start_ns",
                "request_end_ns",
                "output_sequence_length",
            ]
            if not all(col in run.requests.columns for col in required_cols):
                return False

        return True

    def _create_plot_from_spec(
        self, spec: PlotSpec, run: RunData, available_metrics: dict
    ) -> go.Figure:
        """
        Create a plot figure from a plot specification.

        Args:
            spec: Plot specification
            run: RunData object
            available_metrics: Dictionary with display_names and units for metrics

        Returns:
            Plotly figure object
        """
        if spec.plot_type == PlotType.SCATTER:
            return self._create_scatter_plot(spec, run, available_metrics)
        elif spec.plot_type == PlotType.AREA:
            return self._create_area_plot(spec, run, available_metrics)
        elif spec.plot_type == PlotType.HISTOGRAM:
            return self._create_histogram_plot(spec, run, available_metrics)
        elif spec.plot_type == PlotType.DUAL_AXIS:
            return self._create_dual_axis_plot(spec, run, available_metrics)
        elif spec.plot_type == PlotType.SCATTER_WITH_PERCENTILES:
            return self._create_scatter_with_percentiles_plot(
                spec, run, available_metrics
            )
        else:
            raise ValueError(f"Unsupported plot type: {spec.plot_type}")

    def _create_scatter_plot(
        self, spec: PlotSpec, run: RunData, available_metrics: dict
    ) -> go.Figure:
        """Create a scatter plot from specification."""
        x_metric = next(m for m in spec.metrics if m.axis == "x")
        y_metric = next(m for m in spec.metrics if m.axis == "y")

        df = self._prepare_data_for_source(x_metric.source, run)

        return self.plot_generator.create_time_series_scatter(
            df=df,
            x_col=x_metric.name,
            y_metric=y_metric.name,
            title=spec.title,
            x_label=self._get_axis_label(x_metric, available_metrics),
            y_label=self._get_axis_label(y_metric, available_metrics),
        )

    def _create_area_plot(
        self, spec: PlotSpec, run: RunData, available_metrics: dict
    ) -> go.Figure:
        """Create an area plot from specification."""
        x_metric = next(m for m in spec.metrics if m.axis == "x")
        y_metric = next(m for m in spec.metrics if m.axis == "y")

        # Special handling for dispersed throughput
        if y_metric.name == "throughput_tokens_per_sec":
            df = self._per_request_to_dataframe(run)
            throughput_df = self._calculate_dispersed_throughput_events(df)
        else:
            throughput_df = self._prepare_data_for_source(x_metric.source, run)

        return self.plot_generator.create_time_series_area(
            df=throughput_df,
            x_col=x_metric.name,
            y_metric=y_metric.name,
            title=spec.title,
            x_label=self._get_axis_label(x_metric, available_metrics),
            y_label=self._get_axis_label(y_metric, available_metrics),
        )

    def _create_histogram_plot(
        self, spec: PlotSpec, run: RunData, available_metrics: dict
    ) -> go.Figure:
        """Create a histogram plot from specification (for timeslices)."""
        x_metric = next(m for m in spec.metrics if m.axis == "x")
        y_metric = next(m for m in spec.metrics if m.axis == "y")

        # Filter timeslice data for specific metric and stat
        metric_data = run.timeslices[
            (run.timeslices["Metric"] == y_metric.name)
            & (run.timeslices["Stat"] == y_metric.stat)
        ].copy()

        if metric_data.empty:
            raise ValueError(f"No timeslice data for {y_metric.name} ({y_metric.stat})")

        # Prepare data for histogram
        plot_df = metric_data[["Timeslice", "Value"]].copy()
        plot_df = plot_df.rename(columns={"Value": y_metric.stat})

        # Get unit for y-axis label
        unit = metric_data["Unit"].iloc[0] if not metric_data.empty else ""
        y_label = f"{y_metric.name} ({unit})" if unit else y_metric.name

        # Check if we need warning for throughput plot
        warning_message = None
        if "throughput" in spec.name.lower():
            _, warning_message = self._check_request_uniformity(run)

        # Determine if we should use slice_duration
        use_slice_duration = (
            isinstance(spec, TimeSlicePlotSpec) and spec.use_slice_duration
        )

        return self.plot_generator.create_time_series_histogram(
            df=plot_df,
            x_col=x_metric.name,
            y_col=y_metric.stat,
            title=spec.title,
            x_label=self._get_axis_label(x_metric, available_metrics),
            y_label=y_label,
            slice_duration=run.slice_duration if use_slice_duration else None,
            warning_text=warning_message,
        )

    def _create_dual_axis_plot(
        self, spec: PlotSpec, run: RunData, available_metrics: dict
    ) -> go.Figure:
        """Create a dual-axis plot from specification."""
        y1_metric = next(m for m in spec.metrics if m.axis == "y")
        y2_metric = next(m for m in spec.metrics if m.axis == "y2")

        # Handle GPU utilization with throughput overlay
        if spec.name == "gpu_utilization_and_throughput_over_time":
            # Prepare GPU data
            gpu_df = run.gpu_telemetry.copy()
            if "gpu_index" in gpu_df.columns:
                gpu_df = (
                    gpu_df.groupby("timestamp_s")
                    .agg({"gpu_utilization": "mean"})
                    .reset_index()
                )

            # Prepare throughput data
            requests_df = self._per_request_to_dataframe(run)
            throughput_df = self._calculate_dispersed_throughput_events(requests_df)

            if throughput_df.empty:
                raise ValueError("No throughput data available")

            return self.plot_generator.create_gpu_dual_axis_plot(
                df_primary=throughput_df,
                df_secondary=gpu_df,
                x_col_primary="timestamp_s",
                x_col_secondary="timestamp_s",
                y1_metric=y1_metric.name,
                y2_metric=y2_metric.name,
                active_count_col="active_requests",
                title=spec.title,
                x_label="Time (s)",
                y1_label="Output Tokens/sec",
                y2_label="GPU Utilization (%)",
            )
        else:
            raise ValueError(f"Unsupported dual-axis plot: {spec.name}")

    def _create_scatter_with_percentiles_plot(
        self, spec: PlotSpec, run: RunData, available_metrics: dict
    ) -> go.Figure:
        """Create a scatter plot with percentile overlays."""
        x_metric = next(m for m in spec.metrics if m.axis == "x")
        y_metric = next(m for m in spec.metrics if m.axis == "y")

        df = self._prepare_data_for_source(x_metric.source, run)
        df_sorted = df.sort_values(x_metric.name).copy()

        # Calculate rolling percentiles
        window_size = min(10, len(df_sorted))
        df_sorted["p50"] = (
            df_sorted[y_metric.name]
            .rolling(window=window_size, min_periods=1)
            .quantile(0.50)
        )
        df_sorted["p95"] = (
            df_sorted[y_metric.name]
            .rolling(window=window_size, min_periods=1)
            .quantile(0.95)
        )
        df_sorted["p99"] = (
            df_sorted[y_metric.name]
            .rolling(window=window_size, min_periods=1)
            .quantile(0.99)
        )

        return self.plot_generator.create_latency_scatter_with_percentiles(
            df=df_sorted,
            x_col=x_metric.name,
            y_metric=y_metric.name,
            percentile_cols=["p50", "p95", "p99"],
            title=spec.title,
            x_label=self._get_axis_label(x_metric, available_metrics),
            y_label=self._get_axis_label(y_metric, available_metrics),
        )

    def _prepare_data_for_source(
        self, source: DataSource, run: RunData
    ) -> pd.DataFrame:
        """
        Prepare data from a specific source.

        Args:
            source: Data source to prepare
            run: RunData object

        Returns:
            Prepared DataFrame
        """
        if source == DataSource.REQUESTS:
            return self._per_request_to_dataframe(run)
        elif source == DataSource.TIMESLICES:
            return run.timeslices
        elif source == DataSource.GPU_TELEMETRY:
            return run.gpu_telemetry
        else:
            raise ValueError(f"Unsupported data source: {source}")

    def _get_axis_label(self, metric_spec, available_metrics: dict) -> str:
        """
        Get axis label for a metric.

        Args:
            metric_spec: MetricSpec object
            available_metrics: Dictionary with display_names and units

        Returns:
            Formatted axis label
        """
        if metric_spec.name == "request_number":
            return "Request Number"
        elif metric_spec.name == "timestamp":
            return "Time (seconds)"
        elif metric_spec.name == "timestamp_s" or metric_spec.name == "Timeslice":
            return "Time (s)"
        else:
            return self._get_metric_label(
                metric_spec.name, metric_spec.stat, available_metrics
            )

    def _check_request_uniformity(self, run: RunData) -> tuple[bool, str | None]:
        """
        Check if requests have uniform ISL/OSL or if they vary.

        If per-request data is not loaded in run.requests, attempts to load
        ISL/OSL values from profile_export.jsonl on-demand.

        Args:
            run: RunData object with requests DataFrame (may be None)

        Returns:
            Tuple of (is_uniform, warning_message) where:
            - is_uniform: True if all ISL and OSL are identical, False otherwise
            - warning_message: Warning text if non-uniform, None if uniform
        """
        if run.requests is not None and not run.requests.empty:
            df = run.requests
            has_isl = "input_sequence_length" in df.columns
            has_osl = "output_sequence_length" in df.columns

            if not has_isl and not has_osl:
                return True, None

            isl_values = (
                df["input_sequence_length"].dropna() if has_isl else pd.Series()
            )
            osl_values = (
                df["output_sequence_length"].dropna() if has_osl else pd.Series()
            )
        else:
            profile_path = run.metadata.run_path / "profile_export.jsonl"
            if not profile_path.exists():
                return True, None

            try:
                isl_values = []
                osl_values = []

                with open(profile_path) as f:
                    for line in f:
                        try:
                            record = json.loads(line)
                            metrics = record.get("metrics", {})

                            isl = metrics.get("input_sequence_length", {})
                            if isinstance(isl, dict) and "value" in isl:
                                isl_values.append(isl["value"])

                            osl = metrics.get("output_sequence_length", {})
                            if isinstance(osl, dict) and "value" in osl:
                                osl_values.append(osl["value"])
                        except json.JSONDecodeError:
                            continue

                if not isl_values and not osl_values:
                    return True, None

                isl_values = pd.Series(isl_values) if isl_values else pd.Series()
                osl_values = pd.Series(osl_values) if osl_values else pd.Series()

            except Exception as e:
                self.warning(f"Could not load ISL/OSL data for uniformity check: {e}")
                return True, None

        is_uniform = True

        if len(isl_values) > 1 and isl_values.nunique() > 1:
            is_uniform = False

        if len(osl_values) > 1 and osl_values.nunique() > 1:
            is_uniform = False

        if not is_uniform:
            warning_message = (
                "⚠ Requests have varying ISL/OSL. "
                "Req/sec throughput may not accurately represent workload capacity."
            )
            return False, warning_message

        return True, None

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
            # Handle both datetime and numeric timestamp formats
            if pd.api.types.is_datetime64_any_dtype(df["request_end_ns"]):
                # Already datetime, convert to seconds
                df["timestamp"] = df["request_end_ns"].astype("int64") / 1_000_000_000
            else:
                # Numeric nanoseconds, divide directly
                df["timestamp"] = df["request_end_ns"] / 1_000_000_000

            # Normalize to start from 0
            if len(df) > 0:
                df["timestamp"] = df["timestamp"] - df["timestamp"].min()

        return df

    def _calculate_dispersed_throughput_events(
        self, requests_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate throughput using event-based approach with evenly dispersed tokens.

        Tokens are evenly distributed across the generation phase (from TTFT to request_end)
        rather than being counted at a single event. This creates smooth throughput curves
        that accurately represent the token generation rate over time.

        Args:
            requests_df: DataFrame with request data including timestamps and metrics

        Returns:
            DataFrame with columns: timestamp_s, throughput_tokens_per_sec, active_requests
        """
        events = []

        for _, row in requests_df.iterrows():
            request_start_ns = row.get("request_start_ns")
            request_end_ns = row.get("request_end_ns")

            if pd.isna(request_start_ns) or pd.isna(request_end_ns):
                continue

            if isinstance(request_start_ns, pd.Timestamp):
                request_start_ns = request_start_ns.value
            if isinstance(request_end_ns, pd.Timestamp):
                request_end_ns = request_end_ns.value

            request_start_ns = int(request_start_ns)
            request_end_ns = int(request_end_ns)

            ttft_ms = row.get("time_to_first_token", 0)
            if pd.isna(ttft_ms):
                ttft_ms = 0

            generation_start_ns = (
                request_start_ns + int(ttft_ms * 1e6)
                if ttft_ms > 0
                else request_start_ns
            )

            output_tokens = row.get("output_sequence_length", 0)
            if pd.isna(output_tokens):
                output_tokens = 0

            generation_duration_ns = request_end_ns - generation_start_ns

            if generation_duration_ns > 0 and output_tokens > 0:
                token_rate = output_tokens / (generation_duration_ns / 1e9)

                events.append(
                    {
                        "timestamp_ns": generation_start_ns,
                        "delta_rate": token_rate,
                        "active_delta": 1,
                    }
                )
                events.append(
                    {
                        "timestamp_ns": request_end_ns,
                        "delta_rate": -token_rate,
                        "active_delta": -1,
                    }
                )

        if not events:
            return pd.DataFrame(
                columns=["timestamp_s", "throughput_tokens_per_sec", "active_requests"]
            )

        events_df = pd.DataFrame(events).sort_values("timestamp_ns")

        events_df["throughput_tokens_per_sec"] = events_df["delta_rate"].cumsum()
        events_df["active_requests"] = events_df["active_delta"].cumsum()

        start_ns = events_df["timestamp_ns"].min()
        events_df["timestamp_s"] = (events_df["timestamp_ns"] - start_ns) / 1e9

        return events_df[
            ["timestamp_s", "throughput_tokens_per_sec", "active_requests"]
        ].reset_index(drop=True)
