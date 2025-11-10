# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Single-run PNG exporter for time series plots.

Generates static PNG images for analyzing a single profiling run.
"""

import json
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

        # Plot 4-7: Timeslice plots (if data available)
        generated_files.extend(self._generate_timeslices_plots(run, available_metrics))

        # Plot 8-9: GPU telemetry plots (if data available)
        generated_files.extend(self._generate_gpu_plots(run, available_metrics))

        # Plot 10: Dispersed throughput over time (if request data available)
        generated_files.extend(
            self._generate_dispersed_throughput_over_time(run, available_metrics)
        )

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
                title="TTFT Per Request Over Time",
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
                title="Inter-Token Latency Per Request Over Time",
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
        """Generate latency over time plot with percentile overlays."""
        try:
            df_sorted = df.sort_values("timestamp").copy()

            # Calculate rolling percentiles (window of 10 requests, minimum 1)
            window_size = min(10, len(df_sorted))
            df_sorted["p50"] = (
                df_sorted["request_latency"]
                .rolling(window=window_size, min_periods=1)
                .quantile(0.50)
            )
            df_sorted["p95"] = (
                df_sorted["request_latency"]
                .rolling(window=window_size, min_periods=1)
                .quantile(0.95)
            )
            df_sorted["p99"] = (
                df_sorted["request_latency"]
                .rolling(window=window_size, min_periods=1)
                .quantile(0.99)
            )

            fig = self.plot_generator.create_latency_scatter_with_percentiles(
                df=df_sorted,
                x_col="timestamp",
                y_metric="request_latency",
                percentile_cols=["p50", "p95", "p99"],
                title="Request Latency Over Time with Percentiles",
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

    def _generate_timeslices_plots(
        self, run: RunData, available_metrics: dict
    ) -> list[Path]:
        """Generate timeslice histogram plots for all available metrics.

        Creates separate PNG files for each metric showing bar charts across time slices.

        Args:
            run: RunData object with timeslices DataFrame
            available_metrics: Dictionary with display_names and units for metrics

        Returns:
            List of Path objects for generated PNG files
        """
        if run.timeslices is None or run.timeslices.empty:
            self.debug("No timeslice data available for plotting")
            return []

        self.info("Generating timeslice histogram plots")
        generated_files = []

        # Statistic to use for histogram (configurable for future)
        # Options: "avg", "p50", "p90", "p95", "p99", "min", "max"
        stat_to_plot = "avg"

        _, warning_message = self._check_request_uniformity(run)

        # Define metrics to plot with their display names
        metrics_to_plot = [
            ("Time to First Token", "timeslices_ttft.png", "time_to_first_token"),
            ("Inter Token Latency", "timeslices_itl.png", "inter_token_latency"),
            ("Request Throughput", "timeslices_throughput.png", "request_throughput"),
            ("Request Latency", "timeslices_latency.png", "request_latency"),
        ]

        for metric_name, filename, metric_tag in metrics_to_plot:
            try:
                # Filter for this specific metric and statistic
                metric_data = run.timeslices[
                    (run.timeslices["Metric"] == metric_name)
                    & (run.timeslices["Stat"] == stat_to_plot)
                ].copy()

                if metric_data.empty:
                    self.debug(
                        f"No timeslice data found for {metric_name} ({stat_to_plot})"
                    )
                    continue

                # Prepare data for histogram (no pivoting needed)
                plot_df = metric_data[["Timeslice", "Value"]].copy()
                plot_df = plot_df.rename(columns={"Value": stat_to_plot})

                # Get unit for y-axis label
                unit = metric_data["Unit"].iloc[0] if not metric_data.empty else ""
                y_label = f"{metric_name} ({unit})" if unit else metric_name

                plot_warning = (
                    warning_message if metric_tag == "request_throughput" else None
                )

                # Create the histogram
                fig = self.plot_generator.create_time_series_histogram(
                    df=plot_df,
                    x_col="Timeslice",
                    y_col=stat_to_plot,
                    title=f"{metric_name} Across Time Slices",
                    x_label="Time (s)",
                    y_label=y_label,
                    slice_duration=run.slice_duration,
                    warning_text=plot_warning,
                )

                path = self.output_dir / filename
                self._export_figure(fig, path)
                self.info(f"✓ Generated {filename}")
                generated_files.append(path)

            except Exception as e:
                self.error(f"Failed to generate timeslice plot for {metric_name}: {e}")

        return generated_files

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

    def _generate_gpu_plots(self, run: RunData, available_metrics: dict) -> list[Path]:
        """
        Generate GPU telemetry plots.

        Args:
            run: RunData object with gpu_telemetry DataFrame
            available_metrics: Dictionary with display_names and units for metrics

        Returns:
            List of Path objects for generated PNG files
        """
        if run.gpu_telemetry is None or run.gpu_telemetry.empty:
            self.debug("No GPU telemetry data available for plotting")
            return []

        self.info("Generating GPU telemetry plots")
        generated_files = []

        # Plot 1: GPU Utilization with Output Token Throughput overlay
        generated_files.extend(
            self._generate_gpu_utilization_with_throughput(run, available_metrics)
        )

        # Plot 2: GPU Metrics Overlay
        generated_files.extend(
            self._generate_gpu_metrics_overlay(run, available_metrics)
        )

        return generated_files

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

    def _generate_gpu_utilization_with_throughput(
        self, run: RunData, available_metrics: dict
    ) -> list[Path]:
        """
        Generate throughput plot with GPU utilization overlay.

        Uses event-based dispersed throughput calculation where tokens are evenly
        distributed across the generation phase for accurate representation.

        Args:
            run: RunData object with gpu_telemetry and requests DataFrames
            available_metrics: Dictionary with display_names and units for metrics

        Returns:
            List of Path objects for generated PNG files
        """
        try:
            gpu_df = run.gpu_telemetry.copy()

            if "gpu_index" in gpu_df.columns:
                gpu_df = (
                    gpu_df.groupby("timestamp_s")
                    .agg({"gpu_utilization": "mean"})
                    .reset_index()
                )

            if run.requests is not None and not run.requests.empty:
                requests_df = run.requests.copy()

                required_cols = [
                    "request_start_ns",
                    "request_end_ns",
                    "output_sequence_length",
                ]
                if all(col in requests_df.columns for col in required_cols):
                    throughput_df = self._calculate_dispersed_throughput_events(
                        requests_df
                    )

                    if not throughput_df.empty:
                        fig = self.plot_generator.create_gpu_dual_axis_plot(
                            df_primary=throughput_df,
                            df_secondary=gpu_df,
                            x_col_primary="timestamp_s",
                            x_col_secondary="timestamp_s",
                            y1_metric="throughput_tokens_per_sec",
                            y2_metric="gpu_utilization",
                            active_count_col="active_requests",
                            title="Output Token Throughput with GPU Utilization",
                            x_label="Time (s)",
                            y1_label="Output Tokens/sec",
                            y2_label="GPU Utilization (%)",
                        )

                        path = self.output_dir / "gpu_utilization_throughput.png"
                        self._export_figure(fig, path)
                        self.info("✓ Generated gpu_utilization_throughput.png")
                        return [path]

            self.warning(
                "Could not calculate throughput overlay - plotting GPU utilization only"
            )
            return []

        except Exception as e:
            self.error(f"Failed to generate GPU utilization with throughput: {e}")
            return []

    def _generate_gpu_metrics_overlay(
        self, run: RunData, available_metrics: dict
    ) -> list[Path]:
        """
        Generate GPU metrics overlay plot with power, temperature, and clock speeds.

        Args:
            run: RunData object with gpu_telemetry DataFrame
            available_metrics: Dictionary with display_names and units for metrics

        Returns:
            List of Path objects for generated PNG files
        """
        try:
            gpu_df = run.gpu_telemetry.copy()

            required_cols = [
                "gpu_power_usage",
                "gpu_temperature",
                "sm_clock_frequency",
                "memory_clock_frequency",
            ]
            missing_cols = [col for col in required_cols if col not in gpu_df.columns]

            if missing_cols:
                self.warning(
                    f"GPU metrics columns not found in telemetry data: {missing_cols}"
                )
                return []

            fig = self.plot_generator.create_gpu_metrics_overlay(
                df=gpu_df,
                x_col="timestamp_s",
                power_col="gpu_power_usage",
                temp_col="gpu_temperature",
                sm_clock_col="sm_clock_frequency",
                mem_clock_col="memory_clock_frequency",
                gpu_id_col="gpu_uuid",
                title="GPU Metrics Over Time",
                x_label="Time (s)",
            )

            path = self.output_dir / "gpu_metrics_overlay.png"
            self._export_figure(fig, path)
            self.info("✓ Generated gpu_metrics_overlay.png")
            return [path]

        except Exception as e:
            self.error(f"Failed to generate GPU metrics overlay: {e}")
            return []

    def _generate_dispersed_throughput_over_time(
        self, run: RunData, available_metrics: dict
    ) -> list[Path]:
        """
        Generate standalone dispersed throughput over time plot.

        Shows output token throughput with tokens evenly distributed across
        the generation phase (from TTFT to request_end).

        Args:
            run: RunData object with requests DataFrame
            available_metrics: Dictionary with display_names and units for metrics

        Returns:
            List of Path objects for generated PNG files
        """
        try:
            if run.requests is None or run.requests.empty:
                self.debug("No requests data available for dispersed throughput plot")
                return []

            requests_df = run.requests.copy()

            required_cols = [
                "request_start_ns",
                "request_end_ns",
                "output_sequence_length",
            ]
            if not all(col in requests_df.columns for col in required_cols):
                self.warning(
                    f"Missing required columns for dispersed throughput plot. "
                    f"Required: {required_cols}"
                )
                return []

            throughput_df = self._calculate_dispersed_throughput_events(requests_df)

            if throughput_df.empty:
                self.warning("No throughput data to plot")
                return []

            fig = self.plot_generator.create_time_series_area(
                df=throughput_df,
                x_col="timestamp_s",
                y_metric="throughput_tokens_per_sec",
                title="Dispersed Output Token Throughput Over Time",
                x_label="Time (s)",
                y_label="Output Tokens/sec",
            )

            path = self.output_dir / "dispersed_throughput_over_time.png"
            self._export_figure(fig, path)
            self.info("✓ Generated dispersed_throughput_over_time.png")
            return [path]

        except Exception as e:
            self.error(f"Failed to generate dispersed throughput plot: {e}")
            return []
