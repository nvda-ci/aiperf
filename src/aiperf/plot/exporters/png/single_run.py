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

        # Plot 2: GPU Memory Breakdown
        generated_files.extend(
            self._generate_gpu_memory_breakdown(run, available_metrics)
        )

        return generated_files

    def _generate_gpu_utilization_with_throughput(
        self, run: RunData, available_metrics: dict
    ) -> list[Path]:
        """
        Generate GPU utilization plot with output token throughput overlay.

        Args:
            run: RunData object with gpu_telemetry and requests DataFrames
            available_metrics: Dictionary with display_names and units for metrics

        Returns:
            List of Path objects for generated PNG files
        """
        try:
            gpu_df = run.gpu_telemetry.copy()

            # Aggregate by GPU if multiple GPUs present
            if "gpu_index" in gpu_df.columns:
                # Group by timestamp_s and calculate mean across GPUs
                gpu_df = (
                    gpu_df.groupby("timestamp_s")
                    .agg({"gpu_utilization": "mean"})
                    .reset_index()
                )

            # Calculate throughput from requests data
            if run.requests is not None and not run.requests.empty:
                requests_df = run.requests.copy()

                # Convert request_end_ns to timestamp_s relative to run start
                if "request_end_ns" in requests_df.columns:
                    if pd.api.types.is_datetime64_any_dtype(
                        requests_df["request_end_ns"]
                    ):
                        end_times_ns = requests_df["request_end_ns"].astype("int64")
                    else:
                        end_times_ns = requests_df["request_end_ns"]

                    start_time_ns = end_times_ns.min()
                    requests_df["timestamp_s"] = (end_times_ns - start_time_ns) / 1e9

                    # Calculate throughput: tokens per second in 1-second windows
                    if "output_sequence_length" in requests_df.columns:
                        # Round timestamps to nearest second for binning
                        requests_df["time_bin"] = requests_df["timestamp_s"].round(0)

                        throughput_df = (
                            requests_df.groupby("time_bin")
                            .agg({"output_sequence_length": "sum"})
                            .reset_index()
                        )
                        throughput_df = throughput_df.rename(
                            columns={
                                "time_bin": "timestamp_s",
                                "output_sequence_length": "output_token_throughput",
                            }
                        )

                        # Merge with GPU data
                        # Round GPU timestamps for alignment
                        gpu_df["time_bin"] = gpu_df["timestamp_s"].round(0)
                        merged_df = pd.merge(
                            gpu_df,
                            throughput_df,
                            left_on="time_bin",
                            right_on="timestamp_s",
                            how="left",
                        )
                        merged_df = merged_df.drop(columns=["time_bin"])
                        # Use the original GPU timestamp
                        merged_df["timestamp_s"] = merged_df["timestamp_s_x"]
                        merged_df = merged_df.drop(
                            columns=["timestamp_s_x", "timestamp_s_y"]
                        )
                        # Fill NaN throughput with 0
                        merged_df["output_token_throughput"] = merged_df[
                            "output_token_throughput"
                        ].fillna(0)

                        # Create the plot
                        fig = self.plot_generator.create_gpu_dual_axis_plot(
                            df=merged_df,
                            x_col="timestamp_s",
                            y1_metric="gpu_utilization",
                            y2_metric="output_token_throughput",
                            title="GPU Utilization with Output Token Throughput",
                            x_label="Time (s)",
                            y1_label="GPU Utilization (%)",
                            y2_label="Output Tokens/sec",
                        )

                        path = self.output_dir / "gpu_utilization_throughput.png"
                        self._export_figure(fig, path)
                        self.info("✓ Generated gpu_utilization_throughput.png")
                        return [path]

            # Fallback: plot GPU utilization only if throughput can't be calculated
            self.warning(
                "Could not calculate throughput overlay - plotting GPU utilization only"
            )
            return []

        except Exception as e:
            self.error(f"Failed to generate GPU utilization with throughput: {e}")
            return []

    def _generate_gpu_memory_breakdown(
        self, run: RunData, available_metrics: dict
    ) -> list[Path]:
        """
        Generate GPU memory breakdown plot.

        Args:
            run: RunData object with gpu_telemetry DataFrame
            available_metrics: Dictionary with display_names and units for metrics

        Returns:
            List of Path objects for generated PNG files
        """
        try:
            gpu_df = run.gpu_telemetry.copy()

            # Check if required columns exist
            if (
                "gpu_memory_used" not in gpu_df.columns
                or "gpu_memory_free" not in gpu_df.columns
            ):
                self.warning("GPU memory columns not found in telemetry data")
                return []

            # Aggregate by GPU if multiple GPUs present
            if "gpu_index" in gpu_df.columns:
                # Group by timestamp_s and calculate mean across GPUs
                gpu_df = (
                    gpu_df.groupby("timestamp_s")
                    .agg({"gpu_memory_used": "mean", "gpu_memory_free": "mean"})
                    .reset_index()
                )

            # Create the plot
            fig = self.plot_generator.create_gpu_memory_stacked_area(
                df=gpu_df,
                x_col="timestamp_s",
                used_col="gpu_memory_used",
                free_col="gpu_memory_free",
                title="GPU Memory Usage Over Time",
                x_label="Time (s)",
                y_label="Memory (GB)",
            )

            path = self.output_dir / "gpu_memory_breakdown.png"
            self._export_figure(fig, path)
            self.info("✓ Generated gpu_memory_breakdown.png")
            return [path]

        except Exception as e:
            self.error(f"Failed to generate GPU memory breakdown: {e}")
            return []
