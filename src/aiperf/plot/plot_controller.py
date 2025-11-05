# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plot controller for generating visualizations from profiling data."""

from pathlib import Path

from aiperf.plot.constants import PlotMode
from aiperf.plot.core.data_loader import DataLoader
from aiperf.plot.core.mode_detector import ModeDetector, VisualizationMode
from aiperf.plot.exporters import MultiRunPNGExporter, SingleRunPNGExporter


class PlotController:
    """Controller for generating plots from AIPerf profiling data.

    Orchestrates the plot generation pipeline: mode detection, data loading,
    and export. Designed to support multiple output modes (PNG, HTML, server)
    in the future.

    Args:
        paths: List of paths to profiling run directories
        output_dir: Directory to save generated plots
        mode: Output mode (PNG, HTML, or SERVER - currently only PNG supported)
    """

    def __init__(
        self,
        paths: list[Path],
        output_dir: Path,
        mode: PlotMode = PlotMode.PNG,
    ):
        self.paths = paths
        self.output_dir = output_dir
        self.mode = mode
        self.loader = DataLoader()
        self.mode_detector = ModeDetector()

    def run(self) -> list[Path]:
        """Execute plot generation pipeline.

        Returns:
            List of paths to generated plot files
        """
        if self.mode == PlotMode.PNG:
            return self._generate_png_plots()
        else:
            raise ValueError(
                f"Unsupported mode: {self.mode.value}. Currently only '{PlotMode.PNG.value}' is supported."
            )

    def _validate_paths(self) -> None:
        """Validate that all input paths exist."""
        for path in self.paths:
            if not path.exists():
                raise FileNotFoundError(
                    f"Path does not exist: {path}. Please check the path and try again."
                )

    def _detect_visualization_mode(self) -> tuple[VisualizationMode, list[Path]]:
        """Detect whether to generate single-run or multi-run plots.

        Returns:
            Tuple of (visualization mode, list of run directories)
        """
        mode, run_dirs = self.mode_detector.detect_mode(self.paths)

        if not run_dirs:
            raise ValueError(
                f"No valid profiling runs found in: {self.paths}. "
                "Please ensure the directory contains AIPerf profiling output."
            )

        return mode, run_dirs

    def _generate_png_plots(self) -> list[Path]:
        """Generate static PNG plot images.

        Returns:
            List of paths to generated PNG files
        """
        self._validate_paths()
        viz_mode, run_dirs = self._detect_visualization_mode()

        print(
            f"Detecting mode: {viz_mode.value.replace('_', '-')} "
            f"({len(run_dirs)} run{'s' if len(run_dirs) > 1 else ''} found)"
        )

        if viz_mode == VisualizationMode.MULTI_RUN:
            return self._export_multi_run_plots(run_dirs)
        else:
            return self._export_single_run_plots(run_dirs[0])

    def _export_multi_run_plots(self, run_dirs: list[Path]) -> list[Path]:
        """Export multi-run comparison plots.

        Args:
            run_dirs: List of run directories to compare

        Returns:
            List of paths to generated plot files
        """
        runs = []
        for run_dir in run_dirs:
            try:
                run_data = self.loader.load_run(run_dir, load_per_request_data=False)
                self.loader.add_derived_gpu_metrics(run_data.aggregated)
                runs.append(run_data)
            except Exception as e:
                print(f"Warning: Failed to load run from {run_dir}: {e}")

        if not runs:
            raise ValueError("Failed to load any valid profiling runs")

        available = self.loader.get_available_metrics(runs[0])
        exporter = MultiRunPNGExporter(self.output_dir)
        return exporter.export(runs, available)

    def _export_single_run_plots(self, run_dir: Path) -> list[Path]:
        """Export single-run time series plots.

        Args:
            run_dir: Run directory to generate plots from

        Returns:
            List of paths to generated plot files
        """
        run_data = self.loader.load_run(run_dir, load_per_request_data=True)
        available = self.loader.get_available_metrics(run_data)
        exporter = SingleRunPNGExporter(self.output_dir)
        return exporter.export(run_data, available)
