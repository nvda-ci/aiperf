# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI runner for plot command."""

from pathlib import Path

from aiperf.plot.constants import PlotMode
from aiperf.plot.plot_controller import PlotController


def run_plot_controller(
    paths: list[str] | None = None,
    output: str | None = None,
    mode: PlotMode | str = PlotMode.PNG,
) -> None:
    """Generate plots from AIPerf profiling data.

    Args:
        paths: Paths to profiling run directories. Defaults to ./artifacts if not specified.
        output: Directory to save generated plots. Defaults to <first_path>/plot_export if not specified.
        mode: Output mode for plots (PNG, HTML, or SERVER). Defaults to PNG.
    """
    # Determine input paths
    input_paths = paths or ["./artifacts"]
    input_paths = [Path(p) for p in input_paths]

    # Determine output directory
    output_dir = Path(output) if output else input_paths[0] / "plot_export"

    # Create and run controller
    controller = PlotController(
        paths=input_paths,
        output_dir=output_dir,
        mode=mode,
    )

    generated_files = controller.run()

    # Print success message
    print(f"\n✓ Generated {len(generated_files)} plots")
    print(f"Saved to: {output_dir}")
