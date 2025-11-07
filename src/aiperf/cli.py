# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Main CLI entry point for the AIPerf system."""

################################################################################
# NOTE: Keep the imports here to a minimum. This file is read every time
# the CLI is run, including to generate the help text. Any imports here
# will cause a performance penalty during this process.
################################################################################

from cyclopts import App

from aiperf.cli_utils import exit_on_error
from aiperf.common.config import ServiceConfig, UserConfig

app = App(name="aiperf", help="NVIDIA AIPerf")


@app.command(name="profile")
def profile(
    user_config: UserConfig,
    service_config: ServiceConfig | None = None,
) -> None:
    """Run the Profile subcommand.

    Args:
        user_config: User configuration for the benchmark
        service_config: Service configuration options
    """
    with exit_on_error(title="Error Running AIPerf System"):
        from aiperf.cli_runner import run_system_controller
        from aiperf.common.config import load_service_config

        service_config = service_config or load_service_config()
        run_system_controller(user_config, service_config)


@app.command(name="plot")
def plot(
    paths: list[str] | None = None,
    output: str | None = None,
    theme: str = "light",
) -> None:
    """Generate PNG visualizations from AIPerf profiling data.

    TODO [AIP-546 and AIP-549]: Update for HTML and hosted options.
    Currently creates static PNG plot images from profiling results. Automatically detects
    whether to generate multi-run comparison plots or single-run time series plots
    based on the directory structure.

    Args:
        paths: Paths to profiling run directories. Defaults to ./artifacts if not specified.
        output: Directory to save generated plots. Defaults to <first_path>/plot_export if not specified.
        theme: Plot theme to use: 'light' (white background) or 'dark' (dark background). Defaults to 'light'.
    """
    with exit_on_error(title="Error Running Plot Command"):
        from aiperf.plot.cli_runner import run_plot_controller

        run_plot_controller(paths, output, theme=theme)
