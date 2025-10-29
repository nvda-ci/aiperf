# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Logging configuration for the plot command.

This module provides logging setup specific to the plot functionality,
separate from the main AIPerf benchmark logging. Logs are written to
the output directory alongside generated visualizations.
"""

import logging
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.plot.constants import PLOT_LOG_FILE

_logger = AIPerfLogger(__name__)


def setup_plot_logging(output_dir: Path, log_level: str = "INFO") -> None:
    """
    Set up logging for the plot command.

    Configures logging to output to both console (via RichHandler) and a log
    file in the output directory. This should be called at the start of the
    plot command before any plot operations.

    Args:
        output_dir: Directory where plot outputs (and logs) will be saved.
        log_level: Logging level (e.g., "DEBUG", "INFO", "WARNING"). Defaults to "INFO".

    Examples:
        >>> from pathlib import Path
        >>> setup_plot_logging(Path("./artifacts/plot_export"), log_level="INFO")
    """
    root_logger = logging.getLogger()

    level = log_level.upper()
    root_logger.setLevel(level)

    for existing_handler in root_logger.handlers[:]:
        root_logger.removeHandler(existing_handler)

    rich_handler = RichHandler(
        rich_tracebacks=True,
        show_path=True,
        console=Console(),
        show_time=True,
        show_level=True,
        tracebacks_show_locals=False,
        log_time_format="%H:%M:%S.%f",
        omit_repeated_times=False,
    )
    rich_handler.setLevel(level)
    root_logger.addHandler(rich_handler)

    output_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = output_dir / PLOT_LOG_FILE

    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root_logger.addHandler(file_handler)

    _logger.info(f"Plot logging initialized with level: {level}")
    _logger.info(f"Log file: {log_file_path}")
