# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Multi-run plot type handlers.

Handlers for creating comparison plots from multiple profiling runs.
"""

import pandas as pd
import plotly.graph_objects as go

from aiperf.plot.constants import DEFAULT_PERCENTILE
from aiperf.plot.core.plot_generator import PlotGenerator
from aiperf.plot.core.plot_specs import PlotSpec, PlotType
from aiperf.plot.core.plot_type_handlers import PlotTypeHandlerFactory


class BaseMultiRunHandler:
    """
    Base class for multi-run plot handlers.

    Provides common functionality for working with multi-run DataFrames.
    """

    def __init__(self, plot_generator: PlotGenerator) -> None:
        """
        Initialize the handler.

        Args:
            plot_generator: PlotGenerator instance for rendering plots
        """
        self.plot_generator = plot_generator

    def _get_metric_label(
        self, metric_name: str, stat: str | None, available_metrics: dict
    ) -> str:
        """
        Get formatted metric label.

        Args:
            metric_name: Name of the metric
            stat: Statistic (e.g., "avg", "p50")
            available_metrics: Dictionary with display_names and units

        Returns:
            Formatted metric label
        """
        if metric_name in available_metrics:
            display_name = available_metrics[metric_name].get(
                "display_name", metric_name
            )
            unit = available_metrics[metric_name].get("unit", "")
            if stat and stat not in ["avg", "value"]:
                display_name = f"{display_name} ({stat})"
            if unit:
                return f"{display_name} ({unit})"
            return display_name
        return metric_name


@PlotTypeHandlerFactory.register(PlotType.PARETO)
class ParetoHandler(BaseMultiRunHandler):
    """Handler for Pareto curve plots."""

    def can_handle(self, spec: PlotSpec, data: pd.DataFrame) -> bool:
        """Check if Pareto plot can be generated."""
        for metric in spec.metrics:
            if metric.name not in data.columns and metric.name != "concurrency":
                return False
        return True

    def create_plot(
        self, spec: PlotSpec, data: pd.DataFrame, available_metrics: dict
    ) -> go.Figure:
        """Create a Pareto curve plot."""
        x_metric = next(m for m in spec.metrics if m.axis == "x")
        y_metric = next(m for m in spec.metrics if m.axis == "y")

        if x_metric.name == "concurrency":
            x_label = "Concurrency Level"
        else:
            x_label = self._get_metric_label(
                x_metric.name, x_metric.stat or DEFAULT_PERCENTILE, available_metrics
            )

        y_label = self._get_metric_label(
            y_metric.name, y_metric.stat or "avg", available_metrics
        )

        return self.plot_generator.create_pareto_plot(
            df=data,
            x_metric=x_metric.name,
            y_metric=y_metric.name,
            label_by=spec.label_by,
            group_by=spec.group_by,
            title=spec.title,
            x_label=x_label,
            y_label=y_label,
        )


@PlotTypeHandlerFactory.register(PlotType.SCATTER_LINE)
class ScatterLineHandler(BaseMultiRunHandler):
    """Handler for scatter line plots."""

    def can_handle(self, spec: PlotSpec, data: pd.DataFrame) -> bool:
        """Check if scatter line plot can be generated."""
        for metric in spec.metrics:
            if metric.name not in data.columns and metric.name != "concurrency":
                return False
        return True

    def create_plot(
        self, spec: PlotSpec, data: pd.DataFrame, available_metrics: dict
    ) -> go.Figure:
        """Create a scatter line plot."""
        x_metric = next(m for m in spec.metrics if m.axis == "x")
        y_metric = next(m for m in spec.metrics if m.axis == "y")

        if x_metric.name == "concurrency":
            x_label = "Concurrency Level"
        else:
            x_label = self._get_metric_label(
                x_metric.name, x_metric.stat or DEFAULT_PERCENTILE, available_metrics
            )

        y_label = self._get_metric_label(
            y_metric.name, y_metric.stat or "avg", available_metrics
        )

        return self.plot_generator.create_scatter_line_plot(
            df=data,
            x_metric=x_metric.name,
            y_metric=y_metric.name,
            label_by=spec.label_by,
            group_by=spec.group_by,
            title=spec.title,
            x_label=x_label,
            y_label=y_label,
        )
