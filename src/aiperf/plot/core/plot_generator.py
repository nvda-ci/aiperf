# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Plot generation module for AIPerf visualization.

This module provides the PlotGenerator class which creates Plotly Figure objects
with NVIDIA brand styling for various plot types including pareto curves, scatter
plots, line charts, and time series.
"""

import pandas as pd
import plotly.graph_objects as go
import seaborn as sns

from aiperf.plot.constants import (
    NVIDIA_BORDER,
    NVIDIA_DARK_BG,
    NVIDIA_GOLD,
    NVIDIA_GRAY,
    NVIDIA_GREEN,
    NVIDIA_TEXT_LIGHT,
    PLOT_FONT_FAMILY,
)


def get_nvidia_color_scheme(n_colors: int) -> list[str]:
    """
    Generate color scheme with NVIDIA brand colors first, then seaborn for extras.

    Uses NVIDIA green and gold for the first two colors, then dynamically generates
    additional colors using seaborn's "bright" palette for any remaining colors needed.

    Args:
        n_colors: Number of colors needed

    Returns:
        List of hex color strings
    """
    custom_colors = [NVIDIA_GREEN, NVIDIA_GOLD]

    if n_colors <= len(custom_colors):
        return custom_colors[:n_colors]

    additional_needed = n_colors - len(custom_colors)
    palette = sns.color_palette("bright", additional_needed)

    additional = [
        f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"
        for r, g, b in palette
    ]

    return custom_colors + additional


class PlotGenerator:
    """Generate Plotly figures for AIPerf profiling data with NVIDIA branding.

    This class provides generic, reusable plot functions that can visualize any
    metric combination. All plots use NVIDIA brand colors and dark mode styling
    for professional presentations.
    """

    def _assign_model_colors(self, models: list[str]) -> dict[str, str]:
        """Dynamically assign colors to models from NVIDIA palette.

        Colors are assigned in a consistent order based on sorted model names,
        ensuring the same model always gets the same color across different plots.

        Args:
            models: List of unique model names from the data

        Returns:
            Dictionary mapping model name to color hex code
        """
        sorted_models = sorted(models)  # Sort for consistency
        color_scheme = get_nvidia_color_scheme(len(sorted_models))

        color_map = {}
        for i, model in enumerate(sorted_models):
            color_map[model] = color_scheme[i]

        return color_map

    def _get_base_layout(
        self,
        title: str,
        x_label: str,
        y_label: str,
        hovermode: str | None = None,
    ) -> dict:
        """
        Get base layout configuration with NVIDIA branding.

        Provides consistent styling (fonts, colors, margins, grid) that can be
        applied to all plot types. This is the single source of truth for
        NVIDIA brand styling.

        Args:
            title: Plot title text
            x_label: X-axis label text
            y_label: Y-axis label text
            hovermode: Optional hover mode (e.g., "x unified")

        Returns:
            Dictionary of layout configuration ready for fig.update_layout()
        """
        layout = {
            "title": {
                "text": title,
                "font": {
                    "size": 13,
                    "family": PLOT_FONT_FAMILY,
                    "weight": "bold",
                    "color": NVIDIA_TEXT_LIGHT,
                },
            },
            "xaxis_title": x_label,
            "yaxis_title": y_label,
            "template": "plotly_dark",
            "font": {"size": 10, "family": PLOT_FONT_FAMILY, "color": NVIDIA_GRAY},
            "height": 600,
            "margin": {"l": 50, "r": 10, "t": 40, "b": 40},
            "plot_bgcolor": NVIDIA_DARK_BG,
            "paper_bgcolor": NVIDIA_DARK_BG,
            "xaxis": {
                "gridcolor": "#2a2a2a",
                "showline": True,
                "linecolor": NVIDIA_BORDER,
                "color": NVIDIA_TEXT_LIGHT,
            },
            "yaxis": {
                "gridcolor": "#2a2a2a",
                "showline": True,
                "linecolor": NVIDIA_BORDER,
                "color": NVIDIA_TEXT_LIGHT,
            },
            "legend": {
                "font": {
                    "size": 11,
                    "family": PLOT_FONT_FAMILY,
                    "color": NVIDIA_TEXT_LIGHT,
                },
                "bgcolor": "rgba(37, 37, 37, 0.8)",
                "bordercolor": NVIDIA_BORDER,
                "borderwidth": 1,
                "x": 0.98,
                "y": 0.02,
                "xanchor": "right",
                "yanchor": "bottom",
            },
        }

        if hovermode:
            layout["hovermode"] = hovermode

        return layout

    def _prepare_groups(
        self, df: pd.DataFrame, group_by: str | None
    ) -> tuple[list[str | None], dict[str, str]]:
        """
        Prepare group list and color mapping for multi-series plots.

        Handles grouping logic and color assignment in a consistent way across
        all plot types that support grouping (e.g., by model).

        Args:
            df: DataFrame containing the data
            group_by: Column name to group by (e.g., "model"), or None for no grouping

        Returns:
            Tuple of (groups, model_colors) where:
            - groups: Sorted list of group values, or [None] if no grouping
            - model_colors: Dict mapping group names to color hex codes
        """
        if group_by and group_by in df.columns:
            groups = sorted(df[group_by].unique(), reverse=True)
            model_colors = self._assign_model_colors(list(groups))
        else:
            groups = [None]
            model_colors = {}

        return groups, model_colors

    def create_pareto_plot(
        self,
        df: pd.DataFrame,
        x_metric: str,
        y_metric: str,
        label_by: str = "concurrency",
        group_by: str | None = "model",
        title: str | None = None,
        x_label: str | None = None,
        y_label: str | None = None,
    ) -> go.Figure:
        """Create a Pareto curve plot showing trade-offs between two metrics.

        The Pareto frontier is calculated automatically, highlighting optimal
        configurations where improving one metric doesn't worsen the other.

        Args:
            df: DataFrame containing the metrics
            x_metric: Column name for x-axis metric (e.g., "latency")
            y_metric: Column name for y-axis metric (e.g., "throughput")
            label_by: Column to use for point labels (default: "concurrency")
            group_by: Column to group data by for multi-series (default: "model")
            title: Plot title (auto-generated if None)
            x_label: X-axis label (auto-generated if None)
            y_label: Y-axis label (auto-generated if None)

        Returns:
            Plotly Figure object with pareto curve and data points
        """
        df_sorted = df.sort_values(x_metric)
        fig = go.Figure()

        # Auto-generate labels if not provided
        if title is None:
            title = f"Pareto Curve: {y_metric.replace('_', ' ').title()} vs {x_metric.replace('_', ' ').title()}"
        if x_label is None:
            x_label = x_metric.replace("_", " ").title()
        if y_label is None:
            y_label = y_metric.replace("_", " ").title()

        # Prepare groups and colors
        groups, model_colors = self._prepare_groups(df_sorted, group_by)

        for group in groups:
            if group is None:
                group_data = df_sorted
                group_color = NVIDIA_GREEN
                group_name = "Data"
            else:
                group_data = df_sorted[df_sorted[group_by] == group].sort_values(
                    x_metric
                )
                group_color = model_colors.get(group, NVIDIA_GREEN)
                group_name = group

            # Calculate Pareto frontier for this group using vectorized operations
            max_y_cumulative = group_data[y_metric].cummax()
            is_pareto = group_data[y_metric] == max_y_cumulative
            df_pareto = group_data[is_pareto].copy()

            if not df_pareto.empty:
                # Shadow for Pareto line
                fig.add_trace(
                    go.Scatter(
                        x=df_pareto[x_metric],
                        y=df_pareto[y_metric],
                        mode="lines",
                        line=dict(width=8, color="rgba(255, 255, 255, 0.1)"),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

                # Main Pareto line
                fig.add_trace(
                    go.Scatter(
                        x=df_pareto[x_metric],
                        y=df_pareto[y_metric],
                        mode="lines",
                        line=dict(width=3, color=group_color),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

            # Prepare labels and hover text
            labels = [str(val) for val in group_data[label_by]]
            hovertexts = [
                f"<b>{group_name} - {label}</b><br>{x_label}: {x:.1f}<br>{y_label}: {y:.1f}"
                for label, x, y in zip(
                    labels, group_data[x_metric], group_data[y_metric], strict=False
                )
            ]

            # Shadow layer for markers
            fig.add_trace(
                go.Scatter(
                    x=group_data[x_metric],
                    y=group_data[y_metric],
                    mode="markers",
                    marker=dict(
                        size=14,
                        symbol="circle",
                        color="rgba(255, 255, 255, 0.15)",
                        line=dict(width=0),
                    ),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

            # Main markers
            fig.add_trace(
                go.Scatter(
                    x=group_data[x_metric],
                    y=group_data[y_metric],
                    mode="markers+text",
                    marker=dict(
                        size=9,
                        symbol="circle",
                        color=group_color,
                        line=dict(width=0),
                    ),
                    text=labels,
                    textposition="top center",
                    textfont=dict(
                        size=10,
                        color=NVIDIA_TEXT_LIGHT,
                        family=PLOT_FONT_FAMILY,
                        weight="bold",
                    ),
                    hovertemplate="%{customdata}<extra></extra>",
                    customdata=hovertexts,
                    name=group_name,
                    showlegend=(group is not None),
                    legendgroup=group_name,
                )
            )

        # Apply NVIDIA branding layout
        layout = self._get_base_layout(title, x_label, y_label)
        fig.update_layout(layout)

        return fig

    def create_scatter_line_plot(
        self,
        df: pd.DataFrame,
        x_metric: str,
        y_metric: str,
        label_by: str = "concurrency",
        group_by: str | None = "model",
        title: str | None = None,
        x_label: str | None = None,
        y_label: str | None = None,
    ) -> go.Figure:
        """Create a scatter plot with connecting lines.

        Args:
            df: DataFrame containing the metrics
            x_metric: Column name for x-axis metric
            y_metric: Column name for y-axis metric
            label_by: Column to use for point labels (default: "concurrency")
            group_by: Column to group data by for multi-series (default: "model")
            title: Plot title (auto-generated if None)
            x_label: X-axis label (auto-generated if None)
            y_label: Y-axis label (auto-generated if None)

        Returns:
            Plotly Figure object with scatter plot and lines
        """
        df_sorted = df.sort_values(x_metric)
        fig = go.Figure()

        # Auto-generate labels if not provided
        if title is None:
            title = f"{y_metric.replace('_', ' ').title()} vs {x_metric.replace('_', ' ').title()}"
        if x_label is None:
            x_label = x_metric.replace("_", " ").title()
        if y_label is None:
            y_label = y_metric.replace("_", " ").title()

        # Prepare groups and colors
        groups, model_colors = self._prepare_groups(df_sorted, group_by)

        for group in groups:
            if group is None:
                group_data = df_sorted
                group_color = NVIDIA_GREEN
                group_name = "Data"
            else:
                group_data = df_sorted[df_sorted[group_by] == group].sort_values(
                    x_metric
                )
                group_color = model_colors.get(group, NVIDIA_GREEN)
                group_name = group

            # Shadow layer
            fig.add_trace(
                go.Scatter(
                    x=group_data[x_metric],
                    y=group_data[y_metric],
                    mode="lines+markers",
                    marker=dict(
                        size=14,
                        color="rgba(255, 255, 255, 0.12)",
                        symbol="circle",
                        line=dict(width=0),
                    ),
                    line=dict(width=8, color="rgba(255, 255, 255, 0.08)"),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

            # Main trace
            labels = [str(val) for val in group_data[label_by]]
            fig.add_trace(
                go.Scatter(
                    x=group_data[x_metric],
                    y=group_data[y_metric],
                    mode="lines+markers+text",
                    marker=dict(
                        size=9,
                        color=group_color,
                        symbol="circle",
                        line=dict(width=0),
                    ),
                    line=dict(width=3, color=group_color),
                    text=labels,
                    textposition="top center",
                    textfont=dict(
                        size=9, color=NVIDIA_TEXT_LIGHT, family=PLOT_FONT_FAMILY
                    ),
                    hovertemplate=f"<b>{group_name} - %{{text}}</b><br>{x_label}: %{{x:.1f}}<br>{y_label}: %{{y:.1f}}<extra></extra>",
                    name=group_name,
                    showlegend=(group is not None),
                    legendgroup=group_name,
                )
            )

        # Apply NVIDIA branding layout
        layout = self._get_base_layout(title, x_label, y_label)
        fig.update_layout(layout)

        return fig

    def create_time_series_scatter(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_metric: str,
        title: str | None = None,
        x_label: str | None = None,
        y_label: str | None = None,
    ) -> go.Figure:
        """Create a time series scatter plot.

        Args:
            df: DataFrame containing the time series data
            x_col: Column name for x-axis (e.g., "request_number" or "timestamp")
            y_metric: Column name for y-axis metric
            title: Plot title (auto-generated if None)
            x_label: X-axis label (auto-generated if None)
            y_label: Y-axis label (auto-generated if None)

        Returns:
            Plotly Figure object with time series scatter plot
        """
        fig = go.Figure()

        # Auto-generate labels if not provided
        if title is None:
            title = f"{y_metric.replace('_', ' ').title()} Over Time"
        if x_label is None:
            x_label = x_col.replace("_", " ").title()
        if y_label is None:
            y_label = y_metric.replace("_", " ").title()

        # Main scatter points
        fig.add_trace(
            go.Scatter(
                x=df[x_col],
                y=df[y_metric],
                mode="markers",
                marker=dict(size=8, opacity=0.95, color=NVIDIA_GOLD),
                showlegend=False,
                hovertemplate=f"{x_label} %{{x}}<br>{y_label}: %{{y:.1f}}<extra></extra>",
            )
        )

        # Apply NVIDIA branding layout with unified hover
        layout = self._get_base_layout(title, x_label, y_label, hovermode="x unified")
        fig.update_layout(layout)

        return fig

    def create_time_series_area(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_metric: str,
        title: str | None = None,
        x_label: str | None = None,
        y_label: str | None = None,
    ) -> go.Figure:
        """Create a time series area plot with filled region.

        Args:
            df: DataFrame containing the time series data
            x_col: Column name for x-axis (e.g., "timestamp")
            y_metric: Column name for y-axis metric
            title: Plot title (auto-generated if None)
            x_label: X-axis label (auto-generated if None)
            y_label: Y-axis label (auto-generated if None)

        Returns:
            Plotly Figure object with filled area plot
        """
        fig = go.Figure()

        # Auto-generate labels if not provided
        if title is None:
            title = f"{y_metric.replace('_', ' ').title()} Over Time"
        if x_label is None:
            x_label = x_col.replace("_", " ").title()
        if y_label is None:
            y_label = y_metric.replace("_", " ").title()

        # Main trace with fill
        fig.add_trace(
            go.Scatter(
                x=df[x_col],
                y=df[y_metric],
                mode="lines",
                line=dict(width=2, color=NVIDIA_GREEN),
                fill="tozeroy",
                fillcolor="rgba(118, 185, 0, 0.2)",
                showlegend=False,
                hovertemplate=f"{x_label}: %{{x:.0f}}<br>{y_label}: %{{y:.1f}}<extra></extra>",
            )
        )

        # Apply NVIDIA branding layout
        layout = self._get_base_layout(title, x_label, y_label)
        fig.update_layout(layout)

        return fig
