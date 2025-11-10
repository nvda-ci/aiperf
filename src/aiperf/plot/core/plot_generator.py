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
    DARK_THEME_COLORS,
    LIGHT_THEME_COLORS,
    NVIDIA_BORDER,
    NVIDIA_GOLD,
    NVIDIA_GREEN,
    NVIDIA_TEXT_LIGHT,
    PLOT_FONT_FAMILY,
    PlotTheme,
)


def get_nvidia_color_scheme(
    n_colors: int, secondary_color: str = NVIDIA_GOLD
) -> list[str]:
    """
    Generate color scheme with NVIDIA brand colors first, then seaborn for extras.

    Uses NVIDIA green and a secondary color (gold for dark theme, grey for light theme)
    for the first two colors, then dynamically generates additional colors using
    seaborn's "bright" palette for any remaining colors needed.

    Args:
        n_colors: Number of colors needed
        secondary_color: Secondary color to use (NVIDIA_GOLD or NVIDIA_GRAY)

    Returns:
        List of hex color strings
    """
    custom_colors = [NVIDIA_GREEN, secondary_color]

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
    metric combination. Plots can use either light mode (default) or dark mode
    styling for professional presentations.

    Args:
        theme: Theme to use for plots (LIGHT or DARK). Defaults to LIGHT.
    """

    def __init__(self, theme: PlotTheme = PlotTheme.LIGHT):
        """Initialize PlotGenerator with specified theme.

        Args:
            theme: Theme to use for plots (LIGHT or DARK). Defaults to LIGHT.
        """
        self.theme = theme
        self.colors = (
            LIGHT_THEME_COLORS if theme == PlotTheme.LIGHT else DARK_THEME_COLORS
        )

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
        color_scheme = get_nvidia_color_scheme(
            len(sorted_models), self.colors["secondary"]
        )

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
        template = "plotly_dark" if self.theme == PlotTheme.DARK else "plotly_white"

        layout = {
            "title": {
                "text": title,
                "font": {
                    "size": 18,
                    "family": PLOT_FONT_FAMILY,
                    "weight": "bold",
                    "color": self.colors["text"],
                },
            },
            "xaxis_title": x_label,
            "yaxis_title": y_label,
            "template": template,
            "font": {
                "size": 10,
                "family": PLOT_FONT_FAMILY,
                "color": self.colors["text"],
            },
            "height": 600,
            "margin": {"l": 50, "r": 10, "t": 60, "b": 40},
            "plot_bgcolor": self.colors["background"],
            "paper_bgcolor": self.colors["paper"],
            "xaxis": {
                "gridcolor": self.colors["grid"],
                "showline": True,
                "linecolor": self.colors["border"],
                "color": self.colors["text"],
            },
            "yaxis": {
                "gridcolor": self.colors["grid"],
                "showline": True,
                "linecolor": self.colors["border"],
                "color": self.colors["text"],
            },
            "legend": {
                "font": {
                    "size": 11,
                    "family": PLOT_FONT_FAMILY,
                    "color": self.colors["text"],
                },
                "bgcolor": f"rgba({int(self.colors['paper'][1:3], 16)}, {int(self.colors['paper'][3:5], 16)}, {int(self.colors['paper'][5:7], 16)}, 0.8)",
                "bordercolor": self.colors["border"],
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
                        color=self.colors["text"],
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
                        size=9, color=self.colors["text"], family=PLOT_FONT_FAMILY
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
                marker=dict(size=8, opacity=0.95, color=self.colors["secondary"]),
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
                line=dict(width=2, color=NVIDIA_GREEN, shape="hv"),
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

    def create_time_series_histogram(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        title: str | None = None,
        x_label: str | None = None,
        y_label: str | None = None,
        slice_duration: float | None = None,
        warning_text: str | None = None,
    ) -> go.Figure:
        """Create a time series histogram/bar chart.

        Args:
            df: DataFrame containing the time series data
            x_col: Column name for x-axis (e.g., "Timeslice")
            y_col: Column name for y-axis values (e.g., "avg", "p50", "p90")
            title: Plot title (auto-generated if None)
            x_label: X-axis label (auto-generated if None)
            y_label: Y-axis label (auto-generated if None)
            slice_duration: Duration of each slice in seconds (for time-based x-axis)
            warning_text: Optional warning text to display at bottom of plot

        Returns:
            Plotly Figure object with bar chart
        """
        fig = go.Figure()

        # Auto-generate labels if not provided
        if title is None:
            title = f"{y_col.replace('_', ' ').title()} Over Time"
        if x_label is None:
            x_label = "Time (s)" if slice_duration else x_col.replace("_", " ").title()
        if y_label is None:
            y_label = y_col.replace("_", " ").title()

        # Prepare x-axis values and bar configuration
        if slice_duration is not None:
            # Use continuous time scale
            slice_indices = df[x_col].values
            # X-values are the center of each slice (bars are centered on x-value in plotly)
            x_values = slice_indices * slice_duration + slice_duration / 2
            # Bar width equals slice duration for continuous coverage
            bar_width = slice_duration

            # Prepare hover data with time ranges and slice indices
            slice_start_times = slice_indices * slice_duration
            time_ranges = [
                f"{int(start)}s-{int(start + slice_duration)}s"
                for start in slice_start_times
            ]
            hover_template = (
                f"Time: %{{customdata[0]}}<br>"
                f"Slice: %{{customdata[1]}}<br>"
                f"{y_label}: %{{y:.2f}}<extra></extra>"
            )
            customdata = list(zip(time_ranges, slice_indices.astype(int), strict=False))

            # Transparent bars with borders
            marker_config = dict(
                color="rgba(118, 185, 0, 0.7)",  # 70% opacity
                line=dict(color=NVIDIA_GREEN, width=2),
            )
        else:
            # Fallback for non-time-based data
            x_values = df[x_col]
            bar_width = None
            hover_template = (
                f"{x_label}: %{{x}}<br>{y_label}: %{{y:.2f}}<extra></extra>"
            )
            customdata = None
            marker_config = dict(
                color=NVIDIA_GREEN,
                line=dict(color=NVIDIA_GREEN, width=0),
            )

        # Create bar chart
        fig.add_trace(
            go.Bar(
                x=x_values,
                y=df[y_col],
                width=bar_width,
                marker=marker_config,
                showlegend=False,
                hovertemplate=hover_template,
                customdata=customdata,
            )
        )

        # Apply NVIDIA branding layout
        layout = self._get_base_layout(title, x_label, y_label, hovermode="x unified")

        # Configure x-axis for continuous time
        layout["bargap"] = 0
        layout["bargroupgap"] = 0
        if slice_duration is not None:
            # Primary x-axis: Time values at boundaries
            slice_indices = df[x_col].values
            max_slice = slice_indices.max()
            layout["xaxis"]["dtick"] = slice_duration
            layout["xaxis"]["tick0"] = 0
            layout["xaxis"]["range"] = [0, (max_slice + 1) * slice_duration]

            # Add slice index labels as annotations at the top of each bar
            slice_centers = slice_indices * slice_duration + slice_duration / 2
            bar_heights = df[y_col].values
            annotations = []
            for idx, center, height in zip(
                slice_indices, slice_centers, bar_heights, strict=False
            ):
                annotations.append(
                    dict(
                        x=center,
                        y=height,
                        yshift=5,
                        xref="x",
                        yref="y",
                        text=str(int(idx)),
                        showarrow=False,
                        font=dict(
                            size=12,
                            family=PLOT_FONT_FAMILY,
                            color=self.colors["text"],
                            weight="bold",
                        ),
                        xanchor="center",
                        yanchor="bottom",
                    )
                )
            layout["annotations"] = annotations

            # Add legend entry explaining slice numbers
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="none",
                    showlegend=True,
                    name="Bar numbers indicate time slice index",
                    hoverinfo="skip",
                )
            )
            layout["showlegend"] = True

        if warning_text:
            if "annotations" not in layout:
                layout["annotations"] = []

            layout["margin"]["b"] = 140

            warning_annotation = dict(
                x=0.5,
                y=-0.10,
                xref="paper",
                yref="paper",
                text=warning_text,
                showarrow=False,
                font=dict(
                    size=11, family=PLOT_FONT_FAMILY, color=self.colors["secondary"]
                ),
                bgcolor=f"rgba({int(self.colors['secondary'][1:3], 16)}, {int(self.colors['secondary'][3:5], 16)}, {int(self.colors['secondary'][5:7], 16)}, 0.1)",
                bordercolor=self.colors["secondary"],
                borderwidth=2,
                borderpad=8,
                xanchor="center",
                yanchor="top",
            )
            layout["annotations"] = list(layout.get("annotations", [])) + [
                warning_annotation
            ]

        fig.update_layout(layout)

        return fig

    def create_gpu_dual_axis_plot(
        self,
        df_primary: pd.DataFrame,
        df_secondary: pd.DataFrame,
        x_col_primary: str,
        x_col_secondary: str,
        y1_metric: str,
        y2_metric: str,
        active_count_col: str | None = None,
        title: str | None = None,
        x_label: str | None = None,
        y1_label: str | None = None,
        y2_label: str | None = None,
    ) -> go.Figure:
        """
        Create a dual Y-axis plot with independent data sources.

        Primary metric (left Y-axis, typically throughput) is plotted as a step function.
        Secondary metric (right Y-axis, typically GPU utilization) is plotted as filled area.

        Args:
            df_primary: DataFrame for primary metric (left Y-axis)
            df_secondary: DataFrame for secondary metric (right Y-axis)
            x_col_primary: Column name for x-axis in primary DataFrame
            x_col_secondary: Column name for x-axis in secondary DataFrame
            y1_metric: Column name for primary y-axis (left)
            y2_metric: Column name for secondary y-axis (right)
            active_count_col: Optional column name in df_primary for active request count (for tooltip)
            title: Plot title (auto-generated if None)
            x_label: X-axis label (auto-generated if None)
            y1_label: Primary Y-axis label (auto-generated if None)
            y2_label: Secondary Y-axis label (auto-generated if None)

        Returns:
            Plotly Figure object with dual Y-axes
        """
        fig = go.Figure()

        if title is None:
            title = f"{y1_metric.replace('_', ' ').title()} with {y2_metric.replace('_', ' ').title()}"
        if x_label is None:
            x_label = "Time (s)"
        if y1_label is None:
            y1_label = y1_metric.replace("_", " ").title()
        if y2_label is None:
            y2_label = y2_metric.replace("_", " ").title()

        primary_hover = f"{x_label}: %{{x:.1f}}s<br>{y1_label}: %{{y:.1f}}"
        if active_count_col and active_count_col in df_primary.columns:
            primary_hover += "<br>Active Requests: %{customdata}"

        primary_hover += "<extra></extra>"

        customdata = df_primary[active_count_col] if active_count_col else None

        fig.add_trace(
            go.Scatter(
                x=df_primary[x_col_primary],
                y=df_primary[y1_metric],
                mode="lines",
                line=dict(width=2, color=NVIDIA_GREEN, shape="hv"),
                name=y1_label,
                yaxis="y",
                customdata=customdata,
                hovertemplate=primary_hover,
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df_secondary[x_col_secondary],
                y=df_secondary[y2_metric],
                mode="lines",
                line=dict(width=2, color=self.colors["secondary"]),
                fill="tozeroy",
                fillcolor=f"rgba({int(self.colors['secondary'][1:3], 16)}, {int(self.colors['secondary'][3:5], 16)}, {int(self.colors['secondary'][5:7], 16)}, 0.3)",
                name=y2_label,
                yaxis="y2",
                hovertemplate=f"{x_label}: %{{x:.1f}}s<br>{y2_label}: %{{y:.1f}}<extra></extra>",
            )
        )

        layout = self._get_base_layout(title, x_label, y1_label, hovermode="x unified")

        layout["yaxis2"] = {
            "title": y2_label,
            "overlaying": "y",
            "side": "right",
            "gridcolor": "rgba(42, 42, 42, 0.3)",
            "showline": True,
            "linecolor": NVIDIA_BORDER,
            "color": NVIDIA_TEXT_LIGHT,
        }

        layout["legend"]["x"] = 0.02
        layout["legend"]["xanchor"] = "left"

        fig.update_layout(layout)

        return fig

    def create_latency_scatter_with_percentiles(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_metric: str,
        percentile_cols: list[str],
        title: str | None = None,
        x_label: str | None = None,
        y_label: str | None = None,
    ) -> go.Figure:
        """
        Create a scatter plot with rolling percentile overlays for latency analysis.

        Displays individual request latencies as scatter points with overlaid percentile
        lines to provide statistical context. This visualization is ideal for identifying
        tail latency, temporal patterns, and debugging anomalies.

        Args:
            df: DataFrame containing the time series data with percentile columns
            x_col: Column name for x-axis (e.g., "timestamp")
            y_metric: Column name for y-axis metric (e.g., "request_latency")
            percentile_cols: List of column names for percentile lines (e.g., ["p50", "p95", "p99"])
            title: Plot title (auto-generated if None)
            x_label: X-axis label (auto-generated if None)
            y_label: Y-axis label (auto-generated if None)

        Returns:
            Plotly Figure object with scatter points and percentile lines
        """
        fig = go.Figure()

        # Auto-generate labels if not provided
        if title is None:
            title = f"{y_metric.replace('_', ' ').title()} Over Time with Percentiles"
        if x_label is None:
            x_label = x_col.replace("_", " ").title()
        if y_label is None:
            y_label = y_metric.replace("_", " ").title()

        # Get NVIDIA color scheme for percentile lines
        n_percentiles = len(percentile_cols)
        percentile_colors = get_nvidia_color_scheme(
            n_percentiles, self.colors["secondary"]
        )

        # Individual request scatter points (semi-transparent)
        fig.add_trace(
            go.Scatter(
                x=df[x_col],
                y=df[y_metric],
                mode="markers",
                marker=dict(
                    size=6,
                    opacity=0.4,
                    color=self.colors["secondary"],
                    line=dict(width=0),
                ),
                name="Individual Requests",
                hovertemplate=f"{x_label}: %{{x:.2f}}<br>{y_label}: %{{y:.2f}}<extra></extra>",
            )
        )

        # Add percentile lines with NVIDIA color palette
        for idx, percentile_col in enumerate(percentile_cols):
            if percentile_col not in df.columns:
                continue

            # Extract percentile number from column name (e.g., "p95" -> "p95")
            percentile_display = percentile_col.upper()
            color = percentile_colors[idx % len(percentile_colors)]

            fig.add_trace(
                go.Scatter(
                    x=df[x_col],
                    y=df[percentile_col],
                    mode="lines",
                    line=dict(width=2.5, color=color),
                    name=percentile_display,
                    hovertemplate=f"{x_label}: %{{x:.2f}}<br>{percentile_display}: %{{y:.2f}}<extra></extra>",
                )
            )

        # Apply NVIDIA branding layout with unified hover
        layout = self._get_base_layout(title, x_label, y_label, hovermode="x unified")
        fig.update_layout(layout)

        return fig

    def create_gpu_memory_stacked_area(
        self,
        df: pd.DataFrame,
        x_col: str,
        used_col: str,
        free_col: str,
        title: str | None = None,
        x_label: str | None = None,
        y_label: str | None = None,
    ) -> go.Figure:
        """
        Create a stacked area plot showing GPU memory breakdown.

        Args:
            df: DataFrame containing the time series data
            x_col: Column name for x-axis (e.g., "timestamp_s")
            used_col: Column name for used memory
            free_col: Column name for free memory
            title: Plot title (auto-generated if None)
            x_label: X-axis label (auto-generated if None)
            y_label: Y-axis label (auto-generated if None)

        Returns:
            Plotly Figure object with stacked area plot
        """
        fig = go.Figure()

        # Auto-generate labels if not provided
        if title is None:
            title = "GPU Memory Usage Over Time"
        if x_label is None:
            x_label = "Time (s)"
        if y_label is None:
            y_label = "Memory (GB)"

        # Used memory (bottom layer)
        fig.add_trace(
            go.Scatter(
                x=df[x_col],
                y=df[used_col],
                mode="lines",
                line=dict(width=0),
                fill="tozeroy",
                fillcolor="rgba(118, 185, 0, 0.6)",
                name="Used",
                hovertemplate=f"{x_label}: %{{x:.1f}}s<br>Used: %{{y:.2f}} GB<extra></extra>",
                stackgroup="one",
            )
        )

        # Free memory (top layer)
        fig.add_trace(
            go.Scatter(
                x=df[x_col],
                y=df[free_col],
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(153, 153, 153, 0.3)",
                name="Free",
                hovertemplate=f"{x_label}: %{{x:.1f}}s<br>Free: %{{y:.2f}} GB<extra></extra>",
                stackgroup="one",
            )
        )

        # Apply NVIDIA branding layout
        layout = self._get_base_layout(title, x_label, y_label, hovermode="x unified")
        fig.update_layout(layout)

        return fig

    def create_gpu_metrics_overlay(
        self,
        df: pd.DataFrame,
        x_col: str,
        power_col: str,
        temp_col: str,
        sm_clock_col: str,
        mem_clock_col: str,
        gpu_id_col: str = "gpu_uuid",
        title: str | None = None,
        x_label: str | None = None,
    ) -> go.Figure:
        """
        Create a multi-metric overlay plot for GPU telemetry data.

        Shows power, temperature, and clock speeds on dual Y-axes with separate
        lines per GPU UUID. Left axis shows power (W) and temperature (°C).
        Right axis shows clock frequencies (MHz).

        Args:
            df: DataFrame containing GPU telemetry time series data
            x_col: Column name for x-axis (e.g., "timestamp_s")
            power_col: Column name for power usage (W)
            temp_col: Column name for GPU temperature (°C)
            sm_clock_col: Column name for SM clock frequency (MHz)
            mem_clock_col: Column name for memory clock frequency (MHz)
            gpu_id_col: Column name for GPU identifier (default: "gpu_uuid")
            title: Plot title (auto-generated if None)
            x_label: X-axis label (auto-generated if None)

        Returns:
            Plotly Figure object with dual Y-axes and per-GPU traces
        """
        fig = go.Figure()

        if title is None:
            title = "GPU Metrics Over Time"
        if x_label is None:
            x_label = "Time (s)"

        gpu_ids = (
            sorted(df[gpu_id_col].unique()) if gpu_id_col in df.columns else [None]
        )
        n_gpus = len(gpu_ids)

        colors = get_nvidia_color_scheme(n_gpus, self.colors["secondary"])

        metric_configs = [
            {
                "col": power_col,
                "name": "Power",
                "unit": "W",
                "yaxis": "y",
                "dash": None,
                "color_idx": 0,
            },
            {
                "col": temp_col,
                "name": "Temperature",
                "unit": "°C",
                "yaxis": "y",
                "dash": "dot",
                "color_idx": 1,
            },
            {
                "col": sm_clock_col,
                "name": "SM Clock",
                "unit": "MHz",
                "yaxis": "y2",
                "dash": None,
                "color_idx": 2,
            },
            {
                "col": mem_clock_col,
                "name": "Memory Clock",
                "unit": "MHz",
                "yaxis": "y2",
                "dash": "dash",
                "color_idx": 3,
            },
        ]

        for gpu_idx, gpu_id in enumerate(gpu_ids):
            if gpu_id is None:
                gpu_df = df
                gpu_label = ""
            else:
                gpu_df = df[df[gpu_id_col] == gpu_id]
                gpu_label = f" (GPU-{gpu_idx})"

            base_color = colors[gpu_idx % len(colors)]

            for metric in metric_configs:
                if metric["col"] not in gpu_df.columns:
                    continue

                metric_name = f"{metric['name']} ({metric['unit']}){gpu_label}"

                fig.add_trace(
                    go.Scatter(
                        x=gpu_df[x_col],
                        y=gpu_df[metric["col"]],
                        mode="lines",
                        line=dict(
                            width=2,
                            color=base_color,
                            dash=metric["dash"],
                        ),
                        name=metric_name,
                        yaxis=metric["yaxis"],
                        hovertemplate=f"{x_label}: %{{x:.1f}}s<br>{metric['name']}: %{{y:.1f}} {metric['unit']}<extra></extra>",
                        legendgroup=f"gpu{gpu_idx}",
                    )
                )

        layout = self._get_base_layout(
            title, x_label, "Power (W) / Temp (°C)", hovermode="x unified"
        )

        layout["yaxis2"] = {
            "title": "Clock Frequency (MHz)",
            "overlaying": "y",
            "side": "right",
            "gridcolor": self.colors["grid"],
            "showline": True,
            "linecolor": self.colors["border"],
            "color": self.colors["text"],
        }

        layout["legend"]["x"] = 0.02
        layout["legend"]["xanchor"] = "left"
        layout["legend"]["tracegroupgap"] = 10

        fig.update_layout(layout)

        return fig
