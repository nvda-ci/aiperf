#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Generate plotly plots using actual AIPerf PlotGenerator for comparison with matplotlib.
"""

import sys
import pandas as pd
from pathlib import Path

# Add src to path so we can import aiperf modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

from aiperf.plot.core.plot_generator import PlotGenerator
from aiperf.plot.constants import PlotTheme


def example_comparison():
    """Generate light and dark theme plots using actual PlotGenerator."""

    # Sample data (mimicking AIPerf metrics)
    df = pd.DataFrame({
        'concurrency': [1, 2, 4, 8, 16] * 2,
        'request_throughput': [10, 22, 38, 60, 85, 12, 25, 42, 68, 95],
        'time_to_first_token': [45, 48, 52, 58, 70, 42, 46, 50, 55, 65],
        'model': ['Qwen-0.6B']*5 + ['Llama-3B']*5,
    })

    # Light theme
    print("Generating Light Theme plot (Plotly)...")
    generator_light = PlotGenerator(theme=PlotTheme.LIGHT, color_pool_size=10)
    fig_light = generator_light.create_scatter_line_plot(
        df=df,
        x_metric='request_throughput',
        y_metric='time_to_first_token',
        label_by='concurrency',
        group_by='model',
        title='TTFT vs Throughput (Light Theme - Plotly)',
        x_label='Request Throughput (req/s)',
        y_label='Time to First Token (ms)',
    )
    fig_light.write_image(
        '/tmp/nvidia_plotly_light.png',
        width=1600,
        height=800,
        scale=150/100,
    )
    print("✅ Light theme saved to /tmp/nvidia_plotly_light.png")

    # Dark theme
    print("\nGenerating Dark Theme plot (Plotly)...")
    generator_dark = PlotGenerator(theme=PlotTheme.DARK, color_pool_size=10)
    fig_dark = generator_dark.create_scatter_line_plot(
        df=df,
        x_metric='request_throughput',
        y_metric='time_to_first_token',
        label_by='concurrency',
        group_by='model',
        title='TTFT vs Throughput (Dark Theme - Plotly)',
        x_label='Request Throughput (req/s)',
        y_label='Time to First Token (ms)',
    )
    fig_dark.write_image(
        '/tmp/nvidia_plotly_dark.png',
        width=1600,
        height=800,
        scale=150/100,
    )
    print("✅ Dark theme saved to /tmp/nvidia_plotly_dark.png")

    print("\n" + "="*70)
    print("PLOTLY PLOTS GENERATED")
    print("="*70)
    print("\nUsing actual AIPerf PlotGenerator:")
    print("  ✅ NVIDIA brand colors and styling")
    print("  ✅ Light & dark theme support")
    print("  ✅ Professional quality")
    print("  ⚠️  Requires Chrome (346MB binary + 294MB libs)")
    print("\nImage sizes:")
    import os
    light_size = os.path.getsize('/tmp/nvidia_plotly_light.png') / 1024
    dark_size = os.path.getsize('/tmp/nvidia_plotly_dark.png') / 1024
    print(f"  Light theme: {light_size:.1f} KB")
    print(f"  Dark theme: {dark_size:.1f} KB")

    print("\n" + "="*70)
    print("COMPARISON FILES")
    print("="*70)
    print("\nMatplotlib (no Chrome):")
    print("  /tmp/nvidia_matplotlib_light.png")
    print("  /tmp/nvidia_matplotlib_dark.png")
    print("\nPlotly (requires Chrome):")
    print("  /tmp/nvidia_plotly_light.png")
    print("  /tmp/nvidia_plotly_dark.png")
    print("\n👀 Open them side-by-side to compare!")


if __name__ == "__main__":
    example_comparison()

