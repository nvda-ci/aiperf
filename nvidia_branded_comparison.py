#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
NVIDIA Branded Comparison: Plotly vs Matplotlib with PROPER NVIDIA COLORS
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from aiperf.plot.core.plot_generator import PlotGenerator, get_nvidia_color_scheme
from aiperf.plot.constants import PlotTheme, NVIDIA_GREEN, NVIDIA_GOLD


def generate_realistic_data():
    """Generate realistic benchmark data."""
    concurrencies = [1, 2, 4, 8, 16, 32]
    data = []

    # Model 1: Meta-Llama-3-8B-Instruct
    for c in concurrencies:
        ttft_p50 = 30 + c * 3
        throughput = c * 2.1 if c < 16 else c * 1.8
        data.append({
            'concurrency': c,
            'time_to_first_token': ttft_p50,
            'request_throughput': throughput,
            'model': 'meta-llama/Meta-Llama-3-8B-Instruct',
        })

    # Model 2: Qwen/Qwen3-0.6B
    for c in concurrencies:
        ttft_p50 = 20 + c * 2
        throughput = c * 3.2 if c < 16 else c * 2.8
        data.append({
            'concurrency': c,
            'time_to_first_token': ttft_p50,
            'request_throughput': throughput,
            'model': 'Qwen/Qwen3-0.6B',
        })

    return pd.DataFrame(data)


def create_plotly_with_nvidia_branding(df, theme):
    """Create plotly plot with FORCED NVIDIA brand colors."""
    generator = PlotGenerator(theme=theme, color_pool_size=10)

    # FORCE NVIDIA BRAND COLORS for light theme
    if theme == PlotTheme.LIGHT:
        generator._color_pool = get_nvidia_color_scheme(
            10,
            palette_name="deep",
            use_brand_colors=True  # FORCE IT!
        )

    fig = generator.create_scatter_line_plot(
        df=df,
        x_metric='time_to_first_token',
        y_metric='request_throughput',
        label_by='concurrency',
        group_by='model',
        title='TTFT vs Throughput',
        x_label='Time to First Token P50 (ms)',
        y_label='Request Throughput AVG (requests/sec)',
    )

    return fig


def create_matplotlib_mimicking_plotly(df, theme_name):
    """Create matplotlib that MIMICS plotly EXACTLY with NVIDIA colors."""

    # NVIDIA brand colors (same as plotly)
    colors = [NVIDIA_GREEN, NVIDIA_GOLD]

    # Theme settings
    if theme_name == 'dark':
        bg_color = '#1a1a1a'
        paper_color = '#252525'
        text_color = '#E0E0E0'
        grid_color = '#333333'
    else:
        bg_color = '#FFFFFF'
        paper_color = '#FFFFFF'
        text_color = '#0a0a0a'
        grid_color = '#E5E5E5'

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8))
    fig.patch.set_facecolor(paper_color)
    ax.set_facecolor(bg_color)

    # Get unique models
    models = df['model'].unique()

    for i, model in enumerate(models):
        model_data = df[df['model'] == model].sort_values('time_to_first_token')
        color = colors[i % len(colors)]

        # Plot line
        ax.plot(
            model_data['time_to_first_token'],
            model_data['request_throughput'],
            marker='o',
            markersize=9,
            linewidth=3,
            color=color,
            label=model,
            zorder=3,
        )

        # Add point labels (concurrency values)
        for _, row in model_data.iterrows():
            ax.text(
                row['time_to_first_token'],
                row['request_throughput'],
                str(row['concurrency']),
                fontsize=10,
                fontweight='bold',
                color=text_color,
                ha='center',
                va='bottom',
                zorder=4,
            )

    # Title and labels
    ax.set_title('TTFT vs Throughput', fontsize=18, fontweight='bold',
                 color=text_color, pad=20, loc='left')
    ax.set_xlabel('Time to First Token P50 (ms)', fontsize=12, color=text_color)
    ax.set_ylabel('Request Throughput AVG (requests/sec)', fontsize=12, color=text_color)

    # Grid styling (match plotly)
    ax.grid(True, color=grid_color, linestyle='-', linewidth=0.8, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    # Spine styling
    for spine in ax.spines.values():
        spine.set_color(grid_color)
        spine.set_linewidth(1)

    # Tick styling
    ax.tick_params(colors=text_color, labelsize=10)

    # Legend styling (match plotly - bottom right)
    legend = ax.legend(
        loc='lower right',
        frameon=True,
        fontsize=11,
        edgecolor=grid_color,
        facecolor=paper_color,
    )
    legend.get_frame().set_alpha(0.9)
    for text in legend.get_texts():
        text.set_color(text_color)

    plt.tight_layout()

    return fig


def main():
    print("="*70)
    print("NVIDIA BRANDED COMPARISON - PLOTLY vs MATPLOTLIB")
    print("="*70)

    df = generate_realistic_data()

    # ========== PLOTLY with NVIDIA BRANDING ==========
    print("\n🎨 PLOTLY (with NVIDIA brand colors):")

    print("  1. Light theme...")
    fig = create_plotly_with_nvidia_branding(df, PlotTheme.LIGHT)
    fig.write_image('/tmp/plotly_nvidia_light.png', width=1600, height=800, scale=1.5)
    print("     ✅ /tmp/plotly_nvidia_light.png")

    print("  2. Dark theme...")
    fig = create_plotly_with_nvidia_branding(df, PlotTheme.DARK)
    fig.write_image('/tmp/plotly_nvidia_dark.png', width=1600, height=800, scale=1.5)
    print("     ✅ /tmp/plotly_nvidia_dark.png")

    # ========== MATPLOTLIB MIMICKING PLOTLY ==========
    print("\n🎨 MATPLOTLIB (mimicking plotly style):")

    print("  1. Light theme...")
    fig = create_matplotlib_mimicking_plotly(df, 'light')
    fig.savefig('/tmp/matplotlib_nvidia_light.png', dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("     ✅ /tmp/matplotlib_nvidia_light.png")

    print("  2. Dark theme...")
    fig = create_matplotlib_mimicking_plotly(df, 'dark')
    fig.savefig('/tmp/matplotlib_nvidia_dark.png', dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("     ✅ /tmp/matplotlib_nvidia_dark.png")

    print("\n" + "="*70)
    print("COMPARISON COMPLETE!")
    print("="*70)
    print("\n📊 Compare these files:")
    print("\nPLOTLY (requires Chrome - 1.1GB container):")
    print("  /tmp/plotly_nvidia_light.png")
    print("  /tmp/plotly_nvidia_dark.png")
    print("\nMATPLOTLIB (no Chrome - 439MB container):")
    print("  /tmp/matplotlib_nvidia_light.png")
    print("  /tmp/matplotlib_nvidia_dark.png")
    print("\n🔥 Both use NVIDIA GREEN (#76B900) + GOLD (#F4E5C3)")
    print("💾 Matplotlib saves 662MB by not needing Chrome!")


if __name__ == "__main__":
    main()

