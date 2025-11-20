#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
NVIDIA Branded Comparison - FIXED COLORS
Light theme: GREEN + GRAY (not that pale gold shit)
Dark theme: GREEN + GOLD (looks good on dark)
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from aiperf.plot.core.plot_generator import PlotGenerator
from aiperf.plot.constants import PlotTheme, NVIDIA_GREEN, NVIDIA_GOLD, NVIDIA_GRAY


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


def create_plotly_fixed_colors(df, theme):
    """Create plotly with PROPER colors: Light=GREEN+GRAY, Dark=GREEN+GOLD."""
    generator = PlotGenerator(theme=theme, color_pool_size=10)

    # Override color pool with correct colors
    if theme == PlotTheme.LIGHT:
        # Light theme: GREEN + GRAY (not pale gold!)
        generator._color_pool = [NVIDIA_GREEN, NVIDIA_GRAY] + generator._color_pool[2:]
    else:
        # Dark theme: GREEN + GOLD (gold looks good on dark)
        generator._color_pool = [NVIDIA_GREEN, NVIDIA_GOLD] + generator._color_pool[2:]

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


def create_matplotlib_fixed_colors(df, theme_name):
    """Create matplotlib with PROPER colors: Light=GREEN+GRAY, Dark=GREEN+GOLD."""

    # FIXED: Light theme uses GRAY, dark theme uses GOLD
    if theme_name == 'light':
        colors = [NVIDIA_GREEN, NVIDIA_GRAY]
        bg_color = '#FFFFFF'
        paper_color = '#FFFFFF'
        text_color = '#0a0a0a'
        grid_color = '#E5E5E5'
    else:
        colors = [NVIDIA_GREEN, NVIDIA_GOLD]
        bg_color = '#1a1a1a'
        paper_color = '#252525'
        text_color = '#E0E0E0'
        grid_color = '#333333'

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8))
    fig.patch.set_facecolor(paper_color)
    ax.set_facecolor(bg_color)

    # Get unique models and reverse order to match plotly legend
    models = df['model'].unique()[::-1]  # Reverse order!

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

        # Add point labels (concurrency values) - BIGGER font
        for _, row in model_data.iterrows():
            ax.text(
                row['time_to_first_token'],
                row['request_throughput'],
                str(row['concurrency']),
                fontsize=13,
                fontweight='bold',
                color=text_color,
                ha='center',
                va='bottom',
                zorder=4,
            )

    # Title and labels (match plotly styling with BIGGER fonts)
    ax.set_title('TTFT vs Throughput', fontsize=24, fontweight='bold',
                 color=text_color, pad=20, loc='left')
    ax.set_xlabel('Time to First Token P50 (ms)', fontsize=16, color=text_color, labelpad=10)
    ax.set_ylabel('Request Throughput AVG (requests/sec)', fontsize=16, color=text_color, labelpad=10)

    # Grid styling (match plotly)
    ax.grid(True, color=grid_color, linestyle='-', linewidth=0.8, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    # Spine styling
    for spine in ax.spines.values():
        spine.set_color(grid_color)
        spine.set_linewidth(1)

    # Tick styling (BIGGER tick labels)
    ax.tick_params(colors=text_color, labelsize=14, width=1, length=6)

    # Legend styling (match plotly - bottom right, BIGGER font)
    legend = ax.legend(
        loc='lower right',
        frameon=True,
        fontsize=14,
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
    print("NVIDIA BRANDED COMPARISON - FIXED COLOR SCHEME")
    print("="*70)
    print("\n🎨 COLOR CHOICES:")
    print(f"  Light theme: GREEN ({NVIDIA_GREEN}) + GRAY ({NVIDIA_GRAY})")
    print(f"  Dark theme:  GREEN ({NVIDIA_GREEN}) + GOLD ({NVIDIA_GOLD})")
    print("\n  ✅ Gold only on dark backgrounds (where it's visible)")
    print("  ✅ Gray on light backgrounds (actually readable)")

    df = generate_realistic_data()

    # ========== PLOTLY ==========
    print("\n" + "="*70)
    print("PLOTLY (with fixed colors)")
    print("="*70)

    print("  1. Light theme (GREEN + GRAY)...")
    fig = create_plotly_fixed_colors(df, PlotTheme.LIGHT)
    fig.write_image('/tmp/plotly_fixed_light.png', width=1600, height=800, scale=1.5)
    print("     ✅ /tmp/plotly_fixed_light.png")

    print("  2. Dark theme (GREEN + GOLD)...")
    fig = create_plotly_fixed_colors(df, PlotTheme.DARK)
    fig.write_image('/tmp/plotly_fixed_dark.png', width=1600, height=800, scale=1.5)
    print("     ✅ /tmp/plotly_fixed_dark.png")

    # ========== MATPLOTLIB ==========
    print("\n" + "="*70)
    print("MATPLOTLIB (mimicking plotly with fixed colors)")
    print("="*70)

    print("  1. Light theme (GREEN + GRAY)...")
    fig = create_matplotlib_fixed_colors(df, 'light')
    fig.savefig('/tmp/matplotlib_fixed_light.png', dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("     ✅ /tmp/matplotlib_fixed_light.png")

    print("  2. Dark theme (GREEN + GOLD)...")
    fig = create_matplotlib_fixed_colors(df, 'dark')
    fig.savefig('/tmp/matplotlib_fixed_dark.png', dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("     ✅ /tmp/matplotlib_fixed_dark.png")

    print("\n" + "="*70)
    print("✅ COMPARISON COMPLETE - NOW WITH PROPER COLORS!")
    print("="*70)
    print("\nFiles:")
    print("  PLOTLY:     /tmp/plotly_fixed_*.png     (1.1GB container)")
    print("  MATPLOTLIB: /tmp/matplotlib_fixed_*.png (439MB container)")
    print("\n🔥 These should look ACTUALLY GOOD now!")


if __name__ == "__main__":
    main()

