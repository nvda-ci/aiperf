#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Fixed comparison with REALISTIC data that shows what the plots actually look like.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from aiperf.plot.core.plot_generator import PlotGenerator
from aiperf.plot.constants import PlotTheme


def generate_realistic_data():
    """Generate realistic benchmark data that looks like actual AIPerf results."""
    
    # Realistic concurrency values
    concurrencies = [1, 2, 4, 8, 16, 32]
    
    data = []
    
    # Model 1: Meta-Llama-3-8B-Instruct (higher latency, lower throughput)
    for c in concurrencies:
        ttft_p50 = 30 + c * 3  # Latency increases with concurrency
        throughput = c * 2.1 if c < 16 else c * 1.8  # Throughput gains slow down
        
        data.append({
            'concurrency': c,
            'time_to_first_token': ttft_p50,
            'request_throughput': throughput,
            'model': 'meta-llama/Meta-Llama-3-8B-Instruct',
        })
    
    # Model 2: Qwen/Qwen3-0.6B (lower latency, higher throughput)
    for c in concurrencies:
        ttft_p50 = 20 + c * 2  # Better latency
        throughput = c * 3.2 if c < 16 else c * 2.8  # Better throughput
        
        data.append({
            'concurrency': c,
            'time_to_first_token': ttft_p50,
            'request_throughput': throughput,
            'model': 'Qwen/Qwen3-0.6B',
        })
    
    return pd.DataFrame(data)


def main():
    print("="*70)
    print("GENERATING FIXED COMPARISON WITH REALISTIC DATA")
    print("="*70)
    
    df = generate_realistic_data()
    
    # PLOTLY - Light Theme
    print("\n1. Generating Plotly Light Theme...")
    generator_light = PlotGenerator(theme=PlotTheme.LIGHT, color_pool_size=10)
    fig = generator_light.create_scatter_line_plot(
        df=df,
        x_metric='time_to_first_token',
        y_metric='request_throughput',
        label_by='concurrency',
        group_by='model',
        title='TTFT vs Throughput',
        x_label='Time to First Token P50 (ms)',
        y_label='Request Throughput AVG (requests/sec)',
    )
    fig.write_image('/tmp/realistic_plotly_light.png', width=1600, height=800, scale=1.5)
    print("   ✅ /tmp/realistic_plotly_light.png")
    
    # PLOTLY - Dark Theme
    print("\n2. Generating Plotly Dark Theme...")
    generator_dark = PlotGenerator(theme=PlotTheme.DARK, color_pool_size=10)
    fig = generator_dark.create_scatter_line_plot(
        df=df,
        x_metric='time_to_first_token',
        y_metric='request_throughput',
        label_by='concurrency',
        group_by='model',
        title='TTFT vs Throughput',
        x_label='Time to First Token P50 (ms)',
        y_label='Request Throughput AVG (requests/sec)',
    )
    fig.write_image('/tmp/realistic_plotly_dark.png', width=1600, height=800, scale=1.5)
    print("   ✅ /tmp/realistic_plotly_dark.png")
    
    print("\n" + "="*70)
    print("DONE - These should look EXACTLY like your docs examples")
    print("="*70)
    print("\nCompare with your actual plots:")
    print("  verify-cli-plots/docs/diagrams/plot_examples/multi_run/ttft_vs_throughput.png")
    print("\nNew realistic plots:")
    print("  /tmp/realistic_plotly_light.png")
    print("  /tmp/realistic_plotly_dark.png")


if __name__ == "__main__":
    main()

