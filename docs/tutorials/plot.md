<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Visualization and Plotting with AIPerf

Generate PNG visualizations from AIPerf profiling data with automatic mode detection (single-run analysis or multi-run comparison), NVIDIA brand styling, and support for multi-run comparisons and single-run analysis.

## Overview

The `aiperf plot` command generates static PNG visualizations from your profiling results. It automatically detects whether to show multi-run comparison plots or single-run time series analysis based on your directory structure, making it easy to visualize performance trends without manual configuration.

**Key Features:**
- **Automatic mode detection**: Compares multiple runs or analyzes single runs based on directory structure
- **GPU telemetry integration**: Visualize power, utilization, memory, and temperature metrics
- **Timeslice support**: View performance evolution across time windows

## Basic Usage

### Generate Multi-Run Comparison Plots

```bash
# Generate plots from all runs in the sweep directory
aiperf plot artifacts/sweep_qwen

# Or specify the directory explicitly
aiperf plot artifacts/sweep_qwen --output artifacts/sweep_qwen/plot_export
```

This generates PNG files in `artifacts/sweep_qwen/plot_export/` with all default multi-run comparison plots.

### Generate Single-Run Analysis Plots

```bash
# Analyze a single profiling run
aiperf plot artifacts/sweep_qwen/Qwen3-0.6B-concurrency8
```

This generates PNG files in `artifacts/sweep_qwen/Qwen3-0.6B-concurrency8/plot_export/` with time series plots showing performance over the duration of that specific run.

## Visualization Modes

The plot command automatically detects the visualization mode based on directory structure:

### Multi-Run Comparison Mode

**Detected when:**
- Directory contains multiple run subdirectories
- Multiple paths are specified as arguments

**Example directory structures:**
```
artifacts/sweep_qwen/          # Contains multiple runs
├── Qwen3-0.6B-concurrency1/
├── Qwen3-0.6B-concurrency2/
└── Qwen3-0.6B-concurrency4/
```

**Generated plots (5 default):**
1. **Pareto Curve** - Throughput vs latency trade-offs with Pareto frontier highlighting optimal configurations
2. **TTFT vs Throughput** - Time to first token vs request throughput across concurrency levels
3. **Output Token Throughput per User** - Per-user output token throughput at different concurrency levels
4. **Token Throughput per GPU vs Latency** - GPU efficiency vs latency (when GPU telemetry available)
5. **Token Throughput per GPU vs Interactivity** - GPU efficiency vs TTFT (when GPU telemetry available)

#### Example Multi-Run Visualizations

![TTFT vs Throughput](../diagrams/plot_examples/multi_run/ttft_vs_throughput.png)

The TTFT vs Throughput plot shows how time to first token varies with request throughput across different concurrency levels, helping identify configurations that balance responsiveness with system load.

![Pareto Curve: Throughput per GPU vs Latency](../diagrams/plot_examples/multi_run/pareto_curve_throughput_per_gpu_vs_latency.png)

The Pareto curve highlights optimal configurations that maximize GPU efficiency while minimizing latency. Points on the Pareto frontier represent the best trade-offs between these metrics.

![Pareto Curve: Throughput per GPU vs Interactivity](../diagrams/plot_examples/multi_run/pareto_curve_throughput_per_gpu_vs_interactivity.png)

This Pareto curve shows the trade-off between GPU efficiency (tokens/sec/GPU) and interactivity (TTFT), helping identify configurations that maximize GPU utilization while maintaining acceptable first-token latency.

### Single-Run Analysis Mode

**Detected when:**
- Directory contains `profile_export.jsonl` directly
- Path points to a single profiling run

**Example directory structure:**
```
artifacts/single_run/          # Single run directory
└── profile_export.jsonl
```

**Generated plots (4+ default):**
1. **TTFT Over Time** - Scatter plot of time to first token for each request
2. **Inter-Token Latency Over Time** - Scatter plot of ITL for each request
3. **Request Latency Over Time** - Area chart showing end-to-end latency progression
4. **Dispersed Throughput Over Time** - Event-based throughput showing continuous token generation rate

**Additional plots (when data available):**
- **Timeslice plots**: TTFT, ITL, throughput, and latency metrics across time windows (when `--slice-duration` was used)
- **GPU telemetry plots**: GPU utilization and memory usage over time (when `--gpu-telemetry` was used)

#### Example Single-Run Time Series Visualizations

![TTFT Over Time](../diagrams/plot_examples/single_run/time_series/ttft_over_time.png)

The TTFT Over Time scatter plot shows the time to first token for each request throughout the benchmark run, helping identify patterns in prefill latency and potential warm-up or degradation effects.

![Inter-Token Latency Over Time](../diagrams/plot_examples/single_run/time_series/itl_over_time.png)

The ITL Over Time scatter plot displays inter-token latency for each request, revealing generation performance consistency and identifying outliers or performance variations over the run duration.

![Request Latency Over Time](../diagrams/plot_examples/single_run/time_series/latency_over_time.png)

The Request Latency Over Time area chart shows end-to-end latency progression throughout the run, providing a holistic view of system performance including both prefill and generation phases.

### Understanding Dispersed Throughput Visualization

The **Dispersed Throughput Over Time** plot uses an event-based approach to accurately represent output token generation rates. Unlike traditional binning methods that count all tokens at specific events (like request completion), this visualization distributes tokens evenly across the time period they were actually generated.

#### How It Works

For each request, tokens are distributed as follows:

1. **Prefill Phase** (`request_start` → `TTFT`): No output tokens generated (0 tok/sec)
2. **Generation Phase** (`TTFT` → `request_end`): Tokens evenly distributed at constant rate
   - Token rate = `output_tokens / (request_end - TTFT)`

**Example Request Timeline:**
```
Request Lifecycle:
├─────────────────┬──────────────────────────────────────────────┐
│  Prefill Phase  │         Token Generation Phase               │
│                 │                                               │
request_start     TTFT                                      request_end
    │              │                                              │
    0 tokens       │         16 tokens @ 181 tok/sec             │
```

At any point in time, the plot shows the **sum of all active token generation rates** from concurrent requests. Throughput only changes at discrete events:
- **Increases** when a request starts generating (reaches TTFT)
- **Decreases** when a request finishes

This creates a step function that accurately represents the instantaneous throughput at every moment.

#### Why Dispersed Throughput?

**Problem with traditional binning:**
```
Traditional Approach (1-second bins):
├─────────┬─────────┬─────────┬─────────┐
    0s        1s        2s        3s
    │         │         │         │
    ▼         ▼         ▼         ▼
  [spike]   [spike]   [zero]   [spike]
```
- All tokens counted when request ends (creates artificial spikes)
- Bin size is arbitrary (1s? 100ms? How to choose?)
- Misses true generation patterns between bins
- Hard to correlate with GPU metrics

**Dispersed approach:**
```
Event-Based (Accurate):
    ╔════════╗
    ║        ║
╔═══╝        ╚════╗
║                 ║
║                 ╚═══
```
- Smooth, continuous representation
- No arbitrary bin size decisions
- Shows actual generation rate over time
- Accurate correlation with server metrics (GPU utilization, queue depth, etc.)

#### Performance Benefits

- **O(k log k)** complexity where k = 2×number_of_requests
- Much faster than binning approach O(n×m) where n = bins, m = requests
- Exact state transitions (no approximation errors)

#### Use Cases

1. **Identifying bottlenecks**: Flat throughput indicates server saturation
2. **Correlating with GPU metrics**: Accurate alignment with GPU utilization changes
3. **Analyzing ramp-up**: See how throughput increases as requests enter generation
4. **Detecting anomalies**: Sudden drops indicate request completions or failures

#### Example Visualization

![Dispersed Throughput Over Time](../diagrams/plot_examples/single_run/dispersed_throughput_over_time.png)

The plot shows:
- **Step function**: Throughput changes only at request events (TTFT or completion)
- **Filled area**: Visual emphasis on throughput magnitude
- **NVIDIA Green**: Consistent branding with other AIPerf visualizations
- **Time-aligned**: X-axis matches other time-series plots for easy comparison

> [!TIP]
> Compare the dispersed throughput plot with GPU utilization to identify:
> - **GPU idle time**: Low throughput with low GPU utilization suggests insufficient load
> - **GPU saturation**: Flat throughput with high GPU utilization indicates capacity limit
> - **Efficiency issues**: High GPU utilization with low throughput suggests inefficient batching

## Command Options

### Basic Options

```bash
# Default behavior: plots from ./artifacts directory
aiperf plot

# Specify one or more paths
aiperf plot path/to/results

# Compare specific runs from different directories
aiperf plot dir1/run1 dir2/run2 dir3/run3

# Custom output directory
aiperf plot path/to/results --output path/to/custom/output

# Dark theme for presentations
aiperf plot path/to/results --theme dark
```

### Output Directory Logic

The output directory follows this logic:
1. If `--output` is specified, use that path
2. Otherwise, use `{first_input_path}/plot_export/`
3. Default first input path is `./artifacts` if no paths specified

**Examples:**
```bash
# Outputs to: ./artifacts/plot_export/
aiperf plot

# Outputs to: sweep_results/plot_export/
aiperf plot sweep_results

# Outputs to: /custom/location/
aiperf plot sweep_results --output /custom/location
```

## Theme Options

Choose between light and dark themes for your plots:

```bash
# Light theme (default)
aiperf plot

aiperf plot --theme light

# Dark theme
aiperf plot --theme dark
```

### Dark Theme Examples

The dark theme uses a dark background optimized for presentations and low-light environments while maintaining NVIDIA brand colors and readability.

#### Multi-Run Dark Theme

![TTFT vs Throughput (Dark)](../diagrams/plot_examples/multi_run/theme_dark_mode/ttft_vs_throughput.png)

![Pareto Curve: Throughput per GPU vs Latency (Dark)](../diagrams/plot_examples/multi_run/theme_dark_mode/pareto_curve_throughput_per_gpu_vs_latency.png)

![Pareto Curve: Throughput per GPU vs Interactivity (Dark)](../diagrams/plot_examples/multi_run/theme_dark_mode/pareto_curve_throughput_per_gpu_vs_interactivity.png)

#### Single-Run Dark Theme

![GPU Utilization and Throughput Over Time (Dark)](../diagrams/plot_examples/single_run/time_series/theme_dark_mode/gpu_utilization_and_throughput_over_time.png)

![Inter-Token Latency Over Time (Dark)](../diagrams/plot_examples/single_run/time_series/theme_dark_mode/itl_over_time.png)

![ITL Across Timeslices (Dark)](../diagrams/plot_examples/single_run/time_series/theme_dark_mode/timeslices_itl.png)

## Integration with GPU Telemetry

When you collect GPU telemetry during profiling, the plot command automatically includes GPU metrics in visualizations.

### Collecting GPU Telemetry

```bash
# Enable GPU telemetry during profiling
aiperf profile \
  --model Qwen/Qwen3-0.6B \
  --url localhost:8000 \
  --concurrency 8 \
  --request-count 64 \
  --gpu-telemetry
```

### Multi-Run GPU Plots

When GPU telemetry is available across multiple runs, you get additional comparison plots:

- **Token Throughput per GPU vs Latency**: Shows GPU efficiency (tokens/sec/GPU) against request latency
- **Token Throughput per GPU vs Interactivity**: Shows GPU efficiency against TTFT

These plots help identify configurations that maximize GPU utilization while maintaining acceptable latency.

### Single-Run GPU Plots

For single runs with GPU telemetry, time series plots show:

- **GPU Utilization Over Time**: Percentage of GPU compute resources used
- **GPU Memory Usage Over Time**: Memory consumption in GB

These help diagnose GPU resource usage patterns, bottlenecks, and idle periods.

#### Example GPU Telemetry Visualization

![GPU Utilization and Throughput Over Time](../diagrams/plot_examples/single_run/time_series/gpu_utilization_and_throughput_over_time.png)

This dual-axis plot correlates GPU utilization (percentage) with request throughput over time, enabling identification of GPU efficiency patterns, saturation points, and opportunities for optimization.

> [!TIP]
> For detailed GPU telemetry setup instructions, see the [GPU Telemetry Tutorial](gpu-telemetry.md).

## Integration with Timeslices

Timeslice metrics enable performance analysis across sequential time windows during a benchmark run. See the [Timeslices Tutorial](timeslices.md).

The plot command automatically generates timeslice visualizations when timeslice data is available.

For example:
- **TTFT Across Timeslices**: Shows how time to first token evolves over time windows
- **ITL Across Timeslices**: Shows inter-token latency trends across slices
- **Throughput Across Timeslices**: Shows request throughput evolution
- **Latency Across Timeslices**: Shows end-to-end latency trends

These plots help identify:
- **Warm-up effects**: Higher latency in early slices
- **Performance degradation**: Increasing latency in later slices
- **Steady-state behavior**: Stable metrics after initial warm-up

#### Example Timeslice Visualizations

![TTFT Across Timeslices](../diagrams/plot_examples/single_run/timeslices/timeslices_ttft.png)

![ITL Across Timeslices](../diagrams/plot_examples/single_run/timeslices/timeslices_itl.png)

![Throughput Across Timeslices](../diagrams/plot_examples/single_run/timeslices/timeslices_throughput_warning.png)

> [!NOTE]
> **Throughput Warning**: The throughput plot displays a warning when requests have varying input sequence lengths (ISL) or output sequence lengths (OSL). In these cases, requests per second (req/sec) throughput may not accurately represent workload capacity since different requests have different computational costs. For more accurate capacity measurement with non-uniform workloads, consider using token throughput metrics instead.

![Latency Across Timeslices](../diagrams/plot_examples/single_run/timeslices/timeslices_latency.png)


## Output Files

The plot command generates the following files in the output directory:

```
plot_export/
├── pareto_curve_latency_vs_throughput.png
├── ttft_vs_throughput.png
├── output_token_throughput_per_user_vs_concurrency.png
├── dispersed_throughput_over_time.png (for single-run analysis)
├── token_throughput_per_gpu_vs_latency.png (if GPU telemetry available)
├── token_throughput_per_gpu_vs_ttft.png (if GPU telemetry available)
└── ... (additional plots based on mode and available data)
```



## Best Practices

> [!TIP]
> **Consistent Configurations**: When comparing runs, keep all parameters identical except the one you're testing (e.g., only vary concurrency). This ensures plots show the impact of that specific parameter.
> Future features in interactive mode will allow pop-ups to show specific configurations of plotted runs.

> [!TIP]
> **Include Warmup**: Use `--warmup-request-count` to ensure the server reaches steady state before measurement. This reduces noise in your visualizations.

> [!WARNING]
> **Directory Structure**: The plot command relies on consistent directory naming. Ensure all runs you want to compare are in subdirectories of a common parent directory.

> [!NOTE]
> **GPU Metrics**: GPU telemetry plots only appear when telemetry data is available. Make sure DCGM is running and accessible during profiling. See [GPU Telemetry Tutorial](gpu-telemetry.md).

## Troubleshooting

### No Plots Generated

**Problem**: Running `aiperf plot` but no PNG files appear.

**Solutions**:
- Verify the input directory contains valid profiling data (`profile_export.jsonl` files)
- Check that the output directory is writable
- Look for error messages in the console output

### Missing GPU Plots

**Problem**: Expected GPU telemetry plots but they don't appear.

**Solutions**:
- Verify GPU telemetry was collected during profiling (check `gpu_telemetry_export.jsonl` for telemetry data)
- Ensure DCGM exporter was running and accessible during profiling
- Confirm telemetry data is present in the profile exports

### Incorrect Mode Detection

**Problem**: Multi-run data showing single-run plots or vice versa.

**Solutions**:
- Check directory structure matches expected format:
  - Multi-run: parent directory with multiple run subdirectories
  - Single-run: directory with `profile_export.jsonl` directly inside
- Ensure all run directories contain valid `profile_export.jsonl` files

## Related Documentation

- [Working with Profile Exports](working-with-profile-exports.md) - Understanding profiling data format
- [GPU Telemetry](gpu-telemetry.md) - Collecting GPU metrics during profiling
- [Timeslices](timeslices.md) - Time-windowed performance analysis
- [Request Rate and Concurrency](request-rate-concurrency.md) - Load generation strategies for sweeps
