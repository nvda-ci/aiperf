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

### Setting Up a Benchmark Sweep

To demonstrate the plot command, let's run a concurrency sweep that generates multiple profiling runs:

```bash
# Start vLLM server
docker pull vllm/vllm-openai:latest
docker run -d --name vllm-server \
  --gpus all -p 8000:8000 \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen3-0.6B \
  --host 0.0.0.0 --port 8000
```

```bash
# Wait for server to be ready
timeout 900 bash -c 'while [ "$(curl -s -o /dev/null -w "%{http_code}" localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"Qwen/Qwen3-0.6B\",\"messages\":[{\"role\":\"user\",\"content\":\"test\"}],\"max_tokens\":1}")" != "200" ]; do sleep 2; done'
```

```bash
# Run concurrency sweep
for concurrency in 1 2 4 8 16 32; do
  aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --streaming \
    --url localhost:8000 \
    --concurrency $concurrency \
    --request-count 64 \
    --warmup-request-count 1 \
    --output-artifact-dir "artifacts/sweep_qwen/Qwen3-0.6B-concurrency${concurrency}"
done
```

This creates a directory structure like:
```
artifacts/sweep_qwen/
├── Qwen3-0.6B-concurrency1/
│   └── profile_export.jsonl
├── Qwen3-0.6B-concurrency2/
│   └── profile_export.jsonl
├── Qwen3-0.6B-concurrency4/
│   └── profile_export.jsonl
└── ...
```

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

### Single-Run Analysis Mode

**Detected when:**
- Directory contains `profile_export.jsonl` directly
- Path points to a single profiling run

**Example directory structure:**
```
artifacts/single_run/          # Single run directory
└── profile_export.jsonl
```

**Generated plots (3+ default):**
1. **TTFT Over Time** - Scatter plot of time to first token for each request
2. **Inter-Token Latency Over Time** - Scatter plot of ITL for each request
3. **Request Latency Over Time** - Area chart showing end-to-end latency progression

**Additional plots (when data available):**
- **Timeslice plots**: TTFT, ITL, throughput, and latency metrics across time windows (when `--slice-duration` was used)
- **GPU telemetry plots**: GPU utilization and memory usage over time (when `--gpu-telemetry` was used)

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
# Light theme (default) - best for documentation and reports
aiperf plot --theme light

# Dark theme - best for presentations and dark-mode displays
aiperf plot --theme dark
```

Both themes use NVIDIA brand colors and maintain consistent styling for professional visualizations.

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

> [!TIP]
> For detailed GPU telemetry setup instructions, see the [GPU Telemetry Tutorial](gpu-telemetry.md).

## Integration with Timeslices

Timeslice metrics enable performance analysis across sequential time windows during a benchmark run.

### Collecting Timeslice Data

```bash
# Run with timeslicing enabled
aiperf profile \
  --model Qwen/Qwen3-0.6B \
  --url localhost:8000 \
  --concurrency 8 \
  --benchmark-duration 60 \
  --slice-duration 10 \
  --request-count 64
```

### Timeslice Plots

The plot command automatically generates timeslice visualizations when timeslice data is available:

- **TTFT Across Timeslices**: Shows how time to first token evolves over time windows
- **ITL Across Timeslices**: Shows inter-token latency trends across slices
- **Throughput Across Timeslices**: Shows request throughput evolution
- **Latency Across Timeslices**: Shows end-to-end latency trends

These plots help identify:
- **Warm-up effects**: Higher latency in early slices
- **Performance degradation**: Increasing latency in later slices
- **Steady-state behavior**: Stable metrics after initial warm-up

> [!TIP]
> For detailed timeslice configuration, see the [Timeslices Tutorial](timeslices.md).

## Output Files

The plot command generates the following files in the output directory:

```
plot_export/
├── pareto_curve_latency_vs_throughput.png
├── ttft_vs_throughput.png
├── output_token_throughput_per_user_vs_concurrency.png
├── token_throughput_per_gpu_vs_latency.png (if GPU telemetry available)
├── token_throughput_per_gpu_vs_ttft.png (if GPU telemetry available)
└── ... (additional plots based on mode and available data)
```

Each PNG file is a high-resolution, publication-ready visualization with:
- NVIDIA brand colors and styling
- Clear axis labels with units
- Legend showing run configurations
- Grid lines for precise value reading

## Use Cases

### Parameter Sweep Analysis

Compare different configurations to find optimal settings:

```bash
# Run sweep across concurrency levels
for c in 1 2 4 8 16 32 64; do
  aiperf profile --concurrency $c --output-artifact-dir "sweep/c${c}" [other options...]
done

# Generate comparison plots
aiperf plot sweep
```

Use the Pareto curve to identify configurations that offer the best throughput/latency trade-off.

### Performance Debugging

Analyze a single run to identify anomalies:

```bash
# Profile with detailed tracking
aiperf profile --concurrency 16 --request-count 1000 --output-artifact-dir debug_run

# Generate time series plots
aiperf plot debug_run
```

Look for:
- Outliers in TTFT/ITL scatter plots
- Latency spikes in area charts
- GPU utilization gaps suggesting inefficiencies

### Model Comparison

Compare different models under identical conditions:

```bash
# Profile multiple models
for model in "Qwen/Qwen3-0.6B" "Qwen/Qwen3-1.8B" "Qwen/Qwen3-7B"; do
  model_slug=${model//\//_}
  aiperf profile --model $model --output-artifact-dir "comparison/${model_slug}" [other options...]
done

# Compare models
aiperf plot comparison
```

The plots will show performance differences across model sizes.

### Regression Testing

Track performance changes across inference server versions:

```bash
# Baseline
aiperf profile --output-artifact-dir baseline_v1.0 [options...]

# After upgrade
aiperf profile --output-artifact-dir new_v2.0 [options...]

# Compare
aiperf plot baseline_v1.0 new_v2.0 --output regression_comparison
```

## Best Practices

> [!TIP]
> **Consistent Configurations**: When comparing runs, keep all parameters identical except the one you're testing (e.g., only vary concurrency). This ensures plots show the impact of that specific parameter.

> [!TIP]
> **Include Warmup**: Use `--warmup-request-count` to ensure the server reaches steady state before measurement. This reduces noise in your visualizations.

> [!TIP]
> **Theme Selection**: Use `--theme light` for documentation and reports, `--theme dark` for presentations and screen sharing on dark-mode displays.

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
- Ensure you have sufficient disk space
- Look for error messages in the console output

### Missing GPU Plots

**Problem**: Expected GPU telemetry plots but they don't appear.

**Solutions**:
- Verify GPU telemetry was collected during profiling (check `profile_export.jsonl` for telemetry data)
- Ensure DCGM exporter was running and accessible during profiling
- Confirm telemetry data is present in the profile exports

### Incorrect Mode Detection

**Problem**: Multi-run data showing single-run plots or vice versa.

**Solutions**:
- Check directory structure matches expected format:
  - Multi-run: parent directory with multiple run subdirectories
  - Single-run: directory with `profile_export.jsonl` directly inside
- Ensure all run directories contain valid `profile_export.jsonl` files
- Try specifying paths explicitly: `aiperf plot run1 run2 run3`

## Related Documentation

- [Working with Profile Exports](working-with-profile-exports.md) - Understanding profiling data format
- [GPU Telemetry](gpu-telemetry.md) - Collecting GPU metrics during profiling
- [Timeslices](timeslices.md) - Time-windowed performance analysis
- [Request Rate and Concurrency](request-rate-concurrency.md) - Load generation strategies for sweeps
