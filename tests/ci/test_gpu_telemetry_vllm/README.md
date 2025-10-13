<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# GPU Telemetry Test - vLLM Pathway

End-to-end test for GPU telemetry collection with vLLM backend and custom DCGM exporter setup.

## Overview

This test validates that AIPerf can successfully collect GPU metrics (power usage, utilization, memory, temperature, etc.) when using vLLM as the inference backend with a separately configured DCGM exporter.

## Test Components

- **`setup_test.sh`**: CI wrapper that installs aiperf and runs the test
- **`run_test.sh`**: Main test orchestration script
- **`docker-compose.yml`**: Infrastructure definition (DCGM exporter + vLLM server)
- **`custom_gpu_metrics.csv`**: DCGM metrics configuration
- **`../gpu_telemetry_common/verify_results.py`**: Shared validation script

## Test Flow

1. **Setup**: Install aiperf in virtual environment
2. **Infrastructure**: Start DCGM exporter (port 9401) and vLLM server (port 8000)
3. **Health Checks**: Wait for DCGM metrics and vLLM API readiness
4. **Benchmark**: Run AIPerf with `--gpu-telemetry` flag (20 requests, Qwen/Qwen3-0.6B model)
5. **Validation**: Verify GPU metrics were collected correctly
6. **Cleanup**: Stop containers and clean up resources

## Prerequisites

- NVIDIA GPU with CUDA support
- Docker installed and configured for GPU access
- Python 3.10+ available

## Running Locally

```bash
cd tests/ci/test_gpu_telemetry_vllm
./setup_test.sh
```

**Expected runtime**: ~5-10 minutes (depending on GPU and model download time)

## GitLab CI Integration

Add to `.gitlab-ci.yml`:

```yaml
test_gpu_telemetry_vllm:
  extends: .test_template
```

The `.test_template` automatically calls `setup_test.sh`.

## Test Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Model | `Qwen/Qwen3-0.6B` | Small, fast model for quick tests |
| Request Count | `20` | Number of requests to send |
| Concurrency | `4` | Concurrent requests |
| DCGM Port | `9401` | DCGM exporter port |
| vLLM Port | `8000` | vLLM server port |

## Debugging

Check container logs if test fails:
```bash
docker logs dcgm-exporter
docker logs vllm-server
```

Verify DCGM metrics are available:
```bash
curl localhost:9401/metrics | grep DCGM_FI_DEV_GPU_UTIL
```

Check vLLM server health:
```bash
curl localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"test"}],"max_tokens":1}'
```

## Related

- [Dynamo Pathway Test](../test_gpu_telemetry_dynamo/README.md)
- [GPU Telemetry Tutorial](../../docs/tutorials/gpu-telemetry.md)
