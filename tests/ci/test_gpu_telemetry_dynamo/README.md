<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# GPU Telemetry Test - Dynamo Pathway

End-to-end test for GPU telemetry collection with Dynamo backend using its built-in DCGM exporter.

## Overview

This test validates that AIPerf can successfully collect GPU metrics (power usage, utilization, memory, temperature, etc.) when using Dynamo as the inference backend. Dynamo includes DCGM out-of-the-box on port 9401, so no separate DCGM setup is required.

## Test Components

- **`setup_test.sh`**: CI wrapper that installs aiperf and runs the test
- **`run_test.sh`**: Main test orchestration script
- **`../gpu_telemetry_common/verify_results.py`**: Shared validation script

## Test Flow

1. **Setup**: Install aiperf in virtual environment
2. **Infrastructure**: Download and start Dynamo services + launch Dynamo server
3. **Health Checks**: Wait for Dynamo API (port 8080) and DCGM metrics (port 9401) readiness
4. **Benchmark**: Run AIPerf with `--gpu-telemetry` flag (20 requests, Qwen/Qwen3-0.6B model)
5. **Validation**: Verify GPU metrics were collected correctly
6. **Cleanup**: Stop containers and clean up resources

## Prerequisites

- NVIDIA GPU with CUDA support
- Docker installed and configured for GPU access
- Python 3.10+ available

## Running Locally

```bash
cd tests/ci/test_gpu_telemetry_dynamo
./setup_test.sh
```

**Expected runtime**: ~10-20 minutes (first run may take longer due to image downloads)

### Custom Dynamo Image

To test with a different Dynamo version:

```bash
export DYNAMO_PREBUILT_IMAGE_TAG="nvcr.io/nvidia/ai-dynamo/vllm-runtime:X.Y.Z"
./setup_test.sh
```

## GitLab CI Integration

Add to `.gitlab-ci.yml`:

```yaml
test_gpu_telemetry_dynamo:
  extends: .test_template
```

The `.test_template` automatically calls `setup_test.sh`.

## Test Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Model | `Qwen/Qwen3-0.6B` | Small, fast model for quick tests |
| Request Count | `20` | Number of requests to send |
| Concurrency | `4` | Concurrent requests |
| DCGM Port | `9401` | Built-in DCGM exporter port |
| Dynamo Port | `8080` | Dynamo API port |

## Debugging

Check Dynamo logs if test fails:
```bash
docker logs dynamo-server
```

Verify DCGM metrics are available:
```bash
curl localhost:9401/metrics | grep DCGM_FI_DEV_GPU_UTIL
```

Check Dynamo server health:
```bash
curl localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"test"}],"max_completion_tokens":1}'
```

## Related

- [vLLM Pathway Test](../test_gpu_telemetry_vllm/README.md)
- [GPU Telemetry Tutorial](../../docs/tutorials/gpu-telemetry.md)
