#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e  # Exit on any error
set -u  # Exit on undefined variable

echo "=========================================="
echo "Setting up environment for vLLM GPU telemetry test"
echo "=========================================="

# Build AIPerf Docker container
echo "Building AIPerf container..."
cd ${AIPERF_SOURCE_DIR}
docker build -t aiperf:test .

echo "✓ AIPerf Docker image built successfully"

# Start the container with bash entrypoint to keep it running
CONTAINER_NAME="aiperf-gpu-telemetry-vllm-$$"
echo "Starting AIPerf container: ${CONTAINER_NAME}..."
docker run -d --name ${CONTAINER_NAME} --network host --entrypoint bash aiperf:test -c 'tail -f /dev/null'

# Export container name for the run script
export AIPERF_CONTAINER_NAME=${CONTAINER_NAME}

# Verify aiperf works in the container
echo "Verifying aiperf in container..."
docker exec ${CONTAINER_NAME} bash -c 'source /opt/aiperf/venv/bin/activate && aiperf --version'

echo "✓ Environment setup complete"
echo ""

# Run the actual test
echo "Starting vLLM pathway test..."
bash -x ${AIPERF_SOURCE_DIR}/tests/ci/test_gpu_telemetry_vllm/run_test.sh

# Cleanup container
echo "Cleaning up AIPerf container..."
docker stop ${CONTAINER_NAME} || true
docker rm ${CONTAINER_NAME} || true
