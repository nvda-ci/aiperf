#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e  # Exit on any error
set -u  # Exit on undefined variable

echo "=========================================="
echo "Setting up environment for vLLM GPU telemetry test"
echo "=========================================="

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env
else
    echo "✓ uv is already installed"
fi

# Create and activate virtual environment
echo "Creating Python 3.10 virtual environment..."
uv venv --python 3.10

echo "Activating virtual environment..."
source .venv/bin/activate

# Install aiperf from source
echo "Installing aiperf from ${AIPERF_SOURCE_DIR}..."
uv pip install ${AIPERF_SOURCE_DIR}

echo "✓ Environment setup complete"
echo ""

# Run the actual test
echo "Starting vLLM pathway test..."
bash -x ${AIPERF_SOURCE_DIR}/tests/ci/test_gpu_telemetry_vllm/run_test.sh
