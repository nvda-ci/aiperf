#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# GPU Telemetry E2E Test - Dynamo Pathway
# Tests GPU telemetry collection with Dynamo backend (built-in DCGM)

set -e  # Exit on any error
set -u  # Exit on undefined variable

# Configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly MODEL="Qwen/Qwen3-0.6B"
readonly DYNAMO_PORT=8080
readonly DCGM_PORT=9401
readonly OUTPUT_DIR="aiperf_output"
readonly DYNAMO_PREBUILT_IMAGE_TAG="${DYNAMO_PREBUILT_IMAGE_TAG:-nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.5.0}"

# Color output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $*"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*"
}

cleanup() {
    log_info "Cleaning up containers..."
    cd "$SCRIPT_DIR"

    # Stop Dynamo container if running
    docker ps -a | grep -q dynamo-server && docker stop dynamo-server 2>/dev/null || true
    docker ps -a | grep -q dynamo-server && docker rm dynamo-server 2>/dev/null || true

    # Stop docker-compose services
    if [ -f "docker-compose.yml" ]; then
        docker compose -f docker-compose.yml down -v 2>/dev/null || true
    fi

    log_info "Cleanup complete"
}

# Register cleanup on exit
trap cleanup EXIT

main() {
    log_info "=========================================="
    log_info "GPU Telemetry E2E Test - Dynamo Pathway"
    log_info "=========================================="
    log_info "Model: ${MODEL}"
    log_info "Dynamo Port: ${DYNAMO_PORT}"
    log_info "DCGM Port: ${DCGM_PORT}"
    log_info "Dynamo Image: ${DYNAMO_PREBUILT_IMAGE_TAG}"
    log_info ""

    # Change to script directory
    cd "$SCRIPT_DIR"

    # Clean up any existing containers
    log_info "Stopping any existing containers..."
    cleanup

    # Remove old output directory
    if [ -d "$OUTPUT_DIR" ]; then
        log_info "Removing old output directory..."
        rm -rf "$OUTPUT_DIR"
    fi

    # Download Dynamo container
    log_info "Pulling Dynamo container image..."
    docker pull "${DYNAMO_PREBUILT_IMAGE_TAG}"

    # Get Dynamo repository tag
    log_info "Retrieving Dynamo version..."
    DYNAMO_REPO_TAG=$(docker run --rm --entrypoint "" "${DYNAMO_PREBUILT_IMAGE_TAG}" \
        cat /workspace/version.txt | cut -d'+' -f2)
    log_info "Dynamo version: ${DYNAMO_REPO_TAG}"

    # Download docker-compose.yml
    log_info "Downloading Dynamo docker-compose.yml..."
    curl -fsSL -o docker-compose.yml \
        "https://raw.githubusercontent.com/ai-dynamo/dynamo/${DYNAMO_REPO_TAG}/deploy/docker-compose.yml"

    # Start required services
    log_info "Starting Dynamo required services..."
    docker compose -f docker-compose.yml up -d

    # Launch Dynamo in the background
    log_info "Starting Dynamo server with model ${MODEL}..."
    docker run \
        -d \
        --rm \
        --name dynamo-server \
        --gpus all \
        --network host \
        "${DYNAMO_PREBUILT_IMAGE_TAG}" \
        /bin/bash -c "python3 -m dynamo.frontend & python3 -m dynamo.vllm --model ${MODEL} --enforce-eager --no-enable-prefix-caching"

    # Wait for Dynamo API to be ready
    if ! timeout 900 bash -c "while [ \"\$(curl -s -o /dev/null -w '%{http_code}' \
        localhost:${DYNAMO_PORT}/v1/chat/completions \
        -H 'Content-Type: application/json' \
        -d '{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"a\"}],\"max_completion_tokens\":1}')\" != '200' ]; do \
        sleep 2; \
    done"; then
        log_error "Dynamo not ready after 15 minutes"
        docker logs dynamo-server 2>&1 | tail -50
        exit 1
    fi
    log_info "✓ Dynamo API is ready"

    # Wait for DCGM Exporter to be ready
    if ! timeout 120 bash -c "while true; do \
        STATUS=\$(curl -s -o /dev/null -w '%{http_code}' localhost:${DCGM_PORT}/metrics); \
        if [ \"\$STATUS\" = '200' ]; then \
            if curl -s localhost:${DCGM_PORT}/metrics | grep -q 'DCGM_FI_DEV_GPU_UTIL'; then \
                break; \
            fi; \
        fi; \
        sleep 5; \
    done"; then
        log_error "GPU utilization metrics not found after 2 minutes"
        exit 1
    fi
    log_info "✓ DCGM GPU metrics are available"

    # Run AIPerf benchmark
    log_info "Running AIPerf benchmark with GPU telemetry..."
    aiperf profile \
        --model "${MODEL}" \
        --endpoint-type chat \
        --endpoint /v1/chat/completions \
        --streaming \
        --url "localhost:${DYNAMO_PORT}" \
        --synthetic-input-tokens-mean 100 \
        --synthetic-input-tokens-stddev 0 \
        --output-tokens-mean 200 \
        --output-tokens-stddev 0 \
        --extra-inputs min_tokens:200 \
        --extra-inputs ignore_eos:true \
        --concurrency 4 \
        --request-count 20 \
        --warmup-request-count 1 \
        --conversation-num 4 \
        --random-seed 100 \
        --gpu-telemetry

    log_info "✓ AIPerf benchmark completed"

    # Verify results
    log_info "Verifying GPU telemetry results..."
    if python3 ${AIPERF_SOURCE_DIR}/tests/ci/gpu_telemetry_common/verify_results.py --base-dir "${SCRIPT_DIR}"; then
        log_info "=========================================="
        log_info "✓ DYNAMO PATHWAY TEST PASSED"
        log_info "=========================================="
        exit 0
    else
        log_error "=========================================="
        log_error "✗ DYNAMO PATHWAY TEST FAILED"
        log_error "=========================================="
        exit 1
    fi
}

main "$@"
