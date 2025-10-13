#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# GPU Telemetry E2E Test - vLLM Pathway
# Tests GPU telemetry collection with vLLM backend and separate DCGM exporter

set -e  # Exit on any error
set -u  # Exit on undefined variable

# Configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly MODEL="Qwen/Qwen3-0.6B"
readonly DCGM_PORT=9401
readonly VLLM_PORT=8000
readonly OUTPUT_DIR="aiperf_output"

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
    docker compose -f docker-compose.yml down -v 2>/dev/null || true
    log_info "Cleanup complete"
}

# Register cleanup on exit
trap cleanup EXIT

main() {
    log_info "=========================================="
    log_info "GPU Telemetry E2E Test - vLLM Pathway"
    log_info "=========================================="
    log_info "Model: ${MODEL}"
    log_info "DCGM Port: ${DCGM_PORT}"
    log_info "vLLM Port: ${VLLM_PORT}"
    log_info ""

    # Change to script directory
    cd "$SCRIPT_DIR"

    # Clean up any existing containers
    log_info "Stopping any existing containers..."
    docker compose -f docker-compose.yml down -v 2>/dev/null || true

    # Remove old output directory
    if [ -d "$OUTPUT_DIR" ]; then
        log_info "Removing old output directory..."
        rm -rf "$OUTPUT_DIR"
    fi

    # Start infrastructure
    log_info "Starting DCGM exporter and vLLM server..."
    docker compose -f docker-compose.yml up -d

    # Wait for DCGM exporter to be ready
    if ! timeout 60 bash -c "while true; do \
        STATUS=\$(curl -s -o /dev/null -w '%{http_code}' localhost:${DCGM_PORT}/metrics); \
        if [ \"\$STATUS\" = '200' ]; then \
            if curl -s localhost:${DCGM_PORT}/metrics | grep -q 'DCGM_FI_DEV_GPU_UTIL'; then \
                break; \
            fi; \
        fi; \
        sleep 5; \
    done"; then
        log_error "DCGM metrics not available after 60 seconds"
        docker logs dcgm-exporter
        exit 1
    fi
    log_info "✓ DCGM metrics are available"

    # Wait for vLLM server to be ready
    if ! timeout 120 bash -c "while [ \"\$(curl -s -o /dev/null -w '%{http_code}' \
        localhost:${VLLM_PORT}/v1/chat/completions \
        -H 'Content-Type: application/json' \
        -d '{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"test\"}],\"max_tokens\":1}')\" != '200' ]; do \
        sleep 2; \
    done"; then
        log_error "vLLM server not ready after 120 seconds"
        docker logs vllm-server
        exit 1
    fi
    log_info "✓ vLLM server is ready"

    # Run AIPerf benchmark
    log_info "Running AIPerf benchmark with GPU telemetry..."
    aiperf profile \
        --model "${MODEL}" \
        --endpoint-type chat \
        --endpoint /v1/chat/completions \
        --streaming \
        --url "localhost:${VLLM_PORT}" \
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
        log_info "✓ vLLM PATHWAY TEST PASSED"
        log_info "=========================================="
        exit 0
    else
        log_error "=========================================="
        log_error "✗ vLLM PATHWAY TEST FAILED"
        log_error "=========================================="
        exit 1
    fi
}

main "$@"
