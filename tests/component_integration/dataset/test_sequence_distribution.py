# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Component integration tests for sequence length distribution.

Tests the --sequence-distribution parameter which specifies distribution of
input/output sequence lengths using format: "isl|osl:weight;isl|osl:weight;..."
"""

import pytest

from tests.component_integration.conftest import (
    ComponentIntegrationTestDefaults as defaults,
)
from tests.harness.utils import AIPerfCLI


@pytest.mark.component_integration
class TestSequenceLengthDistribution:
    """Test sequence length distribution functionality."""

    def test_sequence_length_distribution_basic(self, cli: AIPerfCLI):
        """Test basic sequence length distribution with three buckets."""
        result = cli.run_sync(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --endpoint-type chat \
                --streaming \
                --random-seed 42 \
                --sequence-distribution "64|10,32|8:70;256|40,128|20:20;1024|100,512|50:10" \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """,
            timeout=120.0,
        )

        # Verify all requests have sequence length metrics
        assert result.jsonl is not None
        assert result.jsonl

        for record in result.jsonl:
            isl = record.metrics.get("input_sequence_length")
            osl = record.metrics.get("output_sequence_length")

            assert isl is not None, "Missing input_sequence_length metric"
            assert osl is not None, "Missing output_sequence_length metric"
            assert isl.value > 0, f"Invalid ISL: {isl.value}"
            assert osl.value > 0, f"Invalid OSL: {osl.value}"

    def test_sequence_distribution_single_bucket(self, cli: AIPerfCLI):
        """Test sequence distribution with a single bucket."""
        result = cli.run_sync(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --endpoint-type chat \
                --streaming \
                --random-seed 42 \
                --sequence-distribution "128|50,64|25:100" \
                --num-sessions 20 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """,
            timeout=60.0,
        )

        # With a single bucket (100% weight), all requests should have these lengths
        for record in result.jsonl:
            isl = record.metrics.get("input_sequence_length").value
            osl = record.metrics.get("output_sequence_length").value

            # Should be within range of 128 ± 3*stddev and 64 ± 3*stddev (99.7% of values)
            # Mean 128, stddev 50 → range [0, 278]
            # Mean 64, stddev 25 → range [0, 139]
            assert 0 < isl <= 281, f"ISL {isl} outside expected range"
            assert 0 < osl <= 140, f"OSL {osl} outside expected range"

    def test_sequence_distribution_weights(self, cli: AIPerfCLI):
        """Test sequence distribution weight distribution."""
        result = cli.run_sync(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --endpoint-type chat \
                --streaming \
                --random-seed 42 \
                --sequence-distribution "100|20,50|10:50;200|40,100|20:50" \
                --num-sessions 100 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """,
            timeout=120.0,
        )

        # Collect sequence lengths
        isl_values = []
        osl_values = []

        for record in result.jsonl:
            isl_values.append(record.metrics.get("input_sequence_length").value)
            osl_values.append(record.metrics.get("output_sequence_length").value)

        # With 50:50 weight split and fixed seed, we should have roughly
        # equal distribution between the two buckets
        # Bucket 1: ISL ~ 100, OSL ~ 50
        # Bucket 2: ISL ~ 200, OSL ~ 100

        # Count requests in each bucket (rough heuristic based on ISL)
        bucket1_count = sum(1 for isl in isl_values if isl < 150)
        bucket2_count = sum(1 for isl in isl_values if isl >= 150)

        # With 50:50 weights, expect roughly equal counts (allow 30% tolerance)
        expected_per_bucket = len(isl_values) / 2
        tolerance = expected_per_bucket * 0.30

        assert abs(bucket1_count - expected_per_bucket) < tolerance, (
            f"Bucket 1 count {bucket1_count} deviates from expected {expected_per_bucket}"
        )
        assert abs(bucket2_count - expected_per_bucket) < tolerance, (
            f"Bucket 2 count {bucket2_count} deviates from expected {expected_per_bucket}"
        )

    def test_sequence_distribution_with_multi_turn(self, cli: AIPerfCLI):
        """Test sequence distribution with multi-turn conversations."""
        result = cli.run_sync(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --endpoint-type chat \
                --streaming \
                --random-seed 42 \
                --sequence-distribution "64|10,32|8:100" \
                --num-sessions 10 \
                --session-turns-mean 3 \
                --session-turns-stddev 0 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """,
            timeout=60.0,
        )

        # Should have 10 sessions × 3 turns = 30 requests
        assert len(result.jsonl) == 30

        # All requests should have sequence length metrics
        for record in result.jsonl:
            isl = record.metrics.get("input_sequence_length")
            osl = record.metrics.get("output_sequence_length")

            assert isl is not None
            assert osl is not None
            assert isl.value > 0
            assert osl.value > 0

    def test_sequence_distribution_with_rate_limiting(self, cli: AIPerfCLI):
        """Test sequence distribution with request rate limiting."""
        result = cli.run_sync(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --endpoint-type chat \
                --streaming \
                --random-seed 42 \
                --sequence-distribution "128|20,64|10:100" \
                --request-rate 10 \
                --request-rate-mode constant \
                --num-sessions 15 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """,
            timeout=60.0,
        )

        # Verify sequence distribution applied correctly with rate limiting
        assert len(result.jsonl) == 15

        for record in result.jsonl:
            isl = record.metrics.get("input_sequence_length")
            osl = record.metrics.get("output_sequence_length")

            assert isl is not None
            assert osl is not None
            # Values should be around 128 and 64 with some stddev
            assert 50 <= isl.value <= 200
            assert 20 <= osl.value <= 100
