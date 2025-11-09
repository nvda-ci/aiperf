# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.enums import ResultsProcessorType
from aiperf.common.models import MetricResult, ProfileResults
from aiperf.common.models.processor_summary_results import (
    MetricSummaryResult,
    TimesliceSummaryResult,
)
from aiperf.records.records_manager import ProcessRecordsResult


class TestRecordsManagerTimeslice:
    """Test cases for RecordsManager timeslice functionality."""

    @pytest.mark.asyncio
    async def test_process_records_result_with_timeslice_summary(self):
        """Test that ProcessRecordsResult can contain timeslice results in summary_results."""

        metric_result = MetricResult(
            tag="request_latency",
            header="Request Latency",
            unit="ms",
            avg=100.0,
            count=10,
        )

        timeslice_results = {
            0: [metric_result],
            1: [metric_result],
        }

        # Create a ProcessRecordsResult with timeslice summary in summary_results
        timeslice_summary = TimesliceSummaryResult(
            timeslice_results=timeslice_results,
        )

        result = ProcessRecordsResult(
            summary_results={
                ResultsProcessorType.TIMESLICE: timeslice_summary,
            },
            profile_summary=ProfileResults(
                completed=2,
                start_ns=1000000000,
                end_ns=2000000000,
            ),
        )

        # Extract timeslice results from summary_results
        assert ResultsProcessorType.TIMESLICE in result.summary_results
        timeslice_summary_result = result.summary_results[
            ResultsProcessorType.TIMESLICE
        ]
        assert isinstance(timeslice_summary_result, TimesliceSummaryResult)
        assert timeslice_summary_result.timeslice_results is not None
        assert len(timeslice_summary_result.timeslice_results) == 2

    @pytest.mark.asyncio
    async def test_process_records_result_with_metric_summary(self):
        """Test that ProcessRecordsResult can contain metric results in summary_results."""
        metric_result = MetricResult(
            tag="request_latency",
            header="Request Latency",
            unit="ms",
            avg=100.0,
            count=10,
        )

        # Create a ProcessRecordsResult with metric summary in summary_results
        metric_summary = MetricSummaryResult(
            results=[metric_result, metric_result],
        )

        result = ProcessRecordsResult(
            summary_results={
                ResultsProcessorType.METRIC_RESULTS: metric_summary,
            },
            profile_summary=ProfileResults(
                completed=2,
                start_ns=1000000000,
                end_ns=2000000000,
            ),
        )

        # Extract metric results from summary_results
        assert ResultsProcessorType.METRIC_RESULTS in result.summary_results
        metric_summary_result = result.summary_results[
            ResultsProcessorType.METRIC_RESULTS
        ]
        assert isinstance(metric_summary_result, MetricSummaryResult)
        assert metric_summary_result.results is not None
        assert len(metric_summary_result.results) == 2

    @pytest.mark.asyncio
    async def test_timeslice_summary_serialization(self):
        """Test that TimesliceSummaryResult can be serialized."""
        metric_result = MetricResult(
            tag="request_latency",
            header="Request Latency",
            unit="ms",
            avg=100.0,
            count=10,
        )

        timeslice_results = {
            0: [metric_result],
            1: [metric_result],
        }

        timeslice_summary = TimesliceSummaryResult(
            timeslice_results=timeslice_results,
        )

        # Test that it can be converted to dict (for JSON serialization)
        result_dict = timeslice_summary.model_dump()

        assert "timeslice_results" in result_dict
        assert result_dict["timeslice_results"] is not None
        assert 0 in result_dict["timeslice_results"]
        assert 1 in result_dict["timeslice_results"]
