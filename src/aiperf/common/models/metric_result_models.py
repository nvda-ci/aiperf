# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metric result models.

This module contains metric-related models that are used across multiple modules.
It's extracted to avoid circular dependencies between processor_summary_results
and record_models.
"""

from pydantic import Field

from aiperf.common.constants import STAT_KEYS
from aiperf.common.enums import CreditPhase
from aiperf.common.enums.metric_enums import MetricValueTypeT
from aiperf.common.models.base_models import AIPerfBaseModel
from aiperf.common.models.error_models import ErrorDetails
from aiperf.common.models.export_models import JsonMetricResult
from aiperf.common.types import MetricTagT


class MetricResult(JsonMetricResult):
    """The result values of a single metric."""

    tag: MetricTagT = Field(description="The unique identifier of the metric")
    # NOTE: We do not use a MetricUnitT here, as that is harder to de-serialize from JSON strings with pydantic.
    #       If we need an instance of a MetricUnitT, lookup the unit based on the tag in the MetricRegistry.
    header: str = Field(
        description="The user friendly name of the metric (e.g. 'Inter Token Latency')"
    )
    count: int | None = Field(
        default=None,
        description="The total number of records used to calculate the metric",
    )
    current: float | None = Field(
        default=None,
        description="The most recent value of the metric (used for realtime dashboard display only)",
    )

    def to_display_unit(self) -> "MetricResult":
        """Convert the metric result to its display unit."""
        from aiperf.exporters.display_units_utils import to_display_unit
        from aiperf.metrics.metric_registry import MetricRegistry

        return to_display_unit(self, MetricRegistry)

    def to_json_result(self) -> JsonMetricResult:
        """Convert the metric result to a JsonMetricResult."""
        result = JsonMetricResult(unit=self.unit)
        for stat in STAT_KEYS:
            setattr(result, stat, getattr(self, stat, None))
        return result


class MetricValue(AIPerfBaseModel):
    """The value of a metric converted to display units for export."""

    value: MetricValueTypeT
    unit: str


class MetricRecordMetadata(AIPerfBaseModel):
    """The metadata of a metric record for export."""

    session_num: int = Field(
        ...,
        description="The sequential number of the session in the benchmark. For single-turn datasets, this will be the"
        " request index. For multi-turn datasets, this will be the session index.",
    )
    x_request_id: str | None = Field(
        default=None,
        description="The X-Request-ID header of the request. This is a unique ID for the request.",
    )
    x_correlation_id: str | None = Field(
        default=None,
        description="The X-Correlation-ID header of the request. This is a shared ID for each user session/conversation in multi-turn.",
    )
    conversation_id: str | None = Field(
        default=None,
        description="The ID of the conversation (if applicable). This can be used to lookup the original request data from the inputs.json file.",
    )
    turn_index: int | None = Field(
        default=None,
        description="The index of the turn in the conversation (if applicable). This can be used to lookup the original request data from the inputs.json file.",
    )
    request_start_ns: int = Field(
        ...,
        description="The wall clock timestamp of the request start time measured as time.time_ns().",
    )
    request_ack_ns: int | None = Field(
        default=None,
        description="The wall clock timestamp of the request acknowledgement from the server, measured as time.time_ns(), if applicable. "
        "This is only applicable to streaming requests, and servers that send 200 OK back immediately after the request is received.",
    )
    request_end_ns: int = Field(
        ...,
        description="The wall clock timestamp of the request end time measured as time.time_ns(). If the request failed, "
        "this will be the time of the error.",
    )
    worker_id: str = Field(
        ..., description="The ID of the AIPerf worker that processed the request."
    )
    record_processor_id: str = Field(
        ...,
        description="The ID of the AIPerf record processor that processed the record.",
    )
    benchmark_phase: CreditPhase = Field(
        ...,
        description="The benchmark phase of the record, either warmup or profiling.",
    )
    was_cancelled: bool = Field(
        default=False,
        description="Whether the request was cancelled during execution.",
    )
    cancellation_time_ns: int | None = Field(
        default=None,
        description="The wall clock timestamp of the request cancellation time measured as time.time_ns(), if applicable. "
        "This is only applicable to requests that were cancelled.",
    )


class MetricRecordInfo(AIPerfBaseModel):
    """The full info of a metric record including the metadata, metrics, and error for export."""

    metadata: MetricRecordMetadata = Field(
        ...,
        description="The metadata of the record. Should match the metadata in the MetricRecordsMessage.",
    )
    metrics: dict[str, MetricValue] = Field(
        ...,
        description="A dictionary containing all metric values along with their units.",
    )
    error: ErrorDetails | None = Field(
        default=None,
        description="The error details if the request failed.",
    )
