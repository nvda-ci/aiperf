# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from time import perf_counter_ns, time_ns
from typing import Any, ClassVar, Literal

from pydantic import ConfigDict, Field, computed_field

from aiperf.common.models.base_models import AIPerfBaseModel


class BaseTraceData(AIPerfBaseModel):
    """Base trace data model.

    Captures timing information for trace data lifecycle using perf_counter_ns().

    Fields organized by phase:

    Reference Timestamps (time synchronization):
      - reference_time_ns: Wall-clock sync point (time.time_ns)
      - reference_perf_ns: Performance counter sync point

    Request Send Phase:
      - request_send_start_perf_ns: Request send start
      - request_headers: Request headers dictionary
      - request_headers_sent_perf_ns: Request headers sent complete
      - request_write_timestamps_perf_ns: List of chunk write timestamps
      - request_write_sizes_bytes: List of chunk sizes
      - request_send_end_perf_ns: Request send end

    Response Receive Phase:
      - response_receive_start_perf_ns: Response receive start
      - response_status_code: Response status code
      - response_headers: Response headers dictionary
      - response_headers_received_perf_ns: Response headers received complete
      - response_receive_timestamps_perf_ns: List of chunk receive timestamps
      - response_receive_sizes_bytes: List of chunk sizes
      - response_receive_end_perf_ns: Response receive end

    Error Tracking:
      - error_timestamp_perf_ns: Error timestamp (if any)

    Note: All *_perf_ns fields use time.perf_counter_ns() for high-precision measurements.
    """

    # For auto-routed-model serialization and deserialization
    discriminator_field: ClassVar[str] = "trace_type"

    trace_type: str = Field(
        ...,
        description="The type of the trace. This is typically the name of the library used",
        frozen=True,
    )

    # Reference Timestamps for converting between wall-clock and perf_counter time.
    reference_time_ns: int | None = Field(
        default=None,
        description="A reference timestamp in wall-clock time for helping with timing calculations (time.time_ns()).",
    )
    reference_perf_ns: int | None = Field(
        default=None,
        description="A reference perf timestamp for helping with timing calculations (time.perf_counter_ns()). "
        "This is the perf_counter_ns value when the reference_time_ns was set.",
    )

    def model_post_init(self, __context: Any) -> None:
        """Initialize the reference time_ns and perf_counter_ns timestamps."""
        if self.reference_time_ns is None or self.reference_perf_ns is None:
            # NOTE: perf_counter is slightly faster than time_ns, so we do it first for tighter coupling.
            #       We also do them as a single operation to avoid timing gaps between the two functions.
            self.reference_perf_ns, self.reference_time_ns = (
                perf_counter_ns(),
                time_ns(),
            )

    # Request Send Phase
    request_send_start_perf_ns: int | None = Field(
        default=None,
        description="When the HTTP request started being sent (perf_counter_ns).",
    )
    request_headers: dict[str, str] | None = Field(
        default=None,
        description="The headers of the request.",
    )
    request_headers_sent_perf_ns: int | None = Field(
        default=None,
        description="When the request headers were sent to the server (perf_counter_ns).",
    )
    request_write_timestamps_perf_ns: list[int] = Field(
        default_factory=list,
        description="Timestamps when request data was written to transport (perf_counter_ns). Useful for tracking upload progress. "
        "Maps to aiohttp's on_request_chunk_sent events. These are transport-layer writes, not application messages.",
    )
    request_write_sizes_bytes: list[int] = Field(
        default_factory=list,
        description="Sizes in bytes of each request write to transport. These are transport-layer writes, not application messages.",
    )
    request_send_end_perf_ns: int | None = Field(
        default=None,
        description="When the HTTP request finished being sent (perf_counter_ns).",
    )

    # Response Receive Phase
    response_status_code: int | None = Field(
        default=None,
        description="The status code of the response.",
    )
    response_receive_start_perf_ns: int | None = Field(
        default=None,
        description="When the response started being received from the server (perf_counter_ns).",
    )
    response_headers: dict[str, str] | None = Field(
        default=None,
        description="The headers of the response.",
    )
    response_headers_received_perf_ns: int | None = Field(
        default=None,
        description="When the response headers were received from the server (perf_counter_ns).",
    )
    response_receive_timestamps_perf_ns: list[int] = Field(
        default_factory=list,
        description="Timestamps when response data was received from the server (perf_counter_ns). Useful for tracking download progress. "
        "Maps to aiohttp's on_response_chunk_received events. These are transport-layer reads, not application messages (SSE events).",
    )
    response_receive_sizes_bytes: list[int] = Field(
        default_factory=list,
        description="Sizes in bytes of each response received from transport. These are transport-layer reads, not application messages (SSE events).",
    )
    response_receive_end_perf_ns: int | None = Field(
        default=None,
        description="When the response finished being received from the server (perf_counter_ns).",
    )

    # Errors
    error_timestamp_perf_ns: int | None = Field(
        default=None,
        description="When an exception occurred during the request (perf_counter_ns). "
        "Maps to aiohttp's on_request_exception event.",
    )

    def _convert_perf_to_wall(self, perf_ns: int | None) -> int | None:
        """Convert perf_counter timestamp to wall-clock timestamp.

        Args:
            perf_ns: Perf counter timestamp to convert

        Returns:
            Wall-clock timestamp or None if input is None
        """
        if perf_ns is None:
            return None
        if self.reference_time_ns is None or self.reference_perf_ns is None:
            raise ValueError(
                "Cannot convert without reference timestamps. "
                "Ensure reference_time_ns and reference_perf_ns are set."
            )
        return self.reference_time_ns + (perf_ns - self.reference_perf_ns)

    def to_export(self) -> "TraceDataExport":
        """Convert to export model with wall-clock timestamps.

        Uses TraceDataExport's _model_lookup_table to auto-detect the correct export class
        based on trace_type (leverages the auto-routed-model infrastructure).

        Returns:
            Export model with all perf_counter timestamps converted to wall-clock time
        """
        # Auto-detect export class using the trace_type discriminator
        # This leverages the existing auto-routed-model infrastructure
        export_class = TraceDataExport._model_lookup_table.get(
            self.trace_type, TraceDataExport
        )

        # Get all fields from the model
        data = self.model_dump()

        # Convert all *_perf_ns fields to *_ns (wall-clock)
        export_data = {}
        for key, value in data.items():
            if key in ("reference_time_ns", "reference_perf_ns"):
                # Skip reference fields (not needed in export)
                continue
            elif key.endswith("_perf_ns"):
                # Smart conversion: just replace _perf_ns with _ns
                export_key = key.replace("_perf_ns", "_ns")

                # Convert timestamp field
                if isinstance(value, list):
                    # Convert list of timestamps
                    export_data[export_key] = [
                        self._convert_perf_to_wall(ts) for ts in value
                    ]
                else:
                    # Convert single timestamp
                    export_data[export_key] = self._convert_perf_to_wall(value)
            else:
                # Copy non-timestamp fields as-is
                export_data[key] = value

        return export_class(**export_data)


class AioHttpTraceData(BaseTraceData):
    """Comprehensive trace data for aiohttp requests using the tracing event system.

    Extends BaseTraceData with aiohttp-specific connection and network timing details.

    Additional fields organized by phase:

    Connection Pool Phase:
      - connection_pool_wait_start_perf_ns: Queue wait start for an available connection
      - connection_pool_wait_end_perf_ns: Queue wait end for an available connection

    Connection Reuse:
      - connection_reused_perf_ns: Existing connection reused timestamp

    DNS Resolution Phase:
      - dns_lookup_start_perf_ns: DNS resolution start
      - dns_lookup_end_perf_ns: DNS resolution complete
      - dns_cache_hit_perf_ns: DNS cache hit occurred
      - dns_cache_miss_perf_ns: DNS cache miss occurred

    TCP Connection Phase (only for new connections):
      - tcp_connect_start_perf_ns: TCP connection establishment start
      - tcp_connect_end_perf_ns: TCP connection established
        Note: For HTTPS, this includes both TCP and TLS handshake time combined.

    Note: Inherits all fields from BaseTraceData (reference, request, response, error).
          All timestamps use perf_counter_ns() for high-precision measurements.
    """

    trace_type: str = "aiohttp"

    # Connection Pool
    connection_pool_wait_start_perf_ns: int | None = Field(
        default=None,
        description="When the request started waiting for an available connection from the pool (perf_counter_ns). "
        "Maps to aiohttp's on_connection_queued_start event.",
    )
    connection_pool_wait_end_perf_ns: int | None = Field(
        default=None,
        description="When an available connection was obtained from the pool (perf_counter_ns). "
        "Maps to aiohttp's on_connection_queued_end event.",
    )

    # TCP Connection (only set if a new connection is created)
    tcp_connect_start_perf_ns: int | None = Field(
        default=None,
        description="When TCP connection establishment started (perf_counter_ns). "
        "Maps to aiohttp's on_connection_create_start event.",
    )
    tcp_connect_end_perf_ns: int | None = Field(
        default=None,
        description="When TCP connection establishment completed (perf_counter_ns). "
        "Maps to aiohttp's on_connection_create_end event.",
    )

    # Connection Reuse
    connection_reused_perf_ns: int | None = Field(
        default=None,
        description="When an existing connection was reused from the pool (perf_counter_ns). "
        "Maps to aiohttp's on_connection_reuseconn event.",
    )

    # DNS Resolution
    dns_lookup_start_perf_ns: int | None = Field(
        default=None,
        description="When DNS resolution started for the hostname (perf_counter_ns). "
        "Maps to aiohttp's on_dns_resolvehost_start event.",
    )
    dns_lookup_end_perf_ns: int | None = Field(
        default=None,
        description="When DNS resolution completed for the hostname (perf_counter_ns). "
        "Maps to aiohttp's on_dns_resolvehost_end event.",
    )
    dns_cache_hit_perf_ns: int | None = Field(
        default=None,
        description="When a DNS cache hit occurred (perf_counter_ns). "
        "Maps to aiohttp's on_dns_cache_hit event.",
    )
    dns_cache_miss_perf_ns: int | None = Field(
        default=None,
        description="When a DNS cache miss occurred (perf_counter_ns). "
        "Maps to aiohttp's on_dns_cache_miss event.",
    )


class TraceDataExport(AIPerfBaseModel):
    """Export model with wall-clock timestamps following k6 and HAR conventions.

    All timestamps are converted from perf_counter to wall-clock time (time.time_ns())
    for correlation with logs, metadata, and cross-system analysis.

    Create from BaseTraceData using trace_data.to_export() method.

    Fields match BaseTraceData exactly, but with _perf_ns replaced by _ns (wall-clock).

    Computed Durations (k6/HAR compatible):
      - sending_ns: Request send time (k6: http_req_sending, HAR: send)
      - waiting_ns: TTFB / server processing (k6: http_req_waiting, HAR: wait)
      - receiving_ns: Response transfer time (k6: http_req_receiving, HAR: receive)
      - total_ns: Total request duration (k6: http_req_duration, HAR: time)

    Note: All timestamps are in wall-clock time (time.time_ns()) for cross-system correlation.
    """

    # For auto-routed-model serialization and deserialization
    discriminator_field: ClassVar[str] = "trace_type"

    trace_type: str = Field(
        ...,
        description="The type of the trace. This is typically the name of the library used"
        "and must match the trace_type of the corresponding trace data model.",
    )

    # Enable computed fields in serialization
    model_config = ConfigDict(use_attribute_docstrings=True)

    # Request Send Phase (matches BaseTraceData field names)
    request_send_start_ns: int | None = Field(
        default=None,
        description="When the HTTP request started being sent (wall-clock time.time_ns()).",
    )
    request_headers: dict[str, str] | None = Field(
        default=None,
        description="The headers of the request.",
    )
    request_headers_sent_ns: int | None = Field(
        default=None,
        description="When the request headers were sent to the server (wall-clock time.time_ns()).",
    )
    request_write_timestamps_ns: list[int] = Field(
        default_factory=list,
        description="Timestamps when request data was written to transport (wall-clock time.time_ns()). "
        "Transport-layer writes, not application messages.",
    )
    request_write_sizes_bytes: list[int] = Field(
        default_factory=list,
        description="Sizes in bytes of each request write to transport.",
    )
    request_send_end_ns: int | None = Field(
        default=None,
        description="When the HTTP request finished being sent (wall-clock time.time_ns()).",
    )

    # Response Receive Phase (matches BaseTraceData field names)
    response_status_code: int | None = Field(
        default=None,
        description="The status code of the response.",
    )
    response_receive_start_ns: int | None = Field(
        default=None,
        description="When the response started being received from the server (wall-clock time.time_ns()).",
    )
    response_headers: dict[str, str] | None = Field(
        default=None,
        description="The headers of the response.",
    )
    response_headers_received_ns: int | None = Field(
        default=None,
        description="When the response headers were received from the server (wall-clock time.time_ns()).",
    )
    response_receive_timestamps_ns: list[int] = Field(
        default_factory=list,
        description="Timestamps when response data was received from the server (wall-clock time.time_ns()). "
        "Transport-layer reads, not application messages.",
    )
    response_receive_sizes_bytes: list[int] = Field(
        default_factory=list,
        description="Sizes in bytes of each response received from transport.",
    )
    response_receive_end_ns: int | None = Field(
        default=None,
        description="When the response finished being received from the server (wall-clock time.time_ns()).",
    )

    # Error Tracking
    error_timestamp_ns: int | None = Field(
        default=None,
        description="When an exception occurred during the request (wall-clock time.time_ns()).",
    )

    # Computed Durations (k6/HAR compatible)
    @computed_field  # type: ignore[prop-decorator]
    @property
    def sending_ns(self) -> int | None:
        """Request send time (k6: http_req_sending, HAR: send)."""
        if self.request_send_start_ns and self.request_send_end_ns:
            return self.request_send_end_ns - self.request_send_start_ns
        return None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def waiting_ns(self) -> int | None:
        """TTFB / server processing time (k6: http_req_waiting, HAR: wait)."""
        if self.request_send_end_ns and self.response_receive_timestamps_ns:
            return self.response_receive_timestamps_ns[0] - self.request_send_end_ns
        return None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def receiving_ns(self) -> int | None:
        """Response transfer time (k6: http_req_receiving, HAR: receive)."""
        if self.response_receive_timestamps_ns:
            if len(self.response_receive_timestamps_ns) > 1:
                return (
                    self.response_receive_timestamps_ns[-1]
                    - self.response_receive_timestamps_ns[0]
                )
            return 0  # Single chunk - no transfer time
        return None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_ns(self) -> int | None:
        """Total request duration (k6: http_req_duration, HAR: time)."""
        if self.request_send_start_ns and self.response_receive_end_ns:
            return self.response_receive_end_ns - self.request_send_start_ns
        return None


class AioHttpTraceDataExport(TraceDataExport):
    """Export model for aiohttp with connection-level timing following k6/HAR conventions.

    Extends TraceDataExport with aiohttp-specific connection pool, DNS, TCP, and TLS timing.

    Fields match AioHttpTraceData exactly, but with _perf_ns replaced by _ns (wall-clock).

    Additional Computed Durations (k6/HAR compatible):
      - blocked_ns: Connection pool wait time (k6: http_req_blocked, HAR: blocked)
      - dns_lookup_ns: DNS lookup time (k6: http_req_looking_up, HAR: dns)
      - connecting_ns: TCP connection time including TLS (k6: http_req_connecting, HAR: connect)

    Note: Inherits all fields from TraceDataExport (request, response, error, durations).
          All timestamps are in wall-clock time (time.time_ns()).
    """

    trace_type: Literal["aiohttp"] = "aiohttp"

    # Connection Pool (matches AioHttpTraceData field names)
    connection_pool_wait_start_ns: int | None = Field(
        default=None,
        description="When the request started waiting for an available connection from the pool (wall-clock time.time_ns()).",
    )
    connection_pool_wait_end_ns: int | None = Field(
        default=None,
        description="When an available connection was obtained from the pool (wall-clock time.time_ns()).",
    )

    # TCP Connection (matches AioHttpTraceData field names)
    tcp_connect_start_ns: int | None = Field(
        default=None,
        description="When TCP connection establishment started (wall-clock time.time_ns()).",
    )
    tcp_connect_end_ns: int | None = Field(
        default=None,
        description="When TCP connection establishment completed (wall-clock time.time_ns()).",
    )

    # Connection Reuse (matches AioHttpTraceData field names)
    connection_reused_ns: int | None = Field(
        default=None,
        description="When an existing connection was reused from the pool (wall-clock time.time_ns()).",
    )

    # DNS Resolution (matches AioHttpTraceData field names)
    dns_lookup_start_ns: int | None = Field(
        default=None,
        description="When DNS resolution started for the hostname (wall-clock time.time_ns()).",
    )
    dns_lookup_end_ns: int | None = Field(
        default=None,
        description="When DNS resolution completed for the hostname (wall-clock time.time_ns()).",
    )
    dns_cache_hit_ns: int | None = Field(
        default=None,
        description="When a DNS cache hit occurred (wall-clock time.time_ns()).",
    )
    dns_cache_miss_ns: int | None = Field(
        default=None,
        description="When a DNS cache miss occurred (wall-clock time.time_ns()).",
    )

    # Additional Computed Durations
    @computed_field  # type: ignore[prop-decorator]
    @property
    def blocked_ns(self) -> int | None:
        """Connection pool wait time (k6: http_req_blocked, HAR: blocked)."""
        if self.connection_pool_wait_start_ns and self.connection_pool_wait_end_ns:
            return self.connection_pool_wait_end_ns - self.connection_pool_wait_start_ns
        return None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def dns_lookup_ns(self) -> int | None:
        """DNS lookup time (k6: http_req_looking_up, HAR: dns)."""
        if self.dns_lookup_start_ns and self.dns_lookup_end_ns:
            return self.dns_lookup_end_ns - self.dns_lookup_start_ns
        return None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def connecting_ns(self) -> int | None:
        """TCP connection time including TLS for HTTPS (k6: http_req_connecting, HAR: connect)."""
        if self.tcp_connect_start_ns and self.tcp_connect_end_ns:
            return self.tcp_connect_end_ns - self.tcp_connect_start_ns
        return None
