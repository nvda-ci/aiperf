# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pydantic import Field

from aiperf.common.models.base_models import AIPerfBaseModel

################################################################################
# AioHTTP Trace Timestamps Model
################################################################################


class AioHttpTraceTimestamps(AIPerfBaseModel):
    """Comprehensive timestamp tracking for aiohttp requests using the tracing event system.

    This model captures all timing information from various stages of an HTTP request lifecycle,
    including connection pooling, DNS resolution, connection establishment, and data transfer.
    All timestamps are captured using time.perf_counter_ns() for high-precision measurements.
    """

    # Connection Pool Timestamps
    connection_queued_start_ns: int | None = Field(
        default=None,
        description="When the request started waiting for an available connection from the pool (perf_counter_ns).",
    )
    connection_queued_end_ns: int | None = Field(
        default=None,
        description="When an available connection was obtained from the pool (perf_counter_ns).",
    )

    # Connection Creation Timestamps (only set if a new connection needs to be created)
    connection_create_start_ns: int | None = Field(
        default=None,
        description="When the creation of a new connection started (perf_counter_ns).",
    )
    connection_create_end_ns: int | None = Field(
        default=None,
        description="When the creation of a new connection completed (perf_counter_ns).",
    )

    # Connection Reuse
    connection_reuseconn_ns: int | None = Field(
        default=None,
        description="When an existing connection was reused from the pool (perf_counter_ns).",
    )

    # DNS Resolution Timestamps
    dns_resolvehost_start_ns: int | None = Field(
        default=None,
        description="When DNS resolution started for the hostname (perf_counter_ns).",
    )
    dns_resolvehost_end_ns: int | None = Field(
        default=None,
        description="When DNS resolution completed for the hostname (perf_counter_ns).",
    )
    dns_cache_hit_ns: int | None = Field(
        default=None,
        description="When a DNS cache hit occurred (perf_counter_ns).",
    )
    dns_cache_miss_ns: int | None = Field(
        default=None,
        description="When a DNS cache miss occurred (perf_counter_ns).",
    )

    # Request Timestamps
    request_start_ns: int | None = Field(
        default=None,
        description="When the HTTP request started being sent (perf_counter_ns).",
    )
    request_headers_sent_ns: int | None = Field(
        default=None,
        description="When the HTTP request headers finished being sent (perf_counter_ns).",
    )
    request_end_ns: int | None = Field(
        default=None,
        description="When the HTTP request finished being sent (perf_counter_ns).",
    )

    # Request Chunk Tracking
    request_chunk_sent_ns: list[int] = Field(
        default_factory=list,
        description="Timestamps of when each request chunk was sent (perf_counter_ns). Useful for tracking upload progress.",
    )

    # Response Timestamps
    response_chunk_received_ns: list[int] = Field(
        default_factory=list,
        description="Timestamps of when each response chunk was received (perf_counter_ns). Useful for tracking download progress.",
    )

    # Redirect Tracking
    request_redirect_ns: list[int] = Field(
        default_factory=list,
        description="Timestamps of when redirects occurred (perf_counter_ns).",
    )

    # Exception Tracking
    request_exception_ns: int | None = Field(
        default=None,
        description="When an exception occurred during the request (perf_counter_ns).",
    )

    # Metadata extracted from trace params
    dns_host: str | None = Field(
        default=None,
        description="The hostname that was resolved via DNS.",
    )
    connection_host: str | None = Field(
        default=None,
        description="The actual host (IP:port or hostname:port) of the connection established.",
    )
    request_method: str | None = Field(
        default=None,
        description="The HTTP method of the request (GET, POST, etc.).",
    )
    request_url: str | None = Field(
        default=None,
        description="The full URL of the request.",
    )
    request_chunk_sizes: list[int] = Field(
        default_factory=list,
        description="Sizes in bytes of each request chunk sent.",
    )
    response_chunk_sizes: list[int] = Field(
        default_factory=list,
        description="Sizes in bytes of each response chunk received.",
    )
    redirect_urls: list[str] = Field(
        default_factory=list,
        description="URLs of redirects that occurred during the request.",
    )
    redirect_status_codes: list[int] = Field(
        default_factory=list,
        description="HTTP status codes of redirect responses.",
    )
    exception_type: str | None = Field(
        default=None,
        description="The type of exception that occurred, if any.",
    )
    exception_message: str | None = Field(
        default=None,
        description="The message of the exception that occurred, if any.",
    )
    response_headers: dict[str, str] | None = Field(
        default=None,
        description="Response headers from the HTTP response.",
    )
    response_status: int | None = Field(
        default=None,
        description="HTTP status code from the response.",
    )
    response_reason: str | None = Field(
        default=None,
        description="HTTP reason phrase from the response.",
    )

    @property
    def total_request_bytes(self) -> int:
        """Calculate total bytes sent in request chunks."""
        return sum(self.request_chunk_sizes)

    @property
    def total_response_bytes(self) -> int:
        """Calculate total bytes received in response chunks."""
        return sum(self.response_chunk_sizes)

    @property
    def connection_queue_wait_ns(self) -> int | None:
        """Calculate how long the request waited for an available connection from the pool."""
        if self.connection_queued_start_ns and self.connection_queued_end_ns:
            return self.connection_queued_end_ns - self.connection_queued_start_ns
        return None

    @property
    def connection_create_duration_ns(self) -> int | None:
        """Calculate how long it took to create a new connection."""
        if self.connection_create_start_ns and self.connection_create_end_ns:
            return self.connection_create_end_ns - self.connection_create_start_ns
        return None

    @property
    def dns_resolution_duration_ns(self) -> int | None:
        """Calculate how long DNS resolution took."""
        if self.dns_resolvehost_start_ns and self.dns_resolvehost_end_ns:
            return self.dns_resolvehost_end_ns - self.dns_resolvehost_start_ns
        return None

    @property
    def request_send_duration_ns(self) -> int | None:
        """Calculate how long it took to send the request."""
        if self.request_start_ns and self.request_end_ns:
            return self.request_end_ns - self.request_start_ns
        return None

    @property
    def request_headers_duration_ns(self) -> int | None:
        """Calculate how long it took to send request headers."""
        if self.request_start_ns and self.request_headers_sent_ns:
            return self.request_headers_sent_ns - self.request_start_ns
        return None

    @property
    def request_body_duration_ns(self) -> int | None:
        """Calculate how long it took to send the request body (after headers)."""
        if self.request_headers_sent_ns and self.request_end_ns:
            return self.request_end_ns - self.request_headers_sent_ns
        return None

    @property
    def total_request_chunks(self) -> int:
        """Get the total number of request chunks sent."""
        return len(self.request_chunk_sent_ns)

    @property
    def total_response_chunks(self) -> int:
        """Get the total number of response chunks received."""
        return len(self.response_chunk_received_ns)

    @property
    def total_redirects(self) -> int:
        """Get the total number of redirects."""
        return len(self.request_redirect_ns)

    # ============================================================================
    # CRITICAL TIMING BREAKDOWNS
    # ============================================================================

    @property
    def time_to_first_byte_ns(self) -> int | None:
        """Time to First Byte (TTFB) - Critical server responsiveness metric.

        Measures time from request end to first response byte received.
        Lower is better - indicates server processing + network time.
        """
        if self.request_end_ns and self.response_chunk_received_ns:
            return self.response_chunk_received_ns[0] - self.request_end_ns
        return None

    @property
    def time_to_last_byte_ns(self) -> int | None:
        """Time to Last Byte (TTLB) - Full download completion time.

        Measures time from request end to last response byte received.
        """
        if self.request_end_ns and self.response_chunk_received_ns:
            return self.response_chunk_received_ns[-1] - self.request_end_ns
        return None

    @property
    def server_processing_time_ns(self) -> int | None:
        """Estimated server processing time (TTFB minus request send time).

        Approximates how long the server took to process and start responding.
        """
        ttfb = self.time_to_first_byte_ns
        send_duration = self.request_send_duration_ns
        if ttfb is not None and send_duration is not None:
            return max(0, ttfb - send_duration)
        return None

    @property
    def network_transfer_time_ns(self) -> int | None:
        """Pure data transfer time (TTLB minus TTFB).

        Time spent transferring response data after first byte received.
        """
        ttlb = self.time_to_last_byte_ns
        ttfb = self.time_to_first_byte_ns
        if ttlb is not None and ttfb is not None:
            return ttlb - ttfb
        return None

    @property
    def connection_establishment_time_ns(self) -> int | None:
        """Total connection establishment overhead.

        Sum of: queue wait + DNS resolution + connection creation.
        """
        total = 0
        count = 0

        if self.connection_queue_wait_ns:
            total += self.connection_queue_wait_ns
            count += 1
        if self.dns_resolution_duration_ns:
            total += self.dns_resolution_duration_ns
            count += 1
        if self.connection_create_duration_ns:
            total += self.connection_create_duration_ns
            count += 1

        return total if count > 0 else None

    # ============================================================================
    # BANDWIDTH & THROUGHPUT
    # ============================================================================

    @property
    def upload_rate_bytes_per_sec(self) -> float | None:
        """Upload bandwidth in bytes per second."""
        if self.total_request_bytes > 0 and self.request_send_duration_ns:
            return self.total_request_bytes / (
                self.request_send_duration_ns / 1_000_000_000
            )
        return None

    @property
    def download_rate_bytes_per_sec(self) -> float | None:
        """Download bandwidth in bytes per second."""
        transfer_time = self.network_transfer_time_ns
        if self.total_response_bytes > 0 and transfer_time and transfer_time > 0:
            return self.total_response_bytes / (transfer_time / 1_000_000_000)
        return None

    @property
    def avg_request_chunk_rate_bytes_per_sec(self) -> float | None:
        """Average upload rate per chunk."""
        if not self.request_chunk_sent_ns or len(self.request_chunk_sent_ns) < 2:
            return None

        total_bytes = self.total_request_bytes
        time_span = self.request_chunk_sent_ns[-1] - self.request_chunk_sent_ns[0]

        if total_bytes > 0 and time_span > 0:
            return total_bytes / (time_span / 1_000_000_000)
        return None

    @property
    def avg_response_chunk_rate_bytes_per_sec(self) -> float | None:
        """Average download rate per chunk."""
        if (
            not self.response_chunk_received_ns
            or len(self.response_chunk_received_ns) < 2
        ):
            return None

        total_bytes = self.total_response_bytes
        time_span = (
            self.response_chunk_received_ns[-1] - self.response_chunk_received_ns[0]
        )

        if total_bytes > 0 and time_span > 0:
            return total_bytes / (time_span / 1_000_000_000)
        return None

    # ============================================================================
    # STATISTICAL ANALYSIS
    # ============================================================================

    @property
    def response_inter_chunk_latencies_ns(self) -> list[int]:
        """Calculate latency between each response chunk."""
        if len(self.response_chunk_received_ns) < 2:
            return []

        return [
            self.response_chunk_received_ns[i] - self.response_chunk_received_ns[i - 1]
            for i in range(1, len(self.response_chunk_received_ns))
        ]

    @property
    def response_chunk_latency_min_ns(self) -> int | None:
        """Minimum inter-chunk latency."""
        latencies = self.response_inter_chunk_latencies_ns
        return min(latencies) if latencies else None

    @property
    def response_chunk_latency_max_ns(self) -> int | None:
        """Maximum inter-chunk latency."""
        latencies = self.response_inter_chunk_latencies_ns
        return max(latencies) if latencies else None

    @property
    def response_chunk_latency_avg_ns(self) -> float | None:
        """Average inter-chunk latency."""
        latencies = self.response_inter_chunk_latencies_ns
        return sum(latencies) / len(latencies) if latencies else None

    @property
    def response_chunk_latency_jitter_ns(self) -> float | None:
        """Inter-chunk latency variance (jitter) - indicates network stability."""
        latencies = self.response_inter_chunk_latencies_ns
        if len(latencies) < 2:
            return None

        avg = sum(latencies) / len(latencies)
        variance = sum((x - avg) ** 2 for x in latencies) / len(latencies)
        return variance**0.5  # Standard deviation

    # ============================================================================
    # HEADER & BODY INTELLIGENCE
    # ============================================================================

    @property
    def request_headers_size_bytes(self) -> int:
        """Estimated size of request headers in bytes."""
        if not self.request_method or not self.request_url:
            return 0

        # Rough estimate: method + URL + HTTP/1.1 + headers
        size = len(self.request_method) + len(self.request_url) + 10
        return size

    @property
    def response_headers_size_bytes(self) -> int:
        """Estimated size of response headers in bytes."""
        if not self.response_headers:
            return 0

        # Rough estimate: status line + all headers
        size = 15  # "HTTP/1.1 200 OK"
        for key, value in self.response_headers.items():
            size += len(key) + len(value) + 4  # ": " + "\r\n"
        return size

    @property
    def compression_type(self) -> str | None:
        """Detected compression type from Content-Encoding header."""
        if self.response_headers:
            return self.response_headers.get("Content-Encoding")
        return None

    @property
    def response_content_length(self) -> int | None:
        """Expected response size from Content-Length header."""
        if self.response_headers:
            length = self.response_headers.get("Content-Length")
            if length:
                try:
                    return int(length)
                except ValueError:
                    pass
        return None

    @property
    def response_content_type(self) -> str | None:
        """Response content type."""
        if self.response_headers:
            return self.response_headers.get("Content-Type")
        return None

    @property
    def transfer_encoding(self) -> str | None:
        """Transfer encoding type (e.g., 'chunked')."""
        if self.response_headers:
            return self.response_headers.get("Transfer-Encoding")
        return None

    @property
    def compression_ratio(self) -> float | None:
        """Compression ratio if both content-length and actual size available."""
        expected = self.response_content_length
        actual = self.total_response_bytes

        if expected and actual and expected > 0:
            return actual / expected
        return None

    # ============================================================================
    # CONNECTION INSIGHTS
    # ============================================================================

    @property
    def connection_was_reused(self) -> bool:
        """Whether an existing connection was reused."""
        return self.connection_reuseconn_ns is not None

    @property
    def dns_was_cached(self) -> bool:
        """Whether DNS result came from cache."""
        return self.dns_cache_hit_ns is not None

    @property
    def connection_overhead_percentage(self) -> float | None:
        """Connection establishment time as percentage of total request time."""
        establishment = self.connection_establishment_time_ns

        if establishment and self.request_end_ns and self.request_start_ns:
            total = self.request_end_ns - self.request_start_ns
            if total > 0:
                return (establishment / total) * 100
        return None

    @property
    def queue_wait_percentage(self) -> float | None:
        """Connection queue wait as percentage of total request time."""
        queue_wait = self.connection_queue_wait_ns

        if queue_wait and self.request_end_ns and self.request_start_ns:
            total = self.request_end_ns - self.request_start_ns
            if total > 0:
                return (queue_wait / total) * 100
        return None

    # ============================================================================
    # DERIVED QUALITY METRICS
    # ============================================================================

    @property
    def is_streaming_response(self) -> bool:
        """Whether response was streamed (multiple chunks)."""
        return len(self.response_chunk_received_ns) > 1

    @property
    def network_vs_server_time_ratio(self) -> float | None:
        """Ratio of network time to server processing time.

        < 1.0 = Network-bound (server is fast)
        > 1.0 = Server-bound (network is fast)
        """
        network = self.network_transfer_time_ns
        server = self.server_processing_time_ns

        if network and server and server > 0:
            return network / server
        return None

    @property
    def request_efficiency_score(self) -> float | None:
        """Overall request efficiency (0-100).

        Higher is better. Factors in:
        - Connection reuse
        - DNS caching
        - Low connection overhead
        - Stable chunk delivery
        """
        score = 0.0
        factors = 0

        # Connection reuse (30 points)
        if self.connection_was_reused:
            score += 30
            factors += 1
        elif self.connection_create_duration_ns:
            factors += 1

        # DNS caching (20 points)
        if self.dns_was_cached:
            score += 20
            factors += 1
        elif self.dns_resolution_duration_ns:
            factors += 1

        # Low connection overhead (30 points)
        overhead = self.connection_overhead_percentage
        if overhead is not None:
            score += max(0, 30 - overhead)
            factors += 1

        # Stable chunk delivery (20 points)
        jitter = self.response_chunk_latency_jitter_ns
        avg = self.response_chunk_latency_avg_ns
        if jitter is not None and avg and avg > 0:
            stability = max(0, 1 - (jitter / avg))
            score += stability * 20
            factors += 1

        return score / factors if factors > 0 else None
