# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HAR (HTTP Archive) 1.2 format exporter for Chrome DevTools compatibility."""

from urllib.parse import parse_qsl, urlparse

import orjson

from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.decorators import implements_protocol
from aiperf.common.enums import ExportLevel, ResultsProcessorType
from aiperf.common.exceptions import PostProcessorDisabled
from aiperf.common.factories import ResultsProcessorFactory
from aiperf.common.messages.inference_messages import MetricRecordsData
from aiperf.common.mixins import AIPerfLifecycleMixin
from aiperf.common.models.har_models import (
    HAR,
    HARCache,
    HARContent,
    HARCreator,
    HAREntry,
    HARLog,
    HARNameValuePair,
    HARQueryString,
    HARRequest,
    HARResponse,
    HARTimings,
    format_iso8601_timestamp,
    ns_to_ms,
)
from aiperf.common.models.model_endpoint_info import ModelEndpointInfo
from aiperf.common.models.record_models import MetricResult
from aiperf.common.models.trace_models import AioHttpTraceDataExport
from aiperf.common.protocols import ResultsProcessorProtocol


@implements_protocol(ResultsProcessorProtocol)
@ResultsProcessorFactory.register(ResultsProcessorType.HAR)
class HARResultsProcessor(AIPerfLifecycleMixin):
    """Exports benchmark results to HAR (HTTP Archive) 1.2 format.

    Creates a standards-compliant HAR file with comprehensive timing information
    from trace data. Compatible with Chrome DevTools Network panel and other
    HAR analysis tools.

    Only enabled when export_level is RECORDS or RAW and trace data is available.
    """

    def __init__(
        self,
        service_id: str,
        service_config: ServiceConfig,
        user_config: UserConfig,
        **kwargs,
    ):
        super().__init__(**kwargs)

        export_level = user_config.output.export_level
        if export_level not in (ExportLevel.RECORDS, ExportLevel.RAW):
            raise PostProcessorDisabled(
                f"HAR results processor is disabled for export level {export_level}"
            )

        self.user_config = user_config
        self.output_file = user_config.output.profile_export_har_file
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.output_file.unlink(missing_ok=True)

        # Store endpoint info for URL construction
        self._model_endpoint = ModelEndpointInfo.from_user_config(user_config)

        # Collect all entries during processing
        self._entries: list[HAREntry] = []
        self._entry_count = 0
        self._entries_with_trace = 0

        self.info(f"HAR export enabled: {self.output_file}")

    async def process_result(self, record_data: MetricRecordsData) -> None:
        """Process a single record and add it to the HAR entries list.

        Args:
            record_data: Record containing metrics, metadata, and trace data
        """
        try:
            # Only export records with trace data
            if not record_data.trace_data:
                return

            # Convert trace data to export format (wall-clock timestamps)
            trace_data_export = record_data.trace_data.to_export()

            # Only support AioHttpTraceDataExport for now
            if not isinstance(trace_data_export, AioHttpTraceDataExport):
                self.warning(
                    lambda: f"Skipping non-aiohttp trace data: {type(trace_data_export).__name__}"
                )
                return

            entry = self._create_har_entry(record_data, trace_data_export)
            self._entries.append(entry)
            self._entries_with_trace += 1

        except Exception as e:
            self.error(f"Failed to create HAR entry: {e!r}")
        finally:
            self._entry_count += 1

    async def summarize(self) -> list[MetricResult]:
        """Finalize and write the HAR file.

        Returns:
            Empty list (HAR processor doesn't produce metric results)
        """
        if not self._entries:
            self.warning("No entries with trace data collected. HAR file not created.")
            return []

        try:
            # Sort entries by startedDateTime (oldest first) per HAR spec recommendation
            self._entries.sort(key=lambda e: e.startedDateTime)

            # Create HAR structure
            har = HAR(
                log=HARLog(
                    version="1.2",
                    creator=HARCreator(
                        name="AIPerf",
                        version="0.3.0",
                        comment="AI Benchmarking Tool - https://github.com/NVIDIA/AIPerf",
                    ),
                    browser=None,  # Not a browser-based tool
                    pages=[],  # No page grouping for API benchmarks
                    entries=self._entries,
                    comment=f"AI inference benchmark - {self._entries_with_trace} requests with trace data out of {self._entry_count} total",
                )
            )

            # Write HAR file as formatted JSON
            har_json = har.model_dump(mode="json", exclude_none=True)
            har_bytes = orjson.dumps(
                har_json,
                option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS,
            )

            self.output_file.write_bytes(har_bytes)

            self.info(
                f"HAR export complete: {len(self._entries)} entries written to {self.output_file}"
            )

        except Exception as e:
            self.error(f"Failed to write HAR file: {e!r}")

        return []

    def _create_har_entry(
        self, record_data: MetricRecordsData, trace_data: AioHttpTraceDataExport
    ) -> HAREntry:
        """Create a HAR entry from record data and trace data.

        Args:
            record_data: Record containing metrics and metadata
            trace_data: Exported trace data with wall-clock timestamps

        Returns:
            HAR entry object
        """
        metadata = record_data.metadata

        # Build full URL
        url = self._build_url()

        # Create request object
        request = self._create_har_request(url, trace_data)

        # Create response object
        response = self._create_har_response(trace_data)

        # Create timings object
        timings = self._create_har_timings(trace_data)

        # Calculate total time (sum of all non-null timings)
        total_time_ms = sum(
            t
            for t in [
                timings.blocked,
                timings.dns,
                timings.connect,
                timings.send,
                timings.wait,
                timings.receive,
            ]
            if t is not None
        )

        # Create entry
        entry = HAREntry(
            pageref=None,  # No page grouping
            startedDateTime=format_iso8601_timestamp(metadata.request_start_ns),
            time=total_time_ms,
            request=request,
            response=response,
            cache=HARCache(),  # Empty cache object (no cache info available)
            timings=timings,
            serverIPAddress=None,  # Not available in trace data
            connection=None,  # Not available in trace data
            comment=f"Session {metadata.session_num}, Worker {metadata.worker_id}"
            + (
                f", X-Request-ID: {metadata.x_request_id}"
                if metadata.x_request_id
                else ""
            ),
        )

        return entry

    def _build_url(self) -> str:
        """Build the full URL from endpoint configuration.

        Returns:
            Full URL string
        """
        endpoint = self._model_endpoint.endpoint
        base_url = endpoint.base_url or ""

        if endpoint.custom_endpoint:
            # Ensure proper URL joining
            if not base_url.endswith("/") and not endpoint.custom_endpoint.startswith(
                "/"
            ):
                return f"{base_url}/{endpoint.custom_endpoint}"
            return f"{base_url}{endpoint.custom_endpoint}"

        return base_url

    def _create_har_request(
        self, url: str, trace_data: AioHttpTraceDataExport
    ) -> HARRequest:
        """Create HAR request object from trace data.

        Args:
            url: Full request URL
            trace_data: Trace data with request information

        Returns:
            HAR request object
        """
        # Parse query string from URL
        parsed = urlparse(url)
        query_params = [
            HARQueryString(name=name, value=value)
            for name, value in parse_qsl(parsed.query)
        ]

        # Convert headers from dict to HAR format
        headers = [
            HARNameValuePair(name=name, value=value)
            for name, value in (trace_data.request_headers or {}).items()
        ]

        # Calculate request size from trace data
        request_body_size = (
            sum(trace_data.request_write_sizes_bytes)
            if trace_data.request_write_sizes_bytes
            else 0
        )

        # Headers size is not directly available, estimate or use -1
        headers_size = -1

        return HARRequest(
            method="POST",  # LLM APIs typically use POST
            url=url,
            httpVersion="HTTP/1.1",  # Default, could be HTTP/2
            cookies=[],  # Cookies not tracked in current trace data
            headers=headers,
            queryString=query_params,
            postData=None,  # Request body not stored in trace data
            headersSize=headers_size,
            bodySize=request_body_size,
        )

    def _create_har_response(self, trace_data: AioHttpTraceDataExport) -> HARResponse:
        """Create HAR response object from trace data.

        Args:
            trace_data: Trace data with response information

        Returns:
            HAR response object
        """
        # Convert headers from dict to HAR format
        headers = [
            HARNameValuePair(name=name, value=value)
            for name, value in (trace_data.response_headers or {}).items()
        ]

        # Calculate response size from trace data
        response_body_size = (
            sum(trace_data.response_receive_sizes_bytes)
            if trace_data.response_receive_sizes_bytes
            else 0
        )

        # Get content type from headers
        content_type = "application/json"  # Default for LLM APIs
        for name, value in (trace_data.response_headers or {}).items():
            if name.lower() == "content-type":
                content_type = value
                break

        # Create content object
        content = HARContent(
            size=response_body_size,
            compression=None,  # Compression info not directly available
            mimeType=content_type,
            text=None,  # Response body not stored in trace data
            encoding=None,
        )

        # Get redirect URL from Location header
        redirect_url = ""
        for name, value in (trace_data.response_headers or {}).items():
            if name.lower() == "location":
                redirect_url = value
                break

        # Status code and text
        status_code = trace_data.response_status_code or 0
        status_text = self._get_status_text(status_code)

        # Headers size is not directly available, estimate or use -1
        headers_size = -1

        return HARResponse(
            status=status_code,
            statusText=status_text,
            httpVersion="HTTP/1.1",  # Default, could be HTTP/2
            cookies=[],  # Cookies not tracked in current trace data
            headers=headers,
            content=content,
            redirectURL=redirect_url,
            headersSize=headers_size,
            bodySize=response_body_size,
        )

    def _create_har_timings(self, trace_data: AioHttpTraceDataExport) -> HARTimings:
        """Create HAR timings object from trace data.

        Converts nanosecond timings to milliseconds as required by HAR spec.

        Args:
            trace_data: Trace data with timing information

        Returns:
            HAR timings object
        """
        # Convert all timings from nanoseconds to milliseconds
        # Use -1 for unavailable timings (but HAR spec says use None, we'll use None)
        blocked_ms = ns_to_ms(trace_data.blocked_ns)
        dns_ms = ns_to_ms(trace_data.dns_lookup_ns)
        connect_ms = ns_to_ms(trace_data.connecting_ns)
        send_ms = ns_to_ms(trace_data.sending_ns) or 0  # Required field
        wait_ms = ns_to_ms(trace_data.waiting_ns) or 0  # Required field
        receive_ms = ns_to_ms(trace_data.receiving_ns) or 0  # Required field

        # SSL time is included in connect time per HAR spec
        # We don't have separate SSL timing, so set to None
        ssl_ms = None

        return HARTimings(
            blocked=blocked_ms,
            dns=dns_ms,
            connect=connect_ms,
            send=send_ms,
            wait=wait_ms,
            receive=receive_ms,
            ssl=ssl_ms,
            comment=None,
        )

    @staticmethod
    def _get_status_text(status_code: int) -> str:
        """Get HTTP status text for a status code.

        Args:
            status_code: HTTP status code

        Returns:
            Status text description
        """
        status_texts = {
            200: "OK",
            201: "Created",
            202: "Accepted",
            204: "No Content",
            301: "Moved Permanently",
            302: "Found",
            304: "Not Modified",
            400: "Bad Request",
            401: "Unauthorized",
            403: "Forbidden",
            404: "Not Found",
            405: "Method Not Allowed",
            408: "Request Timeout",
            429: "Too Many Requests",
            500: "Internal Server Error",
            502: "Bad Gateway",
            503: "Service Unavailable",
            504: "Gateway Timeout",
        }
        return status_texts.get(status_code, "Unknown")
