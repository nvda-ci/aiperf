# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Comprehensive tests for HAR (HTTP Archive) 1.2 models."""

from datetime import datetime

import orjson
import pytest

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


class TestHelperFunctions:
    """Tests for HAR helper functions."""

    def test_format_iso8601_timestamp(self):
        """ISO 8601 timestamp formatting is correct."""
        # 2024-01-01 00:00:00 UTC
        timestamp_ns = 1704067200000000000
        formatted = format_iso8601_timestamp(timestamp_ns)

        assert formatted == "2024-01-01T00:00:00.000Z"
        assert formatted.endswith("Z")
        assert "T" in formatted

    def test_format_iso8601_timestamp_with_milliseconds(self):
        """ISO 8601 timestamp includes millisecond precision."""
        # 2024-01-01 00:00:00.123 UTC
        timestamp_ns = 1704067200123000000
        formatted = format_iso8601_timestamp(timestamp_ns)

        assert formatted == "2024-01-01T00:00:00.123Z"

    def test_format_iso8601_timestamp_roundtrip(self):
        """Formatted timestamp can be parsed back."""
        timestamp_ns = 1704067200123000000
        formatted = format_iso8601_timestamp(timestamp_ns)

        # Parse back
        dt = datetime.fromisoformat(formatted.replace("Z", "+00:00"))
        reconstructed_ns = int(dt.timestamp() * 1_000_000_000)

        # Should match original (within millisecond precision)
        assert abs(reconstructed_ns - timestamp_ns) < 1_000_000

    def test_ns_to_ms_conversion(self):
        """Nanoseconds to milliseconds conversion is correct."""
        assert ns_to_ms(1_000_000) == 1.0
        assert ns_to_ms(1_500_000) == 1.5
        assert ns_to_ms(1_234_567) == pytest.approx(1.234567)

    def test_ns_to_ms_with_none(self):
        """ns_to_ms handles None correctly."""
        assert ns_to_ms(None) is None

    def test_ns_to_ms_with_zero(self):
        """ns_to_ms handles zero correctly."""
        assert ns_to_ms(0) == 0.0


class TestHARNameValuePair:
    """Tests for HARNameValuePair model."""

    def test_creation_with_required_fields(self):
        """Name/value pair can be created with required fields."""
        pair = HARNameValuePair(name="Content-Type", value="application/json")

        assert pair.name == "Content-Type"
        assert pair.value == "application/json"
        assert pair.comment is None

    def test_creation_with_comment(self):
        """Name/value pair can include optional comment."""
        pair = HARNameValuePair(name="X-Custom", value="test", comment="Custom header")

        assert pair.comment == "Custom header"

    def test_serialization(self):
        """Name/value pair serializes correctly."""
        pair = HARNameValuePair(name="Accept", value="*/*")
        data = pair.model_dump()

        assert data["name"] == "Accept"
        assert data["value"] == "*/*"


class TestHARTimings:
    """Tests for HARTimings model."""

    def test_creation_with_required_fields_only(self):
        """Timings can be created with only required fields."""
        timings = HARTimings(send=10.5, wait=50.2, receive=30.8)

        assert timings.send == 10.5
        assert timings.wait == 50.2
        assert timings.receive == 30.8
        assert timings.blocked is None
        assert timings.dns is None
        assert timings.connect is None
        assert timings.ssl is None

    def test_creation_with_all_fields(self):
        """Timings can be created with all fields."""
        timings = HARTimings(
            blocked=5.0,
            dns=2.5,
            connect=15.0,
            send=10.5,
            wait=50.2,
            receive=30.8,
            ssl=8.0,
        )

        assert timings.blocked == 5.0
        assert timings.dns == 2.5
        assert timings.connect == 15.0
        assert timings.ssl == 8.0

    def test_total_time_calculation(self):
        """Total time equals sum of all non-null timings."""
        timings = HARTimings(
            blocked=5.0,
            dns=2.5,
            connect=15.0,
            send=10.5,
            wait=50.2,
            receive=30.8,
        )

        total = sum(
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

        assert total == pytest.approx(114.0)

    def test_ssl_included_in_connect(self):
        """SSL time is documented to be included in connect time."""
        # Per HAR spec, ssl time is included in connect for backward compatibility
        timings = HARTimings(
            connect=15.0,
            send=10.0,
            wait=20.0,
            receive=5.0,
            ssl=8.0,  # Should be <= connect
        )

        assert timings.ssl <= timings.connect


class TestHARContent:
    """Tests for HARContent model."""

    def test_creation_with_required_fields(self):
        """Content can be created with required fields."""
        content = HARContent(size=1024, mimeType="application/json")

        assert content.size == 1024
        assert content.mimeType == "application/json"
        assert content.compression is None
        assert content.text is None
        assert content.encoding is None

    def test_creation_with_compression(self):
        """Content can include compression info."""
        content = HARContent(size=2048, compression=1024, mimeType="application/json")

        assert content.compression == 1024
        # Original size would be size + compression
        assert content.size + content.compression == 3072

    def test_creation_with_text_content(self):
        """Content can include text."""
        content = HARContent(
            size=100,
            mimeType="text/plain",
            text="Hello, World!",
        )

        assert content.text == "Hello, World!"
        assert content.encoding is None

    def test_creation_with_base64_encoding(self):
        """Content can include base64 encoded binary data."""
        content = HARContent(
            size=100,
            mimeType="application/octet-stream",
            text="SGVsbG8gV29ybGQh",  # "Hello World!" in base64
            encoding="base64",
        )

        assert content.encoding == "base64"
        assert content.text is not None


class TestHARRequest:
    """Tests for HARRequest model."""

    def test_creation_with_minimal_fields(self):
        """Request can be created with minimal required fields."""
        request = HARRequest(
            method="POST",
            url="https://api.example.com/v1/completions",
            httpVersion="HTTP/1.1",
            headersSize=-1,
            bodySize=-1,
        )

        assert request.method == "POST"
        assert request.url == "https://api.example.com/v1/completions"
        assert request.httpVersion == "HTTP/1.1"
        assert request.headersSize == -1  # Not available
        assert request.bodySize == -1  # Not available
        assert request.cookies == []
        assert request.headers == []
        assert request.queryString == []

    def test_creation_with_headers(self):
        """Request can include headers."""
        request = HARRequest(
            method="POST",
            url="https://api.example.com/v1/completions",
            httpVersion="HTTP/1.1",
            headers=[
                HARNameValuePair(name="Content-Type", value="application/json"),
                HARNameValuePair(name="Authorization", value="Bearer token"),
            ],
            headersSize=100,
            bodySize=500,
        )

        assert len(request.headers) == 2
        assert request.headers[0].name == "Content-Type"

    def test_creation_with_query_string(self):
        """Request can include query parameters."""
        request = HARRequest(
            method="GET",
            url="https://api.example.com/search?q=test&limit=10",
            httpVersion="HTTP/1.1",
            queryString=[
                HARQueryString(name="q", value="test"),
                HARQueryString(name="limit", value="10"),
            ],
            headersSize=50,
            bodySize=0,
        )

        assert len(request.queryString) == 2
        assert request.queryString[0].name == "q"
        assert request.queryString[1].value == "10"


class TestHARResponse:
    """Tests for HARResponse model."""

    def test_creation_with_minimal_fields(self):
        """Response can be created with minimal required fields."""
        response = HARResponse(
            status=200,
            statusText="OK",
            httpVersion="HTTP/1.1",
            content=HARContent(size=1024, mimeType="application/json"),
            headersSize=-1,
            bodySize=1024,
        )

        assert response.status == 200
        assert response.statusText == "OK"
        assert response.content.size == 1024
        assert response.redirectURL == ""
        assert response.cookies == []
        assert response.headers == []

    def test_creation_with_headers(self):
        """Response can include headers."""
        response = HARResponse(
            status=200,
            statusText="OK",
            httpVersion="HTTP/1.1",
            headers=[
                HARNameValuePair(name="Content-Type", value="application/json"),
                HARNameValuePair(name="Content-Length", value="1024"),
            ],
            content=HARContent(size=1024, mimeType="application/json"),
            headersSize=100,
            bodySize=1024,
        )

        assert len(response.headers) == 2

    def test_redirect_response(self):
        """Redirect response includes Location header."""
        response = HARResponse(
            status=301,
            statusText="Moved Permanently",
            httpVersion="HTTP/1.1",
            content=HARContent(size=0, mimeType="text/html"),
            redirectURL="https://example.com/new-location",
            headersSize=100,
            bodySize=0,
        )

        assert response.status == 301
        assert response.redirectURL == "https://example.com/new-location"

    def test_cached_response(self):
        """Cached response (304) has zero body size."""
        response = HARResponse(
            status=304,
            statusText="Not Modified",
            httpVersion="HTTP/1.1",
            content=HARContent(size=0, mimeType="application/json"),
            headersSize=100,
            bodySize=0,
        )

        assert response.status == 304
        assert response.bodySize == 0


class TestHAREntry:
    """Tests for HAREntry model."""

    def test_creation_with_minimal_fields(self):
        """Entry can be created with required fields."""
        entry = HAREntry(
            startedDateTime="2024-01-01T00:00:00.000Z",
            time=100.0,
            request=HARRequest(
                method="POST",
                url="https://api.example.com/v1/completions",
                httpVersion="HTTP/1.1",
                headersSize=-1,
                bodySize=500,
            ),
            response=HARResponse(
                status=200,
                statusText="OK",
                httpVersion="HTTP/1.1",
                content=HARContent(size=1024, mimeType="application/json"),
                headersSize=-1,
                bodySize=1024,
            ),
            cache=HARCache(),
            timings=HARTimings(send=10.0, wait=50.0, receive=40.0),
        )

        assert entry.startedDateTime == "2024-01-01T00:00:00.000Z"
        assert entry.time == 100.0
        assert entry.request.method == "POST"
        assert entry.response.status == 200
        assert entry.pageref is None

    def test_creation_with_optional_fields(self):
        """Entry can include optional fields."""
        entry = HAREntry(
            pageref="page_1",
            startedDateTime="2024-01-01T00:00:00.000Z",
            time=100.0,
            request=HARRequest(
                method="POST",
                url="https://api.example.com/v1/completions",
                httpVersion="HTTP/1.1",
                headersSize=100,
                bodySize=500,
            ),
            response=HARResponse(
                status=200,
                statusText="OK",
                httpVersion="HTTP/1.1",
                content=HARContent(size=1024, mimeType="application/json"),
                headersSize=100,
                bodySize=1024,
            ),
            cache=HARCache(),
            timings=HARTimings(
                blocked=5.0, dns=2.0, connect=10.0, send=10.0, wait=50.0, receive=23.0
            ),
            serverIPAddress="192.168.1.100",
            connection="443",
            comment="Test request",
        )

        assert entry.pageref == "page_1"
        assert entry.serverIPAddress == "192.168.1.100"
        assert entry.connection == "443"
        assert entry.comment == "Test request"

    def test_time_matches_timings_sum(self):
        """Entry time should equal sum of all timings."""
        timings = HARTimings(
            blocked=5.0, dns=2.0, connect=10.0, send=10.0, wait=50.0, receive=23.0
        )

        total = sum(
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

        entry = HAREntry(
            startedDateTime="2024-01-01T00:00:00.000Z",
            time=total,
            request=HARRequest(
                method="POST",
                url="https://api.example.com/v1/completions",
                httpVersion="HTTP/1.1",
                headersSize=-1,
                bodySize=500,
            ),
            response=HARResponse(
                status=200,
                statusText="OK",
                httpVersion="HTTP/1.1",
                content=HARContent(size=1024, mimeType="application/json"),
                headersSize=-1,
                bodySize=1024,
            ),
            cache=HARCache(),
            timings=timings,
        )

        assert entry.time == pytest.approx(total)


class TestHARLog:
    """Tests for HARLog model."""

    def test_creation_with_minimal_fields(self):
        """Log can be created with minimal required fields."""
        log = HARLog(
            creator=HARCreator(name="AIPerf", version="0.3.0"),
            entries=[],
        )

        assert log.version == "1.2"
        assert log.creator.name == "AIPerf"
        assert log.browser is None
        assert log.pages == []
        assert log.entries == []

    def test_creation_with_entries(self):
        """Log can include entries."""
        entries = [
            HAREntry(
                startedDateTime="2024-01-01T00:00:00.000Z",
                time=100.0,
                request=HARRequest(
                    method="POST",
                    url="https://api.example.com/v1/completions",
                    httpVersion="HTTP/1.1",
                    headersSize=-1,
                    bodySize=500,
                ),
                response=HARResponse(
                    status=200,
                    statusText="OK",
                    httpVersion="HTTP/1.1",
                    content=HARContent(size=1024, mimeType="application/json"),
                    headersSize=-1,
                    bodySize=1024,
                ),
                cache=HARCache(),
                timings=HARTimings(send=10.0, wait=50.0, receive=40.0),
            )
        ]

        log = HARLog(
            creator=HARCreator(name="AIPerf", version="0.3.0"),
            entries=entries,
        )

        assert len(log.entries) == 1
        assert log.entries[0].request.method == "POST"


class TestHAR:
    """Tests for HAR root model."""

    def test_creation_with_minimal_log(self):
        """HAR can be created with minimal log."""
        har = HAR(
            log=HARLog(
                creator=HARCreator(name="AIPerf", version="0.3.0"),
                entries=[],
            )
        )

        assert har.log.version == "1.2"
        assert har.log.creator.name == "AIPerf"

    def test_full_har_structure(self):
        """Complete HAR structure with entries."""
        har = HAR(
            log=HARLog(
                version="1.2",
                creator=HARCreator(
                    name="AIPerf",
                    version="0.3.0",
                    comment="AI Benchmarking Tool",
                ),
                entries=[
                    HAREntry(
                        startedDateTime="2024-01-01T00:00:00.000Z",
                        time=100.0,
                        request=HARRequest(
                            method="POST",
                            url="https://api.example.com/v1/completions",
                            httpVersion="HTTP/1.1",
                            headers=[
                                HARNameValuePair(
                                    name="Content-Type", value="application/json"
                                ),
                            ],
                            headersSize=100,
                            bodySize=500,
                        ),
                        response=HARResponse(
                            status=200,
                            statusText="OK",
                            httpVersion="HTTP/1.1",
                            headers=[
                                HARNameValuePair(
                                    name="Content-Type", value="application/json"
                                ),
                            ],
                            content=HARContent(size=1024, mimeType="application/json"),
                            headersSize=100,
                            bodySize=1024,
                        ),
                        cache=HARCache(),
                        timings=HARTimings(
                            blocked=5.0,
                            dns=2.0,
                            connect=10.0,
                            send=10.0,
                            wait=50.0,
                            receive=23.0,
                        ),
                        serverIPAddress="192.168.1.100",
                        connection="443",
                    )
                ],
            )
        )

        assert len(har.log.entries) == 1
        assert har.log.entries[0].response.status == 200

    def test_serialization_to_json(self):
        """HAR serializes to valid JSON."""
        har = HAR(
            log=HARLog(
                creator=HARCreator(name="AIPerf", version="0.3.0"),
                entries=[],
            )
        )

        # Serialize
        har_dict = har.model_dump(mode="json", exclude_none=True)
        har_json = orjson.dumps(har_dict, option=orjson.OPT_INDENT_2)

        # Should be valid JSON
        parsed = orjson.loads(har_json)
        assert "log" in parsed
        assert parsed["log"]["version"] == "1.2"
        assert parsed["log"]["creator"]["name"] == "AIPerf"

    def test_deserialization_from_json(self):
        """HAR can be deserialized from JSON."""
        har_json = {
            "log": {
                "version": "1.2",
                "creator": {"name": "AIPerf", "version": "0.3.0"},
                "entries": [],
            }
        }

        har = HAR(**har_json)

        assert har.log.version == "1.2"
        assert har.log.creator.name == "AIPerf"

    def test_chrome_devtools_compatible_structure(self):
        """HAR structure is compatible with Chrome DevTools."""
        # Chrome DevTools requires specific fields
        har = HAR(
            log=HARLog(
                version="1.2",  # Must be "1.2"
                creator=HARCreator(name="AIPerf", version="0.3.0"),
                browser=None,  # Optional
                pages=[],  # Can be empty
                entries=[
                    HAREntry(
                        startedDateTime="2024-01-01T00:00:00.000Z",  # ISO 8601 format
                        time=100.0,  # Total time in ms
                        request=HARRequest(
                            method="POST",
                            url="https://api.example.com/v1/completions",
                            httpVersion="HTTP/1.1",
                            cookies=[],
                            headers=[],
                            queryString=[],
                            headersSize=-1,  # -1 for not available
                            bodySize=500,
                        ),
                        response=HARResponse(
                            status=200,
                            statusText="OK",
                            httpVersion="HTTP/1.1",
                            cookies=[],
                            headers=[],
                            content=HARContent(size=1024, mimeType="application/json"),
                            redirectURL="",
                            headersSize=-1,
                            bodySize=1024,
                        ),
                        cache=HARCache(),  # Can be empty
                        timings=HARTimings(
                            send=10.0,
                            wait=50.0,
                            receive=40.0,  # Required
                        ),
                    )
                ],
            )
        )

        # Verify Chrome DevTools requirements
        assert har.log.version == "1.2"
        assert isinstance(har.log.entries, list)
        assert har.log.entries[0].time == 100.0
        assert har.log.entries[0].timings.send is not None
        assert har.log.entries[0].timings.wait is not None
        assert har.log.entries[0].timings.receive is not None


class TestHARValidation:
    """Tests for HAR model validation."""

    def test_missing_required_fields_raises_error(self):
        """Missing required fields raises validation error."""
        with pytest.raises(Exception):  # Pydantic validation error
            HARRequest(
                # Missing method, url, httpVersion
                headersSize=-1,
                bodySize=-1,
            )

    def test_exclude_none_removes_optional_fields(self):
        """Serialization with exclude_none removes None fields."""
        timings = HARTimings(
            send=10.0,
            wait=50.0,
            receive=40.0,
            # blocked, dns, connect, ssl are None
        )

        data = timings.model_dump(exclude_none=True)

        assert "send" in data
        assert "wait" in data
        assert "receive" in data
        assert "blocked" not in data
        assert "dns" not in data
        assert "connect" not in data
        assert "ssl" not in data

    def test_har_spec_field_names_preserved(self):
        """HAR spec uses camelCase field names (e.g., redirectURL, headersSize)."""
        response = HARResponse(
            status=200,
            statusText="OK",
            httpVersion="HTTP/1.1",
            content=HARContent(size=1024, mimeType="application/json"),
            redirectURL="",
            headersSize=-1,
            bodySize=1024,
        )

        data = response.model_dump()

        # HAR spec requires these exact field names
        assert "redirectURL" in data
        assert "headersSize" in data
        assert "bodySize" in data
        assert "httpVersion" in data
        assert "statusText" in data
