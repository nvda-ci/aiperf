# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HAR (HTTP Archive) 1.2 format models for Chrome DevTools compatibility.

This module implements the complete HAR 1.2 specification as defined at:
https://w3c.github.io/web-performance/specs/HAR/Overview.html

The models follow AIPerf conventions using Pydantic with Field descriptions
and proper typing. All models are compatible with Chrome DevTools Network panel.
"""

from datetime import datetime, timezone
from typing import Literal

from pydantic import Field

from aiperf.common.models.base_models import AIPerfBaseModel


class HARNameValuePair(AIPerfBaseModel):
    """Generic name/value pair used for headers, cookies, and query parameters."""

    name: str = Field(
        ...,
        description="The name of the pair (e.g., header name, cookie name, parameter name).",
    )
    value: str = Field(
        ...,
        description="The value of the pair.",
    )
    comment: str | None = Field(
        default=None,
        description="A comment provided by the user or application.",
    )


class HARCookie(AIPerfBaseModel):
    """Cookie information as sent or received."""

    name: str = Field(
        ...,
        description="The name of the cookie.",
    )
    value: str = Field(
        ...,
        description="The value of the cookie.",
    )
    path: str | None = Field(
        default=None,
        description="The path pertaining to the cookie.",
    )
    domain: str | None = Field(
        default=None,
        description="The host of the cookie.",
    )
    expires: str | None = Field(
        default=None,
        description="Cookie expiration time in ISO 8601 format. Omit for session cookies.",
    )
    httpOnly: bool | None = Field(  # noqa: N815
        default=None,
        description="Set to true if the cookie is HTTP only, false otherwise.",
    )
    secure: bool | None = Field(
        default=None,
        description="True if the cookie was transmitted over SSL, false otherwise.",
    )
    comment: str | None = Field(
        default=None,
        description="A comment provided by the user or application.",
    )


class HARQueryString(AIPerfBaseModel):
    """Query string parameter in the URL."""

    name: str = Field(
        ...,
        description="The name of the query parameter.",
    )
    value: str = Field(
        ...,
        description="The value of the query parameter.",
    )
    comment: str | None = Field(
        default=None,
        description="A comment provided by the user or application.",
    )


class HARPostDataParam(AIPerfBaseModel):
    """Post data parameter (for application/x-www-form-urlencoded or multipart/form-data)."""

    name: str = Field(
        ...,
        description="Name of a posted parameter.",
    )
    value: str | None = Field(
        default=None,
        description="Value of a posted parameter or content of a posted file.",
    )
    fileName: str | None = Field(  # noqa: N815
        default=None,
        description="Name of a posted file.",
    )
    contentType: str | None = Field(  # noqa: N815
        default=None,
        description="Content type of a posted file.",
    )
    comment: str | None = Field(
        default=None,
        description="A comment provided by the user or application.",
    )


class HARPostData(AIPerfBaseModel):
    """Request POST data details."""

    mimeType: str = Field(  # noqa: N815
        ...,
        description="MIME type of posted data.",
    )
    params: list[HARPostDataParam] = Field(
        default_factory=list,
        description="List of posted parameters (parsed from text field when possible).",
    )
    text: str = Field(
        default="",
        description="Plain text posted data.",
    )
    comment: str | None = Field(
        default=None,
        description="A comment provided by the user or application.",
    )


class HARContent(AIPerfBaseModel):
    """Response content details."""

    size: int = Field(
        ...,
        description="Length of the returned content in bytes. "
        "Should equal the number of bytes in response.bodySize if available. "
        "Set to 0 in case of 304 (Not Modified).",
    )
    compression: int | None = Field(
        default=None,
        description="Number of bytes saved by compression. "
        "Calculated as (uncompressed size) - (actual transfer size).",
    )
    mimeType: str = Field(  # noqa: N815
        ...,
        description="MIME type of the response text (value of the Content-Type response header). "
        "Includes charset if available.",
    )
    text: str | None = Field(
        default=None,
        description="Response body sent from the server or loaded from the browser cache. "
        "This field is populated with textual content only. "
        "Binary content is encoded using base64 encoding and the encoding field is set.",
    )
    encoding: str | None = Field(
        default=None,
        description="Encoding used for response text field, e.g., 'base64'. "
        "Omit if text is HTTP response decoded and trans-coded to UTF-8.",
    )
    comment: str | None = Field(
        default=None,
        description="A comment provided by the user or application.",
    )


class HARRequest(AIPerfBaseModel):
    """HTTP request details."""

    method: str = Field(
        ...,
        description="Request method (GET, POST, PUT, DELETE, etc.).",
    )
    url: str = Field(
        ...,
        description="Absolute URL of the request (fragments are not included).",
    )
    httpVersion: str = Field(  # noqa: N815
        ...,
        description="Request HTTP version (e.g., 'HTTP/1.1', 'HTTP/2').",
    )
    cookies: list[HARCookie] = Field(
        default_factory=list,
        description="List of cookie objects sent with the request.",
    )
    headers: list[HARNameValuePair] = Field(
        default_factory=list,
        description="List of header objects sent with the request.",
    )
    queryString: list[HARQueryString] = Field(  # noqa: N815
        default_factory=list,
        description="List of query parameter objects parsed from URL.",
    )
    postData: HARPostData | None = Field(  # noqa: N815
        default=None,
        description="Posted data info. Only present for POST, PUT, PATCH methods.",
    )
    headersSize: int = Field(  # noqa: N815
        ...,
        description="Total number of bytes from the start of the HTTP request message "
        "until (and including) the double CRLF before the body. "
        "Set to -1 if the info is not available.",
    )
    bodySize: int = Field(  # noqa: N815
        ...,
        description="Size of the request body (POST data payload) in bytes. "
        "Set to -1 if the info is not available.",
    )
    comment: str | None = Field(
        default=None,
        description="A comment provided by the user or application.",
    )


class HARResponse(AIPerfBaseModel):
    """HTTP response details."""

    status: int = Field(
        ...,
        description="Response status code (e.g., 200, 404, 500).",
    )
    statusText: str = Field(  # noqa: N815
        ...,
        description="Response status description (e.g., 'OK', 'Not Found', 'Internal Server Error').",
    )
    httpVersion: str = Field(  # noqa: N815
        ...,
        description="Response HTTP version (e.g., 'HTTP/1.1', 'HTTP/2').",
    )
    cookies: list[HARCookie] = Field(
        default_factory=list,
        description="List of cookie objects received in the response.",
    )
    headers: list[HARNameValuePair] = Field(
        default_factory=list,
        description="List of header objects received in the response.",
    )
    content: HARContent = Field(
        ...,
        description="Details about the response body.",
    )
    redirectURL: str = Field(  # noqa: N815
        default="",
        description="Redirection target URL from the Location response header.",
    )
    headersSize: int = Field(  # noqa: N815
        ...,
        description="Total number of bytes from the start of the HTTP response message "
        "until (and including) the double CRLF before the body. "
        "Set to -1 if the info is not available.",
    )
    bodySize: int = Field(  # noqa: N815
        ...,
        description="Size of the received response body in bytes. "
        "Set to zero in case of responses coming from the cache (304). "
        "Set to -1 if the info is not available.",
    )
    comment: str | None = Field(
        default=None,
        description="A comment provided by the user or application.",
    )


class HARCacheEntry(AIPerfBaseModel):
    """Cache entry state before or after request."""

    expires: str | None = Field(
        default=None,
        description="Expiration time of the cache entry in ISO 8601 format.",
    )
    lastAccess: str = Field(  # noqa: N815
        ...,
        description="The last time the cache entry was accessed in ISO 8601 format.",
    )
    eTag: str = Field(  # noqa: N815
        ...,
        description="Entity tag identifier for the cached entry.",
    )
    hitCount: int = Field(  # noqa: N815
        ...,
        description="Number of times the cache entry has been accessed.",
    )
    comment: str | None = Field(
        default=None,
        description="A comment provided by the user or application.",
    )


class HARCache(AIPerfBaseModel):
    """Cache usage information for a request/response pair."""

    beforeRequest: HARCacheEntry | None = Field(  # noqa: N815
        default=None,
        description="State of cache entry before the request. "
        "Leave null if the information is not available.",
    )
    afterRequest: HARCacheEntry | None = Field(  # noqa: N815
        default=None,
        description="State of cache entry after the request was completed. "
        "Leave null if the information is not available.",
    )
    comment: str | None = Field(
        default=None,
        description="A comment provided by the user or application.",
    )


class HARTimings(AIPerfBaseModel):
    """Timing information for different phases of the request/response cycle.

    All times are in milliseconds. Use -1 if the timing is not available.
    Per HAR spec: entry.time == sum of all timing phases (excluding -1 values).

    For SSL/TLS connections, the ssl time is included in the connect time
    for backward compatibility with HAR 1.1.
    """

    blocked: float | None = Field(
        default=None,
        description="Time spent in a queue waiting for a network connection. "
        "Use -1 if the timing is not available.",
    )
    dns: float | None = Field(
        default=None,
        description="DNS resolution time. The time required to resolve a host name. "
        "Use -1 if the timing is not available.",
    )
    connect: float | None = Field(
        default=None,
        description="Time required to create TCP connection. "
        "For HTTPS, this includes SSL/TLS handshake time. "
        "Use -1 if the timing is not available.",
    )
    send: float = Field(
        ...,
        description="Time required to send HTTP request to the server.",
    )
    wait: float = Field(
        ...,
        description="Waiting for a response from the server (TTFB - Time To First Byte).",
    )
    receive: float = Field(
        ...,
        description="Time required to read entire response from the server (or cache).",
    )
    ssl: float | None = Field(
        default=None,
        description="Time required for SSL/TLS negotiation. "
        "If this field is defined, the time is also included in the connect field "
        "to ensure backward compatibility with HAR 1.1. "
        "Use -1 if the timing is not available.",
    )
    comment: str | None = Field(
        default=None,
        description="A comment provided by the user or application.",
    )


class HAREntry(AIPerfBaseModel):
    """A single HTTP request/response pair (transaction).

    Entries should be sorted by startedDateTime (oldest first) for optimal
    import performance in analysis tools.
    """

    pageref: str | None = Field(
        default=None,
        description="Reference to the parent page. Leave null if the application does not support grouping by pages.",
    )
    startedDateTime: str = Field(  # noqa: N815
        ...,
        description="Date and time stamp of the request start in ISO 8601 format (YYYY-MM-DDThh:mm:ss.sTZD).",
    )
    time: float = Field(
        ...,
        description="Total elapsed time of the request in milliseconds. "
        "This is the sum of all timings available in the timings object "
        "(excluding any -1 values).",
    )
    request: HARRequest = Field(
        ...,
        description="Detailed info about the request.",
    )
    response: HARResponse = Field(
        ...,
        description="Detailed info about the response.",
    )
    cache: HARCache = Field(
        ...,
        description="Info about cache usage.",
    )
    timings: HARTimings = Field(
        ...,
        description="Detailed timing info about request/response round trip.",
    )
    serverIPAddress: str | None = Field(  # noqa: N815
        default=None,
        description="IP address of the server that was connected (result of DNS resolution).",
    )
    connection: str | None = Field(
        default=None,
        description="Unique ID of the parent TCP/IP connection. "
        "Can be the client port number or any unique identifier. "
        "Note: A page may download resources from multiple connections.",
    )
    comment: str | None = Field(
        default=None,
        description="A comment provided by the user or application.",
    )


class HARPageTimings(AIPerfBaseModel):
    """Timing information about page load events.

    All times are in milliseconds relative to page.startedDateTime.
    Use -1 if the timing is not available.
    """

    onContentLoad: float | None = Field(  # noqa: N815
        default=None,
        description="Number of milliseconds since page load started (page.startedDateTime) "
        "when the DOMContentLoad event is fired (document.readyState == 'interactive'). "
        "Use -1 if the timing is not applicable.",
    )
    onLoad: float | None = Field(  # noqa: N815
        default=None,
        description="Number of milliseconds since page load started (page.startedDateTime) "
        "when the load event is fired (document.readyState == 'complete'). "
        "Use -1 if the timing is not applicable.",
    )
    comment: str | None = Field(
        default=None,
        description="A comment provided by the user or application.",
    )


class HARPage(AIPerfBaseModel):
    """Page grouping for requests (represents a single page or application state).

    Pages are optional but useful for grouping related requests.
    """

    startedDateTime: str = Field(  # noqa: N815
        ...,
        description="Date and time stamp for the beginning of the page load in ISO 8601 format (YYYY-MM-DDThh:mm:ss.sTZD).",
    )
    id: str = Field(
        ...,
        description="Unique identifier of a page within the HAR log. "
        "Entries use this to reference their parent page.",
    )
    title: str = Field(
        ...,
        description="Page title.",
    )
    pageTimings: HARPageTimings = Field(  # noqa: N815
        ...,
        description="Detailed timing info about page load.",
    )
    comment: str | None = Field(
        default=None,
        description="A comment provided by the user or application.",
    )


class HARCreator(AIPerfBaseModel):
    """Information about the application that created the HAR log."""

    name: str = Field(
        ...,
        description="Name of the application that created the log.",
    )
    version: str = Field(
        ...,
        description="Version number of the application that created the log.",
    )
    comment: str | None = Field(
        default=None,
        description="A comment provided by the user or application.",
    )


class HARBrowser(AIPerfBaseModel):
    """Information about the browser used (if applicable)."""

    name: str = Field(
        ...,
        description="Name of the browser used.",
    )
    version: str = Field(
        ...,
        description="Version of the browser used.",
    )
    comment: str | None = Field(
        default=None,
        description="A comment provided by the user or application.",
    )


class HARLog(AIPerfBaseModel):
    """Root object of the HAR log containing all captured data."""

    version: Literal["1.2"] = Field(
        default="1.2",
        description="Version number of the HAR format. Currently always '1.2'.",
    )
    creator: HARCreator = Field(
        ...,
        description="Information about the application that created the log.",
    )
    browser: HARBrowser | None = Field(
        default=None,
        description="Information about the browser used to generate the log. "
        "Optional for non-browser applications.",
    )
    pages: list[HARPage] = Field(
        default_factory=list,
        description="List of pages tracked during the session. "
        "Optional - can be empty if the application does not support page grouping.",
    )
    entries: list[HAREntry] = Field(
        ...,
        description="List of all exported HTTP requests/responses. "
        "Should be sorted by startedDateTime (oldest first) for optimal import performance.",
    )
    comment: str | None = Field(
        default=None,
        description="A comment provided by the user or application.",
    )


class HAR(AIPerfBaseModel):
    """HTTP Archive (HAR) 1.2 format - root container.

    HAR files must be saved in UTF-8 encoding.

    Compatible with Chrome DevTools Network panel and other HAR analysis tools.
    See: https://w3c.github.io/web-performance/specs/HAR/Overview.html
    """

    log: HARLog = Field(
        ...,
        description="The root HAR log object containing all captured HTTP transactions.",
    )


def format_iso8601_timestamp(timestamp_ns: int) -> str:
    """Convert nanosecond timestamp to ISO 8601 format string.

    Args:
        timestamp_ns: Timestamp in nanoseconds (from time.time_ns())

    Returns:
        ISO 8601 formatted string with millisecond precision and timezone (YYYY-MM-DDThh:mm:ss.sssZ)
    """
    timestamp_s = timestamp_ns / 1_000_000_000
    dt = datetime.fromtimestamp(timestamp_s, tz=timezone.utc)
    # Format with millisecond precision
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def ns_to_ms(nanoseconds: int | None) -> float | None:
    """Convert nanoseconds to milliseconds.

    Args:
        nanoseconds: Time in nanoseconds, or None

    Returns:
        Time in milliseconds with fractional precision, or None if input is None
    """
    if nanoseconds is None:
        return None
    return nanoseconds / 1_000_000
