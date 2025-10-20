# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time
from typing import Any

from aiperf.common.enums import TransportType
from aiperf.common.factories import TransportFactory
from aiperf.common.mixins import AIPerfLifecycleMixin
from aiperf.common.models import ErrorDetails, RequestInfo, RequestRecord
from aiperf.transports.base_transports import BaseTransport, TransportMetadata


@TransportFactory.register(TransportType.HTTP2)
class Http2Transport(BaseTransport, AIPerfLifecycleMixin):
    """HTTP/2 transport (placeholder - implement with httpx)."""

    @classmethod
    def metadata(cls) -> TransportMetadata:
        return TransportMetadata(
            transport_type=TransportType.HTTP2,
            url_schemes=["http2"],
        )

    def get_transport_headers(self, request_info: RequestInfo) -> dict[str, str]:
        accept = (
            "text/event-stream"
            if request_info.model_endpoint.endpoint.streaming
            else "application/json"
        )
        return {"Content-Type": "application/json", "Accept": accept}

    def get_url(self, request_info: RequestInfo) -> str:
        url = request_info.model_endpoint.url
        if not url.startswith("http"):
            url = f"https://{url}"
        return url

    async def send_request(
        self, request_info: RequestInfo, payload: dict[str, Any]
    ) -> RequestRecord:
        start_perf_ns = time.perf_counter_ns()
        try:
            raise NotImplementedError("HTTP/2 transport placeholder")
        except Exception as e:
            return RequestRecord(
                start_perf_ns=start_perf_ns,
                end_perf_ns=time.perf_counter_ns(),
                error=ErrorDetails(type=e.__class__.__name__, message=str(e)),
            )
