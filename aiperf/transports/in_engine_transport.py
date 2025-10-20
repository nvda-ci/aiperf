# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time
from typing import Any

from aiperf.common.enums import TransportType
from aiperf.common.factories import TransportFactory
from aiperf.common.mixins import AIPerfLifecycleMixin
from aiperf.common.models import ErrorDetails, RequestInfo, RequestRecord
from aiperf.transports.base_transports import BaseTransport, TransportMetadata


@TransportFactory.register(TransportType.IN_ENGINE)
class InEngineTransport(BaseTransport, AIPerfLifecycleMixin):
    """In-process inference (placeholder - implement with vLLM/transformers)."""

    @classmethod
    def metadata(cls) -> TransportMetadata:
        return TransportMetadata(
            transport_type=TransportType.IN_ENGINE,
            url_schemes=["file", "local"],
        )

    def get_url(self, request_info: RequestInfo) -> str:
        return request_info.model_endpoint.url

    async def send_request(
        self, request_info: RequestInfo, payload: dict[str, Any]
    ) -> RequestRecord:
        start_perf_ns = time.perf_counter_ns()
        try:
            raise NotImplementedError("In-engine transport placeholder")
        except Exception as e:
            return RequestRecord(
                start_perf_ns=start_perf_ns,
                end_perf_ns=time.perf_counter_ns(),
                error=ErrorDetails(type=e.__class__.__name__, message=str(e)),
            )
