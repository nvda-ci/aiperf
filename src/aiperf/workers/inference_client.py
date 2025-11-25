# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable

from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.factories import EndpointFactory, TransportFactory
from aiperf.common.mixins import AIPerfLifecycleMixin
from aiperf.common.models import (
    ErrorDetails,
    ModelEndpointInfo,
    RequestInfo,
    RequestRecord,
)


class InferenceClient(AIPerfLifecycleMixin):
    """Inference client for the worker."""

    def __init__(self, model_endpoint: ModelEndpointInfo, service_id: str, **kwargs):
        super().__init__(model_endpoint=model_endpoint, service_id=service_id, **kwargs)
        self.model_endpoint = model_endpoint
        self.service_id = service_id

        # Detect and set transport type if not explicitly set
        if not model_endpoint.transport:
            model_endpoint.transport = TransportFactory.detect_from_url(
                model_endpoint.endpoint.base_url
            )
            if not model_endpoint.transport:
                raise ValueError(
                    f"No transport found for URL: {model_endpoint.endpoint.base_url}"
                )

        # Create endpoint and transport instances
        self.endpoint = EndpointFactory.create_instance(
            self.model_endpoint.endpoint.type,
            model_endpoint=self.model_endpoint,
        )
        self.transport = TransportFactory.create_instance(
            self.model_endpoint.transport,
            model_endpoint=self.model_endpoint,
        )
        self.attach_child_lifecycle(self.transport)

    async def _send_request_to_transport(
        self, request_info: RequestInfo
    ) -> RequestRecord:
        """Send request via transport.

        Handles the complete request lifecycle:
        1. Populates endpoint headers and params on request_info
        2. Formats the payload using the endpoint
        3. Sends the request via the transport

        Args:
            request_info: The request information.

        Returns:
            RequestRecord containing the response data and metadata.
        """
        request_info.endpoint_headers = self.endpoint.get_endpoint_headers(request_info)
        request_info.endpoint_params = self.endpoint.get_endpoint_params(request_info)
        formatted_payload = self.endpoint.format_payload(request_info)
        # TODO: THE TRANSPORT SHOULD HANDLE CANCELLATION, NOT THE INFERENCE CLIENT.
        return await self.transport.send_request(
            request_info, payload=formatted_payload
        )

    async def _send_request_internal(self, request_info: RequestInfo) -> RequestRecord:
        pre_send_perf_ns, pre_send_timestamp_ns = None, None
        try:
            # Save the current perf_ns before sending the request so it can be used to calculate
            # the start_perf_ns of the request in case of an exception.
            pre_send_perf_ns, pre_send_timestamp_ns = (
                time.perf_counter_ns(),
                time.time_ns(),
            )

            send_coroutine = self._send_request_to_transport(
                request_info=request_info,
            )

            maybe_result: RequestRecord | None = await self._send_with_optional_cancel(
                send_coroutine=send_coroutine,
                cancel_after_ns=request_info.cancel_after_ns,
            )

            # TODO: this needs triple checking after the refactor!
            if maybe_result is not None:
                result = maybe_result
                if self.is_debug_enabled:
                    self.debug(
                        f"pre_send_perf_ns to start_perf_ns latency: {result.start_perf_ns - pre_send_perf_ns} ns"
                    )
                result.turns = request_info.turns
                return result
            else:
                cancellation_perf_ns = time.perf_counter_ns()
                if self.is_debug_enabled:
                    delay_s = request_info.cancel_after_ns / NANOS_PER_SECOND
                    self.debug(f"Request cancelled after {delay_s:.3f}s")

                return RequestRecord(
                    request_info=request_info,
                    turns=request_info.turns,
                    timestamp_ns=pre_send_timestamp_ns,
                    start_perf_ns=pre_send_perf_ns,
                    end_perf_ns=cancellation_perf_ns,
                    cancellation_perf_ns=cancellation_perf_ns,
                    error=ErrorDetails(
                        type="RequestCancellationError",
                        message=(
                            f"Request was cancelled after "
                            f"{request_info.cancel_after_ns / NANOS_PER_SECOND:.3f} seconds"
                        ),
                        code=499,  # Client Closed Request
                    ),
                )
        except Exception as e:
            self.error(
                f"Error calling inference server API at {self.model_endpoint.endpoint.base_url}: {e!r}"
            )
            return RequestRecord(
                request_info=request_info,
                turns=request_info.turns,
                timestamp_ns=pre_send_timestamp_ns or time.time_ns(),
                # Try and use the pre_send_perf_ns if it is available, otherwise use the current time.
                start_perf_ns=pre_send_perf_ns or time.perf_counter_ns(),
                end_perf_ns=time.perf_counter_ns(),
                error=ErrorDetails.from_exception(e),
            )

    async def send_request(self, request_info: RequestInfo) -> RequestRecord:
        """Send a request to the inference API. Will return an error record if the call fails."""
        if self.is_trace_enabled:
            self.trace(
                f"Calling inference API for turn: {request_info.turns[request_info.turn_index]}"
            )
        record = await self._send_request_internal(request_info)
        return self._enrich_request_record(record=record, request_info=request_info)

    async def _send_with_optional_cancel(
        self,
        *,
        send_coroutine: Awaitable[RequestRecord],
        cancel_after_ns: int | None,
    ) -> RequestRecord | None:
        """Send a coroutine with optional cancellation after a delay.
        Args:
            send_coroutine: The coroutine object to send.
            cancel_after_ns: The delay in nanoseconds after which to cancel the request. If None, no cancellation is performed.
        Returns:
            The result of the send_coroutine, or None if it was cancelled.
        """
        if cancel_after_ns is None:
            return await send_coroutine

        timeout_s = cancel_after_ns / NANOS_PER_SECOND
        try:
            return await asyncio.wait_for(send_coroutine, timeout=timeout_s)
        except asyncio.TimeoutError:
            return None

    def _enrich_request_record(
        self,
        *,
        record: RequestRecord,
        request_info: RequestInfo,
    ) -> RequestRecord:
        """Enrich a RequestRecord with the original request info."""
        record.model_name = (
            request_info.turns[request_info.turn_index].model
            or self.model_endpoint.primary_model_name
        )
        record.request_info = request_info
        # If this is the first turn, calculate the credit drop latency
        if request_info.turn_index == 0 and request_info.drop_perf_ns is not None:
            record.credit_drop_latency = (
                record.start_perf_ns - request_info.drop_perf_ns
            )
        # Preserve headers set by transport; only use endpoint headers if not set
        if record.request_headers is None:
            record.request_headers = request_info.endpoint_headers
        return record
