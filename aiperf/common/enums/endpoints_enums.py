# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums.base_enums import CaseInsensitiveStrEnum


class EndpointType(CaseInsensitiveStrEnum):
    """Endpoint types supported by AIPerf.

    Simple string enum for endpoint types. All metadata is retrieved dynamically
    from the registered endpoint classes via EndpointFactory.metadata().
    """

    CHAT = "chat"
    COMPLETIONS = "completions"
    EMBEDDINGS = "embeddings"
    RANKINGS = "rankings"


class TransportType(CaseInsensitiveStrEnum):
    """The various types of transports for an endpoint."""

    HTTP = "http"
    HTTP2 = "http2"
    GRPC = "grpc"
    IN_ENGINE = "in-engine"
