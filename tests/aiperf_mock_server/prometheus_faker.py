# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prometheus faker with base class and subclasses for different metric types.

This module provides a factory function for backward compatibility while the actual
implementations are split across multiple files.
"""

# Import base classes and constants
from aiperf_mock_server.prometheus_base import (
    DURATION_BUCKETS,
    ITL_BUCKETS,
    LATENCY_BUCKETS,
    OUTPUT_TOKEN_BUCKETS,
    SERVER_CONFIGS,
    SUMMARY_QUANTILES,
    TOKEN_BUCKETS,
    PrometheusMetricsFaker,
    ServerConfig,
    ServerState,
)

# Import individual faker implementations
from aiperf_mock_server.prometheus_dynamo_component import (
    DynamoComponentMetricsFaker,
    DynamoComponentServer,
)
from aiperf_mock_server.prometheus_dynamo_frontend import (
    DynamoFrontendMetricsFaker,
    DynamoFrontendServer,
)
from aiperf_mock_server.prometheus_generic import (
    GenericMetricsFaker,
    GenericServer,
)
from aiperf_mock_server.prometheus_kvbm import KVBMMetricsFaker, KVBMServer
from aiperf_mock_server.prometheus_sglang import (
    SGLangMetricsFaker,
    SGLangServer,
)
from aiperf_mock_server.prometheus_triton import (
    TritonMetricsFaker,
    TritonServer,
)
from aiperf_mock_server.prometheus_vllm import VLLMMetricsFaker, VLLMServer

# Export all classes and constants for backward compatibility
__all__ = [
    # Base classes and constants
    "PrometheusMetricsFaker",
    "ServerConfig",
    "ServerState",
    "SERVER_CONFIGS",
    "DURATION_BUCKETS",
    "LATENCY_BUCKETS",
    "ITL_BUCKETS",
    "TOKEN_BUCKETS",
    "OUTPUT_TOKEN_BUCKETS",
    "SUMMARY_QUANTILES",
    # Generic
    "GenericServer",
    "GenericMetricsFaker",
    # vLLM
    "VLLMServer",
    "VLLMMetricsFaker",
    # Triton
    "TritonServer",
    "TritonMetricsFaker",
    # SGLang
    "SGLangServer",
    "SGLangMetricsFaker",
    # KVBM
    "KVBMServer",
    "KVBMMetricsFaker",
    # Dynamo Frontend
    "DynamoFrontendServer",
    "DynamoFrontendMetricsFaker",
    # Dynamo Component
    "DynamoComponentServer",
    "DynamoComponentMetricsFaker",
    # Factory function
    "PrometheusFaker",
]


# =============================================================================
# Factory function for backward compatibility
# =============================================================================


def PrometheusFaker(
    server_type: str = "medium",
    num_servers: int = 2,
    seed: int | None = None,
    initial_load: float = 0.7,
    metric_type: str = "generic",
) -> PrometheusMetricsFaker:
    """Factory function to create the appropriate metrics faker.

    Args:
        server_type: Type of server config (small, medium, large, vllm, dynamo, triton, sglang, kvbm)
        num_servers: Number of server instances to simulate
        seed: Random seed for reproducibility
        initial_load: Initial load level (0.0=idle, 1.0=max)
        metric_type: Type of metrics to generate (generic, vllm, triton, sglang, kvbm, dynamo_frontend, dynamo_component)
                     Note: "dynamo" is an alias for "dynamo_frontend" for backward compatibility

    Returns:
        Appropriate PrometheusMetricsFaker subclass instance
    """
    metric_types = {
        "generic": GenericMetricsFaker,
        "vllm": VLLMMetricsFaker,
        "triton": TritonMetricsFaker,
        "sglang": SGLangMetricsFaker,
        "kvbm": KVBMMetricsFaker,
        "dynamo": DynamoFrontendMetricsFaker,  # Default to frontend for backward compatibility
        "dynamo_frontend": DynamoFrontendMetricsFaker,
        "dynamo_component": DynamoComponentMetricsFaker,
    }

    if metric_type not in metric_types:
        raise ValueError(
            f"Invalid metric type: {metric_type}. Must be one of {list(metric_types.keys())}"
        )

    faker_class = metric_types[metric_type]
    return faker_class(server_type, num_servers, seed, initial_load)
