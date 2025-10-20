# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import EndpointType
from aiperf.common.plugins import AIPerfPluginManager
from aiperf.common.protocols import EndpointProtocol

_pm = AIPerfPluginManager()

_pm.register_lazy(
    EndpointProtocol,
    EndpointType.CHAT,
    module_path="aiperf.endpoints.openai_chat",
    class_name="ChatEndpoint",
)

_pm.register_lazy(
    EndpointProtocol,
    EndpointType.COMPLETIONS,
    module_path="aiperf.endpoints.openai_completions",
    class_name="CompletionsEndpoint",
)

_pm.register_lazy(
    EndpointProtocol,
    EndpointType.EMBEDDINGS,
    module_path="aiperf.endpoints.openai_embeddings",
    class_name="EmbeddingsEndpoint",
)

_pm.register_lazy(
    EndpointProtocol,
    EndpointType.RANKINGS,
    module_path="aiperf.endpoints.nim_rankings",
    class_name="RankingsEndpoint",
)
