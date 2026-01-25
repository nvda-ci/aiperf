# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC

from aiperf.common import plugin_registry
from aiperf.common.config import ServiceConfig
from aiperf.common.mixins.aiperf_lifecycle_mixin import AIPerfLifecycleMixin
from aiperf.common.protocols import CommunicationProtocol


class CommunicationMixin(AIPerfLifecycleMixin, ABC):
    """Mixin to provide access to a CommunicationProtocol instance. This mixin should be inherited
    by any mixin that needs access to the communication layer to create Communication clients.
    """

    def __init__(self, service_config: ServiceConfig, **kwargs) -> None:
        super().__init__(service_config=service_config, **kwargs)
        self.service_config = service_config
        CommClass = plugin_registry.get_class(
            "communication", self.service_config.comm_config.comm_backend
        )
        # CommClass extends Singleton, so repeated calls return the same per-process instance
        self.comms: CommunicationProtocol = CommClass(
            config=self.service_config.comm_config
        )
        self.attach_child_lifecycle(self.comms)
