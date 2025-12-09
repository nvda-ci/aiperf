# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import inspect
import socket
from dataclasses import dataclass
from typing import Any

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.environment import Environment

_logger = AIPerfLogger(__name__)


@dataclass(frozen=True)
class SocketDefaults:
    """
    Default values for socket options.
    """

    TCP_NODELAY = 1  # Disable Nagle's algorithm
    TCP_QUICKACK = 1  # Quick ACK mode
    SO_KEEPALIVE = 1  # Enable keepalive
    SO_LINGER = 0  # Disable linger
    SO_REUSEADDR = 1  # Enable reuse address
    SO_REUSEPORT = 1  # Enable reuse port

    @classmethod
    def apply_to_socket(cls, sock: socket.socket) -> None:
        """Apply the default socket options to the given socket."""

        # Low-latency optimizations for streaming
        sock.setsockopt(socket.SOL_TCP, socket.TCP_NODELAY, cls.TCP_NODELAY)

        # Connection keepalive settings for long-lived SSE connections
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, cls.SO_KEEPALIVE)

        # Fine-tune keepalive timing (Linux-specific)
        if hasattr(socket, "TCP_KEEPIDLE"):
            sock.setsockopt(
                socket.SOL_TCP, socket.TCP_KEEPIDLE, Environment.HTTP.TCP_KEEPIDLE
            )
            sock.setsockopt(
                socket.SOL_TCP, socket.TCP_KEEPINTVL, Environment.HTTP.TCP_KEEPINTVL
            )
            sock.setsockopt(
                socket.SOL_TCP, socket.TCP_KEEPCNT, Environment.HTTP.TCP_KEEPCNT
            )

        # Buffer size optimizations for streaming
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, Environment.HTTP.SO_RCVBUF)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, Environment.HTTP.SO_SNDBUF)

        # Linux-specific TCP optimizations
        if hasattr(socket, "TCP_QUICKACK"):
            sock.setsockopt(socket.SOL_TCP, socket.TCP_QUICKACK, cls.TCP_QUICKACK)

        if hasattr(socket, "TCP_USER_TIMEOUT"):
            sock.setsockopt(
                socket.SOL_TCP,
                socket.TCP_USER_TIMEOUT,
                Environment.HTTP.TCP_USER_TIMEOUT,
            )


@dataclass(frozen=True)
class AioHttpDefaults:
    """Default values for aiohttp.ClientSession."""

    LIMIT = (
        Environment.HTTP.CONNECTION_LIMIT
    )  # Maximum number of concurrent connections
    LIMIT_PER_HOST = (
        0  # Maximum number of concurrent connections per host (0 will set to LIMIT)
    )
    TTL_DNS_CACHE = Environment.HTTP.TTL_DNS_CACHE  # Time to live for DNS cache
    USE_DNS_CACHE = True  # Enable DNS cache
    ENABLE_CLEANUP_CLOSED = False  # Disable cleanup of closed connections
    FORCE_CLOSE = False  # Disable force close connections
    KEEPALIVE_TIMEOUT = Environment.HTTP.KEEPALIVE_TIMEOUT  # Keepalive timeout
    HAPPY_EYEBALLS_DELAY = None  # Happy eyeballs delay (None = disabled)
    SOCKET_FAMILY = socket.AF_INET  # Family of the socket (IPv4)

    @staticmethod
    def _get_supported_tcp_connector_params() -> set[str]:
        """Get the set of parameter names supported by TCPConnector."""
        from aiohttp import TCPConnector

        sig = inspect.signature(TCPConnector.__init__)
        return set(sig.parameters.keys())

    @staticmethod
    def supports_tcp_connector_param(param_name: str) -> bool:
        """Check if TCPConnector supports a given parameter."""
        return param_name in AioHttpDefaults._get_supported_tcp_connector_params()

    @staticmethod
    def _filter_supported_tcp_connector_kwargs(
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Filter kwargs to only include parameters supported by TCPConnector.

        This handles aiohttp version compatibility by inspecting the TCPConnector
        signature and filtering out unsupported parameters.
        """
        supported_params = AioHttpDefaults._get_supported_tcp_connector_params()

        unsupported_params = set(kwargs.keys()) - supported_params
        if unsupported_params:
            _logger.warning(
                f"This version of aiohttp does not support the following parameters: {unsupported_params}"
            )

        return {k: v for k, v in kwargs.items() if k in supported_params}

    @classmethod
    def get_default_kwargs(cls) -> dict[str, Any]:
        """Get the default keyword arguments for aiohttp.TCPConnector.

        Returns only parameters supported by the installed aiohttp version.
        """
        kwargs: dict[str, Any] = {
            "limit": cls.LIMIT,
            "limit_per_host": cls.LIMIT_PER_HOST,
            "ttl_dns_cache": cls.TTL_DNS_CACHE,
            "use_dns_cache": cls.USE_DNS_CACHE,
            "enable_cleanup_closed": cls.ENABLE_CLEANUP_CLOSED,
            "force_close": cls.FORCE_CLOSE,
            "keepalive_timeout": cls.KEEPALIVE_TIMEOUT,
            "happy_eyeballs_delay": cls.HAPPY_EYEBALLS_DELAY,  # Added in aiohttp 3.9.0
            "family": cls.SOCKET_FAMILY,
        }
        return cls._filter_supported_tcp_connector_kwargs(kwargs)
