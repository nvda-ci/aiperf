# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import socket
from dataclasses import dataclass

from aiperf.common import constants


@dataclass(frozen=True)
class SocketDefaults:
    """
    Default values for socket options.
    """

    TCP_NODELAY = 1  # Disable Nagle's algorithm
    TCP_QUICKACK = 1  # Quick ACK mode

    SO_KEEPALIVE = 1  # Enable keepalive
    TCP_KEEPIDLE = 60  # Start keepalive after 1 min idle
    TCP_KEEPINTVL = 30  # Keepalive interval: 30 seconds
    TCP_KEEPCNT = 1  # 1 failed keepalive probes = dead

    SO_LINGER = 0  # Disable linger
    SO_REUSEADDR = 1  # Enable reuse address
    SO_REUSEPORT = 1  # Enable reuse port

    SO_RCVBUF = 1024 * 1024 * 10  # 10MB receive buffer
    SO_SNDBUF = 1024 * 1024 * 10  # 10MB send buffer

    SO_RCVTIMEO = 30  # 30 second receive timeout
    SO_SNDTIMEO = 30  # 30 second send timeout
    TCP_USER_TIMEOUT = 30000  # 30 sec user timeout

    @classmethod
    def build_socket_options(cls) -> list[tuple[int, int, int]]:
        """Build socket options as a list of tuples (level, optname, value).

        The options are optimized for low-latency streaming with large buffers
        and platform-specific TCP optimizations.
        """
        socket_options = [
            # Disable Nagle's algorithm for low-latency streaming
            (socket.SOL_TCP, socket.TCP_NODELAY, cls.TCP_NODELAY),
            # Enable socket keepalive for long-lived connections
            (socket.SOL_SOCKET, socket.SO_KEEPALIVE, cls.SO_KEEPALIVE),
            # Large buffers for high-throughput streaming
            (socket.SOL_SOCKET, socket.SO_RCVBUF, cls.SO_RCVBUF),
            (socket.SOL_SOCKET, socket.SO_SNDBUF, cls.SO_SNDBUF),
        ]

        # Linux-specific TCP optimizations
        if hasattr(socket, "TCP_QUICKACK"):
            socket_options.append(
                (socket.SOL_TCP, socket.TCP_QUICKACK, cls.TCP_QUICKACK)
            )

        if hasattr(socket, "TCP_KEEPIDLE"):
            socket_options.extend(
                [
                    (socket.SOL_TCP, socket.TCP_KEEPIDLE, cls.TCP_KEEPIDLE),
                    (socket.SOL_TCP, socket.TCP_KEEPINTVL, cls.TCP_KEEPINTVL),
                    (socket.SOL_TCP, socket.TCP_KEEPCNT, cls.TCP_KEEPCNT),
                ]
            )

        if hasattr(socket, "TCP_USER_TIMEOUT"):
            socket_options.append(
                (socket.SOL_TCP, socket.TCP_USER_TIMEOUT, cls.TCP_USER_TIMEOUT)
            )

        return socket_options

    @classmethod
    def apply_to_socket(cls, sock: socket.socket) -> None:
        """Apply the default socket options to the given socket."""
        socket_options = cls.build_socket_options()

        for option_level, option_name, option_value in socket_options:
            sock.setsockopt(option_level, option_name, option_value)


@dataclass(frozen=True)
class AioHttpDefaults:
    """Default values for aiohttp.ClientSession."""

    LIMIT = (
        constants.AIPERF_HTTP_CONNECTION_LIMIT
    )  # Maximum number of concurrent connections
    LIMIT_PER_HOST = (
        0  # Maximum number of concurrent connections per host (0 will set to LIMIT)
    )
    TTL_DNS_CACHE = 300  # Time to live for DNS cache
    USE_DNS_CACHE = True  # Enable DNS cache
    ENABLE_CLEANUP_CLOSED = False  # Disable cleanup of closed connections
    FORCE_CLOSE = False  # Disable force close connections
    KEEPALIVE_TIMEOUT = 300  # Keepalive timeout
    HAPPY_EYEBALLS_DELAY = None  # Happy eyeballs delay (None = disabled)
    SOCKET_FAMILY = socket.AF_INET  # Family of the socket (IPv4)


@dataclass(frozen=True)
class HttpCoreDefaults:
    """Default values for httpcore.AsyncConnectionPool with HTTP/2 support."""

    HTTP1 = True  # Enable HTTP/1.1 support
    HTTP2 = True  # Enable HTTP/2 support
    STREAMS_PER_CONNECTION = 100  # Approx concurrent streams per HTTP/2 connection
    KEEPALIVE_EXPIRY = 300.0  # Keep connections alive for 5 minutes
    RETRIES = 0  # No automatic retries (handled at higher level)
    MIN_CONNECTIONS = 10  # Minimum number of connections to maintain in the pool

    @classmethod
    def calculate_max_connections(cls) -> int:
        """Calculate maximum connections based on target concurrency.

        Strategy: AIPERF_HTTP_CONNECTION_LIMIT / STREAMS_PER_CONNECTION
        Default: 2500 / 100 = 25 connections

        HTTP/2 supports ~100 concurrent streams per connection (server-negotiated
        via SETTINGS_MAX_CONCURRENT_STREAMS).

        Returns:
            Number of HTTP/2 connections to maintain in the pool
        """
        return max(
            cls.MIN_CONNECTIONS,
            constants.AIPERF_HTTP_CONNECTION_LIMIT // cls.STREAMS_PER_CONNECTION,
        )
