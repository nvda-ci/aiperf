# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""WebSocket connection management for multiple clients.

Provides management of concurrent WebSocket client connections with
concurrent broadcast via asyncio.gather and orjson serialization.
"""

import asyncio
from typing import Any

import aiohttp.web
import orjson

from aiperf.common.aiperf_logger import AIPerfLogger


class ConnectionManager:
    """Manages multiple WebSocket client connections.

    State management methods are synchronous (atomic in asyncio).
    I/O methods are async with concurrent fan-out via gather.
    """

    def __init__(self) -> None:
        """Initialize the connection manager."""
        self.clients: dict[str, aiohttp.web.WebSocketResponse] = {}
        self.logger = AIPerfLogger(self.__class__.__name__)

    def add_client(
        self, client_id: str, websocket: aiohttp.web.WebSocketResponse
    ) -> None:
        """Add a new WebSocket client."""
        self.clients[client_id] = websocket
        self.logger.info(
            f"Client {client_id} connected (total clients: {len(self.clients)})"
        )

    def remove_client(self, client_id: str) -> None:
        """Remove a WebSocket client."""
        if self.clients.pop(client_id, None) is not None:
            self.logger.info(
                f"Client {client_id} disconnected (total clients: {len(self.clients)})"
            )

    def get_client(self, client_id: str) -> aiohttp.web.WebSocketResponse | None:
        """Get a WebSocket client by ID."""
        return self.clients.get(client_id)

    def get_client_count(self) -> int:
        """Get the number of connected clients."""
        return len(self.clients)

    async def send_to_client(self, client_id: str, message: dict[str, Any]) -> bool:
        """Send a message to a specific client.

        Returns:
            True if the message was sent successfully, False otherwise.
        """
        ws = self.clients.get(client_id)
        if ws and not ws.closed:
            try:
                await ws.send_str(orjson.dumps(message).decode())
                return True
            except Exception as e:
                self.logger.warning(f"Failed to send to client {client_id}: {e}")
                self.remove_client(client_id)
        return False

    async def broadcast(self, message: dict[str, Any]) -> int:
        """Broadcast a message to all connected clients concurrently.

        Returns:
            Number of clients that received the message successfully.
        """
        if not self.clients:
            return 0

        results = await asyncio.gather(
            *(self.send_to_client(cid, message) for cid in list(self.clients.keys())),
            return_exceptions=True,
        )
        return sum(1 for r in results if r is True)

    async def close_all(self) -> None:
        """Close all WebSocket connections gracefully."""
        if not self.clients:
            return

        self.logger.info(f"Closing {len(self.clients)} WebSocket connections")

        async def close_one(ws: aiohttp.web.WebSocketResponse) -> None:
            if not ws.closed:
                await ws.close()

        await asyncio.gather(
            *(close_one(ws) for ws in self.clients.values()),
            return_exceptions=True,
        )
        self.clients.clear()
        self.logger.info("All WebSocket connections closed")
