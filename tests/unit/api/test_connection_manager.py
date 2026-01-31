# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the WebSocket connection manager."""

from unittest.mock import AsyncMock

import orjson
import pytest

from aiperf.api.connection_manager import ConnectionManager


def make_mock_ws(closed: bool = False) -> AsyncMock:
    """Create a mock WebSocket with given closed state."""
    ws = AsyncMock()
    ws.closed = closed
    return ws


@pytest.fixture
def connection_manager() -> ConnectionManager:
    """Create a ConnectionManager instance for testing."""
    return ConnectionManager()


@pytest.fixture
def mock_websocket() -> AsyncMock:
    """Create a mock WebSocket connection."""
    return make_mock_ws()


class TestConnectionManager:
    """Tests for ConnectionManager."""

    def test_add_client(
        self, connection_manager: ConnectionManager, mock_websocket: AsyncMock
    ) -> None:
        """Test adding a client."""
        connection_manager.add_client("client1", mock_websocket)

        assert "client1" in connection_manager.clients
        assert connection_manager.get_client_count() == 1

    def test_remove_client(
        self, connection_manager: ConnectionManager, mock_websocket: AsyncMock
    ) -> None:
        """Test removing a client."""
        connection_manager.add_client("client1", mock_websocket)
        connection_manager.remove_client("client1")

        assert "client1" not in connection_manager.clients
        assert connection_manager.get_client_count() == 0

    def test_remove_nonexistent_client(
        self, connection_manager: ConnectionManager
    ) -> None:
        """Test removing a nonexistent client does not raise."""
        connection_manager.remove_client("nonexistent")
        assert connection_manager.get_client_count() == 0

    @pytest.mark.asyncio
    async def test_send_to_client(
        self, connection_manager: ConnectionManager, mock_websocket: AsyncMock
    ) -> None:
        """Test sending message to specific client."""
        connection_manager.add_client("client1", mock_websocket)

        result = await connection_manager.send_to_client("client1", {"test": "data"})

        assert result is True
        mock_websocket.send_str.assert_called_once_with(
            orjson.dumps({"test": "data"}).decode()
        )

    @pytest.mark.asyncio
    async def test_send_to_nonexistent_client(
        self, connection_manager: ConnectionManager
    ) -> None:
        """Test sending to nonexistent client returns False."""
        result = await connection_manager.send_to_client(
            "nonexistent", {"test": "data"}
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_send_to_closed_client(
        self, connection_manager: ConnectionManager
    ) -> None:
        """Test sending to closed client returns False."""
        ws = make_mock_ws(closed=True)
        connection_manager.add_client("client1", ws)

        result = await connection_manager.send_to_client("client1", {"test": "data"})

        assert result is False
        ws.send_str.assert_not_called()

    @pytest.mark.asyncio
    async def test_broadcast(self, connection_manager: ConnectionManager) -> None:
        """Test broadcasting to all clients."""
        ws1, ws2 = make_mock_ws(), make_mock_ws()
        connection_manager.add_client("client1", ws1)
        connection_manager.add_client("client2", ws2)

        await connection_manager.broadcast({"type": "broadcast"})

        ws1.send_str.assert_called_once_with(
            orjson.dumps({"type": "broadcast"}).decode()
        )
        ws2.send_str.assert_called_once_with(
            orjson.dumps({"type": "broadcast"}).decode()
        )

    @pytest.mark.asyncio
    async def test_broadcast_removes_failed_clients(
        self, connection_manager: ConnectionManager
    ) -> None:
        """Test that broadcast removes clients that fail to receive."""
        ws1 = make_mock_ws()
        ws2 = make_mock_ws()
        ws2.send_str.side_effect = Exception("Connection error")

        connection_manager.add_client("client1", ws1)
        connection_manager.add_client("client2", ws2)

        await connection_manager.broadcast({"type": "broadcast"})

        assert "client1" in connection_manager.clients
        assert "client2" not in connection_manager.clients

    @pytest.mark.asyncio
    async def test_close_all(self, connection_manager: ConnectionManager) -> None:
        """Test closing all connections."""
        ws1, ws2 = make_mock_ws(), make_mock_ws()
        connection_manager.add_client("client1", ws1)
        connection_manager.add_client("client2", ws2)

        await connection_manager.close_all()

        ws1.close.assert_called_once()
        ws2.close.assert_called_once()
        assert connection_manager.get_client_count() == 0

    def test_get_client_count(
        self, connection_manager: ConnectionManager, mock_websocket: AsyncMock
    ) -> None:
        """Test getting client count."""
        assert connection_manager.get_client_count() == 0

        connection_manager.add_client("client1", mock_websocket)
        assert connection_manager.get_client_count() == 1

        connection_manager.add_client("client2", make_mock_ws())
        assert connection_manager.get_client_count() == 2
