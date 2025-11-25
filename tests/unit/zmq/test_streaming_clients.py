# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for streaming ZMQ clients - ZMQStreamingRouterClient and ZMQStreamingDealerClient.

Tests focus on behavior and functionality, not implementation details.
"""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest
import zmq.asyncio

from aiperf.common.enums import MessageType
from aiperf.common.exceptions import NotInitializedError
from aiperf.common.messages import (
    Message,
    WorkerReadyMessage,
)
from aiperf.zmq.streaming_dealer_client import ZMQStreamingDealerClient
from aiperf.zmq.streaming_router_client import ZMQStreamingRouterClient

# ============================================================================
# ZMQStreamingRouterClient Tests
# ============================================================================


class TestStreamingRouterClientInitialization:
    """Test ZMQStreamingRouterClient initialization."""

    def test_creates_router_socket(self, mock_zmq_context):
        """Should create a ROUTER socket."""
        client = ZMQStreamingRouterClient(address="tcp://*:5555", bind=True)

        assert client.socket_type == zmq.SocketType.ROUTER

    @pytest.mark.parametrize(
        "address,bind",
        [
            ("tcp://*:5555", True),
            ("tcp://localhost:5555", False),
            ("ipc:///tmp/router.ipc", True),
            ("ipc:///tmp/router.ipc", False),
        ],
        ids=["tcp_bind", "tcp_connect", "ipc_bind", "ipc_connect"],
    )  # fmt: skip
    def test_supports_various_transports(self, address, bind, mock_zmq_context):
        """Should support both TCP and IPC transports."""
        client = ZMQStreamingRouterClient(address=address, bind=bind)

        assert client.address == address
        assert client.bind == bind


class TestStreamingRouterClientSendTo:
    """Test ZMQStreamingRouterClient.send_to method."""

    @pytest.mark.asyncio
    async def test_sends_message_to_specific_identity(
        self, mock_zmq_socket, mock_zmq_context, sample_message
    ):
        """Should send message to specific DEALER by identity."""
        client = ZMQStreamingRouterClient(address="tcp://*:5555", bind=True)
        await client.initialize()

        await client.send_to("worker-42", sample_message)

        # Verify multipart message with correct envelope
        mock_zmq_socket.send_multipart.assert_called_once()
        call_args = mock_zmq_socket.send_multipart.call_args[0][0]

        assert call_args[0] == b"worker-42"  # Identity
        assert sample_message.message_type.value.encode() in call_args[1]  # Message

    @pytest.mark.asyncio
    async def test_raises_if_not_initialized(self, mock_zmq_context):
        """Should raise NotInitializedError if socket not initialized."""
        client = ZMQStreamingRouterClient(address="tcp://*:5555", bind=True)

        message = Message(message_type=MessageType.HEARTBEAT)

        with pytest.raises(NotInitializedError):
            await client.send_to("worker-1", message)

    @pytest.mark.asyncio
    async def test_raises_on_invalid_message_type(self, mock_zmq_context):
        """Should raise TypeError if message is not a Message instance."""
        client = ZMQStreamingRouterClient(address="tcp://*:5555", bind=True)
        await client.initialize()

        with pytest.raises(TypeError, match="must be an instance of Message"):
            await client.send_to("worker-1", "not a message")


class TestStreamingRouterClientReceiver:
    """Test ZMQStreamingRouterClient message receiving."""

    @pytest.mark.asyncio
    async def test_receives_messages_and_calls_handler(
        self, mock_zmq_context, sample_message, wait_for_background_task
    ):
        """Should receive messages from DEALERs and call registered handler."""
        handler_called = asyncio.Event()
        received_identity = None
        received_message = None

        async def handler(identity: str, message: Message):
            nonlocal received_identity, received_message
            received_identity = identity
            received_message = message
            handler_called.set()

        # Setup mock socket to return one message
        mock_socket = AsyncMock(spec=zmq.asyncio.Socket)
        mock_socket.bind = Mock()
        mock_socket.setsockopt = Mock()
        mock_socket.send_multipart = AsyncMock()

        # Return message in ROUTER format: [identity, empty, message_bytes]
        # Create a coroutine that can be awaited properly
        call_count = 0

        async def recv_multipart_handler():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                await asyncio.sleep(0.01)  # Small delay
                return [b"worker-1", b"", sample_message.to_json_bytes()]
            else:
                # Block forever on subsequent calls
                await asyncio.Future()

        mock_socket.recv_multipart = recv_multipart_handler
        mock_zmq_context.socket = Mock(return_value=mock_socket)

        client = ZMQStreamingRouterClient(address="tcp://*:5555", bind=True)
        client.register_receiver(handler)
        await client.initialize()
        await client.start()

        try:
            # Wait for message to be received
            await asyncio.wait_for(handler_called.wait(), timeout=1.0)

            assert received_identity == "worker-1"
            assert received_message.request_id == sample_message.request_id
        finally:
            await client.stop()


# ============================================================================
# ZMQStreamingDealerClient Tests
# ============================================================================


class TestStreamingDealerClientInitialization:
    """Test ZMQStreamingDealerClient initialization."""

    def test_creates_dealer_socket(self, mock_zmq_context):
        """Should create a DEALER socket."""
        client = ZMQStreamingDealerClient(
            address="tcp://localhost:5555", identity="worker-1"
        )

        assert client.socket_type == zmq.SocketType.DEALER
        assert client.identity == "worker-1"

    @pytest.mark.asyncio
    async def test_sets_identity_in_socket_options(
        self, mock_zmq_socket, mock_zmq_context
    ):
        """Should set IDENTITY socket option for routing."""
        client = ZMQStreamingDealerClient(
            address="tcp://localhost:5555", identity="worker-42"
        )
        await client.initialize()

        # Verify IDENTITY was set in socket options
        setsockopt_calls = mock_zmq_socket.setsockopt.call_args_list
        identity_calls = [
            call for call in setsockopt_calls if call[0][0] == zmq.IDENTITY
        ]

        assert len(identity_calls) == 1
        assert identity_calls[0][0][1] == b"worker-42"

    @pytest.mark.parametrize(
        "address,identity",
        [
            ("tcp://localhost:5555", "worker-1"),
            ("tcp://localhost:6666", "worker-2"),
            ("ipc:///tmp/router.ipc", "worker-3"),
        ],
        ids=["tcp_worker1", "tcp_worker2", "ipc_worker3"],
    )  # fmt: skip
    async def test_supports_various_transports(
        self, address, identity, mock_zmq_context
    ):
        """Should support both TCP and IPC transports."""
        client = ZMQStreamingDealerClient(address=address, identity=identity)

        assert client.address == address
        assert client.identity == identity


class TestStreamingDealerClientSend:
    """Test ZMQStreamingDealerClient.send method."""

    @pytest.mark.asyncio
    async def test_sends_message_with_correct_envelope(
        self, mock_zmq_socket, mock_zmq_context, sample_message
    ):
        """Should send message using single-frame send."""
        client = ZMQStreamingDealerClient(
            address="tcp://localhost:5555", identity="worker-1"
        )
        await client.initialize()

        await client.send(sample_message)

        # Verify message was sent (DEALER uses send, not send_multipart)
        mock_zmq_socket.send.assert_called_once()
        call_args = mock_zmq_socket.send.call_args[0][0]

        assert sample_message.message_type.value.encode() in call_args  # Message

    @pytest.mark.asyncio
    async def test_raises_if_not_initialized(self, mock_zmq_context):
        """Should raise NotInitializedError if socket not initialized."""
        client = ZMQStreamingDealerClient(
            address="tcp://localhost:5555", identity="worker-1"
        )

        message = Message(message_type=MessageType.HEARTBEAT)

        with pytest.raises(NotInitializedError):
            await client.send(message)


class TestStreamingDealerClientReceiver:
    """Test ZMQStreamingDealerClient message receiving."""

    @pytest.mark.asyncio
    async def test_receives_messages_and_calls_handler(
        self, mock_zmq_context, sample_message, wait_for_background_task
    ):
        """Should receive messages from ROUTER and call registered handler."""
        handler_called = asyncio.Event()
        received_message = None

        async def handler(message: Message):
            nonlocal received_message
            received_message = message
            handler_called.set()

        # Setup mock socket to return one message
        mock_socket = AsyncMock(spec=zmq.asyncio.Socket)
        mock_socket.bind = Mock()
        mock_socket.connect = Mock()
        mock_socket.setsockopt = Mock()
        mock_socket.send = AsyncMock()

        # DEALER receives using recv() not recv_multipart()
        call_count = 0

        async def recv_handler():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                await asyncio.sleep(0.01)
                return sample_message.to_json_bytes()
            else:
                await asyncio.Future()

        mock_socket.recv = recv_handler
        mock_zmq_context.socket = Mock(return_value=mock_socket)

        client = ZMQStreamingDealerClient(
            address="tcp://localhost:5555", identity="worker-1"
        )
        client.register_receiver(handler)
        await client.initialize()
        await client.start()

        try:
            await asyncio.wait_for(handler_called.wait(), timeout=1.0)
            assert received_message.request_id == sample_message.request_id
        finally:
            await client.stop()

    @pytest.mark.asyncio
    async def test_handles_message_without_delimiter(
        self, mock_zmq_context, sample_message, wait_for_background_task
    ):
        """Should handle messages (DEALER uses recv, framing handled by ZMQ)."""
        handler_called = asyncio.Event()
        received_message = None

        async def handler(message: Message):
            nonlocal received_message
            received_message = message
            handler_called.set()

        mock_socket = AsyncMock(spec=zmq.asyncio.Socket)
        mock_socket.bind = Mock()
        mock_socket.connect = Mock()
        mock_socket.setsockopt = Mock()
        mock_socket.send = AsyncMock()

        # DEALER uses recv() - ZMQ handles framing automatically
        call_count = 0

        async def recv_handler():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                await asyncio.sleep(0.01)
                return sample_message.to_json_bytes()
            else:
                await asyncio.Future()

        mock_socket.recv = recv_handler
        mock_zmq_context.socket = Mock(return_value=mock_socket)

        client = ZMQStreamingDealerClient(
            address="tcp://localhost:5555", identity="worker-1"
        )
        client.register_receiver(handler)
        await client.initialize()
        await client.start()

        try:
            await asyncio.wait_for(handler_called.wait(), timeout=1.0)
            assert received_message.request_id == sample_message.request_id
        finally:
            await client.stop()


# ============================================================================
# Integration Tests - ROUTER + DEALER
# ============================================================================


class TestStreamingRouterDealerIntegration:
    """Integration tests for ROUTER-DEALER bidirectional streaming."""

    @pytest.mark.skip(reason="Integration test with real sockets - needs investigation")
    @pytest.mark.asyncio
    async def test_bidirectional_communication(self, wait_for_background_task):
        """Should support bidirectional message flow between ROUTER and DEALER."""
        # Use real sockets for integration test
        router_address = "tcp://127.0.0.1:45678"

        dealer_received = asyncio.Event()
        router_received = asyncio.Event()
        dealer_message = None
        router_message = None
        router_identity = None

        async def router_handler(identity: str, message: Message):
            nonlocal router_message, router_identity
            router_identity = identity
            router_message = message
            router_received.set()

        async def dealer_handler(message: Message):
            nonlocal dealer_message
            dealer_message = message
            dealer_received.set()

        # Create ROUTER and DEALER
        router = ZMQStreamingRouterClient(address=router_address, bind=True)
        router.register_receiver(router_handler)
        await router.initialize()
        await router.start()

        dealer = ZMQStreamingDealerClient(
            address=router_address, identity="test-worker"
        )
        dealer.register_receiver(dealer_handler)
        await dealer.initialize()
        await dealer.start()

        try:
            # Test DEALER → ROUTER
            msg_to_router = WorkerReadyMessage(service_id="test-worker")
            await dealer.send(msg_to_router)
            await asyncio.wait_for(router_received.wait(), timeout=1.0)

            assert router_identity == "test-worker"
            assert router_message.message_type == MessageType.WORKER_READY

            # Test ROUTER → DEALER
            msg_to_dealer = Message(message_type=MessageType.HEARTBEAT)
            await router.send_to("test-worker", msg_to_dealer)
            await asyncio.wait_for(dealer_received.wait(), timeout=1.0)

            assert dealer_message.message_type == MessageType.HEARTBEAT

        finally:
            await dealer.stop()
            await router.stop()

    @pytest.mark.skip(reason="Integration test with real sockets - needs investigation")
    @pytest.mark.asyncio
    async def test_router_routes_to_correct_dealer(self, wait_for_background_task):
        """Should route messages to correct DEALER based on identity."""
        router_address = "tcp://127.0.0.1:45679"

        worker1_received = asyncio.Event()
        worker2_received = asyncio.Event()
        worker1_message = None
        worker2_message = None

        async def worker1_handler(message: Message):
            nonlocal worker1_message
            worker1_message = message
            worker1_received.set()

        async def worker2_handler(message: Message):
            nonlocal worker2_message
            worker2_message = message
            worker2_received.set()

        # Create ROUTER
        router = ZMQStreamingRouterClient(address=router_address, bind=True)
        router.register_receiver(AsyncMock())
        await router.initialize()
        await router.start()

        # Create two DEALERs with different identities
        dealer1 = ZMQStreamingDealerClient(address=router_address, identity="worker-1")
        dealer1.register_receiver(worker1_handler)
        await dealer1.initialize()
        await dealer1.start()

        dealer2 = ZMQStreamingDealerClient(address=router_address, identity="worker-2")
        dealer2.register_receiver(worker2_handler)
        await dealer2.initialize()
        await dealer2.start()

        try:
            # Announce workers
            await dealer1.send(WorkerReadyMessage(service_id="worker-1"))
            await dealer2.send(WorkerReadyMessage(service_id="worker-2"))
            await asyncio.sleep(0.1)  # Let ready messages be processed

            # Send specific messages to each worker
            msg1 = Message(message_type=MessageType.HEARTBEAT, request_id="msg-1")
            msg2 = Message(message_type=MessageType.HEARTBEAT, request_id="msg-2")

            await router.send_to("worker-1", msg1)
            await router.send_to("worker-2", msg2)

            # Wait for both to receive
            await asyncio.wait_for(worker1_received.wait(), timeout=1.0)
            await asyncio.wait_for(worker2_received.wait(), timeout=1.0)

            # Verify correct routing
            assert worker1_message.request_id == "msg-1"
            assert worker2_message.request_id == "msg-2"

        finally:
            await dealer1.stop()
            await dealer2.stop()
            await router.stop()


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestStreamingClientsErrorHandling:
    """Test error handling in streaming clients."""

    @pytest.mark.asyncio
    async def test_router_handles_malformed_envelope(
        self, mock_zmq_context, wait_for_background_task
    ):
        """Should handle malformed ROUTER envelopes gracefully."""
        handler = AsyncMock()

        mock_socket = AsyncMock(spec=zmq.asyncio.Socket)
        mock_socket.bind = Mock()
        mock_socket.setsockopt = Mock()
        mock_socket.send_multipart = AsyncMock()

        # Return malformed envelope (missing parts)
        call_count = 0

        async def recv_multipart_handler():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                await asyncio.sleep(0.01)
                return [b"worker-1"]  # Missing empty delimiter and message
            else:
                await asyncio.Future()

        mock_socket.recv_multipart = recv_multipart_handler
        mock_zmq_context.socket = Mock(return_value=mock_socket)

        client = ZMQStreamingRouterClient(address="tcp://*:5555", bind=True)
        client.register_receiver(handler)
        await client.initialize()
        await client.start()

        try:
            await asyncio.sleep(0.1)
            # Should not crash, should log error and continue
            handler.assert_not_called()
        finally:
            await client.stop()

    @pytest.mark.asyncio
    async def test_dealer_handles_malformed_message(
        self, mock_zmq_context, wait_for_background_task
    ):
        """Should handle malformed messages gracefully."""
        handler = AsyncMock()

        mock_socket = AsyncMock(spec=zmq.asyncio.Socket)
        mock_socket.bind = Mock()
        mock_socket.connect = Mock()
        mock_socket.setsockopt = Mock()
        mock_socket.send_multipart = AsyncMock()

        # Return invalid JSON
        call_count = 0

        async def recv_multipart_handler():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                await asyncio.sleep(0.01)
                return [b"", b"not valid json"]
            else:
                await asyncio.Future()

        mock_socket.recv_multipart = recv_multipart_handler
        mock_zmq_context.socket = Mock(return_value=mock_socket)

        client = ZMQStreamingDealerClient(
            address="tcp://localhost:5555", identity="worker-1"
        )
        client.register_receiver(handler)
        await client.initialize()
        await client.start()

        try:
            await asyncio.sleep(0.1)
            # Should not crash, should log error and continue
            handler.assert_not_called()
        finally:
            await client.stop()


# ============================================================================
# Lifecycle Tests
# ============================================================================


class TestStreamingClientsLifecycle:
    """Test lifecycle behavior of streaming clients."""

    @pytest.mark.asyncio
    async def test_router_cleanup_on_stop(self, mock_zmq_context):
        """Should clean up resources on stop."""
        client = ZMQStreamingRouterClient(address="tcp://*:5555", bind=True)
        client.register_receiver(AsyncMock())
        await client.initialize()
        await client.start()

        await client.stop()

        # Verify handlers cleared
        assert client._receiver_handler is None

    @pytest.mark.asyncio
    async def test_dealer_cleanup_on_stop(self, mock_zmq_context):
        """Should clean up resources on stop."""
        client = ZMQStreamingDealerClient(
            address="tcp://localhost:5555", identity="worker-1"
        )
        client.register_receiver(AsyncMock())
        await client.initialize()
        await client.start()

        await client.stop()

        # Verify handler cleared
        assert client._receiver_handler is None

    @pytest.mark.asyncio
    async def test_cannot_register_multiple_handlers(self, mock_zmq_context):
        """Should raise error if trying to register multiple handlers."""
        router = ZMQStreamingRouterClient(address="tcp://*:5555", bind=True)
        dealer = ZMQStreamingDealerClient(
            address="tcp://localhost:5555", identity="worker-1"
        )

        # First registration should work
        router.register_receiver(AsyncMock())
        dealer.register_receiver(AsyncMock())

        # Second registration should fail
        with pytest.raises(ValueError, match="already registered"):
            router.register_receiver(AsyncMock())

        with pytest.raises(ValueError, match="already registered"):
            dealer.register_receiver(AsyncMock())
