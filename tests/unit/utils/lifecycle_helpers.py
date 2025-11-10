# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared lifecycle testing utilities.

This module provides context managers and helpers for testing components
that use AIPerfLifecycleMixin. With global ZMQ mocking, we can now test
components through their full lifecycle (initialize -> start -> run -> stop)
without complex patching overhead.

Usage:
    @pytest.mark.asyncio
    async def test_component_lifecycle():
        component = MyComponent(service_config=ServiceConfig(), user_config=UserConfig())

        async with aiperf_lifecycle(component) as running_component:
            # Component is now RUNNING
            result = await running_component.do_something()
            assert result == expected

        # Component is now STOPPED and cleaned up
        assert component.state == LifecycleState.STOPPED
"""

from contextlib import asynccontextmanager, suppress
from typing import TypeVar

from aiperf.common.enums import LifecycleState

T = TypeVar("T")


@asynccontextmanager
async def aiperf_lifecycle(component: T) -> T:
    """Universal async context manager for AIPerfLifecycleMixin components.

    This context manager handles the full lifecycle of any component that uses
    AIPerfLifecycleMixin:
    1. Calls initialize() to transition CREATED -> INITIALIZED
    2. Calls start() to transition INITIALIZED -> RUNNING
    3. Yields the running component for testing
    4. Calls stop() to transition RUNNING -> STOPPED on exit

    Args:
        component: Any component with AIPerfLifecycleMixin (services, processors, etc.)

    Yields:
        The component in RUNNING state

    Example:
        async with aiperf_lifecycle(telemetry_manager) as manager:
            # manager.state == LifecycleState.RUNNING
            await manager.do_something()
        # manager.state == LifecycleState.STOPPED

    Note:
        With global ZMQ mocking enabled, components can be instantiated directly
        without patching BaseComponentService or CommunicationMixin. This context
        manager then provides safe lifecycle management for testing.
    """
    try:
        await component.initialize()
        await component.start()
        yield component
    finally:
        await component.stop()


@asynccontextmanager
async def aiperf_initialized(component: T) -> T:
    """Context manager for components that need initialization but not start/stop.

    Similar to aiperf_lifecycle but only calls initialize(). Useful for testing
    components that don't need to be fully running.

    Args:
        component: Any component with AIPerfLifecycleMixin

    Yields:
        The component in INITIALIZED state

    Example:
        async with aiperf_initialized(processor) as proc:
            # proc.state == LifecycleState.INITIALIZED
            result = proc.process_data(data)
            assert result.is_valid
    """
    try:
        await component.initialize()
        yield component
    finally:
        # Best-effort cleanup - some components may not support stop from INITIALIZED
        if hasattr(component, "stop"):
            with suppress(Exception):
                await component.stop()


def assert_lifecycle_state(component, expected_state: LifecycleState) -> None:
    """Assert that a component is in the expected lifecycle state.

    Args:
        component: Component with a state attribute
        expected_state: Expected LifecycleState

    Raises:
        AssertionError: If component state doesn't match expected

    Example:
        assert_lifecycle_state(service, LifecycleState.RUNNING)
    """
    assert hasattr(component, "state"), (
        f"{component.__class__.__name__} has no state attribute"
    )
    assert component.state == expected_state, (
        f"Expected {component.__class__.__name__} to be in {expected_state}, "
        f"but was in {component.state}"
    )


async def test_full_lifecycle_transitions(component: T) -> None:
    """Test that a component goes through all lifecycle states correctly.

    This is a helper function that tests the full lifecycle state machine:
    CREATED -> INITIALIZED -> RUNNING -> STOPPED

    Args:
        component: Freshly instantiated component (in CREATED state)

    Example:
        @pytest.mark.asyncio
        async def test_telemetry_manager_lifecycle():
            manager = TelemetryManager(ServiceConfig(), UserConfig())
            await test_full_lifecycle_transitions(manager)
    """
    # Verify initial state
    assert_lifecycle_state(component, LifecycleState.CREATED)

    # Test initialize transition
    await component.initialize()
    assert_lifecycle_state(component, LifecycleState.INITIALIZED)

    # Test start transition
    await component.start()
    assert_lifecycle_state(component, LifecycleState.RUNNING)

    # Test stop transition
    await component.stop()
    assert_lifecycle_state(component, LifecycleState.STOPPED)
