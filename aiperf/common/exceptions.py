# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiperf.common.types import ServiceTypeT


class AIPerfError(Exception):
    """Base class for all exceptions raised by AIPerf."""

    def __str__(self) -> str:
        """Return the string representation of the exception with the class name."""
        return super().__str__()


class AIPerfMultiError(AIPerfError):
    """Exception raised when running multiple tasks and one or more fail."""

    def __init__(self, message: str | None, exceptions: list[Exception]) -> None:
        self.exceptions = exceptions

        err_strings = [str(e) for e in exceptions]
        if message:
            super().__init__(f"{message}: {','.join(err_strings)}")
        else:
            super().__init__(",".join(err_strings))


class HookError(AIPerfError):
    """Exception raised when a hook encounters an error."""

    def __init__(self, hook_class_name: str, hook_func_name: str, e: Exception) -> None:
        self.hook_class_name = hook_class_name
        self.hook_func_name = hook_func_name
        self.exception = e
        super().__init__(f"{hook_class_name}.{hook_func_name}: {e}")


class ServiceError(AIPerfError):
    """Generic service error."""

    def __init__(
        self,
        message: str,
        service_type: "ServiceTypeT",
        service_id: str,
    ) -> None:
        super().__init__(
            f"{message} for service of type {service_type} with id {service_id}"
        )
        self.service_type = service_type
        self.service_id = service_id


class LifecycleOperationError(AIPerfError):
    """Exception raised when a lifecycle operation fails and the lifecycle should stop gracefully."""

    def __init__(
        self,
        operation: str,
        original_exception: Exception | None,
        lifecycle_id: str,
    ) -> None:
        self.operation = operation
        self.original_exception = original_exception
        self.lifecycle_id = lifecycle_id
        super().__init__(
            str(original_exception)
            if original_exception
            else f"Failed to perform operation '{operation}'"
        )


class CommunicationError(AIPerfError):
    """Generic communication error."""


class ConfigurationError(AIPerfError):
    """Exception raised when something fails to configure, or there is a configuration error."""


class DatasetLoaderError(AIPerfError):
    """Generic dataset loader error."""


class FactoryCreationError(AIPerfError):
    """Exception raised when a factory encounters an error while creating a class."""


class InitializationError(AIPerfError):
    """Exception raised when something fails to initialize."""


class InvalidOperationError(AIPerfError):
    """Exception raised when an operation is invalid."""


class InvalidStateError(AIPerfError):
    """Exception raised when something is in an invalid state."""


class MetricTypeError(AIPerfError):
    """Exception raised when a metric type encounters an error while creating a class."""


class MetricUnitError(AIPerfError):
    """Exception raised when trying to convert a metric to or from a unit that is does not support it."""


class NotInitializedError(AIPerfError):
    """Exception raised when something that should be initialized is not."""


class NoMetricValue(AIPerfError):
    """Raised when a metric value is not available."""


class PostProcessorDisabled(AIPerfError):
    """Raised when initializing a post processor to indicate to the caller that it is disabled and should not be used."""


class UnsupportedHookError(AIPerfError):
    """Exception raised when a hook is defined on a class that does not have any base classes that provide that hook type."""


class ValidationError(AIPerfError):
    """Exception raised when something fails validation."""
