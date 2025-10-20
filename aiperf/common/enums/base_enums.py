# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from enum import Enum, EnumMeta
from functools import cached_property
from typing import Any

from pydantic import BaseModel, Field
from typing_extensions import Self


class CaseInsensitiveStrEnum(str, Enum):
    """
    CaseInsensitiveStrEnum is a custom enumeration class that extends `str` and `Enum` to provide case-insensitive
    lookup functionality for its members.
    """

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}.{self.name}"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self.value.lower() == other.lower()
        if isinstance(other, Enum):
            return self.value.lower() == other.value.lower()
        return super().__eq__(other)

    def __hash__(self) -> int:
        return hash(self.value.lower())

    @classmethod
    def _missing_(cls, value):
        """
        Handles cases where a value is not directly found in the enumeration.

        This method is called when an attempt is made to access an enumeration
        member using a value that does not directly match any of the defined
        members. It provides custom logic to handle such cases.

        Returns:
            The matching enumeration member if a case-insensitive match is found
            for string values; otherwise, returns None.
        """
        if isinstance(value, str):
            for member in cls:
                if member.value.lower() == value.lower():
                    return member
        return None


class BasePydanticEnumInfo(BaseModel):
    """Base class for all enum info classes that extend `BasePydanticBackedStrEnum`. By default, it
    provides a `tag` for the enum member, which is used for lookup and string comparison,
    and the subclass can provide additional information as needed."""

    tag: str = Field(
        ...,
        min_length=1,
        description="The string value of the enum member used for lookup, serialization, and string insensitive comparison.",
    )

    def __str__(self) -> str:
        return self.tag


class BasePydanticBackedStrEnum(CaseInsensitiveStrEnum):
    """
    Custom enumeration class that extends `CaseInsensitiveStrEnum`
    and is backed by a `BasePydanticEnumInfo` that contains the `tag`, and any other information that is needed
    to represent the enum member.
    """

    # Override the __new__ method to store the `BasePydanticEnumInfo` subclass model as an attribute. This is a python feature that
    # allows us to modify the behavior of the enum class's constructor. We use this to ensure the the enums still look like
    # a regular string enum, but also have the additional information stored as an attribute.
    def __new__(cls, info: BasePydanticEnumInfo) -> Self:
        # Create a new string object based on this class and the tag value.
        obj = str.__new__(cls, info.tag)
        # Ensure string value is set for comparison. This is how enums work internally.
        obj._value_ = info.tag
        # Store the Pydantic model as an attribute.
        obj._info: BasePydanticEnumInfo = info  # type: ignore
        return obj

    @cached_property
    def info(self) -> BasePydanticEnumInfo:
        """Get the enum info for the enum member."""
        # This is the Pydantic model that was stored as an attribute in the __new__ method.
        return self._info  # type: ignore


class DynamicEnumMeta(EnumMeta):
    """Metaclass for dynamic enums with IDE support.

    Features:
    - Supports dynamic member creation at runtime
    - IDE-friendly attribute access for type hints
    - Thread-safe member access
    - Compatible with standard Enum behavior

    Usage:
        class MyDynamicEnum(str, metaclass=DynamicEnumMeta):
            '''Members are generated dynamically at runtime.'''
            pass
    """

    def __getattr__(cls, name: str) -> Any:
        """Allow IDE to understand dynamic enum members while supporting runtime access.

        Sequence:
        1. Try standard enum member lookup (for actual enum members)
        2. For IDE hints, return a DynamicMemberProxy (for unknown but potentially valid members)
        3. Raise AttributeError for truly invalid attributes (starting with _)

        Args:
            name: The attribute name being accessed

        Returns:
            The enum member if it exists, or a DynamicMemberProxy for IDE support

        Raises:
            AttributeError: If the attribute is private (starts with _)
        """
        # Try standard enum lookup first
        try:
            return super().__getattribute__(name)
        except AttributeError:
            pass

        # Skip private/dunder attributes
        if name.startswith("_"):
            raise AttributeError(f"'{cls.__name__}' has no attribute '{name}'")

        # For IDE support: return a proxy that acts like an enum member
        # This allows IDEs to understand dynamic members without errors
        return DynamicMemberProxy(cls, name)


class DynamicMemberProxy:
    """Proxy for dynamic enum members that don't exist yet.

    Provides IDE-friendly duck typing while remaining runtime-compatible.
    Useful for:
    - IDE autocomplete suggestions
    - Type checking with dynamic enums
    - Graceful handling of unknown members

    This is NOT an actual enum member - it's a placeholder for IDE tooling.
    When used at runtime, it will fail loudly to indicate the member is missing.
    """

    __slots__ = ("_class", "_name")

    def __init__(self, enum_class: type, name: str) -> None:
        object.__setattr__(self, "_class", enum_class)
        object.__setattr__(self, "_name", name)

    @property
    def value(self) -> str:
        """Return the name as the value."""
        return object.__getattribute__(self, "_name")

    @property
    def name(self) -> str:
        """Return the name."""
        return object.__getattribute__(self, "_name")

    def __str__(self) -> str:
        """String representation."""
        name = object.__getattribute__(self, "_name")
        return name.lower()

    def __repr__(self) -> str:
        """Representation for debugging."""
        enum_class = object.__getattribute__(self, "_class")
        name = object.__getattribute__(self, "_name")
        return f"{enum_class.__name__}.{name} (dynamic)"

    def __eq__(self, other: Any) -> bool:
        """Compare with strings and other proxies."""
        if isinstance(other, str):
            name = object.__getattribute__(self, "_name")
            return name.lower() == other.lower()
        if isinstance(other, DynamicMemberProxy):
            self_name = object.__getattribute__(self, "_name")
            other_name = object.__getattribute__(other, "_name")
            return self_name == other_name
        return NotImplemented

    def __hash__(self) -> int:
        """Hash for use in sets/dicts."""
        name = object.__getattribute__(self, "_name")
        return hash(name.lower())

    def __setattr__(self, name: str, value: Any) -> None:
        """Prevent modification."""
        raise TypeError(f"Cannot set attribute '{name}' on dynamic member")

    def __getattr__(self, name: str) -> Any:
        """Raise informative error for missing attributes."""
        if name in ("_class", "_name"):
            return object.__getattribute__(self, name)
        raise AttributeError(
            f"Dynamic member '{object.__getattribute__(self, '_name')}' "
            f"has no attribute '{name}'. This is likely a missing enum member."
        )


class DynamicStrEnum(str, Enum, metaclass=DynamicEnumMeta):
    """A string enum that supports dynamic members and IDE autocomplete.

    This enum type:
    - Stores members as strings
    - Supports case-insensitive comparison
    - Provides IDE hints for dynamic members via DynamicMemberProxy
    - Maintains thread safety
    - Works with existing enum patterns

    Usage:
        class EndpointType(DynamicStrEnum):
            OPENAI_CHAT = "openai_chat"
            ANTHROPIC_CHAT = "anthropic_chat"

        # Dynamic access (works at runtime)
        impl = EndpointType.OPENAI_CHAT

        # IDE sees all attributes due to metaclass
        # Type checkers understand potential members
    """

    def __str__(self) -> str:
        """Return string value."""
        return self.value

    def __repr__(self) -> str:
        """Return representation."""
        return f"{self.__class__.__name__}.{self.name}"

    def __eq__(self, other: Any) -> bool:
        """Case-insensitive comparison."""
        if isinstance(other, str):
            return self.value.lower() == other.lower()
        if isinstance(other, Enum):
            return self.value.lower() == other.value.lower()
        return super().__eq__(other)

    def __hash__(self) -> int:
        """Hash based on lowercase value."""
        return hash(self.value.lower())

    @classmethod
    def _missing_(cls, value: Any) -> Any:
        """Handle case-insensitive lookup."""
        if isinstance(value, str):
            for member in cls:
                if member.value.lower() == value.lower():
                    return member
        return None
