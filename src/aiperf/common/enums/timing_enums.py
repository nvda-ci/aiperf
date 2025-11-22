# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums.base_enums import CaseInsensitiveStrEnum


class TimingMode(CaseInsensitiveStrEnum):
    """The different ways the TimingManager should generate requests."""

    FIXED_SCHEDULE = "fixed_schedule"
    """A mode where the TimingManager will send requests according to a fixed schedule."""

    REQUEST_RATE = "request_rate"
    """A mode where the TimingManager will send requests using a request rate generator based on various modes.
    Optionally, a max concurrency limit can be specified as well.
    """


class RequestRateMode(CaseInsensitiveStrEnum):
    """The different ways the RequestRateStrategy should generate requests."""

    CONSTANT = "constant"
    """Generate requests at a constant rate."""

    POISSON = "poisson"
    """Generate requests using a poisson process."""

    CONCURRENCY_BURST = "concurrency_burst"
    """Generate requests as soon as possible, up to a max concurrency limit. Only allowed when a request rate is not specified."""


class CreditPhase(CaseInsensitiveStrEnum):
    """The type of credit phase. This is used to identify which phase of the
    benchmark the credit is being used in, for tracking and reporting purposes."""

    WARMUP = "warmup"
    """The credit phase while the warmup is active. This is used to warm up the model and
    ensure that the model is ready to be profiled."""

    PROFILING = "profiling"
    """The credit phase while profiling is active. This is the primary phase of the
    benchmark, and what is used to calculate the final results."""


class CreditScope(CaseInsensitiveStrEnum):
    """The scope of a credit - whether it represents a single turn or an entire conversation.

    This determines how workers process credits and when they return them:
    - TURN: Credit represents 1 turn (1 request). Worker processes the turn and returns immediately.
            Used by REQUEST_RATE mode for precise concurrency control at the request level.
    - CONVERSATION: Credit represents an entire conversation (N turns). Worker processes all turns
                    sequentially with inter-turn delays and returns when conversation completes.
                    Used by FIXED_SCHEDULE and other modes to avoid blocking TimingManager.
    """

    TURN = "turn"
    """Credit represents a single turn (1 request).
    Worker processes one turn and returns the credit immediately after completion.
    Concurrency control operates at the turn/request level.
    Used by REQUEST_RATE mode."""

    CONVERSATION = "conversation"
    """Credit represents an entire conversation (N turns).
    Worker processes all turns sequentially (with inter-turn delays) and returns
    the credit only after the entire conversation completes.
    Concurrency control operates at the conversation level.
    Used by FIXED_SCHEDULE and other timing modes."""
