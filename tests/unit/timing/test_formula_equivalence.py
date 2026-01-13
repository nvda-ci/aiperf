# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify new simplified virtual history formula matches original exactly."""

from typing import NamedTuple

import pytest


class OriginalResult(NamedTuple):
    start_sec: float
    turns_remaining: int


class NewResult(NamedTuple):
    first_send_offset: float
    turns_to_send: int


def original_formula(
    num_users: int, qps: float, session_turns: int, i: int
) -> OriginalResult:
    """Original formula from before simplification."""
    turn_gap = num_users / qps
    session_lifetime_sec = turn_gap * (session_turns - 1)
    user_spawn_gap = session_lifetime_sec / num_users
    min_gap_sec = turn_gap / num_users
    user_spawn_gap = max(min_gap_sec, user_spawn_gap)

    virtual_age = num_users - i
    offset_sec = virtual_age * user_spawn_gap
    start_sec = turn_gap - (offset_sec % turn_gap)

    virtual_done = (virtual_age * (session_turns - 1)) // num_users + 1
    turns_remaining = session_turns - virtual_done

    return OriginalResult(start_sec, turns_remaining)


def new_formula(num_users: int, qps: float, session_turns: int, i: int) -> NewResult:
    """New simplified formula."""
    # effective_turns is for staggering math only (floor of 1 for single-turn case)
    effective_turns = max(1, session_turns - 1)
    virtual_history = (num_users - i) * effective_turns
    completed_turns, phase_offset = divmod(virtual_history, num_users)

    # Use original session_turns for actual turn count (clamped for single-turn edge case)
    turns_to_send = max(0, (session_turns - 1) - completed_turns)
    first_send_offset = (num_users - phase_offset) / qps

    return NewResult(first_send_offset, turns_to_send)


# Test cases with various prime/coprime relationships
@pytest.mark.parametrize(
    "num_users,qps,session_turns,desc",
    [
        # Basic cases
        (10, 1.0, 5, "simple baseline"),
        (10, 2.0, 5, "higher qps"),
        (10, 0.5, 5, "fractional qps"),
        # Prime numbers
        (7, 1.0, 11, "prime users, prime turns"),
        (13, 1.0, 17, "larger primes"),
        (2, 1.0, 3, "smallest primes"),
        (97, 1.0, 101, "large primes"),
        # Coprime relationships (GCD = 1)
        (8, 1.0, 9, "coprime: 8 and 9"),
        (9, 1.0, 16, "coprime: 9 and 16"),
        (15, 1.0, 22, "coprime: 15 and 22"),
        (25, 1.0, 36, "coprime: 25 and 36"),
        # Non-coprime (shared factors)
        (12, 1.0, 18, "GCD=6: 12 and 18"),
        (15, 1.0, 20, "GCD=5: 15 and 20"),
        (16, 1.0, 24, "GCD=8: 16 and 24"),
        (100, 1.0, 150, "GCD=50: 100 and 150"),
        # Edge cases
        (1, 1.0, 5, "single user"),
        (5, 1.0, 1, "single turn"),
        (1, 1.0, 1, "single user, single turn"),
        (5, 1.0, 2, "two turns (T-1=1)"),
        # Adversarial QPS values
        (10, 0.1, 5, "very low qps"),
        (10, 100.0, 5, "very high qps"),
        (7, 0.333, 11, "irrational-ish qps"),
        (13, 3.14159, 17, "pi qps"),
        (10, 2.71828, 5, "e qps"),
        # Large scale
        (1000, 1.0, 50, "many users"),
        (50, 1.0, 1000, "many turns"),
        (500, 10.0, 100, "large scale, high qps"),
        # Adversarial: users >> turns
        (100, 1.0, 3, "many users, few turns"),
        (50, 1.0, 2, "50 users, 2 turns"),
        # Adversarial: turns >> users
        (3, 1.0, 100, "few users, many turns"),
        (2, 1.0, 50, "2 users, 50 turns"),
        # Perfect divisibility
        (10, 1.0, 11, "turns-1 = users"),
        (10, 1.0, 21, "turns-1 = 2*users"),
        (10, 1.0, 6, "turns-1 = users/2"),
        # Fibonacci-related (often adversarial)
        (8, 1.0, 13, "fibonacci: 8 and 13"),
        (13, 1.0, 21, "fibonacci: 13 and 21"),
        (21, 1.0, 34, "fibonacci: 21 and 34"),
    ],
)
def test_formula_equivalence(num_users: int, qps: float, session_turns: int, desc: str):
    """Verify new formula matches original for all users."""
    for i in range(num_users):
        orig = original_formula(num_users, qps, session_turns, i)
        new = new_formula(num_users, qps, session_turns, i)

        assert abs(orig.start_sec - new.first_send_offset) < 1e-10, (
            f"{desc}: i={i} start_sec mismatch: {orig.start_sec} vs {new.first_send_offset}"
        )
        assert orig.turns_remaining == new.turns_to_send, (
            f"{desc}: i={i} turns mismatch: {orig.turns_remaining} vs {new.turns_to_send}"
        )
