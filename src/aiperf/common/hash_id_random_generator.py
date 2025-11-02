# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hash-ID-based random generator for parallel processing with reproducibility.

This module provides a specialized RandomGenerator subclass that enables parallel
processing of requests with hash IDs while maintaining perfect reproducibility.
Each hash ID generates a deterministic, independent random sequence regardless
of which worker processes it.

Key features:
- Deterministic seeding per hash_id using stable hash approach (base_seed + hash_id)
- Efficient re-seeding of the same RNG instance (no new instances created)
- Perfect reproducibility across runs and workers
- Isolated random sequences per hash block

Architecture:
    Global Seed → Base Identifier → Hash ID → Re-seed RNG per hash block

Usage:
    >>> from aiperf.common import random_generator as rng
    >>> from aiperf.common.hash_id_random_generator import HashIdRandomGenerator
    >>>
    >>> # In your component __init__
    >>> hash_rng = HashIdRandomGenerator.from_base_rng(rng.derive("dataset.prompt.corpus"))
    >>>
    >>> # For each hash_id, re-seed the RNG deterministically
    >>> hash_rng.reseed_for_hash_id(123)
    >>> value = hash_rng.randrange(1000)
"""

import hashlib

from aiperf.common.random_generator import RandomGenerator

__all__ = ["HashIdRandomGenerator"]


class _FakeNumpyRNG:
    """Fake NumPy RNG that raises an exception when any NumPy RNG operations are attempted."""

    def __getattr__(self, name):
        raise RuntimeError(
            "NumPy RNG is not used for HashIdRandomGenerator. Use Python RNG based operations instead, "
            "or implement usage of NumPy RNG in the subclass."
        )


class HashIdRandomGenerator(RandomGenerator):
    """RandomGenerator subclass that can re-seed per hash_id for parallel processing.

    This subclass extends RandomGenerator with the ability to efficiently re-seed
    for different hash_ids, enabling reproducible random generation in parallel
    processing scenarios where multiple workers need to generate identical content
    for the same hash_id.

    Each hash_id produces a completely independent, deterministic random sequence
    derived from a stable hash of (base_seed + hash_id), ensuring:
    1. Same hash_id always produces same random sequence
    2. Different hash_ids produce independent sequences
    3. Worker order doesn't affect reproducibility
    4. Each worker can maintain its own cache independently
    5. Minimal memory overhead (re-uses same RNG instance)

    Thread Safety:
        This class is NOT thread-safe. Each worker process should have its own
        HashIdRandomGenerator instance. Do not share across threads or processes.
    """

    @classmethod
    def from_base_rng(cls, base_rng: RandomGenerator) -> "HashIdRandomGenerator":
        """Create a HashIdRandomGenerator from a base RandomGenerator.

        Args:
            base_rng: Base RandomGenerator obtained via rng.derive(identifier).
                     Its seed is used as the base for all hash_id derivations.

        Returns:
            A new HashIdRandomGenerator initialized with the base seed.
        """
        # If the base RNG is not seeded, use a random seed
        # This ensures that the HashIdRandomGenerator can use a non-deterministic base seed
        # while, still maintaining reproducibile hash_id seeds among workers.
        base_seed = base_rng.seed or base_rng.randrange(0, 2**64)
        instance = cls(base_seed, _internal=True)
        return instance

    def __init__(self, base_seed: int, *, _internal: bool = False):
        """Initialize hash-id random generator.

        Note:
            Use from_base_rng() class method instead of direct construction.
        """
        super().__init__(base_seed, _internal=_internal)
        # NumPy RNG is not used for HashIdRandomGenerator, so we set it to a fake NumPy RNG
        # This will prevent any accidental use of NumPy RNG operations, which will
        # instead raise an exception.
        self._numpy_rng = _FakeNumpyRNG()

    def reseed_for_hash_id(self, hash_id: int) -> None:
        """Re-seed this RNG for a specific hash_id.

        This method re-seeds the RandomGenerator with a seed deterministically
        derived from the base seed and hash_id. The same hash_id will always
        produce the same derived seed, ensuring reproducible random sequences
        across different workers and runs.

        After calling this method, all random operations on this instance will
        use the new seed until reseed_for_hash_id is called again.

        Args:
            hash_id: The hash identifier for which to re-seed the RNG.
                    This is typically a KV block hash ID from Mooncake traces.

        Example:
            >>> hash_rng = HashIdRandomGenerator.from_base_rng(base_rng)
            >>> hash_rng.reseed_for_hash_id(123)
            >>> val1 = hash_rng.randrange(1000)
            >>> hash_rng.reseed_for_hash_id(456)
            >>> val2 = hash_rng.randrange(1000)
            >>> # val1 and val2 come from independent, deterministic sequences
        """
        # Deterministic: derive seed from base_seed + hash_id
        seed_string = f"{self.seed}:hash_id:{hash_id}"
        hash_bytes = hashlib.sha256(seed_string.encode("utf-8")).digest()
        derived_seed = int.from_bytes(hash_bytes[:8], byteorder="big")

        # Re-seed Python RNG only since NumPy is not used for this RNG
        self._python_rng.seed(derived_seed)
