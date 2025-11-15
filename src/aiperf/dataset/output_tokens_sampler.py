# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Output tokens sampler for AIPerf benchmarking.

Provides sampling of output token counts for synthetic turn generation.
Handles normal distribution sampling with mean/stddev configuration.

Example:
    ```python
    from aiperf.dataset.output_tokens_sampler import OutputTokensSampler

    sampler = OutputTokensSampler(
        mean=100,
        stddev=10,
    )
    tokens = sampler.sample()  # Returns positive integer or None
    ```
"""

from aiperf.common import random_generator as rng


class OutputTokensSampler:
    """Samples output token counts based on normal distribution configuration.

    When mean is None, returns None (indicating no sampling required).
    When mean is provided, samples positive integers from normal distribution.

    Attributes:
        mean: Mean of the normal distribution for output tokens (None if disabled).
        stddev: Standard deviation for output token sampling.
        rng: RandomGenerator for sampling.
    """

    def __init__(
        self,
        mean: float | None,
        stddev: float,
    ):
        """Initialize the output tokens sampler.

        Args:
            mean: Mean of output tokens distribution. If None, sample() returns None.
            stddev: Standard deviation for output token sampling.

        Example:
            # Enabled sampling
            sampler = OutputTokensSampler(
                mean=100,
                stddev=10,
            )

            # Disabled sampling (mean=None)
            sampler = OutputTokensSampler(
                mean=None,
                stddev=0,
            )
        """
        self.mean = mean
        self.stddev = stddev
        self._rng = rng.derive("dataset.output_tokens")

    def sample(self) -> int | None:
        """Sample an output token count from the configured distribution.

        Returns:
            Positive integer >= 1 sampled from normal distribution, or None if
            mean is None (indicating output token sampling is disabled).

        Note:
            Uses ceiling to ensure result is always >= 1 even when sample
            approaches 0. For stddev <= 0, returns max(1, round(mean)).
        """
        if self.mean is None:
            return None

        return self._rng.sample_positive_normal_integer(self.mean, self.stddev)
