# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from aiperf.common.enums.base_enums import CaseInsensitiveStrEnum


class ModelSelectionStrategy(CaseInsensitiveStrEnum):
    """Strategy for selecting the model to use for the request."""

    ROUND_ROBIN = "round_robin"
    """Cycle through the models in order, wrapping around to the beginning."""

    RANDOM = "random"
    """Randomly select a model from the list with replacement."""

    SHUFFLE = "shuffle"
    """Shuffle the list of models and cycle through them without replacement. Re-shuffle after the end of the list."""
