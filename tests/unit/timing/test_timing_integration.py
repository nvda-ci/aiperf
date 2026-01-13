# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import tempfile
from pathlib import Path

from aiperf.common.config import (
    EndpointConfig,
    InputConfig,
    LoadGeneratorConfig,
    UserConfig,
)
from aiperf.common.enums import CustomDatasetType, TimingMode
from aiperf.timing.config import TimingConfig


class TestTimingConfigurationIntegration:
    """Test timing configuration integration with effective request count."""

    def test_effective_request_count_in_timing_config(self, create_mooncake_trace_file):
        """Test that TimingConfig honors explicit user request_count over dataset size."""
        filename = create_mooncake_trace_file(3)  # 3 entries

        try:
            user_config = UserConfig(
                endpoint=EndpointConfig(model_names=["test-model"]),
                loadgen=LoadGeneratorConfig(
                    request_count=100
                ),  # User's explicit value should be honored
                input=InputConfig(
                    file=filename, custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE
                ),
            )

            timing_config = TimingConfig.from_user_config(user_config)

            # Should use user's explicit request_count (100), not dataset size (3)
            assert timing_config.phase_configs[0].total_expected_requests == 100

        finally:
            Path(filename).unlink(missing_ok=True)

    def test_timing_mode_selection_with_timestamps(self, create_mooncake_trace_file):
        """Test that timing mode switches to fixed schedule when timestamps present."""
        filename = create_mooncake_trace_file(3, include_timestamps=True)

        try:
            user_config = UserConfig(
                endpoint=EndpointConfig(model_names=["test-model"]),
                input=InputConfig(
                    file=filename, custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE
                ),
            )

            timing_config = TimingConfig.from_user_config(user_config)

            # Should auto-detect fixed schedule due to timestamps
            assert (
                timing_config.phase_configs[0].timing_mode == TimingMode.FIXED_SCHEDULE
            )
            assert timing_config.phase_configs[0].total_expected_requests == 3

        finally:
            Path(filename).unlink(missing_ok=True)

    def test_timing_mode_selection_without_timestamps(self, create_mooncake_trace_file):
        """Test that timing mode uses request rate when no timestamps."""
        filename = create_mooncake_trace_file(3, include_timestamps=False)

        try:
            user_config = UserConfig(
                endpoint=EndpointConfig(model_names=["test-model"]),
                input=InputConfig(
                    file=filename, custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE
                ),
            )

            timing_config = TimingConfig.from_user_config(user_config)

            # Should use request rate (default) since no timestamps
            assert timing_config.phase_configs[0].timing_mode == TimingMode.REQUEST_RATE
            # When no explicit request_count is provided and no timestamps, defaults to 10
            assert timing_config.phase_configs[0].total_expected_requests == 10

        finally:
            Path(filename).unlink(missing_ok=True)

    def test_non_custom_dataset_uses_original_count(self):
        """Test that non-custom datasets use explicitly configured request_count."""
        user_config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            loadgen=LoadGeneratorConfig(request_count=42),
            # No custom dataset configuration
        )

        timing_config = TimingConfig.from_user_config(user_config)

        # Should use original request_count since no custom dataset
        assert timing_config.phase_configs[0].total_expected_requests == 42

    def test_empty_dataset_file_behavior(self, create_mooncake_trace_file):
        """Test behavior with empty dataset file."""
        # Create empty file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            filename = f.name

        try:
            user_config = UserConfig(
                endpoint=EndpointConfig(model_names=["test-model"]),
                input=InputConfig(
                    file=filename, custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE
                ),
            )

            timing_config = TimingConfig.from_user_config(user_config)

            # When dataset file is empty and no request_count provided, defaults to 10
            assert timing_config.phase_configs[0].total_expected_requests == 10
            # Timing mode should still be REQUEST_RATE (no timestamps means no fixed schedule)
            assert timing_config.phase_configs[0].timing_mode == TimingMode.REQUEST_RATE

        finally:
            Path(filename).unlink(missing_ok=True)

    def test_mixed_custom_dataset_timing_integration(self, create_mooncake_trace_file):
        """Test end-to-end timing integration with mixed content."""
        filename = create_mooncake_trace_file(2, include_timestamps=True)

        try:
            user_config = UserConfig(
                endpoint=EndpointConfig(model_names=["test-model"]),
                loadgen=LoadGeneratorConfig(
                    request_rate=10  # Should be ignored due to fixed schedule
                ),
                input=InputConfig(
                    file=filename, custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE
                ),
            )

            # Test that all the pieces work together
            effective_count = user_config._count_dataset_entries()
            should_use_fixed = (
                user_config._should_use_fixed_schedule_for_mooncake_trace()
            )
            timing_config = TimingConfig.from_user_config(user_config)

            assert effective_count == 2
            assert should_use_fixed is True
            assert timing_config.phase_configs[0].total_expected_requests == 2
            assert (
                timing_config.phase_configs[0].timing_mode == TimingMode.FIXED_SCHEDULE
            )

        finally:
            Path(filename).unlink(missing_ok=True)
