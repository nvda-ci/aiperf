# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for huggingface_generate endpoint."""

from pathlib import Path

import pytest

from tests.component_integration.conftest import (
    ComponentIntegrationTestDefaults as defaults,
)
from tests.harness.utils import AIPerfCLI


@pytest.mark.component_integration
class TestHuggingFaceGenerateEndpoint:
    """Tests for huggingface_generate endpoint."""

    def _create_input_file(self, lines: list[str], tmp_path: Path) -> Path:
        """Helper to create a temporary input file with given text lines."""
        input_file = tmp_path / "inputs.jsonl"
        input_file.write_text("".join(f'{{"text": "{line}"}}\n' for line in lines))
        return input_file

    def _run_profile(
        self,
        cli: AIPerfCLI,
        streaming: bool,
        input_file: Path | None = None,
    ):
        """Helper to run CLI profile for huggingface_generate."""
        stream_flag = "--streaming" if streaming else ""
        dataset_flag = ""
        input_flag = ""

        if input_file:
            dataset_flag = "--custom-dataset-type single_turn"
            input_flag = f"--input-file {input_file}"

        result = cli.run_sync(
            f"""
            aiperf profile \
                --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
                --endpoint-type huggingface_generate \
                {stream_flag} \
                {input_flag} \
                {dataset_flag} \
                --request-count {defaults.request_count} \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        assert result.request_count == defaults.request_count
        return result

    def test_synthetic_non_streaming(self, cli: AIPerfCLI):
        """Synthetic (no input file) non-streaming run."""
        result = self._run_profile(cli, streaming=False)
        assert not result.has_streaming_metrics

    def test_synthetic_streaming(self, cli: AIPerfCLI):
        """Synthetic (no input file) streaming run."""
        result = self._run_profile(cli, streaming=True)
        assert result.has_streaming_metrics

    def test_file_input_non_streaming(self, cli: AIPerfCLI, tmp_path: Path):
        """File input non-streaming run."""
        input_file = self._create_input_file(
            ["Hello TinyLlama!", "Tell me a joke."], tmp_path
        )
        result = self._run_profile(cli, streaming=False, input_file=input_file)
        assert not result.has_streaming_metrics

    def test_file_input_streaming(self, cli: AIPerfCLI, tmp_path: Path):
        """File input streaming run."""
        input_file = self._create_input_file(
            ["Stream something poetic.", "Give me a haiku."], tmp_path
        )
        result = self._run_profile(cli, streaming=True, input_file=input_file)
        assert result.has_streaming_metrics
