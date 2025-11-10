# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Shared fixtures for visualization tests.

This file contains fixtures that are automatically discovered by pytest
and made available to test functions in the same directory and subdirectories.
"""

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from aiperf.plot.core.mode_detector import ModeDetector

logging.getLogger("choreographer").setLevel(logging.WARNING)
logging.getLogger("kaleido").setLevel(logging.WARNING)

# Path constants for static fixture data
FIXTURES_DIR = Path(__file__).parent / "fixtures"
QWEN_CONCURRENCY1_DIR = FIXTURES_DIR / "qwen_concurrency1"
QWEN_CONCURRENCY4_DIR = FIXTURES_DIR / "qwen_concurrency4"
LLAMA_CONCURRENCY1_DIR = FIXTURES_DIR / "llama_concurrency1"
LLAMA_CONCURRENCY4_DIR = FIXTURES_DIR / "llama_concurrency4"


@pytest.fixture
def mode_detector() -> ModeDetector:
    """
    Create a ModeDetector instance for testing.

    Returns:
        ModeDetector instance.
    """
    return ModeDetector()


@pytest.fixture
def tmp_run_dir(tmp_path: Path) -> Path:
    """
    Create a temporary directory structure for a single run.

    Args:
        tmp_path: Pytest's tmp_path fixture.

    Returns:
        Path to the temporary run directory.
    """
    run_dir = tmp_path / "test_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


@pytest.fixture
def sample_jsonl_data() -> list[dict[str, Any]]:
    """
    Generate sample JSONL data for testing.

    Returns:
        List of dictionaries representing JSONL records.
    """
    return [
        {
            "metadata": {
                "session_num": 0,
                "x_request_id": "req-1",
                "request_start_ns": 1000000000000,
                "request_ack_ns": 1000000100000,
                "request_end_ns": 1000001000000,
                "benchmark_phase": "profiling",
                "was_cancelled": False,
                "worker_id": "0",
                "record_processor_id": "0",
            },
            "metrics": {
                "time_to_first_token": {"value": 45.5, "unit": "ms"},
                "inter_token_latency": {"value": 18.2, "unit": "ms"},
                "request_latency": {"value": 900.0, "unit": "ms"},
                "output_sequence_length": {"value": 100, "unit": "tokens"},
                "input_sequence_length": {"value": 50, "unit": "tokens"},
            },
            "error": None,
        },
        {
            "metadata": {
                "session_num": 1,
                "x_request_id": "req-2",
                "request_start_ns": 1000000500000,
                "request_ack_ns": 1000000600000,
                "request_end_ns": 1000001500000,
                "benchmark_phase": "profiling",
                "was_cancelled": False,
                "worker_id": "0",
                "record_processor_id": "0",
            },
            "metrics": {
                "time_to_first_token": {"value": 50.2, "unit": "ms"},
                "inter_token_latency": {"value": 19.5, "unit": "ms"},
                "request_latency": {"value": 1000.0, "unit": "ms"},
                "output_sequence_length": {"value": 120, "unit": "tokens"},
                "input_sequence_length": {"value": 60, "unit": "tokens"},
            },
            "error": None,
        },
    ]


@pytest.fixture
def sample_aggregated_data() -> dict[str, Any]:
    """
    Generate sample aggregated JSON data for testing.

    Returns:
        Dictionary representing aggregated statistics.
    """
    return {
        "input_config": {
            "endpoint": {
                "model_names": ["test-model"],
                "type": "chat",
                "streaming": True,
            },
            "loadgen": {
                "concurrency": 4,
                "request_count": 100,
            },
            "input": {
                "prompt": {
                    "input_tokens": {"mean": 100, "stddev": 10},
                    "output_tokens": {"mean": 200, "stddev": 20},
                }
            },
        },
        "was_cancelled": False,
        "error_summary": [],
    }


@pytest.fixture
def sample_input_config() -> dict[str, Any]:
    """
    Generate sample inputs.json data for testing.

    Returns:
        Dictionary representing input configuration.
    """
    return {
        "data": [
            {
                "session_id": "session-1",
                "payloads": [
                    {
                        "messages": [{"role": "user", "content": "Test prompt 1"}],
                        "model": "test-model",
                        "stream": True,
                    }
                ],
            },
            {
                "session_id": "session-2",
                "payloads": [
                    {
                        "messages": [{"role": "user", "content": "Test prompt 2"}],
                        "model": "test-model",
                        "stream": True,
                    }
                ],
            },
        ]
    }


@pytest.fixture
def populated_run_dir(
    tmp_run_dir: Path,
    sample_jsonl_data: list[dict[str, Any]],
    sample_aggregated_data: dict[str, Any],
    sample_input_config: dict[str, Any],
) -> Path:
    """
    Create a fully populated run directory with all required files.

    Args:
        tmp_run_dir: Temporary run directory.
        sample_jsonl_data: Sample JSONL records.
        sample_aggregated_data: Sample aggregated data.
        sample_input_config: Sample input config.

    Returns:
        Path to the populated run directory.
    """
    # Write profile_export.jsonl
    jsonl_file = tmp_run_dir / "profile_export.jsonl"
    with open(jsonl_file, "w") as f:
        for record in sample_jsonl_data:
            f.write(json.dumps(record) + "\n")

    # Write profile_export_aiperf.json
    json_file = tmp_run_dir / "profile_export_aiperf.json"
    with open(json_file, "w") as f:
        json.dump(sample_aggregated_data, f)

    # Write inputs.json
    inputs_file = tmp_run_dir / "inputs.json"
    with open(inputs_file, "w") as f:
        json.dump(sample_input_config, f)

    return tmp_run_dir


@pytest.fixture
def multiple_run_dirs(
    tmp_path: Path,
    sample_jsonl_data: list[dict[str, Any]],
    sample_aggregated_data: dict[str, Any],
) -> list[Path]:
    """
    Create multiple run directories for testing multi-run scenarios.

    Args:
        tmp_path: Pytest's tmp_path fixture.
        sample_jsonl_data: Sample JSONL records.
        sample_aggregated_data: Sample aggregated data.

    Returns:
        List of paths to run directories.
    """
    run_dirs = []

    # Create 3 runs with varying concurrency
    for i, concurrency in enumerate([2, 4, 8], start=1):
        run_dir = tmp_path / f"run_{i}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Write JSONL
        jsonl_file = run_dir / "profile_export.jsonl"
        with open(jsonl_file, "w") as f:
            for record in sample_jsonl_data:
                f.write(json.dumps(record) + "\n")

        # Write aggregated JSON with different concurrency
        agg_data = sample_aggregated_data.copy()
        agg_data["input_config"]["loadgen"]["concurrency"] = concurrency
        json_file = run_dir / "profile_export_aiperf.json"
        with open(json_file, "w") as f:
            json.dump(agg_data, f)

        run_dirs.append(run_dir)

    return run_dirs


@pytest.fixture
def parent_dir_with_runs(multiple_run_dirs: list[Path]) -> Path:
    """
    Get parent directory containing multiple run subdirectories.

    Args:
        multiple_run_dirs: List of run directories.

    Returns:
        Path to parent directory.
    """
    return multiple_run_dirs[0].parent


@pytest.fixture
def parent_dir_with_single_run(
    tmp_path: Path,
    sample_jsonl_data: list[dict[str, Any]],
    sample_aggregated_data: dict[str, Any],
) -> Path:
    """
    Create a parent directory containing only a single run directory.

    Args:
        tmp_path: Pytest's tmp_path fixture.
        sample_jsonl_data: Sample JSONL records.
        sample_aggregated_data: Sample aggregated data.

    Returns:
        Path to parent directory containing one run.
    """
    parent = tmp_path / "single_run_parent"
    parent.mkdir(parents=True, exist_ok=True)

    run_dir = parent / "run_1"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Write JSONL
    jsonl_file = run_dir / "profile_export.jsonl"
    with open(jsonl_file, "w") as f:
        for record in sample_jsonl_data:
            f.write(json.dumps(record) + "\n")

    # Write aggregated JSON
    json_file = run_dir / "profile_export_aiperf.json"
    with open(json_file, "w") as f:
        json.dump(sample_aggregated_data, f)

    return parent


@pytest.fixture
def nested_run_dirs(
    tmp_path: Path,
    sample_jsonl_data: list[dict[str, Any]],
) -> Path:
    """
    Create nested run directories (run containing another run).

    Args:
        tmp_path: Pytest's tmp_path fixture.
        sample_jsonl_data: Sample JSONL records.

    Returns:
        Path to parent directory containing nested runs.
    """
    parent = tmp_path / "nested_runs"
    parent.mkdir(parents=True, exist_ok=True)

    # Create outer run
    outer = parent / "outer_run"
    outer.mkdir(parents=True, exist_ok=True)
    outer_jsonl = outer / "profile_export.jsonl"
    with open(outer_jsonl, "w") as f:
        for record in sample_jsonl_data:
            f.write(json.dumps(record) + "\n")

    # Create inner run nested inside outer
    inner = outer / "inner_run"
    inner.mkdir(parents=True, exist_ok=True)
    inner_jsonl = inner / "profile_export.jsonl"
    with open(inner_jsonl, "w") as f:
        for record in sample_jsonl_data:
            f.write(json.dumps(record) + "\n")

    return parent


# Static fixtures using real data from actual aiperf runs
@pytest.fixture
def real_qwen_concurrency1_dir() -> Path:
    """
    Path to real Qwen run directory with concurrency=1.

    Returns:
        Path to the Qwen concurrency 1 fixture directory.
    """
    return QWEN_CONCURRENCY1_DIR


@pytest.fixture
def real_qwen_concurrency4_dir() -> Path:
    """
    Path to real Qwen run directory with concurrency=4.

    Returns:
        Path to the Qwen concurrency 4 fixture directory.
    """
    return QWEN_CONCURRENCY4_DIR


@pytest.fixture
def real_llama_concurrency1_dir() -> Path:
    """
    Path to real Llama run directory with concurrency=1.

    Returns:
        Path to the Llama concurrency 1 fixture directory.
    """
    return LLAMA_CONCURRENCY1_DIR


@pytest.fixture
def real_llama_concurrency4_dir() -> Path:
    """
    Path to real Llama run directory with concurrency=4.

    Returns:
        Path to the Llama concurrency 4 fixture directory.
    """
    return LLAMA_CONCURRENCY4_DIR


@pytest.fixture
def real_qwen_sweep_dirs() -> list[Path]:
    """
    Paths to Qwen concurrency sweep run directories for comparison testing.

    Returns:
        List of paths to Qwen run fixture directories with different concurrency.
    """
    return [QWEN_CONCURRENCY1_DIR, QWEN_CONCURRENCY4_DIR]


@pytest.fixture
def real_llama_sweep_dirs() -> list[Path]:
    """
    Paths to Llama concurrency sweep run directories for comparison testing.

    Returns:
        List of paths to Llama run fixture directories with different concurrency.
    """
    return [LLAMA_CONCURRENCY1_DIR, LLAMA_CONCURRENCY4_DIR]


@pytest.fixture
def real_data_parent_dir() -> Path:
    """
    Path to parent directory containing real run subdirectories.

    Returns:
        Path to the fixtures directory containing all real runs.
    """
    return FIXTURES_DIR


@pytest.fixture
def real_qwen_gpu_telemetry() -> Path:
    """
    Path to real GPU telemetry data from Qwen concurrency1 run.

    Returns:
        Path to gpu_telemetry_export.jsonl file.
    """
    gpu_file = QWEN_CONCURRENCY1_DIR / "gpu_telemetry_export.jsonl"
    if not gpu_file.exists():
        pytest.skip(f"GPU telemetry file not found: {gpu_file}")
    return gpu_file


@pytest.fixture
def real_qwen_profile_data() -> Path:
    """
    Path to real profile data from Qwen concurrency1 run.

    Returns:
        Path to profile_export.jsonl file.
    """
    profile_file = QWEN_CONCURRENCY1_DIR / "profile_export.jsonl"
    if not profile_file.exists():
        pytest.skip(f"Profile file not found: {profile_file}")
    return profile_file


@pytest.fixture
def real_qwen_aggregated_data() -> Path:
    """
    Path to real aggregated data from Qwen concurrency1 run.

    Returns:
        Path to profile_export_aiperf.json file.
    """
    agg_file = QWEN_CONCURRENCY1_DIR / "profile_export_aiperf.json"
    if not agg_file.exists():
        pytest.skip(f"Aggregated file not found: {agg_file}")
    return agg_file


@pytest.fixture
def qwen_concurrency8_dir() -> Path:
    """
    Path to Qwen concurrency 8 run directory for higher concurrency testing.

    Returns:
        Path to the Qwen concurrency 8 fixture directory.
    """
    qwen_c8_dir = FIXTURES_DIR / "qwen_concurrency8"
    if not qwen_c8_dir.exists():
        pytest.skip(f"Qwen concurrency 8 fixture not found: {qwen_c8_dir}")
    return qwen_c8_dir
