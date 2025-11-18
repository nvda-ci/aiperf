# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from pathlib import Path

import pytest

from aiperf.common.config import (
    EndpointConfig,
    InputConfig,
    LoadGeneratorConfig,
    OutputConfig,
    UserConfig,
)
from aiperf.common.enums import EndpointType

"""
Comprehensive test suite for artifact directory computation in UserConfig.
"""


@pytest.fixture(autouse=True)
def mock_path_is_file(monkeypatch):
    """Automatically mock Path.is_file for all tests in this module."""
    monkeypatch.setattr("pathlib.Path.is_file", lambda self: True)


def create_user_config(
    model_names: list[str] | None = None,
    endpoint_type: EndpointType = EndpointType.CHAT,
    base_artifact_directory: Path = Path("/tmp/artifacts"),
    concurrency: int | None = None,
    request_rate: float | None = None,
    request_count: int | None = None,
    fixed_schedule: bool = False,
    input_file: str | None = None,
) -> UserConfig:
    """Create a UserConfig with sensible defaults for testing."""
    if model_names is None:
        model_names = ["testmodel"]

    loadgen_kwargs = {}
    if concurrency is not None:
        loadgen_kwargs["concurrency"] = concurrency
    if request_rate is not None:
        loadgen_kwargs["request_rate"] = request_rate
    if request_count is not None:
        loadgen_kwargs["request_count"] = request_count
    if not loadgen_kwargs and not fixed_schedule:
        loadgen_kwargs["concurrency"] = 1

    input_kwargs = {}
    if fixed_schedule:
        input_kwargs["fixed_schedule"] = True
        input_kwargs["file"] = input_file or "/tmp/dummy_input.txt"

    return UserConfig(
        endpoint=EndpointConfig(
            model_names=model_names,
            type=endpoint_type,
            custom_endpoint="custom_endpoint",
            url="http://custom-url",
        ),
        output=OutputConfig(base_artifact_directory=base_artifact_directory),
        loadgen=LoadGeneratorConfig(**loadgen_kwargs)
        if loadgen_kwargs
        else LoadGeneratorConfig(),
        input=InputConfig(**input_kwargs) if input_kwargs else InputConfig(),
    )


@dataclass
class FullArtifactTestCase:
    """Complete test case for artifact directory computation."""

    model_names: list[str]
    endpoint_type: EndpointType
    fixed_schedule: bool
    concurrency: int | None
    request_rate: float | None
    expected_dir: Path
    description: str = ""


class TestComputeArtifactDirectory:
    """Comprehensive test suite for artifact directory computation."""

    @pytest.mark.parametrize(
        "model_names,expected_model_part",
        [
            pytest.param(["simple_model"], "simple_model", id="simple model name"),
            pytest.param(["hf/model"], "hf_model", id="model name with single slash"),
            pytest.param(
                ["org/repo/model"],
                "org_repo_model",
                id="model name with multiple slashes",
            ),
            pytest.param(
                ["nvidia/llama-3.1-nemotron-70b-instruct"],
                "nvidia_llama-3.1-nemotron-70b-instruct",
                id="realistic huggingface model name",
            ),
            pytest.param(
                ["model-with-dashes"], "model-with-dashes", id="model name with dashes"
            ),
            pytest.param(
                ["model_with_underscores"],
                "model_with_underscores",
                id="model name with underscores",
            ),
            pytest.param(
                ["MODEL123"], "MODEL123", id="model name with uppercase and numbers"
            ),
            pytest.param(["model.v1.0"], "model.v1.0", id="model name with dots"),
        ],
    )
    def test_model_name_handling(self, model_names, expected_model_part):
        """Test various model name formats and special character handling."""
        config = create_user_config(model_names=model_names)
        artifact_dir = config.computed_artifact_directory
        assert artifact_dir.name.startswith(expected_model_part)

    @pytest.mark.parametrize(
        "model_names,expected_model_part",
        [
            pytest.param(["model1", "model2"], "model1_multi", id="two models"),
            pytest.param(
                ["model1", "model2", "model3"], "model1_multi", id="three models"
            ),
            pytest.param(
                ["hf/model1", "hf/model2"],
                "hf_model1_multi",
                id="multi models with slashes",
            ),
            pytest.param(["a", "b", "c", "d", "e"], "a_multi", id="five models"),
        ],
    )
    def test_multi_model_naming(self, model_names, expected_model_part):
        """Test multi-model scenarios with _multi suffix."""
        config = create_user_config(model_names=model_names)
        artifact_dir = config.computed_artifact_directory
        assert artifact_dir.name.startswith(expected_model_part)
        assert "_multi" in artifact_dir.name

    @pytest.mark.parametrize(
        "endpoint_type,expected_service_part",
        [
            (EndpointType.CHAT, "openai-chat"),
            (EndpointType.COMPLETIONS, "openai-completions"),
            (EndpointType.EMBEDDINGS, "openai-embeddings"),
        ],
    )
    def test_endpoint_type_in_artifact_path(self, endpoint_type, expected_service_part):
        """Test that different endpoint types are correctly reflected in artifact path."""
        config = create_user_config(endpoint_type=endpoint_type)
        artifact_dir = config.computed_artifact_directory
        assert expected_service_part in str(artifact_dir)

    @pytest.mark.parametrize(
        "concurrency,request_rate,expected_stimulus_part",
        [
            pytest.param(1, None, "concurrency1", id="concurrency only mode"),
            pytest.param(None, 10.0, "request_rate10.0", id="request rate only mode"),
            pytest.param(
                5,
                10.0,
                "concurrency5-request_rate10.0",
                id="both concurrency and request rate",
            ),
            pytest.param(10, None, "concurrency10", id="high concurrency"),
            pytest.param(None, 0.5, "request_rate0.5", id="fractional request rate"),
            pytest.param(
                None, 1000.0, "request_rate1000.0", id="very high request rate"
            ),
            pytest.param(
                8,
                25.5,
                "concurrency8-request_rate25.5",
                id="decimal request rate with concurrency",
            ),
        ],
    )
    def test_stimulus_patterns(self, concurrency, request_rate, expected_stimulus_part):
        """Test various stimulus (timing mode) patterns in artifact directories."""
        config = create_user_config(
            concurrency=concurrency,
            request_rate=request_rate,
        )
        artifact_dir = config.computed_artifact_directory
        assert expected_stimulus_part in str(artifact_dir)

    def test_stimulus_fixed_schedule(self):
        """Test fixed schedule stimulus pattern."""
        config = create_user_config(fixed_schedule=True)
        artifact_dir = config.computed_artifact_directory
        assert "fixed_schedule" in str(artifact_dir)

    @pytest.mark.parametrize(
        "test_case",
        [
            FullArtifactTestCase(
                model_names=["hf/model"],
                endpoint_type=EndpointType.CHAT,
                fixed_schedule=False,
                concurrency=5,
                request_rate=10.0,
                expected_dir=Path(
                    "/tmp/artifacts/hf_model-openai-chat-concurrency5-request_rate10.0"
                ),
                description="model name with slash",
            ),
            FullArtifactTestCase(
                model_names=["model1", "model2"],
                endpoint_type=EndpointType.COMPLETIONS,
                fixed_schedule=False,
                concurrency=8,
                request_rate=25.5,
                expected_dir=Path(
                    "/tmp/artifacts/model1_multi-openai-completions-concurrency8-request_rate25.5"
                ),
                description="multi-model",
            ),
            FullArtifactTestCase(
                model_names=["singlemodel"],
                endpoint_type=EndpointType.EMBEDDINGS,
                fixed_schedule=True,
                concurrency=None,
                request_rate=None,
                expected_dir=Path(
                    "/tmp/artifacts/singlemodel-openai-embeddings-fixed_schedule"
                ),
                description="single model with fixed schedule",
            ),
        ],
        ids=lambda test_case: test_case.description,
    )
    def test_full_artifact_directory_path(self, test_case: FullArtifactTestCase):
        """Test complete artifact directory paths (backward compatibility test)."""
        config = create_user_config(
            model_names=test_case.model_names,
            endpoint_type=test_case.endpoint_type,
            concurrency=test_case.concurrency,
            request_rate=test_case.request_rate,
            fixed_schedule=test_case.fixed_schedule,
        )
        assert config.computed_artifact_directory == test_case.expected_dir

    @pytest.mark.parametrize(
        "base_dir,expected_parent",
        [
            (Path("/tmp/artifacts"), Path("/tmp/artifacts")),
            (Path("/custom/output"), Path("/custom/output")),
            (Path("/a/b/c/d/e"), Path("/a/b/c/d/e")),
            (Path("relative/path"), Path("relative/path")),
            (Path("."), Path(".")),
        ],
    )
    def test_base_artifact_directory_variations(self, base_dir, expected_parent):
        """Test that different base artifact directories are correctly used."""
        config = create_user_config(base_artifact_directory=base_dir)
        artifact_dir = config.computed_artifact_directory
        assert artifact_dir.parent == expected_parent

    def test_artifact_directory_caching(self):
        """Test that artifact directory is computed once and cached."""
        config = create_user_config()
        dir1 = config.computed_artifact_directory
        dir2 = config.computed_artifact_directory
        assert dir1 is dir2
        assert dir1 == dir2

    def test_artifact_directory_manual_override(self):
        """Test that artifact directory can be manually set."""
        config = create_user_config()
        custom_dir = Path("/custom/override/path")
        config.computed_artifact_directory = custom_dir
        assert config.computed_artifact_directory == custom_dir

    @pytest.mark.parametrize(
        "concurrency,request_rate,expected_substring",
        [
            (None, None, "concurrency1"),  # Default concurrency
            (1, None, "concurrency1"),
            (10, None, "concurrency10"),
            (None, 5.0, "request_rate5.0"),
            (5, 10.0, "concurrency5-request_rate10.0"),
        ],
    )
    def test_concurrency_and_request_rate_combinations(
        self, concurrency, request_rate, expected_substring
    ):
        """Test various combinations of concurrency and request rate values."""
        config = create_user_config(
            concurrency=concurrency,
            request_rate=request_rate,
        )
        artifact_dir = config.computed_artifact_directory
        assert expected_substring in str(artifact_dir)

    def test_artifact_directory_with_all_endpoint_types(self):
        """Test artifact directory generation for all supported endpoint types."""
        for endpoint_type in [
            EndpointType.CHAT,
            EndpointType.COMPLETIONS,
            EndpointType.EMBEDDINGS,
        ]:
            config = create_user_config(endpoint_type=endpoint_type)
            artifact_dir = config.computed_artifact_directory
            assert artifact_dir.parent == Path("/tmp/artifacts")
            assert "testmodel" in artifact_dir.name
            assert artifact_dir.exists() is False

    @pytest.mark.parametrize(
        "model_names",
        [
            ["a" * 100],
            ["model-" + "x" * 50],
        ],
    )
    def test_long_model_names(self, model_names):
        """Test artifact directory generation with very long model names."""
        config = create_user_config(model_names=model_names)
        artifact_dir = config.computed_artifact_directory
        assert len(str(artifact_dir)) > 0
        assert artifact_dir.parent == Path("/tmp/artifacts")

    @pytest.mark.parametrize(
        "concurrency,request_rate",
        [
            (1, 1.0),
            (1, 0.1),
            (10, 100.0),
            (100, 1000.0),
            (999, 9999.99),
        ],
    )
    def test_extreme_concurrency_and_rate_values(self, concurrency, request_rate):
        """Test artifact directory with extreme but valid concurrency and rate values."""
        config = create_user_config(
            concurrency=concurrency,
            request_rate=request_rate,
            request_count=max(concurrency, 100),
        )
        artifact_dir = config.computed_artifact_directory
        assert f"concurrency{concurrency}" in str(artifact_dir)
        assert f"request_rate{request_rate}" in str(artifact_dir)

    def test_artifact_directory_structure_consistency(self):
        """Test that artifact directory always follows the expected structure pattern."""
        config = create_user_config(concurrency=5, request_rate=10.0)
        artifact_dir = config.computed_artifact_directory
        parts = artifact_dir.name.split("-")
        assert len(parts) >= 3
        assert parts[0] == "testmodel"
        assert "openai" in parts[1]
        assert "chat" in parts[2]

    @pytest.mark.parametrize(
        "model_name",
        [
            "///",
            "org////repo",
            "/leading",
            "trailing/",
        ],
    )
    def test_edge_case_slash_handling(self, model_name):
        """Test edge cases in slash handling for model names."""
        config = create_user_config(model_names=[model_name])
        artifact_dir = config.computed_artifact_directory
        assert "/" not in artifact_dir.name
        assert len(artifact_dir.name) > 0

    def test_artifact_directory_with_fixed_schedule_and_file(self):
        """Test artifact directory when using fixed schedule mode with input file."""
        config = create_user_config(
            fixed_schedule=True,
            input_file="/tmp/schedule.txt",
        )
        artifact_dir = config.computed_artifact_directory
        assert "fixed_schedule" in str(artifact_dir)
        assert "concurrency" not in str(artifact_dir)
        assert "request_rate" not in str(artifact_dir)

    def test_high_concurrency_with_sufficient_request_count(self):
        """Test artifact directory with high concurrency and sufficient request count."""
        config = create_user_config(concurrency=100, request_count=200)
        artifact_dir = config.computed_artifact_directory
        assert "concurrency100" in str(artifact_dir)

    @pytest.mark.parametrize(
        "model_names,expected_first_part",
        [
            (["gpt-4"], "gpt-4"),
            (["claude-3-opus"], "claude-3-opus"),
            (["meta-llama/Llama-2-7b-hf"], "meta-llama_Llama-2-7b-hf"),
            (["mistralai/Mistral-7B-v0.1"], "mistralai_Mistral-7B-v0.1"),
        ],
    )
    def test_realistic_model_names(self, model_names, expected_first_part):
        """Test artifact directory with realistic production model names."""
        config = create_user_config(model_names=model_names)
        artifact_dir = config.computed_artifact_directory
        assert artifact_dir.name.startswith(expected_first_part)

    def test_artifact_path_uniqueness_across_configs(self):
        """Test that different configurations produce unique artifact directories."""
        configs = [
            create_user_config(model_names=["model1"], concurrency=5),
            create_user_config(
                model_names=["model1"], concurrency=10, request_count=20
            ),
            create_user_config(
                model_names=["model1"],
                endpoint_type=EndpointType.COMPLETIONS,
                concurrency=5,
            ),
        ]
        artifact_dirs = [config.computed_artifact_directory for config in configs]
        assert len(artifact_dirs) == len(set(artifact_dirs))
        assert artifact_dirs[0] != artifact_dirs[1]
        assert artifact_dirs[0] != artifact_dirs[2]
        assert artifact_dirs[1] != artifact_dirs[2]

    @pytest.mark.parametrize(
        "request_rate,expected_part",
        [
            (0.001, "request_rate0.001"),
            (0.01, "request_rate0.01"),
            (0.1, "request_rate0.1"),
            (1.5, "request_rate1.5"),
            (10.5, "request_rate10.5"),
            (100.99, "request_rate100.99"),
        ],
    )
    def test_fractional_request_rates(self, request_rate, expected_part):
        """Test artifact directory with various fractional request rates."""
        config = create_user_config(request_rate=request_rate)
        artifact_dir = config.computed_artifact_directory
        assert expected_part in str(artifact_dir)

    def test_directory_name_does_not_contain_invalid_chars(self):
        """Test that artifact directory name contains only filesystem-safe characters."""
        config = create_user_config(
            model_names=["test/model"],
            concurrency=5,
            request_rate=10.5,
        )
        artifact_dir = config.computed_artifact_directory
        invalid_chars = ["<", ">", ":", '"', "|", "?", "*"]
        for char in invalid_chars:
            assert char not in artifact_dir.name

    def test_multi_model_first_model_with_special_chars(self):
        """Test multi-model where first model has special characters."""
        config = create_user_config(
            model_names=["org/model-v1.0", "model2", "model3"],
        )
        artifact_dir = config.computed_artifact_directory
        assert artifact_dir.name.startswith("org_model-v1.0_multi")
        assert "/" not in artifact_dir.name
