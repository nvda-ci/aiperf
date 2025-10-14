# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest

from aiperf.common.exceptions import NotInitializedError
from aiperf.common.tokenizer import Tokenizer


class TestTokenizer:
    def test_empty_tokenizer(self):
        tokenizer = Tokenizer()
        assert tokenizer._tokenizer is None

        with pytest.raises(NotInitializedError):
            tokenizer("test")
        with pytest.raises(NotInitializedError):
            tokenizer.encode("test")
        with pytest.raises(NotInitializedError):
            tokenizer.decode([1])
        with pytest.raises(NotInitializedError):
            _ = tokenizer.bos_token_id

    def test_non_empty_tokenizer(self, mock_tokenizer_cls):
        tokenizer = mock_tokenizer_cls.from_pretrained("gpt2")
        assert tokenizer._tokenizer is not None

        assert tokenizer("This is a test")["input_ids"] == [10, 11, 12, 13]
        assert tokenizer.encode("This is a test") == [10, 11, 12, 13]
        assert (
            tokenizer.decode([10, 11, 12, 13]) == "token_10 token_11 token_12 token_13"
        )
        assert tokenizer.bos_token_id == 1

    def test_all_args(self, mock_tokenizer_cls):
        tokenizer = mock_tokenizer_cls.from_pretrained(
            name="gpt2",
            trust_remote_code=True,
            revision="11c5a3d5811f50298f278a704980280950aedb10",
        )
        assert tokenizer._tokenizer is not None


class TestTokenizerAliasResolution:
    """Tests for HuggingFace Hub alias resolution."""

    def test_resolve_alias_successful(self, mock_model_info):
        """Test successful alias resolution."""
        mock_info = MagicMock()
        mock_info.id = "openai-community/gpt2"
        mock_model_info.return_value = mock_info

        resolved = Tokenizer.resolve_alias("gpt2")
        assert resolved == "openai-community/gpt2"
        mock_model_info.assert_called_once_with("gpt2", token=None)

    def test_resolve_alias_with_token(self, mock_model_info):
        """Test alias resolution with authentication token."""
        mock_info = MagicMock()
        mock_info.id = "private-org/private-model"
        mock_model_info.return_value = mock_info

        resolved = Tokenizer.resolve_alias("private-model", token="hf_token")
        assert resolved == "private-org/private-model"
        mock_model_info.assert_called_once_with("private-model", token="hf_token")

    def test_resolve_alias_not_found(self, mock_model_info):
        """Test alias resolution when repository is not found."""
        from huggingface_hub.utils import RepositoryNotFoundError

        mock_model_info.side_effect = RepositoryNotFoundError("Not found")

        resolved = Tokenizer.resolve_alias("nonexistent-model")
        assert resolved == "nonexistent-model"

    def test_resolve_alias_http_error(self, mock_model_info):
        """Test alias resolution when HTTP error occurs."""
        from huggingface_hub.utils import HfHubHTTPError

        mock_model_info.side_effect = HfHubHTTPError("HTTP error")

        resolved = Tokenizer.resolve_alias("problematic-model")
        assert resolved == "problematic-model"

    def test_resolve_alias_generic_exception(self, mock_model_info):
        """Test alias resolution with unexpected exception."""
        mock_model_info.side_effect = Exception("Unexpected error")

        resolved = Tokenizer.resolve_alias("some-model")
        assert resolved == "some-model"

    def test_from_pretrained_with_alias_resolution(
        self, mock_auto_tokenizer, mock_model_info
    ):
        """Test from_pretrained with alias resolution enabled."""
        mock_info = MagicMock()
        mock_info.id = "openai-community/gpt2"
        mock_model_info.return_value = mock_info

        mock_tokenizer = MagicMock()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        tokenizer = Tokenizer.from_pretrained("gpt2", resolve_alias=True)

        mock_model_info.assert_called_once_with("gpt2", token=None)

        mock_auto_tokenizer.from_pretrained.assert_called_once_with(
            "openai-community/gpt2",
            trust_remote_code=False,
            revision="main",
            token=None,
        )

        assert tokenizer._tokenizer is not None

    def test_from_pretrained_without_alias_resolution(
        self, mock_auto_tokenizer, mock_model_info
    ):
        """Test from_pretrained with alias resolution disabled."""
        mock_tokenizer = MagicMock()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        tokenizer = Tokenizer.from_pretrained("gpt2", resolve_alias=False)

        mock_model_info.assert_not_called()

        mock_auto_tokenizer.from_pretrained.assert_called_once_with(
            "gpt2",
            trust_remote_code=False,
            revision="main",
            token=None,
        )

        assert tokenizer._tokenizer is not None

    def test_from_pretrained_alias_resolution_with_token(
        self, mock_auto_tokenizer, mock_model_info
    ):
        """Test from_pretrained with alias resolution and authentication token."""
        mock_info = MagicMock()
        mock_info.id = "private-org/private-model"
        mock_model_info.return_value = mock_info

        mock_tokenizer = MagicMock()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        tokenizer = Tokenizer.from_pretrained(
            "private-model", resolve_alias=True, token="hf_token"
        )

        mock_model_info.assert_called_once_with("private-model", token="hf_token")

        mock_auto_tokenizer.from_pretrained.assert_called_once_with(
            "private-org/private-model",
            trust_remote_code=False,
            revision="main",
            token="hf_token",
        )

        assert tokenizer._tokenizer is not None

    def test_from_pretrained_alias_resolution_fallback(
        self, mock_auto_tokenizer, mock_model_info
    ):
        """Test from_pretrained fallback when alias resolution fails."""
        from huggingface_hub.utils import RepositoryNotFoundError

        mock_model_info.side_effect = RepositoryNotFoundError("Not found")

        mock_tokenizer = MagicMock()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        tokenizer = Tokenizer.from_pretrained("local/path", resolve_alias=True)

        mock_auto_tokenizer.from_pretrained.assert_called_once_with(
            "local/path",
            trust_remote_code=False,
            revision="main",
            token=None,
        )

        assert tokenizer._tokenizer is not None

    def test_resolve_alias_with_search(self, mock_model_info, mock_list_models):
        """Test alias resolution using search when direct lookup fails."""
        from huggingface_hub.utils import RepositoryNotFoundError

        mock_model_info.side_effect = RepositoryNotFoundError("Not found")

        mock_model = MagicMock()
        mock_model.id = "openai/gpt-oss-20b"
        mock_list_models.return_value = iter([mock_model])

        resolved = Tokenizer.resolve_alias("gpt-oss-20b")
        assert resolved == "openai/gpt-oss-20b"
        mock_list_models.assert_called_once_with(
            search="gpt-oss-20b", limit=50, token=None
        )

    def test_resolve_alias_with_search_multiple_results(
        self, mock_model_info, mock_list_models
    ):
        """Test alias resolution chooses the correct model from multiple search results."""
        from huggingface_hub.utils import RepositoryNotFoundError

        mock_model_info.side_effect = RepositoryNotFoundError("Not found")

        mock_model1 = MagicMock()
        mock_model1.id = "other-org/gpt-oss-20b-variant"
        mock_model2 = MagicMock()
        mock_model2.id = "openai/gpt-oss-20b"
        mock_model3 = MagicMock()
        mock_model3.id = "another/model"
        mock_list_models.return_value = iter([mock_model1, mock_model2, mock_model3])

        resolved = Tokenizer.resolve_alias("gpt-oss-20b")
        assert resolved == "openai/gpt-oss-20b"

    def test_resolve_alias_with_search_no_match(
        self, mock_model_info, mock_list_models
    ):
        """Test alias resolution returns original name when search finds no match."""
        from huggingface_hub.utils import RepositoryNotFoundError

        mock_model_info.side_effect = RepositoryNotFoundError("Not found")

        mock_model = MagicMock()
        mock_model.id = "other-org/different-model"
        mock_list_models.return_value = iter([mock_model])

        resolved = Tokenizer.resolve_alias("gpt-oss-20b")
        assert resolved == "gpt-oss-20b"

    def test_resolve_alias_with_search_token(self, mock_model_info, mock_list_models):
        """Test alias resolution with search passes token correctly."""
        from huggingface_hub.utils import RepositoryNotFoundError

        mock_model_info.side_effect = RepositoryNotFoundError("Not found")

        mock_model = MagicMock()
        mock_model.id = "private-org/private-model"
        mock_list_models.return_value = iter([mock_model])

        resolved = Tokenizer.resolve_alias("private-model", token="hf_token")
        assert resolved == "private-org/private-model"
        mock_list_models.assert_called_once_with(
            search="private-model", limit=50, token="hf_token"
        )

    def test_resolve_alias_with_search_failure(self, mock_model_info, mock_list_models):
        """Test alias resolution returns original name when search fails."""
        from huggingface_hub.utils import RepositoryNotFoundError

        mock_model_info.side_effect = RepositoryNotFoundError("Not found")

        mock_list_models.side_effect = Exception("Search failed")

        resolved = Tokenizer.resolve_alias("some-model")
        assert resolved == "some-model"

    def test_resolve_alias_skips_network_for_local_paths(
        self, mock_model_info, mock_list_models
    ):
        """Test that local paths skip network requests entirely."""
        # Test absolute path
        resolved = Tokenizer.resolve_alias("/home/user/models/my-tokenizer")
        assert resolved == "/home/user/models/my-tokenizer"
        mock_model_info.assert_not_called()
        mock_list_models.assert_not_called()

        # Reset mocks
        mock_model_info.reset_mock()
        mock_list_models.reset_mock()

        # Test relative path with ./
        resolved = Tokenizer.resolve_alias("./local-model")
        assert resolved == "./local-model"
        mock_model_info.assert_not_called()
        mock_list_models.assert_not_called()

        # Reset mocks
        mock_model_info.reset_mock()
        mock_list_models.reset_mock()

        # Test relative path with ../
        resolved = Tokenizer.resolve_alias("../another-model")
        assert resolved == "../another-model"
        mock_model_info.assert_not_called()
        mock_list_models.assert_not_called()
