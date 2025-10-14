# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import io
from pathlib import Path
from typing import TYPE_CHECKING

# Use TYPE_CHECKING to import BatchEncoding only during static type checks
if TYPE_CHECKING:
    from transformers import BatchEncoding

from aiperf.common.exceptions import (
    InitializationError,
    NotInitializedError,
)


class Tokenizer:
    """
    This class provides a simplified interface for using Huggingface
    tokenizers, with default arguments for common operations.
    """

    def __init__(self) -> None:
        """
        Initialize the tokenizer with default values for call, encode, and decode.
        """
        self._tokenizer = None
        self._call_args = {"add_special_tokens": False}
        self._encode_args = {"add_special_tokens": False}
        self._decode_args = {"skip_special_tokens": True}

    @staticmethod
    def resolve_alias(name: str, token: str | None = None) -> str:
        """
        Resolve a tokenizer name alias to its canonical repository ID.

        This method queries the HuggingFace Hub to resolve model aliases
        (e.g., "gpt2" -> "openai-community/gpt2", "gpt-oss-20b" -> "openai/gpt-oss-20b").
        If the name is already a canonical ID or if resolution fails, the original name is returned.

        Args:
            name: The tokenizer name or alias to resolve.
            token: Optional HuggingFace API token for private repositories.

        Returns:
            The canonical repository ID if found, otherwise the original name.
        """
        # Check if this looks like a local path to avoid unnecessary network requests
        # A name is considered a local path if:
        # 1. It's an absolute path (starts with / or drive letter on Windows)
        # 2. It's a relative path (starts with ./ or ../)
        # 3. It exists as a directory on the filesystem
        path = Path(name)
        is_local_path = (
            path.is_absolute()
            or name.startswith("./")
            or name.startswith("../")
            or path.is_dir()
        )

        if is_local_path:
            # Skip network requests for local paths
            return name

        # Lazy import HuggingFace Hub to avoid loading it at module import time
        from huggingface_hub import list_models, model_info
        from huggingface_hub.utils import (
            HfHubHTTPError,
            RepositoryNotFoundError,
        )

        try:
            # Try direct lookup first (for full IDs like "openai/gpt-oss-20b")
            info = model_info(name, token=token)
            # The 'id' field contains the canonical repository name
            return info.id
        except (RepositoryNotFoundError, HfHubHTTPError):
            # If direct lookup fails, try searching for the model
            try:
                # Search for models matching the name, get more results for better matching
                models = list(list_models(search=name, limit=50, token=token))

                if not models:
                    return name

                # Prioritize matches in order:
                # 1. Exact match (case-insensitive)
                # 2. Ends with /{name} (e.g., "gpt-oss-20b" -> "openai/gpt-oss-20b")
                # 3. Most downloaded/popular if multiple matches exist
                name_lower = name.lower()
                suffix_matches = []

                for model in models:
                    model_id_lower = model.id.lower()

                    # Check for exact match
                    if model_id_lower == name_lower:
                        return model.id

                    # Check for suffix match (org/model-name pattern)
                    if model_id_lower.endswith(f"/{name_lower}"):
                        suffix_matches.append(model)

                # If we found suffix matches, return the most popular one
                if suffix_matches:
                    # Sort by downloads (most popular first) if available
                    suffix_matches.sort(
                        key=lambda m: getattr(m, "downloads", 0) or 0, reverse=True
                    )
                    return suffix_matches[0].id

                # If no exact or suffix match, return the original name
                return name
            except Exception:
                # If search fails, return the original name
                return name
        except Exception:
            # For any other exception, return the original name
            return name

    @classmethod
    def from_pretrained(
        cls,
        name: str,
        trust_remote_code: bool = False,
        revision: str = "main",
        resolve_alias: bool = True,
        token: str | None = None,
    ) -> "Tokenizer":
        """
        Factory to load a tokenizer for the given pretrained model name.

        Args:
            name: The name or path of the pretrained tokenizer model.
            trust_remote_code: Whether to trust remote code when loading the tokenizer.
            revision: The specific model version to use.
            resolve_alias: Whether to resolve model aliases to canonical names. Default is True.
            token: Optional HuggingFace API token for private repositories.
        """
        import contextlib

        # Lazy import AutoTokenizer to avoid loading transformers at module import time
        with (
            # Silence tokenizer warning on import and first use
            contextlib.redirect_stdout(io.StringIO()) as _,
            contextlib.redirect_stderr(io.StringIO()),
        ):
            from transformers import AutoTokenizer

        try:
            tokenizer_cls = cls()

            # Resolve alias if requested
            resolved_name = name
            if resolve_alias:
                resolved_name = cls.resolve_alias(name, token=token)

            tokenizer_cls._tokenizer = AutoTokenizer.from_pretrained(
                resolved_name,
                trust_remote_code=trust_remote_code,
                revision=revision,
                token=token,
            )
        except Exception as e:
            raise InitializationError(e) from e
        return tokenizer_cls

    def __call__(self, text, **kwargs) -> "BatchEncoding":
        """
        Call the underlying Huggingface tokenizer with default arguments,
        which can be overridden by kwargs.

        Args:
            text: The input text to tokenize.

        Returns:
            A BatchEncoding object containing the tokenized output.
        """
        if self._tokenizer is None:
            raise NotInitializedError("Tokenizer is not initialized.")
        return self._tokenizer(text, **{**self._call_args, **kwargs})

    def encode(self, text, **kwargs) -> list[int]:
        """
        Encode the input text into a list of token IDs.

        This method calls the underlying Huggingface tokenizer's encode
        method with default arguments, which can be overridden by kwargs.

        Args:
            text: The input text to encode.

        Returns:
            A list of token IDs.
        """
        if self._tokenizer is None:
            raise NotInitializedError("Tokenizer is not initialized.")
        return self._tokenizer.encode(text, **{**self._encode_args, **kwargs})

    def decode(self, token_ids, **kwargs) -> str:
        """
        Decode a list of token IDs back into a string.

        This method calls the underlying Huggingface tokenizer's decode
        method with default arguments, which can be overridden by kwargs.

        Args:
            token_ids: A list of token IDs to decode.

        Returns:
            The decoded string.
        """
        if self._tokenizer is None:
            raise NotInitializedError("Tokenizer is not initialized.")
        return self._tokenizer.decode(token_ids, **{**self._decode_args, **kwargs})

    @property
    def bos_token_id(self) -> int:
        """
        Return the beginning-of-sequence (BOS) token ID.
        """
        if self._tokenizer is None:
            raise NotInitializedError("Tokenizer is not initialized.")
        return self._tokenizer.bos_token_id

    @property
    def eos_token_id(self) -> int:
        """
        Return the end-of-sequence (EOS) token ID.
        """
        if self._tokenizer is None:
            raise NotInitializedError("Tokenizer is not initialized.")
        return self._tokenizer.eos_token_id

    @property
    def block_separation_token_id(self) -> int | None:
        """
        Returns BOS, EOS, or None if none are available.
        """
        if self._tokenizer is None:
            raise NotInitializedError("Tokenizer is not initialized.")

        if self.bos_token_id is not None:
            return self.bos_token_id
        if self.eos_token_id is not None:
            return self.eos_token_id
        return None

    def __repr__(self) -> str:
        """
        Return a string representation of the underlying tokenizer.

        Returns:
            The string representation of the tokenizer.
        """
        return self._tokenizer.__repr__()

    def __str__(self) -> str:
        """
        Return a user-friendly string representation of the underlying tokenizer.

        Returns:
            The string representation of the tokenizer.
        """
        return self._tokenizer.__str__()
