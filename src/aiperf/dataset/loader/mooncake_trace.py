# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from pathlib import Path

from pydantic import ValidationError

from aiperf.common.config.user_config import UserConfig
from aiperf.common.enums import DatasetLoaderType, DatasetSamplingStrategy
from aiperf.common.factories import DatasetLoaderFactory
from aiperf.common.models import Conversation, Text, Turn
from aiperf.common.tokenizer import Tokenizer
from aiperf.dataset.generator import PromptGenerator
from aiperf.dataset.loader.file.base import BaseFileLoader
from aiperf.dataset.loader.models import MooncakeTrace


@DatasetLoaderFactory.register(DatasetLoaderType.MOONCAKE_TRACE)
class MooncakeTraceDatasetLoader(BaseFileLoader):
    """A dataset loader that loads Mooncake trace data from a file.

    Loads Mooncake trace data from a file and converts the data into
    a list of conversations for dataset manager.

    Each line in the file represents a single trace entry and will be
    converted to a separate conversation with a unique session ID.

    Example:
    Fixed schedule version
    ```json
    {"timestamp": 1000, "input_length": 300, "output_length": 40, "hash_ids": [123, 456]}
    ```

    Multi-turn version
    ```json
    {"session_id": "abc-123", "input_length": 300, "output_length": 40},
    {"session_id": "abc-123", "delay": 2, "input_length": 150, "output_length": 20}
    ```
    """

    def __init__(
        self,
        config: UserConfig,
        tokenizer: Tokenizer,
        filename: str,
        prompt_generator: PromptGenerator,
        **kwargs,
    ):
        super().__init__(config, tokenizer, filename, **kwargs)
        self.prompt_generator = prompt_generator
        self._skipped_traces = 0
        self._start_offset = config.input.fixed_schedule_start_offset
        self._end_offset = config.input.fixed_schedule_end_offset

    @classmethod
    def can_load_file(cls, path: Path) -> bool:
        """Check if this loader can handle the given file.

        For mooncake trace data, validates first non-empty line against the MooncakeTrace model.
        This will handle all of the validation logic for the different input combinations.

        Args:
            path: Path to the file to check.

        Returns:
            True if this loader can handle the file, False otherwise.
        """
        if not path.is_file():
            return False

        try:
            with open(path) as f:
                for line in f:
                    if (line := line.strip()) == "":
                        continue  # Skip empty lines
                    MooncakeTrace.model_validate_json(line)
                    return True  # Successfully validated first line
            return False  # File is empty or has only empty lines
        except (ValidationError, Exception):
            return False

    @classmethod
    def get_preferred_sampling_strategy(cls) -> DatasetSamplingStrategy:
        """Get the preferred dataset sampling strategy for MooncakeTrace."""
        return DatasetSamplingStrategy.SEQUENTIAL

    def parse_and_validate(self) -> list[MooncakeTrace]:
        """Parse and validate Mooncake trace data from a file.

        Returns:
            A list of validated MooncakeTrace objects (filtered by offset).
        """
        data: list[MooncakeTrace] = []

        with open(self.filename) as f:
            for line in f:
                if (line := line.strip()) == "":
                    continue  # Skip empty lines

                trace_data = MooncakeTrace.model_validate_json(line)

                if (
                    trace_data.timestamp is not None
                    and not self._timestamp_within_offsets(trace_data.timestamp)
                ):
                    self._skipped_traces += 1
                    continue  # Skip traces before or after the fixed schedule offset

                data.append(trace_data)

        if self._skipped_traces > 0:
            self.info(
                f"Skipped {self._skipped_traces:,} traces because they were "
                f"before the start offset of {self._start_offset} or "
                f"after the end offset of {self._end_offset}"
            )
        self.debug(lambda: f"Loaded {len(data):,} traces from {self.filename}")

        return data

    def _timestamp_within_offsets(self, timestamp: int) -> bool:
        return (self._start_offset is None or timestamp >= self._start_offset) and (
            self._end_offset is None or timestamp <= self._end_offset
        )

    def convert_to_conversations(self, data: list[MooncakeTrace]) -> list[Conversation]:
        """Convert all the Mooncake trace data to conversation objects.

        Groups traces by session_id and creates conversations.

        Args:
            data: A list of MooncakeTrace objects.

        Returns:
            A list of conversations.
        """
        # Group by session_id
        grouped_data: dict[str, list[MooncakeTrace]] = defaultdict(list)
        for trace in data:
            session_id = trace.session_id or self.session_id_generator.next()
            grouped_data[session_id].append(trace)

        # Convert grouped data to conversations
        conversations = []
        for session_id, traces in grouped_data.items():
            conversation = Conversation(session_id=session_id)
            for trace in traces:
                # Handle both text_input and input_length formats
                if trace.text_input is not None:
                    prompt = trace.text_input
                else:
                    prompt = self.prompt_generator.generate(
                        mean=trace.input_length,
                        stddev=0,
                        hash_ids=trace.hash_ids
                        or [],  # Use empty list if hash_ids is None
                    )

                turn = Turn(
                    timestamp=trace.timestamp,
                    delay=trace.delay,
                    texts=[Text(name="text", contents=[prompt])],
                    max_tokens=trace.output_length,
                )
                conversation.turns.append(turn)
            conversations.append(conversation)
        return conversations
