# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common import random_generator as rng
from aiperf.common.config import UserConfig
from aiperf.common.enums import DatasetLoaderType, DatasetSamplingStrategy
from aiperf.common.factories import DatasetLoaderFactory
from aiperf.common.models import Audio, Conversation, Image, Text, Turn, Video
from aiperf.common.tokenizer import Tokenizer
from aiperf.dataset.loader.synthetic.base import BaseSyntheticLoader


@DatasetLoaderFactory.register(DatasetLoaderType.SYNTHETIC_MULTIMODAL)
class SyntheticMultiModalLoader(BaseSyntheticLoader):
    """Generates synthetic multi-modal conversations.

    Creates synthetic conversations with configurable:
    - Number of turns per conversation (with variance)
    - Multi-modal payloads: text, image, audio, video
    - Turn delays
    - ISL/OSL distribution for consistent prompt/response pairing

    At least one modality must be enabled (prompt, image, or audio).

    Args:
        config: The user configuration
        tokenizer: The tokenizer instance
    """

    def __init__(self, config: UserConfig, tokenizer: Tokenizer):
        super().__init__(config, tokenizer)

        # Derived RNGs for sampling
        self._turn_sampler_rng = rng.derive("loader.conversation.turn_count")
        self._delay_sampler_rng = rng.derive("loader.conversation.turn_delay")

        # Validate at least one modality is enabled
        if (
            not self.include_prompt
            and not self.include_image
            and not self.include_audio
        ):
            raise ValueError(
                "All synthetic data modalities are disabled. "
                "Please enable at least one of prompt, image, or audio by "
                "setting the mean to a positive value."
            )

    @classmethod
    def can_load(cls, config: UserConfig) -> bool:
        """Check if this loader should be used.

        Returns True if:
        - No file specified
        - No public dataset specified
        - Not a rankings endpoint

        Args:
            config: The user configuration to check.

        Returns:
            True if this loader should generate synthetic multi-modal data.
        """
        return (
            config.input.file is None
            and config.input.public_dataset is None
            and "rankings" not in config.endpoint.type.value
        )

    def load(self) -> list[Conversation]:
        """Generate synthetic multi-modal conversations.

        Returns:
            List of Conversation objects with synthetic data.
        """
        conversations = []

        for _ in range(self.config.input.conversation.num_dataset_entries):
            session_id = self.session_id_generator.next()

            # Sample number of turns for this conversation
            num_turns = self._turn_sampler_rng.sample_positive_normal_integer(
                self.config.input.conversation.turn.mean,
                self.config.input.conversation.turn.stddev,
            )
            self.debug(
                lambda num_turns=num_turns: f"Creating conversation with {num_turns} turns"
            )

            turns = []
            for turn_idx in range(num_turns):
                turn = self._create_turn(is_first=(turn_idx == 0))
                turns.append(turn)

            conversation = Conversation(session_id=session_id, turns=turns)
            conversations.append(conversation)

        return conversations

    def _create_turn(self, is_first: bool) -> Turn:
        """Create a single turn with multi-modal payloads.

        Args:
            is_first: Whether this is the first turn in the conversation.

        Returns:
            Turn object with generated media payloads.
        """
        turn = Turn()

        # Generate multi-modal payloads
        if self.include_prompt:
            turn.texts.append(self._generate_text_payloads(turn, is_first))
        if self.include_image:
            turn.images.append(self._generate_image_payloads())
        if self.include_audio:
            turn.audios.append(self._generate_audio_payloads())
        if self.include_video:
            turn.videos.append(self._generate_video_payloads())

        # Set delay for non-first turns
        if not is_first and self.config.input.conversation.turn.delay.mean > 0:
            delay = self._delay_sampler_rng.sample_positive_normal_integer(
                self.config.input.conversation.turn.delay.mean,
                self.config.input.conversation.turn.delay.stddev,
            )
            turn.delay = delay * self.config.input.conversation.turn.delay.ratio

        # Validate at least one payload was generated
        if not turn.texts and not turn.images and not turn.audios and not turn.videos:
            self.warning(
                "No synthetic payloads generated. Please enable at least one modality."
            )

        # Set turn metadata
        turn.model = self.model_selector.select(turn)
        turn.max_tokens = self.get_max_tokens_for_turn(turn)

        # Clear cached sequence lengths to free memory
        self.clear_turn_cache(turn)

        return turn

    def _generate_text_payloads(self, turn: Turn, is_first: bool) -> Text:
        """Generate text payloads for a turn with ISL/OSL consistency.

        Args:
            turn: The turn object (used for caching sequence lengths)
            is_first: Whether this is the first turn in the conversation

        Returns:
            Text object with generated content.
        """
        text = Text(name="text")

        # Sample ISL/OSL pair for this turn (cached for consistency)
        turn_id = id(turn)
        isl, _ = self._get_turn_sequence_lengths(turn_id)

        # Preserve original variance unless sequence distribution is active
        stddev = (
            0
            if self._seq_distribution is not None
            else self.config.input.prompt.input_tokens.stddev
        )

        for _ in range(self.config.input.prompt.batch_size):
            # Generate prompt content using the sampled input sequence length
            content = self.prompt_generator.generate(mean=isl, stddev=stddev)

            # Add prefix prompt if this is the first turn and prefix is enabled
            if is_first and self.prefix_prompt_enabled:
                prefix = self.prompt_generator.get_random_prefix_prompt()
                content = f"{prefix} {content}"

            text.contents.append(content)

        return text

    def _generate_image_payloads(self) -> Image:
        """Generate synthetic images.

        Returns:
            Image object with generated data.
        """
        image = Image(name="image_url")
        for _ in range(self.config.input.image.batch_size):
            data = self.image_generator.generate()
            image.contents.append(data)
        return image

    def _generate_audio_payloads(self) -> Audio:
        """Generate synthetic audio.

        Returns:
            Audio object with generated data.
        """
        audio = Audio(name="input_audio")
        for _ in range(self.config.input.audio.batch_size):
            data = self.audio_generator.generate()
            audio.contents.append(data)
        return audio

    def _generate_video_payloads(self) -> Video:
        """Generate synthetic video.

        Returns:
            Video object with generated data.
        """
        video = Video(name="video_url")
        for _ in range(self.config.input.video.batch_size):
            data = self.video_generator.generate()
            if data:  # Only append if video was actually generated
                video.contents.append(data)
        return video

    def get_preferred_sampling_strategy(self) -> DatasetSamplingStrategy:
        """Return preferred sampling strategy for synthetic data.

        Returns:
            DatasetSamplingStrategy.SHUFFLE for variety in synthetic data.
        """
        return DatasetSamplingStrategy.SHUFFLE

    # Helper properties to check if modalities are enabled

    @property
    def include_prompt(self) -> bool:
        """Check if prompt generation is enabled."""
        return self.config.input.prompt.input_tokens.mean > 0

    @property
    def include_image(self) -> bool:
        """Check if image generation is enabled."""
        return (
            self.config.input.image.width.mean > 0
            and self.config.input.image.height.mean > 0
        )

    @property
    def include_audio(self) -> bool:
        """Check if audio generation is enabled."""
        return self.config.input.audio.length.mean > 0

    @property
    def include_video(self) -> bool:
        """Check if video generation is enabled."""
        return bool(self.config.input.video.width and self.config.input.video.height)
