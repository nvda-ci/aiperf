# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import time
from collections import defaultdict

import orjson

from aiperf.common.base_component_service import BaseComponentService
from aiperf.common.config import OutputDefaults, ServiceConfig, UserConfig
from aiperf.common.decorators import implements_protocol
from aiperf.common.enums import (
    CommAddress,
    CommandType,
    DatasetLoaderType,
    MessageType,
    ServiceType,
)
from aiperf.common.environment import Environment
from aiperf.common.factories import (
    DatasetLoaderFactory,
    DatasetSamplingStrategyFactory,
    EndpointFactory,
    PublicDatasetFactory,
    ServiceFactory,
)
from aiperf.common.hooks import on_command, on_request
from aiperf.common.messages import (
    ConversationRequestMessage,
    ConversationResponseMessage,
    ConversationTurnRequestMessage,
    ConversationTurnResponseMessage,
    DatasetConfiguredNotification,
    ProfileConfigureCommand,
)
from aiperf.common.mixins import ReplyClientMixin
from aiperf.common.models import (
    Conversation,
    DatasetMetadata,
    InputsFile,
    ModelEndpointInfo,
    RequestInfo,
    SessionPayloads,
)
from aiperf.common.protocols import ServiceProtocol
from aiperf.common.tokenizer import Tokenizer


@implements_protocol(ServiceProtocol)
@ServiceFactory.register(ServiceType.DATASET_MANAGER)
class DatasetManager(ReplyClientMixin, BaseComponentService):
    """
    The DatasetManager primary responsibility is to manage the data generation or acquisition.
    For synthetic generation, it contains the code to generate the prompts or tokens.
    It will have an API for dataset acquisition of a dataset if available in a remote repository or database.
    """

    def __init__(
        self,
        service_config: ServiceConfig,
        user_config: UserConfig,
        service_id: str | None = None,
    ) -> None:
        super().__init__(
            service_config=service_config,
            user_config=user_config,
            service_id=service_id,
            reply_client_address=CommAddress.DATASET_MANAGER_PROXY_BACKEND,
            reply_client_bind=False,
        )
        self.user_config = user_config
        self.tokenizer: Tokenizer | None = None
        self.dataset: dict[str, Conversation] = {}  # session ID -> Conversation mapping
        self.dataset_metadata: DatasetMetadata | None = None
        self._session_ids_cache: list[str] = []
        self.dataset_configured = asyncio.Event()

    @on_command(CommandType.PROFILE_CONFIGURE)
    async def _profile_configure_command(
        self, message: ProfileConfigureCommand
    ) -> None:
        """Configure the dataset."""
        self.info("Configuring tokenizer(s) for dataset manager")
        begin = time.perf_counter()
        self._configure_tokenizer()
        self.info(
            lambda: f"Tokenizer(s) configured in {time.perf_counter() - begin:.2f}s"
        )

        self.info(lambda: f"Configuring dataset for {self.service_id}")
        begin = time.perf_counter()
        await self._configure_dataset()
        await self._generate_inputs_json_file()
        self.info(lambda: f"Dataset configured in {time.perf_counter() - begin:.2f}s")

    def _configure_tokenizer(self) -> None:
        """Configure the tokenizer for the dataset manager."""
        tokenizer_name = (
            self.user_config.tokenizer.name or self.user_config.endpoint.model_names[0]
        )
        self.tokenizer = Tokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=self.user_config.tokenizer.trust_remote_code,
            revision=self.user_config.tokenizer.revision,
        )

    def _generate_input_payloads(self, model_endpoint: ModelEndpointInfo) -> InputsFile:
        """Generate input payloads from the dataset for use in the inputs.json file."""
        endpoint = EndpointFactory.create_instance(
            model_endpoint.endpoint.type,
            model_endpoint=model_endpoint,
        )
        self.debug(f"Created endpoint: {endpoint.__class__.__name__}")

        session_payloads: dict[str, list] = defaultdict(list)
        for conversation in self.dataset.values():
            for i, turn in enumerate(conversation.turns):
                request_info = RequestInfo(
                    model_endpoint=model_endpoint, turns=[turn], turn_index=i
                )
                request_info.endpoint_headers = endpoint.get_endpoint_headers(
                    request_info
                )
                request_info.endpoint_params = endpoint.get_endpoint_params(
                    request_info
                )
                session_payloads[conversation.session_id].append(
                    endpoint.format_payload(request_info)
                )

        return InputsFile(
            data=[
                SessionPayloads(session_id=sid, payloads=payloads)
                for sid, payloads in session_payloads.items()
            ]
        )

    async def _generate_inputs_json_file(self) -> None:
        """Generate inputs.json file in the artifact directory."""
        file_path = (
            self.user_config.output.artifact_directory / OutputDefaults.INPUTS_JSON_FILE
        )
        temp_file_path = file_path.with_suffix(".tmp")
        self.info(f"Generating inputs.json file at {file_path.resolve()}")

        try:
            start_time = time.perf_counter()
            file_path.parent.mkdir(parents=True, exist_ok=True)

            model_endpoint = ModelEndpointInfo.from_user_config(self.user_config)
            inputs = self._generate_input_payloads(model_endpoint)

            temp_file_path.write_bytes(
                orjson.dumps(
                    inputs.model_dump(exclude_none=True, mode="json"),
                    option=orjson.OPT_INDENT_2,
                )
            )
            temp_file_path.replace(file_path)

            duration = time.perf_counter() - start_time
            self.info(f"inputs.json file generated in {duration:.2f} seconds")

        except OSError as e:
            self.exception(
                f"Error generating inputs.json file at {file_path.resolve()}: {e!r}"
            )
            # NOTE: We don't raise an error here, as we want to continue to run the benchmark
        except Exception as e:
            # This is a fatal error, as later in the benchmark, errors will occur while trying to convert the payloads
            self.exception(
                f"Error generating inputs.json file at {file_path.resolve()}: {e!r}"
            )
            raise
        finally:
            if temp_file_path.exists():
                temp_file_path.unlink()

    def _load_dataset(self) -> list[Conversation]:
        """Load dataset using the appropriate loader.

        Handles three loading paths in order of precedence:
        1. Public dataset: Download and use specified loader
        2. File-based: Find loader that can handle the input file
        3. Synthetic: Find loader for synthetic data generation

        Returns:
            list[Conversation]: Loaded conversations.

        Raises:
            ValueError: If no loader can handle the configuration.
        """
        # Path 1: Public dataset (has special download logic)
        if self.user_config.input.public_dataset is not None:
            return self._load_public_dataset()

        # Path 2 & 3: Find loader via can_load() auto-detection
        return self._load_via_auto_detection()

    def _load_public_dataset(self) -> list[Conversation]:
        """Load public dataset: download → loader → conversations."""
        from aiperf.dataset.public_datasets import download_public_dataset

        dataset = PublicDatasetFactory.get_instance(
            self.user_config.input.public_dataset
        )
        file_path = download_public_dataset(dataset)

        loader = DatasetLoaderFactory.create_instance(
            dataset.loader_type,
            config=self.user_config,
            tokenizer=self.tokenizer,
            filename=str(file_path),
        )
        self._apply_sampling_strategy(loader)
        return loader.load()

    def _load_via_auto_detection(self) -> list[Conversation]:
        """Find and use a loader via can_load() auto-detection.

        Iterates through all registered loaders and uses the first one
        that returns True from can_load().

        Returns:
            list[Conversation]: Loaded conversations.

        Raises:
            ValueError: If no loader can handle the configuration.
        """
        from aiperf.dataset.utils import check_file_exists

        # Check file exists if specified
        if self.user_config.input.file is not None:
            check_file_exists(self.user_config.input.file)

        # If explicit type is set, use that loader directly
        if self.user_config.input.custom_dataset_type is not None:
            loader_type = DatasetLoaderType(self.user_config.input.custom_dataset_type)
            return self._create_and_load(loader_type)

        # Auto-detect: iterate through loaders and find one that can_load()
        matching_loaders: list[tuple[type, DatasetLoaderType]] = []

        for (
            loader_class,
            loader_type,
        ) in DatasetLoaderFactory.get_all_classes_and_types():
            try:
                if loader_class.can_load(self.user_config):
                    matching_loaders.append((loader_class, loader_type))
                    self.debug(
                        f"Loader {loader_class.__name__} can handle this configuration"
                    )
            except Exception as e:
                self.debug(f"Loader {loader_class.__name__} check failed: {e!r}")

        if not matching_loaders:
            raise ValueError(
                "No loader can handle the data format. "
                "Please specify --custom-dataset-type or --input-file explicitly."
            )

        if len(matching_loaders) > 1:
            loader_names = [cls.__name__ for cls, _ in matching_loaders]
            raise ValueError(
                f"Multiple loaders can handle the data format: {loader_names}. "
                "Please specify --custom-dataset-type explicitly."
            )

        loader_class, loader_type = matching_loaders[0]
        self.info(f"Auto-detected loader: {loader_class.__name__}")
        return self._create_and_load(loader_type)

    def _create_and_load(self, loader_type: DatasetLoaderType) -> list[Conversation]:
        """Create loader instance and load conversations.

        Args:
            loader_type: The type of loader to create.

        Returns:
            list[Conversation]: Loaded conversations.
        """
        # Build kwargs based on whether loader needs filename
        kwargs: dict = {"config": self.user_config, "tokenizer": self.tokenizer}
        if self.user_config.input.file is not None:
            kwargs["filename"] = self.user_config.input.file

        loader = DatasetLoaderFactory.create_instance(loader_type, **kwargs)
        self._apply_sampling_strategy(loader)
        return loader.load()

    def _apply_sampling_strategy(self, loader) -> None:
        """Apply loader's preferred sampling strategy if not explicitly set."""
        if self.user_config.input.dataset_sampling_strategy is None:
            strategy = loader.get_preferred_sampling_strategy()
            self.user_config.input.dataset_sampling_strategy = strategy
            self.info(
                f"Using {loader.__class__.__name__} preferred strategy: {strategy}"
            )

    async def _configure_dataset(self) -> None:
        if self.user_config is None:
            raise self._service_error("User config is required for dataset manager")

        self.dataset_configured.clear()
        self.dataset = {conv.session_id: conv for conv in self._load_dataset()}

        self._dataset_sampler = DatasetSamplingStrategyFactory.create_instance(
            self.user_config.input.dataset_sampling_strategy,
            conversation_ids=list(self.dataset.keys()),
        )
        self.dataset_metadata = DatasetMetadata(
            conversations=[
                conversation.metadata() for conversation in self.dataset.values()
            ],
            sampling_strategy=self.user_config.input.dataset_sampling_strategy,
        )
        metadata = self.dataset_metadata
        self.info(
            f"sampling strategy: {metadata.sampling_strategy}, "
            f"unique conversations: {len(metadata.conversations)}, "
            f"unique turn count: {sum(len(conversation.turns) for conversation in metadata.conversations)}"
        )
        self.dataset_configured.set()
        await self.publish(
            DatasetConfiguredNotification(service_id=self.service_id, metadata=metadata)
        )

    @on_request(MessageType.CONVERSATION_REQUEST)
    async def _handle_conversation_request(
        self, message: ConversationRequestMessage
    ) -> ConversationResponseMessage:
        """Handle a conversation request."""
        self.debug(lambda: f"Handling conversation request: {message}")
        await self._wait_for_dataset_configuration()

        if not self.dataset:
            raise self._service_error("Dataset is empty")

        return self._return_conversation_by_id(
            request_id=message.request_id,
            conversation_id=message.conversation_id,
        )

    def _get_conversation(self, conversation_id: str | None) -> Conversation:
        """Get a conversation by ID, or sample one if ID is None."""
        if conversation_id is None:
            if self._dataset_sampler is None:
                raise self._service_error("Dataset sampler not configured")
            conversation_id = self._dataset_sampler.next_conversation_id()

        if conversation_id not in self.dataset:
            raise self._service_error(f"Conversation {conversation_id} not found")

        return self.dataset[conversation_id]

    @on_request(MessageType.CONVERSATION_TURN_REQUEST)
    async def _handle_conversation_turn_request(
        self, message: ConversationTurnRequestMessage
    ) -> ConversationTurnResponseMessage:
        """Handle a turn request."""
        self.debug(lambda: f"Handling turn request: {message}")

        conversation = self._get_conversation(message.conversation_id)
        if message.turn_index >= len(conversation.turns):
            raise self._service_error(
                f"Turn index {message.turn_index} out of range for {message.conversation_id}"
            )

        self.trace_or_debug(
            lambda: f"Sending turn response: {conversation.turns[message.turn_index]}",
            "Sending turn response",
        )
        return ConversationTurnResponseMessage(
            service_id=self.service_id,
            request_id=message.request_id,
            turn=conversation.turns[message.turn_index],
        )

    async def _wait_for_dataset_configuration(self) -> None:
        """Wait for the dataset to be configured if it is not already."""
        if not self.dataset_configured.is_set():
            self.debug(
                "Dataset not configured. Waiting for dataset to be configured..."
            )
            await asyncio.wait_for(
                self.dataset_configured.wait(),
                timeout=Environment.DATASET.CONFIGURATION_TIMEOUT,
            )


def main() -> None:
    """Main entry point for the dataset manager."""

    from aiperf.common.bootstrap import bootstrap_and_run_service

    bootstrap_and_run_service(DatasetManager)


if __name__ == "__main__":
    main()
