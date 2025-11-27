# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import time

import aiofiles

import aiperf.dataset.public_datasets  # noqa: F401
from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.base_component_service import BaseComponentService
from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.config.config_defaults import OutputDefaults
from aiperf.common.decorators import implements_protocol
from aiperf.common.enums import (
    CommAddress,
    CommandType,
    CustomDatasetType,
    DatasetLoaderType,
    MessageType,
    ServiceType,
)
from aiperf.common.environment import Environment
from aiperf.common.factories import (
    DatasetLoaderFactory,
    DatasetSamplingStrategyFactory,
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
    DatasetTimingRequest,
    DatasetTimingResponse,
    ProfileConfigureCommand,
)
from aiperf.common.mixins import ReplyClientMixin
from aiperf.common.models import Conversation, InputsFile
from aiperf.common.models.dataset_models import SessionPayloads
from aiperf.common.models.model_endpoint_info import ModelEndpointInfo
from aiperf.common.models.record_models import RequestInfo
from aiperf.common.protocols import DatasetSamplingStrategyProtocol, ServiceProtocol
from aiperf.common.tokenizer import Tokenizer

# Import to ensure factory registrations - loaders and datasets self-register on import
from aiperf.dataset.loader import (  # noqa: F401
    MooncakeTraceDatasetLoader,
    MultiTurnDatasetLoader,
    RandomPoolDatasetLoader,
    ShareGPTLoader,
    SingleTurnDatasetLoader,
    SyntheticMultiModalLoader,
    SyntheticRankingsLoader,
)

_logger = AIPerfLogger(__name__)


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
        self.debug("Dataset manager __init__")
        self.user_config = user_config
        self.tokenizer: Tokenizer | None = None
        self.dataset: dict[str, Conversation] = {}  # session ID -> Conversation mapping
        self._session_ids_cache: list[str] = []
        self.dataset_configured = asyncio.Event()
        self._dataset_sampler: DatasetSamplingStrategyProtocol | None = None

    @on_command(CommandType.PROFILE_CONFIGURE)
    async def _profile_configure_command(
        self, message: ProfileConfigureCommand
    ) -> None:
        """Configure the dataset."""

        self.info("Configuring tokenizer(s) for dataset manager")
        begin = time.perf_counter()
        await self._configure_tokenizer()
        duration = time.perf_counter() - begin
        self.info(lambda: f"Tokenizer(s) configured in {duration:.2f} seconds")

        self.info(lambda: f"Configuring dataset for {self.service_id}")
        begin = time.perf_counter()
        await self._configure_dataset()
        await self._generate_inputs_json_file()
        duration = time.perf_counter() - begin
        self.info(lambda: f"Dataset configured in {duration:.2f} seconds")

    async def _configure_tokenizer(self) -> None:
        """Configure the tokenizer for the dataset manager."""
        tokenizer_name = self.user_config.tokenizer.name
        if tokenizer_name is None:
            # TODO: What do we do if there are multiple models?
            # How will we know which tokenizer to use?
            tokenizer_name = self.user_config.endpoint.model_names[0]

        self.tokenizer = Tokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=self.user_config.tokenizer.trust_remote_code,
            revision=self.user_config.tokenizer.revision,
        )

    def _generate_input_payloads(
        self,
        model_endpoint: ModelEndpointInfo,
    ) -> InputsFile:
        """Generate input payloads from the dataset for use in the inputs.json file."""
        inputs = InputsFile()
        from aiperf.common.factories import EndpointFactory
        from aiperf.common.protocols import EndpointProtocol

        endpoint: EndpointProtocol = EndpointFactory.create_instance(
            model_endpoint.endpoint.type,
            model_endpoint=model_endpoint,
        )
        self.debug(
            lambda: f"Created endpoint protocol for {model_endpoint.endpoint.type}, "
            f"class: {endpoint.__class__.__name__}",
        )
        session_payloads_map: dict[str, list] = {}
        for conversation in self.dataset.values():
            session_id = conversation.session_id
            if session_id not in session_payloads_map:
                session_payloads_map[session_id] = []

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
                payload = endpoint.format_payload(request_info)
                session_payloads_map[session_id].append(payload)

        for session_id, payloads in session_payloads_map.items():
            inputs.data.append(
                SessionPayloads(session_id=session_id, payloads=payloads)
            )
        return inputs

    async def _generate_inputs_json_file(self) -> None:
        """Generate inputs.json file in the artifact directory."""
        file_path = (
            self.user_config.output.artifact_directory / OutputDefaults.INPUTS_JSON_FILE
        )
        self.info(f"Generating inputs.json file at {file_path.resolve()}")

        try:
            start_time = time.perf_counter()
            file_path.parent.mkdir(parents=True, exist_ok=True)

            model_endpoint = ModelEndpointInfo.from_user_config(self.user_config)
            inputs = self._generate_input_payloads(model_endpoint)

            async with aiofiles.open(file_path, "w") as f:
                await f.write(inputs.model_dump_json(indent=2, exclude_none=True))

            duration = time.perf_counter() - start_time
            self.info(f"inputs.json file generated in {duration:.2f} seconds")

        except Exception as e:
            # Log as warning, but continue to run the benchmark
            self.warning(
                f"Error generating inputs.json file at {file_path.resolve()}: {e}"
            )

    def _load_public_dataset(self) -> list[Conversation]:
        """Load public dataset: metadata → download → loader → conversations.

        Returns:
            list[Conversation]: Loaded conversations.
        """
        from aiperf.dataset.public_datasets import download_public_dataset

        # 1. Get dataset metadata
        dataset = PublicDatasetFactory.get_instance(
            self.user_config.input.public_dataset
        )

        # 2. Download file
        file_path = download_public_dataset(dataset)

        # 3. Set sampling strategy from loader
        self._set_sampling_strategy_from_loader(dataset.loader_type, dataset.name)

        # 4. Create and run loader
        loader = DatasetLoaderFactory.create_instance(
            dataset.loader_type,
            config=self.user_config,
            tokenizer=self.tokenizer,
            filename=str(file_path),
        )
        return loader.load()

    def _infer_dataset_type(self) -> "CustomDatasetType":
        """Infer the dataset type by querying all registered loaders.

        Iterates through all file loaders and checks which one can handle the
        configured input file.

        Returns:
            CustomDatasetType: The inferred dataset type.

        Raises:
            ValueError: If no loader or multiple loaders can handle the format.
        """
        # Check for explicit type field first by reading first line
        from pathlib import Path

        import orjson

        from aiperf.common.enums import CustomDatasetType

        file_path = Path(self.user_config.input.file)
        if file_path.is_file():
            try:
                with open(file_path) as f:
                    for line in f:
                        if not (line := line.strip()):
                            continue
                        try:
                            data = orjson.loads(line)
                            if "type" in data:
                                explicit_type = CustomDatasetType(data["type"])
                                self.info(f"Using explicit type field: {explicit_type}")
                                return explicit_type
                        except (orjson.JSONDecodeError, ValueError):
                            pass
                        break  # Only check first line
            except Exception as e:
                self.debug(f"Could not read first line for type detection: {e!r}")

        # Iterate through all loaders to find one that can handle this config
        detected_type = None
        for (
            loader_class,
            loader_type,
        ) in DatasetLoaderFactory.get_all_classes_and_types():
            # Only check file loaders (skip synthetic loaders)
            if loader_type in [
                DatasetLoaderType.SYNTHETIC_MULTIMODAL,
                DatasetLoaderType.SYNTHETIC_RANKINGS,
                DatasetLoaderType.SHAREGPT,  # ShareGPT handled via public dataset path
            ]:
                continue

            try:
                if loader_class.can_load(self.user_config):
                    self.info(
                        f"Loader {loader_class.__name__} can handle the input file format."
                    )
                    if detected_type is not None:
                        raise ValueError(
                            f"Multiple loaders can handle the data format: {detected_type} and {loader_type}. "
                            "Please specify --custom-dataset-type explicitly."
                        )
                    # Convert DatasetLoaderType back to CustomDatasetType
                    detected_type = CustomDatasetType(loader_type.value)
            except Exception as e:
                self.debug(f"Loader {loader_class.__name__} cannot handle file: {e!r}")

        if detected_type is None:
            raise ValueError(
                "No loader can handle the data format. Please specify --custom-dataset-type explicitly."
            )

        return detected_type

    def _load_custom_dataset(self) -> list[Conversation]:
        """Load custom dataset: infer type → strategy → loader → conversations.

        Returns:
            list[Conversation]: Loaded conversations.
        """
        from aiperf.dataset.utils import check_file_exists

        check_file_exists(self.user_config.input.file)

        # 1. Determine dataset type (explicit or auto-infer)
        dataset_type = self.user_config.input.custom_dataset_type
        if dataset_type is None:
            dataset_type = self._infer_dataset_type()
            self.info(f"Auto-detected dataset type: {dataset_type}")

        # Convert CustomDatasetType to DatasetLoaderType
        loader_type = DatasetLoaderType(dataset_type.value)

        # 2. Set sampling strategy from loader
        self._set_sampling_strategy_from_loader(loader_type, str(dataset_type))

        # 3. Build loader-specific kwargs
        kwargs = self._build_loader_kwargs(loader_type)

        # 4. Create and run loader
        loader = DatasetLoaderFactory.create_instance(
            loader_type,
            config=self.user_config,
            tokenizer=self.tokenizer,
            filename=self.user_config.input.file,
            **kwargs,
        )
        return loader.load()

    def _is_rankings_endpoint(self, endpoint_type: str) -> bool:
        return "rankings" in endpoint_type.lower()

    def _load_synthetic_dataset(self) -> list[Conversation]:
        endpoint_type = self.user_config.endpoint.type

        if self._is_rankings_endpoint(endpoint_type):
            loader_type = DatasetLoaderType.SYNTHETIC_RANKINGS
        else:
            loader_type = DatasetLoaderType.SYNTHETIC_MULTIMODAL

        loader = DatasetLoaderFactory.create_instance(
            loader_type,
            config=self.user_config,
            tokenizer=self.tokenizer,
        )
        return loader.load()

    def _set_sampling_strategy_from_loader(
        self, loader_type: DatasetLoaderType, source_name: str
    ) -> None:
        """Set sampling strategy from loader's preference if not explicitly set.

        Args:
            loader_type: The loader type to get strategy from
            source_name: Name for logging (e.g., "ShareGPT", "single_turn")
        """
        if self.user_config.input.dataset_sampling_strategy is None:
            loader_class = DatasetLoaderFactory.get_class_from_type(loader_type)
            preferred_strategy = loader_class.get_preferred_sampling_strategy()
            self.user_config.input.dataset_sampling_strategy = preferred_strategy
            self.info(
                f"Using preferred sampling strategy for {source_name}: {preferred_strategy}"
            )

    def _build_loader_kwargs(self, loader_type: DatasetLoaderType) -> dict:
        """Build loader-specific kwargs.

        Args:
            loader_type: The loader type to build kwargs for

        Returns:
            dict: Loader-specific kwargs
        """
        if loader_type == DatasetLoaderType.RANDOM_POOL:
            return {"num_conversations": self.user_config.input.conversation.num}
        elif loader_type == DatasetLoaderType.MOONCAKE_TRACE:
            from aiperf.dataset.generator import PromptGenerator

            return {
                "prompt_generator": PromptGenerator(
                    config=self.user_config.input.prompt,
                    tokenizer=self.tokenizer,
                )
            }
        return {}

    async def _configure_dataset(self) -> None:
        if self.user_config is None:
            raise self._service_error("User config is required for dataset manager")

        self.dataset_configured.clear()

        if self.user_config.input.public_dataset is not None:
            conversations = await self._load_public_dataset()
        elif (
            self.user_config.input.custom_dataset_type is not None
            or self.user_config.input.file is not None
        ):
            # Use CUSTOM composer if either:
            # 1. custom_dataset_type is explicitly set, OR
            # 2. input file is provided (composer will auto-infer type)
            conversations = self._load_custom_dataset()
        else:
            conversations = self._load_synthetic_dataset()

        self.dataset = {conv.session_id: conv for conv in conversations}
        self._session_ids_cache = list(self.dataset.keys())

        self._dataset_sampler = DatasetSamplingStrategyFactory.create_instance(
            self.user_config.input.dataset_sampling_strategy,
            conversation_ids=self._session_ids_cache,
        )

        self.dataset_configured.set()
        await self.publish(DatasetConfiguredNotification(service_id=self.service_id))

    @on_request(MessageType.CONVERSATION_REQUEST)
    async def _handle_conversation_request(
        self, message: ConversationRequestMessage
    ) -> ConversationResponseMessage:
        """Handle a conversation request."""
        self.debug(lambda: f"Handling conversation request: {message}")

        await self._wait_for_dataset_configuration()

        if not self.dataset:
            raise self._service_error(
                "Dataset is empty and must be configured before handling requests.",
            )

        if message.conversation_id is None:
            return self._return_any_conversation(
                request_id=message.request_id,
            )
        else:
            return self._return_conversation_by_id(
                request_id=message.request_id,
                conversation_id=message.conversation_id,
            )

    def _return_any_conversation(
        self, request_id: str | None
    ) -> ConversationResponseMessage:
        """Return any conversation from the dataset based on the user specified method."""

        if self._dataset_sampler is None:
            raise self._service_error(
                "Dataset sampler is not configured. Must be configured before handling requests.",
            )
        session_id = self._dataset_sampler.next_conversation_id()
        conversation = self.dataset[session_id]
        self.trace_or_debug(
            lambda: f"Sending conversation response: {conversation}",
            lambda: f"Sending conversation response with id: {conversation.session_id}",
        )
        return ConversationResponseMessage(
            service_id=self.service_id,
            request_id=request_id,
            conversation=conversation,
        )

    def _return_conversation_by_id(
        self, request_id: str | None, conversation_id: str
    ) -> ConversationResponseMessage:
        """Return a conversation if it exists, otherwise raise an error."""

        if conversation_id not in self.dataset:
            raise self._service_error(
                f"Conversation {conversation_id} not found in dataset.",
            )

        conversation = self.dataset[conversation_id]
        self.trace_or_debug(
            lambda: f"Sending conversation response: {conversation}",
            lambda: f"Sending conversation response with id: {conversation.session_id}",
        )
        return ConversationResponseMessage(
            service_id=self.service_id,
            request_id=request_id,
            conversation=conversation,
        )

    @on_request(MessageType.CONVERSATION_TURN_REQUEST)
    async def _handle_conversation_turn_request(
        self, message: ConversationTurnRequestMessage
    ) -> ConversationTurnResponseMessage:
        """Handle a turn request."""
        self.debug(lambda: f"Handling turn request: {message}")

        if message.conversation_id not in self.dataset:
            raise self._service_error(
                f"Conversation {message.conversation_id} not found in dataset.",
            )

        conversation = self.dataset[message.conversation_id]
        if message.turn_index >= len(conversation.turns):
            raise self._service_error(
                f"Turn index {message.turn_index} is out of range for conversation {message.conversation_id}.",
            )

        turn = conversation.turns[message.turn_index]

        self.trace_or_debug(
            lambda: f"Sending turn response: {turn}",
            "Sending turn response",
        )
        return ConversationTurnResponseMessage(
            service_id=self.service_id,
            request_id=message.request_id,
            turn=turn,
        )

    @on_request(MessageType.DATASET_TIMING_REQUEST)
    async def _handle_dataset_timing_request(
        self, message: DatasetTimingRequest
    ) -> DatasetTimingResponse:
        """Handle a dataset timing request."""
        self.trace_or_debug(
            lambda: f"Handling dataset timing request: {message}",
            "Handling dataset timing request",
        )

        await self._wait_for_dataset_configuration()

        if not self.dataset:
            raise self._service_error(
                "Dataset is empty and must be configured before handling timing requests.",
            )

        timing_dataset = []
        for conversation_id, conversation in self.dataset.items():
            if conversation.turns:
                timing_dataset.append(
                    (conversation.turns[0].timestamp, conversation_id)
                )

        return DatasetTimingResponse(
            service_id=self.service_id,
            request_id=message.request_id,
            timing_data=timing_dataset,
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
