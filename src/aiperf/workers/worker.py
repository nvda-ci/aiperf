# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import time

from aiperf.common.base_component_service import BaseComponentService
from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.enums import (
    CommAddress,
    CommandType,
    CreditPhase,
    ServiceType,
)
from aiperf.common.environment import Environment
from aiperf.common.exceptions import NotInitializedError
from aiperf.common.factories import ServiceFactory
from aiperf.common.hooks import background_task, on_command, on_start, on_stop
from aiperf.common.messages import (
    CommandAcknowledgedResponse,
    ConversationRequestMessage,
    ConversationResponseMessage,
    CreditContext,
    CreditDropMessage,
    ErrorMessage,
    InferenceResultsMessage,
    ProfileCancelCommand,
    WorkerHealthMessage,
    WorkerReadyMessage,
    WorkerShutdownMessage,
)
from aiperf.common.mixins import ProcessHealthMixin
from aiperf.common.models import (
    Conversation,
    ErrorDetails,
    ModelEndpointInfo,
    RequestInfo,
    RequestRecord,
    Text,
    Turn,
    WorkerTaskStats,
)
from aiperf.common.protocols import (
    PushClientProtocol,
    RequestClientProtocol,
    StreamingDealerClientProtocol,
)
from aiperf.workers.inference_client import InferenceClient
from aiperf.workers.session_manager import UserSession, UserSessionManager


@ServiceFactory.register(ServiceType.WORKER)
class Worker(BaseComponentService, ProcessHealthMixin):
    """Worker processes credits from the TimingManager and makes API calls to inference servers.

    Responsibilities:
    - Receives credits via DEALER socket from StickyCreditRouter
    - Processes individual turns (1 credit = 1 turn) with session caching for sticky routing
    - Manages conversation state and assistant responses across turns
    - Sends inference results to RecordProcessor for metric calculation
    - Reports health and task statistics to WorkerManager

    Architecture:

      ┌────────────────────┐
      │ StickyCreditRouter  │
      │   (ROUTER socket)  │
      └────┬──────────▲────┘
           │          │
      CreditDrop CreditReturn
           │          │
           ▼          │
      ┌────────────────────┐
      │  Worker (DEALER)   │
      │                    │
      │  1. Check cache    │
      │  2. Advance session│
      │  3. Build request  │
      └────────┬───────────┘
               │
               ▼
      ┌────────────────────┐
      │  InferenceClient   │
      │  (HTTP/streaming)  │
      └────┬───────────▲───┘
           │           │
           ▼           │
      ┌────────────────────┐
      │  Inference Server  │
      │   (vLLM, TRT-LLM)  │
      └────────────────────┘
                        │
                        └────▶ RecordProcessor
    Credit Flow (All Modes):
    ═══════════════════════════════════════════════════════════════════════════
    1. Credit arrives with x_correlation_id (shared across all turns)
    2. Check session cache:
       - Cache HIT:  Reuse session → Sticky routing working!
       - Cache MISS: Fetch conversation → Create & cache session
    3. Advance session to credit.turn_index
    4. Process single turn, return credit immediately
    5. If final_turn: Evict session from cache

    Example timeline for 3-turn conversation:
    T1: credit[turn=0, x_corr=ABC] → cache MISS → fetch & cache session → return
    T2: credit[turn=1, x_corr=ABC] → cache HIT  → reuse session → return
    T3: credit[turn=2, x_corr=ABC] → cache HIT  → reuse session → evict → return
        └─▶ Same worker processes all turns (StickyCreditRouter sticky routing)

    Session Lifecycle:
    - First turn: Create session from DatasetManager, cache by x_correlation_id
    - Subsequent turns: Retrieve from cache, advance to turn_index
    - Final turn: Process and evict from cache
    - SmartRouter ensures all turns route to same worker for cache hits
    """

    def __init__(
        self,
        service_config: ServiceConfig,
        user_config: UserConfig,
        service_id: str | None = None,
        **kwargs,
    ):
        super().__init__(
            service_config=service_config,
            user_config=user_config,
            service_id=service_id,
            **kwargs,
        )

        self.debug(lambda: f"Worker process __init__ (pid: {self._process.pid})")

        self.task_stats: WorkerTaskStats = WorkerTaskStats()

        self.inference_results_push_client: PushClientProtocol = (
            self.comms.create_push_client(
                CommAddress.RAW_INFERENCE_PROXY_FRONTEND,
            )
        )
        self.conversation_request_client: RequestClientProtocol = (
            self.comms.create_request_client(
                CommAddress.DATASET_MANAGER_PROXY_FRONTEND,
            )
        )

        self.model_endpoint = ModelEndpointInfo.from_user_config(self.user_config)

        self.inference_client: InferenceClient = InferenceClient(
            model_endpoint=self.model_endpoint,
            service_id=self.service_id,
        )
        self.attach_child_lifecycle(self.inference_client)
        self.debug(
            lambda: f"Created inference client for {self.model_endpoint.endpoint.type}, "
            f"class: {self.inference_client.__class__.__name__}",
        )

        self.credit_dealer_client: StreamingDealerClientProtocol = (
            self.comms.create_streaming_dealer_client(
                address=CommAddress.CREDIT_ROUTER,
                identity=self.service_id,  # CRITICAL for ROUTER routing!
                bind=False,
            )
        )
        self.credit_dealer_client.register_receiver(self._on_credit_drop_message)

        self.session_manager: UserSessionManager = UserSessionManager()

    @on_start
    async def _send_worker_ready_message(self) -> None:
        """Send WorkerReadyMessage to announce presence."""
        await self.credit_dealer_client.send(
            WorkerReadyMessage(service_id=self.service_id)
        )

    @on_stop
    async def _send_worker_shutdown_message(self) -> None:
        """Send WorkerShutdownMessage to announce shutdown."""
        try:
            await self.credit_dealer_client.send(
                WorkerShutdownMessage(service_id=self.service_id)
            )
            self.info(
                f"Sent WorkerShutdownMessage for graceful disconnect ({self.service_id})"
            )
        except Exception as e:
            self.debug(
                f"Failed to send shutdown message (already disconnected?): {e!r}"
            )

    @background_task(
        immediate=False,
        interval=Environment.WORKER.HEALTH_CHECK_INTERVAL,
    )
    async def _health_check_task(self) -> None:
        """Task to report the health of the worker to the worker manager."""
        await self.publish(self.create_health_message())

    def create_health_message(self) -> WorkerHealthMessage:
        return WorkerHealthMessage(
            service_id=self.service_id,
            health=self.get_process_health(),
            task_stats=self.task_stats,
        )

    @on_command(CommandType.PROFILE_CANCEL)
    async def _handle_profile_cancel_command(
        self, message: ProfileCancelCommand
    ) -> None:
        self.debug(lambda: f"Received profile cancel command: {message}")
        await self.publish(
            CommandAcknowledgedResponse.from_command_message(message, self.service_id)
        )
        await self.stop()

    async def _on_credit_drop_message(self, message: CreditDropMessage) -> None:
        """Handle incoming credit from TimingManager via StickyCreditRouter.

        Flow:
        1. Create CreditContext with drop timestamp
        2. Process single turn:
           - Check session cache by x_correlation_id
           - If cache miss: Fetch conversation and create session
           - Advance session to turn_index
           - Send request to inference server
        3. ALWAYS return credit in finally block, regardless of success/failure

        Credit return is guaranteed via finally block to ensure accurate concurrency tracking.
        """
        credit_context = CreditContext(
            credit=message.credit,
            drop_perf_ns=time.perf_counter_ns(),
            error=None,
        )
        try:
            if not self.inference_client:
                raise NotInitializedError("Inference server client not initialized.")
            await self._process_credit(credit_context)
        except Exception as e:
            self.exception(
                f"Error occurred while processing credit {message.credit}: {e!r}"
            )
        finally:
            # ALWAYS return the credit here to ensure accurate tracking
            return_message = credit_context.to_return_message(self.service_id)
            await self.credit_dealer_client.send(return_message)

    async def _process_credit(self, credit_context: CreditContext) -> None:
        """Process a credit (1 credit = 1 request).

        Flow:
        1. Pre-create CreditTurnResult with credit.id as x_request_id
        2. Check session cache using x_correlation_id:
           - Cache hit: Reuse session (enables conversation caching on inference server)
           - Cache miss: Retrieve conversation from DatasetManager, create new session
        3. Advance session to current turn index
        4. Process the turn (send request, collect response)
        5. On error: Set error in pre-created result
        6. Finally: Evict session from cache if this is the final turn

        Session Lifecycle:
        - First turn: Session created and cached under x_correlation_id
        - Subsequent turns: Session retrieved from cache (sticky routing ensures same worker)
        - Final turn: Session evicted from cache to free memory
        """
        x_request_id = credit_context.credit.id
        x_correlation_id = credit_context.credit.x_correlation_id
        try:
            session = self.session_manager.get(x_correlation_id)
            if session is None:
                _conversation = await self._retrieve_conversation(
                    conversation_id=credit_context.credit.conversation_id,
                    phase=credit_context.credit.phase,
                    credit_context=credit_context,
                )
                session = self.session_manager.create_and_store(
                    x_correlation_id, _conversation
                )

            session.advance_turn(credit_context.credit.turn_index)

            self.task_stats.total += 1
            request_info: RequestInfo = self._create_request_info(
                session=session,
                credit_context=credit_context,
                x_request_id=x_request_id,
            )
            record: RequestRecord = await self.inference_client.send_request(
                request_info
            )
            await self._send_inference_result_message(record)

            if resp_turn := await self._process_response(record):
                session.store_response(resp_turn)

        except Exception as e:
            credit_context.error = ErrorDetails.from_exception(e)
            self.exception(f"Error processing credit: {e!r}")
        finally:
            if credit_context.credit.is_final_turn:
                self.session_manager.evict(x_correlation_id)

    def _create_request_info(
        self,
        *,
        x_request_id: str,
        session: UserSession,
        credit_context: CreditContext,
    ) -> RequestInfo:
        """Create RequestInfo for inference request with session state and credit metadata.

        Consolidates all information needed by InferenceClient and endpoints to:
        - Format the request payload (model, parameters, conversation history)
        - Set HTTP headers (X-Request-ID, X-Correlation-ID, auth)
        - Track request timing (drop_perf_ns for credit drop latency)
        - Handle cancellation (cancel_after_ns if specified)

        Args:
            x_request_id: Unique ID for this request (X-Request-ID header)
            session: Session containing conversation history and current turn index
            credit_context: Context with credit metadata (num, phase, timestamps)

        Returns:
            RequestInfo with all data needed to send inference request
        """
        credit = credit_context.credit
        return RequestInfo(
            model_endpoint=self.model_endpoint,
            credit_num=credit.num,
            credit_phase=credit.phase,
            cancel_after_ns=credit.cancel_after_ns,
            x_request_id=x_request_id,
            x_correlation_id=session.x_correlation_id,
            conversation_id=session.conversation.session_id,
            turn_index=session.turn_index,
            turns=session.turn_list,
            drop_perf_ns=credit_context.drop_perf_ns,
        )

    async def _retrieve_conversation(
        self,
        *,
        conversation_id: str | None,
        phase: CreditPhase,
        credit_context: CreditContext,
    ) -> Conversation:
        """Retrieve conversation from DatasetManager via request-reply pattern.

        Flow:
        1. Send ConversationRequestMessage to DatasetManager
        2. Wait for response (ConversationResponseMessage or ErrorMessage)
        3. If error: Send error record to RecordProcessor and raise exception
        4. If success: Return Conversation object

        Args:
            conversation_id: ID of conversation to retrieve (from dataset)
            phase: Credit phase (WARMUP or PROFILING)
            credit_context: Credit context with metadata needed for error reporting

        Returns:
            Conversation object with turns and metadata

        Raises:
            ValueError: If DatasetManager returns ErrorMessage or conversation not found

        Note: Errors are sent to RecordProcessor as RequestRecords so they appear
        in final metrics and error summary.
        """
        conversation_response: (
            ConversationResponseMessage | ErrorMessage
        ) = await self.conversation_request_client.request(
            ConversationRequestMessage(
                service_id=self.service_id,
                conversation_id=conversation_id,
                credit_phase=phase,
            )
        )
        if self.is_trace_enabled:
            self.trace(f"Received response message: {conversation_response}")

        # Check for error in conversation response
        if isinstance(conversation_response, ErrorMessage):
            await self._send_inference_result_message(
                RequestRecord(
                    request_info=RequestInfo(
                        model_endpoint=self.model_endpoint,
                        conversation_id=conversation_id,
                        turn_index=0,
                        turns=[],
                        credit_num=credit_context.credit.num,
                        credit_phase=credit_context.credit.phase,
                        x_request_id=credit_context.credit.id,
                        x_correlation_id=credit_context.credit.x_correlation_id,
                        drop_perf_ns=credit_context.drop_perf_ns,
                    ),
                    model_name=self.model_endpoint.primary_model_name,
                    timestamp_ns=time.time_ns(),
                    start_perf_ns=time.perf_counter_ns(),
                    end_perf_ns=time.perf_counter_ns(),
                    error=conversation_response.error,
                )
            )
            raise ValueError("Failed to retrieve conversation response")

        return conversation_response.conversation

    async def _process_response(self, record: RequestRecord) -> Turn | None:
        """Extract assistant response from RequestRecord and convert to Turn for session.

        Flow:
        1. Use endpoint to parse responses into structured data
        2. Extract text content from all responses
        3. If text present: Create Turn with role="assistant"
        4. If no text: Return None (error response or no content)

        Args:
            record: RequestRecord with raw responses from inference server

        Returns:
            Turn object for storing in session, or None if no content
        """
        resp = self.inference_client.endpoint.extract_response_data(record)
        # TODO how do we handle reasoning responses in multi turn?
        resp_text = "".join([r.data.get_text() for r in resp if r.data])
        if resp_text:
            return Turn(
                role="assistant",
                texts=[Text(contents=[resp_text])],
            )
        return None

    async def _send_inference_result_message(self, record: RequestRecord) -> None:
        """Send RequestRecord to RecordProcessor for metric calculation.

        All records (success and error) flow through this method to ensure consistent
        metric calculation and error tracking.

        Flow:
        1. Update task statistics (total and success/failure counts)
        2. Wrap record in InferenceResultsMessage
        3. Push to RecordProcessor via PUSH socket (fire-and-forget)

        Note: Uses execute_async() to avoid blocking on network I/O.
        """
        # All records will flow through here to be sent to the inference results push client.
        self.task_stats.task_finished(record.valid)

        msg = InferenceResultsMessage(
            service_id=self.service_id,
            record=record,
        )
        self.execute_async(self.inference_results_push_client.push(msg))


def main() -> None:
    from aiperf.common.bootstrap import bootstrap_and_run_service

    bootstrap_and_run_service(Worker)


if __name__ == "__main__":
    main()
