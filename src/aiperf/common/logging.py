# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multiprocess-safe logging infrastructure with Rich console and file output.

This module provides a comprehensive logging system designed for distributed
multiprocess applications. It combines the visual appeal of Rich terminal
formatting with the reliability of multiprocessing-safe log aggregation.

Key Features:
    - Multiprocess Log Queue: Thread-safe singleton queue that aggregates logs
      from child processes to the main process, enabling centralized log viewing
      in dashboard UIs.
    - Rich Console Output: Custom-styled log rendering with timestamps, colored
      log levels, syntax highlighting, and right-aligned logger source information.
    - Adaptive Line Wrapping: Character-level text wrapping that adjusts to
      terminal width with intelligent continuation line indentation.
    - Service-Aware Log Levels: Per-service debug/trace level overrides via
      environment configuration, useful for targeted debugging.
    - Dual Output: Simultaneous console and file logging with configurable
      artifact directories.

Architecture:
    The logging system operates in two modes based on UI type:

    1. Dashboard Mode (queue-based): Child processes forward log records
       to a multiprocessing.Queue, which the main process consumes for display
       in the dashboard UI's log viewer.

    2. Console Mode (direct output): Logs render directly to the terminal
       using Rich formatting, suitable for CLI and non-dashboard UIs.

Usage:
    For child processes::

        from aiperf.common.logging import setup_child_process_logging, get_global_log_queue

        queue = get_global_log_queue()
        setup_child_process_logging(queue, service_id="worker_001", service_config=cfg)

    For main process with Rich output::

        from aiperf.common.logging import setup_rich_logging

        setup_rich_logging(user_config, service_config)

    Cleanup on shutdown::

        await cleanup_global_log_queue()
"""

import asyncio
import logging
import multiprocessing
import queue
import re
import threading
from datetime import datetime
from pathlib import Path

from rich.console import Console, ConsoleRenderable, Group
from rich.highlighter import RegexHighlighter
from rich.logging import RichHandler
from rich.text import Span, Text
from rich.traceback import Traceback

from aiperf.common.aiperf_logger import _DEBUG, _TRACE, AIPerfLogger
from aiperf.common.config import ServiceConfig, ServiceDefaults, UserConfig
from aiperf.common.config.config_defaults import OutputDefaults
from aiperf.common.enums import AIPerfUIType, ServiceType
from aiperf.common.environment import Environment
from aiperf.common.factories import ServiceFactory

_logger = AIPerfLogger(__name__)
_global_log_queue: "multiprocessing.Queue | None" = None
_log_queue_lock = threading.Lock()


def get_global_log_queue() -> multiprocessing.Queue:
    """Get the global log queue. Will create a new queue if it doesn't exist.

    Thread-safe singleton pattern using double-checked locking.
    """
    global _global_log_queue
    if _global_log_queue is None:
        with _log_queue_lock:
            if _global_log_queue is None:
                _global_log_queue = multiprocessing.Queue(
                    maxsize=Environment.LOGGING.QUEUE_MAXSIZE
                )
    return _global_log_queue


async def cleanup_global_log_queue() -> None:
    """Clean up the global log queue to prevent semaphore leaks.

    This should be called during shutdown to properly close and join the queue,
    which releases the internal semaphores used by multiprocessing.Queue.
    Thread-safe.
    """
    global _global_log_queue
    with _log_queue_lock:
        if _global_log_queue is not None:
            try:
                _global_log_queue.close()
                await asyncio.wait_for(
                    asyncio.to_thread(_global_log_queue.join_thread), timeout=1.0
                )
                _logger.debug("Cleaned up global log queue")
            except Exception as e:
                _logger.debug(f"Error cleaning up log queue: {e}")
            finally:
                _global_log_queue = None


def _is_service_in_types(service_id: str, service_types: set[ServiceType]) -> bool:
    """Check if a service is in a set of services."""
    for service_type in service_types:
        # for cases of service_id being "worker_xxxxxx" and service_type being "worker",
        # we want to set the log level to debug
        if (
            service_id == service_type
            or service_id.startswith(f"{service_type}_")
            and service_id
            != f"{service_type}_manager"  # for worker vs worker_manager, etc.
        ):
            return True

        # Check if the provided logger name is the same as the service's class name
        if ServiceFactory.get_class_from_type(service_type).__name__ == service_id:
            return True
    return False


def setup_child_process_logging(
    log_queue: "multiprocessing.Queue | None" = None,
    service_id: str | None = None,
    service_config: ServiceConfig | None = None,
    user_config: UserConfig | None = None,
) -> None:
    """Set up logging for a child process to send logs to the main process.

    This should be called early in child process initialization.

    Args:
        log_queue: The multiprocessing queue to send logs to. If None, tries to get the global queue.
        service_id: The ID of the service to log under. If None, logs will be under the process name.
        service_config: The service configuration used to determine the log level.
        user_config: The user configuration used to determine the log folder.
    """
    root_logger = logging.getLogger()
    level = ServiceDefaults.LOG_LEVEL.upper()
    if service_config:
        level = service_config.log_level.upper()

        if service_id:
            # If the service is in the trace or debug services, set the level to trace or debug
            if Environment.DEV.TRACE_SERVICES and _is_service_in_types(
                service_id, Environment.DEV.TRACE_SERVICES
            ):
                level = _TRACE
            elif Environment.DEV.DEBUG_SERVICES and _is_service_in_types(
                service_id, Environment.DEV.DEBUG_SERVICES
            ):
                level = _DEBUG

    # Set the root logger level to ensure logs are passed to handlers
    root_logger.setLevel(level)

    # Remove all existing handlers to avoid duplicate logs
    for existing_handler in root_logger.handlers[:]:
        root_logger.removeHandler(existing_handler)

    if (
        log_queue is not None
        and service_config
        and service_config.ui_type == AIPerfUIType.DASHBOARD
    ):
        # For dashboard UI, we want to log to the queue, so it can be displayed in the UI
        # log viewer, instead of the console directly.
        queue_handler = MultiProcessLogHandler(log_queue, service_id)
        queue_handler.setLevel(level)
        root_logger.addHandler(queue_handler)
    else:
        # For all other cases, set up custom rich logging to the console
        rich_handler = CustomRichHandler(
            rich_tracebacks=True,
            show_path=False,
            console=Console(),
            show_time=False,
            show_level=False,
            tracebacks_show_locals=False,
        )
        rich_handler.setLevel(level)
        root_logger.addHandler(rich_handler)

    if user_config and user_config.output.artifact_directory:
        file_handler = create_file_handler(
            user_config.output.artifact_directory / OutputDefaults.LOG_FOLDER, level
        )
        root_logger.addHandler(file_handler)


# TODO: Integrate with the subprocess logging instead of being separate
def setup_rich_logging(user_config: UserConfig, service_config: ServiceConfig) -> None:
    """Set up rich logging with appropriate configuration."""
    # Set logging level for the root logger (affects all loggers)
    level = service_config.log_level.upper()
    logging.root.setLevel(level)

    rich_handler = CustomRichHandler(
        rich_tracebacks=True,
        show_path=False,
        console=Console(),
        show_time=False,
        show_level=False,
        tracebacks_show_locals=False,
    )
    logging.root.addHandler(rich_handler)

    # Enable file logging for services
    # TODO: Use config to determine if file logging is enabled and the folder path.
    log_folder = user_config.output.artifact_directory / OutputDefaults.LOG_FOLDER
    log_folder.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_folder / OutputDefaults.LOG_FILE)
    file_handler.setLevel(level)
    file_handler.formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.root.addHandler(file_handler)

    _logger.debug(lambda: f"Logging initialized with level: {level}")


def create_file_handler(
    log_folder: Path,
    level: str | int,
) -> logging.FileHandler:
    """Configure a file handler for logging."""

    log_folder.mkdir(parents=True, exist_ok=True)
    log_file_path = log_folder / OutputDefaults.LOG_FILE

    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    return file_handler


class LogHighlighter(RegexHighlighter):
    """Lightweight highlighter optimized for log messages.

    Faster than ReprHighlighter while still highlighting important patterns:
    URLs, file paths, UUIDs, IPs, numbers, quoted strings, booleans, key=value,
    and Python object repr patterns (class calls, brackets).

    Uses a single mega-regex for maximum performance - one finditer() call
    instead of multiple pattern matches per message.
    """

    base_style = "repr."

    # Single mega-regex combining all patterns for maximum performance
    _MEGA_PATTERN = re.compile(
        r"(?P<url>(?:https?|file|wss?)://[^\s\]\)'\"]+)"  # URLs
        r"|(?P<path>(?<![/\w.~])(?:/[\w._-]+)+/?)"  # Unix paths (not after word/.~/)
        r"|(?P<filename>\b[\w.-]+\.(?:ya?ml|jsonl?|py|log|txt|csv|ipc)\b)"  # Filenames
        r"|(?P<uuid>[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12})"  # Full UUID
        r"|(?P<ipv4>\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b)"  # IPv4
        r"|(?P<number>(?<![.\w])-?\d+\.?\d*(?:(?:e[+-]?\d+)|(?:ns|us|ms|s|m|h))?\b)"  # Numbers (neg allowed at start)
        r"|(?P<str>\"[^\"]*\"|'[^']*'|`[^`]*`)"  # Quoted strings
        r"|\b(?P<bool_true>True)\b|\b(?P<bool_false>False)\b|\b(?P<none>None)\b"  # Booleans
        r"|(?P<brace>[\[\](){}])"  # Brackets
        r"|(?P<call>[A-Z]\w*)\("  # Class calls
        r"|\b(?P<attrib_name>\w+)=(?P<attrib_value>[^\s,=\[\](){}]+)?"  # key=value (word boundary prevents ReDoS)
    )  # fmt: skip

    # Keep highlights for compatibility with base class
    highlights = [_MEGA_PATTERN]

    def highlight(self, text: Text) -> None:
        """Apply syntax highlighting to text using a single mega-regex.

        Mutates the Text object in-place by appending Span objects for each
        matched pattern. Uses a single compiled regex with named groups for
        maximum performance (one finditer call instead of multiple patterns).
        """
        plain = text.plain
        append_span = text._spans.append
        prefix = self.base_style

        for match in self._MEGA_PATTERN.finditer(plain):
            for name, value in match.groupdict().items():
                if value is not None:
                    start, end = match.span(name)
                    if start != -1:
                        append_span(Span(start, end, f"{prefix}{name}"))


class CustomRichHandler(RichHandler):
    """Rich logging handler with adaptive terminal-aware formatting.

    Extends RichHandler to provide a custom log format optimized for readability
    in terminal environments of varying widths. Each log line displays:

    Example::

        HH:MM:SS.mmm LEVEL    message content (logger_name:lineno)

    Features:
        - Millisecond Timestamps: Compact time-only format for dense log output.
        - Colored Log Levels: Each level (TRACE through CRITICAL) has a distinct
          color style for quick visual scanning.
        - Syntax Highlighting: Message content is highlighted using a custom
          LogHighlighter for URLs, paths, numbers, key=value pairs, quoted strings,
          booleans, and Python object patterns. Highlighting is preserved across
          line wrap boundaries.
        - Logger Name Suffix: Appends `(logger_name:lineno)` in dim italic style
          to trace log origins. The logger name is typically a service ID with hex
          suffix (e.g., `dataset_manager_57638ada`), a class name (e.g.,
          `SyntheticDatasetComposer`), or a service type (e.g., `system_controller`).
        - Character-Level Wrapping: Long messages wrap at exact character
          boundaries (not word boundaries) to maximize terminal real estate.
        - Adaptive Indentation: On wide terminals (>=MIN_CONSOLE_INDENT_WRAP_WIDTH,
          default 90 cols), continuation lines indent to align with the first line's
          message. On narrow terminals, continuation lines use full width without
          indentation.
        - Message Truncation: Messages exceeding `MAX_CONSOLE_MESSAGE_LENGTH`
          are truncated to prevent log spam from dominating the console.
        - Rich Tracebacks: Exception tracebacks render with Rich formatting
          when `rich_tracebacks=True`.

    Example Output::

        12:26:52.092 INFO     Tokenizer(s) configured in 2.21 seconds (dataset_manager_57638ada:89)
        12:26:52.279 INFO     Using default sampling strategy for synthetic dataset: shuffle
                              (SyntheticDatasetComposer:29)
        12:26:52.291 INFO     AIPerf System is CONFIGURED (system_controller:193)

    Attributes:
        LOG_LEVEL_STYLES: Mapping of log level names to Rich style strings.
        highlighter: LogHighlighter instance for syntax highlighting messages.
    """

    LOG_LEVEL_STYLES = {
        "TRACE": "dim",
        "DEBUG": "dim",
        "INFO": "cyan",
        "NOTICE": "blue",
        "WARNING": "yellow",
        "SUCCESS": "green",
        "ERROR": "red",
        "CRITICAL": "bold red",
    }

    ABSOLUTE_MIN_CONSOLE_WIDTH = 40
    PREFIX_LENGTH = 22  # "HH:MM:SS.mmm LEVEL    " is fixed at 22 chars

    # Pre-created objects to avoid repeated allocations in hot path
    _NEWLINE = Text("\n")
    _INDENT = Text(" " * PREFIX_LENGTH)

    # Characters that trigger highlighting (quotes, slashes, brackets, equals)
    _HIGHLIGHT_CHARS = frozenset("\"'/:=-()[]{}")

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the handler with a LogHighlighter for message formatting.

        Args:
            *args: Passed to RichHandler.
            **kwargs: Passed to RichHandler. Common options include:
                - rich_tracebacks: Enable Rich-formatted exception tracebacks.
                - console: Rich Console instance for output.
                - show_time/show_level/show_path: Disabled by design; this handler
                  renders its own prefix format.
        """
        super().__init__(*args, **kwargs)
        self.highlighter = LogHighlighter()

    def render(
        self,
        *,
        record: logging.LogRecord,
        traceback: Traceback | None,
        message_renderable: ConsoleRenderable,
    ) -> ConsoleRenderable:
        """Render a log record into a styled, width-aware Rich renderable.

        Constructs the full log line with timestamp prefix, colored level,
        highlighted message content, and dim logger suffix. Handles multi-line
        output via character-level wrapping with optional continuation indentation.

        The message is highlighted BEFORE wrapping, then sliced into wrapped lines.
        This preserves highlighting styles across line boundaries (e.g., a long
        path in `src=/very/long/path` remains styled as `attrib_value` even when
        wrapped across multiple lines).

        Args:
            record: The log record containing message, level, logger name, etc.
            traceback: Optional Rich Traceback to append after the log message.
            message_renderable: Pre-rendered message (unused; we re-render from record).

        Returns:
            A ConsoleRenderable (Text or Group) ready for printing. If a traceback
            is provided, returns a Group containing both the formatted log line
            and the traceback.
        """
        timestamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S.%f")[:-3]
        level_style = self.LOG_LEVEL_STYLES.get(record.levelname, "white")
        message = record.getMessage()[: Environment.LOGGING.MAX_CONSOLE_MESSAGE_LENGTH]
        logger_suffix = f"({record.name}:{record.lineno})"

        # Calculate widths
        console_width = (
            self.console.size.width
            if self.console
            else Environment.LOGGING.DEFAULT_CONSOLE_WIDTH
        )
        target_width = max(console_width - 2, self.ABSOLUTE_MIN_CONSOLE_WIDTH)
        content_width = target_width - self.PREFIX_LENGTH
        indent_continuations = (
            console_width >= Environment.LOGGING.MIN_CONSOLE_INDENT_WRAP_WIDTH
        )
        continuation_width = content_width if indent_continuations else target_width

        # Split by newlines, filter empty segments
        segments = [s for s in message.split("\n") if s] or [""]

        # Fast path: skip expensive regex highlighting for simple messages
        use_highlighter = bool(self._HIGHLIGHT_CHARS & set(message))

        # Build output parts directly
        parts: list[Text] = []
        is_first_line = True
        num_segments = len(segments)

        for seg_idx, segment in enumerate(segments):
            is_last_segment = seg_idx == num_segments - 1
            text_to_wrap = f"{segment} {logger_suffix}" if is_last_segment else segment
            suffix_boundary = (
                len(segment) + 1 if is_last_segment else len(text_to_wrap) + 1
            )

            # Highlight the FULL segment first (with trailing space), then slice for wrapping
            # This preserves styles across wrap boundaries
            msg_with_space = f"{segment} " if is_last_segment else segment
            if use_highlighter:
                highlighted_msg = Text(msg_with_space)
                self.highlighter.highlight(highlighted_msg)
            else:
                highlighted_msg = Text(msg_with_space)

            char_pos = 0

            while char_pos < len(text_to_wrap):
                # Add line prefix or continuation indent
                if is_first_line:
                    parts.append(Text(f"{timestamp} ", style="log.time"))
                    parts.append(Text(f"{record.levelname:<8} ", style=level_style))
                    is_first_line = False
                    line_width = content_width
                else:
                    parts.append(self._NEWLINE)
                    if indent_continuations:
                        parts.append(self._INDENT)
                    line_width = continuation_width

                line_end = min(char_pos + line_width, len(text_to_wrap))

                # Slice from highlighted text, preserving styles across wrap boundaries
                if char_pos >= suffix_boundary:
                    # Entire line is suffix
                    parts.append(
                        Text(text_to_wrap[char_pos:line_end], style="dim italic")
                    )
                elif line_end <= suffix_boundary:
                    # Entire line is message - slice from highlighted text
                    parts.append(highlighted_msg[char_pos:line_end])
                else:
                    # Line spans message/suffix boundary
                    parts.append(highlighted_msg[char_pos:suffix_boundary])
                    parts.append(
                        Text(text_to_wrap[suffix_boundary:line_end], style="dim italic")
                    )

                char_pos = line_end

        formatted_log = Text.assemble(*parts)
        formatted_log.no_wrap = True

        return Group(formatted_log, traceback) if traceback else formatted_log

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record to the console with custom Rich formatting.

        Overrides the default RichHandler emit to use character-level wrapping
        instead of Rich's default word-based soft wrapping. This ensures consistent
        line lengths and prevents unexpected line breaks in the middle of tokens.

        If the record includes exception info and rich_tracebacks is enabled,
        a Rich Traceback is rendered and appended to the output.

        Args:
            record: The log record to emit.
        """
        # Build traceback if exception info is present
        traceback = None
        if (
            self.rich_tracebacks
            and record.exc_info
            and record.exc_info != (None, None, None)
        ):
            traceback = Traceback.from_exception(*record.exc_info)

        # Render and print (render() gets message directly from record)
        log_renderable = self.render(
            record=record, traceback=traceback, message_renderable=Text("")
        )
        self.console.print(log_renderable, soft_wrap=False)


class MultiProcessLogHandler(RichHandler):
    """Custom logging handler that forwards log records to a multiprocessing queue."""

    def __init__(
        self, log_queue: multiprocessing.Queue, service_id: str | None = None
    ) -> None:
        super().__init__()
        self.log_queue = log_queue
        self.service_id = service_id
        self._proc_name = multiprocessing.current_process().name
        self._proc_id = multiprocessing.current_process().pid

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record to the queue."""
        try:
            # Create a serializable log data structure
            log_data = {
                "name": record.name,
                "levelname": record.levelname,
                "levelno": record.levelno,
                "msg": record.getMessage(),
                "created": record.created,
                "process_name": self._proc_name,
                "process_id": self._proc_id,
                "service_id": self.service_id,
            }
            self.log_queue.put_nowait(log_data)
        except queue.Full:
            # Drop logs if queue is full to prevent blocking. Do not log to prevent recursion.
            pass
        except Exception:
            # Do not log to prevent recursion
            pass
