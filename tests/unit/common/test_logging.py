# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
import sys
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console, Group
from rich.text import Text
from rich.traceback import Traceback

from aiperf.common.logging import CustomRichHandler, LogHighlighter

# The prefix format is "{timestamp} {level:<8} " where timestamp is "HH:MM:SS.mmm"
# Example: "14:32:01.456 INFO     " = 12 + 1 + 8 + 1 = 22 chars
PREFIX_LENGTH = 22


def make_log_record(
    msg: str = "Test message",
    level: int = logging.INFO,
    name: str = "test_logger",
    lineno: int = 42,
    exc_info: tuple | None = None,
) -> logging.LogRecord:
    """Factory for creating LogRecord instances with sensible defaults."""
    return logging.LogRecord(
        name=name,
        level=level,
        pathname="test.py",
        lineno=lineno,
        msg=msg,
        args=(),
        exc_info=exc_info,
    )


def render_to_str(handler: CustomRichHandler, record: logging.LogRecord) -> str:
    """Render a log record and return the string representation."""
    result = handler.render(record=record, traceback=None, message_renderable=Text(""))
    return str(result)


def calculate_suffix_length(name: str, lineno: int) -> int:
    """Calculate the length of the logger suffix '(name:lineno)'."""
    return len(f"({name}:{lineno})")


@pytest.fixture
def mock_console() -> MagicMock:
    """Create a mock console with configurable width."""
    console = MagicMock(spec=Console)
    console.size = MagicMock()
    console.size.width = 120
    return console


@pytest.fixture
def handler(mock_console: MagicMock) -> CustomRichHandler:
    """Create a CustomRichHandler with a mock console."""
    return CustomRichHandler(
        rich_tracebacks=True,
        show_path=False,
        console=mock_console,
        show_time=False,
        show_level=False,
        tracebacks_show_locals=False,
    )


@pytest.fixture
def log_record() -> logging.LogRecord:
    """Create a basic log record for testing."""
    return make_log_record()


class TestCustomRichHandlerLogLevelStyles:
    """Test cases for LOG_LEVEL_STYLES dictionary."""

    @pytest.mark.parametrize(
        "level,expected_style",
        [
            ("TRACE", "dim"),
            ("DEBUG", "dim"),
            ("INFO", "cyan"),
            ("NOTICE", "blue"),
            ("WARNING", "yellow"),
            ("SUCCESS", "green"),
            ("ERROR", "red"),
            ("CRITICAL", "bold red"),
        ],
    )
    def test_log_level_style_mapping(self, level: str, expected_style: str):
        """Test that each log level maps to the correct style."""
        assert CustomRichHandler.LOG_LEVEL_STYLES[level] == expected_style

    def test_log_level_styles_is_complete(self):
        """Test that all expected log levels have styles defined."""
        expected_levels = {
            "TRACE",
            "DEBUG",
            "INFO",
            "NOTICE",
            "WARNING",
            "SUCCESS",
            "ERROR",
            "CRITICAL",
        }
        assert set(CustomRichHandler.LOG_LEVEL_STYLES.keys()) == expected_levels


class TestCustomRichHandlerInitialization:
    """Test cases for CustomRichHandler initialization."""

    def test_initializes_with_log_highlighter(self, handler: CustomRichHandler):
        """Test that handler initializes with LogHighlighter."""
        from aiperf.common.logging import LogHighlighter

        assert isinstance(handler.highlighter, LogHighlighter)


class TestCustomRichHandlerRender:
    """Test cases for CustomRichHandler.render method."""

    def test_render_returns_text_for_simple_message(
        self, handler: CustomRichHandler, log_record: logging.LogRecord
    ):
        """Test that render returns Text for a simple log message."""
        result = handler.render(
            record=log_record, traceback=None, message_renderable=Text("")
        )
        assert isinstance(result, Text)

    def test_render_returns_group_with_traceback(
        self, handler: CustomRichHandler, log_record: logging.LogRecord
    ):
        """Test that render returns Group when traceback is present."""
        mock_traceback = MagicMock(spec=Traceback)
        result = handler.render(
            record=log_record, traceback=mock_traceback, message_renderable=Text("")
        )
        assert isinstance(result, Group)

    @pytest.mark.parametrize(
        "expected_content,description",
        [
            (":", "timestamp with colons"),
            ("INFO", "log level name"),
            ("(test_logger:42)", "logger suffix with name and line"),
            ("Test message", "log message content"),
        ],
    )
    def test_render_includes_expected_content(
        self,
        handler: CustomRichHandler,
        log_record: logging.LogRecord,
        expected_content: str,
        description: str,
    ):
        """Test that rendered output includes expected content."""
        rendered_str = render_to_str(handler, log_record)
        assert expected_content in rendered_str, f"Missing {description}"

    @pytest.mark.parametrize(
        "level_name,level",
        [
            ("DEBUG", logging.DEBUG),
            ("INFO", logging.INFO),
            ("WARNING", logging.WARNING),
            ("ERROR", logging.ERROR),
            ("CRITICAL", logging.CRITICAL),
        ],
    )
    def test_render_with_different_levels(
        self, handler: CustomRichHandler, level_name: str, level: int
    ):
        """Test rendering with different log levels."""
        record = make_log_record(msg="Message", level=level, name="test", lineno=1)
        rendered_str = render_to_str(handler, record)
        assert level_name in rendered_str

    def test_render_sets_no_wrap_on_text(
        self, handler: CustomRichHandler, log_record: logging.LogRecord
    ):
        """Test that render sets no_wrap=True on the result to prevent Rich re-wrapping."""
        result = handler.render(
            record=log_record, traceback=None, message_renderable=Text("")
        )
        assert isinstance(result, Text)
        assert result.no_wrap is True


class TestCustomRichHandlerMessageTruncation:
    """Test cases for message truncation behavior."""

    @pytest.mark.parametrize(
        "msg_length,max_length,should_truncate",
        [
            (100, 50, True),
            (50, 50, False),
            (49, 50, False),
            (51, 50, True),
            (200, 100, True),
        ],
    )
    def test_message_truncation(
        self,
        handler: CustomRichHandler,
        msg_length: int,
        max_length: int,
        should_truncate: bool,
    ):
        """Test message truncation at various lengths.

        The code truncates via: message = record.getMessage()[:MAX_CONSOLE_MESSAGE_LENGTH]
        So a 100-char message with max_length=50 becomes exactly 50 chars.
        """
        with patch(
            "aiperf.common.logging.Environment.LOGGING.MAX_CONSOLE_MESSAGE_LENGTH",
            max_length,
        ):
            full_message = "A" * msg_length
            record = make_log_record(msg=full_message, lineno=1, name="test")
            rendered_str = render_to_str(handler, record)
            # Remove newlines and whitespace from wrapping for character count
            content_only = rendered_str.replace("\n", "").replace(" ", "")

            if should_truncate:
                # Full message should not appear (accounting for wrapping)
                assert full_message not in rendered_str.replace("\n", "").replace(
                    " ", ""
                )
                # Truncated content should have exactly max_length A's
                assert content_only.count("A") == max_length
            else:
                assert content_only.count("A") == msg_length


class TestCustomRichHandlerWrapping:
    """Test cases for line wrapping behavior.

    The render() method calculates widths as:
        target_width = max(console_width - 2, 40)
        content_width = target_width - prefix_len  (prefix_len ~= 22)

    For indent behavior:
        indent_continuations = console_width >= MIN_CONSOLE_INDENT_WRAP_WIDTH

    When indented, continuation lines get prefix_len spaces prepended.
    When not indented, continuation lines use full target_width.
    """

    @pytest.mark.parametrize(
        "console_width,min_indent_width,expect_indent",
        [
            # Below threshold - no indent
            (60, 90, False),
            (89, 90, False),
            # At threshold boundary - should indent (>= comparison)
            (90, 90, True),
            # Above threshold - indent
            (91, 90, True),
            (150, 90, True),
            # Different thresholds
            (100, 100, True),
            (99, 100, False),
            (50, 50, True),
            (49, 50, False),
        ],
    )
    def test_indent_wrapping_threshold(
        self,
        handler: CustomRichHandler,
        mock_console: MagicMock,
        console_width: int,
        min_indent_width: int,
        expect_indent: bool,
    ):
        """Test indentation behavior at various console width thresholds.

        Code logic: indent_continuations = console_width >= MIN_CONSOLE_INDENT_WRAP_WIDTH
        When True, continuation lines get ' ' * prefix_len prepended.
        """
        mock_console.size.width = console_width
        with patch(
            "aiperf.common.logging.Environment.LOGGING.MIN_CONSOLE_INDENT_WRAP_WIDTH",
            min_indent_width,
        ):
            record = make_log_record(msg="X" * 300, lineno=1, name="test")
            result = handler.render(
                record=record, traceback=None, message_renderable=Text("")
            )
            rendered_str = str(result)

            assert "\n" in rendered_str, "Message should wrap"

            lines = rendered_str.split("\n")
            assert len(lines) > 1, "Should have continuation lines"

            continuation_line = lines[1]
            if expect_indent:
                # Continuation should have exactly PREFIX_LENGTH spaces
                leading_spaces = len(continuation_line) - len(
                    continuation_line.lstrip(" ")
                )
                assert leading_spaces == PREFIX_LENGTH, (
                    f"Expected {PREFIX_LENGTH} spaces, got {leading_spaces}"
                )
            else:
                # Continuation should NOT start with spaces
                assert not continuation_line.startswith(" "), (
                    "Narrow console should not indent continuation lines"
                )

    @pytest.mark.parametrize(
        "console_width,msg_length,name,lineno,expected_lines",
        [
            # Wide console (120): target=118, content_width=96
            # Short message fits on one line
            (120, 10, "t", 1, 1),
            # Message + suffix fits: 70 + 1 + 5 = 76 chars < 96
            (120, 70, "t", 1, 1),
            # Message + suffix wraps once: 100 + 1 + 5 = 106 chars / 96 = 2 lines
            (120, 100, "t", 1, 2),
            # Long message: 200 + 1 + 5 = 206 / 96 = 3 lines
            (120, 200, "t", 1, 3),
            # Narrow console (60): target=58, content_width=36
            # 50 + 1 + 5 = 56 / 36 = 2 lines (no indent, so continuation gets full 58)
            # Actually first line is 36, continuation lines are 58: 36 + 20 = 56 = 1 line?
            # Let's be more precise: 56 chars total, first line 36, remaining 20, fits in 58 = 2 lines
            (60, 50, "t", 1, 2),
        ],
    )
    def test_line_wrapping_count(
        self,
        handler: CustomRichHandler,
        mock_console: MagicMock,
        console_width: int,
        msg_length: int,
        name: str,
        lineno: int,
        expected_lines: int,
    ):
        """Test that messages wrap to expected number of lines.

        Calculations:
            target_width = max(console_width - 2, 40)
            content_width = target_width - PREFIX_LENGTH
            full_content = message + ' ' + suffix

        For wide consoles (indent), continuation lines also use content_width.
        For narrow consoles (no indent), continuation lines use target_width.
        """
        mock_console.size.width = console_width
        record = make_log_record(msg="M" * msg_length, lineno=lineno, name=name)
        rendered_str = render_to_str(handler, record)
        actual_lines = len(rendered_str.split("\n"))
        assert actual_lines == expected_lines, (
            f"Expected {expected_lines} lines, got {actual_lines} for msg_len={msg_length}, "
            f"console_width={console_width}"
        )

    def test_uses_default_console_width_when_console_is_none(
        self, handler: CustomRichHandler, mock_console: MagicMock
    ):
        """Test that DEFAULT_CONSOLE_WIDTH is used when console is None.

        Code: console_width = self.console.size.width if self.console else DEFAULT_CONSOLE_WIDTH

        We verify by comparing line counts with different default widths for the same long message.
        A wider default should result in fewer lines.
        """
        handler.console = None
        long_msg = "X" * 300

        with patch(
            "aiperf.common.logging.Environment.LOGGING.DEFAULT_CONSOLE_WIDTH", 200
        ):
            record = make_log_record(msg=long_msg, lineno=1, name="test")
            wide_result = render_to_str(handler, record)
            wide_lines = len(wide_result.split("\n"))

        with patch(
            "aiperf.common.logging.Environment.LOGGING.DEFAULT_CONSOLE_WIDTH", 60
        ):
            record = make_log_record(msg=long_msg, lineno=1, name="test")
            narrow_result = render_to_str(handler, record)
            narrow_lines = len(narrow_result.split("\n"))

        # Wider console = fewer lines
        assert wide_lines < narrow_lines, (
            f"Wider console should have fewer lines: {wide_lines} vs {narrow_lines}"
        )

    @pytest.mark.parametrize(
        "console_width,expected_target_width",
        [
            (30, 40),  # max(28, 40) = 40 (floored)
            (41, 40),  # max(39, 40) = 40 (floored)
            (42, 40),  # max(40, 40) = 40 (boundary)
            (43, 41),  # max(41, 40) = 41 (above floor)
            (50, 48),  # max(48, 40) = 48
            (100, 98),  # max(98, 40) = 98
        ],
    )
    def test_target_width_floor_is_40(
        self,
        handler: CustomRichHandler,
        mock_console: MagicMock,
        console_width: int,
        expected_target_width: int,
    ):
        """Test that target_width has a minimum floor of 40.

        Code: target_width = max(console_width - 2, 40)

        We verify by comparing line counts for console widths at and below the floor.
        Widths that floor to 40 should produce identical output.
        """
        mock_console.size.width = console_width
        record = make_log_record(msg="A" * 100, lineno=1, name="test")
        result = handler.render(
            record=record, traceback=None, message_renderable=Text("")
        )
        rendered_str = str(result)

        lines = rendered_str.split("\n")
        # Verify no empty lines (which would happen if content_width <= 0)
        for line in lines:
            assert len(line) > 0, "Should not have empty lines"

        # Verify it produces output and wraps (doesn't crash on narrow console)
        assert len(lines) >= 1, "Should produce at least one line"

        # For floored widths (30, 41, 42), all should have same target_width of 40
        # So they should produce the same number of lines
        if expected_target_width == 40:
            # First line uses content_width = 40 - 22 = 18
            # Continuation lines use target_width = 40 (no indent since < 90)
            # Full content = 100 + 1 + 8 = 109 chars
            # Line 1: 18, Line 2: 40, Line 3: 40, Line 4: 11 = 4 lines
            assert len(lines) == 4, (
                f"Floored width should produce 4 lines, got {len(lines)}"
            )


class TestCustomRichHandlerEmit:
    """Test cases for CustomRichHandler.emit method."""

    @pytest.mark.parametrize(
        "msg,level",
        [
            ("Simple message", logging.INFO),
            ("Debug info", logging.DEBUG),
            ("Warning!", logging.WARNING),
            ("Error occurred", logging.ERROR),
            ("", logging.INFO),  # Empty message
            ("A" * 500, logging.INFO),  # Long message
        ],
    )
    def test_emit_prints_to_console_with_soft_wrap_disabled(
        self, handler: CustomRichHandler, mock_console: MagicMock, msg: str, level: int
    ):
        """Test that emit prints to console with soft_wrap=False.

        Code: self.console.print(log_renderable, soft_wrap=False)

        soft_wrap=False prevents Rich from re-wrapping our carefully formatted output.
        """
        record = make_log_record(msg=msg, level=level, lineno=1, name="test")
        handler.emit(record)
        mock_console.print.assert_called_once()
        _, kwargs = mock_console.print.call_args
        assert kwargs.get("soft_wrap") is False

    def test_emit_with_exception_creates_traceback(
        self, handler: CustomRichHandler, mock_console: MagicMock
    ):
        """Test that emit creates a Traceback when exc_info is present.

        Code: if self.rich_tracebacks and record.exc_info and record.exc_info != (None, None, None):
                  traceback = Traceback.from_exception(*record.exc_info)
        """
        try:
            raise ValueError("Test error")
        except ValueError:
            exc_info = sys.exc_info()

        record = make_log_record(
            msg="Error occurred",
            level=logging.ERROR,
            lineno=1,
            name="test",
            exc_info=exc_info,
        )
        handler.emit(record)
        mock_console.print.assert_called_once()
        call_args = mock_console.print.call_args[0]
        # When there's a traceback, render() returns Group(formatted_log, traceback)
        assert isinstance(call_args[0], Group)

    @pytest.mark.parametrize(
        "rich_tracebacks,expected_type",
        [
            (True, Group),  # With rich_tracebacks=True, exception produces Group
            (False, Text),  # With rich_tracebacks=False, no Traceback is created
        ],
    )
    def test_emit_respects_rich_tracebacks_setting(
        self, mock_console: MagicMock, rich_tracebacks: bool, expected_type: type
    ):
        """Test that emit respects the rich_tracebacks constructor argument.

        Code: if self.rich_tracebacks and record.exc_info and ...:
                  traceback = Traceback.from_exception(...)
        """
        handler = CustomRichHandler(
            rich_tracebacks=rich_tracebacks, console=mock_console
        )
        try:
            raise ValueError("Test error")
        except ValueError:
            exc_info = sys.exc_info()

        record = make_log_record(
            msg="Error", level=logging.ERROR, lineno=1, name="test", exc_info=exc_info
        )
        handler.emit(record)
        mock_console.print.assert_called_once()
        call_args = mock_console.print.call_args[0]
        assert isinstance(call_args[0], expected_type)

    @pytest.mark.parametrize(
        "has_exc_info,expected_type",
        [
            (True, Group),  # With exception, produces Group
            (False, Text),  # Without exception, produces Text
        ],
    )
    def test_emit_output_type_based_on_exception_presence(
        self,
        handler: CustomRichHandler,
        mock_console: MagicMock,
        has_exc_info: bool,
        expected_type: type,
    ):
        """Test that emit returns correct type based on exception presence.

        With exc_info: render() is called with a Traceback, returns Group
        Without exc_info: render() is called with traceback=None, returns Text
        """
        exc_info = None
        if has_exc_info:
            try:
                raise ValueError("Test")
            except ValueError:
                exc_info = sys.exc_info()

        record = make_log_record(
            msg="Message", level=logging.ERROR, lineno=1, name="test", exc_info=exc_info
        )
        handler.emit(record)
        call_args = mock_console.print.call_args[0]
        assert isinstance(call_args[0], expected_type)


class TestCustomRichHandlerEmbeddedNewlines:
    """Test cases for messages containing embedded newlines.

    Messages can contain newlines from multiline strings, exception messages,
    or formatted output. The handler should properly indent each line segment
    rather than letting raw newlines break the formatting.
    """

    def test_single_newline_creates_two_lines(
        self, handler: CustomRichHandler, mock_console: MagicMock
    ):
        """Test that a single embedded newline creates two output lines."""
        mock_console.size.width = 120
        record = make_log_record(msg="Line1\nLine2", lineno=1, name="test")
        rendered_str = render_to_str(handler, record)
        lines = rendered_str.split("\n")
        assert len(lines) == 2

    def test_multiple_newlines_create_multiple_lines(
        self, handler: CustomRichHandler, mock_console: MagicMock
    ):
        """Test that multiple embedded newlines create correct number of lines."""
        mock_console.size.width = 120
        record = make_log_record(
            msg="Line1\nLine2\nLine3\nLine4", lineno=1, name="test"
        )
        rendered_str = render_to_str(handler, record)
        lines = rendered_str.split("\n")
        assert len(lines) == 4

    def test_newline_continuation_lines_are_indented_on_wide_console(
        self, handler: CustomRichHandler, mock_console: MagicMock
    ):
        """Test that lines after newlines get proper indentation on wide consoles.

        On wide consoles (>= MIN_CONSOLE_INDENT_WRAP_WIDTH), continuation lines
        should be indented with PREFIX_LENGTH spaces to align with the message.
        """
        mock_console.size.width = 120
        record = make_log_record(msg="First\nSecond\nThird", lineno=1, name="test")
        rendered_str = render_to_str(handler, record)
        lines = rendered_str.split("\n")

        assert len(lines) == 3
        # First line has timestamp prefix
        assert "INFO" in lines[0]
        # Continuation lines should be indented
        assert lines[1].startswith(" " * PREFIX_LENGTH)
        assert lines[2].startswith(" " * PREFIX_LENGTH)

    def test_newline_continuation_lines_not_indented_on_narrow_console(
        self, handler: CustomRichHandler, mock_console: MagicMock
    ):
        """Test that lines after newlines have no indent on narrow consoles.

        On narrow consoles (< MIN_CONSOLE_INDENT_WRAP_WIDTH), continuation lines
        should NOT be indented to maximize usable width.
        """
        mock_console.size.width = 60
        record = make_log_record(msg="First\nSecond\nThird", lineno=1, name="test")
        rendered_str = render_to_str(handler, record)
        lines = rendered_str.split("\n")

        assert len(lines) == 3
        # Continuation lines should NOT start with spaces
        assert not lines[1].startswith(" ")
        assert not lines[2].startswith(" ")

    def test_suffix_appears_only_at_end_with_newlines(
        self, handler: CustomRichHandler, mock_console: MagicMock
    ):
        """Test that the logger suffix (name:lineno) only appears at the end."""
        mock_console.size.width = 120
        record = make_log_record(msg="Line1\nLine2\nLine3", lineno=42, name="mylogger")
        rendered_str = render_to_str(handler, record)

        # Suffix should appear exactly once, at the end
        assert rendered_str.count("(mylogger:42)") == 1
        assert rendered_str.endswith("(mylogger:42)")

    def test_empty_segments_from_consecutive_newlines_are_collapsed(
        self, handler: CustomRichHandler, mock_console: MagicMock
    ):
        """Test that consecutive newlines collapse empty segments.

        Empty segments are skipped to avoid visual clutter in logs.
        "Before\n\nAfter" produces 2 lines, not 3.
        """
        mock_console.size.width = 120
        record = make_log_record(msg="Before\n\nAfter", lineno=1, name="test")
        rendered_str = render_to_str(handler, record)
        lines = rendered_str.split("\n")

        # Empty segment is skipped, so only 2 lines
        assert len(lines) == 2
        assert "Before" in lines[0]
        assert "After" in lines[1]

    def test_leading_newline_is_skipped(
        self, handler: CustomRichHandler, mock_console: MagicMock
    ):
        """Test that leading empty segment from newline at start is skipped."""
        mock_console.size.width = 120
        record = make_log_record(msg="\nStartsWithNewline", lineno=1, name="test")
        rendered_str = render_to_str(handler, record)
        lines = rendered_str.split("\n")

        # Leading empty segment is skipped
        assert len(lines) == 1
        assert "StartsWithNewline" in lines[0]
        assert "INFO" in lines[0]

    def test_trailing_newline_is_ignored(
        self, handler: CustomRichHandler, mock_console: MagicMock
    ):
        """Test that trailing newline doesn't create extra empty line."""
        mock_console.size.width = 120
        record = make_log_record(msg="EndsWithNewline\n", lineno=1, name="test")
        rendered_str = render_to_str(handler, record)
        lines = rendered_str.split("\n")

        # Trailing empty segment is skipped, so only 1 line
        assert len(lines) == 1
        assert "(test:1)" in lines[0]
        assert "EndsWithNewline" in lines[0]

    @pytest.mark.parametrize(
        "msg,expected_line_count",
        [
            ("No newlines", 1),
            ("One\nNewline", 2),
            ("Two\nNew\nlines", 3),
            ("\nStarting newline", 1),  # Leading empty segment skipped
            ("Ending newline\n", 1),  # Trailing empty segment skipped
            ("\n\n\n", 1),  # All empty segments skipped, just suffix remains
            ("1\n2\n3\n4\n5", 5),
            ("X\n\nY\n\nZ", 3),  # Empty segments between content are skipped
        ],
    )
    def test_newline_count_variations(
        self,
        handler: CustomRichHandler,
        mock_console: MagicMock,
        msg: str,
        expected_line_count: int,
    ):
        """Test various newline patterns produce correct line counts.

        Empty segments (from consecutive newlines or leading/trailing newlines)
        are skipped to avoid visual clutter.
        """
        mock_console.size.width = 120
        record = make_log_record(msg=msg, lineno=1, name="test")
        rendered_str = render_to_str(handler, record)
        lines = rendered_str.split("\n")
        assert len(lines) == expected_line_count

    def test_long_line_with_newlines_wraps_correctly(
        self, handler: CustomRichHandler, mock_console: MagicMock
    ):
        """Test that long segments after newlines also wrap correctly."""
        mock_console.size.width = 120
        # First segment short, second segment long enough to wrap
        long_segment = "Y" * 150
        record = make_log_record(msg=f"Short\n{long_segment}", lineno=1, name="t")
        rendered_str = render_to_str(handler, record)
        lines = rendered_str.split("\n")

        # Should have: 1 for "Short" + 2 for wrapped long segment = 3+ lines
        assert len(lines) >= 3
        # All lines after first should be indented (wide console)
        for line in lines[1:]:
            assert line.startswith(" " * PREFIX_LENGTH)

    def test_wrapping_preserves_highlighting_across_boundaries(
        self, handler: CustomRichHandler, mock_console: MagicMock
    ):
        """Test that highlighting styles are preserved when text wraps.

        When a highlighted pattern (like path=/very/long/path) wraps across
        multiple lines, both parts should retain the same style. Previously,
        each wrapped chunk was highlighted independently, causing the second
        part to be incorrectly identified as a standalone 'path' instead of
        continuing as 'attrib_value'.
        """
        # Force narrow width to ensure wrapping
        mock_console.size.width = 60
        record = make_log_record(
            msg="path=/very/long/path/to/some/file/that/will/wrap.txt",
            lineno=1,
            name="t",
        )
        rendered = handler.render(
            record=record, traceback=None, message_renderable=Text("")
        )

        # Get all spans with attrib_value style
        attrib_value_spans = [s for s in rendered._spans if "attrib_value" in s.style]

        # Should have attrib_value spans (the path after =)
        assert len(attrib_value_spans) >= 1, "Expected attrib_value highlighting"

        # If message wrapped, there should be multiple attrib_value spans
        # all with the same style (not one as attrib_value and one as path)
        path_spans = [s for s in rendered._spans if s.style == "repr.path"]
        # The path should NOT be re-highlighted as a standalone path
        # (it should remain as attrib_value across the wrap)
        for span in path_spans:
            matched_text = rendered.plain[span.start : span.end]
            # Standalone path matches should start with / and not be part of key=value
            # After = we expect attrib_value, not path
            assert "=" not in rendered.plain[: span.start] or matched_text.startswith(
                "/"
            ), f"Path incorrectly split from attrib_value: {matched_text}"


class TestLogHighlighterCorrectness:
    """Correctness tests for LogHighlighter regex patterns.

    Tests adversarial inputs to ensure no false positives or incorrect matches.
    """

    @pytest.fixture
    def highlighter(self) -> LogHighlighter:
        return LogHighlighter()

    def get_matches(
        self, highlighter: LogHighlighter, msg: str
    ) -> list[tuple[str, str]]:
        """Extract matches as (matched_text, style_suffix) tuples."""
        from rich.text import Text

        text = Text(msg)
        highlighter.highlight(text)
        return [(msg[s.start : s.end], s.style.split(".")[-1]) for s in text._spans]

    def has_style(
        self, matches: list[tuple[str, str]], style: str, text: str | None = None
    ) -> bool:
        """Check if a style appears in matches, optionally for specific text."""
        return any(m[1] == style and (text is None or m[0] == text) for m in matches)

    # --- False Positive Path Tests ---

    @pytest.mark.parametrize(
        "msg",
        [
            "a/b/c",  # Not an absolute path
            "input/output",  # Words with slash
            "10/20",  # Numbers with slash (date-like)
            "./config.yaml",  # Relative path
            "../parent",  # Parent directory
            "~/Documents",  # Tilde expansion
        ],
    )
    def test_no_false_positive_paths(self, highlighter: LogHighlighter, msg: str):
        """Ensure non-absolute paths are not matched as paths."""
        matches = self.get_matches(highlighter, msg)
        # Should not have any path match that doesn't start with /
        false_paths = [
            m for m in matches if m[1] == "path" and not m[0].startswith("/")
        ]
        assert not false_paths, f"False positive path matches: {false_paths}"

    @pytest.mark.parametrize(
        "msg,expected_path",
        [
            ("/path/to/file", "/path/to/file"),
            ("/etc/config.yaml", "/etc/config.yaml"),
            ("/home/user/.config", "/home/user/.config"),
        ],
    )
    def test_real_paths_matched(
        self, highlighter: LogHighlighter, msg: str, expected_path: str
    ):
        """Ensure real absolute paths are correctly matched."""
        matches = self.get_matches(highlighter, msg)
        assert self.has_style(matches, "path", expected_path)

    def test_path_in_key_value_matched_as_value(self, highlighter: LogHighlighter):
        """Paths after = are matched as attrib_value, not path."""
        matches = self.get_matches(highlighter, "src=/etc/config")
        assert self.has_style(matches, "attrib_name", "src")
        assert self.has_style(matches, "attrib_value", "/etc/config")

    # --- Key=Value Edge Cases ---

    def test_normal_key_value(self, highlighter: LogHighlighter):
        """Normal key=value should match both parts."""
        matches = self.get_matches(highlighter, "key=value")
        assert self.has_style(matches, "attrib_name", "key")
        assert self.has_style(matches, "attrib_value", "value")

    def test_double_equals_no_value(self, highlighter: LogHighlighter):
        """key==value should only match key, not =value as value."""
        matches = self.get_matches(highlighter, "key==value")
        assert self.has_style(matches, "attrib_name", "key")
        # Should NOT have =value as the value
        assert not any(m[0] == "=value" for m in matches)

    def test_triple_equals_no_value(self, highlighter: LogHighlighter):
        """key===value should only match key."""
        matches = self.get_matches(highlighter, "key===value")
        assert self.has_style(matches, "attrib_name", "key")
        assert not any("==" in m[0] for m in matches)

    def test_empty_value(self, highlighter: LogHighlighter):
        """key= with no value should match just the key."""
        matches = self.get_matches(highlighter, "key=")
        assert self.has_style(matches, "attrib_name", "key")

    def test_no_key(self, highlighter: LogHighlighter):
        """=value without a key should not match as key=value."""
        matches = self.get_matches(highlighter, "=value")
        assert not self.has_style(matches, "attrib_name")
        assert not self.has_style(matches, "attrib_value")

    # --- Quote Handling ---

    @pytest.mark.parametrize(
        "msg,expected_str",
        [
            ('msg="hello"', '"hello"'),
            ("msg='hello'", "'hello'"),
            ("msg=`hello`", "`hello`"),
        ],
    )
    def test_quoted_strings_matched(
        self, highlighter: LogHighlighter, msg: str, expected_str: str
    ):
        """All quote styles (double, single, backtick) should be matched."""
        matches = self.get_matches(highlighter, msg)
        # Should be matched as attrib_value (since it's after =)
        assert self.has_style(matches, "attrib_value", expected_str)

    def test_standalone_quoted_strings(self, highlighter: LogHighlighter):
        """Standalone quoted strings should be matched as str."""
        for msg, expected in [
            ('"hello world"', '"hello world"'),
            ("'single quotes'", "'single quotes'"),
            ("`backticks`", "`backticks`"),
        ]:
            matches = self.get_matches(highlighter, msg)
            assert self.has_style(matches, "str", expected), f"Failed for: {msg}"

    def test_unclosed_quote_as_value(self, highlighter: LogHighlighter):
        """Unclosed quote after = should be captured as value (not crash)."""
        matches = self.get_matches(highlighter, 'msg="unclosed')
        assert self.has_style(matches, "attrib_name", "msg")
        # The unclosed quote should be captured as attrib_value
        assert self.has_style(matches, "attrib_value", '"unclosed')

    # --- Type Value Handling ---

    def test_number_value(self, highlighter: LogHighlighter):
        """Numeric values in key=value should be matched."""
        matches = self.get_matches(highlighter, "count=42")
        assert self.has_style(matches, "attrib_name", "count")
        assert self.has_style(matches, "attrib_value", "42")

    def test_boolean_value(self, highlighter: LogHighlighter):
        """Boolean values in key=value should be matched."""
        matches = self.get_matches(highlighter, "enabled=True")
        assert self.has_style(matches, "attrib_name", "enabled")
        assert self.has_style(matches, "attrib_value", "True")

    def test_none_value(self, highlighter: LogHighlighter):
        """None values in key=value should be matched."""
        matches = self.get_matches(highlighter, "data=None")
        assert self.has_style(matches, "attrib_name", "data")
        assert self.has_style(matches, "attrib_value", "None")

    def test_list_value_components(self, highlighter: LogHighlighter):
        """List values should have components highlighted separately."""
        matches = self.get_matches(highlighter, "items=[1, 2, 3]")
        assert self.has_style(matches, "attrib_name", "items")
        assert self.has_style(matches, "brace", "[")
        assert self.has_style(matches, "brace", "]")
        assert self.has_style(matches, "number", "1")
        assert self.has_style(matches, "number", "2")
        assert self.has_style(matches, "number", "3")

    def test_class_call(self, highlighter: LogHighlighter):
        """Class instantiation patterns should be matched."""
        matches = self.get_matches(highlighter, "Point(x=1, y=2)")
        assert self.has_style(matches, "call", "Point")
        assert self.has_style(matches, "attrib_name", "x")
        assert self.has_style(matches, "attrib_name", "y")

    # --- Edge Cases That Should Not Crash ---

    @pytest.mark.parametrize(
        "msg",
        [
            "",  # Empty string
            "   ",  # Only whitespace
            "\n\n\n",  # Only newlines
            "🔥💀🎉",  # Unicode emoji
            "a\x00b",  # Null byte
            "normal message with no special patterns",
        ],
    )
    def test_edge_cases_no_crash(self, highlighter: LogHighlighter, msg: str):
        """Edge cases should not crash the highlighter."""
        from rich.text import Text

        text = Text(msg)
        # Should not raise
        highlighter.highlight(text)

    # --- URL Handling ---

    @pytest.mark.parametrize(
        "msg,expected_url",
        [
            ("http://example.com", "http://example.com"),
            ("https://example.com/path?query=1", "https://example.com/path?query=1"),
            ("file:///path/to/file", "file:///path/to/file"),
            ("wss://socket.example.com", "wss://socket.example.com"),
        ],
    )
    def test_urls_matched(
        self, highlighter: LogHighlighter, msg: str, expected_url: str
    ):
        """Various URL schemes should be matched."""
        matches = self.get_matches(highlighter, msg)
        assert self.has_style(matches, "url", expected_url)

    def test_ftp_not_matched_as_url(self, highlighter: LogHighlighter):
        """FTP URLs are not in our supported schemes."""
        matches = self.get_matches(highlighter, "ftp://server/file")
        assert not self.has_style(matches, "url")

    # --- Number Formats ---

    @pytest.mark.parametrize(
        "msg,expected_number",
        [
            ("42", "42"),
            ("-42", "-42"),
            ("3.14", "3.14"),
            ("-3.14", "-3.14"),
            ("1.5e-10", "1.5e-10"),
            ("1e+99", "1e+99"),
            ("123ns", "123ns"),
            ("500ms", "500ms"),
            ("1.5s", "1.5s"),
        ],
    )
    def test_number_formats(
        self, highlighter: LogHighlighter, msg: str, expected_number: str
    ):
        """Various number formats should be matched."""
        matches = self.get_matches(highlighter, msg)
        assert self.has_style(matches, "number", expected_number)

    # --- Booleans and None ---

    @pytest.mark.parametrize(
        "msg,style",
        [
            ("True", "bool_true"),
            ("False", "bool_false"),
            ("None", "none"),
        ],
    )
    def test_booleans_and_none(self, highlighter: LogHighlighter, msg: str, style: str):
        """Boolean and None literals should be matched."""
        matches = self.get_matches(highlighter, msg)
        assert self.has_style(matches, style, msg)
