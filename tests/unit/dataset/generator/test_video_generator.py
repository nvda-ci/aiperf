# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
from contextlib import ExitStack
from unittest.mock import Mock, patch

import pytest
from PIL import Image

from aiperf.common.config import VideoConfig
from aiperf.common.enums import VideoFormat, VideoSynthType
from aiperf.dataset.generator.video import VideoGenerator


@pytest.fixture
def base_config():
    """Base configuration for VideoGenerator tests."""
    return VideoConfig(
        width=64,
        height=64,
        duration=0.5,
        fps=2,
        format=VideoFormat.WEBM,
        codec="libvpx-vp9",
        synth_type=VideoSynthType.MOVING_SHAPES,
    )


class TestVideoGenerator:
    """Test suite for VideoGenerator class."""

    def test_init_with_config(self, base_config):
        """Test VideoGenerator initialization with valid config."""
        generator = VideoGenerator(base_config)
        assert generator.config == base_config

    @pytest.mark.parametrize(
        "ffmpeg_path,expected",
        [
            ("/usr/bin/ffmpeg", True),
            (None, False),
        ],
    )
    def test_check_ffmpeg_availability(self, base_config, ffmpeg_path, expected):
        """Test FFmpeg availability check."""
        with patch("shutil.which", return_value=ffmpeg_path):
            generator = VideoGenerator(base_config)
            assert generator._check_ffmpeg_availability() is expected

    @pytest.mark.parametrize(
        "platform_name,patches_and_expected",
        [
            # Linux distributions
            ("Linux", ({"open_return": io.StringIO("ID=ubuntu")}, "apt")),
            ("Linux", ({"open_return": io.StringIO("ID=fedora")}, "dnf")),
            ("Linux", ({"open_return": io.StringIO("ID=arch")}, "pacman")),
            ("Linux", ({"open_side_effect": FileNotFoundError}, "apt")),  # fallback
            # macOS
            ("Darwin",({"which": lambda x: "/brew" if x == "brew" else None}, "brew install")),
            ("Darwin",({"which": lambda x: "/port" if x == "port" else None}, "port install")),
            ("Darwin", ({"which": lambda x: None}, "brew.sh")),
            # Windows
            ("Windows",({"which": lambda x: "choco" if x == "choco" else None}, "choco install")),
            ("Windows",({"which": lambda x: "winget" if x == "winget" else None}, "winget install")),
            ("Windows", ({"which": lambda x: None}, "ffmpeg.org")),
            # Unknown OS
            ("UnknownOS", ({}, "ffmpeg.org")),
        ],
    )  # fmt: skip
    def test_get_ffmpeg_install_instructions(
        self, base_config, platform_name, patches_and_expected
    ):
        """Test platform-specific FFmpeg installation instructions."""
        patches_dict, expected_keyword = patches_and_expected
        generator = VideoGenerator(base_config)

        with ExitStack() as stack:
            stack.enter_context(patch("platform.system", return_value=platform_name))

            if "open_return" in patches_dict:
                stack.enter_context(
                    patch(
                        "builtins.open",
                        create=True,
                        return_value=patches_dict["open_return"],
                    )
                )
            elif "open_side_effect" in patches_dict:
                stack.enter_context(
                    patch("builtins.open", side_effect=patches_dict["open_side_effect"])
                )

            if "which" in patches_dict:
                stack.enter_context(
                    patch("shutil.which", side_effect=patches_dict["which"])
                )

            instructions = generator._get_ffmpeg_install_instructions()
            assert expected_keyword in instructions
            assert "ffmpeg" in instructions

    def test_generate_with_disabled_video(self):
        """Test that generate returns empty string when video is disabled."""
        config = VideoConfig(
            width=None,
            height=None,
            duration=1.0,
            fps=4,
            format=VideoFormat.WEBM,
            codec="libvpx-vp9",
            synth_type=VideoSynthType.MOVING_SHAPES,
        )
        generator = VideoGenerator(config)
        result = generator.generate()
        assert result == ""

    @pytest.mark.parametrize(
        "synth_type,width,height,duration,fps",
        [
            (VideoSynthType.MOVING_SHAPES, 64, 64, 0.5, 2),
            (VideoSynthType.GRID_CLOCK, 128, 128, 1.0, 4),
        ],
    )
    def test_generate_frames(self, synth_type, width, height, duration, fps):
        """Test frame generation for different synthesis types."""
        config = VideoConfig(
            width=width,
            height=height,
            duration=duration,
            fps=fps,
            format=VideoFormat.WEBM,
            codec="libvpx-vp9",
            synth_type=synth_type,
        )
        generator = VideoGenerator(config)
        frames = generator._generate_frames()

        expected_frame_count = int(duration * fps)
        assert len(frames) == expected_frame_count

        # Verify all frames are PIL Images with correct dimensions
        for frame in frames:
            assert isinstance(frame, Image.Image)
            assert frame.size == (width, height)
            assert frame.mode == "RGB"

    def test_generate_frames_unknown_type(self, base_config):
        """Test that unknown synthesis type raises ValueError."""
        base_config.synth_type = "unknown_type"
        generator = VideoGenerator(base_config)

        with pytest.raises(ValueError, match="Unknown synthesis type"):
            generator._generate_frames()

    def test_encode_frames_to_base64_empty_frames(self, base_config):
        """Test encoding empty frame list returns empty string."""
        generator = VideoGenerator(base_config)
        result = generator._encode_frames_to_base64([])
        assert result == ""

    def test_encode_frames_to_base64_unsupported_format(self, base_config):
        """Test that unsupported format raises ValueError."""
        base_config.format = Mock(name="UNSUPPORTED", value="unsupported")
        generator = VideoGenerator(base_config)
        frames = [Image.new("RGB", (64, 64), (0, 0, 0))]

        with pytest.raises(ValueError, match="Unsupported video format"):
            generator._encode_frames_to_base64(frames)

    def test_encode_frames_ffmpeg_not_available(self, base_config):
        """Test that encoding fails gracefully when FFmpeg is not available."""
        generator = VideoGenerator(base_config)
        frames = [Image.new("RGB", (64, 64), (0, 0, 0))]

        with (
            patch.object(generator, "_check_ffmpeg_availability", return_value=False),
            pytest.raises(RuntimeError, match="FFmpeg binary not found"),
        ):
            generator._encode_frames_to_base64(frames)

    def test_encode_frames_codec_error(self, base_config):
        """Test handling of codec errors."""
        generator = VideoGenerator(base_config)
        frames = [Image.new("RGB", (64, 64), (0, 0, 0))]

        with (
            patch.object(generator, "_check_ffmpeg_availability", return_value=True),
            patch.object(
                generator,
                "_create_video_with_pipes",
                side_effect=Exception("Codec not supported"),
            ),
            patch.object(
                generator,
                "_create_video_with_temp_files",
                side_effect=Exception("Codec not supported"),
            ),
            pytest.raises(RuntimeError, match="[Cc]odec"),
        ):
            generator._encode_frames_to_base64(frames)

    def test_create_video_with_pipes_fallback(self, base_config):
        """Test fallback to temp files when pipes fail."""
        generator = VideoGenerator(base_config)
        frames = [Image.new("RGB", (64, 64), (255, 0, 0))]
        mock_result = "data:video/webm;base64,FAKE_BASE64"

        with (
            patch.object(
                generator, "_create_video_with_pipes", side_effect=BrokenPipeError
            ),
            patch.object(
                generator, "_create_video_with_temp_files", return_value=mock_result
            ),
        ):
            result = generator._create_video_with_ffmpeg(frames)
            assert result == mock_result


class TestCreateVideoWithFFmpeg:
    """Test suite specifically for _create_video_with_ffmpeg method."""

    @pytest.fixture
    def frames(self):
        """Create sample frames for testing."""
        return [
            Image.new("RGB", (64, 64), (255, 0, 0)),
            Image.new("RGB", (64, 64), (0, 255, 0)),
        ]

    @pytest.fixture
    def mp4_config(self):
        """Create MP4 video configuration for testing."""
        return VideoConfig(
            width=64,
            height=64,
            duration=0.5,
            fps=2,
            format=VideoFormat.MP4,
            codec="libx264",
            synth_type=VideoSynthType.MOVING_SHAPES,
        )

    @pytest.fixture
    def webm_config(self):
        """Create WebM video configuration for testing."""
        return VideoConfig(
            width=64,
            height=64,
            duration=0.5,
            fps=2,
            format=VideoFormat.WEBM,
            codec="libvpx-vp9",
            synth_type=VideoSynthType.MOVING_SHAPES,
        )

    def test_mp4_uses_temp_files(self, frames, mp4_config):
        """Test that MP4 format always uses temp file method."""
        generator = VideoGenerator(mp4_config)
        mock_result = "data:video/mp4;base64,FAKE_MP4_DATA"

        with (
            patch.object(
                generator, "_create_video_with_temp_files", return_value=mock_result
            ) as mock_temp,
            patch.object(generator, "_create_video_with_pipes") as mock_pipes,
        ):
            result = generator._create_video_with_ffmpeg(frames)

            mock_temp.assert_called_once_with(frames)
            mock_pipes.assert_not_called()
            assert result == mock_result

    def test_webm_tries_pipes_first(self, frames, webm_config):
        """Test that WebM format attempts pipe method first."""
        generator = VideoGenerator(webm_config)
        mock_result = "data:video/webm;base64,FAKE_WEBM_DATA"

        with (
            patch.object(
                generator, "_create_video_with_pipes", return_value=mock_result
            ) as mock_pipes,
            patch.object(generator, "_create_video_with_temp_files") as mock_temp,
        ):
            result = generator._create_video_with_ffmpeg(frames)

            mock_pipes.assert_called_once_with(frames)
            mock_temp.assert_not_called()
            assert result == mock_result

    @pytest.mark.parametrize(
        "exception_type",
        [BrokenPipeError, OSError, RuntimeError],
    )
    def test_webm_fallback_on_pipe_errors(self, frames, webm_config, exception_type):
        """Test that WebM falls back to temp files on pipe errors."""
        generator = VideoGenerator(webm_config)
        mock_result = "data:video/webm;base64,FAKE_WEBM_DATA"
        error_msg = f"Test {exception_type.__name__}"

        with (
            patch.object(
                generator,
                "_create_video_with_pipes",
                side_effect=exception_type(error_msg),
            ) as mock_pipes,
            patch.object(
                generator, "_create_video_with_temp_files", return_value=mock_result
            ) as mock_temp,
        ):
            result = generator._create_video_with_ffmpeg(frames)

            mock_pipes.assert_called_once_with(frames)
            mock_temp.assert_called_once_with(frames)
            assert result == mock_result

    def test_empty_frames_raises_error(self, webm_config):
        """Test that empty frame list raises ValueError."""
        generator = VideoGenerator(webm_config)

        with pytest.raises(
            ValueError, match="Cannot create video from empty frame list"
        ):
            generator._create_video_with_ffmpeg([])

    def test_mp4_propagates_temp_file_errors(self, frames, mp4_config):
        """Test that MP4 encoding errors are propagated."""
        generator = VideoGenerator(mp4_config)
        error_msg = "FFmpeg temp file error"

        with (
            patch.object(
                generator,
                "_create_video_with_temp_files",
                side_effect=RuntimeError(error_msg),
            ),
            pytest.raises(RuntimeError, match=error_msg),
        ):
            generator._create_video_with_ffmpeg(frames)

    def test_webm_propagates_both_method_failures(self, frames, webm_config):
        """Test that errors are propagated when both methods fail."""
        generator = VideoGenerator(webm_config)
        pipe_error = "Pipe encoding failed"
        temp_error = "Temp file encoding failed"

        with (
            patch.object(
                generator,
                "_create_video_with_pipes",
                side_effect=RuntimeError(pipe_error),
            ),
            patch.object(
                generator,
                "_create_video_with_temp_files",
                side_effect=RuntimeError(temp_error),
            ),
            pytest.raises(RuntimeError, match=temp_error),
        ):
            generator._create_video_with_ffmpeg(frames)

    def test_logging_messages_mp4(self, frames, mp4_config, caplog):
        """Test that appropriate log messages are generated for MP4."""
        generator = VideoGenerator(mp4_config)
        mock_result = "data:video/mp4;base64,FAKE_DATA"

        with patch.object(
            generator, "_create_video_with_temp_files", return_value=mock_result
        ):
            generator._create_video_with_ffmpeg(frames)

            assert "file-based encoding for MP4" in caplog.text

    def test_logging_messages_webm_pipe_success(self, frames, webm_config, caplog):
        """Test log messages for successful pipe-based WebM encoding."""
        generator = VideoGenerator(webm_config)
        mock_result = "data:video/webm;base64,FAKE_DATA"

        with patch.object(
            generator, "_create_video_with_pipes", return_value=mock_result
        ):
            generator._create_video_with_ffmpeg(frames)

            assert "pipe-based encoding for webm" in caplog.text.lower()

    def test_logging_messages_webm_fallback(self, frames, webm_config, caplog):
        """Test log messages when falling back from pipes to temp files."""
        generator = VideoGenerator(webm_config)
        mock_result = "data:video/webm;base64,FAKE_DATA"

        with (
            patch.object(
                generator,
                "_create_video_with_pipes",
                side_effect=BrokenPipeError("Test error"),
            ),
            patch.object(
                generator, "_create_video_with_temp_files", return_value=mock_result
            ),
        ):
            generator._create_video_with_ffmpeg(frames)

            assert "pipe-based encoding" in caplog.text.lower()
            assert "falling back to file-based" in caplog.text.lower()
            assert "BrokenPipeError" in caplog.text
