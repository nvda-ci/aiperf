# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for mode detection functionality.
"""

import sys
from pathlib import Path

import pytest

from aiperf.plot.core.mode_detector import (
    ModeDetector,
    VisualizationMode,
)
from aiperf.plot.exceptions import ModeDetectionError


class TestModeDetection:
    """Tests for detect_mode method."""

    def test_single_run_detection(
        self, mode_detector: ModeDetector, populated_run_dir: Path
    ) -> None:
        """Test detection of single run mode."""
        mode = mode_detector.detect_mode([populated_run_dir])
        assert mode == VisualizationMode.SINGLE_RUN

    def test_multiple_runs_explicit_paths(
        self, mode_detector: ModeDetector, multiple_run_dirs: list[Path]
    ) -> None:
        """Test detection of multi-run mode with explicit paths."""
        mode = mode_detector.detect_mode(multiple_run_dirs)
        assert mode == VisualizationMode.MULTI_RUN

    def test_multiple_runs_parent_directory(
        self, mode_detector: ModeDetector, parent_dir_with_runs: Path
    ) -> None:
        """Test detection of multi-run mode from parent directory."""
        mode = mode_detector.detect_mode([parent_dir_with_runs])
        assert mode == VisualizationMode.MULTI_RUN

    def test_parent_directory_with_single_run(
        self, mode_detector: ModeDetector, parent_dir_with_single_run: Path
    ) -> None:
        """Test that parent directory with only one run is detected as SINGLE_RUN."""
        mode = mode_detector.detect_mode([parent_dir_with_single_run])
        assert mode == VisualizationMode.SINGLE_RUN

    def test_empty_paths_raises_error(self, mode_detector: ModeDetector) -> None:
        """Test that empty path list raises error."""
        with pytest.raises(ModeDetectionError, match="No paths provided"):
            mode_detector.detect_mode([])

    def test_nonexistent_path_raises_error(self, mode_detector: ModeDetector) -> None:
        """Test that nonexistent path raises error."""
        fake_path = Path("/nonexistent/path")
        with pytest.raises(ModeDetectionError, match="Path does not exist"):
            mode_detector.detect_mode([fake_path])

    def test_non_directory_path_raises_error(
        self, mode_detector: ModeDetector, tmp_path: Path
    ) -> None:
        """Test that file path raises error."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("test")

        with pytest.raises(ModeDetectionError, match="Path is not a directory"):
            mode_detector.detect_mode([file_path])

    def test_invalid_run_directory_raises_error(
        self, mode_detector: ModeDetector, tmp_path: Path
    ) -> None:
        """Test that directory without required files raises error."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        with pytest.raises(
            ModeDetectionError,
            match="No valid run directories found",
        ):
            mode_detector.detect_mode([empty_dir])

    def test_multiple_invalid_paths_raises_error(
        self, mode_detector: ModeDetector, tmp_path: Path
    ) -> None:
        """Test that multiple paths with invalid run raises error."""
        dir1 = tmp_path / "dir1"
        dir1.mkdir()
        dir2 = tmp_path / "dir2"
        dir2.mkdir()

        with pytest.raises(ModeDetectionError, match="No valid run directories found"):
            mode_detector.detect_mode([dir1, dir2])


class TestFindRunDirectories:
    """Tests for find_run_directories method."""

    def test_find_single_run(
        self, mode_detector: ModeDetector, populated_run_dir: Path
    ) -> None:
        """Test finding single run directory."""
        runs = mode_detector.find_run_directories([populated_run_dir])
        assert len(runs) == 1
        assert runs[0] == populated_run_dir

    def test_find_multiple_runs_explicit(
        self, mode_detector: ModeDetector, multiple_run_dirs: list[Path]
    ) -> None:
        """Test finding multiple run directories from explicit paths."""
        runs = mode_detector.find_run_directories(multiple_run_dirs)
        assert len(runs) == 3
        assert set(runs) == set(multiple_run_dirs)

    def test_find_runs_from_parent(
        self, mode_detector: ModeDetector, parent_dir_with_runs: Path
    ) -> None:
        """Test finding run directories from parent directory."""
        runs = mode_detector.find_run_directories([parent_dir_with_runs])
        assert len(runs) == 3
        # Verify all runs are in the parent directory
        for run in runs:
            assert run.parent == parent_dir_with_runs

    def test_find_runs_sorted(
        self, mode_detector: ModeDetector, parent_dir_with_runs: Path
    ) -> None:
        """Test that found runs are sorted."""
        runs = mode_detector.find_run_directories([parent_dir_with_runs])
        run_names = [r.name for r in runs]
        assert run_names == sorted(run_names)

    def test_nonexistent_path_raises_error(self, mode_detector: ModeDetector) -> None:
        """Test that nonexistent path raises error."""
        fake_path = Path("/nonexistent/path")
        with pytest.raises(ModeDetectionError, match="Path does not exist"):
            mode_detector.find_run_directories([fake_path])

    def test_non_directory_raises_error(
        self, mode_detector: ModeDetector, tmp_path: Path
    ) -> None:
        """Test that file path raises error."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("test")

        with pytest.raises(ModeDetectionError, match="Path is not a directory"):
            mode_detector.find_run_directories([file_path])

    def test_invalid_directory_raises_error(
        self, mode_detector: ModeDetector, tmp_path: Path
    ) -> None:
        """Test that directory without valid runs raises error."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        with pytest.raises(
            ModeDetectionError,
            match="does not contain any valid run directories",
        ):
            mode_detector.find_run_directories([empty_dir])

    def test_mixed_valid_and_invalid_paths(
        self, mode_detector: ModeDetector, populated_run_dir: Path, tmp_path: Path
    ) -> None:
        """Test handling of mixed valid and invalid paths."""
        invalid_dir = tmp_path / "invalid"
        invalid_dir.mkdir()

        # Should raise error on first invalid path
        with pytest.raises(ModeDetectionError):
            mode_detector.find_run_directories([populated_run_dir, invalid_dir])


class TestIsRunDirectory:
    """Tests for _is_run_directory helper."""

    def test_valid_run_directory(
        self, mode_detector: ModeDetector, populated_run_dir: Path
    ) -> None:
        """Test that valid run directory is detected."""

        assert mode_detector._is_run_directory(populated_run_dir) is True

    def test_directory_without_required_file(
        self, mode_detector: ModeDetector, tmp_path: Path
    ) -> None:
        """Test that directory without profile_export.jsonl is not a run."""

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        assert mode_detector._is_run_directory(empty_dir) is False

    def test_file_path_returns_false(
        self, mode_detector: ModeDetector, tmp_path: Path
    ) -> None:
        """Test that file path returns False."""

        file_path = tmp_path / "test.txt"
        file_path.write_text("test")

        assert mode_detector._is_run_directory(file_path) is False


class TestSymlinkEdgeCases:
    """Tests for symlink edge cases."""

    def test_symlink_to_run_directory(
        self, mode_detector: ModeDetector, populated_run_dir: Path, tmp_path: Path
    ) -> None:
        """Test symlink to valid run directory."""
        symlink = tmp_path / "symlink_run"
        symlink.symlink_to(populated_run_dir)
        mode = mode_detector.detect_mode([symlink])
        assert mode == VisualizationMode.SINGLE_RUN

    def test_broken_symlink_directory(
        self, mode_detector: ModeDetector, tmp_path: Path
    ) -> None:
        """Test broken symlink to directory."""
        symlink = tmp_path / "broken"
        symlink.symlink_to(tmp_path / "nonexistent")
        with pytest.raises(ModeDetectionError, match="Path does not exist"):
            mode_detector.detect_mode([symlink])

    def test_symlinked_profile_export_file(
        self, mode_detector: ModeDetector, tmp_path: Path
    ) -> None:
        """Test run with symlinked profile_export.jsonl."""
        # Create actual file
        real_file = tmp_path / "real_profile.jsonl"
        real_file.write_text('{"test": "data"}\n')

        # Create run dir with symlink
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        symlink_file = run_dir / "profile_export.jsonl"
        symlink_file.symlink_to(real_file)

        assert mode_detector._is_run_directory(run_dir)

    def test_broken_symlink_profile_export(
        self, mode_detector: ModeDetector, tmp_path: Path
    ) -> None:
        """Test run with broken symlink to profile_export.jsonl."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        symlink_file = run_dir / "profile_export.jsonl"
        symlink_file.symlink_to(tmp_path / "nonexistent.jsonl")

        assert not mode_detector._is_run_directory(run_dir)


class TestDuplicatePaths:
    """Tests for duplicate path handling."""

    def test_same_path_twice(
        self, mode_detector: ModeDetector, populated_run_dir: Path
    ) -> None:
        """Test same path specified multiple times."""
        mode = mode_detector.detect_mode([populated_run_dir, populated_run_dir])
        # Should deduplicate to single run
        assert mode == VisualizationMode.SINGLE_RUN

        # Should deduplicate
        runs = mode_detector.find_run_directories(
            [populated_run_dir, populated_run_dir]
        )
        assert len(runs) == 1

    def test_resolved_path_duplicates(
        self, mode_detector: ModeDetector, populated_run_dir: Path
    ) -> None:
        """Test paths that resolve to same directory."""
        # Create paths that look different but resolve to same location
        path1 = populated_run_dir
        path2 = populated_run_dir / ".." / populated_run_dir.name

        runs = mode_detector.find_run_directories([path1, path2])
        assert len(runs) == 1

    def test_parent_and_child_paths_mixed(
        self, mode_detector: ModeDetector, parent_dir_with_runs: Path
    ) -> None:
        """Test parent directory + explicit child paths."""
        # Get first child run directory
        children = [d for d in parent_dir_with_runs.iterdir() if d.is_dir()]
        child = children[0]

        runs = mode_detector.find_run_directories([parent_dir_with_runs, child])
        # Should deduplicate - child appears once (3 total, not 4)
        assert len(runs) == 3


class TestNestedRunDirectories:
    """Tests for nested run directory handling."""

    def test_nested_run_directories(
        self, mode_detector: ModeDetector, nested_run_dirs: Path
    ) -> None:
        """Test run directory containing another run."""
        # Should find both outer and inner runs
        runs = mode_detector.find_run_directories([nested_run_dirs])
        assert len(runs) == 2

        # Should detect as multi-run (2 runs found)
        mode = mode_detector.detect_mode([nested_run_dirs])
        assert mode == VisualizationMode.MULTI_RUN

    def test_nested_runs_counted_separately(
        self, mode_detector: ModeDetector, nested_run_dirs: Path, sample_jsonl_data
    ) -> None:
        """Test that nested runs are counted separately."""
        # Add another standalone run at the parent level
        standalone = nested_run_dirs / "standalone_run"
        standalone.mkdir()
        jsonl_file = standalone / "profile_export.jsonl"
        with open(jsonl_file, "w") as f:
            for record in sample_jsonl_data:
                f.write(f"{record}\n")

        # Should find 3 runs total: outer, inner, standalone
        runs = mode_detector.find_run_directories([nested_run_dirs])
        assert len(runs) == 3

    def test_deeply_nested_runs(
        self, mode_detector: ModeDetector, tmp_path: Path, sample_jsonl_data
    ) -> None:
        """Test multiple levels of nesting."""
        # Create three levels of nesting
        level1 = tmp_path / "level1"
        level1.mkdir()
        (level1 / "profile_export.jsonl").write_text('{"test": "level1"}\n')

        level2 = level1 / "level2"
        level2.mkdir()
        (level2 / "profile_export.jsonl").write_text('{"test": "level2"}\n')

        level3 = level2 / "level3"
        level3.mkdir()
        (level3 / "profile_export.jsonl").write_text('{"test": "level3"}\n')

        # Should find all 3 levels
        runs = mode_detector.find_run_directories([tmp_path])
        assert len(runs) == 3


class TestPermissionErrors:
    """Tests for permission error handling."""

    @pytest.mark.skipif(
        sys.platform == "win32", reason="Permission handling different on Windows"
    )
    def test_unreadable_directory(
        self, mode_detector: ModeDetector, tmp_path: Path
    ) -> None:
        """Test directory with no read permissions."""
        no_read_dir = tmp_path / "no_read"
        no_read_dir.mkdir()
        no_read_dir.chmod(0o000)

        try:
            # Should handle gracefully - no runs found
            with pytest.raises(ModeDetectionError, match="No valid run directories"):
                mode_detector.detect_mode([no_read_dir])
        finally:
            no_read_dir.chmod(0o755)  # Cleanup

    def test_unreadable_profile_export(
        self, mode_detector: ModeDetector, tmp_path: Path
    ) -> None:
        """Test run with unreadable profile_export.jsonl."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        profile = run_dir / "profile_export.jsonl"
        profile.write_text('{"test": "data"}\n')
        profile.chmod(0o000)

        try:
            # Should still be detected as valid (only checks existence)
            # DataLoader will fail later when reading
            assert mode_detector._is_run_directory(run_dir)
        finally:
            profile.chmod(0o644)  # Cleanup


class TestFileContentEdgeCases:
    """Tests for file content edge cases."""

    def test_empty_profile_export_file(
        self, mode_detector: ModeDetector, tmp_path: Path
    ) -> None:
        """Test run with empty profile_export.jsonl."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "profile_export.jsonl").write_text("")

        # Mode detection treats as valid
        assert mode_detector._is_run_directory(run_dir)
        mode = mode_detector.detect_mode([run_dir])
        assert mode == VisualizationMode.SINGLE_RUN

        # DataLoader will fail (tested elsewhere)

    def test_corrupted_profile_export_file(
        self, mode_detector: ModeDetector, tmp_path: Path
    ) -> None:
        """Test run with corrupted JSON."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "profile_export.jsonl").write_text("not valid json{{{")

        # Mode detection doesn't validate content
        assert mode_detector._is_run_directory(run_dir)


class TestHiddenDirectories:
    """Tests for hidden directory handling."""

    def test_hidden_run_directory(
        self, mode_detector: ModeDetector, tmp_path: Path, sample_jsonl_data
    ) -> None:
        """Test hidden run directory (starting with .)."""
        hidden_run = tmp_path / ".hidden_run"
        hidden_run.mkdir()
        jsonl_file = hidden_run / "profile_export.jsonl"
        with open(jsonl_file, "w") as f:
            for record in sample_jsonl_data:
                f.write(f"{record}\n")

        # Should be detected
        assert mode_detector._is_run_directory(hidden_run)

        # Hidden directories should be found in scan
        runs = mode_detector.find_run_directories([tmp_path])
        assert len(runs) == 1
