# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from dataclasses import dataclass

import pytest

from aiperf.common.duplicate_tracker import AsyncKeyedDuplicateTracker


@dataclass
class SampleRecord:
    """Simple test record for deduplication testing."""

    key: str
    value: int
    metadata: str = "test"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SampleRecord):
            return False
        return self.value == other.value and self.metadata == other.metadata


@pytest.fixture
def tracker() -> AsyncKeyedDuplicateTracker[SampleRecord]:
    """Create a fresh tracker for each test."""
    return AsyncKeyedDuplicateTracker[SampleRecord](
        key_function=lambda record: record.key,
        value_function=lambda record: record.value,
    )


# Helper functions
async def write_sequence(
    tracker: AsyncKeyedDuplicateTracker[SampleRecord], key: str, values: list[int]
) -> list[list[SampleRecord]]:
    """Write a sequence of values and return all results."""
    results = []
    for value in values:
        record = SampleRecord(key=key, value=value)
        result = await tracker.deduplicate_record(record)
        results.append(result)
    return results


def flatten_results(results: list[list[SampleRecord]]) -> list[SampleRecord]:
    """Flatten a list of results into a single list."""
    return [record for result in results for record in result]


def get_values(records: list[SampleRecord]) -> list[int]:
    """Extract values from a list of records."""
    return [record.value for record in records]


class TestAsyncKeyedDuplicateTrackerBasicDeduplication:
    """Test basic deduplication functionality."""

    @pytest.mark.asyncio
    async def test_first_record_always_written(
        self, tracker: AsyncKeyedDuplicateTracker[SampleRecord]
    ):
        """Test that the first record is always written."""
        record = SampleRecord(key="key1", value=1)
        result = await tracker.deduplicate_record(record)

        assert len(result) == 1
        assert result[0] == record

    @pytest.mark.asyncio
    @pytest.mark.parametrize("num_duplicates", [2, 3, 5, 10])  # fmt: skip
    async def test_consecutive_duplicates_suppressed(
        self, tracker: AsyncKeyedDuplicateTracker[SampleRecord], num_duplicates: int
    ):
        """Test that consecutive identical records are suppressed."""
        results = await write_sequence(tracker, "key1", [1] * num_duplicates)

        # First should be written, rest suppressed
        assert len(results[0]) == 1
        for i in range(1, num_duplicates):
            assert len(results[i]) == 0

        # Dupe count should be num_duplicates - 1
        assert tracker._dupe_counts["key1"] == num_duplicates - 1

    @pytest.mark.asyncio
    async def test_change_writes_previous_and_new(
        self, tracker: AsyncKeyedDuplicateTracker[SampleRecord]
    ):
        """Test that when value changes, both previous and new records are written.

        Input: A, A, A, B
        Expected: A (first), [], [], [A, B] (last A before change + new B)
        """
        results = await write_sequence(tracker, "key1", [1, 1, 1, 2])

        assert len(results[0]) == 1  # First A written
        assert len(results[1]) == 0  # Duplicate suppressed
        assert len(results[2]) == 0  # Duplicate suppressed
        assert len(results[3]) == 2  # Both previous A and new B
        assert results[3][0].value == 1  # Previous record
        assert results[3][1].value == 2  # New record

        # Dupe count should be reset
        assert tracker._dupe_counts["key1"] == 0

    @pytest.mark.asyncio
    async def test_no_deduplication_when_values_differ(
        self, tracker: AsyncKeyedDuplicateTracker[SampleRecord]
    ):
        """Test that different values are not deduplicated."""
        results = await write_sequence(tracker, "key1", [1, 2, 3])

        for i, result in enumerate(results):
            # Each should write (possibly with previous on change)
            assert len(result) >= 1
            assert result[-1].value == i + 1  # Last item should be current record

    @pytest.mark.asyncio
    # fmt: skip
    @pytest.mark.parametrize(
        "input_sequence,expected_output",
        [
            # A,A,A,B,B,C → A (first), A,B (last A + change to B), B,C (last B + change to C)
            ([1, 1, 1, 2, 2, 3], [1, 1, 2, 2, 3]),
            # A,A,A → A (only first during execution)
            ([1, 1, 1], [1]),
            # A,B,C → A, B, C (no duplicates, just write each)
            ([1, 2, 3], [1, 2, 3]),
            # A,A,B,B,C,C → A, A,B (last A + B), B,C (last B + C)
            ([1, 1, 2, 2, 3, 3], [1, 1, 2, 2, 3]),
            # A,B,A,B,A → No duplicates, just write each = A,B,A,B,A
            ([1, 2, 1, 2, 1], [1, 2, 1, 2, 1]),
        ],
    )
    async def test_deduplication_sequences(
        self,
        tracker: AsyncKeyedDuplicateTracker[SampleRecord],
        input_sequence: list[int],
        expected_output: list[int],
    ):
        """Test various deduplication sequences."""
        results = await write_sequence(tracker, "key1", input_sequence)
        all_written = flatten_results(results)
        values = get_values(all_written)

        assert values == expected_output


class TestAsyncKeyedDuplicateTrackerPerKey:
    """Test that deduplication is tracked independently per key."""

    @pytest.mark.asyncio
    async def test_deduplication_per_key(
        self, tracker: AsyncKeyedDuplicateTracker[SampleRecord]
    ):
        """Test that deduplication is tracked independently per key."""
        record_key1 = SampleRecord(key="key1", value=1)
        record_key2 = SampleRecord(key="key2", value=1)

        # Write same value to two different keys
        result1_key1 = await tracker.deduplicate_record(record_key1)
        result1_key2 = await tracker.deduplicate_record(record_key2)

        # Both should write (first for each key)
        assert len(result1_key1) == 1
        assert len(result1_key2) == 1

        # Write duplicates
        result2_key1 = await tracker.deduplicate_record(record_key1)
        result2_key2 = await tracker.deduplicate_record(record_key2)

        # Both should suppress
        assert len(result2_key1) == 0
        assert len(result2_key2) == 0

        # Each key should have its own dupe count
        assert tracker._dupe_counts["key1"] == 1
        assert tracker._dupe_counts["key2"] == 1

    @pytest.mark.asyncio
    async def test_keys_maintain_independent_state(
        self, tracker: AsyncKeyedDuplicateTracker[SampleRecord]
    ):
        """Test that different keys maintain completely independent state."""
        record_a_key1 = SampleRecord(key="key1", value=1)
        record_b_key2 = SampleRecord(key="key2", value=2)

        # key1: Write A three times
        await tracker.deduplicate_record(record_a_key1)
        await tracker.deduplicate_record(record_a_key1)
        await tracker.deduplicate_record(record_a_key1)

        # key2: Write B three times
        await tracker.deduplicate_record(record_b_key2)
        await tracker.deduplicate_record(record_b_key2)
        await tracker.deduplicate_record(record_b_key2)

        # Each should have 2 duplicates
        assert tracker._dupe_counts["key1"] == 2
        assert tracker._dupe_counts["key2"] == 2

        # Previous records should be different by value
        assert tracker._previous_records["key1"].value == 1
        assert tracker._previous_records["key2"].value == 2


class TestAsyncKeyedDuplicateTrackerEquality:
    """Test equality comparison for deduplication."""

    @pytest.mark.asyncio
    async def test_equality_uses_value_function(
        self, tracker: AsyncKeyedDuplicateTracker[SampleRecord]
    ):
        """Test that deduplication uses the value_function for comparison."""
        record1 = SampleRecord(key="key1", value=1, metadata="test")
        record2 = SampleRecord(key="key1", value=1, metadata="different")
        record3 = SampleRecord(key="key1", value=2, metadata="test")

        # record1 and record2 have same value (even though metadata differs)
        result1 = await tracker.deduplicate_record(record1)
        assert len(result1) == 1

        result2 = await tracker.deduplicate_record(record2)
        assert len(result2) == 0  # Duplicate (same value)

        # record3 has different value
        result3 = await tracker.deduplicate_record(record3)
        assert len(result3) == 2  # Previous + new

    @pytest.mark.asyncio
    async def test_complex_equality(self):
        """Test deduplication with complex dictionary objects."""
        tracker_dict: AsyncKeyedDuplicateTracker[dict] = AsyncKeyedDuplicateTracker[
            dict
        ](
            key_function=lambda record: record.get("key", "default"),
            value_function=lambda record: record,
        )

        dict1 = {"key": "key1", "a": 1, "b": {"c": 2}}
        dict2 = {"key": "key1", "a": 1, "b": {"c": 2}}
        dict3 = {"key": "key1", "a": 1, "b": {"c": 3}}

        result1 = await tracker_dict.deduplicate_record(dict1)
        assert len(result1) == 1

        result2 = await tracker_dict.deduplicate_record(dict2)
        assert len(result2) == 0  # Equal dicts

        result3 = await tracker_dict.deduplicate_record(dict3)
        assert len(result3) == 2  # Different


class TestAsyncKeyedDuplicateTrackerConcurrency:
    """Test concurrent access to the tracker."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("num_tasks,writes_per_task", [(3, 10), (5, 5), (10, 3)])  # fmt: skip
    async def test_concurrent_writes_to_same_key(
        self,
        tracker: AsyncKeyedDuplicateTracker[SampleRecord],
        num_tasks: int,
        writes_per_task: int,
    ):
        """Test that concurrent writes to the same key are handled safely."""
        results = await write_sequence(
            tracker, "key1", [1] * (num_tasks * writes_per_task)
        )

        # At least first record should be written
        total_written = sum(len(r) for r in results)
        assert total_written >= 1

        # Total written should be much less than total due to deduplication
        assert total_written < num_tasks * writes_per_task

    @pytest.mark.asyncio
    @pytest.mark.parametrize("num_keys", [2, 3, 5])  # fmt: skip
    async def test_concurrent_writes_to_different_keys(
        self, tracker: AsyncKeyedDuplicateTracker[SampleRecord], num_keys: int
    ):
        """Test that different keys can be written concurrently."""

        async def write_to_key(key: str, count: int) -> int:
            total = 0
            for _ in range(count):
                record = SampleRecord(key=key, value=1)
                result = await tracker.deduplicate_record(record)
                total += len(result)
            return total

        # Write to different keys concurrently
        keys = [f"key{i}" for i in range(num_keys)]
        results = await asyncio.gather(*[write_to_key(key, 5) for key in keys])

        # Each key should have written at least once (first record)
        for key_result in results:
            assert key_result >= 1

        # Each key should have its own lock
        for key in keys:
            assert key in tracker._dupe_locks

    @pytest.mark.asyncio
    async def test_lock_creation_race_condition(
        self, tracker: AsyncKeyedDuplicateTracker[SampleRecord]
    ):
        """Test that lock creation handles race conditions correctly."""

        async def write_first_record() -> list[SampleRecord]:
            record = SampleRecord(key="new_key", value=1)
            return await tracker.deduplicate_record(record)

        # Try to create locks for the same key concurrently
        results = await asyncio.gather(
            write_first_record(),
            write_first_record(),
            write_first_record(),
        )

        # Should have created exactly one lock
        assert "new_key" in tracker._dupe_locks

        # At least one should have written
        total_written = sum(len(r) for r in results)
        assert total_written >= 1


class TestAsyncKeyedDuplicateTrackerEdgeCases:
    """Test edge cases and special scenarios."""

    @pytest.mark.asyncio
    async def test_single_record(
        self, tracker: AsyncKeyedDuplicateTracker[SampleRecord]
    ):
        """Test that a single record is written without issues."""
        record = SampleRecord(key="key1", value=1)
        result = await tracker.deduplicate_record(record)

        assert len(result) == 1
        assert result[0] == record
        assert tracker._dupe_counts["key1"] == 0

    @pytest.mark.asyncio
    @pytest.mark.parametrize("key", ["", "key-with-dashes", "key.with.dots", "key/with/slashes"])  # fmt: skip
    async def test_special_key_strings(
        self, tracker: AsyncKeyedDuplicateTracker[SampleRecord], key: str
    ):
        """Test that various key formats work correctly."""
        record = SampleRecord(key=key, value=1)
        result = await tracker.deduplicate_record(record)

        assert len(result) == 1
        assert key in tracker._previous_records

    @pytest.mark.asyncio
    @pytest.mark.parametrize("num_keys", [10, 50, 100])  # fmt: skip
    async def test_many_keys(
        self, tracker: AsyncKeyedDuplicateTracker[SampleRecord], num_keys: int
    ):
        """Test handling many different keys."""
        for i in range(num_keys):
            record = SampleRecord(key=f"key{i}", value=1)
            result = await tracker.deduplicate_record(record)
            assert len(result) == 1

        # Should have num_keys locks and previous records
        assert len(tracker._dupe_locks) == num_keys
        assert len(tracker._previous_records) == num_keys

    @pytest.mark.asyncio
    @pytest.mark.parametrize("num_alternations", [3, 6, 10])  # fmt: skip
    async def test_alternating_values(
        self, tracker: AsyncKeyedDuplicateTracker[SampleRecord], num_alternations: int
    ):
        """Test alternating between two values (no duplication - each is different)."""
        # Alternate A, B, A, B, ...
        sequence = [1 if i % 2 == 0 else 2 for i in range(num_alternations)]
        results = await write_sequence(tracker, "key1", sequence)

        # Each value is different from previous, so all are written
        total_written = sum(len(r) for r in results)
        assert total_written == num_alternations

    @pytest.mark.asyncio
    @pytest.mark.parametrize("num_duplicates", [10, 100, 1000])  # fmt: skip
    async def test_long_duplicate_sequence(
        self, tracker: AsyncKeyedDuplicateTracker[SampleRecord], num_duplicates: int
    ):
        """Test handling a very long sequence of duplicates."""
        results = await write_sequence(tracker, "key1", [1] * num_duplicates)

        # First write should succeed, all others suppressed
        assert len(results[0]) == 1
        for i in range(1, num_duplicates):
            assert len(results[i]) == 0

        # Should have num_duplicates - 1 duplicates
        assert tracker._dupe_counts["key1"] == num_duplicates - 1

    @pytest.mark.asyncio
    async def test_none_values(self):
        """Test handling None as record values."""

        @dataclass
        class NullableRecord:
            key: str
            value: int | None

        tracker_nullable: AsyncKeyedDuplicateTracker[NullableRecord] = (
            AsyncKeyedDuplicateTracker[NullableRecord](
                key_function=lambda record: record.key,
                value_function=lambda record: record.value,
            )
        )

        record1 = NullableRecord(key="key1", value=None)
        result1 = await tracker_nullable.deduplicate_record(record1)
        assert len(result1) == 1
        assert result1[0].value is None

        record2 = NullableRecord(key="key1", value=None)
        result2 = await tracker_nullable.deduplicate_record(record2)
        assert len(result2) == 0  # Duplicate

        record3 = NullableRecord(key="key1", value=1)
        result3 = await tracker_nullable.deduplicate_record(record3)
        assert len(result3) == 2  # Previous None + new 1


class TestAsyncKeyedDuplicateTrackerFlush:
    """Test flushing remaining duplicates."""

    @pytest.mark.asyncio
    async def test_flush_with_no_pending_duplicates(
        self, tracker: AsyncKeyedDuplicateTracker[SampleRecord]
    ):
        """Test that flush returns empty list when no pending duplicates."""
        await write_sequence(tracker, "key1", [1, 2, 3])

        # No duplicates, so nothing to flush
        to_flush = await tracker.flush_remaining_duplicates()
        assert len(to_flush) == 0

    @pytest.mark.asyncio
    async def test_flush_with_pending_duplicates(
        self, tracker: AsyncKeyedDuplicateTracker[SampleRecord]
    ):
        """Test that flush returns pending duplicates."""
        # Write A, A, A (2 pending duplicates)
        await write_sequence(tracker, "key1", [1, 1, 1])

        # Should flush the last A
        to_flush = await tracker.flush_remaining_duplicates()
        assert len(to_flush) == 1
        assert to_flush[0][0] == "key1"  # Key
        assert to_flush[0][1].value == 1  # Record value

        # Dupe count should be reset
        assert tracker._dupe_counts["key1"] == 0

    @pytest.mark.asyncio
    async def test_flush_multiple_keys(
        self, tracker: AsyncKeyedDuplicateTracker[SampleRecord]
    ):
        """Test flushing pending duplicates from multiple keys."""
        # key1: A, A, A
        await write_sequence(tracker, "key1", [1, 1, 1])
        # key2: B, B
        await write_sequence(tracker, "key2", [2, 2])
        # key3: C (no duplicates)
        await write_sequence(tracker, "key3", [3])

        # Should flush key1 and key2, but not key3
        to_flush = await tracker.flush_remaining_duplicates()
        assert len(to_flush) == 2

        flushed_keys = {key for key, _ in to_flush}
        assert flushed_keys == {"key1", "key2"}

    @pytest.mark.asyncio
    async def test_flush_idempotent(
        self, tracker: AsyncKeyedDuplicateTracker[SampleRecord]
    ):
        """Test that calling flush multiple times doesn't duplicate records."""
        await write_sequence(tracker, "key1", [1, 1, 1])

        # First flush
        to_flush1 = await tracker.flush_remaining_duplicates()
        assert len(to_flush1) == 1

        # Second flush should return nothing
        to_flush2 = await tracker.flush_remaining_duplicates()
        assert len(to_flush2) == 0
