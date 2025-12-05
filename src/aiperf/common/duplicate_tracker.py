# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Duplicate tracker for deduplicating records."""

import asyncio
from collections import defaultdict
from collections.abc import Callable
from typing import Any, Generic, TypeVar

from aiperf.common.mixins import AIPerfLoggerMixin

TRecord = TypeVar("TRecord", bound=Any)


class AsyncKeyedDuplicateTracker(AIPerfLoggerMixin, Generic[TRecord]):
    """Tracker for deduplicating records by key and value.

    Args:
        key_function: A function that takes a record and returns a key for tracking duplicates. This is used to group records by key.
        value_function: A function that takes a record and returns a value for comparison. This is used to compare the current value to the previous value.

    Notes:
        The key_function and value_function are used to group records by key and compare values. This is useful for cases where
        the record itself contains timestamps or other metadata that is not relevant to the value being compared for deduplication.

    Tracks the previous record for each key and detects duplicates.

    Deduplication logic:
        Consecutive identical values are suppressed to save
        storage while preserving complete timeline information. The strategy:

        1. First occurrence → always written (marks start of period)
        2. Duplicates → skipped and counted
        3. Change detected → last duplicate written, then new record
           (provides end timestamp of previous period + start of new period)

        Example: Input A,A,A,B,B,C,D,D,D,D → Output A,A,B,B,C,D,D

        Why write the last occurrence? Time-series data needs actual observations:
            Without: A@t1, B@t4 ← You could guess A ended at ~t3, but no proof
            With:    A@t1, A@t3, B@t4 ← A was observed until t3

        Without the last occurrence, you'd rely on interpolation/assumptions rather
        than actual measured data. This enables accurate duration calculations,
        timeline visualization (Grafana), and time-weighted averages. Essential
        for metrics requiring precise change detection.

        Deduplication uses equality (==) on the metrics dictionary for each separate endpoint.
    """

    def __init__(
        self,
        key_function: Callable[[TRecord], str],
        value_function: Callable[[TRecord], Any] = lambda x: x,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        # Lock for safe access to creating dynamic locks for deduplication.
        self._lock_creation_lock = asyncio.Lock()
        self._dupe_locks: dict[str, asyncio.Lock] = {}
        self._dupe_counts: dict[str, int] = defaultdict(int)
        # Keep track of the previous record for each endpoint to detect duplicates.
        self._previous_records: dict[str, TRecord] = {}
        self._key_function = key_function
        self._value_function = value_function

    async def deduplicate_record(self, record: TRecord) -> list[TRecord]:
        """Deduplicate a record and return the records to write.

        Args:
            record: The record to deduplicate.

        Returns:
            A list of records to write containing either an empty list, the current record, or the current and previous records.
        """
        records_to_write: list[TRecord] = [record]

        key = self._key_function(record)
        value = self._value_function(record)

        if key not in self._dupe_locks:
            # Create a lock for this key if it doesn't exist
            async with self._lock_creation_lock:
                # Double check inside the lock to avoid race conditions
                if key not in self._dupe_locks:
                    self.trace(lambda: f"Creating lock for key: {key}")
                    self._dupe_locks[key] = asyncio.Lock()

        # Check for duplicates and update the records to write
        async with self._dupe_locks[key]:
            if key in self._previous_records:
                if self._value_function(self._previous_records[key]) == value:
                    self._dupe_counts[key] += 1
                    self.trace(
                        lambda: f"Duplicate found for key: {key}, incrementing dupe count to {self._dupe_counts[key]}"
                    )
                    # Clear the list instead of return so the previous record is still updated down below
                    records_to_write.clear()

                # If we have duplicates, we need to write the previous record before the current record,
                # in order to know when the change actually occurs.
                elif self._dupe_counts[key] > 0:
                    self._dupe_counts[key] = 0
                    self.trace(
                        lambda: f"New change detected for key: {key}, writing previous record and resetting dupe count"
                    )
                    records_to_write.insert(0, self._previous_records[key])

            self._previous_records[key] = record

        return records_to_write

    async def flush_remaining_duplicates(self) -> list[TRecord]:
        """Flush remaining duplicates for all keys on shutdown.

        When the system is stopping, there may be pending duplicates that haven't
        been written yet (because we're still in a duplicate sequence). This method
        returns the last occurrence for each key that has pending duplicates.

        Returns:
            A list of records that need to be flushed.
        """
        records_to_flush: list[TRecord] = []

        # Iterate through all keys that have pending duplicates
        for key in list(self._dupe_counts.keys()):
            if self._dupe_counts[key] > 0 and key in self._dupe_locks:
                async with self._dupe_locks[key]:
                    if self._dupe_counts[key] > 0 and key in self._previous_records:
                        self.trace(
                            lambda key=key: f"Flushing {self._dupe_counts[key]} remaining duplicates for key: {key}"
                        )
                        records_to_flush.append(self._previous_records[key])
                        self._dupe_counts[key] = 0

        return records_to_flush
