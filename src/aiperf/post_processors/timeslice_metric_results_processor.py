# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from collections import defaultdict
from typing import Any

from aiperf.common.config import UserConfig
from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.decorators import implements_protocol
from aiperf.common.enums import ResultsProcessorType
from aiperf.common.exceptions import NoMetricValue, PostProcessorDisabled
from aiperf.common.factories import ResultsProcessorFactory
from aiperf.common.models.processor_summary_results import TimesliceSummaryResult
from aiperf.common.protocols import ResultsProcessorProtocol
from aiperf.common.types import MetricTagT, TimeSliceT
from aiperf.metrics.base_metric import BaseMetric
from aiperf.metrics.metric_dicts import MetricResultsDict
from aiperf.metrics.metric_registry import MetricRegistry
from aiperf.post_processors.metric_results_processor import MetricResultsProcessor


@implements_protocol(ResultsProcessorProtocol)
@ResultsProcessorFactory.register(ResultsProcessorType.TIMESLICE)
class TimesliceMetricResultsProcessor(MetricResultsProcessor):
    """Processor for metric results in timeslice mode.

    Groups metrics by time slices based on request timestamps and slice_duration.
    """

    def __init__(self, user_config: UserConfig, **kwargs: Any):
        super().__init__(user_config=user_config, **kwargs)

        if self.user_config.output.slice_duration is None:
            raise PostProcessorDisabled(
                "TimesliceMetricResultsProcessor requires slice_duration to be set"
            )

        self._slice_duration_ns: int = int(
            self.user_config.output.slice_duration * NANOS_PER_SECOND
        )

        # Set up aggregate metric object storage with lazy initialization
        # Using defaultdict(dict) instead of eager initialization to save memory
        # when only a small percentage of timeslices are actually used
        self._timeslice_instances_maps: dict[
            TimeSliceT, dict[MetricTagT, BaseMetric]
        ] = defaultdict(dict)

        # Use instance variable with defaultdict for auto-vivification
        self._timeslice_results: dict[TimeSliceT, MetricResultsDict] = defaultdict(
            MetricResultsDict
        )

    async def get_timeslice_index(self, request_start_ns: int):
        return int(request_start_ns / self._slice_duration_ns)

    async def get_instances_map(
        self, request_start_ns: int | None = None
    ) -> dict[MetricTagT, BaseMetric]:
        """Get the appropriate instances map based on mode with lazy initialization.

        Creates metric instances only when first accessed to save memory.
        Memory savings: 3,600 slices × 100 metrics × 1KB/instance = ~360MB saved
        if only 10% of slices are populated.
        """
        if request_start_ns is None:
            raise ValueError(
                "TimesliceMetricResultsProcessor::get_instances_map must be passed a request_start_ns"
            )

        timeslice_index = await self.get_timeslice_index(request_start_ns)

        # Get or create the instances map for this timeslice
        instances_map = self._timeslice_instances_maps[timeslice_index]

        # Lazy create metric instances only if map is empty (first access)
        if not instances_map:
            instances_map.update(
                {
                    tag: MetricRegistry.get_class(tag)()
                    for tag in MetricRegistry.all_tags()
                }
            )

        return instances_map

    async def get_results(
        self, request_start_ns: int | None = None
    ) -> MetricResultsDict:
        """Get the results dict for the appropriate timeslice based on request timestamp."""
        if request_start_ns is None:
            raise ValueError(
                "TimesliceMetricResultsProcessor::get_results must be passed a request_start_ns"
            )

        timeslice_index = await self.get_timeslice_index(request_start_ns)

        # Return (or create) the timeslice results dict for this timeslice
        return self._timeslice_results[timeslice_index]

    async def update_derived_metrics(self) -> None:
        for timeslice_results in self._timeslice_results.values():
            for tag, derive_func in self.derive_funcs.items():
                try:
                    timeslice_results[tag] = derive_func(timeslice_results)
                except NoMetricValue as e:
                    self.debug(f"No metric value for derived metric '{tag}': {e!r}")
                except Exception as e:
                    self.warning(f"Error deriving metric '{tag}': {e!r}")

    async def summarize(self) -> TimesliceSummaryResult:
        """Summarize the results.

        This will compute the values for the derived metrics, and then create the MetricResult objects for each metric.
        """
        await self.update_derived_metrics()

        # Compute and return the metric results.
        timeslice_metric_results = {}

        # Start timeslice indices at zero
        for counter, timeslice_index in enumerate(
            sorted(self._timeslice_results.keys())
        ):
            metric_results = [
                self._create_metric_result(tag, values)
                for tag, values in self._timeslice_results[timeslice_index].items()
            ]
            timeslice_metric_results[counter] = metric_results

        return TimesliceSummaryResult(timeslice_results=timeslice_metric_results)
