# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from aiperf.common.enums import MetricFlags
from aiperf.common.enums.metric_enums import GenericMetricUnit, MetricOverTimeUnit
from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models import ParsedResponseRecord
from aiperf.metrics.base_record_metric import BaseRecordMetric
from aiperf.metrics.metric_dicts import MetricRecordDict
from aiperf.metrics.types.request_latency_metric import RequestLatencyMetric


class NumVideosMetric(BaseRecordMetric[int]):
    """Number of videos metric."""

    tag = "num_videos"
    header = "Number of Videos"
    short_header = "Num Videos"
    unit = GenericMetricUnit.VIDEOS
    flags = MetricFlags.SUPPORTS_VIDEO_ONLY | MetricFlags.NO_CONSOLE

    def _parse_record(
        self, record: ParsedResponseRecord, record_metrics: MetricRecordDict
    ) -> int:
        """Parse the number of videos from the record by summing the number of videos in each turn."""
        num_videos = sum(
            len(video.contents)
            for turn in record.request.turns
            for video in turn.videos
        )
        if num_videos == 0:
            raise NoMetricValue(
                "Record must have at least one video in at least one turn."
            )
        return num_videos


class VideoThroughputMetric(BaseRecordMetric[float]):
    """Video throughput metric."""

    tag = "video_throughput"
    header = "Video Throughput"
    display_order = 870
    unit = MetricOverTimeUnit.VIDEOS_PER_SECOND
    flags = MetricFlags.SUPPORTS_VIDEO_ONLY
    required_metrics = {
        NumVideosMetric.tag,
        RequestLatencyMetric.tag,
    }

    def _parse_record(
        self, record: ParsedResponseRecord, record_metrics: MetricRecordDict
    ) -> float:
        """Parse the video throughput from the record by dividing the number of videos by the request latency."""
        num_videos = record_metrics.get_or_raise(NumVideosMetric)
        request_latency_sec = record_metrics.get_converted_or_raise(
            RequestLatencyMetric, self.unit.time_unit
        )
        return num_videos / request_latency_sec


class VideoLatencyMetric(BaseRecordMetric[float]):
    """Video latency metric."""

    tag = "video_latency"
    header = "Video Latency"
    short_header = "Video Latency"
    display_order = 871
    unit = MetricOverTimeUnit.MS_PER_VIDEO
    flags = MetricFlags.SUPPORTS_VIDEO_ONLY
    required_metrics = {
        NumVideosMetric.tag,
        RequestLatencyMetric.tag,
    }

    def _parse_record(
        self, record: ParsedResponseRecord, record_metrics: MetricRecordDict
    ) -> float:
        """Parse the video latency from the record by dividing the request latency by the number of videos."""
        num_videos = record_metrics.get_or_raise(NumVideosMetric)
        request_latency_ms = record_metrics.get_converted_or_raise(
            RequestLatencyMetric, self.unit.time_unit
        )
        return request_latency_ms / num_videos
