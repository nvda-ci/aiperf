# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NIM Image Embedding usage metrics.

These metrics track image and patch counts as reported in the API response's usage field
for NIM Image Embedding endpoints (e.g., C-RADIO).
"""

from aiperf.common.enums import GenericMetricUnit, MetricFlags
from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models import ParsedResponseRecord
from aiperf.metrics import BaseRecordMetric
from aiperf.metrics.derived_sum_metric import DerivedSumMetric
from aiperf.metrics.metric_dicts import MetricRecordDict


class UsageNumImagesMetric(BaseRecordMetric[int]):
    """
    API usage field num_images metric for NIM Image Embeddings.

    This represents the number of images processed as reported in the
    API response's usage field. Only applicable to image embedding endpoints.

    Formula:
        Usage Num Images = response.usage.num_images (last non-None)
    """

    tag = "usage_num_images"
    header = "Usage Num Images"
    short_header = "Usage Images"
    short_header_hide_unit = True
    unit = GenericMetricUnit.IMAGES
    flags = MetricFlags.LARGER_IS_BETTER | MetricFlags.NO_CONSOLE
    required_metrics = None

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> int:
        """
        Extract the API-reported num_images count from the record.

        Raises:
            NoMetricValue: If the API did not provide num_images count.
        """
        for response in reversed(record.responses):
            if response.usage:
                num_images = response.usage.num_images
                if num_images is not None:
                    return num_images

        raise NoMetricValue("Usage num_images is not available in the record.")


class UsageNumPatchesMetric(BaseRecordMetric[int]):
    """
    API usage field num_patches metric for NIM Image Embeddings.

    This represents the number of patches processed as reported in the
    API response's usage field. Applicable when using pyramidal patching
    (e.g., pyramid: [[1,1],[3,3],[5,5]] = 35 patches per image).

    Formula:
        Usage Num Patches = response.usage.num_patches (last non-None)
    """

    tag = "usage_num_patches"
    header = "Usage Num Patches"
    short_header = "Usage Patches"
    short_header_hide_unit = True
    unit = GenericMetricUnit.PATCHES
    flags = MetricFlags.LARGER_IS_BETTER | MetricFlags.NO_CONSOLE
    required_metrics = None

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> int:
        """
        Extract the API-reported num_patches count from the record.

        Raises:
            NoMetricValue: If the API did not provide num_patches count.
        """
        for response in reversed(record.responses):
            if response.usage:
                num_patches = response.usage.num_patches
                if num_patches is not None:
                    return num_patches

        raise NoMetricValue("Usage num_patches is not available in the record.")


class TotalUsageNumImagesMetric(DerivedSumMetric[int, UsageNumImagesMetric]):
    """
    Total API-reported num_images across all requests.

    Formula:
        ```
        Total Usage Num Images = Sum(Usage Num Images)
        ```
    """

    tag = "total_usage_num_images"
    header = "Total Usage Num Images"
    short_header = "Total Usage Images"
    short_header_hide_unit = True


class TotalUsageNumPatchesMetric(DerivedSumMetric[int, UsageNumPatchesMetric]):
    """
    Total API-reported num_patches across all requests.

    Formula:
        ```
        Total Usage Num Patches = Sum(Usage Num Patches)
        ```
    """

    tag = "total_usage_num_patches"
    header = "Total Usage Num Patches"
    short_header = "Total Usage Patches"
    short_header_hide_unit = True
