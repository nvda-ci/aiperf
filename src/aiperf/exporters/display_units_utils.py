# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from urllib.parse import urlparse

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.constants import STAT_KEYS
from aiperf.common.exceptions import MetricUnitError
from aiperf.common.models import MetricResult
from aiperf.common.types import MetricTagT
from aiperf.metrics.metric_registry import MetricRegistry

_logger = AIPerfLogger(__name__)


def normalize_endpoint_display(url: str) -> str:
    """Normalize endpoint URL for display by removing scheme and trimming /metrics suffix.

    Args:
        url: The full URL to normalize (e.g., "https://host:9400/api/metrics")

    Returns:
        Normalized display string with netloc and trimmed path (e.g., "host:9400/api")
    """
    parsed = urlparse(url)
    path = parsed.path

    if path.endswith("/metrics"):
        path = path[: -len("/metrics")]

    display = parsed.netloc
    if path:
        display += path

    return display


# Mapping of Prometheus metric name suffixes to unit strings
# Note: Suffixes are sorted by length (longest first) for matching,
# so compound suffixes like _tokens_total will match before _total
_METRIC_SUFFIX_TO_UNIT: dict[str, str] = {
    # Time units
    "_seconds": "seconds",
    "_seconds_total": "seconds",  # Compound: seconds counter
    "_milliseconds": "milliseconds",
    "_ms": "milliseconds",
    "_ms_total": "milliseconds",  # Compound: milliseconds counter
    "_nanoseconds": "nanoseconds",
    "_ns": "nanoseconds",
    "_ns_total": "nanoseconds",  # Compound: nanoseconds counter
    # Size/data units
    "_bytes": "bytes",
    "_kilobytes": "kilobytes",
    "_megabytes": "megabytes",
    "_gigabytes": "gigabytes",
    "_bytes_total": "bytes",  # Compound: bytes counter
    # Count/quantity units
    "_total": "count",
    "_count": "count",
    "_tokens": "tokens",
    "_tokens_total": "tokens",  # Compound: token counter
    "_requests": "requests",
    "_requests_total": "requests",  # Compound: request counter
    "_messages": "messages",
    "_messages_total": "messages",  # Compound: message counter
    "_blocks": "blocks",
    "_connections": "connections",
    "_clients": "clients",
    "_services": "services",
    "_endpoints": "endpoints",
    "_errors": "errors",
    "_errors_total": "errors",  # Compound: error counter
    "_hits": "hits",
    "_hits_total": "hits",  # Compound: hit counter
    "_misses": "misses",
    "_misses_total": "misses",  # Compound: miss counter
    "_queries": "queries",
    "_preemptions": "preemptions",
    "_preemptions_total": "preemptions",  # Compound: preemption counter
    # Ratio/percentage units
    "_ratio": "ratio",
    "_percent": "percent",
    "_perc": "percent",  # Common shorthand (e.g., vllm:kv_cache_usage_perc)
    # Physical units
    "_celsius": "celsius",
    "_volts": "volts",
    "_amperes": "amperes",
    "_joules": "joules",
    "_watts": "watts",
    "_meters": "meters",
    # Special types
    "_info": "info",
}

# Pre-compute sorted suffixes (longest first) for efficient matching
_SORTED_SUFFIXES = sorted(_METRIC_SUFFIX_TO_UNIT.keys(), key=len, reverse=True)


def parse_unit_from_metric_name(metric_name: str) -> str | None:
    """Infer unit from Prometheus metric name suffix.

    Prometheus naming conventions use suffixes like _seconds, _bytes, _total
    to indicate the unit of a metric.

    Args:
        metric_name: Full metric name (e.g., 'request_duration_seconds')

    Returns:
        Unit string (e.g., 'seconds') or None if no recognized suffix
    """
    name_lower = metric_name.lower()
    # Check longer suffixes first to avoid partial matches
    # e.g., "_milliseconds" should match before "_seconds"
    for suffix in _SORTED_SUFFIXES:
        if name_lower.endswith(suffix):
            return _METRIC_SUFFIX_TO_UNIT[suffix]
    return None


def to_display_unit(result: MetricResult, registry: MetricRegistry) -> MetricResult:
    """
    Return a new MetricResult converted to its display unit (if different).
    """
    metric_cls = registry.get_class(result.tag)
    if result.unit and result.unit != metric_cls.unit.value:
        _logger.error(
            f"Metric {result.tag} has a unit ({result.unit}) that does not match the expected unit ({metric_cls.unit.value}). "
            f"({metric_cls.unit.value}) will be used for conversion."
        )

    display_unit = metric_cls.display_unit or metric_cls.unit

    if display_unit == metric_cls.unit:
        return result

    record = result.model_copy(deep=True)
    record.unit = display_unit.value

    for stat in STAT_KEYS:
        val = getattr(record, stat, None)
        if val is None:
            continue
        # Only convert numeric values
        if isinstance(val, int | float):
            try:
                new_value = metric_cls.unit.convert_to(display_unit, val)
            except MetricUnitError as e:
                _logger.warning(
                    f"Error converting {stat} for {result.tag} from {metric_cls.unit.value} to {display_unit.value}: {e}"
                )
                continue
            setattr(record, stat, new_value)
    return record


def convert_all_metrics_to_display_units(
    records: Iterable[MetricResult], registry: MetricRegistry
) -> dict[MetricTagT, MetricResult]:
    """Helper for exporters that want a tag->result mapping in display units."""
    out: dict[MetricTagT, MetricResult] = {}
    for r in records:
        out[r.tag] = to_display_unit(r, registry)
    return out
