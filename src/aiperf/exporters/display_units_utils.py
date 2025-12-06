# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
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
    "_reqs": "requests",  # Common shorthand (e.g., sglang:num_running_reqs)
    # Rate units (unambiguous suffixes only)
    "_gb_s": "GB/s",  # Throughput in GB/s (e.g., sglang:cache_transfer_gb_s)
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


# Description-based unit patterns
# Maps regex patterns to unit strings. Patterns are checked in order.
# Uses named groups for clarity. All patterns are case-insensitive.
_DESCRIPTION_UNIT_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Explicit unit mentions: "in seconds", "in ms", etc.
    (re.compile(r"\bin\s+seconds?\b", re.IGNORECASE), "seconds"),
    (re.compile(r"\bin\s+milliseconds?\b", re.IGNORECASE), "milliseconds"),
    (re.compile(r"\bin\s+ms\b", re.IGNORECASE), "milliseconds"),
    (re.compile(r"\bin\s+nanoseconds?\b", re.IGNORECASE), "nanoseconds"),
    (re.compile(r"\bin\s+ns\b", re.IGNORECASE), "nanoseconds"),
    (re.compile(r"\bin\s+bytes?\b", re.IGNORECASE), "bytes"),
    (re.compile(r"\bin\s+GB/s\b", re.IGNORECASE), "GB/s"),
    (re.compile(r"\bin\s+MB/s\b", re.IGNORECASE), "MB/s"),
    (re.compile(r"\bin\s+tokens?/s(?:ec(?:ond)?)?\b", re.IGNORECASE), "tokens/s"),
    (re.compile(r"\bin\s+requests?/s(?:ec(?:ond)?)?\b", re.IGNORECASE), "requests/s"),
    # Parenthetical unit mentions: "(seconds)", "(tokens/s)", etc.
    (re.compile(r"\(seconds?\)", re.IGNORECASE), "seconds"),
    (re.compile(r"\(milliseconds?\)", re.IGNORECASE), "milliseconds"),
    (re.compile(r"\(ms\)", re.IGNORECASE), "milliseconds"),
    (re.compile(r"\(nanoseconds?\)", re.IGNORECASE), "nanoseconds"),
    (re.compile(r"\(ns\)", re.IGNORECASE), "nanoseconds"),
    (re.compile(r"\(bytes?\)", re.IGNORECASE), "bytes"),
    (re.compile(r"\(GB/s\)", re.IGNORECASE), "GB/s"),
    (re.compile(r"\(MB/s\)", re.IGNORECASE), "MB/s"),
    (re.compile(r"\(tokens?/s(?:ec(?:ond)?)?\)", re.IGNORECASE), "tokens/s"),
    (re.compile(r"\(requests?/s(?:ec(?:ond)?)?\)", re.IGNORECASE), "requests/s"),
]

# Patterns for detecting ratio (0-1 scale) vs percent (0-100 scale)
# These override suffix-based inference when present in descriptions.
# Order matters: check ratio patterns first since they're more specific.
_RATIO_RANGE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\(0\.0\s*[-–—to]+\s*1\.0\)", re.IGNORECASE),  # (0.0-1.0), (0.0 to 1.0)
    re.compile(r"\(0\s*[-–—to]+\s*1\)", re.IGNORECASE),  # (0-1), (0 to 1)
    re.compile(r"\b0\.0\s*[-–—to]+\s*1\.0\b", re.IGNORECASE),  # 0.0-1.0 without parens
    re.compile(
        r"\brange\s+0(?:\.0)?\s*(?:[-–—]|to)\s*1(?:\.0)?\b", re.IGNORECASE
    ),  # range 0-1
    # "1 means/is/equals/= 100 percent/%" patterns
    re.compile(
        r"(?:\b|\()1(?:\.0)?\s*(?:means|is|equals?|==?)\s*100\s*(?:%|percent)",
        re.IGNORECASE,
    ),
]

_PERCENT_RANGE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\(0\s*[-–—to]+\s*100\)", re.IGNORECASE),  # (0-100)
    re.compile(r"\(0\.0\s*[-–—to]+\s*100\.0\)", re.IGNORECASE),  # (0.0-100.0)
    re.compile(r"\b0\s*[-–—to]+\s*100\s*%", re.IGNORECASE),  # 0-100%
    re.compile(r"\brange\s+0\s*[-–—to]+\s*100\b", re.IGNORECASE),  # range 0-100
]


def parse_scale_from_description(description: str | None) -> str | None:
    """Detect ratio vs percent scale from description range indicators.

    This function looks for explicit range indicators in descriptions to
    distinguish between:
    - ratio: Values in [0.0, 1.0] range (statistical convention)
    - percent: Values in [0, 100] range

    Args:
        description: Metric description text, or None

    Returns:
        "ratio" if 0-1 range detected, "percent" if 0-100 range detected,
        None if no range indicator found
    """
    if not description:
        return None

    # Check ratio patterns first (more specific)
    for pattern in _RATIO_RANGE_PATTERNS:
        if pattern.search(description):
            return "ratio"

    # Check percent patterns
    for pattern in _PERCENT_RANGE_PATTERNS:
        if pattern.search(description):
            return "percent"

    return None


def parse_unit_from_description(description: str | None) -> str | None:
    """Extract unit from metric description text.

    Looks for explicit unit mentions like "in seconds", "(tokens/s)", etc.
    This is more authoritative than suffix-based inference since descriptions
    often contain the exact unit specification.

    Args:
        description: Metric description text, or None

    Returns:
        Unit string if a recognized pattern is found, None otherwise
    """
    if not description:
        return None

    for pattern, unit in _DESCRIPTION_UNIT_PATTERNS:
        if pattern.search(description):
            return unit

    return None


def infer_unit(
    metric_name: str,
    description: str | None = None,
    existing_unit: str | None = None,
) -> str | None:
    """Infer the unit for a metric using multiple sources.

    Priority order:
    1. Scale from description (ratio vs percent range indicators)
    2. Unit from description (explicit "in seconds", etc.)
    3. Existing unit (if already set)
    4. Unit from metric name suffix

    The scale detection (step 1) can override suffix-based inference when
    a metric has a suffix like "_percent" but the description indicates
    a 0-1 range (which should be "ratio", not "percent").

    Args:
        metric_name: Full metric name (e.g., 'cache_hit_rate')
        description: Optional description text from the metric
        existing_unit: Optional pre-existing unit (e.g., from HELP text)

    Returns:
        Inferred unit string, or None if no unit can be determined
    """
    # Check for explicit scale indicators in description first
    # This takes priority because it's the most authoritative source
    scale = parse_scale_from_description(description)
    if scale:
        return scale

    # Check for explicit unit in description
    desc_unit = parse_unit_from_description(description)
    if desc_unit:
        return desc_unit

    # Use existing unit if set
    if existing_unit:
        return existing_unit

    # Fall back to suffix-based inference
    return parse_unit_from_metric_name(metric_name)


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
