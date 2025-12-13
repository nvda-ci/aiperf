# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Data serialization utilities for HTML export.

This module provides functions to serialize RunData, DataFrames, and metric
results to JSON-compatible dictionaries for embedding in HTML files.
"""

from pathlib import Path
from typing import Any

import orjson
import pandas as pd

from aiperf.common.constants import STAT_KEYS
from aiperf.common.models.record_models import MetricResult
from aiperf.plot.constants import ALL_STAT_KEYS, NON_METRIC_KEYS, STAT_LABELS
from aiperf.plot.core.data_loader import RunData, RunMetadata
from aiperf.plot.core.plot_specs import ExperimentClassificationConfig
from aiperf.plot.metric_names import get_metric_display_name


class HTMLDataSerializer:
    """
    Utilities for serializing RunData to JSON for HTML embedding.

    Handles:
    - DataFrame to records conversion
    - MetricResult serialization
    - RunMetadata serialization
    - Multi-run aggregated data serialization
    """

    @staticmethod
    def serialize_dataframe(
        df: pd.DataFrame | None,
        columns: list[str] | None = None,
    ) -> dict | None:
        """
        Serialize DataFrame to JSON-compatible format.

        Args:
            df: DataFrame to serialize
            columns: Specific columns to include (None = all)

        Returns:
            Dict with 'columns', 'data', 'dtypes', or None if df is None
        """
        if df is None or df.empty:
            return None

        if columns:
            available_cols = [c for c in columns if c in df.columns]
            df = df[available_cols]

        records = df.to_dict("records")

        dtypes = {}
        for col in df.columns:
            dtype = str(df[col].dtype)
            if "int" in dtype:
                dtypes[col] = "int"
            elif "float" in dtype:
                dtypes[col] = "float"
            elif "datetime" in dtype:
                dtypes[col] = "datetime"
            elif "bool" in dtype:
                dtypes[col] = "bool"
            else:
                dtypes[col] = "string"

        return {
            "columns": list(df.columns),
            "data": records,
            "dtypes": dtypes,
            "rowCount": len(df),
        }

    @staticmethod
    def serialize_metric_result(metric: MetricResult | dict | Any) -> dict | None:
        """
        Serialize MetricResult to JSON dict.

        Args:
            metric: MetricResult object or dict with metric values

        Returns:
            Dict with stat values, or None if metric is None
        """
        if metric is None:
            return None

        result = {}

        if isinstance(metric, MetricResult):
            result["unit"] = metric.unit
            for stat in STAT_KEYS:
                val = getattr(metric, stat, None)
                if val is not None:
                    result[stat] = val
        elif isinstance(metric, dict):
            if "unit" in metric:
                result["unit"] = metric["unit"]
            for stat in STAT_KEYS:
                if stat in metric and metric[stat] is not None:
                    result[stat] = metric[stat]
            if "value" in metric:
                result["value"] = metric["value"]

        return result if result else None

    @staticmethod
    def serialize_run_metadata(metadata: RunMetadata) -> dict:
        """
        Serialize RunMetadata to JSON dict.

        Args:
            metadata: RunMetadata object

        Returns:
            Dict with metadata fields
        """
        return {
            "runName": metadata.run_name,
            "runPath": str(metadata.run_path),
            "model": metadata.model,
            "concurrency": metadata.concurrency,
            "requestCount": metadata.request_count,
            "durationSeconds": metadata.duration_seconds,
            "endpointType": metadata.endpoint_type,
            "startTime": metadata.start_time,
            "endTime": metadata.end_time,
            "wasCancelled": metadata.was_cancelled,
            "experimentType": metadata.experiment_type,
            "experimentGroup": metadata.experiment_group,
        }

    @staticmethod
    def serialize_aggregated(aggregated: dict[str, Any]) -> dict:
        """
        Serialize aggregated statistics to JSON dict.

        Args:
            aggregated: Aggregated data dictionary from RunData

        Returns:
            Dict with serialized metrics
        """
        result = {}

        for key, value in aggregated.items():
            if key in NON_METRIC_KEYS:
                continue

            if (
                isinstance(value, MetricResult)
                or isinstance(value, dict)
                and "unit" in value
            ):
                serialized = HTMLDataSerializer.serialize_metric_result(value)
                if serialized:
                    result[key] = serialized

        return result

    @staticmethod
    def serialize_available_metrics(available_metrics: dict) -> dict:
        """
        Serialize available metrics info for JavaScript consumption.

        Args:
            available_metrics: Dict with 'display_names' and 'units'

        Returns:
            Dict with metric info formatted for JS
        """
        display_names = available_metrics.get("display_names", {})
        units = available_metrics.get("units", {})

        metrics = {}
        for metric_name in display_names:
            metrics[metric_name] = {
                "displayName": display_names.get(
                    metric_name, get_metric_display_name(metric_name)
                ),
                "unit": units.get(metric_name, ""),
            }

        return metrics

    @staticmethod
    def serialize_single_run(
        run: RunData,
        available_metrics: dict,
    ) -> dict:
        """
        Serialize a single run with full per-request data.

        Args:
            run: RunData object with per-request data
            available_metrics: Dict with display_names and units

        Returns:
            Dict with all run data for single-run HTML export
        """
        per_request_columns = (
            list(run.requests.columns) if run.requests is not None else []
        )

        return {
            "mode": "single",
            "metadata": HTMLDataSerializer.serialize_run_metadata(run.metadata),
            "requests": HTMLDataSerializer.serialize_dataframe(run.requests),
            "timeslices": HTMLDataSerializer.serialize_dataframe(run.timeslices),
            "gpuTelemetry": HTMLDataSerializer.serialize_dataframe(run.gpu_telemetry),
            "aggregated": HTMLDataSerializer.serialize_aggregated(run.aggregated),
            "availableMetrics": HTMLDataSerializer.serialize_available_metrics(
                available_metrics
            ),
            "perRequestColumns": per_request_columns,
            "sliceDuration": run.slice_duration,
            "statLabels": STAT_LABELS,
            "statKeys": ALL_STAT_KEYS,
        }

    @staticmethod
    def serialize_multi_run(
        runs: list[RunData],
        available_metrics: dict,
        classification_config: ExperimentClassificationConfig | None = None,
    ) -> dict:
        """
        Serialize multiple runs for comparison (aggregated data only).

        Args:
            runs: List of RunData objects
            available_metrics: Dict with display_names and units
            classification_config: Optional experiment classification config

        Returns:
            Dict with multi-run data for HTML export
        """
        serialized_runs = []
        for run in runs:
            serialized_runs.append(
                {
                    "metadata": HTMLDataSerializer.serialize_run_metadata(run.metadata),
                    "aggregated": HTMLDataSerializer.serialize_aggregated(
                        run.aggregated
                    ),
                }
            )

        swept_params = HTMLDataSerializer._detect_swept_parameters(runs)

        result = {
            "mode": "multi",
            "runs": serialized_runs,
            "availableMetrics": HTMLDataSerializer.serialize_available_metrics(
                available_metrics
            ),
            "sweptParameters": swept_params,
            "statLabels": STAT_LABELS,
            "statKeys": ALL_STAT_KEYS,
        }

        if classification_config:
            result["classification"] = {
                "baselines": classification_config.baselines,
                "treatments": classification_config.treatments,
                "default": classification_config.default,
                "groupDisplayNames": classification_config.group_display_names,
            }

        return result

    @staticmethod
    def _detect_swept_parameters(runs: list[RunData]) -> list[str]:
        """
        Detect parameters that vary across runs.

        Checks: model, concurrency, endpoint_type, request_count, experiment_group.

        Args:
            runs: List of RunData objects

        Returns:
            List of parameter names that have different values across runs
        """
        if len(runs) < 2:
            return []

        param_attrs = [
            ("model", "model"),
            ("concurrency", "concurrency"),
            ("endpoint_type", "endpoint_type"),
            ("request_count", "request_count"),
            ("experiment_group", "experiment_group"),
        ]

        param_values: dict[str, set] = {}

        for run in runs:
            for param_name, attr_name in param_attrs:
                value = getattr(run.metadata, attr_name, None)
                if value is not None:
                    param_values.setdefault(param_name, set()).add(value)

        swept = [param for param, values in param_values.items() if len(values) > 1]

        return swept

    @staticmethod
    def to_json_string(data: dict) -> str:
        """
        Serialize data dict to JSON string using orjson.

        Args:
            data: Dictionary to serialize

        Returns:
            JSON string
        """

        def default_serializer(obj: Any) -> Any:
            if isinstance(obj, Path):
                return str(obj)
            if hasattr(obj, "model_dump"):
                return obj.model_dump()
            if pd.isna(obj):
                return None
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        return orjson.dumps(data, default=default_serializer).decode("utf-8")
