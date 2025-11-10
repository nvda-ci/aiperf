# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from pathlib import Path

from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.models import ProcessRecordsResult
from aiperf.common.models.telemetry_models import TelemetryResults


@dataclass
class ExporterConfig:
    user_config: UserConfig
    service_config: ServiceConfig
    process_records_result: ProcessRecordsResult | None = None
    telemetry_results: TelemetryResults | None = None
    results: ProcessRecordsResult | None = None  # Legacy parameter (deprecated)

    def __post_init__(self):
        """Support legacy 'results' parameter and validate required fields."""
        # Handle legacy 'results' parameter
        if self.process_records_result is None and self.results is not None:
            object.__setattr__(self, "process_records_result", self.results)
        elif self.process_records_result is None:
            raise ValueError(
                "ExporterConfig requires 'process_records_result' parameter"
            )

        # If both provided, ensure they match
        if self.results is not None and self.results is not self.process_records_result:
            raise ValueError(
                "Cannot provide both 'results' and 'process_records_result' with different values"
            )


@dataclass
class FileExportInfo:
    export_type: str
    file_path: Path
