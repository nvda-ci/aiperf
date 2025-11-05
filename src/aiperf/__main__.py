# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys

from aiperf.cli import app
from aiperf.common.environment import Environment


def main() -> int:
    # TODO: HACK: Remove this once we can upgrade to v4 of cyclopts
    # This is a hack to allow the --gpu-telemetry flag to be used without a value
    # and it will be set to the default endpoints, which will inform the telemetry
    # exporter to print the telemetry to the console
    if "--gpu-telemetry" in sys.argv:
        idx = sys.argv.index("--gpu-telemetry")
        if idx >= len(sys.argv) - 1 or sys.argv[idx + 1].startswith("-"):
            for endpoint in reversed(Environment.GPU.DEFAULT_DCGM_ENDPOINTS):
                sys.argv.insert(idx + 1, endpoint)

    # Similar hack for --server-metrics flag to enable console output
    # When used without a value, insert "console" marker to enable console display
    # The actual endpoint is auto-derived from --url, so we just need a presence marker
    if "--server-metrics" in sys.argv:
        idx = sys.argv.index("--server-metrics")
        if idx >= len(sys.argv) - 1 or sys.argv[idx + 1].startswith("-"):
            sys.argv.insert(idx + 1, "console")

    return app(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
