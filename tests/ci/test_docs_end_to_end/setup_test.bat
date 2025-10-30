@echo off
REM SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
REM SPDX-License-Identifier: Apache-2.0

echo Running python main.py --all-servers
cd /d "%AIPERF_SOURCE_DIR%\tests\ci\%CI_JOB_NAME%"
python main.py --all-servers
