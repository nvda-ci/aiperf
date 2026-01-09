#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark sequential vs parallel Mooncake trace conversion.

Usage:
    python scripts/benchmark_parallel_conversion.py --use-mooncake
    python scripts/benchmark_parallel_conversion.py --num-sessions 20000 --workers 8 16
"""

from __future__ import annotations

import argparse
import os
import statistics
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

MOONCAKE_TRACE_URL = (
    "https://raw.githubusercontent.com/kvcache-ai/Mooncake/refs/heads/main/"
    "FAST25-release/arxiv-trace/mooncake_trace.jsonl"
)
MOONCAKE_CACHE_DIR = Path.home() / ".cache" / "aiperf" / "mooncake"
MOONCAKE_CACHE_FILE = MOONCAKE_CACHE_DIR / "mooncake_trace.jsonl"

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class BenchmarkResult:
    method: str
    num_workers: int | None
    num_sessions: int
    elapsed_seconds: float
    sessions_per_second: float


def download_mooncake_trace() -> str:
    if MOONCAKE_CACHE_FILE.exists():
        print(f"Using cached: {MOONCAKE_CACHE_FILE}")
        return str(MOONCAKE_CACHE_FILE)

    print(f"Downloading from {MOONCAKE_TRACE_URL}...")
    MOONCAKE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(MOONCAKE_TRACE_URL, MOONCAKE_CACHE_FILE)
    print(f"Cached to: {MOONCAKE_CACHE_FILE}")
    return str(MOONCAKE_CACHE_FILE)


def generate_synthetic_trace(num_sessions: int) -> str:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for i in range(num_sessions):
            f.write(
                f'{{"session_id": "sess_{i:06d}", "input_length": 768, '
                f'"output_length": 50, "hash_ids": [{i * 2}, {i * 2 + 1}]}}\n'
            )
        return f.name


def run_benchmark(
    trace_file: str,
    tokenizer_name: str,
    num_runs: int = 2,
    worker_counts: list[int] | None = None,
    batch_size: int = 100,
    skip_sequential: bool = False,
) -> list[BenchmarkResult]:
    from aiperf.common import random_generator as rng
    from aiperf.common.config import (
        EndpointConfig,
        PrefixPromptConfig,
        PromptConfig,
        UserConfig,
    )
    from aiperf.common.tokenizer import Tokenizer
    from aiperf.dataset.generator.prompt import PromptGenerator
    from aiperf.dataset.loader.mooncake_trace import MooncakeTraceDatasetLoader

    worker_counts = worker_counts or [8, 16]

    config = PromptConfig(
        mean=100,
        stddev=20,
        block_size=512,
        prefix_prompt=PrefixPromptConfig(pool_size=0, length=0),
    )
    user_config = UserConfig(endpoint=EndpointConfig(model_names=["benchmark"]))

    print(f"Loading tokenizer: {tokenizer_name}...")
    tokenizer = Tokenizer.from_pretrained(tokenizer_name)
    print("Tokenizer loaded.\n")

    # Get session count
    rng.reset()
    rng.init(42)
    generator = PromptGenerator(config, tokenizer)
    loader = MooncakeTraceDatasetLoader(
        filename=trace_file, prompt_generator=generator, user_config=user_config
    )
    trace_data = loader.load_dataset()
    num_sessions = len(trace_data)
    print(f"Trace file contains {num_sessions} sessions\n")

    results: list[BenchmarkResult] = []

    # Sequential
    if skip_sequential:
        avg_sequential = num_sessions / 780.0
        print("=" * 60)
        print("SEQUENTIAL (skipped, using ~780 sessions/sec baseline)")
        print("=" * 60)
        results.append(
            BenchmarkResult("sequential", None, num_sessions, avg_sequential, 780.0)
        )
    else:
        print("=" * 60)
        print("SEQUENTIAL")
        print("=" * 60)
        times = []
        for run in range(num_runs):
            rng.reset()
            rng.init(42)
            gen = PromptGenerator(config, tokenizer)
            ldr = MooncakeTraceDatasetLoader(
                filename=trace_file, prompt_generator=gen, user_config=user_config
            )
            data = ldr.load_dataset()

            start = time.perf_counter()
            ldr.convert_to_conversations(data)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            print(
                f"  Run {run + 1}/{num_runs}: {elapsed:.2f}s ({num_sessions / elapsed:.0f} sess/s)"
            )

        avg_sequential = statistics.mean(times)
        print(
            f"  Average: {avg_sequential:.2f}s ({num_sessions / avg_sequential:.0f} sess/s)\n"
        )
        results.append(
            BenchmarkResult(
                "sequential",
                None,
                num_sessions,
                avg_sequential,
                num_sessions / avg_sequential,
            )
        )

    # Parallel
    for nw in worker_counts:
        print("=" * 60)
        print(f"PARALLEL ({nw} workers, batch={batch_size})")
        print("=" * 60)
        times = []
        for run in range(num_runs):
            rng.reset()
            rng.init(42)
            gen = PromptGenerator(config, tokenizer)
            ldr = MooncakeTraceDatasetLoader(
                filename=trace_file, prompt_generator=gen, user_config=user_config
            )
            data = ldr.load_dataset()

            start = time.perf_counter()
            ldr.convert_to_conversations(data, num_workers=nw, batch_size=batch_size)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            print(
                f"  Run {run + 1}/{num_runs}: {elapsed:.2f}s ({num_sessions / elapsed:.0f} sess/s)"
            )

        avg = statistics.mean(times)
        speedup = avg_sequential / avg
        print(
            f"  Average: {avg:.2f}s ({num_sessions / avg:.0f} sess/s) - {speedup:.2f}x speedup\n"
        )
        results.append(
            BenchmarkResult(
                f"parallel-{nw}w", nw, num_sessions, avg, num_sessions / avg
            )
        )

    return results


def print_summary(results: list[BenchmarkResult]) -> None:
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    seq = next((r for r in results if r.method == "sequential"), None)
    if not seq:
        return

    print(f"{'Method':<20} {'Time':<10} {'Sess/s':<12} {'Speedup':<10}")
    print("-" * 52)
    for r in results:
        speedup = seq.elapsed_seconds / r.elapsed_seconds
        sp_str = f"{speedup:.2f}x" if r.method != "sequential" else "baseline"
        print(
            f"{r.method:<20} {r.elapsed_seconds:<10.2f} {r.sessions_per_second:<12.0f} {sp_str:<10}"
        )

    parallel = [r for r in results if r.num_workers]
    if parallel:
        best = max(parallel, key=lambda r: r.sessions_per_second)
        print(
            f"\nBest: {best.method} ({seq.elapsed_seconds / best.elapsed_seconds:.2f}x speedup)"
        )


def main():
    parser = argparse.ArgumentParser(description="Benchmark Mooncake trace conversion")
    parser.add_argument("--trace-file", type=str, help="Path to trace file")
    parser.add_argument(
        "--num-sessions", type=int, default=20000, help="Sessions for synthetic data"
    )
    parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument(
        "--workers", type=int, nargs="+", default=[8, 16], help="Worker counts to test"
    )
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--skip-sequential", action="store_true")
    parser.add_argument(
        "--use-mooncake", action="store_true", help="Use real Mooncake trace"
    )
    args = parser.parse_args()

    if args.use_mooncake:
        trace_file = download_mooncake_trace()
        temp_file = None
    elif args.trace_file:
        trace_file = args.trace_file
        temp_file = None
    else:
        print(f"Generating synthetic trace ({args.num_sessions} sessions)...")
        trace_file = generate_synthetic_trace(args.num_sessions)
        temp_file = trace_file

    try:
        results = run_benchmark(
            trace_file,
            args.tokenizer,
            args.runs,
            args.workers,
            args.batch_size,
            args.skip_sequential,
        )
        print_summary(results)
    finally:
        if temp_file:
            Path(temp_file).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
