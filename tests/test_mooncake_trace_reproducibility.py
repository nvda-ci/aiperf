# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test/benchmark for Mooncake trace parallel reproducibility.

This test downloads the actual mooncake trace file from GitHub and validates
that prompt generation is 100% identical between sequential and parallel processing
modes. This ensures that the HashIdRandomGenerator provides perfect reproducibility
across different worker ordering scenarios.
"""

import random
import tempfile
from pathlib import Path
from urllib.request import urlretrieve

import pytest

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

# URL to the actual Mooncake trace file
MOONCAKE_TRACE_URL = "https://raw.githubusercontent.com/kvcache-ai/Mooncake/refs/heads/main/FAST25-release/arxiv-trace/mooncake_trace.jsonl"

# Model to use for tokenization - using a small, fast model
DEFAULT_TOKENIZER_MODEL = "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="module")
def real_tokenizer():
    """Create a real tokenizer instance (cached for the entire module)."""
    print(f"\nLoading tokenizer: {DEFAULT_TOKENIZER_MODEL}...")
    tokenizer = Tokenizer.from_pretrained(DEFAULT_TOKENIZER_MODEL)
    print("Tokenizer loaded successfully")
    return tokenizer


@pytest.fixture(scope="module")
def mooncake_trace_file():
    """Download and cache the actual mooncake trace file."""
    # Create a temp directory for the trace file
    temp_dir = Path(tempfile.gettempdir()) / "aiperf_mooncake_test"
    temp_dir.mkdir(exist_ok=True)

    trace_file = temp_dir / "mooncake_trace.jsonl"

    # Download if not already cached
    if not trace_file.exists():
        print(f"\nDownloading Mooncake trace from {MOONCAKE_TRACE_URL}...")
        urlretrieve(MOONCAKE_TRACE_URL, trace_file)
        print(f"Downloaded to {trace_file}")
    else:
        print(f"\nUsing cached Mooncake trace from {trace_file}")

    yield str(trace_file)

    # Optional: cleanup after all tests in module complete
    # trace_file.unlink(missing_ok=True)


@pytest.fixture
def user_config():
    """Create a minimal UserConfig for testing."""
    return UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))


@pytest.fixture
def prompt_generator_with_config(real_tokenizer):
    """Create a PromptGenerator with test configuration."""
    config = PromptConfig(
        mean=100,
        stddev=20,
        block_size=512,
        prefix_prompt=PrefixPromptConfig(pool_size=0, length=0),
    )

    # Reset RNG to ensure clean state
    rng.reset()
    rng.init(42)  # Use fixed seed for reproducibility

    generator = PromptGenerator(config, real_tokenizer)
    return generator


class TestMooncakeTraceReproducibility:
    """Test suite for Mooncake trace reproducibility with parallel processing."""

    def test_sequential_vs_parallel_identical_prompts(
        self, mooncake_trace_file, real_tokenizer, user_config
    ):
        """
        Test that sequential and parallel (shuffled) processing produce identical prompts.

        This is the core reproducibility test: we generate prompts in two different orders
        (simulating different worker processing orders) and verify that the output is
        100% identical for each hash_id.
        """
        # ============================================================
        # SEQUENTIAL MODE: Process traces in original order using REAL loader
        # ============================================================
        print(
            "\n[Sequential Mode] Generating prompts in original order using real loader..."
        )
        sequential_prompts = self._load_and_generate_prompts(
            mooncake_trace_file, real_tokenizer, user_config, seed=42, shuffle=False
        )

        if not sequential_prompts:
            pytest.skip("No traces with hash_ids found in the mooncake trace file")

        print(f"\nTesting reproducibility with {len(sequential_prompts)} traces...")

        # ============================================================
        # PARALLEL MODE: Process traces in shuffled order using REAL loader
        # (simulates different workers processing in different orders)
        # ============================================================
        print(
            "\n[Parallel Mode] Generating prompts in shuffled order using real loader..."
        )
        parallel_prompts = self._load_and_generate_prompts(
            mooncake_trace_file, real_tokenizer, user_config, seed=42, shuffle=True
        )

        # ============================================================
        # VERIFICATION: Compare all generated prompts
        # ============================================================
        print("\n[Verification] Comparing prompts...")

        # Ensure we generated prompts for the same traces
        assert set(sequential_prompts.keys()) == set(parallel_prompts.keys()), (
            "Sequential and parallel modes generated prompts for different trace indices"
        )

        # Compare each prompt
        mismatches = []
        for trace_idx in sequential_prompts:
            seq_prompt = sequential_prompts[trace_idx]
            par_prompt = parallel_prompts[trace_idx]

            if seq_prompt != par_prompt:
                mismatches.append(
                    {
                        "trace_idx": trace_idx,
                        "seq_prompt": seq_prompt[:100],  # First 100 chars for debugging
                        "par_prompt": par_prompt[:100],
                    }
                )

        # Report results
        total_prompts = len(sequential_prompts)
        matching_prompts = total_prompts - len(mismatches)

        print(f"\n{'=' * 70}")
        print("REPRODUCIBILITY TEST RESULTS")
        print(f"{'=' * 70}")
        print(f"Total prompts generated: {total_prompts}")
        print(f"Matching prompts: {matching_prompts}")
        print(f"Mismatched prompts: {len(mismatches)}")
        print(f"Match rate: {100.0 * matching_prompts / total_prompts:.2f}%")
        print(f"{'=' * 70}\n")

        # Assert 100% identical
        assert len(mismatches) == 0, (
            f"Found {len(mismatches)} mismatched prompts between sequential and parallel modes. "
            f"First mismatch: {mismatches[0] if mismatches else 'N/A'}"
        )

        print(
            "✓ SUCCESS: All prompts are 100% identical between sequential and parallel modes!"
        )

    def test_same_seed_same_output(
        self, mooncake_trace_file, real_tokenizer, user_config
    ):
        """
        Test that using the same seed produces identical outputs across multiple runs.

        This verifies that the RNG is properly deterministic.
        """
        print("\nTesting seed determinism with real loader...")

        # Run 1
        prompts_run1 = self._load_and_generate_prompts(
            mooncake_trace_file, real_tokenizer, user_config, seed=42, shuffle=False
        )

        if not prompts_run1:
            pytest.skip("No traces with hash_ids found")

        # Run 2 with same seed
        prompts_run2 = self._load_and_generate_prompts(
            mooncake_trace_file, real_tokenizer, user_config, seed=42, shuffle=False
        )

        print(f"Compared {len(prompts_run1)} traces")

        # Verify identical
        assert prompts_run1 == prompts_run2, (
            "Same seed produced different outputs across runs"
        )

        print("✓ SUCCESS: Same seed produces identical outputs!")

    def test_different_seeds_different_output(
        self, mooncake_trace_file, real_tokenizer, user_config
    ):
        """
        Test that different seeds produce different outputs.

        This verifies that the RNG is actually working and not stuck.
        """
        print("\nTesting seed variation with real loader...")

        # Run with seed 42
        prompts_seed42 = self._load_and_generate_prompts(
            mooncake_trace_file, real_tokenizer, user_config, seed=42, shuffle=False
        )

        if not prompts_seed42:
            pytest.skip("No traces with hash_ids found")

        # Run with seed 123
        prompts_seed123 = self._load_and_generate_prompts(
            mooncake_trace_file, real_tokenizer, user_config, seed=123, shuffle=False
        )

        # Count differences
        different_count = sum(
            1 for idx in prompts_seed42 if prompts_seed42[idx] != prompts_seed123[idx]
        )

        print(f"Different prompts: {different_count}/{len(prompts_seed42)}")

        # We expect at least some differences
        assert different_count > 0, (
            "Different seeds produced identical outputs (RNG may not be working)"
        )

        print("✓ SUCCESS: Different seeds produce different outputs!")

    def test_hash_id_independence(self, real_tokenizer):
        """
        Test that each hash_id produces independent, deterministic sequences.

        This directly tests the HashIdRandomGenerator's core functionality.
        """
        print("\nTesting hash_id independence...")

        config = PromptConfig(
            mean=100,
            stddev=20,
            block_size=512,
            prefix_prompt=PrefixPromptConfig(pool_size=0, length=0),
        )

        # Test with different hash_ids
        test_hash_ids = [1, 2, 3, 100, 999, 12345]

        for seed in [42, 123]:
            print(f"\n  Testing with seed {seed}...")

            # Generate prompts in original order
            rng.reset()
            rng.init(seed)
            generator1 = PromptGenerator(config, real_tokenizer)

            prompts_order1 = {}
            for hash_id in test_hash_ids:
                prompt = generator1.generate(mean=100, stddev=0, hash_ids=[hash_id])
                prompts_order1[hash_id] = prompt

            # Generate prompts in shuffled order
            shuffled_ids = test_hash_ids.copy()
            random.shuffle(shuffled_ids)

            rng.reset()
            rng.init(seed)
            generator2 = PromptGenerator(config, real_tokenizer)

            prompts_order2 = {}
            for hash_id in shuffled_ids:
                prompt = generator2.generate(mean=100, stddev=0, hash_ids=[hash_id])
                prompts_order2[hash_id] = prompt

            # Verify each hash_id produces the same prompt regardless of order
            for hash_id in test_hash_ids:
                assert prompts_order1[hash_id] == prompts_order2[hash_id], (
                    f"Hash ID {hash_id} produced different prompts in different orders"
                )

            print(
                f"    ✓ All {len(test_hash_ids)} hash_ids produced consistent prompts"
            )

        print("\n✓ SUCCESS: Hash IDs are independent and deterministic!")

    # ============================================================
    # Helper Methods
    # ============================================================

    def _load_and_generate_prompts(
        self,
        trace_file: str,
        tokenizer: Tokenizer,
        user_config: UserConfig,
        seed: int,
        shuffle: bool = False,
    ) -> dict[str, str]:
        """
        Load traces and generate prompts using the REAL MooncakeTraceDatasetLoader.

        This simulates the actual production code path for loading and generating prompts.

        Args:
            trace_file: Path to the mooncake trace file
            tokenizer: Real tokenizer instance
            user_config: UserConfig instance
            seed: Random seed for reproducibility
            shuffle: Whether to shuffle the trace order (simulates parallel processing)

        Returns:
            Dictionary mapping session_id to generated prompt
        """
        # Reset RNG with specified seed
        rng.reset()
        rng.init(seed)

        # Create a PromptGenerator with the real tokenizer
        prompt_config = PromptConfig(
            mean=100,
            stddev=20,
            block_size=512,
            prefix_prompt=PrefixPromptConfig(pool_size=0, length=0),
        )
        generator = PromptGenerator(prompt_config, tokenizer)

        # Use the REAL MooncakeTraceDatasetLoader from production code
        loader = MooncakeTraceDatasetLoader(
            filename=trace_file, prompt_generator=generator, user_config=user_config
        )

        # Load traces using real loader
        trace_data = loader.load_dataset()

        # Shuffle the session order if requested (simulates different worker processing)
        session_ids = list(trace_data.keys())
        if shuffle:
            rng.derive("test.shuffle").shuffle(session_ids)
        else:
            # Keep original order
            session_ids = sorted(session_ids)

        # Convert to conversations using the REAL convert_to_conversations method
        # Process sessions in the specified order
        prompts = {}
        for session_id in session_ids:
            traces = trace_data[session_id]

            # Only process traces with hash_ids for this test
            if not traces or not traces[0].hash_ids:
                continue

            # Use the REAL conversion logic
            conversation_data = {session_id: traces}
            conversations = loader.convert_to_conversations(conversation_data)

            # Extract the generated prompt from the conversation
            if conversations and conversations[0].turns:
                prompt = conversations[0].turns[0].texts[0].contents[0]
                prompts[session_id] = prompt

        return prompts


@pytest.mark.benchmark
class TestMooncakeTraceBenchmark:
    """
    Benchmark tests for measuring performance of prompt generation.

    Use pytest -m benchmark to run these tests.
    """

    def test_benchmark_full_trace_generation(
        self, mooncake_trace_file, real_tokenizer, user_config, benchmark
    ):
        """
        Benchmark prompt generation for the entire mooncake trace file using REAL loader.

        This measures the performance of generating prompts for all traces using
        the actual production code path.
        """

        def generate_all_prompts_with_real_loader():
            # Reset RNG
            rng.reset()
            rng.init(42)

            # Create PromptGenerator
            config = PromptConfig(
                mean=100,
                stddev=20,
                block_size=512,
                prefix_prompt=PrefixPromptConfig(pool_size=0, length=0),
            )
            generator = PromptGenerator(config, real_tokenizer)

            # Use REAL loader
            loader = MooncakeTraceDatasetLoader(
                filename=mooncake_trace_file,
                prompt_generator=generator,
                user_config=user_config,
            )

            # Load and convert using real code paths
            trace_data = loader.load_dataset()
            conversations = loader.convert_to_conversations(trace_data)

            # Extract all prompts
            prompts = []
            for conv in conversations:
                if conv.turns and conv.turns[0].texts:
                    prompts.append(conv.turns[0].texts[0].contents[0])

            return prompts, len(trace_data)

        # Run the benchmark
        result, total_sessions = benchmark(generate_all_prompts_with_real_loader)

        print("\nBenchmark results:")
        print(f"  Total sessions: {total_sessions}")
        print(f"  Prompts generated: {len(result)}")


class TestParallelConversion:
    """Test suite for parallel conversion with HashIdRandomGenerator."""

    def test_sequential_vs_parallel_identical(
        self, mooncake_trace_file, real_tokenizer, user_config
    ):
        """
        Test that sequential and parallel conversion produce identical results.

        This is the core test for the parallel implementation: it verifies that
        convert_to_conversations() and convert_to_conversations_parallel()
        produce exactly the same conversations.
        """
        # Reset RNG
        rng.reset()
        rng.init(42)

        # Create PromptGenerator
        config = PromptConfig(
            mean=100,
            stddev=20,
            block_size=512,
            prefix_prompt=PrefixPromptConfig(pool_size=0, length=0),
        )
        generator = PromptGenerator(config, real_tokenizer)

        # Create loader
        loader = MooncakeTraceDatasetLoader(
            filename=mooncake_trace_file,
            prompt_generator=generator,
            user_config=user_config,
        )

        # Load trace data
        trace_data = loader.load_dataset()

        if not trace_data:
            pytest.skip("No trace data loaded")

        # ============================================================
        # Sequential conversion
        # ============================================================
        rng.reset()
        rng.init(42)
        generator_seq = PromptGenerator(config, real_tokenizer)
        loader_seq = MooncakeTraceDatasetLoader(
            filename=mooncake_trace_file,
            prompt_generator=generator_seq,
            user_config=user_config,
        )
        trace_data_seq = loader_seq.load_dataset()
        sequential_conversations = loader_seq.convert_to_conversations(trace_data_seq)

        # ============================================================
        # Parallel conversion
        # ============================================================
        rng.reset()
        rng.init(42)
        generator_par = PromptGenerator(config, real_tokenizer)
        loader_par = MooncakeTraceDatasetLoader(
            filename=mooncake_trace_file,
            prompt_generator=generator_par,
            user_config=user_config,
        )
        trace_data_par = loader_par.load_dataset()
        parallel_conversations = loader_par.convert_to_conversations_parallel(
            trace_data_par, num_workers=4
        )

        # ============================================================
        # Verification
        # ============================================================
        assert len(sequential_conversations) == len(parallel_conversations), (
            f"Different number of conversations: "
            f"sequential={len(sequential_conversations)}, parallel={len(parallel_conversations)}"
        )

        # Sort by session_id for comparison
        seq_by_session = {c.session_id: c for c in sequential_conversations}
        par_by_session = {c.session_id: c for c in parallel_conversations}

        assert set(seq_by_session.keys()) == set(par_by_session.keys()), (
            "Different session IDs between sequential and parallel"
        )

        mismatches = []
        for session_id in seq_by_session:
            seq_conv = seq_by_session[session_id]
            par_conv = par_by_session[session_id]

            if len(seq_conv.turns) != len(par_conv.turns):
                mismatches.append(f"Session {session_id}: different turn count")
                continue

            for i, (seq_turn, par_turn) in enumerate(
                zip(seq_conv.turns, par_conv.turns, strict=False)
            ):
                seq_prompt = seq_turn.texts[0].contents[0] if seq_turn.texts else ""
                par_prompt = par_turn.texts[0].contents[0] if par_turn.texts else ""

                if seq_prompt != par_prompt:
                    mismatches.append(f"Session {session_id}, turn {i}: prompts differ")

        assert len(mismatches) == 0, (
            f"Found {len(mismatches)} mismatches between sequential and parallel: "
            f"{mismatches[:5]}"
        )

        print(
            f"\n✓ SUCCESS: Sequential and parallel produced identical results "
            f"for {len(sequential_conversations)} conversations!"
        )

    def test_parallel_different_worker_counts_identical(
        self, mooncake_trace_file, real_tokenizer, user_config
    ):
        """
        Test that different worker counts produce identical results.

        This verifies that the HashIdRandomGenerator ensures reproducibility
        regardless of how work is distributed across workers.
        """
        config = PromptConfig(
            mean=100,
            stddev=20,
            block_size=512,
            prefix_prompt=PrefixPromptConfig(pool_size=0, length=0),
        )

        results_by_workers = {}

        for num_workers in [2, 4, 8]:
            rng.reset()
            rng.init(42)

            generator = PromptGenerator(config, real_tokenizer)
            loader = MooncakeTraceDatasetLoader(
                filename=mooncake_trace_file,
                prompt_generator=generator,
                user_config=user_config,
            )

            trace_data = loader.load_dataset()
            conversations = loader.convert_to_conversations_parallel(
                trace_data, num_workers=num_workers
            )

            # Extract prompts by session_id for comparison
            prompts_by_session = {}
            for conv in conversations:
                if conv.turns and conv.turns[0].texts:
                    prompts_by_session[conv.session_id] = (
                        conv.turns[0].texts[0].contents[0]
                    )

            results_by_workers[num_workers] = prompts_by_session

        # Compare all results
        baseline = results_by_workers[2]
        for num_workers, prompts in results_by_workers.items():
            if num_workers == 2:
                continue

            assert set(prompts.keys()) == set(baseline.keys()), (
                f"Different sessions with {num_workers} workers vs baseline"
            )

            for session_id in baseline:
                assert prompts[session_id] == baseline[session_id], (
                    f"Different prompt for session {session_id} with {num_workers} workers"
                )

        print("\n✓ SUCCESS: All worker counts (2, 4, 8) produced identical results!")

    def test_parallel_empty_data(self, real_tokenizer, user_config):
        """Test that parallel conversion handles empty data gracefully."""
        rng.reset()
        rng.init(42)

        config = PromptConfig(
            mean=100,
            stddev=20,
            block_size=512,
            prefix_prompt=PrefixPromptConfig(pool_size=0, length=0),
        )
        generator = PromptGenerator(config, real_tokenizer)

        # Create a temporary empty trace file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write("")  # Empty file
            temp_file = f.name

        try:
            loader = MooncakeTraceDatasetLoader(
                filename=temp_file, prompt_generator=generator, user_config=user_config
            )

            trace_data = loader.load_dataset()
            conversations = loader.convert_to_conversations_parallel(trace_data)

            assert conversations == [], "Expected empty list for empty trace data"
            print("\n✓ SUCCESS: Parallel conversion handles empty data correctly!")
        finally:
            Path(temp_file).unlink(missing_ok=True)

    def test_parallel_small_dataset_fallback(self, real_tokenizer, user_config):
        """
        Test that parallel conversion falls back to sequential for small datasets.

        When the number of sessions is less than num_workers, the parallel
        method should fall back to sequential processing.
        """
        rng.reset()
        rng.init(42)

        config = PromptConfig(
            mean=100,
            stddev=20,
            block_size=512,
            prefix_prompt=PrefixPromptConfig(pool_size=0, length=0),
        )
        generator = PromptGenerator(config, real_tokenizer)

        # Create a small trace file with just 2 sessions
        # Note: input_length must be compatible with block_size (512) and hash_ids count
        # With 2 hash_ids and block_size=512, input_length should be around 600-1024
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(
                '{"session_id": "sess1", "input_length": 600, "output_length": 20, "hash_ids": [1, 2]}\n'
            )
            f.write(
                '{"session_id": "sess2", "input_length": 600, "output_length": 20, "hash_ids": [3, 4]}\n'
            )
            temp_file = f.name

        try:
            loader = MooncakeTraceDatasetLoader(
                filename=temp_file, prompt_generator=generator, user_config=user_config
            )

            trace_data = loader.load_dataset()

            # Request more workers than sessions - should fall back to sequential
            conversations = loader.convert_to_conversations_parallel(
                trace_data, num_workers=8
            )

            assert len(conversations) == 2, (
                f"Expected 2 conversations, got {len(conversations)}"
            )
            print(
                "\n✓ SUCCESS: Parallel conversion falls back to sequential for small datasets!"
            )
        finally:
            Path(temp_file).unlink(missing_ok=True)


class TestGetSeed:
    """Test the get_seed() function in random_generator module."""

    def test_get_seed_returns_correct_seed(self):
        """Test that get_seed() returns the seed used to initialize RNG."""
        rng.reset()
        rng.init(12345)

        assert rng.get_seed() == 12345, (
            "get_seed() should return the initialization seed"
        )

        rng.reset()

    def test_get_seed_returns_none_for_non_deterministic(self):
        """Test that get_seed() returns None for non-deterministic mode."""
        rng.reset()
        rng.init(None)

        assert rng.get_seed() is None, (
            "get_seed() should return None for non-deterministic mode"
        )

        rng.reset()

    def test_get_seed_raises_before_init(self):
        """Test that get_seed() raises InvalidStateError before init."""
        rng.reset()

        with pytest.raises(Exception):  # InvalidStateError
            rng.get_seed()


class TestTokenizerModelName:
    """Test that Tokenizer stores and returns model name."""

    def test_tokenizer_stores_model_name(self, real_tokenizer):
        """Test that Tokenizer.model_name returns the model name used to create it."""
        assert real_tokenizer.model_name == DEFAULT_TOKENIZER_MODEL, (
            f"Expected model_name to be {DEFAULT_TOKENIZER_MODEL}, "
            f"got {real_tokenizer.model_name}"
        )
