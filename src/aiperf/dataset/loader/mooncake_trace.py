# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import uuid
from collections import defaultdict
from multiprocessing import Pool, shared_memory
from typing import Any

import numpy as np

from aiperf.common import random_generator as rng
from aiperf.common.config.user_config import UserConfig
from aiperf.common.decorators import implements_protocol
from aiperf.common.enums import CustomDatasetType
from aiperf.common.factories import CustomDatasetFactory
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.models import Conversation, Text, Turn
from aiperf.dataset.generator import PromptGenerator
from aiperf.dataset.loader.models import MooncakeTrace
from aiperf.dataset.loader.protocol import CustomDatasetLoaderProtocol

# Worker state
_w: dict[str, Any] = {}


def _init_worker(shm_name, shape, dtype, tokenizer_name, seed, block_size, sep_token):
    from aiperf.common.tokenizer import Tokenizer

    shm = shared_memory.SharedMemory(name=shm_name)
    _w.update(
        tokenizer=Tokenizer.from_pretrained(tokenizer_name),
        corpus=np.ndarray(shape, dtype=dtype, buffer=shm.buf),
        shm=shm,
        seed=seed,
        block_size=block_size,
        sep_token=sep_token,
    )


def _process_batch(batch):
    from aiperf.common import random_generator as wrng
    from aiperf.common.hash_id_random_generator import HashIdRandomGenerator
    from aiperf.dataset.generator.prompt import sample_tokens_from_corpus

    wrng.reset()
    wrng.init(_w["seed"])
    hrng = HashIdRandomGenerator.from_base_rng(wrng.derive("dataset.prompt.corpus"))

    corpus = _w["corpus"]
    block_size = _w["block_size"]
    sep = _w["sep_token"]
    decode = _w["tokenizer"].decode
    cache = {}

    def sample_block(hash_id, size):
        if hash_id in cache:
            return cache[hash_id]
        hrng.reseed_for_hash_id(hash_id)
        tokens = sample_tokens_from_corpus(corpus, size, hrng, sep)
        cache[hash_id] = tokens
        return tokens

    results = []
    for session_id, traces in batch:
        turns = []
        for t in traces:
            if t.get("text_input"):
                prompt = t["text_input"]
            elif t.get("hash_ids"):
                hids = t["hash_ids"]
                n_tokens = t["input_length"]
                last_size = n_tokens - (len(hids) - 1) * block_size
                tokens = []
                for i, hid in enumerate(hids):
                    tokens.extend(
                        sample_block(
                            hid, last_size if i == len(hids) - 1 else block_size
                        )
                    )
                prompt = decode(tokens, skip_special_tokens=False)
            else:
                prompt = ""

            turns.append(
                (t.get("timestamp"), t.get("delay"), prompt, t.get("output_length"))
            )
        results.append((session_id, turns))

    return results


@implements_protocol(CustomDatasetLoaderProtocol)
@CustomDatasetFactory.register(CustomDatasetType.MOONCAKE_TRACE)
class MooncakeTraceDatasetLoader(AIPerfLoggerMixin):
    def __init__(
        self,
        filename: str,
        prompt_generator: PromptGenerator,
        user_config: UserConfig,
        **kwargs,
    ):
        self.filename = filename
        self.prompt_generator = prompt_generator
        self.user_config = user_config
        self._skipped = 0
        self._start = user_config.input.fixed_schedule_start_offset
        self._end = user_config.input.fixed_schedule_end_offset
        super().__init__(user_config=user_config, **kwargs)

    def load_dataset(self) -> dict[str, list[MooncakeTrace]]:
        data: dict[str, list[MooncakeTrace]] = defaultdict(list)
        with open(self.filename) as f:
            for line in f:
                if not (line := line.strip()):
                    continue
                trace = MooncakeTrace.model_validate_json(line)
                ts = trace.timestamp
                if ts is not None and not (
                    (self._start is None or ts >= self._start)
                    and (self._end is None or ts <= self._end)
                ):
                    self._skipped += 1
                    continue
                data[trace.session_id or str(uuid.uuid4())].append(trace)

        if self._skipped:
            self.info(
                f"Skipped {self._skipped:,} traces outside [{self._start}, {self._end}]"
            )
        return data

    def convert_to_conversations(
        self,
        data: dict[str, list[MooncakeTrace]],
        num_workers: int | None = None,
        batch_size: int = 100,
    ) -> list[Conversation]:
        sessions = list(data.items())
        if not sessions:
            return []

        pg = self.prompt_generator
        corpus = np.array(pg._tokenized_corpus, dtype=np.int32)
        shm = shared_memory.SharedMemory(create=True, size=corpus.nbytes)

        try:
            np.copyto(
                np.ndarray(corpus.shape, dtype=corpus.dtype, buffer=shm.buf), corpus
            )

            batches = [
                [
                    (sid, [t.model_dump() for t in traces])
                    for sid, traces in sessions[i : i + batch_size]
                ]
                for i in range(0, len(sessions), batch_size)
            ]

            with Pool(
                num_workers or min(os.cpu_count() or 4, 16),
                _init_worker,
                (
                    shm.name,
                    corpus.shape,
                    corpus.dtype,
                    pg.tokenizer.model_name,
                    rng.get_seed(),
                    pg.config.input_tokens.block_size,
                    pg.tokenizer.block_separation_token_id,
                ),
            ) as pool:
                results = pool.map(_process_batch, batches)

            return [
                Conversation(
                    session_id=sid,
                    turns=[
                        Turn(
                            timestamp=ts,
                            delay=d,
                            texts=[Text(name="text", contents=[p])],
                            max_tokens=mt,
                        )
                        for ts, d, p, mt in turns
                    ],
                )
                for batch in results
                for sid, turns in batch
            ]
        finally:
            shm.close()
            shm.unlink()
