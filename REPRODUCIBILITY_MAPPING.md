<!--
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# AIPerf Reproducibility: Configuration to RNG Mapping

## Visual RNG Flow

```
USER INPUT: --random-seed VALUE
       |
       v
BOOTSTRAP: rng.init(value)  [bootstrap.py:106]
       |
       v
CREATE RNG MANAGER: _RNGManager(root_seed=value)
       |
       +----------+----------+----------+----------+
       |          |          |          |          |
       v          v          v          v          v
   Image      Audio       Prompt     Dataset    Timing
Generator   Generator   Generator   Samplers   Generator
   |          |          |          |          |
   +--+        +--+       +--+       +--+       +--+
      |           |          |          |          |
      v           v          v          v          v
 CHILD RNGs (SHA256-derived seeds)
   |           |          |          |          |
   ├─width     ├─duration ├─length  ├─random  ├─poisson
   ├─height    ├─bits     ├─corpus  ├─shuffle ├─cancel
   ├─format    ├─rate     └─prefix  └─...     └─...
   └─source    └─waveform
```

## Configuration to Component Mapping

### 1. PROMPT GENERATION

| Config Option | Component | RNG? | Child RNG |
|---|---|---|---|
| `--prompt-input-tokens-mean` | PromptGenerator | - | (no RNG) |
| `--prompt-input-tokens-stddev` | PromptGenerator | **YES** | `generator.prompt.length` |
| `--prompt-output-tokens-mean` | PromptGenerator | - | (no RNG) |
| `--prompt-output-tokens-stddev` | PromptGenerator | **YES** | `generator.prompt.length` |
| `--prompt-prefix-pool-size` | PromptGenerator | - | (no RNG) |
| `--prompt-prefix-length` | PromptGenerator | **YES** | `generator.prompt.prefix` |
| `--prompt-batch-size` | N/A | - | (no RNG) |

**RNG Operations**:
```
generator.prompt.length
  ├─ sample_positive_normal_integer(mean, stddev)  # For token counts
  └─ randint()  # Corpus position selection

generator.prompt.corpus
  ├─ randint()  # Which corpus tokens to use
  └─ shuffle()  # (not used currently)

generator.prompt.prefix
  ├─ generate_prompt()  # Entire prompts from corpus
  └─ (uses corpus RNG internally)
```

---

### 2. IMAGE GENERATION

| Config Option | Component | RNG? | Child RNG |
|---|---|---|---|
| `--image-width-mean` | ImageGenerator | - | (no RNG) |
| `--image-width-stddev` | ImageGenerator | **YES** | `dataset.image_generator` |
| `--image-height-mean` | ImageGenerator | - | (no RNG) |
| `--image-height-stddev` | ImageGenerator | **YES** | `dataset.image_generator` |
| `--image-format` | ImageGenerator | **YES** (if RANDOM) | `dataset.image_generator` |
| `--image-batch-size` | N/A | - | (no RNG) |

**RNG Operations**:
```
dataset.image_generator
  ├─ sample_positive_normal_integer(mean, stddev)  # Width
  ├─ sample_positive_normal_integer(mean, stddev)  # Height
  ├─ choice(formats)  # If format is RANDOM
  └─ choice(source_images)  # Select source image to resize
```

---

### 3. AUDIO GENERATION

| Config Option | Component | RNG? | Child RNG |
|---|---|---|---|
| `--audio-length-mean` | AudioGenerator | - | (no RNG) |
| `--audio-length-stddev` | AudioGenerator | **YES** | `dataset.audio_generator` |
| `--audio-format` | AudioGenerator | - | (no RNG) |
| `--audio-depths` | AudioGenerator | **YES** | `dataset.audio_generator` |
| `--audio-sample-rates` | AudioGenerator | **YES** | `dataset.audio_generator` |
| `--audio-num-channels` | AudioGenerator | - | (no RNG) |
| `--audio-batch-size` | N/A | - | (no RNG) |

**RNG Operations**:
```
dataset.audio_generator
  ├─ sample_normal(mean, stddev)  # Audio duration
  ├─ choice(depths)  # Bit depth selection
  ├─ choice(sample_rates)  # Sample rate selection
  └─ (waveform generation details)
```

---

### 4. DATASET SAMPLING

| Config Option | Component | RNG? | Child RNG |
|---|---|---|---|
| `--public-dataset` | DatasetLoader | - | (no RNG) |
| `--custom-dataset-type` | DatasetLoader | - | (no RNG) |
| `--input-file` | DatasetLoader | - | (no RNG) |
| `--dataset-sampling-strategy sequential` | SequentialSampler | NO | (not used) |
| `--dataset-sampling-strategy random` | RandomSampler | **YES** | `dataset.random_sampler` |
| `--dataset-sampling-strategy shuffle` | ShuffleSampler | **YES** | `dataset.shuffle_sampler` |
| `--conversation-num` | (config only) | - | (no RNG) |
| `--num-dataset-entries` | (config only) | - | (no RNG) |
| `--conversation-turn-mean` | ConversationGenerator | - | (no RNG) |
| `--conversation-turn-stddev` | ConversationGenerator | **YES** | `generator.conversation.turn` |
| `--conversation-turn-delay-mean` | ConversationGenerator | - | (no RNG) |
| `--conversation-turn-delay-stddev` | ConversationGenerator | **YES** | `generator.conversation.delay` |

**RNG Operations**:
```
dataset.random_sampler
  └─ choice(conversation_ids)  # Select with replacement

dataset.shuffle_sampler
  ├─ shuffle(conversation_ids)  # Reorder
  └─ shuffle(...)  # Repeat when exhausted

generator.conversation.turn
  └─ sample_positive_normal_integer(mean, stddev)

generator.conversation.delay
  └─ sample_positive_normal(mean, stddev)
```

---

### 5. REQUEST TIMING

| Config Option | Component | RNG? | Child RNG |
|---|---|---|---|
| `--request-rate` | TimingConfig | - | (no RNG) |
| `--request-rate-mode constant` | RequestRateStrategy | NO | (not used) |
| `--request-rate-mode poisson` | PoissonGenerator | **YES** | `timing.request_rate_poisson` |
| `--benchmark-duration` | (config only) | - | (no RNG) |
| `--benchmark-grace-period` | (config only) | - | (no RNG) |
| `--concurrency` | (config only) | - | (no RNG) |
| `--request-count` | (config only) | - | (no RNG) |
| `--warmup-request-count` | (config only) | - | (no RNG) |
| `--request-cancellation-rate` | CancellationStrategy | **YES** | `timing.request_cancellation` |
| `--request-cancellation-delay` | CancellationStrategy | - | (no RNG) |

**RNG Operations**:
```
timing.request_rate_poisson
  └─ expovariate(lambd)  # Exponential inter-arrival times

timing.request_cancellation
  └─ random()  # Determine if request should be cancelled
```

---

## Component Isolation Matrix

### "If I change X, what else changes?"

```
PROMPT GENERATION CHANGES
  ├─ Input stddev change
  │  ├─ AFFECTS: Prompt token lengths
  │  ├─ AFFECTS: Prompt content (corpus sampling)
  │  └─ NO EFFECT: Images, timing, dataset order
  │
  ├─ Output stddev change
  │  ├─ AFFECTS: Output specifications (no effect on actual output)
  │  └─ NO EFFECT: Images, timing, dataset order, input
  │
  └─ Prefix pool size change
     ├─ AFFECTS: Prefix prompts generated
     └─ NO EFFECT: Main prompts, images, timing, dataset order

IMAGE GENERATION CHANGES
  ├─ Width stddev change
  │  ├─ AFFECTS: Image dimensions
  │  ├─ AFFECTS: Image file size (pixel data)
  │  └─ NO EFFECT: Prompts, timing, dataset order
  │
  ├─ Height stddev change
  │  ├─ AFFECTS: Image dimensions
  │  └─ NO EFFECT: Prompts, timing, dataset order
  │
  └─ Format change to RANDOM
     ├─ AFFECTS: Image encoding (PNG vs JPEG)
     └─ NO EFFECT: Prompts, timing, dataset order

AUDIO GENERATION CHANGES
  ├─ Length stddev change
  │  ├─ AFFECTS: Audio duration
  │  └─ NO EFFECT: Prompts, images, timing, dataset order
  │
  ├─ Depths/sample_rates list change
  │  ├─ AFFECTS: Audio properties
  │  └─ NO EFFECT: Prompts, images, timing, dataset order

DATASET SAMPLING CHANGES
  ├─ Strategy: sequential → random
  │  ├─ AFFECTS: Conversation selection order
  │  └─ NO EFFECT: Prompt content, images, timing
  │
  └─ Strategy: random → shuffle
     ├─ AFFECTS: Conversation selection order
     └─ NO EFFECT: Prompt content, images, timing

REQUEST TIMING CHANGES
  ├─ Rate mode: constant → poisson
  │  ├─ AFFECTS: Request inter-arrival times
  │  └─ NO EFFECT: Prompts, images, dataset order
  │
  └─ Cancellation rate change
     ├─ AFFECTS: Which requests are cancelled
     └─ NO EFFECT: Prompts, images, timing (request sending)

RANDOM SEED CHANGES
  ├─ No seed → with seed
  │  ├─ AFFECTS: Everything becomes deterministic
  │  ├─ DATA SAME: Configuration unchanged
  │  └─ BEHAVIOR: Reproducible instead of random
  │
  └─ Seed value change (42 → 99)
     ├─ AFFECTS: All random values change
     ├─ AFFECTS: All random selections change
     └─ CONFIGURATION: Conceptually unchanged
```

---

## RNG Component Execution Order

### What Initializes When?

```
BOOTSTRAP (bootstrap.py:106)
  1. rng.reset()
  2. rng.init(user_config.input.random_seed)

COMPONENT INITIALIZATION (during system startup)
  1. ImageGenerator.__init__()
     ├─ self._rng = rng.derive("dataset.image_generator")
     └─ Registers: dataset.image_generator RNG

  2. AudioGenerator.__init__()
     ├─ self._rng = rng.derive("dataset.audio_generator")
     └─ Registers: dataset.audio_generator RNG

  3. PromptGenerator.__init__()
     ├─ self._length_rng = rng.derive("generator.prompt.length")
     ├─ self._corpus_rng = rng.derive("generator.prompt.corpus")
     ├─ self._prefix_rng = rng.derive("generator.prompt.prefix")
     └─ Registers: 3 prompt RNGs

  4. DatasetSamplers.__init__()
     ├─ RandomSampler: rng.derive("dataset.random_sampler")
     ├─ ShuffleSampler: rng.derive("dataset.shuffle_sampler")
     └─ SequentialSampler: NO RNG

  5. TimingGenerators.__init__()
     ├─ PoissonGenerator: rng.derive("timing.request_rate_poisson")
     ├─ ConstantGenerator: NO RNG
     └─ CancellationStrategy: rng.derive("timing.request_cancellation")

REQUEST GENERATION (during benchmark)
  - Each component uses its pre-derived RNG
  - RNG stream is deterministic (if seed was set)
  - RNG sequences are independent per component
```

**Key Point**: Order of initialization doesn't matter (SHA256-based, not counter-based)

---

## Seed Derivation Formula

For each component:

```
root_seed = user_config.input.random_seed  # e.g., 42

identifier = "dataset.image_generator"  # Unique per component

seed_string = f"{root_seed}:{identifier}"
            = "42:dataset.image_generator"

child_seed = SHA256(seed_string.encode('utf-8'))[0:8] as uint64
           = uint64(SHA256("42:dataset.image_generator"))
           = DETERMINISTIC VALUE X

RandomGenerator(child_seed)  # Now seeded with X
```

**Property**: Same (root_seed, identifier) pair always produces same child_seed

---

## Example: Tracing a Single Request

```
Request #1 with --random-seed 42 --prompt-input-tokens-stddev 50:

1. PromptGenerator.generate() called
   └─ Calls: self._length_rng.sample_positive_normal_integer(550, 50)
      └─ Uses child RNG seeded by SHA256("42:generator.prompt.length")
      └─ Produces: Token count = 523 (example)

2. For each prompt token:
   └─ Calls: self._corpus_rng.randint(0, corpus_size)
      └─ Uses child RNG seeded by SHA256("42:generator.prompt.corpus")
      └─ Produces: Token ID = 4523 (example)
      └─ Repeats 523 times (different values from sequence)

3. ImageGenerator.generate() called
   └─ Calls: self._rng.sample_positive_normal_integer(512, 0)
      └─ Uses child RNG seeded by SHA256("42:dataset.image_generator")
      └─ Produces: Width = 512 (stddev=0, so no variation)

4. Request sent at time T
   └─ If poisson: inter-arrival from timing.request_rate_poisson RNG
   └─ If constant: T = previous_time + fixed_interval (no RNG)

Run 2 with --random-seed 42 (identical seed):
  └─ All RNGs produce identical sequences
  └─ Request #1 identical to first run
  └─ All prompts identical
  └─ All image dimensions identical
  └─ All timing identical (if poisson)

Run 3 with --random-seed 99 (different seed):
  └─ All child seeds different
  └─ Token counts might be 531 instead of 523
  └─ Different corpus tokens selected
  └─ Different image properties
  └─ Different timing (if poisson)
```

---

## Summary: What to Remember

1. **Single Knob**: `--random-seed` controls ALL reproducibility
2. **Component Isolation**: Changing one component's config doesn't affect others
3. **Hash-Based**: Seed derivation is deterministic, not order-dependent
4. **Cross-Machine**: Same seed on any machine produces identical results
5. **Partially Deterministic**: Some options work without seed (sequential, constant rate)

