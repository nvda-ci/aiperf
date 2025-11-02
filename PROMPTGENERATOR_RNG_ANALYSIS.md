<!--
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# PromptGenerator RNG Split Analysis
## Deep Dive: KV Cache Reuse and Benchmarking Implications

**Date**: 2025-01-XX
**Component**: `src/aiperf/dataset/generator/prompt.py`
**Question**: Should length, corpus position, and prefix use separate RNGs?

---

## Current Design

```python
self._length_rng = rng.derive("dataset.prompt.length")    # Token count sampling
self._corpus_rng = rng.derive("dataset.prompt.corpus")    # Corpus position selection
self._prefix_rng = rng.derive("dataset.prompt.prefix")    # Prefix selection
```

---

## Research Findings: KV Cache Reuse

### How KV Cache Reuse Works

Based on extensive research into TensorRT-LLM, vLLM, and prefix caching mechanisms:

**Key Principle**:
> "KV cache pages can be shared and reused by requests that start with the **same prompt**"

**Critical Requirements**:
1. **Exact prefix matching** - "only when the prefix is exactly identical, including whitespace and formatting"
2. **Token-level identity** - "Even a single character difference breaks the cache"
3. **Sequential dependency** - Full blocks (typically 128 tokens) must match

### Does KV Cache Care About Corpus Position?

**Answer: NO** ✅

KV cache reuse depends on:
- ✅ **Token sequence identity**: Same tokens in same order
- ❌ **NOT source provenance**: Where tokens came from in corpus is irrelevant

**Example**:
```
Request A: Samples "To be or not to be" from position 1000 in corpus
Request B: Samples "To be or not to be" from position 5000 in corpus
```

If both produce **identical token sequences**, KV cache can be reused. The corpus position is transparent to the model.

### Does KV Cache Care About Length Changes?

**Answer: YES - Length Changes Break Cache** ❌

**Key Finding**:
> "Truncated versions represent different prompts" - even with identical content

**Example**:
```
Request A: "To be or not to be, that is the question whether" (10 tokens)
Request B: "To be or not to be, that is" (7 tokens)
```

These are **different prompts** - no cache reuse occurs.

**Why it matters**: Prefix caching requires exact prefix match. Changing length means:
- Different number of cached KV states
- Cannot reuse partial cache
- Must recompute from scratch

---

## Benchmarking Implications

### Real-World LLM Deployment Patterns

Based on research into production LLM systems:

#### Pattern 1: System Prompts (Shared Prefix)
```
System: "You are a helpful assistant. Answer concisely."
User 1: "What is the capital of France?"
User 2: "How do I bake bread?"
User 3: "Explain quantum computing"
```

- **Same prefix**: System prompt is identical
- **Different continuations**: User queries vary
- **KV cache benefit**: Huge! System prompt cached, only compute user query

#### Pattern 2: Few-Shot Learning
```
Examples: "Translate English to French: dog -> chien, cat -> chat, bird -> oiseau"
Query 1: "tree"
Query 2: "house"
```

- **Same prefix**: Examples are identical
- **Different continuations**: Queries vary
- **KV cache benefit**: Examples cached across all queries

#### Pattern 3: Document QA (Shared Context)
```
Context: [5000 tokens of document text]
Question 1: "What is the main topic?"
Question 2: "Who are the key figures?"
```

- **Same prefix**: Document context
- **Different continuations**: Questions
- **KV cache benefit**: Massive! Document processed once

### What AIPerf is Testing

AIPerf's PromptGenerator creates **synthetic workloads** by:
1. **Length**: Sample token count from normal distribution
2. **Corpus**: Sample starting position in Shakespeare corpus
3. **Content**: Extract `length` tokens starting from `corpus_position`

**Key Insight**: These are **NOT** shared-prefix patterns!

Each generated prompt is:
- ❌ Different starting position in corpus
- ❌ Different length
- ❌ No shared prefix across requests

**Conclusion**: AIPerf is benchmarking **diverse workload patterns**, NOT prefix caching scenarios.

---

## Analysis: Should PromptGenerator Split RNGs?

### Scenario Analysis

Let's evaluate what happens with split vs single RNG when changing parameters:

#### Scenario 1: Change Length Distribution
```
Config A: mean=256, stddev=20
Config B: mean=512, stddev=20
```

**With Split RNGs** (current):
- Length RNG: Different → samples different token counts ✓
- Corpus RNG: **Same** → samples **same corpus positions**
- Result: Different-length excerpts from **same locations**

**With Single RNG** (proposed):
- All sampling: Different → different lengths AND different positions
- Result: Different-length excerpts from **different locations**

**Question**: Which is more realistic for benchmarking?

#### Scenario 2: Test Prefix Caching Sensitivity
```
Test A: 60% short prompts (256 tokens)
Test B: 60% long prompts (512 tokens)
```

**With Split RNGs** (current):
- When same pair selected → **Same corpus content, different length**
- Effect: Tests length sensitivity while controlling for content
- Useful for: "Does length alone affect KV cache hit rate?"

**With Single RNG** (proposed):
- When same pair selected → **Different corpus content AND different length**
- Effect: Tests both variables simultaneously
- Realistic for: "What's overall system performance?"

---

## Position Bias Research Findings

### LLMs Exhibit Position Bias

Critical finding from research:

> "Position bias is a prevalent issue in modern language models where models prioritize content based on its position within the given context"

**Implications for Corpus Sampling**:

**Position-Dependent Phenomena**:
1. **Causal attention** causes models to favor distant content
2. **Relative positional encodings (RoPE)** prefer nearby content
3. **Quality varies** across corpus positions

**Example from Research**:
```
Story A at position 1: Wins 100% of comparisons
Story B at position 2: Loses 100% of comparisons

[Reverse order]

Story A at position 2: Loses 100% of comparisons
Story B at position 1: Wins 100% of comparisons
```

**Key Insight**: **Content source DOES matter** for LLM behavior!

### Corpus Position is NOT Neutral

Sampling from different corpus positions produces:
- ✅ **Semantically different** content (Shakespeare corpus has themes, plots, character arcs)
- ✅ **Stylistically different** content (early vs late plays, comedy vs tragedy)
- ⚠️ **Position-bias affected** responses (LLMs process content differently based on position)

**Conclusion**: Corpus position RNG isolation is **scientifically justified** for controlling content variance.

---

## Verdict: PromptGenerator RNG Split Analysis

### Re-evaluation Based on KV Cache Research

| Concern | Split Benefit | Verdict |
|---------|---------------|---------|
| **Length RNG** | Controls token count independently | ✅ **KEEP** |
| **Corpus RNG** | Controls content source independently | ✅ **KEEP** |
| **Prefix RNG** | Controls prefix selection independently | ✅ **KEEP** |

### Updated Reasoning

#### 1. **Length RNG** ✅ Correct

**Why Split?**
- Length changes break KV cache (no reuse with truncated prompts)
- Length is a **primary performance variable** in LLM benchmarking
- Independent control enables **sensitivity analysis**: "How does length affect throughput?"

**Real Scenario**:
```
Test: "Impact of increasing prompt length on first-token latency"
- Keep corpus content constant → isolate length effect
- Vary length → measure latency delta
```

**Verdict**: **Essential for controlled experiments**

#### 2. **Corpus RNG** ✅ Correct

**Why Split?**
- Corpus position determines **semantic content**
- LLMs exhibit **position bias** - same content at different positions produces different behavior
- Content quality varies across corpus (Shakespeare has themes, plot arcs)

**Real Scenario**:
```
Test: "Length scaling behavior with consistent content themes"
- Keep content source constant → control semantic variables
- Vary length → measure how theme complexity scales
```

**Alternative Perspective**:
- If testing "realistic diverse workload" → might want correlated content changes
- But for **scientific experiments** → need independent variable control

**Verdict**: **Scientifically justified for controlled studies**

#### 3. **Prefix RNG** ✅ Correct

**Why Split?**
- Prefix caching is a **major optimization** in production LLMs
- Prefix selection is **orthogonal** to content and length
- Enables testing: "Impact of prefix pool size on cache hit rate"

**Real Scenario**:
```
Test: "Prefix cache reuse patterns"
- Keep content and length constant
- Vary prefix selection → measure cache hit rates
```

**Verdict**: **Enables realistic production scenario testing**

---

## Counterargument: When Single RNG Makes Sense

### Scenario: "Realistic Production Workload"

If goal is to simulate **true production traffic**:

**Characteristics**:
- ❌ Variables are **NOT** independent
- ❌ Short queries tend to have different content than long queries
- ❌ Different prefixes correlate with different use cases

**Example**:
```
Short prompts (256 tokens): Quick questions, simple tasks
Long prompts (2048 tokens): Complex analysis, multi-step reasoning
```

In reality, these differ in:
- Length ← User intent
- Content ← Task type
- Prefix ← Application domain

**For this scenario**: Single RNG creates **realistic correlation** between variables.

---

## Final Recommendation

### Context Matters: What is AIPerf Testing?

**Option A: Scientific Benchmarking** (Current Design) ✅

Goal: **Controlled experiments** to understand LLM behavior
- Use Case: "How does prompt length affect latency?"
- Use Case: "How does content complexity affect quality?"
- Use Case: "How does prefix caching affect throughput?"

**Design**: ✅ **KEEP split RNGs**
- Enables isolating individual variables
- Supports ablation studies
- Provides scientific rigor

**Option B: Production Workload Simulation**

Goal: **Realistic traffic patterns** for capacity planning
- Use Case: "Can we handle 10K RPS of real user traffic?"
- Use Case: "What's p99 latency under production load?"

**Design**: ❌ **Single RNG** might be better
- Variables naturally correlate in production
- More realistic request distribution
- Better capacity planning

---

## Conclusion

### Updated Verdict: ✅ **SPLIT RNGs ARE CORRECT FOR AIPERF**

After deep research into KV cache reuse, position bias, and benchmarking practices:

**PromptGenerator's split RNG design is scientifically sound because**:

1. **Length independence enables**: Testing KV cache behavior, latency scaling, throughput optimization
2. **Corpus independence enables**: Content quality control, position bias studies, semantic consistency
3. **Prefix independence enables**: Cache hit rate testing, production pattern simulation

**Key Insight from KV Cache Research**:
- KV cache doesn't care about corpus position (only token identity)
- KV cache DOES care about length (truncation breaks reuse)
- Position bias IS affected by corpus location

**These properties make split RNGs valuable for**:
- 🔬 Scientific experiments (ablation studies)
- 🎯 Sensitivity analysis (which variable matters most?)
- ⚙️ Optimization testing (cache effectiveness, batching strategies)

### Caveat

If AIPerf's goal shifts toward **pure production simulation**, reconsider single RNG for realistic variable correlation. But for **benchmarking and analysis** (current purpose), split design is optimal.

---

## Comparison: PromptGenerator vs SequenceLengthDistribution

| Aspect | PromptGenerator | SequenceLengthDistribution |
|--------|----------------|---------------------------|
| **What it controls** | How prompts are generated | Which ISL/OSL pairs are used |
| **Variables** | Length, content source, prefix | Pair selection, ISL variation, OSL variation |
| **Use case** | Generating test data | Defining workload mix |
| **Split justification** | ✅ Independent generation concerns | ❌ Creates spurious correlation |
| **Verdict** | ✅ **CORRECT** | ❌ **INCORRECT** |

**Why the difference?**

- **PromptGenerator**: Controls **how** to generate prompts (generation process)
  - Split = Independent control over generation parameters
  - Benefit = Scientific control for ablation studies

- **SequenceLengthDistribution**: Controls **what** workload to test (experimental config)
  - Split = Shared randomness across different experiments
  - Problem = Couples unrelated experimental configurations

---

## References

1. **KV Cache Reuse**: NVIDIA TensorRT-LLM documentation on KV cache page sharing
2. **Prefix Caching**: BentoML LLM Inference Handbook on prefix caching requirements
3. **Position Bias**: "Eliminating Position Bias of Language Models: A Mechanistic Approach" (arXiv)
4. **Production Patterns**: "BatchLLM: Optimizing Large Batched LLM Inference with Global Prefix Sharing"
5. **Benchmark Design**: "Design and Analysis of Benchmarking Experiments for Distributed Internet Services"

---

## Takeaway

**PromptGenerator's split RNG design is NOT like SequenceLengthDistribution's split**:

- PromptGenerator = Splitting **generation concerns** within a single experiment ✅
- SequenceLengthDistribution = Splitting **experimental configurations** across different tests ❌

**The distinction is critical**: One enables scientific control, the other creates unintended coupling.

**Final Status**: ✅ **PromptGenerator RNG split is VALIDATED and CORRECT**
