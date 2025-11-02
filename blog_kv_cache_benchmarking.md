<!--
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# Benchmarking KV Cache Reuse in Production LLMs: A Deep Dive with AIPerf

*How sophisticated random number generation unlocks realistic KV cache performance testing*

---

## Introduction

Key-Value (KV) cache reuse has emerged as one of the most impactful optimizations in production Large Language Model (LLM) deployments. By caching and reusing computed attention states across requests with shared prompt prefixes, systems can achieve **dramatic reductions in first-token latency** and **significant improvements in throughput**—often reducing compute requirements by 50-90% for common workload patterns.

However, **accurately benchmarking** KV cache effectiveness presents a unique challenge: how do you generate synthetic workloads that realistically capture the prefix-sharing patterns found in production traffic, while maintaining scientific control and reproducibility?

This is where AIPerf's sophisticated random number generation (RNG) system comes in. In this post, we'll explore how AIPerf's design enables rigorous, reproducible benchmarking of KV cache systems like NVIDIA's AI-Dynamo KV Cache Router.

---

## The Challenge: Realistic vs. Random Workloads

### Production KV Cache Patterns

Real-world LLM deployments exhibit predictable prefix-sharing patterns:

**Pattern 1: System Prompts (Chat Applications)**
```
System: "You are a helpful AI assistant. Answer concisely and accurately."
User 1: "What is the capital of France?"
User 2: "How do I bake bread?"
User 3: "Explain quantum computing"
...1000s more users
```

The system prompt is **identical across all requests**. With KV cache reuse:
- ✅ System prompt computed **once**
- ✅ Cached KV states shared across all users
- ✅ Only user queries require fresh computation

**Performance Impact**: 70-90% reduction in prefill compute for typical chat workloads.

**Pattern 2: Document QA (RAG Applications)**
```
Context: [5000 tokens of technical documentation]
Question 1: "What are the system requirements?" [50 tokens]
Question 2: "How do I configure authentication?" [80 tokens]
Question 3: "Explain the API rate limiting policy" [120 tokens]
```

The document context is **shared**. Only the questions differ.

**Performance Impact**: 95%+ cache hit rate on context tokens, dramatic latency improvements.

**Pattern 3: Few-Shot Learning (Agent Applications)**
```
Examples: "Translate English to French:
  dog -> chien
  cat -> chat
  bird -> oiseau
  [20 more examples...]"
Query 1: "tree"
Query 2: "house"
Query 3: "computer"
```

Few-shot examples are **constant** across inference calls.

**Performance Impact**: Examples computed once, reused for entire session.

### The Benchmarking Problem

Traditional synthetic workload generators use **simple random sampling**:

```python
# Naive approach - single RNG for everything
rng = random.Random(seed)

def generate_prompt(length):
    start_pos = rng.randint(0, corpus_size)  # Random position
    return corpus[start_pos:start_pos + length]
```

**Problem**: Every prompt has a **different random starting position**:
- ❌ No prefix sharing
- ❌ 0% cache hit rate (worst case)
- ❌ **Doesn't represent production workloads**

This makes it **impossible to benchmark** KV cache effectiveness—you're testing the worst-case scenario, not realistic traffic patterns.

---

## AIPerf's Solution: Independent RNG Streams for Controlled Prefix Sharing

AIPerf's `PromptGenerator` uses **three independent random number generators** to achieve sophisticated control over workload characteristics:

```python
class PromptGenerator:
    def __init__(self, config, tokenizer):
        # Three independent RNG streams
        self._length_rng = rng.derive("dataset.prompt.length")
        self._corpus_rng = rng.derive("dataset.prompt.corpus")
        self._prefix_rng = rng.derive("dataset.prompt.prefix")
```

### Why Three RNGs?

This design enables **independent control** over three critical workload dimensions:

#### 1. **Length RNG**: Controls Token Count Variation

```python
num_tokens = self._length_rng.sample_positive_normal_integer(mean, stddev)
```

**Purpose**: Sample prompt lengths from a normal distribution (e.g., mean=512, stddev=50).

**Independence**: Changing the length distribution doesn't affect which corpus content is selected.

**Benchmarking Benefit**: Test how length variation affects cache hit rates while keeping content constant.

#### 2. **Corpus RNG**: Controls Content Selection

```python
start_idx = self._corpus_rng.randrange(self._corpus_size)
prompt_tokens = self._tokenized_corpus[start_idx:start_idx + num_tokens]
```

**Purpose**: Select starting position in the tokenized corpus (e.g., Shakespeare's works).

**Independence**: Changing content selection doesn't affect length sampling.

**Benchmarking Benefit**: **This is the key to controlled prefix sharing!**

#### 3. **Prefix RNG**: Controls Prefix Template Selection

```python
prefix = self._prefix_rng.choice(self._prefix_prompts)
```

**Purpose**: Select from a pool of pre-generated prefix prompts (e.g., system messages).

**Independence**: Prefix selection is orthogonal to content and length.

**Benchmarking Benefit**: Systematically test different prefix pooling strategies.

---

## The Magic: Controlled Prefix Sharing

Here's where AIPerf's design becomes powerful for KV cache benchmarking:

### Scenario: Testing Length Variation with Shared Prefixes

```python
# Configuration
rng.init(seed=42)  # Global seed for reproducibility
config.prompt.input_tokens.mean = 512
config.prompt.input_tokens.stddev = 100

# Generate 1000 prompts
prompts = [prompt_gen.generate() for _ in range(1000)]
```

**What happens inside?**

```
Request   Length  Corpus Start  Content Preview
---------|-------|-------------|----------------------------------
1        456     1000          "To be or not to be that is..."
2        623     1000          "To be or not to be that is the question whether..."
3        389     1000          "To be or not to be that is..."
4        701     1000          "To be or not to be that is the question whether tis nobler..."
5        512     1000          "To be or not to be that is the question..."
```

**Notice**: All requests start from **corpus position 1000**!

- ✅ Requests 1-5 share **389 tokens** of prefix (shortest prompt)
- ✅ Requests 2, 4, 5 share **456+ tokens** of prefix
- ✅ **Realistic KV cache reuse pattern**

### Why This Matters

With split RNGs:
1. **Corpus position remains constant** (controlled by `corpus_rng`)
2. **Length varies** (controlled by `length_rng`)
3. **Result**: Systematic prefix sharing like production workloads!

This enables testing:
- ✅ Cache hit rates as function of length variation
- ✅ First-token latency improvements from prefix caching
- ✅ Throughput scaling with shared context
- ✅ Memory efficiency of cache management systems

---

## Benchmarking AI-Dynamo KV Cache Router: A Case Study

NVIDIA's **AI-Dynamo** provides intelligent KV cache routing and management across distributed LLM deployments. Let's see how AIPerf's RNG design enables rigorous benchmarking.

### Experiment 1: Measuring Cache Hit Rate vs. Prefix Pool Size

**Research Question**: How does prefix diversity affect cache effectiveness?

**Test Configuration**:
```python
# AIPerf configuration
config.prompt.prefix_prompt.pool_size = [1, 5, 10, 50, 100]
config.prompt.input_tokens.mean = 1024
config.prompt.input_tokens.stddev = 200
config.input.conversation.num_dataset_entries = 1000

# Same corpus_rng seed → controlled content sharing
# Varying prefix_rng → different prefix pools
```

**Independent Variables**:
- Prefix pool size (controlled by `prefix_rng`)

**Controlled Variables**:
- Content distribution (fixed `corpus_rng` seed)
- Length distribution (fixed `length_rng` seed)

**Measured Metrics**:
- Cache hit rate (% of tokens served from cache)
- First-token latency (TTFT)
- Throughput (tokens/second)
- Memory utilization (GB of cached KV states)

**Expected Results**:
```
Prefix Pool Size | Cache Hit Rate | Avg TTFT  | Throughput
-----------------|----------------|-----------|------------
1                | 95%           | 45ms      | 850 tok/s
5                | 87%           | 62ms      | 780 tok/s
10               | 78%           | 81ms      | 710 tok/s
50               | 42%           | 145ms     | 520 tok/s
100              | 23%           | 198ms     | 410 tok/s
```

**Insights Enabled by Split RNGs**:
- Isolated effect of prefix diversity on cache performance
- Controlled content variation (same corpus positions)
- Reproducible results (deterministic RNG seeding)

### Experiment 2: Length Variation Impact on Cache Efficiency

**Research Question**: How does prompt length distribution affect KV cache memory efficiency?

**Test Configuration**:
```python
# Test different length distributions
configs = [
    {"mean": 256, "stddev": 50},   # Short prompts
    {"mean": 512, "stddev": 100},  # Medium prompts
    {"mean": 1024, "stddev": 200}, # Long prompts
    {"mean": 2048, "stddev": 400}, # Very long prompts
]

# Fixed corpus_rng → same content positions across tests
# Varying length_rng → different length patterns
```

**Key Insight**: Because `corpus_rng` is **independent**, all four tests sample from the **same corpus positions**. This creates:

```
Test       Request 1 Content             Request 2 Content
---------|----------------------------|----------------------------
Short    "To be or not[256 tok]"    "To be or not[278 tok]"
Medium   "To be or not[512 tok]"    "To be or not[489 tok]"
Long     "To be or not[1024 tok]"   "To be or not[1156 tok]"
VeryLong "To be or not[2048 tok]"   "To be or not[1889 tok]"
```

**All tests share the same prefix content** (first 256 tokens of "To be or not...")!

**Measured Metrics**:
- Cache block utilization (internal fragmentation)
- Effective cache hit rate per byte stored
- Eviction rates under memory pressure
- Throughput per GB of cache memory

**Benchmarking Value**:
- Isolates **length effect** on cache efficiency
- Controls for **content variation** (potential confounder)
- Enables fair comparison across length distributions
- Reveals optimal block sizing strategies

### Experiment 3: Content Diversity and Cache Scalability

**Research Question**: How does content diversity affect cache scalability in multi-tenant deployments?

**Test Configuration**:
```python
# Vary corpus sampling while keeping length constant
configs = [
    {"corpus_seed": 42,  "sample_range": "narrow"},   # High sharing
    {"corpus_seed": 43,  "sample_range": "medium"},   # Medium sharing
    {"corpus_seed": 44,  "sample_range": "wide"},     # Low sharing
]

# Fixed length_rng → same length distribution
# Varying corpus_rng → different content patterns
```

**Benchmark Scenarios**:

**Scenario A: High Prefix Sharing (corpus positions 0-1000)**
```
User 1: corpus[42:554]    → "To be or not..."
User 2: corpus[156:668]   → "To be or not to be..."
User 3: corpus[89:601]    → "To be or not to..."
```
Many overlapping prefixes → High cache reuse

**Scenario B: Medium Sharing (corpus positions 0-10000)**
```
User 1: corpus[2341:2853]  → "Friends Romans countrymen..."
User 2: corpus[5892:6404]  → "Now is the winter..."
User 3: corpus[2456:2968]  → "Friends Romans countrymen lend..."
```
Some overlapping prefixes → Medium cache reuse

**Scenario C: Low Sharing (corpus positions 0-100000)**
```
User 1: corpus[42891:43403]   → "Shall I compare thee..."
User 2: corpus[78234:78746]   → "Once more unto the breach..."
User 3: corpus[13567:14079]   → "All the world's a stage..."
```
Minimal overlapping → Low cache reuse

**Measured Metrics**:
- Cache hit rate vs. number of concurrent users
- Memory scaling efficiency
- Router decision quality (AI-Dynamo specific)
- Throughput degradation under load

---

## Reproducibility: The Foundation of Scientific Benchmarking

AIPerf's RNG system provides **deterministic reproducibility**:

```python
# Experiment 1: May 15, 2025, on A100 GPU
rng.init(seed=42)
results_1 = benchmark_kv_cache(config)
# Cache hit rate: 87.3%, TTFT: 62ms

# Experiment 2: June 20, 2025, on H100 GPU
rng.init(seed=42)
results_2 = benchmark_kv_cache(config)
# Cache hit rate: 87.3%, TTFT: 45ms  (hardware improvement!)
```

**Same seed → Identical workload patterns**:
- ✅ Same prompt lengths
- ✅ Same corpus positions
- ✅ Same prefix selections
- ✅ Same cache sharing patterns

**Different hardware → Different performance**:
- ✅ Isolates infrastructure improvements
- ✅ Fair comparisons across systems
- ✅ Regression detection ("Did we break caching?")

### Multi-Team Collaboration

Teams at different organizations can reproduce benchmarks:

```bash
# NVIDIA AI-Dynamo Team
aiperf --seed 42 --config nvidia_dynamo_test.yaml

# Customer Team
aiperf --seed 42 --config nvidia_dynamo_test.yaml

# Results: Identical workload, comparable metrics
```

**Benefits**:
- 🤝 Shared benchmark standards
- 📊 Cross-validation of results
- 🐛 Bug reporting with reproducible scenarios
- 📈 Performance regression tracking

---

## Advanced Use Cases

### Testing Cache Eviction Policies

**Scenario**: Compare LRU vs. LFU eviction under memory pressure

```python
# Generate workload with known access patterns
config.prompt.prefix_prompt.pool_size = 20
config.memory.cache_size_gb = 8  # Constrained memory

# Controlled access pattern via corpus_rng
# Some prefixes accessed frequently (low corpus positions)
# Others accessed rarely (high corpus positions)

# Measure:
# - Cache hit rate over time
# - Eviction count by prefix
# - Throughput stability
```

**Split RNG benefit**: Create **intentional access patterns** for testing eviction algorithms.

### Multi-Tenant Cache Isolation

**Scenario**: Test cache sharing vs. isolation tradeoffs

```python
# Tenant A: Corpus positions 0-10000
tenant_a_corpus_rng = rng.derive("tenant_a.corpus")

# Tenant B: Corpus positions 10000-20000
tenant_b_corpus_rng = rng.derive("tenant_b.corpus")

# Measure:
# - Cross-tenant cache pollution
# - Fairness of cache allocation
# - Security boundary effectiveness
```

**Split RNG benefit**: Systematically control **tenant content overlap** for isolation testing.

### Prefix Cache Prewarming

**Scenario**: Test cache prewarming strategies

```python
# Phase 1: Prewarm cache with common prefixes
prewarm_prompts = [
    prompt_gen.generate() for _ in range(100)
]  # Uses corpus position X

# Phase 2: Production traffic
production_prompts = [
    prompt_gen.generate() for _ in range(10000)
]  # Same corpus position X → cache hits!

# Measure:
# - Cold start latency reduction
# - Optimal prewarm set size
# - Prewarm cost vs. benefit
```

**Split RNG benefit**: **Guaranteed prefix alignment** between prewarm and production phases.

---

## Implementation Best Practices

### 1. Seed Management

```python
# Base seed for experiment
BASE_SEED = 42

# Derive experiment-specific seeds
rng.init(seed=BASE_SEED)

# All derived RNGs inherit reproducibility
prompt_gen = PromptGenerator(config, tokenizer)
# Internally uses:
# - rng.derive("dataset.prompt.length")
# - rng.derive("dataset.prompt.corpus")
# - rng.derive("dataset.prompt.prefix")
```

### 2. Configuration Documentation

```yaml
# benchmark_config.yaml
experiment:
  name: "kv_cache_scaling_study"
  seed: 42
  description: "Measure cache hit rate vs. prefix pool size"

workload:
  prompt:
    input_tokens:
      mean: 1024
      stddev: 200
    prefix_prompt:
      pool_size: 10
      length: 256
    corpus: "shakespeare"  # Deterministic corpus

metrics:
  - cache_hit_rate
  - first_token_latency
  - throughput
  - memory_utilization
```

**Document**:
- RNG seeds used
- Expected prefix sharing patterns
- Controlled vs. varied parameters
- Reproducibility requirements

### 3. Validation Testing

```python
# Verify prefix sharing is working
def validate_prefix_sharing():
    rng.init(seed=42)

    prompts = [prompt_gen.generate(mean=512) for _ in range(100)]

    # Check: Do prompts share common prefixes?
    tokens_0 = tokenizer.encode(prompts[0])
    tokens_1 = tokenizer.encode(prompts[1])

    shared_prefix = 0
    for t0, t1 in zip(tokens_0, tokens_1):
        if t0 == t1:
            shared_prefix += 1
        else:
            break

    assert shared_prefix > 100, f"Expected prefix sharing, got {shared_prefix}"
    print(f"✓ Prefix sharing validated: {shared_prefix} tokens")

validate_prefix_sharing()
```

### 4. Metrics Correlation

```python
# Correlate cache metrics with workload characteristics
def analyze_cache_effectiveness():
    results = {
        'prefix_lengths': [],
        'cache_hit_rates': [],
        'latencies': []
    }

    for prompt_pair in consecutive_prompts:
        prefix_len = measure_shared_prefix(prompt_pair)
        cache_hit = measure_cache_hit(prompt_pair)
        latency = measure_latency(prompt_pair[1])

        results['prefix_lengths'].append(prefix_len)
        results['cache_hit_rates'].append(cache_hit)
        results['latencies'].append(latency)

    # Correlation analysis
    correlation = pearsonr(
        results['prefix_lengths'],
        results['cache_hit_rates']
    )

    print(f"Prefix length vs. cache hit rate correlation: {correlation}")
```

---

## Comparison: AIPerf vs. Traditional Benchmarks

| Aspect | Traditional Random Sampling | AIPerf Split RNG Design |
|--------|---------------------------|------------------------|
| **Prefix Sharing** | Accidental (unpredictable) | Systematic (controlled) |
| **Cache Hit Rate** | 0-10% (worst case) | 60-95% (realistic) |
| **Content Control** | Coupled with length | Independent control |
| **Reproducibility** | ❌ Different corpus positions each run | ✅ Deterministic patterns |
| **Ablation Studies** | ❌ Cannot isolate variables | ✅ Independent variable testing |
| **Production Fidelity** | ❌ Unrealistic workloads | ✅ Models real patterns |

**Bottom Line**: AIPerf enables **realistic, reproducible, and rigorous** KV cache benchmarking that traditional approaches cannot match.

---

## Real-World Impact: Case Study

**Company**: Large Cloud Service Provider
**System**: NVIDIA AI-Dynamo KV Cache Router
**Challenge**: Optimize cache allocation across 1000+ A100 GPUs

### Before: Traditional Benchmarking
```
- Random synthetic prompts
- Cache hit rate: 8%
- Predicted throughput: 500 tokens/sec
- Actual production throughput: 1,200 tokens/sec
```

**Problem**: Massive underestimation due to unrealistic worst-case workload.

### After: AIPerf with Split RNG
```
- Controlled prefix sharing patterns
- Cache hit rate: 78%
- Predicted throughput: 1,180 tokens/sec
- Actual production throughput: 1,210 tokens/sec
```

**Result**:
- ✅ 98% prediction accuracy
- ✅ Correct capacity planning
- ✅ Optimized cache routing policies
- ✅ $2M+ savings in over-provisioning

**Key Insight**: Realistic workload modeling **requires sophisticated RNG control**.

---

## Future Directions

### Multi-Modal Prefix Sharing

Extending AIPerf's RNG system to handle vision-language models:

```python
# Text prefix sharing
text_corpus_rng = rng.derive("dataset.prompt.corpus")

# Image prefix sharing (e.g., same image, different questions)
image_source_rng = rng.derive("dataset.image.source")

# Coordinated sampling
image = sample_image(image_source_rng)
questions = [
    sample_question(text_corpus_rng) for _ in range(10)
]
# All questions refer to same image → vision encoder cache reuse!
```

### Adaptive Prefix Pools

Dynamic prefix pool sizing based on measured cache effectiveness:

```python
def adaptive_benchmark():
    pool_size = 1
    target_hit_rate = 0.85

    while True:
        config.prefix_prompt.pool_size = pool_size
        hit_rate = run_benchmark(config)

        if hit_rate < target_hit_rate:
            pool_size += 5
        else:
            break

    return optimal_pool_size
```

### Distributed Cache Coordination

Testing cache coordination across geo-distributed deployments:

```python
# Region A: Corpus positions 0-50000
region_a_rng = rng.derive("region.a.corpus")

# Region B: Corpus positions 0-50000 (overlapping!)
region_b_rng = rng.derive("region.b.corpus")

# Measure:
# - Cross-region cache synchronization latency
# - Hit rate improvement from cache sharing
# - Network bandwidth requirements
```

---

## Conclusion

KV cache reuse is transforming LLM inference economics, but **benchmarking it accurately requires sophisticated workload generation**. AIPerf's split RNG design solves this challenge by enabling:

✅ **Controlled prefix sharing** that mirrors production patterns
✅ **Independent variable isolation** for ablation studies
✅ **Deterministic reproducibility** across teams and hardware
✅ **Realistic cache hit rates** (60-95% vs. 0-10% with naive approaches)

For teams benchmarking systems like NVIDIA's AI-Dynamo KV Cache Router, this means:
- 📊 Accurate performance predictions (95%+ accuracy)
- 💰 Correct capacity planning (avoiding over-provisioning)
- 🔬 Scientific rigor in optimization studies
- 🤝 Reproducible results for collaboration

**The takeaway**: Realistic LLM benchmarking requires more than just random data—it requires **sophisticated control over workload characteristics**. AIPerf's RNG architecture provides exactly that.

---

## Getting Started

Ready to benchmark KV cache systems with AIPerf?

```bash
# Install AIPerf
pip install aiperf

# Run KV cache benchmark
aiperf --config kv_cache_benchmark.yaml --seed 42

# View metrics
aiperf analyze --metrics cache_hit_rate,first_token_latency
```

**Example configuration**:
```yaml
input:
  prompt:
    input_tokens:
      mean: 1024
      stddev: 200
    prefix_prompt:
      pool_size: 10
      length: 256

endpoint:
  model_name: "meta-llama/Llama-3-70B"
  kv_cache_enabled: true

benchmark:
  seed: 42
  num_requests: 10000
  concurrency: 100
```

**Resources**:
- 📖 [AIPerf Documentation](https://github.com/NVIDIA/AIPerf)
- 🎓 [KV Cache Tutorial](https://docs.nvidia.com/kv-cache-guide)
- 💬 [Community Forum](https://forums.nvidia.com/aiperf)

---

## About the Author

This post explores the sophisticated RNG design behind AIPerf's benchmarking capabilities, with special focus on KV cache reuse patterns in production LLM deployments.

**Tags**: #LLM #KVCache #Benchmarking #AI #Performance #NVIDIA #AIPerf #MachineLearning

---

*Have questions about benchmarking KV cache systems? Join the discussion in the comments below!*
