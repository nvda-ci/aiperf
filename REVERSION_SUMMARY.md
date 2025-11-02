<!--
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# SequenceLengthDistribution RNG Reversion Summary

**Date**: 2025-01-XX
**Status**: ✅ **COMPLETED**

---

## What Was Done

Successfully reverted `SequenceLengthDistribution` from a **3-RNG split design** back to a **single RNG design**, based on comprehensive research into Common Random Numbers (CRN) variance reduction techniques.

---

## Changes Made

### 1. Source Code Changes

**File**: `src/aiperf/common/models/sequence_distribution.py`

**Before (Split RNG)**:
```python
def __init__(self, pairs: list[SequenceLengthPair]) -> None:
    self._pair_rng = rng.derive("models.sequence.distribution.pair")
    self._isl_rng = rng.derive("models.sequence.distribution.isl")
    self._osl_rng = rng.derive("models.sequence.distribution.osl")
```

**After (Single RNG)**:
```python
def __init__(self, pairs: list[SequenceLengthPair]) -> None:
    self._rng = rng.derive("models.sequence.distribution")
```

**Methods Updated**:
- `__init__()` - Changed from 3 RNGs to 1 RNG
- `sample()` - Updated to use `self._rng` instead of split RNGs
- `sample_batch()` - Updated to use `self._rng` instead of split RNGs

### 2. Test Updates

**File**: `tests/test_sequence_distribution.py`

Updated 4 tests with new expected values due to different RNG sequence:

1. **test_multi_pair_distribution_sampling**
   - First 10 samples changed
   - Count distributions changed (6113/3887 vs. 5965/4035)

2. **test_stddev_distribution_sampling**
   - First 10 samples changed
   - Statistical values updated

3. **test_end_to_end_workflow**
   - Count distributions changed (3014/4957/2029 vs. 2974/5048/1978)

4. **test_end_to_end_workflow_with_stddev**
   - Statistical values updated

---

## Test Results

### ✅ All Tests Pass

```bash
tests/test_sequence_distribution.py::51 tests - 51 PASSED ✅
tests/composers/test_base_composer.py::12 tests - 12 PASSED ✅
tests/config/test_prompt_config.py::13 tests - 13 PASSED ✅

Total: 76/76 tests passing ✅
```

---

## Why This Change Was Made

### Research Findings on Common Random Numbers (CRN)

Based on extensive research into CRN variance reduction techniques:

**CRN is appropriate when**:
> "Comparing alternative scenarios directly where you need to reduce stochastic noise between model runs"

**Example**: Disease simulation where individuals serve as their own controls across different treatment policies.

**CRN is NOT appropriate when**:
> "Comparing fundamentally different experimental configurations"

**Example**: Different workload distributions are **different experiments**, not counterfactuals.

### Why SequenceLengthDistribution Split Was Wrong

1. **Different Distributions = Different Experiments**
   - `256,128:60;512,256:40` vs `256,128:40;512,256:60`
   - These represent different workload definitions, not alternative scenarios

2. **Spurious Correlation**
   - Split RNG created shared randomness across independent experiments
   - Violated principle that different configs should be fully independent

3. **User Expectation Violation**
   - Users expect: Same seed + same config = same result ✅
   - Users don't expect: Different configs sharing underlying randomness ❌

4. **Risk of Negative Correlation**
   - Research shows: "If CRN induces negative correlation, variance increases (backfires)"
   - Applying same ISL/OSL samples to potentially different pairs is semantically incorrect

### Why PromptGenerator Split IS Correct

In contrast, `PromptGenerator`'s split RNG design **remains correct** because:

1. **Within-Experiment Splitting**
   - Splits different **aspects of prompt generation** (length, content, prefix)
   - NOT splitting across different experimental configurations

2. **Enables Prefix Sharing**
   - Same corpus position + different lengths = shared prefixes
   - Critical for realistic KV cache benchmarking
   - Models production patterns (document + varying query lengths)

3. **Scientific Control**
   - Enables ablation studies: "How does length alone affect latency?"
   - Independent variables for sensitivity analysis

**Key Distinction**:
- PromptGenerator: Splits **generation concerns** within experiment ✅
- SequenceLengthDistribution: Was splitting **experimental configs** across experiments ❌

---

## Design Principle Clarification

### When TO Split RNGs ✅

**Within a single component/experiment**:
- Independent physical properties (width vs height)
- Independent functional concerns (content vs length)
- Enables ablation studies and scientific control
- Example: PromptGenerator, ImageGenerator, AudioGenerator

### When NOT TO Split RNGs ❌

**Across different experimental configurations**:
- Different workload distributions
- Different system configs being compared
- Creates spurious correlation
- Example: SequenceLengthDistribution (now fixed)

---

## Impact Assessment

### ✅ Benefits of Reversion

1. **Clearer Semantics**
   - Each distribution config gets independent random behavior
   - No hidden coupling between experiments

2. **User Expectation Alignment**
   - Changing config = completely independent experiment
   - Follows standard benchmarking practices

3. **Simpler Mental Model**
   - One config = one RNG sequence
   - Easier to understand and debug

4. **Eliminates Risk**
   - No risk of negative correlation backfire
   - No spurious correlation artifacts

### ⚠️ What We Lose

**Nothing significant**:
- The split RNG wasn't providing useful variance reduction
- It was creating unintended correlation, not useful control
- Single RNG maintains full reproducibility (same seed = same results)

---

## Related Documentation

### Files Created During This Work

1. **RNG_AUDIT_REPORT.md**
   - Comprehensive audit of all RNG usage in codebase
   - Analysis of 7 components (6 correct, 1 needed fix)
   - Design principles for when to split vs. not split

2. **PROMPTGENERATOR_RNG_ANALYSIS.md**
   - Deep dive into KV cache reuse and prefix sharing
   - Validates PromptGenerator's split RNG design
   - Explains why same corpus position creates prefix sharing

3. **blog_kv_cache_benchmarking.md**
   - Professional blog post on KV cache benchmarking
   - Showcases AIPerf's sophisticated RNG design
   - Real-world case studies and implementation guidance

4. **REVERSION_SUMMARY.md** (this file)
   - Summary of reversion changes
   - Test results and validation
   - Impact assessment

---

## Commits

This work should be committed with the following message:

```
fix: revert SequenceLengthDistribution to single RNG design

Based on research into Common Random Numbers (CRN) variance reduction
techniques, the split RNG design in SequenceLengthDistribution was
creating spurious correlation between independent experimental
configurations.

Changes:
- Reverted from 3 RNGs (pair, isl, osl) to single RNG
- Updated test expectations for new random sequence
- All 76 tests passing

Rationale:
- Different distribution configs are different experiments, not
  counterfactuals of the same experiment
- Split RNG was misapplying CRN technique
- Single RNG maintains reproducibility while eliminating unintended
  coupling

Research references:
- "Common Random Numbers for Disease Simulation Modeling" (PMC2761656)
- "On the Effectiveness of Common Random Numbers" (Management Science)
- "Variance Reduction Techniques" (multiple sources)

See RNG_AUDIT_REPORT.md for full analysis.
```

---

## Final Status

✅ **Reversion Complete and Validated**

- Source code updated ✅
- Tests updated ✅
- All tests passing (76/76) ✅
- Documentation complete ✅
- Ready for commit ✅

**Summary**: Successfully corrected a misapplication of Common Random Numbers technique while validating that all other RNG splits in the codebase are correct and well-designed.
