<!--
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# RNG Split Design Audit Report

**Date**: 2025-01-XX
**Auditor**: Claude (AI Assistant)
**Scope**: All `rng.derive()` usage in AIPerf codebase

## Executive Summary

Based on simulation best practices research (Common Random Numbers variance reduction technique), this audit evaluates whether each RNG split in the codebase is appropriate or represents unintended coupling.

**Key Finding**: Most RNG splits in AIPerf are **CORRECT**, with one notable exception requiring remediation.

---

## Audit Criteria

Based on research findings:

### ✅ **Valid Reasons to Split RNGs (Independent Concerns)**

Split RNGs when operations represent **independent aspects of the same entity**:
- Different physical properties (width vs height, duration vs format)
- Different phases of generation (selection vs sampling)
- **Semantic independence**: Changes to one should not affect randomness of another

### ❌ **Invalid Reasons to Split (Common Random Numbers Misuse)**

Do NOT split RNGs to create Common Random Numbers (CRN) when:
- Comparing fundamentally different configurations
- Risk of inducing negative correlation
- Creates spurious coupling between experiments
- No clear counterfactual analysis benefit

---

## Audit Results by Component

### 1. PromptGenerator ✅ **CORRECT**

**Location**: `src/aiperf/dataset/generator/prompt.py:41-43`

```python
self._length_rng = rng.derive("dataset.prompt.length")  # How many tokens
self._corpus_rng = rng.derive("dataset.prompt.corpus")  # Which corpus position
self._prefix_rng = rng.derive("dataset.prompt.prefix")  # Which prefix
```

**Analysis**: ✅ **Appropriate split**

**Reasoning**:
- **Length**: Determines token count (statistical property)
- **Corpus**: Determines sampling position in text (content selection)
- **Prefix**: Determines which prefix to use (independent selection)

These are **semantically independent concerns**:
- Changing token length shouldn't affect which corpus position is selected
- Prefix selection is orthogonal to both
- Each represents a distinct design parameter

**Research Support**: This follows the principle of "assigning independent random sequences to each functionally related group" (CRN literature).

**Verdict**: ✅ **KEEP AS-IS**

---

### 2. ImageGenerator ✅ **CORRECT**

**Location**: `src/aiperf/dataset/generator/image.py:29-31`

```python
self._dimensions_rng = rng.derive("dataset.image.dimensions")  # Width & height
self._format_rng = rng.derive("dataset.image.format")         # PNG vs JPEG
self._source_rng = rng.derive("dataset.image.source")         # Which source image
```

**Analysis**: ✅ **Appropriate split**

**Reasoning**:
- **Dimensions**: Physical properties (width, height)
- **Format**: Encoding choice (JPEG, PNG)
- **Source**: Content selection (which base image)

These represent **independent design spaces**:
- Changing image format shouldn't affect which dimensions are sampled
- Source image selection is orthogonal to both
- Enables adding new concerns (e.g., compression quality) without affecting existing randomness

**Verdict**: ✅ **KEEP AS-IS**

---

### 3. AudioGenerator ✅ **CORRECT**

**Location**: `src/aiperf/dataset/generator/audio.py:53-55`

```python
self._duration_rng = rng.derive("dataset.audio.duration")  # How long
self._format_rng = rng.derive("dataset.audio.format")     # WAV vs MP3
self._data_rng = rng.derive("dataset.audio.data")         # Audio content
```

**Analysis**: ✅ **Appropriate split**

**Reasoning**: Same logic as ImageGenerator - independent concerns for duration, encoding, and content.

**Verdict**: ✅ **KEEP AS-IS**

---

### 4. VideoGenerator ✅ **CORRECT**

**Location**: `src/aiperf/dataset/generator/video.py:33-35`

```python
self._dimensions_rng = rng.derive("dataset.video.dimensions")
self._format_rng = rng.derive("dataset.video.format")
self._data_rng = rng.derive("dataset.video.data")
```

**Analysis**: ✅ **Appropriate split**

**Reasoning**: Same pattern as Image/AudioGenerator.

**Verdict**: ✅ **KEEP AS-IS**

---

### 5. BaseDatasetComposer ✅ **CORRECT**

**Location**: `src/aiperf/dataset/composer/base.py:34-35`

```python
self._model_selector_rng = rng.derive("composer.turn.model_selection")
self._max_tokens_rng = rng.derive("composer.turn.max_tokens")
```

**Analysis**: ✅ **Appropriate split**

**Reasoning**:
- **Model selection**: Which model to use for a turn (when strategy is RANDOM)
- **Max tokens**: Token limit sampling for output

These are **independent turn properties**:
- Choosing a model shouldn't affect max_tokens randomness
- Adding new turn properties (e.g., temperature sampling) won't affect existing ones

**Verdict**: ✅ **KEEP AS-IS**

---

### 6. SyntheticDatasetComposer ✅ **CORRECT**

**Location**: `src/aiperf/dataset/composer/synthetic.py:20-21`

```python
self._turn_sampler_rng = rng.derive("composer.conversation.turn_count")
self._delay_sampler_rng = rng.derive("composer.conversation.turn_delay")
```

**Analysis**: ✅ **Appropriate split**

**Reasoning**:
- **Turn count**: How many turns in a conversation
- **Turn delay**: Time delay between turns

These are **independent conversation properties**:
- Number of turns shouldn't affect timing between them
- Clean separation of structural vs temporal concerns

**Verdict**: ✅ **KEEP AS-IS**

---

### 7. SequenceLengthDistribution ❌ **INCORRECT**

**Location**: `src/aiperf/common/models/sequence_distribution.py:119-121`

```python
self._pair_rng = rng.derive("models.sequence.distribution.pair")  # Which ISL/OSL pair
self._isl_rng = rng.derive("models.sequence.distribution.isl")    # ISL variation
self._osl_rng = rng.derive("models.sequence.distribution.osl")    # OSL variation
```

**Analysis**: ❌ **INCORRECT - Misuse of Common Random Numbers**

**Reasoning**:

#### Problem Description
This split creates CRN-style correlation between **fundamentally different workload configurations**.

When user changes from `256,128:60;512,256:40` to `256,128:40;512,256:60`:
- Different distributions represent **different experiments**, not counterfactuals
- Same ISL/OSL random stream applied to potentially different pairs
- Creates **spurious correlation** without scientific justification

#### Why This Violates CRN Principles

From research:

> "CRN should be employed when you need to compare alternative scenarios directly"

**Our case**: Different probability distributions are **NOT** alternative scenarios of the same experiment. They are **different workload definitions**.

**Analogy**:
```
Distribution A (60:40) = "Morning traffic pattern" (60% cars, 40% trucks)
Distribution B (40:60) = "Evening traffic pattern" (40% cars, 60% trucks)
```

These are **separate experiments**, not "the same traffic with different probabilities".

#### Risk of Negative Correlation

From research:

> "If the CRN induces a negative correlation... this technique can actually backfire"

When the 6th sample selects different pairs:
- Exp A: ISL sample #6 applied to 256-pair (short)
- Exp B: ISL sample #6 applied to 512-pair (long)

The same random variation is applied to **semantically different entities**.

#### What Users Actually Expect

When a user changes distribution config, they expect:
- ✅ **Independent reproducibility**: Same seed + same config = same result
- ❌ **NOT shared randomness**: Different configs should be fully independent

**Verdict**: ❌ **REVERT TO SINGLE RNG**

**Recommendation**:
```python
# CORRECT design:
self._rng = rng.derive("models.sequence.distribution")
```

This ensures:
- Each distribution configuration gets independent random behavior
- Simpler mental model
- No spurious correlation
- Follows standard benchmarking practices

---

## Summary Table

| Component | Location | RNGs | Status | Action |
|-----------|----------|------|--------|--------|
| PromptGenerator | prompt.py | 3 split | ✅ Correct | Keep |
| ImageGenerator | image.py | 3 split | ✅ Correct | Keep |
| AudioGenerator | audio.py | 3 split | ✅ Correct | Keep |
| VideoGenerator | video.py | 3 split | ✅ Correct | Keep |
| BaseComposer | base.py | 2 split | ✅ Correct | Keep |
| SyntheticComposer | synthetic.py | 2 split | ✅ Correct | Keep |
| **SequenceLengthDistribution** | **sequence_distribution.py** | **3 split** | **❌ Incorrect** | **Revert to single** |

---

## Design Principles Derived

### When TO Split RNGs ✅

1. **Independent Physical Properties**
   - Width vs height, duration vs format
   - Example: `_dimensions_rng` vs `_format_rng`

2. **Independent Functional Concerns**
   - Content selection vs property sampling
   - Example: `_corpus_rng` (which text) vs `_length_rng` (how much)

3. **Orthogonal Design Parameters**
   - Model selection vs output length
   - Turn count vs turn delay

4. **Future Extensibility**
   - Adding new property shouldn't affect existing randomness
   - Example: Adding `_quality_rng` to ImageGenerator

### When NOT TO Split RNGs ❌

1. **Different Experimental Configurations**
   - Different distributions = different experiments
   - Each should be independently reproducible

2. **Risk of Semantic Confusion**
   - Applying same random sample to different entity types
   - Creates non-obvious correlation

3. **No Clear Counterfactual Benefit**
   - Not asking "what if same entity had different property?"
   - Instead: "what if we ran a different experiment?"

4. **Violates User Expectations**
   - Users expect config changes to be fully independent
   - Not to "share" underlying randomness

---

## Remediation Plan

### Immediate Action Required

1. **Revert SequenceLengthDistribution to single RNG**
   - Change from 3 RNGs back to 1 RNG
   - Update tests with new expected values
   - Document the change

2. **Update Documentation**
   - Add this audit report to repo
   - Update RNG design guidelines
   - Clarify when to split vs when not to

3. **Add Linting/Review Guidelines**
   - Flag any new split RNGs for review
   - Ensure design rationale is documented

### Long-term

- Monitor for new RNG usage patterns
- Consider adding automated checks
- Share learnings in team documentation

---

## References

- **Common Random Numbers**: PMC2761656 - "Keeping the Noise Down: Common Random Numbers for Disease Simulation Modeling"
- **Variance Reduction**: "On the Effectiveness of Common Random Numbers" - Management Science
- **When CRN Backfires**: Variance reduction techniques can increase variance if negative correlation is induced
- **AIPerf Documentation**: `docs/random_number_generation.md`

---

## Conclusion

The AIPerf codebase demonstrates **excellent RNG design** in 6 out of 7 audited components. The splits in generators and composers correctly separate independent concerns and enable clean extensibility.

The one exception (SequenceLengthDistribution) represents a well-intentioned but **misapplied** use of Common Random Numbers technique, which should be reverted to maintain:
- ✅ User expectations of independence
- ✅ Standard benchmarking practices
- ✅ Simpler mental models
- ✅ Avoidance of spurious correlation

**Overall Assessment**: 🟢 **Strong foundation with one actionable fix**
