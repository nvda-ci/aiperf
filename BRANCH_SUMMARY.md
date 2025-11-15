<!--
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# Branch Summary: ajc/refactor-loaders

## 🎯 Complete Architectural Transformation

This branch contains a complete redesign of the dataset loading architecture, executed in three major phases with comprehensive cleanup.

---

## Phase 1: Unified Loader Architecture ✅

### Goal
Eliminate the confusing "Composer" abstraction layer and unify all dataset loading under a single "Loader" pattern.

### What Was Done
- **Created new base classes**: `BaseDatasetLoader`, `BaseSyntheticLoader`, `BaseFileLoader`
- **Migrated 7 loaders** to new architecture
- **Eliminated entire Composer layer**: Deleted composer/, public_dataset/ directories
- **Single factory**: `DatasetLoaderFactory` for all loaders
- **Smart auto-detection**: DatasetManager automatically infers dataset types

### Key Files
- `src/aiperf/dataset/loader/base.py` - Root loader base class
- `src/aiperf/dataset/loader/synthetic/` - Synthetic loaders
- `src/aiperf/dataset/loader/file/` - File loaders
- `src/aiperf/dataset/dataset_manager.py` - Updated to use DatasetLoaderFactory

### Documentation
- `REFACTOR_SUMMARY.md`
- `CLEANUP_SUMMARY.md`

---

## Phase 2: Public Dataset Decoupling ✅

### Goal
Decouple public dataset metadata from parsing logic so multiple datasets can share the same loader.

### What Was Done
- **Separated concerns**: Metadata ≠ Parsing ≠ Download
- **Created download utility**: `download_public_dataset()` - reusable for any dataset
- **Refactored ShareGPTLoader**: Pure parsing logic only (extends BaseFileLoader)
- **Removed BaseRemoteDatasetLoader**: No longer needed

### Key Innovation
```python
# Dataset = metadata
ShareGPTDataset.name = "ShareGPT"
ShareGPTDataset.url = "https://..."
ShareGPTDataset.loader_type = DatasetLoaderType.SHAREGPT  # References loader!

# Loader = parsing logic (completely separate!)
class ShareGPTLoader(BaseFileLoader):
    def parse_and_validate(self): ...
    def convert_to_conversations(self): ...
```

### Documentation
- `PUBLIC_DATASET_DECOUPLING.md`

---

## Phase 3: Ultra-Simplified Public Datasets ✅

### Goal
Make public datasets as simple as possible - just data, no classes, no inheritance.

### What Was Done
- **Created simple dataclass**: `PublicDataset` with 4 fields
- **Hard-coded instances**: `SHAREGPT = PublicDataset(...)`
- **Instance registration**: `PublicDatasetFactory.register_instance()`
- **Removed all hierarchy**: No BasePublicDataset, no class per dataset

### The Result
```python
# That's it - just a frozen dataclass!
@dataclass(frozen=True)
class PublicDataset:
    name: str
    url: str
    remote_filename: str
    loader_type: DatasetLoaderType

# Hard-coded instances
SHAREGPT = PublicDataset(
    name="ShareGPT",
    url="https://huggingface.co/...",
    remote_filename="ShareGPT_V3_unfiltered_cleaned_split.json",
    loader_type=DatasetLoaderType.SHAREGPT,
)
```

**Adding new datasets**: Just 5 lines! No new classes needed!

---

## Phase 4: Backward Compatibility Cleanup ✅

### Goal
Remove all backward compatibility shims for a completely clean architecture.

### What Was Removed
1. **CustomDatasetFactory** - entire factory class
2. **CustomDatasetLoaderProtocol** - obsolete protocol
3. **Dual registration** - removed from all 4 file loaders
4. **custom_loader_type** - removed from PublicDataset
5. **BasePublicDataset** - removed class hierarchy
6. **ShareGPTDataset** class - replaced with SHAREGPT instance

### Result
- **ONE factory for loaders**: DatasetLoaderFactory
- **ONE registration per loader**: No duplication
- **Pure metadata**: No backward compat fields

---

## Final Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    DatasetManager                       │
│                          ↓                              │
│  ┌──────────────────────┴──────────────────────┐       │
│  ↓                                             ↓       │
│  DatasetLoaderFactory                PublicDatasetFactory│
│  (parsing logic)                     (metadata instances)│
│  ├─→ SyntheticMultiModalLoader       └─→ SHAREGPT      │
│  ├─→ SyntheticRankingsLoader             ├─ name       │
│  ├─→ SingleTurnLoader                    ├─ url        │
│  ├─→ MultiTurnLoader                     ├─ remote_filename│
│  ├─→ RandomPoolLoader                    └─ loader_type │
│  ├─→ MooncakeTraceLoader                     ↓         │
│  └─→ ShareGPTLoader ←────────────────────────┘         │
│       (SHAREGPT.loader_type points here!)              │
│                                                         │
│  Utility: download_public_dataset(dataset_instance)    │
└─────────────────────────────────────────────────────────┘
```

---

## File Structure

```
src/aiperf/dataset/
├── dataset_manager.py          # Uses DatasetLoaderFactory + PublicDatasetFactory
├── loader/
│   ├── base.py                 # BaseDatasetLoader (root)
│   ├── file/
│   │   ├── base.py             # BaseFileLoader
│   │   └── sharegpt.py         # ShareGPTLoader (pure parsing)
│   ├── synthetic/
│   │   ├── base.py             # BaseSyntheticLoader
│   │   ├── multimodal.py       # SyntheticMultiModalLoader
│   │   └── rankings.py         # SyntheticRankingsLoader
│   ├── single_turn.py
│   ├── multi_turn.py
│   ├── random_pool.py
│   └── mooncake_trace.py
└── public_datasets/
    ├── datasets.py             # PublicDataset dataclass + SHAREGPT instance
    └── downloader.py           # download_public_dataset() utility
```

---

## Testing Status

| Test Suite | Status |
|------------|--------|
| DatasetManager Integration | ✅ 4/4 PASSING |
| Import Verification | ✅ PASS |
| Factory Registration | ✅ 7 loaders registered |
| Public Dataset Registration | ✅ SHAREGPT registered |
| Linting (E,F,W) | ✅ PASS |
| Code Formatting | ✅ PASS |

---

## Code Quality Improvements

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Abstraction Layers | 3 | 2 | -33% |
| Factories for Loaders | 2 | 1 | -50% |
| Public Dataset LOC | ~100+ | ~50 | -50% |
| Classes for Metadata | Many | 1 dataclass | ~90% reduction |
| Files Deleted | - | ~20+ | Significant cleanup |

---

## Key Benefits

### 🎯 Simplicity
- Eliminated unnecessary abstraction layers
- Reduced from class hierarchies to simple dataclasses
- ONE factory for loaders, ONE pattern for datasets

### 🔄 Reusability
- Multiple datasets can share the same loader
- Download utility works for ANY dataset
- Loaders are pure parsing logic

### 📈 Extensibility
- Adding new dataset: 5 lines of code
- Adding new loader: Follow BaseFileLoader pattern
- No boilerplate required

### 🧪 Testability
- Each component isolated and testable
- Integration tests verify end-to-end
- Clean separation makes mocking easy

### 🧹 Maintainability
- Clear responsibilities
- No hidden coupling
- Easy to understand and modify

---

## Migration Notes

### For Users
- Public datasets work the same from CLI perspective
- Config parameters unchanged (CustomDatasetType still supported for user configs)
- Internal architecture is completely new

### For Developers
- Use `DatasetLoaderFactory` (not CustomDatasetFactory)
- Add datasets in `public_datasets/datasets.py`
- Loaders extend `BaseFileLoader` or `BaseSyntheticLoader`

---

## Documentation

Three comprehensive documents created:
1. **REFACTOR_SUMMARY.md** - Original refactor details (Phases 1-9)
2. **PUBLIC_DATASET_DECOUPLING.md** - Decoupling architecture
3. **CLEANUP_SUMMARY.md** - Post-refactor cleanup
4. **BRANCH_SUMMARY.md** (this file) - Complete overview

---

## Conclusion

This branch represents a **complete architectural transformation** of the dataset loading system:

✅ **Unified** - Single loader abstraction
✅ **Decoupled** - Metadata separate from logic
✅ **Simplified** - Dataclass instances, not class hierarchies
✅ **Clean** - No backward compatibility cruft
✅ **Tested** - All integration tests passing
✅ **Production-ready** - Linted, formatted, documented

**The codebase is now dramatically simpler, more maintainable, and easier to extend.** 🚀
