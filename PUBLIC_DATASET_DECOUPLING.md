<!--
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# Public Dataset Decoupling - Clean Architecture

## Overview

Successfully decoupled public datasets from loaders, achieving a clean separation where:
- **Public datasets = pure metadata** (URL, name, which loader to use)
- **Loaders = pure parsing logic** (no dataset-specific metadata)
- **Multiple datasets can share the same loader**

## The Problem (Before)

### Coupled Architecture
```python
@DatasetLoaderFactory.register(DatasetLoaderType.SHAREGPT)
@PublicDatasetFactory.register(PublicDatasetType.SHAREGPT)
class ShareGPTLoader(BaseRemoteDatasetLoader):
    # MIXED: Metadata + Parsing + Download logic all in one!
    tag = "ShareGPT"
    url = "https://huggingface.co/..."
    remote_filename = "ShareGPT_V3_unfiltered_cleaned_split.json"

    def __init__(self, config, tokenizer, filename=None):
        # Complex logic to handle both remote and local loading
        if filename is not None:
            BaseFileLoader.__init__(...)  # Local loading
        else:
            super().__init__(...)  # Remote downloading

    def download(self): ...  # Download logic
    def parse_and_validate(self): ...  # Parsing logic
```

**Problems:**
- ❌ Loader contains dataset metadata (URL, name)
- ❌ Can't reuse loader for different datasets with same format
- ❌ Complex dual-initialization logic
- ❌ Mixed concerns: download + parse in same class

## The Solution (After)

### Decoupled Architecture

#### 1. Public Dataset = Pure Metadata
```python
@PublicDatasetFactory.register(PublicDatasetType.SHAREGPT)
class ShareGPTDataset(BasePublicDataset):
    """Pure metadata - no logic!"""
    name = "ShareGPT"
    url = "https://huggingface.co/datasets/anon8231489123/..."
    remote_filename = "ShareGPT_V3_unfiltered_cleaned_split.json"
    loader_type = DatasetLoaderType.SHAREGPT  # Which loader to use!
    custom_loader_type = CustomDatasetType.SHAREGPT  # Backward compat
```

#### 2. Loader = Pure Parsing Logic
```python
@DatasetLoaderFactory.register(DatasetLoaderType.SHAREGPT)
class ShareGPTLoader(BaseFileLoader):
    """Pure parsing logic - no metadata!"""

    def __init__(self, config, tokenizer, filename):
        # Simple: just initialize file loader
        super().__init__(config, tokenizer, filename)

    def can_load_file(self, path): ...  # Format detection
    def parse_and_validate(self): ...  # Parsing
    def convert_to_conversations(self): ...  # Conversion
```

#### 3. Download Utility = Reusable Function
```python
def download_public_dataset(dataset: PublicDatasetProtocol) -> Path:
    """Reusable download utility for ANY public dataset."""
    cache_path = AIPERF_DATASET_CACHE_DIR / dataset.remote_filename

    if cache_path.exists():
        return cache_path  # Use cache

    # Download from dataset.url
    # Save to cache
    return cache_path
```

#### 4. DatasetManager = Orchestrator
```python
def _load_public_dataset(self):
    # 1. Get dataset metadata
    dataset_class = PublicDatasetFactory.get_class_from_type(public_dataset_type)

    # 2. Download using utility
    file_path = download_public_dataset(dataset_class)

    # 3. Get loader specified by dataset
    loader = DatasetLoaderFactory.create_instance(
        dataset_class.loader_type,  # Dataset tells us which loader!
        filename=file_path,
    )

    # 4. Load conversations
    return loader.load()
```

## Benefits

### ✅ Clean Separation of Concerns
- **Metadata** (datasets) separate from **logic** (loaders)
- **Download** (utility) separate from **parsing** (loader)

### ✅ Reusability
Multiple datasets can use the same loader:
```python
@PublicDatasetFactory.register(PublicDatasetType.SHAREGPT)
class ShareGPTDataset(BasePublicDataset):
    loader_type = DatasetLoaderType.SHAREGPT  # Uses ShareGPTLoader

@PublicDatasetFactory.register(PublicDatasetType.SHAREGPT_V4)
class ShareGPTV4Dataset(BasePublicDataset):
    name = "ShareGPT V4"
    url = "https://different-source.com/sharegpt_v4.json"
    loader_type = DatasetLoaderType.SHAREGPT  # SAME loader!
```

### ✅ Extensibility
Adding a new dataset is trivial:
```python
# Just add metadata - no new loader needed if format matches!
@PublicDatasetFactory.register(PublicDatasetType.MY_DATASET)
class MyDataset(BasePublicDataset):
    name = "My Dataset"
    url = "https://my-source.com/data.json"
    loader_type = DatasetLoaderType.SHAREGPT  # Reuse existing loader!
```

### ✅ Testability
- Test loaders in isolation (pure functions)
- Test datasets without download logic
- Mock download utility easily

## Implementation Details

### File Structure
```
src/aiperf/dataset/
├── public_datasets/           # NEW: Pure metadata
│   ├── __init__.py
│   ├── base.py               # BasePublicDataset
│   ├── downloader.py         # download_public_dataset() utility
│   └── sharegpt.py           # ShareGPTDataset (metadata only)
│
└── loader/
    ├── file/
    │   ├── base.py           # BaseFileLoader
    │   └── sharegpt.py       # ShareGPTLoader (parsing only)
    └── synthetic/
        └── ...
```

### Removed Files
- ❌ `loader/file/remote_base.py` - BaseRemoteDatasetLoader (no longer needed)

### Updated Files
- ✅ `protocols.py` - Updated PublicDatasetProtocol (pure metadata)
- ✅ `loader/file/sharegpt.py` - Removed metadata, simplified __init__
- ✅ `dataset_manager.py` - Uses new decoupled pattern
- ✅ All `__init__.py` files - Removed BaseRemoteDatasetLoader exports

## Testing

### Architecture Verification ✅
```
✓ ShareGPTDataset is pure metadata (no download method)
✓ ShareGPTLoader extends BaseFileLoader (not BaseRemoteDatasetLoader)
✓ ShareGPTLoader only registered with DatasetLoaderFactory
✓ ShareGPTDataset only registered with PublicDatasetFactory
✓ Different classes (properly decoupled)
```

### Integration Tests ✅
```
✓ All DatasetManager tests passing (4/4)
✓ Imports work correctly
✓ Formatting and linting pass
```

## Example: Adding a New Public Dataset

```python
# Step 1: Create metadata class
@PublicDatasetFactory.register(PublicDatasetType.ALPACA)
class AlpacaDataset(BasePublicDataset):
    name = "Alpaca"
    url = "https://github.com/tatsu-lab/stanford_alpaca/raw/main/alpaca_data.json"
    remote_filename = "alpaca.json"
    loader_type = DatasetLoaderType.SHAREGPT  # Reuse ShareGPT loader!
    custom_loader_type = CustomDatasetType.SHAREGPT

# Step 2: Import in DatasetManager to trigger registration
from aiperf.dataset.public_datasets import AlpacaDataset  # noqa: F401

# That's it! No new loader needed if format matches!
```

## Comparison

| Aspect | Before (Coupled) | After (Decoupled) |
|--------|------------------|-------------------|
| **Loader class** | Mixed metadata + logic | Pure parsing logic |
| **Dataset class** | N/A (coupled) | Pure metadata |
| **Download logic** | In loader __init__ | Separate utility function |
| **Reusability** | One dataset per loader | Many datasets per loader |
| **Extensibility** | New loader for each dataset | Just add metadata! |
| **Testing** | Complex (mocked download) | Simple (isolated units) |

## Success Metrics

- ✅ **Clean separation**: Metadata ≠ Logic
- ✅ **Reusability**: One loader, many datasets
- ✅ **Extensibility**: Add datasets without new loaders
- ✅ **Simplicity**: Each class has one responsibility
- ✅ **Testability**: Easy to test in isolation
- ✅ **All tests passing**: Integration verified

## Conclusion

The public dataset architecture is now **cleanly decoupled** with proper separation of concerns:
- **PublicDatasetFactory** → Manages dataset metadata
- **DatasetLoaderFactory** → Manages parsing logic
- **download_public_dataset()** → Reusable download utility

This makes it trivial to add new public datasets and promotes code reuse! 🚀
