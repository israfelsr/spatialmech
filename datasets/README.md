# Training Datasets

This directory contains datasets used for training probes and other models.

## Structure

```
datasets/
├── spatial_hf/              # HuggingFace format spatial datasets
├── features/                # Extracted features from vision models
└── README.md               # This file
```

## Dataset Types

### 1. Spatial Reasoning Datasets (HuggingFace Format)

For training spatial relation probes:
```python
from datasets import load_from_disk

# Load spatial reasoning dataset
dataset = load_from_disk("datasets/spatial_hf")

# Access features
for sample in dataset:
    image_path = sample["image_path"]
    spatial_relation = sample["spatial_relation"]
    label = sample["label"]
    features = sample["features"]  # Pre-extracted if available
```

### 2. Feature Datasets

Pre-extracted features from vision models:
```
datasets/features/
├── dinov2/
│   ├── layer_10.pt
│   ├── layer_15.pt
│   └── layer_23.pt
├── clip/
│   └── last_layer.pt
└── qwen/
    └── visual_features.pt
```

## Data Sources

Datasets can come from:
- **HuggingFace Hub**: Download using `datasets` library
- **Local Processing**: Convert raw images to HF format
- **Feature Extraction**: Pre-compute features from vision models

## Usage

### Loading Training Datasets

```python
from datasets import load_from_disk
import torch

# Load HuggingFace dataset
dataset = load_from_disk("datasets/spatial_hf")

# Load pre-extracted features
features = torch.load("datasets/features/dinov2/layer_23.pt")
```

### Creating Training Datasets

```python
from datasets import Dataset

# Create from Python dict
data = {
    "image_path": [...],
    "spatial_relation": [...],
    "label": [...]
}
dataset = Dataset.from_dict(data)

# Save to disk
dataset.save_to_disk("datasets/spatial_hf")
```

## Dataset Format

### HuggingFace Format
```python
Dataset({
    features: [
        'image_path': string,
        'spatial_relation': string (one of: left, right, top, bottom),
        'label': int (0 or 1),
        'captions': list[string],
        'features': optional tensor
    ],
    num_rows: N
})
```

### Feature Tensors
```python
# Shape: (N, D) where N=num_samples, D=feature_dim
{
    'features': torch.Tensor,  # Shape: (N, 768) for DINOv2
    'labels': torch.Tensor,    # Shape: (N,)
    'metadata': dict
}
```

## Adding Training Data

To add new training datasets:

1. **Prepare data** in HuggingFace format
2. **Save to disk**: `dataset.save_to_disk("datasets/your_dataset_hf")`
3. **Document format** in this README
4. **Update configs** in `config/` directory

## Related Directories

- **dataset_zoo/**: Evaluation datasets for VLM testing
- **data/**: Temporary/cache data (gitignored)
- **config/**: Training configurations

See the main README for overall project structure.
