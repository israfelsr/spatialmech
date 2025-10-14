# SpatialMech

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd spatialmech

# Create conda environment
conda create -n spatialmech python=3.11
conda activate spatialmech

# Install dependencies
pip install -r requirements.txt
```

### Set Environment Variables

```bash
# Set path to spatial reasoning evaluation data
export SPATIAL_DATA_DIR="/leonardo_work/EUHPC_D27_102/compmech/whatsup_vlms_data"
```
### Evaluate a VLM

```bash
python scripts/evaluate.py \
    --model-name qwen2vl-vllm \
    --dataset Controlled_Images_A \
    --device cuda
```

## 📚 Key Components

### Model Zoo
Pre-trained VLMs for spatial reasoning evaluation:
- Qwen2-VL (standard & vLLM)
- LLaVA 1.5/1.6
- PaliGemma

```python
from model_zoo import get_model
model, preprocess = get_model("qwen2vl-vllm", device="cuda")
```

### Dataset Zoo
Spatial reasoning benchmarks:
- Controlled Images (A/B)
- COCO-QA (one/two objects)
- Visual Genome QA (one/two objects)
- VSR

```python
from dataset_zoo import get_dataset
dataset = get_dataset("Controlled_Images_A", root_dir="/path/to/data")
```

### Spatial Probes
Linear classifiers for understanding vision model representations:
```python
from src.models import SpatialProbe

probe = SpatialProbe(input_dim=768, num_relations=4)
```

## 📊 Workflows

### 1. Feature Extraction → Probe Training

```bash
# Extract features from a vision model
python scripts/extract_features.py \
    --model dinov2 \
    --dataset datasets/spatial_hf \
    --output datasets/features/dinov2

# Train spatial probes
python scripts/train_probes.py \
    --config config/dinov2.yaml \
    --features datasets/features/dinov2

# Visualize results
python scripts/plot_results.py \
    --results results/dinov2/ \
    --output plots/dinov2_probes.png
```

### 2. VLM Evaluation

```bash
# Evaluate on multiple datasets
for dataset in Controlled_Images_A COCO_QA_one_obj VG_QA_one_obj; do
    python scripts/evaluate.py \
        --model-name qwen2vl-vllm \
        --dataset $dataset \
        --output results/qwen/
done

# Compare models
python scripts/compare_models.py \
    --results results/ \
    --models qwen2vl llava1.5 paligemma
```

### 3. Analysis

```bash
# Launch Jupyter for analysis
jupyter notebook notebooks/

# Or run specific analysis
python scripts/analyze_layers.py --results results/dinov2/
```

## 🛠️ Available Scripts

| Script | Purpose |
|--------|---------|
| `train_probes.py` | Train spatial reasoning probes |
| `extract_features.py` | Extract features from vision models |
| `evaluate.py` | Evaluate VLMs on spatial reasoning |
| `compare_models.py` | Compare multiple models |
| `plot_results.py` | Visualize results |

Run with `--help` for detailed options.

## ⚙️ Configuration

Configurations are in `config/*.yaml`:

```yaml
# config/dinov2.yaml
model:
  name: "dinov2"
  checkpoint: "facebook/dinov2-large"
  layers: [10, 15, 20, 23]

dataset:
  name: "spatial_relations"
  path: "datasets/spatial_hf"
  relations: ["left", "right", "top", "bottom"]

training:
  batch_size: 32
  learning_rate: 0.001
  epochs: 100
```

## 🔍 Key Concepts

### Evaluation vs Training

**Evaluation (model_zoo + dataset_zoo)**:
- 🎯 Test full VLMs on benchmarks
- 📊 Image-text pairs
- 🔍 Accuracy metrics

**Training (src + datasets)**:
- 🎯 Train lightweight probes
- 📊 Feature vectors + labels
- 🔍 Trained probe models

### Probes vs VLMs

**Probes**:
- Linear classifiers on frozen features
- Fast to train (~minutes)
- Show what info is encoded

**VLMs**:
- End-to-end models
- Generate natural language
- Evaluate complex reasoning

## 📦 Dependencies

Key packages:
- PyTorch 2.0+
- Transformers
- vLLM (for fast inference)
- Datasets (HuggingFace)
- PIL/Pillow
- NumPy, Pandas
- Matplotlib, Seaborn

See `requirements.txt` for complete list.

## 📄 Citation

If you use this code, please cite:

## 📧 Contact

For questions or issues, please open a GitHub issue.

## 📜 License

[Specify your license]

---

**Quick Links**:
- 📖 [Project Structure](PROJECT_STRUCTURE.md)
- 🤖 [Model Zoo](model_zoo/README.md)
- 📊 [Dataset Zoo](dataset_zoo/README.md)
- 🗂️ [Training Datasets](datasets/README.md)
