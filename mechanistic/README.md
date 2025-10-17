# Mechanistic Interpretability for Spatial Reasoning

This directory contains tools and experiments for mechanistic interpretability of vision-language models on spatial reasoning tasks.

## Directory Structure

```
mechanistic/
├── attention/           # Attention analysis
│   ├── extractors.py   # Extract attention from models
│   ├── visualizers.py  # Visualize attention patterns
│   └── analyzers.py    # Analyze attention behavior
│
├── attribution/         # Attribution methods
│   ├── cross_attention.py      # Cross-attention attribution
│   ├── integrated_gradients.py # Integrated gradients
│   └── attention_rollout.py    # Attention rollout
│
├── patching/           # Causal interventions
│   ├── activation_patching.py  # Activation patching
│   ├── path_patching.py        # Path patching
│   └── causal_tracing.py       # Causal tracing
│
├── probing/            # Representation analysis
│   ├── linear_probes.py        # Linear probing
│   └── representation_analysis.py
│
├── utils/              # Shared utilities
│   ├── hooks.py        # PyTorch hooks (READY)
│   ├── visualization.py # Viz utilities (READY)
│   └── metrics.py      # Mechanistic metrics (READY)
│
└── experiments/        # Experiment scripts
    ├── attention_maps.py
    ├── cross_attn_attribution.py
    └── activation_patching.py
```

## Quick Start

### 1. Extract Attention Weights

```python
from mechanistic.utils.hooks import AttentionCache

# Create attention cache
cache = AttentionCache()

# Register hooks on your model
hooks = cache.register_attention_hooks(
    model,
    layer_names=['vision_model.encoder.layers.0.self_attn']
)

# Run forward pass
output = model(images, text)

# Get attention weights
attn_weights = cache.get_attention('vision_model.encoder.layers.0.self_attn')

# Clean up
cache.remove_hooks()
```

### 2. Visualize Attention

```python
from mechanistic.utils.visualization import plot_attention_heatmap, plot_attention_on_image

# Plot attention heatmap
fig, ax = plot_attention_heatmap(
    attn_weights[0],  # First sample
    tokens=['[CLS]', 'left', 'of', '[SEP]'],
    layer_name='Layer 0',
    save_path='plots/mechanistic/attention_maps/layer0.png'
)

# Overlay attention on image
fig, ax = plot_attention_on_image(
    image,
    attn_weights[0, 0, 0, 1:],  # First head, CLS token attention to patches
    num_patches=16,
    save_path='plots/mechanistic/attention_maps/overlay.png'
)
```

### 3. Calculate Attention Metrics

```python
from mechanistic.utils.metrics import attention_entropy, attention_concentration

# Calculate entropy (how focused is attention?)
entropy = attention_entropy(attn_weights)
print(f"Attention entropy: {entropy.mean():.3f}")

# Calculate concentration (top-k sum)
concentration = attention_concentration(attn_weights, k=5)
print(f"Top-5 concentration: {concentration.mean():.3f}")
```

## Experiments

### Available Experiments

1. **Attention Map Generation** (`experiments/attention_maps.py`)
   - Extract and visualize attention patterns across layers
   - Compare attention for correct vs incorrect predictions
   - Identify which image regions models attend to for spatial relations

2. **Cross-Attention Attribution** (`experiments/cross_attn_attribution.py`)
   - Analyze cross-attention from text to image
   - Identify which image regions correspond to spatial words
   - Visualize token-to-patch attributions

3. **Activation Patching** (`experiments/activation_patching.py`)
   - Intervene on activations to test causal effects
   - Identify critical layers/components for spatial reasoning
   - Patch activations to understand failure modes

### Running Experiments

```bash
# Extract attention maps for a dataset
python mechanistic/experiments/attention_maps.py \
    --model-name qwen2vl-vllm \
    --dataset Controlled_Images_B \
    --output-dir plots/mechanistic/attention_maps

# Run cross-attention attribution analysis
python mechanistic/experiments/cross_attn_attribution.py \
    --model-name gemma3 \
    --dataset COCO_QA_two_obj \
    --num-samples 100
```

## Integration with Existing Code

The mechanistic interpretability tools integrate seamlessly with your existing evaluation pipeline:

```python
from model_zoo import get_model
from dataset_zoo import get_dataset
from mechanistic.utils.hooks import AttentionCache
from mechanistic.utils.visualization import plot_attention_heatmap

# Load model as usual
model, image_preprocess = get_model("qwen2vl-vllm", device="cuda")
dataset = get_dataset("Controlled_Images_B", image_preprocess=image_preprocess)

# Add attention hooks
cache = AttentionCache()
cache.register_attention_hooks(model.llm.model)  # Hook into vLLM model

# Run evaluation with attention extraction
for batch in dataloader:
    output = model.get_out_scores_wh_batched(...)

    # Extract and visualize attention
    attn = cache.get_attention('layer.0')
    plot_attention_heatmap(attn[0], save_path=f'attn_{idx}.png')

cache.remove_hooks()
```

## Notebooks

Interactive analysis notebooks are in `notebooks/mechanistic/`:

- `attention_analysis.ipynb` - Attention pattern analysis
- `attribution_analysis.ipynb` - Attribution method comparison
- `patching_results.ipynb` - Activation patching results

## Key Features

### Utilities (`utils/`)

✅ **hooks.py** - PyTorch hooks for capturing activations
- `ActivationCache` - Cache intermediate activations
- `AttentionCache` - Specialized for attention weights
- Auto-detection of attention layers

✅ **visualization.py** - Visualization functions
- `plot_attention_heatmap()` - Attention weight heatmaps
- `plot_attention_on_image()` - Overlay attention on images
- `plot_multi_head_attention()` - Multi-head visualization
- `plot_cross_attention_attribution()` - Cross-attention viz

✅ **metrics.py** - Mechanistic metrics
- `attention_entropy()` - Measure attention focus
- `attention_concentration()` - Top-k attention mass
- `causal_effect_strength()` - Patching effect size
- `attribution_sparsity()` - Attribution concentration

## Planned Features

### Attention Module
- [ ] Attention pattern extractors for different model architectures
- [ ] Attention flow analysis across layers
- [ ] Head importance scoring

### Attribution Module
- [ ] Cross-attention attribution
- [ ] Integrated gradients
- [ ] Attention rollout
- [ ] GradCAM for vision models

### Patching Module
- [ ] Activation patching framework
- [ ] Path patching (multiple components)
- [ ] Causal tracing (à la ROME/MEMIT)
- [ ] Noise patching

### Probing Module
- [ ] Linear probes for spatial concepts
- [ ] Representation similarity analysis (RSA)
- [ ] Diagnostic classifiers

## Examples

### Example 1: Attention Analysis for Spatial Relations

```python
from mechanistic.utils.hooks import AttentionCache
from mechanistic.utils.visualization import plot_attention_heatmap

# Setup
cache = AttentionCache()
cache.register_attention_hooks(model, layer_names=['encoder.layer.5'])

# Run model
output = model(image, text="The cup is to the left of the plate")

# Analyze
attn = cache.get_attention('encoder.layer.5')[0]
plot_attention_heatmap(attn[0], tokens=['cup', 'left', 'of', 'plate'])

cache.remove_hooks()
```

### Example 2: Cross-Attention Attribution

```python
from mechanistic.utils.visualization import plot_cross_attention_attribution

# Extract cross-attention (text -> image)
cross_attn = extract_cross_attention(model, image, text)

# Visualize which image regions each word attends to
plot_cross_attention_attribution(
    image=image,
    text_tokens=['the', 'cup', 'left', 'plate'],
    cross_attention=cross_attn,
    num_patches=16
)
```

### Example 3: Activation Patching

```python
from mechanistic.patching import patch_activations

# Run clean forward pass
clean_output = model(clean_image, text)

# Patch with corrupted activations
patched_output = patch_activations(
    model=model,
    clean_input=(clean_image, text),
    corrupt_input=(corrupt_image, text),
    layer_names=['encoder.layer.5']
)

# Measure causal effect
effect = causal_effect_strength(clean_output, patched_output)
print(f"Causal effect of layer 5: {effect:.3f}")
```

## Contributing

When adding new mechanistic experiments:

1. Add core functionality to the appropriate module (`attention/`, `attribution/`, etc.)
2. Add reusable utilities to `utils/`
3. Create experiment scripts in `experiments/`
4. Document in this README
5. Add example notebooks to `notebooks/mechanistic/`

## References

- [Transformer Circuits Thread](https://transformer-circuits.pub/)
- [Causal Tracing (ROME)](https://arxiv.org/abs/2202.05262)
- [Activation Patching](https://www.neelnanda.io/mechanistic-interpretability/activation-patching)
- [Attention Rollout](https://arxiv.org/abs/2005.00928)
