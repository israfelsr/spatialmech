"""
Qwen3-VL Attention Extraction for Spatial Reasoning

This script extracts cross-attention from Qwen3-VL-4B-Instruct to understand
how the model attends to image regions when reasoning about spatial relations.

Usage:
    python mechanistic/experiments/qwen3vl_attention_extraction.py \
        --model-path /path/to/Qwen3-VL-4B-Instruct \
        --data-dir /path/to/whatsup_vlms_data \
        --sample-idx 0 \
        --output-dir plots/mechanistic/qwen3vl_attention \
        --num-samples 10
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import zoom
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from dataset_zoo.aro_datasets import get_controlled_images_b
from mechanistic.utils.visualization import (
    plot_attention_heatmap,
    plot_attention_on_image,
)
from mechanistic.utils.metrics import attention_entropy, attention_concentration


def parse_args():
    parser = argparse.ArgumentParser(description="Extract attention from Qwen3-VL")
    parser.add_argument(
        "--model-path",
        type=str,
        default="/leonardo_work/EUHPC_D27_102/compmech/models/Qwen3-VL-4B-Instruct",
        help="Path to Qwen3-VL model"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/leonardo_work/EUHPC_D27_102/compmech/whatsup_vlms_data",
        help="Root directory for dataset"
    )
    parser.add_argument(
        "--sample-idx",
        type=int,
        default=0,
        help="Sample index to analyze (if not using --num-samples)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to process (if None, only process sample-idx)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots/mechanistic/qwen3vl_attention",
        help="Output directory for visualizations"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu)"
    )
    parser.add_argument(
        "--save-tensors",
        action="store_true",
        help="Save attention tensors to disk"
    )
    return parser.parse_args()


def load_model(model_path, device):
    """Load Qwen3-VL model with eager attention for attention extraction."""
    print(f"Loading Qwen3-VL from: {model_path}")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager"  # Required for output_attentions=True
    )

    processor = AutoProcessor.from_pretrained(model_path)

    print(f"Model loaded successfully on {model.device}")
    print(f"Attention implementation: eager (supports output_attentions)")

    return model, processor


def extract_cross_attention(model, processor, image, question):
    """
    Extract cross-attention from text tokens to image patches.

    Returns:
        cross_attn: [num_text_tokens, num_image_patches] attention weights
        tokens_list: List of all tokens
        text_tokens: List of text tokens
        grid_size: Size of image patch grid
        vision_start_idx: Start index of vision tokens
        vision_end_idx: End index of vision tokens
    """
    # Prepare messages in Qwen format
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]

    # Apply chat template
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Process inputs
    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # Forward pass with attention
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_attentions=True,
            return_dict=True
        )

    # Check if attention was captured
    if not hasattr(outputs, 'attentions') or outputs.attentions is None:
        raise RuntimeError("No attention weights found in outputs!")

    attentions = outputs.attentions
    print(f"Captured attention from {len(attentions)} layers")

    # Decode tokens
    token_ids = inputs['input_ids'][0]
    tokens_list = [processor.tokenizer.decode(t) for t in token_ids]

    # Find vision token boundaries
    vision_start_idx = None
    vision_end_idx = None
    vision_start_token = "<|vision_start|>"
    vision_end_token = "<|vision_end|>"

    for i, token in enumerate(tokens_list):
        if vision_start_token in token:
            vision_start_idx = i
        if vision_end_token in token:
            vision_end_idx = i
            break

    if vision_start_idx is None or vision_end_idx is None:
        raise RuntimeError(f"Could not find vision tokens! Tokens: {tokens_list}")

    print(f"Image tokens span: [{vision_start_idx}, {vision_end_idx}]")
    num_image_tokens = vision_end_idx - vision_start_idx - 1
    grid_size = int(np.sqrt(num_image_tokens))
    print(f"Image grid size: {grid_size}x{grid_size}")

    # Extract cross-attention from last layer
    last_layer_attn = attentions[-1]  # [batch, heads, seq, seq]
    avg_attn = last_layer_attn[0].mean(0)  # [seq, seq]

    # Text tokens are after vision_end_idx
    text_start_idx = vision_end_idx + 1

    # Attention from text to image patches
    cross_attn = avg_attn[text_start_idx:, vision_start_idx+1:vision_end_idx]

    text_tokens = tokens_list[text_start_idx:]

    print(f"Cross-attention shape: {cross_attn.shape}")
    print(f"Format: [num_text_tokens={len(text_tokens)}, num_image_patches={num_image_tokens}]")

    return cross_attn, tokens_list, text_tokens, grid_size, vision_start_idx, vision_end_idx


def visualize_token_attention(image, cross_attn, text_tokens, interesting_indices,
                              interesting_labels, grid_size, caption, save_path):
    """Visualize attention for specific tokens."""
    n_tokens = min(len(interesting_indices), 6)
    fig, axes = plt.subplots(1, n_tokens + 1, figsize=(4 * (n_tokens + 1), 4))

    # Handle single axis case
    if n_tokens == 0:
        return

    if n_tokens == 1:
        axes = [axes] if not isinstance(axes, np.ndarray) else axes

    # Plot original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Plot attention for each token
    for plot_idx, token_idx in enumerate(interesting_indices[:n_tokens]):
        token_attn = cross_attn[token_idx]  # [num_image_patches]

        # Reshape to 2D grid
        attn_grid = token_attn.cpu().reshape(grid_size, grid_size).numpy()

        # Normalize
        attn_grid = (attn_grid - attn_grid.min()) / (attn_grid.max() - attn_grid.min() + 1e-8)

        # Resize to image size
        img_array = np.array(image)
        zoom_factor = (img_array.shape[0] / grid_size, img_array.shape[1] / grid_size)
        attn_resized = zoom(attn_grid, zoom_factor, order=1)

        # Plot
        axes[plot_idx + 1].imshow(img_array)
        axes[plot_idx + 1].imshow(attn_resized, cmap='hot', alpha=0.6)
        axes[plot_idx + 1].set_title(f'Token: {interesting_labels[plot_idx]}')
        axes[plot_idx + 1].axis('off')

    plt.suptitle(f'Cross-Attention: Text Tokens → Image Patches\n{caption}', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved token attention visualization to {save_path}")


def visualize_spatial_attention(image, cross_attn, text_tokens, spatial_words,
                                caption, save_path):
    """Visualize aggregated attention for spatial words."""
    spatial_token_indices = []
    for i, token in enumerate(text_tokens):
        if any(word in token.lower() for word in spatial_words):
            spatial_token_indices.append(i)

    if len(spatial_token_indices) == 0:
        print("No spatial words found in text!")
        return

    # Average attention across all spatial tokens
    spatial_attn = cross_attn[spatial_token_indices].mean(0)
    grid_size = int(np.sqrt(spatial_attn.shape[0]))

    # Reshape and normalize
    attn_grid = spatial_attn.cpu().reshape(grid_size, grid_size).numpy()
    attn_grid = (attn_grid - attn_grid.min()) / (attn_grid.max() - attn_grid.min() + 1e-8)

    # Resize to image size
    img_array = np.array(image)
    zoom_factor = (img_array.shape[0] / grid_size, img_array.shape[1] / grid_size)
    attn_resized = zoom(attn_grid, zoom_factor, order=1)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.imshow(image)
    ax1.set_title('Original Image')
    ax1.axis('off')

    ax2.imshow(img_array)
    im = ax2.imshow(attn_resized, cmap='hot', alpha=0.6)
    ax2.set_title('Averaged Spatial Word Attention')
    ax2.axis('off')

    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    plt.suptitle(f'Where does the model look for spatial reasoning?\n{caption}', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved spatial attention visualization to {save_path}")


def visualize_attention_metrics(cross_attn, save_path):
    """Visualize attention metrics distributions."""
    entropy = attention_entropy(cross_attn, dim=-1)
    concentration = attention_concentration(cross_attn, k=5)

    print(f"Cross-Attention Metrics:")
    print(f"  Average entropy: {entropy.mean():.3f} (lower = more focused)")
    print(f"  Average top-5 concentration: {concentration.mean():.3f} (higher = more concentrated)")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.hist(entropy.cpu().numpy(), bins=30, alpha=0.7, edgecolor='black')
    ax1.axvline(entropy.mean().item(), color='red', linestyle='--', label=f'Mean: {entropy.mean():.3f}')
    ax1.set_xlabel('Attention Entropy')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Attention Entropy\n(per text token)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.hist(concentration.cpu().numpy(), bins=30, alpha=0.7, edgecolor='black', color='orange')
    ax2.axvline(concentration.mean().item(), color='red', linestyle='--', label=f'Mean: {concentration.mean():.3f}')
    ax2.set_xlabel('Top-5 Concentration')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Attention Concentration\n(per text token)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved attention metrics to {save_path}")

    return entropy.mean().item(), concentration.mean().item()


def process_sample(model, processor, dataset, sample_idx, output_dir, save_tensors=False):
    """Process a single sample and extract attention."""
    print(f"\n{'='*80}")
    print(f"Processing sample {sample_idx}")
    print(f"{'='*80}")

    # Get sample
    sample = dataset[sample_idx]
    image = sample['image_options'][0]
    caption = sample['caption_options'][0]

    # Extract objects
    words = caption.split()
    object1 = words[1]
    object2 = words[-1]

    print(f"Caption: {caption}")
    print(f"Object 1: {object1}, Object 2: {object2}")

    # Create question
    question = f"Where is the {object1} in relation to the {object2}? Answer with left, right, front or behind."
    print(f"Question: {question}")

    # Extract attention
    cross_attn, tokens_list, text_tokens, grid_size, vision_start_idx, vision_end_idx = \
        extract_cross_attention(model, processor, image, question)

    # Create sample output directory
    sample_dir = os.path.join(output_dir, f"sample_{sample_idx:04d}")
    os.makedirs(sample_dir, exist_ok=True)

    # Save original image
    image.save(os.path.join(sample_dir, "original_image.png"))

    # Find interesting tokens
    spatial_words = ['left', 'right', 'front', 'behind', 'where']
    object_words = [object1.lower(), object2.lower()]
    interesting_words = spatial_words + object_words

    interesting_indices = []
    interesting_labels = []

    for i, token in enumerate(text_tokens):
        token_clean = token.strip().lower()
        for word in interesting_words:
            if word in token_clean:
                interesting_indices.append(i)
                interesting_labels.append(f"{token}({i})")
                break

    print(f"Found {len(interesting_indices)} interesting tokens: {interesting_labels}")

    # Visualize token attention
    if len(interesting_indices) > 0:
        token_attn_path = os.path.join(sample_dir, "token_attention.png")
        visualize_token_attention(
            image, cross_attn, text_tokens, interesting_indices,
            interesting_labels, grid_size, caption, token_attn_path
        )

    # Visualize spatial word attention
    spatial_attn_path = os.path.join(sample_dir, "spatial_attention.png")
    visualize_spatial_attention(
        image, cross_attn, text_tokens, spatial_words, caption, spatial_attn_path
    )

    # Visualize attention metrics
    metrics_path = os.path.join(sample_dir, "attention_metrics.png")
    entropy_mean, concentration_mean = visualize_attention_metrics(cross_attn, metrics_path)

    # Save attention tensors if requested
    if save_tensors:
        tensor_path = os.path.join(sample_dir, "attention_tensors.pt")
        torch.save({
            'cross_attn': cross_attn.cpu(),
            'text_tokens': text_tokens,
            'grid_size': grid_size,
            'caption': caption,
            'question': question,
            'object1': object1,
            'object2': object2,
            'entropy_mean': entropy_mean,
            'concentration_mean': concentration_mean,
        }, tensor_path)
        print(f"Saved attention tensors to {tensor_path}")

    return {
        'sample_idx': sample_idx,
        'caption': caption,
        'object1': object1,
        'object2': object2,
        'entropy_mean': entropy_mean,
        'concentration_mean': concentration_mean,
        'num_interesting_tokens': len(interesting_indices),
    }


def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    model, processor = load_model(args.model_path, args.device)

    # Load dataset
    print(f"\nLoading dataset from {args.data_dir}")
    dataset = get_controlled_images_b(
        image_preprocess=None,
        download=False,
        root_dir=args.data_dir
    )
    print(f"Dataset size: {len(dataset)}")

    # Determine samples to process
    if args.num_samples is not None:
        sample_indices = range(min(args.num_samples, len(dataset)))
    else:
        sample_indices = [args.sample_idx]

    print(f"\nProcessing {len(sample_indices)} samples")

    # Process samples
    results = []
    for idx in sample_indices:
        result = process_sample(model, processor, dataset, idx, args.output_dir, args.save_tensors)
        results.append(result)

    # Save summary
    summary_path = os.path.join(args.output_dir, "summary.txt")
    with open(summary_path, 'w') as f:
        f.write("Qwen3-VL Attention Extraction Summary\n")
        f.write("=" * 80 + "\n\n")
        for result in results:
            f.write(f"Sample {result['sample_idx']:04d}:\n")
            f.write(f"  Caption: {result['caption']}\n")
            f.write(f"  Objects: {result['object1']}, {result['object2']}\n")
            f.write(f"  Entropy: {result['entropy_mean']:.3f}\n")
            f.write(f"  Concentration: {result['concentration_mean']:.3f}\n")
            f.write(f"  Interesting tokens: {result['num_interesting_tokens']}\n\n")

        # Overall statistics
        avg_entropy = np.mean([r['entropy_mean'] for r in results])
        avg_concentration = np.mean([r['concentration_mean'] for r in results])
        f.write("\nOverall Statistics:\n")
        f.write(f"  Average entropy: {avg_entropy:.3f}\n")
        f.write(f"  Average concentration: {avg_concentration:.3f}\n")

    print(f"\nSaved summary to {summary_path}")
    print(f"All visualizations saved to {args.output_dir}")
    print("\nDone!")


if __name__ == "__main__":
    main()
