"""
Two-Hop Hypothesis Analysis Script

Analyzes attention patterns to test the 2-hop hypothesis:
"Text tokens gather information from the image, and the last token
reads from those text tokens (rather than directly from the image)."

Usage:
    # Analyze a single sample from a saved dataset
    python mechanistic/experiments/analyze_two_hop_hypothesis.py \
        --data-path results/correct_samples.pt \
        --sample-idx 0 \
        --output-dir plots/mechanistic/two_hop_analysis

    # Analyze multiple samples and compute aggregate statistics
    python mechanistic/experiments/analyze_two_hop_hypothesis.py \
        --data-path results/correct_samples.pt \
        --num-samples 50 \
        --output-dir plots/mechanistic/two_hop_analysis \
        --save-aggregate
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from mechanistic.utils.two_hop_metrics import (
    compute_all_two_hop_metrics,
    print_metrics_summary,
    TwoHopMetrics,
    identify_token_ranges,
)
from mechanistic.utils.two_hop_visualization import plot_complete_analysis


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze 2-hop attention hypothesis")
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to saved samples (e.g., results/correct_samples.pt)"
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
        help="Number of samples to analyze (if None, only analyze sample-idx)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots/mechanistic/two_hop_analysis",
        help="Output directory for visualizations and results"
    )
    parser.add_argument(
        "--layer-idx",
        type=int,
        default=-1,
        help="Which layer to analyze (-1 for last layer)"
    )
    parser.add_argument(
        "--analyze-all-layers",
        action="store_true",
        help="Analyze all layers for layer-wise evolution"
    )
    parser.add_argument(
        "--save-aggregate",
        action="store_true",
        help="Save aggregate statistics across multiple samples"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="qwen2vl",
        help="Model name (for loading tokenizer)"
    )
    return parser.parse_args()


def load_tokenizer(model_name: str):
    """Load tokenizer based on model name."""
    from transformers import AutoTokenizer

    model_paths = {
        "qwen2vl": "Qwen/Qwen2-VL-7B-Instruct",
        "qwen3vl": "Qwen/Qwen3-VL-4B-Instruct",
    }

    model_path = model_paths.get(model_name.lower(), model_name)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print(f"Loaded tokenizer from {model_path}")
        return tokenizer
    except Exception as e:
        print(f"Warning: Could not load tokenizer: {e}")
        print("Proceeding without tokenizer (will need to provide token ranges manually)")
        return None


def analyze_sample(
    sample: Dict,
    tokenizer,
    layer_idx: int = -1,
    analyze_all_layers: bool = False,
    verbose: bool = True
) -> TwoHopMetrics:
    """
    Analyze a single sample for 2-hop hypothesis.

    Args:
        sample: Dictionary with 'attentions', 'input_ids', etc.
        tokenizer: Tokenizer for decoding tokens
        layer_idx: Which layer to analyze
        analyze_all_layers: Whether to compute layer-wise evolution
        verbose: Whether to print detailed output

    Returns:
        TwoHopMetrics object
    """
    if verbose:
        print("\n" + "=" * 80)
        print("ANALYZING SAMPLE")
        print("=" * 80)
        print(f"Dataset index: {sample.get('dataset_idx', 'N/A')}")
        print(f"Caption: {sample.get('caption', 'N/A')}")
        print(f"Question: {sample.get('question', 'N/A')}")
        print(f"Prediction: {sample.get('pred', 'N/A')}")
        print(f"Ground Truth: {sample.get('GT', 'N/A')}")

    # Compute metrics
    try:
        metrics = compute_all_two_hop_metrics(
            sample,
            tokenizer=tokenizer,
            layer_idx=layer_idx,
            analyze_all_layers=analyze_all_layers
        )

        if verbose:
            print_metrics_summary(metrics, verbose=True)

        return metrics

    except Exception as e:
        print(f"Error analyzing sample: {e}")
        raise


def plot_aggregate_statistics(
    all_metrics: List[TwoHopMetrics],
    output_dir: str
):
    """
    Plot aggregate statistics across multiple samples.

    Args:
        all_metrics: List of TwoHopMetrics objects
        output_dir: Directory to save plots
    """
    # Extract metrics
    last_token_image_pcts = [m.last_token_image_pct for m in all_metrics]
    last_token_text_pcts = [m.last_token_text_pct for m in all_metrics]
    text_tokens_image_pcts = [m.text_tokens_image_pct_mean for m in all_metrics]
    attention_flow_scores = [m.attention_flow_score for m in all_metrics]
    hub_scores = [m.information_bottleneck_score for m in all_metrics]
    indirect_ratios = [m.indirect_to_direct_ratio for m in all_metrics]

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f'2-Hop Hypothesis: Aggregate Statistics (n={len(all_metrics)} samples)',
        fontsize=16,
        fontweight='bold'
    )

    # 1. Last token image attention distribution
    axes[0, 0].hist(last_token_image_pcts, bins=30, alpha=0.7, color='#FF6B6B', edgecolor='black')
    axes[0, 0].axvline(np.mean(last_token_image_pcts), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(last_token_image_pcts):.2f}%')
    axes[0, 0].set_xlabel('Last Token → Image (%)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Last Token Image Attention')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Text tokens image attention distribution
    axes[0, 1].hist(text_tokens_image_pcts, bins=30, alpha=0.7, color='#4ECDC4', edgecolor='black')
    axes[0, 1].axvline(np.mean(text_tokens_image_pcts), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(text_tokens_image_pcts):.2f}%')
    axes[0, 1].set_xlabel('Text Tokens → Image (mean %)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Text Tokens Image Attention')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Attention flow score distribution
    axes[0, 2].hist(attention_flow_scores, bins=30, alpha=0.7, color='#95E1D3', edgecolor='black')
    axes[0, 2].axvline(np.mean(attention_flow_scores), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(attention_flow_scores):.3f}')
    axes[0, 2].set_xlabel('Attention Flow Score (correlation)')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Attention Flow Score')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # 4. Hub score distribution
    axes[1, 0].hist(hub_scores, bins=30, alpha=0.7, color='#FFB6B9', edgecolor='black')
    axes[1, 0].axvline(np.mean(hub_scores), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(hub_scores):.3f}')
    axes[1, 0].set_xlabel('Information Bottleneck Score')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Hub Token Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 5. Indirect/Direct ratio distribution
    axes[1, 1].hist(indirect_ratios, bins=30, alpha=0.7, color='#A8E6CF', edgecolor='black')
    axes[1, 1].axvline(np.mean(indirect_ratios), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(indirect_ratios):.2f}x')
    axes[1, 1].axvline(1.0, color='gray', linestyle=':', linewidth=2, label='Equal (1.0x)')
    axes[1, 1].set_xlabel('Indirect/Direct Ratio')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Indirect vs Direct Image Attention')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # 6. Scatter: Last token vs Text tokens image attention
    axes[1, 2].scatter(text_tokens_image_pcts, last_token_image_pcts, alpha=0.5, s=50)
    axes[1, 2].set_xlabel('Text Tokens → Image (mean %)')
    axes[1, 2].set_ylabel('Last Token → Image (%)')
    axes[1, 2].set_title('Last Token vs Text Tokens Image Attention')
    axes[1, 2].grid(True, alpha=0.3)

    # Add diagonal line
    max_val = max(max(text_tokens_image_pcts), max(last_token_image_pcts))
    axes[1, 2].plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='Equal attention')
    axes[1, 2].legend()

    plt.tight_layout()
    save_path = os.path.join(output_dir, "aggregate_statistics.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved aggregate statistics to {save_path}")

    # Print summary statistics
    print("\n" + "=" * 80)
    print("AGGREGATE STATISTICS SUMMARY")
    print("=" * 80)
    print(f"Number of samples: {len(all_metrics)}")
    print(f"\nLast Token → Image Attention:")
    print(f"  Mean: {np.mean(last_token_image_pcts):.2f}%")
    print(f"  Std:  {np.std(last_token_image_pcts):.2f}%")
    print(f"  Median: {np.median(last_token_image_pcts):.2f}%")
    print(f"\nText Tokens → Image Attention:")
    print(f"  Mean: {np.mean(text_tokens_image_pcts):.2f}%")
    print(f"  Std:  {np.std(text_tokens_image_pcts):.2f}%")
    print(f"  Median: {np.median(text_tokens_image_pcts):.2f}%")
    print(f"\nAttention Flow Score:")
    print(f"  Mean: {np.mean(attention_flow_scores):.3f}")
    print(f"  Std:  {np.std(attention_flow_scores):.3f}")
    print(f"  Median: {np.median(attention_flow_scores):.3f}")
    print(f"\nInformation Bottleneck Score:")
    print(f"  Mean: {np.mean(hub_scores):.3f}")
    print(f"  Std:  {np.std(hub_scores):.3f}")
    print(f"  Median: {np.median(hub_scores):.3f}")
    print(f"\nIndirect/Direct Ratio:")
    print(f"  Mean: {np.mean(indirect_ratios):.2f}x")
    print(f"  Std:  {np.std(indirect_ratios):.2f}x")
    print(f"  Median: {np.median(indirect_ratios):.2f}x")
    print(f"  % samples with ratio > 1: {np.mean([r > 1 for r in indirect_ratios]) * 100:.1f}%")
    print("=" * 80)


def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    print(f"Loading data from {args.data_path}")
    data = torch.load(args.data_path)

    # Handle different data formats
    if isinstance(data, dict):
        # Single sample
        samples = [data]
    elif isinstance(data, list):
        # List of samples
        samples = data
    else:
        raise ValueError(f"Unexpected data format: {type(data)}")

    print(f"Loaded {len(samples)} samples")

    # Load tokenizer
    tokenizer = load_tokenizer(args.model_name)

    # Determine samples to process
    if args.num_samples is not None:
        sample_indices = range(min(args.num_samples, len(samples)))
    else:
        sample_indices = [args.sample_idx]

    print(f"\nProcessing {len(sample_indices)} samples")

    # Process samples
    all_metrics = []
    for idx in tqdm(sample_indices, desc="Analyzing samples"):
        sample = samples[idx]

        try:
            metrics = analyze_sample(
                sample,
                tokenizer,
                layer_idx=args.layer_idx,
                analyze_all_layers=args.analyze_all_layers,
                verbose=(len(sample_indices) == 1)  # Only verbose for single sample
            )

            all_metrics.append(metrics)

            # Generate visualizations for single sample or first few samples
            if len(sample_indices) == 1 or (len(sample_indices) <= 5 and idx < 5):
                sample_dir = os.path.join(args.output_dir, f"sample_{idx:04d}")
                os.makedirs(sample_dir, exist_ok=True)

                # Get attention tensor for visualization
                attentions = sample["attentions"]
                if isinstance(attentions, tuple):
                    attention = attentions[args.layer_idx]
                else:
                    attention = attentions

                # Identify token ranges
                vision_range, text_range = identify_token_ranges(sample["input_ids"], tokenizer)

                # Generate all plots
                plot_complete_analysis(
                    metrics,
                    attention,
                    vision_range,
                    text_range,
                    output_dir=sample_dir,
                    prefix=f"sample_{idx}"
                )

        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue

    # Save aggregate statistics if requested
    if args.save_aggregate and len(all_metrics) > 1:
        plot_aggregate_statistics(all_metrics, args.output_dir)

        # Save metrics to file
        metrics_path = os.path.join(args.output_dir, "all_metrics.pt")
        torch.save(all_metrics, metrics_path)
        print(f"\nSaved all metrics to {metrics_path}")

    print(f"\nAll results saved to {args.output_dir}")
    print("\nDone!")


if __name__ == "__main__":
    main()
