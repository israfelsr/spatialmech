"""
Visualization utilities for 2-hop attention analysis.

Functions for visualizing:
- Attention distribution pie charts
- Attention flow diagrams
- Hub token identification
- Layer-wise evolution
- Direct vs indirect comparison
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
import seaborn as sns
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from .two_hop_metrics import TwoHopMetrics


def plot_attention_distribution_comparison(
    metrics: TwoHopMetrics,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 5)
):
    """
    Plot pie charts comparing last token vs text tokens attention distribution.

    Args:
        metrics: TwoHopMetrics object
        save_path: Optional path to save figure
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Last token distribution
    last_token_data = [
        metrics.last_token_image_pct,
        metrics.last_token_text_pct,
        100 - metrics.last_token_image_pct - metrics.last_token_text_pct
    ]
    labels = ['Image', 'Text', 'Other']
    colors = ['#FF6B6B', '#4ECDC4', '#95E1D3']

    wedges1, texts1, autotexts1 = axes[0].pie(
        last_token_data,
        labels=labels,
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        textprops={'fontsize': 12}
    )
    axes[0].set_title('Last Token Attention Distribution', fontsize=14, fontweight='bold')

    # Make percentage text bold
    for autotext in autotexts1:
        autotext.set_color('white')
        autotext.set_fontweight('bold')

    # Text tokens distribution (average)
    text_tokens_data = [
        metrics.text_tokens_image_pct_mean,
        metrics.text_tokens_text_pct_mean,
        100 - metrics.text_tokens_image_pct_mean - metrics.text_tokens_text_pct_mean
    ]

    wedges2, texts2, autotexts2 = axes[1].pie(
        text_tokens_data,
        labels=labels,
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        textprops={'fontsize': 12}
    )
    axes[1].set_title(
        f'Text Tokens Attention Distribution\n(Mean across {metrics.num_text_tokens} tokens)',
        fontsize=14,
        fontweight='bold'
    )

    for autotext in autotexts2:
        autotext.set_color('white')
        autotext.set_fontweight('bold')

    plt.suptitle(
        '2-Hop Hypothesis: Attention Distribution Comparison',
        fontsize=16,
        fontweight='bold',
        y=1.02
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved attention distribution comparison to {save_path}")

    return fig


def plot_attention_flow(
    attention: torch.Tensor,
    vision_range: Tuple[int, int],
    text_range: Tuple[int, int],
    metrics: TwoHopMetrics,
    last_token_idx: int = -1,
    top_k_text_tokens: int = 5,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
):
    """
    Visualize 2-hop attention flow: Image → Text Tokens → Last Token.

    Args:
        attention: Attention tensor [seq, seq] (already averaged over heads/batch)
        vision_range: (start, end) for vision tokens
        text_range: (start, end) for text tokens
        metrics: TwoHopMetrics object
        last_token_idx: Index of last token
        top_k_text_tokens: Number of top text tokens to show
        save_path: Optional path to save figure
        figsize: Figure size
    """
    # Handle tensor shapes
    if attention.dim() == 4:
        attention = attention[0].mean(0)
    elif attention.dim() == 3:
        attention = attention.mean(0)

    vision_start, vision_end = vision_range
    text_start, text_end = text_range

    # Get attention weights
    text_to_image = attention[text_start:text_end, vision_start:vision_end].sum(dim=1)  # [num_text]
    last_to_text = attention[last_token_idx, text_start:text_end]  # [num_text]

    # Get top-k text tokens by hub score (product)
    hub_scores = text_to_image * last_to_text
    top_k = min(top_k_text_tokens, len(hub_scores))
    top_k_values, top_k_indices = torch.topk(hub_scores, top_k)

    fig, ax = plt.subplots(figsize=figsize)

    # Draw three columns: Image | Text Tokens | Last Token
    image_x = 0.1
    text_x = 0.45
    last_x = 0.8

    # Draw boxes
    image_box = FancyBboxPatch(
        (image_x - 0.08, 0.4), 0.16, 0.2,
        boxstyle="round,pad=0.01",
        edgecolor='#FF6B6B',
        facecolor='#FFE5E5',
        linewidth=3
    )
    ax.add_patch(image_box)
    ax.text(image_x, 0.5, 'Image\nTokens', ha='center', va='center',
            fontsize=14, fontweight='bold')

    last_box = FancyBboxPatch(
        (last_x - 0.08, 0.4), 0.16, 0.2,
        boxstyle="round,pad=0.01",
        edgecolor='#4ECDC4',
        facecolor='#E5F9F7',
        linewidth=3
    )
    ax.add_patch(last_box)
    ax.text(last_x, 0.5, 'Last\nToken', ha='center', va='center',
            fontsize=14, fontweight='bold')

    # Draw text token boxes
    token_height = 0.08
    token_spacing = 0.1
    start_y = 0.5 + (top_k - 1) * token_spacing / 2

    for i, idx in enumerate(top_k_indices):
        y_pos = start_y - i * token_spacing

        # Text token box
        token_box = FancyBboxPatch(
            (text_x - 0.1, y_pos - token_height/2), 0.2, token_height,
            boxstyle="round,pad=0.005",
            edgecolor='#95E1D3',
            facecolor='#F0FFF0',
            linewidth=2
        )
        ax.add_patch(token_box)

        # Token label
        token_str = metrics.hub_token_strings[metrics.hub_token_indices.index(idx.item())] \
                    if idx.item() in metrics.hub_token_indices else f"T{idx}"
        ax.text(text_x, y_pos, token_str, ha='center', va='center',
                fontsize=11, fontweight='bold')

        # Arrow from image to text token
        text_to_img_weight = text_to_image[idx].item()
        arrow1 = FancyArrowPatch(
            (image_x + 0.08, 0.5), (text_x - 0.1, y_pos),
            arrowstyle='->,head_width=0.4,head_length=0.4',
            color='#FF6B6B',
            alpha=min(1.0, text_to_img_weight * 5),
            linewidth=2 + text_to_img_weight * 10,
            linestyle='-'
        )
        ax.add_patch(arrow1)

        # Arrow from text token to last
        last_to_txt_weight = last_to_text[idx].item()
        arrow2 = FancyArrowPatch(
            (text_x + 0.1, y_pos), (last_x - 0.08, 0.5),
            arrowstyle='->,head_width=0.4,head_length=0.4',
            color='#4ECDC4',
            alpha=min(1.0, last_to_txt_weight * 5),
            linewidth=2 + last_to_txt_weight * 10,
            linestyle='-'
        )
        ax.add_patch(arrow2)

    # Add legend
    ax.text(0.5, 0.95, f'2-Hop Attention Flow (Top-{top_k} Hub Tokens)',
            ha='center', va='top', fontsize=16, fontweight='bold',
            transform=ax.transAxes)

    ax.text(0.5, 0.05,
            f'Attention Flow Score: {metrics.attention_flow_score:.3f} | '
            f'Hub Score: {metrics.information_bottleneck_score:.3f}',
            ha='center', va='bottom', fontsize=12,
            transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # Add arrows for direct path (dashed)
    direct_arrow = FancyArrowPatch(
        (image_x + 0.08, 0.45), (last_x - 0.08, 0.45),
        arrowstyle='->,head_width=0.3,head_length=0.3',
        color='gray',
        alpha=0.3,
        linewidth=1.5,
        linestyle='--'
    )
    ax.add_patch(direct_arrow)
    ax.text(0.5, 0.42, 'Direct (weak)', ha='center', va='top',
            fontsize=9, color='gray', style='italic')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved attention flow diagram to {save_path}")

    return fig


def plot_hub_tokens_analysis(
    metrics: TwoHopMetrics,
    attention: torch.Tensor,
    vision_range: Tuple[int, int],
    text_range: Tuple[int, int],
    last_token_idx: int = -1,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 6)
):
    """
    Visualize which text tokens act as information hubs.

    Args:
        metrics: TwoHopMetrics object
        attention: Attention tensor
        vision_range: (start, end) for vision tokens
        text_range: (start, end) for text tokens
        last_token_idx: Index of last token
        save_path: Optional path to save figure
        figsize: Figure size
    """
    # Handle tensor shapes
    if attention.dim() == 4:
        attention = attention[0].mean(0)
    elif attention.dim() == 3:
        attention = attention.mean(0)

    vision_start, vision_end = vision_range
    text_start, text_end = text_range

    # Get attention weights
    text_to_image = attention[text_start:text_end, vision_start:vision_end].sum(dim=1).cpu().numpy()
    last_to_text = attention[last_token_idx, text_start:text_end].cpu().numpy()

    # Normalize
    text_to_image_norm = (text_to_image - text_to_image.min()) / (text_to_image.max() - text_to_image.min() + 1e-8)
    last_to_text_norm = (last_to_text - last_to_text.min()) / (last_to_text.max() - last_to_text.min() + 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Scatter plot
    axes[0].scatter(text_to_image_norm, last_to_text_norm, alpha=0.6, s=100)

    # Highlight hub tokens
    for i, idx in enumerate(metrics.hub_token_indices):
        axes[0].scatter(
            text_to_image_norm[idx],
            last_to_text_norm[idx],
            color='red',
            s=200,
            marker='*',
            zorder=10,
            edgecolors='darkred',
            linewidths=2
        )
        axes[0].annotate(
            metrics.hub_token_strings[i],
            (text_to_image_norm[idx], last_to_text_norm[idx]),
            xytext=(10, 10),
            textcoords='offset points',
            fontsize=10,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5)
        )

    axes[0].set_xlabel('Text → Image Attention (normalized)', fontsize=12)
    axes[0].set_ylabel('Last → Text Attention (normalized)', fontsize=12)
    axes[0].set_title('Hub Token Identification', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Correlation annotation
    axes[0].text(
        0.05, 0.95,
        f'Correlation: {metrics.attention_flow_score:.3f}',
        transform=axes[0].transAxes,
        fontsize=11,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

    # Bar plot of hub scores
    hub_scores = text_to_image_norm * last_to_text_norm
    token_indices = np.arange(len(hub_scores))

    colors = ['red' if i in metrics.hub_token_indices else 'lightblue'
              for i in range(len(hub_scores))]

    axes[1].bar(token_indices, hub_scores, color=colors, alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Text Token Index', fontsize=12)
    axes[1].set_ylabel('Hub Score (Text→Image × Last→Text)', fontsize=12)
    axes[1].set_title('Hub Scores Across All Text Tokens', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')

    # Highlight top tokens
    for idx in metrics.hub_token_indices:
        axes[1].annotate(
            '',
            xy=(idx, hub_scores[idx]),
            xytext=(idx, hub_scores[idx] + 0.05),
            arrowprops=dict(arrowstyle='->', color='red', lw=2)
        )

    plt.suptitle(
        'Information Bottleneck Analysis: Which Tokens Act as Hubs?',
        fontsize=16,
        fontweight='bold',
        y=1.02
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved hub tokens analysis to {save_path}")

    return fig


def plot_layer_wise_evolution(
    metrics: TwoHopMetrics,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 6)
):
    """
    Plot how attention patterns evolve across layers.

    Args:
        metrics: TwoHopMetrics object (must have layer-wise data)
        save_path: Optional path to save figure
        figsize: Figure size
    """
    if metrics.layer_wise_last_token_image_pct is None:
        raise ValueError("No layer-wise data available. Set analyze_all_layers=True when computing metrics.")

    num_layers = len(metrics.layer_wise_last_token_image_pct)
    layers = np.arange(num_layers)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot 1: Last token image attention across layers
    axes[0].plot(
        layers,
        metrics.layer_wise_last_token_image_pct,
        marker='o',
        linewidth=3,
        markersize=8,
        color='#4ECDC4',
        label='Last Token → Image'
    )
    axes[0].set_xlabel('Layer', fontsize=12)
    axes[0].set_ylabel('Image Attention %', fontsize=12)
    axes[0].set_title('Last Token Image Attention Evolution', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=10)

    # Plot 2: Text tokens mean image attention across layers
    axes[1].plot(
        layers,
        metrics.layer_wise_text_tokens_image_pct,
        marker='s',
        linewidth=3,
        markersize=8,
        color='#FF6B6B',
        label='Text Tokens → Image (mean)'
    )
    axes[1].set_xlabel('Layer', fontsize=12)
    axes[1].set_ylabel('Image Attention %', fontsize=12)
    axes[1].set_title('Text Tokens Image Attention Evolution', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=10)

    plt.suptitle(
        '2-Hop Hypothesis: Layer-wise Attention Evolution',
        fontsize=16,
        fontweight='bold',
        y=1.02
    )

    # Add interpretation note
    fig.text(
        0.5, -0.02,
        'Expected pattern: Text→Image increases in early layers, Last→Image remains low in later layers',
        ha='center',
        fontsize=10,
        style='italic',
        color='gray'
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved layer-wise evolution to {save_path}")

    return fig


def plot_direct_vs_indirect(
    metrics: TwoHopMetrics,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
):
    """
    Visualize direct vs indirect image attention from last token.

    Args:
        metrics: TwoHopMetrics object
        save_path: Optional path to save figure
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    categories = ['Direct\n(Last → Image)', 'Indirect\n(Last → Text → Image)']
    values = [metrics.direct_image_attention, metrics.indirect_image_attention]
    colors = ['#FF6B6B', '#4ECDC4']

    bars = ax.bar(categories, values, color=colors, alpha=0.8, edgecolor='black', linewidth=2)

    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{value:.4f}',
            ha='center',
            va='bottom',
            fontsize=14,
            fontweight='bold'
        )

    ax.set_ylabel('Attention Weight', fontsize=12)
    ax.set_title(
        f'Direct vs Indirect Image Attention\n'
        f'Ratio: {metrics.indirect_to_direct_ratio:.2f}x '
        f'({"Indirect" if metrics.indirect_to_direct_ratio > 1 else "Direct"} is stronger)',
        fontsize=14,
        fontweight='bold'
    )
    ax.grid(True, alpha=0.3, axis='y')

    # Add interpretation box
    interpretation = (
        "Ratio > 1: Supports 2-hop hypothesis\n"
        "(Last token accesses image indirectly via text tokens)\n\n"
        "Ratio < 1: Direct path dominates\n"
        "(Last token looks directly at image)"
    )
    ax.text(
        0.98, 0.97,
        interpretation,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7)
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved direct vs indirect comparison to {save_path}")

    return fig


def plot_complete_analysis(
    metrics: TwoHopMetrics,
    attention: torch.Tensor,
    vision_range: Tuple[int, int],
    text_range: Tuple[int, int],
    last_token_idx: int = -1,
    output_dir: str = ".",
    prefix: str = "sample"
):
    """
    Generate all visualization plots for a complete 2-hop analysis.

    Args:
        metrics: TwoHopMetrics object
        attention: Attention tensor
        vision_range: (start, end) for vision tokens
        text_range: (start, end) for text tokens
        last_token_idx: Index of last token
        output_dir: Directory to save plots
        prefix: Prefix for filenames
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # 1. Attention distribution comparison
    plot_attention_distribution_comparison(
        metrics,
        save_path=os.path.join(output_dir, f"{prefix}_attention_distribution.png")
    )
    plt.close()

    # 2. Attention flow diagram
    plot_attention_flow(
        attention,
        vision_range,
        text_range,
        metrics,
        last_token_idx,
        save_path=os.path.join(output_dir, f"{prefix}_attention_flow.png")
    )
    plt.close()

    # 3. Hub tokens analysis
    plot_hub_tokens_analysis(
        metrics,
        attention,
        vision_range,
        text_range,
        last_token_idx,
        save_path=os.path.join(output_dir, f"{prefix}_hub_tokens.png")
    )
    plt.close()

    # 4. Direct vs indirect
    plot_direct_vs_indirect(
        metrics,
        save_path=os.path.join(output_dir, f"{prefix}_direct_vs_indirect.png")
    )
    plt.close()

    # 5. Layer-wise evolution (if available)
    if metrics.layer_wise_last_token_image_pct is not None:
        plot_layer_wise_evolution(
            metrics,
            save_path=os.path.join(output_dir, f"{prefix}_layer_evolution.png")
        )
        plt.close()

    print(f"\nAll plots saved to {output_dir}/ with prefix '{prefix}'")


__all__ = [
    "plot_attention_distribution_comparison",
    "plot_attention_flow",
    "plot_hub_tokens_analysis",
    "plot_layer_wise_evolution",
    "plot_direct_vs_indirect",
    "plot_complete_analysis",
]
