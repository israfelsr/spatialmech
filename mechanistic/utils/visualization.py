import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, List, Dict
from PIL import Image


def plot_layer_wise_attention_distribution(
    layer_distributions: List[Dict[str, float]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
    title: Optional[str] = None
):
    """
    Plot layer-wise attention distribution showing image vs text percentages.

    Args:
        layer_distributions: List of dicts, one per layer, each containing:
            - 'image_pct': % attention to image tokens
            - 'text_pct': % attention to text tokens
            - 'other_pct': % attention to other tokens
        save_path: Optional path to save figure
        figsize: Figure size
        title: Optional custom title

    Example:
        layer_dists = [
            {'image_pct': 27.15, 'text_pct': 70.31, 'other_pct': 2.54},
            {'image_pct': 35.20, 'text_pct': 62.40, 'other_pct': 2.40},
            ...
        ]
        plot_layer_wise_attention_distribution(layer_dists)
    """
    num_layers = len(layer_distributions)
    layers = np.arange(num_layers)

    # Extract percentages for each category
    image_pcts = [d['image_pct'] for d in layer_distributions]
    text_pcts = [d['text_pct'] for d in layer_distributions]
    other_pcts = [d['other_pct'] for d in layer_distributions]

    # Set up the bar positions
    bar_width = 0.28
    r1 = layers
    r2 = layers + bar_width
    r3 = layers + 2 * bar_width

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create bars
    bars1 = ax.bar(r1, image_pcts, width=bar_width,
                   label='Image', color='#FF6B6B', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(r2, text_pcts, width=bar_width,
                   label='Text', color='#4ECDC4', alpha=0.8, edgecolor='black')
    bars3 = ax.bar(r3, other_pcts, width=bar_width,
                   label='Other', color='#95E1D3', alpha=0.8, edgecolor='black')

    # Customize plot
    ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel('Attention Percentage (%)', fontsize=12, fontweight='bold')

    if title is None:
        title = 'Last Token Attention Distribution Across Layers'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    ax.set_xticks(layers + bar_width)
    ax.set_xticklabels([f'L{i}' for i in layers])

    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 100)

    # Add percentage labels on top of bars (optional, for key layers)
    if num_layers <= 10:  # Only add labels if not too many layers
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                if height > 5:  # Only show label if bar is visible
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}%',
                           ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved layer-wise attention distribution to {save_path}")

    return fig


def plot_layer_wise_attention_stacked(
    layer_distributions: List[Dict[str, float]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
    title: Optional[str] = None
):
    """
    Plot layer-wise attention distribution as stacked bar chart.

    Args:
        layer_distributions: List of dicts, one per layer, each containing:
            - 'image_pct': % attention to image tokens
            - 'text_pct': % attention to text tokens
            - 'other_pct': % attention to other tokens
        save_path: Optional path to save figure
        figsize: Figure size
        title: Optional custom title
    """
    num_layers = len(layer_distributions)
    layers = np.arange(num_layers)

    # Extract percentages for each category
    image_pcts = np.array([d['image_pct'] for d in layer_distributions])
    text_pcts = np.array([d['text_pct'] for d in layer_distributions])
    other_pcts = np.array([d['other_pct'] for d in layer_distributions])

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create stacked bars
    ax.bar(layers, image_pcts, label='Image',
           color='#FF6B6B', alpha=0.8, edgecolor='black')
    ax.bar(layers, text_pcts, bottom=image_pcts, label='Text',
           color='#4ECDC4', alpha=0.8, edgecolor='black')
    ax.bar(layers, other_pcts, bottom=image_pcts+text_pcts, label='Other',
           color='#95E1D3', alpha=0.8, edgecolor='black')

    # Customize plot
    ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel('Attention Percentage (%)', fontsize=12, fontweight='bold')

    if title is None:
        title = 'Last Token Attention Distribution Across Layers (Stacked)'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    ax.set_xticks(layers)
    ax.set_xticklabels([f'L{i}' for i in layers])

    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 100)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved stacked layer-wise attention distribution to {save_path}")

    return fig


def plot_layer_wise_attention_line(
    layer_distributions: List[Dict[str, float]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
    title: Optional[str] = None
):
    """
    Plot layer-wise attention distribution as line plot.

    Args:
        layer_distributions: List of dicts, one per layer, each containing:
            - 'image_pct': % attention to image tokens
            - 'text_pct': % attention to text tokens
            - 'other_pct': % attention to other tokens
        save_path: Optional path to save figure
        figsize: Figure size
        title: Optional custom title
    """
    num_layers = len(layer_distributions)
    layers = np.arange(num_layers)

    # Extract percentages for each category
    image_pcts = [d['image_pct'] for d in layer_distributions]
    text_pcts = [d['text_pct'] for d in layer_distributions]
    other_pcts = [d['other_pct'] for d in layer_distributions]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create line plots
    ax.plot(layers, image_pcts, marker='o', linewidth=3, markersize=8,
            label='Image', color='#FF6B6B', alpha=0.8)
    ax.plot(layers, text_pcts, marker='s', linewidth=3, markersize=8,
            label='Text', color='#4ECDC4', alpha=0.8)
    ax.plot(layers, other_pcts, marker='^', linewidth=3, markersize=8,
            label='Other', color='#95E1D3', alpha=0.8)

    # Customize plot
    ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel('Attention Percentage (%)', fontsize=12, fontweight='bold')

    if title is None:
        title = 'Last Token Attention Distribution Evolution Across Layers'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    ax.set_xticks(layers)
    ax.set_xticklabels([f'L{i}' for i in layers])

    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved line plot of layer-wise attention distribution to {save_path}")

    return fig
