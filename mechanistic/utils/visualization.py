"""
Visualization Utilities for Mechanistic Interpretability

Provides functions for visualizing:
- Attention maps and heatmaps
- Attribution maps
- Activation patterns
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, List
from PIL import Image


def plot_attention_heatmap(
    attention_weights: torch.Tensor,
    tokens: Optional[List[str]] = None,
    layer_name: str = "Attention",
    head_idx: Optional[int] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = "viridis"
):
    """
    Plot attention weights as a heatmap.

    Args:
        attention_weights: Attention tensor [batch, heads, seq_len, seq_len] or [seq_len, seq_len]
        tokens: List of token labels for axes
        layer_name: Name of the layer for title
        head_idx: If provided, plot only this attention head
        save_path: Path to save figure
        figsize: Figure size
        cmap: Colormap name
    """
    # Handle different attention weight shapes
    if attention_weights.dim() == 4:
        # [batch, heads, seq_len, seq_len]
        attn = attention_weights[0]  # Take first batch
        if head_idx is not None:
            attn = attn[head_idx]  # Take specific head
        else:
            attn = attn.mean(0)  # Average over heads
    elif attention_weights.dim() == 3:
        # [heads, seq_len, seq_len]
        if head_idx is not None:
            attn = attention_weights[head_idx]
        else:
            attn = attention_weights.mean(0)
    else:
        attn = attention_weights

    attn = attn.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(attn, cmap=cmap, aspect='auto')

    # Set ticks and labels
    if tokens is not None:
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha='right')
        ax.set_yticklabels(tokens)

    ax.set_xlabel('Key', fontsize=12)
    ax.set_ylabel('Query', fontsize=12)

    title = f'{layer_name} Attention'
    if head_idx is not None:
        title += f' (Head {head_idx})'
    else:
        title += ' (Averaged)'
    ax.set_title(title, fontsize=14, pad=20)

    plt.colorbar(im, ax=ax, label='Attention Weight')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved attention heatmap to {save_path}")

    return fig, ax


def plot_attention_on_image(
    image: Image.Image,
    attention_weights: torch.Tensor,
    num_patches: int = 16,
    save_path: Optional[str] = None,
    alpha: float = 0.6,
    cmap: str = "hot"
):
    """
    Overlay attention weights on an image.

    Args:
        image: PIL Image
        attention_weights: Attention weights for image tokens [num_patches^2]
        num_patches: Number of patches per dimension (assumes square patches)
        save_path: Path to save visualization
        alpha: Transparency of attention overlay
        cmap: Colormap for attention
    """
    # Reshape attention to 2D grid
    attn = attention_weights.detach().cpu().numpy()
    if attn.ndim == 1:
        attn_grid = attn.reshape(num_patches, num_patches)
    else:
        attn_grid = attn

    # Resize attention to match image size
    img_array = np.array(image)
    attn_resized = np.array(Image.fromarray((attn_grid * 255).astype(np.uint8)).resize(
        (img_array.shape[1], img_array.shape[0]), Image.BILINEAR
    ))

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img_array)
    ax.imshow(attn_resized, cmap=cmap, alpha=alpha)
    ax.axis('off')
    ax.set_title('Attention Overlay', fontsize=14, pad=20)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved attention overlay to {save_path}")

    return fig, ax


def plot_multi_head_attention(
    attention_weights: torch.Tensor,
    layer_name: str = "Layer",
    save_path: Optional[str] = None,
    max_heads: int = 12,
    figsize: Tuple[int, int] = (20, 16)
):
    """
    Plot attention patterns for multiple heads.

    Args:
        attention_weights: Attention tensor [batch, heads, seq_len, seq_len]
        layer_name: Name for the plot title
        save_path: Path to save figure
        max_heads: Maximum number of heads to plot
        figsize: Figure size
    """
    if attention_weights.dim() == 4:
        attn = attention_weights[0]  # [heads, seq_len, seq_len]
    else:
        attn = attention_weights

    num_heads = min(attn.shape[0], max_heads)
    rows = int(np.ceil(np.sqrt(num_heads)))
    cols = int(np.ceil(num_heads / rows))

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if num_heads > 1 else [axes]

    for i in range(num_heads):
        ax = axes[i]
        attn_head = attn[i].detach().cpu().numpy()
        im = ax.imshow(attn_head, cmap='viridis', aspect='auto')
        ax.set_title(f'Head {i}', fontsize=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

    # Hide extra subplots
    for i in range(num_heads, len(axes)):
        axes[i].axis('off')

    fig.suptitle(f'{layer_name} - All Attention Heads', fontsize=16, y=0.98)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved multi-head attention to {save_path}")

    return fig, axes


def plot_cross_attention_attribution(
    image: Image.Image,
    text_tokens: List[str],
    cross_attention: torch.Tensor,
    num_patches: int = 16,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5)
):
    """
    Visualize cross-attention from text tokens to image patches.

    Args:
        image: Input image
        text_tokens: List of text tokens
        cross_attention: Cross-attention weights [num_text_tokens, num_image_patches]
        num_patches: Number of patches per dimension
        save_path: Path to save figure
        figsize: Figure size
    """
    cross_attn = cross_attention.detach().cpu().numpy()

    # Create subplots
    num_tokens = len(text_tokens)
    fig, axes = plt.subplots(1, num_tokens + 1, figsize=figsize)

    # Plot original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image', fontsize=10)
    axes[0].axis('off')

    # Plot attention for each token
    for i, token in enumerate(text_tokens):
        token_attn = cross_attn[i].reshape(num_patches, num_patches)

        # Resize to image dimensions
        img_array = np.array(image)
        attn_resized = np.array(Image.fromarray((token_attn * 255).astype(np.uint8)).resize(
            (img_array.shape[1], img_array.shape[0]), Image.BILINEAR
        ))

        axes[i + 1].imshow(img_array)
        axes[i + 1].imshow(attn_resized, cmap='hot', alpha=0.6)
        axes[i + 1].set_title(f'"{token}"', fontsize=10)
        axes[i + 1].axis('off')

    plt.suptitle('Cross-Attention Attribution per Token', fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved cross-attention attribution to {save_path}")

    return fig, axes


__all__ = [
    'plot_attention_heatmap',
    'plot_attention_on_image',
    'plot_multi_head_attention',
    'plot_cross_attention_attribution'
]
