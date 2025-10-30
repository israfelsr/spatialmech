import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, List, Dict
from PIL import Image


def plot_layer_wise_attention(
    layer_distributions: List[Dict[str, float]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
    title: Optional[str] = None,
):
    """
    Plot layer-wise attention distribution as line plot.

    Args:
        layer_distributions: List of dicts, one per layer, each containing:
            - 'image_pct': % attention to image tokens
            - 'text_pct': % attention to text tokens
            - 'bos_pct': % attention to other tokens
        save_path: Optional path to save figure
        figsize: Figure size
        title: Optional custom title
    """
    num_layers = len(layer_distributions)
    layers = np.arange(num_layers)

    # Extract percentages for each category
    image_pcts = [d["image_pct"] for d in layer_distributions]
    text_pcts = [d["text_pct"] for d in layer_distributions]
    bos_pcts = [d["bos_pct"] for d in layer_distributions]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create line plots
    ax.plot(
        layers,
        image_pcts,
        marker="o",
        linewidth=3,
        markersize=8,
        label="Image",
        alpha=0.8,
    )
    ax.plot(
        layers,
        text_pcts,
        marker="s",
        linewidth=3,
        markersize=8,
        label="Text",
        alpha=0.8,
    )
    ax.plot(
        layers,
        bos_pcts,
        marker="^",
        linewidth=3,
        markersize=8,
        label="Other",
        alpha=0.8,
    )

    # Customize plot
    ax.set_xlabel("Layer", fontsize=12, fontweight="bold")
    ax.set_ylabel("Attention Percentage (%)", fontsize=12, fontweight="bold")

    if title is None:
        title = "Last Token Attention Distribution Evolution Across Layers"
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    ax.set_xticks(layers)
    ax.set_xticklabels([f"L{i}" for i in layers])

    ax.legend(fontsize=11, loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved line plot of layer-wise attention distribution to {save_path}")

    return fig
