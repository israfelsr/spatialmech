import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, List, Dict, Union
from PIL import Image
import pandas as pd


def plot_layer_wise_attention(
    data: Union[List[Dict[str, float]], pd.DataFrame],
    sample_idx: Optional[int] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
    title: Optional[str] = None,
):
    """
    Plot layer-wise attention distribution as line plot for a single sample.

    Args:
        data: Either a list of dicts (one per layer) OR a pandas DataFrame
              with columns: [sample_idx, layer_idx, image_pct, text_pct, bos_pct]
        sample_idx: If data is DataFrame, which sample to plot (required)
        save_path: Optional path to save figure
        figsize: Figure size
        title: Optional custom title

    Example with DataFrame:
        plot_layer_wise_attention(df, sample_idx=54, save_path='plots/sample_54.png')

    Example with list:
        layer_dists = [{'image_pct': 27.15, 'text_pct': 70.31, 'bos_pct': 2.54}, ...]
        plot_layer_wise_attention(layer_dists)
    """
    # Handle DataFrame input
    if isinstance(data, pd.DataFrame):
        if sample_idx is None:
            raise ValueError("sample_idx is required when passing a DataFrame")

        # Filter for specific sample
        sample_data = data[data['sample_idx'] == sample_idx].sort_values('layer_idx')

        if len(sample_data) == 0:
            raise ValueError(f"No data found for sample_idx={sample_idx}")

        layers = sample_data['layer_idx'].values
        image_pcts = sample_data['image_pct'].values
        text_pcts = sample_data['text_pct'].values
        bos_pcts = sample_data['bos_pct'].values

        # Get GT label if available for title
        gt_label = sample_data['GT'].iloc[0] if 'GT' in sample_data.columns else None

    # Handle list of dicts input
    else:
        num_layers = len(data)
        layers = np.arange(num_layers)
        image_pcts = [d["image_pct"] for d in data]
        text_pcts = [d["text_pct"] for d in data]
        bos_pcts = [d["bos_pct"] for d in data]
        gt_label = None

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
        label="BOS/Other",
        alpha=0.8,
    )

    # Customize plot
    ax.set_xlabel("Layer", fontsize=12, fontweight="bold")
    ax.set_ylabel("Attention Percentage (%)", fontsize=12, fontweight="bold")

    if title is None:
        title = "Last Token Attention Distribution Across Layers"
        if sample_idx is not None:
            title += f"\nSample {sample_idx}"
        if gt_label is not None:
            title += f" (GT: {gt_label})"
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    ax.set_xticks(layers)
    ax.set_xticklabels([f"L{int(i)}" for i in layers])

    ax.legend(fontsize=11, loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved line plot of layer-wise attention distribution to {save_path}")

    return fig


def plot_layer_wise_attention_with_ci(
    df: pd.DataFrame,
    gt_filter: Optional[Union[str, List[str]]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 8),
    title: Optional[str] = None,
    ci: Union[str, int] = 95,
):
    """
    Plot layer-wise attention with confidence intervals across multiple samples.

    Uses seaborn to show mean ± confidence interval for each attention type across layers.

    Args:
        df: DataFrame with columns [sample_idx, layer_idx, image_pct, text_pct, bos_pct, GT]
        gt_filter: Optional filter for GT column. Can be:
                   - Single string: "left"
                   - List of strings: ["left", "right"]
                   - None: use all samples
        save_path: Optional path to save figure
        figsize: Figure size
        title: Optional custom title
        ci: Confidence interval (e.g., 95 for 95% CI, or 'sd' for standard deviation)

    Example:
        # Plot all samples
        plot_layer_wise_attention_with_ci(df)

        # Plot only "left" samples
        plot_layer_wise_attention_with_ci(df, gt_filter="left")

        # Plot "left" and "right" samples
        plot_layer_wise_attention_with_ci(df, gt_filter=["left", "right"])
    """
    # Filter by GT if requested
    if gt_filter is not None:
        if isinstance(gt_filter, str):
            gt_filter = [gt_filter]
        df_filtered = df[df['GT'].isin(gt_filter)].copy()
        gt_label = ", ".join(gt_filter)
    else:
        df_filtered = df.copy()
        gt_label = "All"

    if len(df_filtered) == 0:
        raise ValueError(f"No data found for GT filter: {gt_filter}")

    # Melt the dataframe for seaborn
    df_melted = df_filtered.melt(
        id_vars=['sample_idx', 'layer_idx', 'GT'],
        value_vars=['image_pct', 'text_pct', 'bos_pct'],
        var_name='attention_type',
        value_name='percentage'
    )

    # Create more readable labels
    label_map = {
        'image_pct': 'Image',
        'text_pct': 'Text',
        'bos_pct': 'BOS/Other'
    }
    df_melted['attention_type'] = df_melted['attention_type'].map(label_map)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create lineplot with confidence intervals
    sns.lineplot(
        data=df_melted,
        x='layer_idx',
        y='percentage',
        hue='attention_type',
        style='attention_type',
        markers=['o', 's', '^'],
        markersize=10,
        linewidth=3,
        ci=ci,
        ax=ax,
        alpha=0.8
    )

    # Customize plot
    ax.set_xlabel("Layer", fontsize=12, fontweight="bold")
    ax.set_ylabel("Attention Percentage (%)", fontsize=12, fontweight="bold")

    if title is None:
        n_samples = df_filtered['sample_idx'].nunique()
        title = f"Last Token Attention Distribution Across Layers\n"
        title += f"GT: {gt_label} | {n_samples} samples | "
        if ci == 'sd':
            title += "Mean ± SD"
        else:
            title += f"Mean ± {ci}% CI"
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    # Set x-axis ticks to show all layers
    layers = sorted(df_filtered['layer_idx'].unique())
    ax.set_xticks(layers)
    ax.set_xticklabels([f"L{int(i)}" for i in layers])

    ax.legend(fontsize=11, loc="best", framealpha=0.9, title="Attention Type")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved confidence interval plot to {save_path}")

    return fig


def plot_layer_wise_attention_comparison(
    df: pd.DataFrame,
    gt_values: List[str],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 10),
    ci: Union[str, int] = 95,
):
    """
    Create subplot comparison of attention patterns for different GT values.

    Args:
        df: DataFrame with columns [sample_idx, layer_idx, image_pct, text_pct, bos_pct, GT]
        gt_values: List of GT values to compare (e.g., ["left", "right", "front", "behind"])
        save_path: Optional path to save figure
        figsize: Figure size
        ci: Confidence interval

    Example:
        plot_layer_wise_attention_comparison(
            df,
            gt_values=["left", "right", "front", "behind"],
            save_path='plots/gt_comparison.png'
        )
    """
    n_gts = len(gt_values)
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    for idx, gt_val in enumerate(gt_values):
        ax = axes[idx]

        # Filter data
        df_filtered = df[df['GT'] == gt_val].copy()

        if len(df_filtered) == 0:
            ax.text(0.5, 0.5, f"No data for GT={gt_val}",
                   ha='center', va='center', fontsize=12)
            ax.set_title(f"GT: {gt_val}")
            continue

        # Melt the dataframe
        df_melted = df_filtered.melt(
            id_vars=['sample_idx', 'layer_idx', 'GT'],
            value_vars=['image_pct', 'text_pct', 'bos_pct'],
            var_name='attention_type',
            value_name='percentage'
        )

        label_map = {
            'image_pct': 'Image',
            'text_pct': 'Text',
            'bos_pct': 'BOS/Other'
        }
        df_melted['attention_type'] = df_melted['attention_type'].map(label_map)

        # Plot
        sns.lineplot(
            data=df_melted,
            x='layer_idx',
            y='percentage',
            hue='attention_type',
            style='attention_type',
            markers=['o', 's', '^'],
            markersize=8,
            linewidth=2.5,
            ci=ci,
            ax=ax,
            alpha=0.8
        )

        # Customize
        n_samples = df_filtered['sample_idx'].nunique()
        ax.set_title(f"GT: {gt_val} ({n_samples} samples)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Layer", fontsize=10)
        ax.set_ylabel("Attention %", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)

        layers = sorted(df_filtered['layer_idx'].unique())
        ax.set_xticks(layers)
        ax.set_xticklabels([f"L{int(i)}" for i in layers])

        ax.legend(fontsize=9, loc="best", framealpha=0.9)

    # Hide unused subplots
    for idx in range(n_gts, len(axes)):
        axes[idx].axis('off')

    plt.suptitle(
        "Layer-wise Attention Comparison by Ground Truth",
        fontsize=16,
        fontweight="bold",
        y=0.995
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved comparison plot to {save_path}")

    return fig
