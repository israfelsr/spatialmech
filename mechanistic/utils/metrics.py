"""
Mechanistic Metrics

Metrics for evaluating mechanistic properties:
- Attention entropy
- Attribution sparsity
- Causal effect strength
"""

import torch
import numpy as np
from typing import Optional


def attention_entropy(attention_weights: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Calculate entropy of attention distribution.

    High entropy = uniform attention (less focused)
    Low entropy = peaked attention (more focused)

    Args:
        attention_weights: Attention weights tensor
        dim: Dimension to calculate entropy over

    Returns:
        Entropy values
    """
    # Add small epsilon for numerical stability
    eps = 1e-10
    attn = attention_weights + eps

    # Calculate entropy: -sum(p * log(p))
    entropy = -(attn * torch.log(attn)).sum(dim=dim)

    return entropy


def attention_sparsity(attention_weights: torch.Tensor, threshold: float = 0.1) -> float:
    """
    Calculate sparsity of attention (fraction of weights above threshold).

    Args:
        attention_weights: Attention weights tensor
        threshold: Threshold for considering a weight "active"

    Returns:
        Sparsity score (0 = all weights above threshold, 1 = no weights above threshold)
    """
    num_above_threshold = (attention_weights > threshold).float().mean().item()
    sparsity = 1.0 - num_above_threshold
    return sparsity


def attention_concentration(
    attention_weights: torch.Tensor,
    k: Optional[int] = None
) -> torch.Tensor:
    """
    Calculate concentration: sum of top-k attention weights.

    Args:
        attention_weights: Attention weights [batch, heads, seq, seq]
        k: Number of top weights to sum (default: 10% of sequence length)

    Returns:
        Concentration scores
    """
    if k is None:
        seq_len = attention_weights.shape[-1]
        k = max(1, seq_len // 10)

    # Get top-k weights
    topk_weights, _ = torch.topk(attention_weights, k, dim=-1)
    concentration = topk_weights.sum(dim=-1)

    return concentration


def causal_effect_strength(
    clean_output: torch.Tensor,
    patched_output: torch.Tensor,
    metric: str = "mse"
) -> float:
    """
    Calculate strength of causal effect from activation patching.

    Args:
        clean_output: Output from clean run
        patched_output: Output after patching
        metric: Distance metric ('mse', 'cosine', 'kl')

    Returns:
        Effect strength
    """
    if metric == "mse":
        effect = torch.nn.functional.mse_loss(patched_output, clean_output).item()
    elif metric == "cosine":
        similarity = torch.nn.functional.cosine_similarity(
            clean_output.flatten(),
            patched_output.flatten(),
            dim=0
        )
        effect = (1.0 - similarity).item()
    elif metric == "kl":
        # KL divergence (assuming log probabilities)
        clean_probs = torch.softmax(clean_output, dim=-1)
        patched_probs = torch.softmax(patched_output, dim=-1)
        effect = torch.nn.functional.kl_div(
            torch.log(patched_probs + 1e-10),
            clean_probs,
            reduction='batchmean'
        ).item()
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return effect


def attribution_sparsity(attribution_map: torch.Tensor, percentile: float = 90) -> float:
    """
    Calculate sparsity of attribution map.

    Args:
        attribution_map: Attribution values
        percentile: Percentile threshold for "important" attributions

    Returns:
        Sparsity ratio (percentage of attributions that are "important")
    """
    threshold = torch.quantile(attribution_map.abs(), percentile / 100.0)
    important_ratio = (attribution_map.abs() > threshold).float().mean().item()
    return important_ratio


__all__ = [
    'attention_entropy',
    'attention_sparsity',
    'attention_concentration',
    'causal_effect_strength',
    'attribution_sparsity'
]
