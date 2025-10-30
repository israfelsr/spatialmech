import torch
import numpy as np
from typing import Optional
from typing import Dict, Tuple, Optional, List


def identify_token_ranges(
    input_ids: torch.Tensor,
    tokenizer,
    vision_start_token: str = "<|vision_start|>",
    vision_end_token: str = "<|vision_end|>",
) -> Tuple[Tuple[int, int], Tuple[int, int]]:

    if input_ids.dim() == 2:
        input_ids = input_ids[0]

    tokens = [tokenizer.decode(t) for t in input_ids]

    vision_start_idx = None
    vision_end_idx = None

    for i, token in enumerate(tokens):
        if vision_start_token in token:
            vision_start_idx = i + 1  # Start after the marker
        if vision_end_token in token:
            vision_end_idx = i  # End before the marker
            break

    if vision_start_idx is None or vision_end_idx is None:
        raise ValueError(f"Could not find vision tokens. Tokens: {tokens[:20]}...")

    # Text tokens start after vision_end marker
    text_start_idx = vision_end_idx + 1
    text_end_idx = len(tokens)

    return (vision_start_idx, vision_end_idx), (text_start_idx, text_end_idx)


def last_token_attention_distribution(
    attention: torch.Tensor,
    vision_range: Tuple[int, int],
    text_range: Tuple[int, int],
    last_token_idx: int = -1,
    average_heads: bool = True,
) -> Dict[str, float]:
    """
    Compute attention distribution for the last token.

    Args:
        attention: Attention tensor [batch, heads, seq, seq] or [heads, seq, seq] or [seq, seq]
        vision_range: (start, end) indices for vision tokens
        text_range: (start, end) indices for text tokens
        last_token_idx: Index of last token (-1 for last, -2 for second-to-last, etc.)
        average_heads: Whether to average across attention heads

    Returns:
        Dictionary with:
            - image_pct: % attention to image tokens
            - text_pct: % attention to text tokens
            - bos_pct: % attention to other tokens
    """
    # Handle different tensor shapes
    if attention.dim() == 4:
        attention = attention[0]  # Remove batch dim

    if attention.dim() == 3 and average_heads:
        attention = attention.mean(0)  # Average over heads: [seq, seq]

    # Get attention from last token
    last_token_attn = attention[last_token_idx]  # [seq]

    vision_start, vision_end = vision_range
    text_start, text_end = text_range

    # Sum attention to different regions
    image_attn = last_token_attn[vision_start:vision_end].sum().item()
    text_attn = last_token_attn[text_start:text_end].sum().item()
    total_attn = last_token_attn.sum().item()

    # Compute percentages
    image_pct = image_attn * 100
    text_pct = text_attn * 100
    bos_pct = 100 - image_pct - text_pct

    return {
        "image_pct": image_pct,
        "text_pct": text_pct,
        "bos_pct": bos_pct,
    }


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


def attention_sparsity(
    attention_weights: torch.Tensor, threshold: float = 0.1
) -> float:
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
    attention_weights: torch.Tensor, k: Optional[int] = None
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
    clean_output: torch.Tensor, patched_output: torch.Tensor, metric: str = "mse"
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
            clean_output.flatten(), patched_output.flatten(), dim=0
        )
        effect = (1.0 - similarity).item()
    elif metric == "kl":
        # KL divergence (assuming log probabilities)
        clean_probs = torch.softmax(clean_output, dim=-1)
        patched_probs = torch.softmax(patched_output, dim=-1)
        effect = torch.nn.functional.kl_div(
            torch.log(patched_probs + 1e-10), clean_probs, reduction="batchmean"
        ).item()
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return effect


def attribution_sparsity(
    attribution_map: torch.Tensor, percentile: float = 90
) -> float:
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
    "attention_entropy",
    "attention_sparsity",
    "attention_concentration",
    "causal_effect_strength",
    "attribution_sparsity",
]
