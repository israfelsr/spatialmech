"""
Two-Hop Attention Metrics

Metrics for analyzing the 2-hop attention hypothesis:
"Text tokens gather information from the image, and the last token
reads from those text tokens (rather than directly from the image)."

Usage:
    from mechanistic.utils.two_hop_metrics import (
        compute_all_two_hop_metrics,
        last_token_attention_distribution,
        text_tokens_attention_distribution,
    )

    # Assuming you have a sample dict with 'attentions' and 'input_ids'
    metrics = compute_all_two_hop_metrics(
        sample,
        vision_token_range=(start_idx, end_idx),
        text_token_range=(text_start_idx, text_end_idx),
        last_token_idx=-1
    )
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class TwoHopMetrics:
    """Container for all 2-hop attention metrics."""

    # Core metrics
    last_token_image_pct: float  # % attention from last token to image
    last_token_text_pct: float   # % attention from last token to text

    # Text tokens attention distribution
    text_tokens_image_pct_mean: float  # Average % attention from text to image
    text_tokens_image_pct_std: float
    text_tokens_text_pct_mean: float   # Average % attention from text to other text
    text_tokens_text_pct_std: float

    # 2-hop hypothesis metrics
    attention_flow_score: float  # Correlation between text→image and last→text
    information_bottleneck_score: float  # Which text tokens act as hubs
    hub_token_indices: List[int]  # Indices of hub tokens
    hub_token_strings: List[str]  # String representation of hub tokens

    # Direct vs indirect image access
    direct_image_attention: float  # Last token's direct attention to image
    indirect_image_attention: float  # Last token's indirect attention via text
    indirect_to_direct_ratio: float  # Ratio of indirect/direct

    # Layer-wise evolution (if multiple layers provided)
    layer_wise_last_token_image_pct: Optional[List[float]] = None
    layer_wise_text_tokens_image_pct: Optional[List[float]] = None

    # Positional analysis
    position_specific_image_attention: Optional[Dict[str, float]] = None

    # Additional context
    num_layers: int = 1
    num_image_tokens: int = 0
    num_text_tokens: int = 0
    num_total_tokens: int = 0


def identify_token_ranges(
    input_ids: torch.Tensor,
    tokenizer,
    vision_start_token: str = "<|vision_start|>",
    vision_end_token: str = "<|vision_end|>"
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Automatically identify vision and text token ranges.

    Args:
        input_ids: Tensor of token IDs [batch, seq_len] or [seq_len]
        tokenizer: Tokenizer to decode tokens
        vision_start_token: Special token marking start of vision tokens
        vision_end_token: Special token marking end of vision tokens

    Returns:
        vision_range: (start_idx, end_idx) for vision tokens (exclusive end)
        text_range: (start_idx, end_idx) for text tokens (exclusive end)
    """
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
    average_heads: bool = True
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
            - other_pct: % attention to other tokens
            - image_attn: Raw attention to image tokens
            - text_attn: Raw attention to text tokens
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
    image_pct = (image_attn / total_attn * 100) if total_attn > 0 else 0
    text_pct = (text_attn / total_attn * 100) if total_attn > 0 else 0
    other_pct = 100 - image_pct - text_pct

    return {
        "image_pct": image_pct,
        "text_pct": text_pct,
        "other_pct": other_pct,
        "image_attn": image_attn,
        "text_attn": text_attn,
    }


def text_tokens_attention_distribution(
    attention: torch.Tensor,
    vision_range: Tuple[int, int],
    text_range: Tuple[int, int],
    average_heads: bool = True,
    exclude_self_attention: bool = True
) -> Dict[str, any]:
    """
    Compute attention distribution for all text tokens.

    Args:
        attention: Attention tensor [batch, heads, seq, seq] or [heads, seq, seq] or [seq, seq]
        vision_range: (start, end) indices for vision tokens
        text_range: (start, end) indices for text tokens
        average_heads: Whether to average across attention heads
        exclude_self_attention: Whether to exclude self-attention when computing text→text

    Returns:
        Dictionary with:
            - image_pct_mean: Mean % attention from text tokens to image
            - image_pct_std: Std % attention from text tokens to image
            - text_pct_mean: Mean % attention from text tokens to other text
            - text_pct_std: Std % attention from text tokens to other text
            - per_token_image_pct: [num_text_tokens] array of image attention %
            - per_token_text_pct: [num_text_tokens] array of text attention %
    """
    # Handle different tensor shapes
    if attention.dim() == 4:
        attention = attention[0]  # Remove batch dim

    if attention.dim() == 3 and average_heads:
        attention = attention.mean(0)  # Average over heads: [seq, seq]

    vision_start, vision_end = vision_range
    text_start, text_end = text_range

    # Get attention from text tokens
    text_token_attn = attention[text_start:text_end]  # [num_text_tokens, seq]

    # Compute attention to image and text for each text token
    num_text_tokens = text_end - text_start
    per_token_image_pct = []
    per_token_text_pct = []

    for i in range(num_text_tokens):
        token_attn = text_token_attn[i]  # [seq]

        image_attn = token_attn[vision_start:vision_end].sum().item()

        if exclude_self_attention:
            # Exclude attention to self
            self_idx = text_start + i
            text_attn_tensor = token_attn[text_start:text_end].clone()
            text_attn_tensor[i] = 0  # Zero out self-attention
            text_attn = text_attn_tensor.sum().item()
        else:
            text_attn = token_attn[text_start:text_end].sum().item()

        total_attn = token_attn.sum().item()

        image_pct = (image_attn / total_attn * 100) if total_attn > 0 else 0
        text_pct = (text_attn / total_attn * 100) if total_attn > 0 else 0

        per_token_image_pct.append(image_pct)
        per_token_text_pct.append(text_pct)

    per_token_image_pct = np.array(per_token_image_pct)
    per_token_text_pct = np.array(per_token_text_pct)

    return {
        "image_pct_mean": per_token_image_pct.mean(),
        "image_pct_std": per_token_image_pct.std(),
        "text_pct_mean": per_token_text_pct.mean(),
        "text_pct_std": per_token_text_pct.std(),
        "per_token_image_pct": per_token_image_pct,
        "per_token_text_pct": per_token_text_pct,
    }


def compute_attention_flow_score(
    attention: torch.Tensor,
    vision_range: Tuple[int, int],
    text_range: Tuple[int, int],
    last_token_idx: int = -1,
    average_heads: bool = True
) -> float:
    """
    Compute correlation between (text→image) and (last→text).

    High correlation suggests 2-hop: text tokens that attend more to image
    are also attended more by the last token.

    Args:
        attention: Attention tensor
        vision_range: (start, end) indices for vision tokens
        text_range: (start, end) indices for text tokens
        last_token_idx: Index of last token
        average_heads: Whether to average across attention heads

    Returns:
        Pearson correlation coefficient between text→image and last→text
    """
    # Handle different tensor shapes
    if attention.dim() == 4:
        attention = attention[0]

    if attention.dim() == 3 and average_heads:
        attention = attention.mean(0)

    vision_start, vision_end = vision_range
    text_start, text_end = text_range

    # Get text→image attention for each text token
    text_to_image = attention[text_start:text_end, vision_start:vision_end].sum(dim=1)  # [num_text]

    # Get last→text attention
    last_to_text = attention[last_token_idx, text_start:text_end]  # [num_text]

    # Compute correlation
    if len(text_to_image) < 2:
        return 0.0

    # Convert to numpy for correlation
    text_to_image_np = text_to_image.cpu().numpy()
    last_to_text_np = last_to_text.cpu().numpy()

    # Pearson correlation
    correlation = np.corrcoef(text_to_image_np, last_to_text_np)[0, 1]

    # Handle NaN (constant arrays)
    if np.isnan(correlation):
        return 0.0

    return float(correlation)


def compute_information_bottleneck_score(
    attention: torch.Tensor,
    vision_range: Tuple[int, int],
    text_range: Tuple[int, int],
    last_token_idx: int = -1,
    average_heads: bool = True,
    top_k: int = 3
) -> Tuple[float, List[int]]:
    """
    Identify which text tokens act as "information hubs".

    Hub tokens have high attention FROM image AND high attention FROM last token.

    Args:
        attention: Attention tensor
        vision_range: (start, end) indices for vision tokens
        text_range: (start, end) indices for text tokens
        last_token_idx: Index of last token
        average_heads: Whether to average across attention heads
        top_k: Number of top hub tokens to identify

    Returns:
        hub_score: Overall hub score (product of normalized attentions)
        hub_indices: Indices of top-k hub tokens (relative to text_start)
    """
    # Handle different tensor shapes
    if attention.dim() == 4:
        attention = attention[0]

    if attention.dim() == 3 and average_heads:
        attention = attention.mean(0)

    vision_start, vision_end = vision_range
    text_start, text_end = text_range

    # Get text→image attention (how much each text token attends to image)
    text_to_image = attention[text_start:text_end, vision_start:vision_end].sum(dim=1)  # [num_text]

    # Get last→text attention (how much last token attends to each text token)
    last_to_text = attention[last_token_idx, text_start:text_end]  # [num_text]

    # Normalize both to [0, 1]
    text_to_image_norm = (text_to_image - text_to_image.min()) / (text_to_image.max() - text_to_image.min() + 1e-8)
    last_to_text_norm = (last_to_text - last_to_text.min()) / (last_to_text.max() - last_to_text.min() + 1e-8)

    # Hub score: product of both (high when both are high)
    hub_scores = text_to_image_norm * last_to_text_norm

    # Get top-k hub tokens
    top_k = min(top_k, len(hub_scores))
    top_k_values, top_k_indices = torch.topk(hub_scores, top_k)

    # Overall hub score (mean of top-k)
    overall_hub_score = top_k_values.mean().item()

    hub_indices = top_k_indices.cpu().tolist()

    return overall_hub_score, hub_indices


def compute_indirect_image_attention(
    attention: torch.Tensor,
    vision_range: Tuple[int, int],
    text_range: Tuple[int, int],
    last_token_idx: int = -1,
    average_heads: bool = True
) -> Dict[str, float]:
    """
    Compare direct vs indirect image attention from last token.

    Direct: last→image attention
    Indirect: Σ(last→text[i] × text[i]→image) - weighted sum through text tokens

    Args:
        attention: Attention tensor
        vision_range: (start, end) indices for vision tokens
        text_range: (start, end) indices for text tokens
        last_token_idx: Index of last token
        average_heads: Whether to average across attention heads

    Returns:
        Dictionary with:
            - direct: Direct attention to image
            - indirect: Indirect attention via text tokens
            - ratio: indirect/direct ratio
    """
    # Handle different tensor shapes
    if attention.dim() == 4:
        attention = attention[0]

    if attention.dim() == 3 and average_heads:
        attention = attention.mean(0)

    vision_start, vision_end = vision_range
    text_start, text_end = text_range

    # Direct attention: last→image
    direct_attn = attention[last_token_idx, vision_start:vision_end].sum().item()

    # Indirect attention: Σ(last→text[i] × text[i]→image)
    last_to_text = attention[last_token_idx, text_start:text_end]  # [num_text]
    text_to_image = attention[text_start:text_end, vision_start:vision_end].sum(dim=1)  # [num_text]

    indirect_attn = (last_to_text * text_to_image).sum().item()

    # Compute ratio
    ratio = indirect_attn / (direct_attn + 1e-8)

    return {
        "direct": direct_attn,
        "indirect": indirect_attn,
        "ratio": ratio,
    }


def compute_positional_analysis(
    attention: torch.Tensor,
    vision_range: Tuple[int, int],
    text_range: Tuple[int, int],
    token_strings: List[str],
    keywords: List[str] = ["left", "right", "front", "behind", "where"],
    average_heads: bool = True
) -> Dict[str, float]:
    """
    Analyze which token positions (e.g., object names, spatial words) attend most to image.

    Args:
        attention: Attention tensor
        vision_range: (start, end) indices for vision tokens
        text_range: (start, end) indices for text tokens
        token_strings: List of decoded token strings for text tokens
        keywords: Keywords to identify (e.g., spatial relation words)
        average_heads: Whether to average across attention heads

    Returns:
        Dictionary mapping keyword to average image attention %
    """
    # Handle different tensor shapes
    if attention.dim() == 4:
        attention = attention[0]

    if attention.dim() == 3 and average_heads:
        attention = attention.mean(0)

    vision_start, vision_end = vision_range
    text_start, text_end = text_range

    # Get text→image attention
    text_to_image = attention[text_start:text_end, vision_start:vision_end].sum(dim=1)  # [num_text]

    # Normalize to percentages
    text_total_attn = attention[text_start:text_end].sum(dim=1)
    text_to_image_pct = (text_to_image / (text_total_attn + 1e-8) * 100).cpu().numpy()

    # Find keywords in token strings
    keyword_attention = {}
    for keyword in keywords:
        keyword_lower = keyword.lower()
        matching_indices = [i for i, token in enumerate(token_strings) if keyword_lower in token.lower()]

        if matching_indices:
            keyword_attention[keyword] = float(text_to_image_pct[matching_indices].mean())
        else:
            keyword_attention[keyword] = 0.0

    return keyword_attention


def compute_layer_wise_evolution(
    attentions: Tuple[torch.Tensor],
    vision_range: Tuple[int, int],
    text_range: Tuple[int, int],
    last_token_idx: int = -1,
    average_heads: bool = True
) -> Dict[str, List[float]]:
    """
    Track how attention patterns evolve across layers.

    Args:
        attentions: Tuple of attention tensors, one per layer
        vision_range: (start, end) indices for vision tokens
        text_range: (start, end) indices for text tokens
        last_token_idx: Index of last token
        average_heads: Whether to average across attention heads

    Returns:
        Dictionary with:
            - last_token_image_pct: List of % per layer
            - text_tokens_image_pct: List of mean % per layer
    """
    last_token_image_pcts = []
    text_tokens_image_pcts = []

    for layer_attn in attentions:
        # Last token distribution
        last_dist = last_token_attention_distribution(
            layer_attn, vision_range, text_range, last_token_idx, average_heads
        )
        last_token_image_pcts.append(last_dist["image_pct"])

        # Text tokens distribution
        text_dist = text_tokens_attention_distribution(
            layer_attn, vision_range, text_range, average_heads
        )
        text_tokens_image_pcts.append(text_dist["image_pct_mean"])

    return {
        "last_token_image_pct": last_token_image_pcts,
        "text_tokens_image_pct": text_tokens_image_pcts,
    }


def compute_all_two_hop_metrics(
    sample: Dict,
    tokenizer=None,
    vision_range: Optional[Tuple[int, int]] = None,
    text_range: Optional[Tuple[int, int]] = None,
    last_token_idx: int = -1,
    layer_idx: int = -1,
    average_heads: bool = True,
    analyze_all_layers: bool = False,
    keywords: List[str] = ["left", "right", "front", "behind", "where"]
) -> TwoHopMetrics:
    """
    Compute all 2-hop attention metrics for a sample.

    Args:
        sample: Dictionary containing 'attentions' and 'input_ids'
        tokenizer: Tokenizer for decoding tokens (required if vision_range not provided)
        vision_range: Optional (start, end) for vision tokens
        text_range: Optional (start, end) for text tokens
        last_token_idx: Index of last token (-1 for last)
        layer_idx: Which layer to analyze (-1 for last layer)
        average_heads: Whether to average across attention heads
        analyze_all_layers: Whether to compute layer-wise evolution
        keywords: Keywords for positional analysis

    Returns:
        TwoHopMetrics object with all computed metrics
    """
    # Get attention from specified layer
    attentions = sample["attentions"]
    if isinstance(attentions, tuple):
        attention = attentions[layer_idx]
    else:
        attention = attentions

    # Auto-detect token ranges if not provided
    if vision_range is None or text_range is None:
        if tokenizer is None:
            raise ValueError("Must provide either vision_range/text_range or tokenizer")
        vision_range, text_range = identify_token_ranges(sample["input_ids"], tokenizer)

    vision_start, vision_end = vision_range
    text_start, text_end = text_range

    num_image_tokens = vision_end - vision_start
    num_text_tokens = text_end - text_start

    # Decode text tokens for positional analysis
    if tokenizer is not None:
        input_ids = sample["input_ids"]
        if input_ids.dim() == 2:
            input_ids = input_ids[0]
        text_token_ids = input_ids[text_start:text_end]
        token_strings = [tokenizer.decode(t) for t in text_token_ids]
    else:
        token_strings = [f"token_{i}" for i in range(num_text_tokens)]

    # 1. Last token attention distribution
    last_token_dist = last_token_attention_distribution(
        attention, vision_range, text_range, last_token_idx, average_heads
    )

    # 2. Text tokens attention distribution
    text_tokens_dist = text_tokens_attention_distribution(
        attention, vision_range, text_range, average_heads
    )

    # 3. Attention flow score
    flow_score = compute_attention_flow_score(
        attention, vision_range, text_range, last_token_idx, average_heads
    )

    # 4. Information bottleneck (hub tokens)
    hub_score, hub_indices = compute_information_bottleneck_score(
        attention, vision_range, text_range, last_token_idx, average_heads
    )
    hub_token_strings = [token_strings[i] for i in hub_indices]

    # 5. Direct vs indirect image attention
    indirect_metrics = compute_indirect_image_attention(
        attention, vision_range, text_range, last_token_idx, average_heads
    )

    # 6. Positional analysis
    positional_metrics = compute_positional_analysis(
        attention, vision_range, text_range, token_strings, keywords, average_heads
    )

    # 7. Layer-wise evolution (if requested)
    layer_wise = None
    num_layers = 1
    if analyze_all_layers and isinstance(attentions, tuple):
        layer_wise = compute_layer_wise_evolution(
            attentions, vision_range, text_range, last_token_idx, average_heads
        )
        num_layers = len(attentions)

    # Get sequence length
    if attention.dim() == 4:
        num_total_tokens = attention.shape[-1]
    elif attention.dim() == 3:
        num_total_tokens = attention.shape[-1]
    else:
        num_total_tokens = len(attention)

    return TwoHopMetrics(
        # Core metrics
        last_token_image_pct=last_token_dist["image_pct"],
        last_token_text_pct=last_token_dist["text_pct"],

        # Text tokens distribution
        text_tokens_image_pct_mean=text_tokens_dist["image_pct_mean"],
        text_tokens_image_pct_std=text_tokens_dist["image_pct_std"],
        text_tokens_text_pct_mean=text_tokens_dist["text_pct_mean"],
        text_tokens_text_pct_std=text_tokens_dist["text_pct_std"],

        # 2-hop hypothesis metrics
        attention_flow_score=flow_score,
        information_bottleneck_score=hub_score,
        hub_token_indices=hub_indices,
        hub_token_strings=hub_token_strings,

        # Direct vs indirect
        direct_image_attention=indirect_metrics["direct"],
        indirect_image_attention=indirect_metrics["indirect"],
        indirect_to_direct_ratio=indirect_metrics["ratio"],

        # Layer-wise (if computed)
        layer_wise_last_token_image_pct=layer_wise["last_token_image_pct"] if layer_wise else None,
        layer_wise_text_tokens_image_pct=layer_wise["text_tokens_image_pct"] if layer_wise else None,

        # Positional
        position_specific_image_attention=positional_metrics,

        # Context
        num_layers=num_layers,
        num_image_tokens=num_image_tokens,
        num_text_tokens=num_text_tokens,
        num_total_tokens=num_total_tokens,
    )


def print_metrics_summary(metrics: TwoHopMetrics, verbose: bool = True):
    """
    Pretty print the metrics summary.

    Args:
        metrics: TwoHopMetrics object
        verbose: Whether to print detailed information
    """
    print("=" * 80)
    print("TWO-HOP ATTENTION METRICS SUMMARY")
    print("=" * 80)

    print("\n📊 CORE METRICS")
    print(f"  Last token attention:")
    print(f"    → Image: {metrics.last_token_image_pct:.2f}%")
    print(f"    → Text:  {metrics.last_token_text_pct:.2f}%")
    print(f"  ")
    print(f"  Text tokens attention (mean ± std):")
    print(f"    → Image: {metrics.text_tokens_image_pct_mean:.2f}% ± {metrics.text_tokens_image_pct_std:.2f}%")
    print(f"    → Text:  {metrics.text_tokens_text_pct_mean:.2f}% ± {metrics.text_tokens_text_pct_std:.2f}%")

    print("\n🔗 TWO-HOP HYPOTHESIS METRICS")
    print(f"  Attention flow score: {metrics.attention_flow_score:.3f}")
    print(f"    (Correlation between text→image and last→text)")
    print(f"    Higher = stronger 2-hop pattern")
    print(f"  ")
    print(f"  Information bottleneck score: {metrics.information_bottleneck_score:.3f}")
    print(f"    (Strength of hub tokens)")
    print(f"    Hub tokens (top-{len(metrics.hub_token_indices)}): {metrics.hub_token_strings}")

    print("\n🔄 DIRECT VS INDIRECT IMAGE ACCESS")
    print(f"  Direct (last→image):   {metrics.direct_image_attention:.4f}")
    print(f"  Indirect (last→text→image): {metrics.indirect_image_attention:.4f}")
    print(f"  Indirect/Direct ratio: {metrics.indirect_to_direct_ratio:.2f}x")
    print(f"    Ratio > 1 suggests indirect path is stronger (supports 2-hop)")

    if verbose and metrics.position_specific_image_attention:
        print("\n📍 POSITIONAL ANALYSIS (keyword → image attention)")
        for keyword, attn_pct in metrics.position_specific_image_attention.items():
            print(f"  '{keyword}': {attn_pct:.2f}%")

    if verbose and metrics.layer_wise_last_token_image_pct is not None:
        print("\n🔬 LAYER-WISE EVOLUTION")
        print(f"  Analyzed {metrics.num_layers} layers")
        print(f"  Last token image attention per layer:")
        for i, pct in enumerate(metrics.layer_wise_last_token_image_pct):
            print(f"    Layer {i:2d}: {pct:.2f}%")

    print("\n📋 CONTEXT")
    print(f"  Total tokens: {metrics.num_total_tokens}")
    print(f"  Image tokens: {metrics.num_image_tokens}")
    print(f"  Text tokens:  {metrics.num_text_tokens}")
    print(f"  Layers analyzed: {metrics.num_layers}")
    print("=" * 80)


__all__ = [
    "TwoHopMetrics",
    "identify_token_ranges",
    "last_token_attention_distribution",
    "text_tokens_attention_distribution",
    "compute_attention_flow_score",
    "compute_information_bottleneck_score",
    "compute_indirect_image_attention",
    "compute_positional_analysis",
    "compute_layer_wise_evolution",
    "compute_all_two_hop_metrics",
    "print_metrics_summary",
]
