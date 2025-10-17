"""
PyTorch Hook Utilities for Mechanistic Interpretability

Provides utilities for registering forward hooks to capture:
- Attention weights
- Activations
- Intermediate representations
"""

import torch
import torch.nn as nn
from typing import Dict, List, Callable, Optional
from collections import defaultdict


class ActivationCache:
    """
    Cache for storing activations from forward hooks.

    Usage:
        cache = ActivationCache()
        hook = model.layer.register_forward_hook(cache.save_hook('layer_name'))
        output = model(input)
        activations = cache.get('layer_name')
    """

    def __init__(self):
        self.activations = defaultdict(list)
        self.hooks = []

    def save_hook(self, name: str) -> Callable:
        """Create a hook function that saves activations to cache."""
        def hook(module, input, output):
            if isinstance(output, tuple):
                # Handle tuple outputs (e.g., attention weights + values)
                self.activations[name].append(tuple(o.detach().cpu() if isinstance(o, torch.Tensor) else o for o in output))
            else:
                self.activations[name].append(output.detach().cpu())
        return hook

    def get(self, name: str) -> List:
        """Retrieve cached activations for a layer."""
        return self.activations[name]

    def clear(self):
        """Clear all cached activations."""
        self.activations.clear()

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


class AttentionCache:
    """
    Specialized cache for capturing attention weights.

    Usage:
        cache = AttentionCache()
        hooks = cache.register_attention_hooks(model, layer_names=['layer.0', 'layer.1'])
        output = model(input)
        attn_weights = cache.get_attention('layer.0')
    """

    def __init__(self):
        self.attention_weights = defaultdict(list)
        self.hooks = []

    def attention_hook(self, name: str) -> Callable:
        """Create a hook that extracts attention weights from attention modules."""
        def hook(module, input, output):
            # Handle different attention output formats
            if isinstance(output, tuple):
                # Usually (attention_output, attention_weights)
                if len(output) >= 2:
                    attn = output[1]  # Attention weights
                    if isinstance(attn, torch.Tensor):
                        self.attention_weights[name].append(attn.detach().cpu())
            elif hasattr(output, 'attentions'):
                # Handle HuggingFace style outputs
                self.attention_weights[name].append(output.attentions.detach().cpu())
        return hook

    def register_attention_hooks(self, model: nn.Module, layer_names: Optional[List[str]] = None):
        """
        Register hooks on attention layers.

        Args:
            model: The model to hook
            layer_names: List of layer names to hook (e.g., ['encoder.layer.0.attention'])
                        If None, attempts to auto-detect attention layers
        """
        if layer_names is None:
            # Auto-detect attention layers
            layer_names = self._find_attention_layers(model)

        for name in layer_names:
            module = self._get_module_by_name(model, name)
            if module is not None:
                hook = module.register_forward_hook(self.attention_hook(name))
                self.hooks.append(hook)

        return self.hooks

    def _find_attention_layers(self, model: nn.Module) -> List[str]:
        """Auto-detect attention layers by name patterns."""
        attention_layers = []
        for name, module in model.named_modules():
            # Common attention layer patterns
            if any(pattern in name.lower() for pattern in ['attention', 'attn', 'self_attn', 'cross_attn']):
                attention_layers.append(name)
        return attention_layers

    def _get_module_by_name(self, model: nn.Module, name: str):
        """Get a module by its dotted name."""
        try:
            for part in name.split('.'):
                model = getattr(model, part)
            return model
        except AttributeError:
            print(f"Warning: Could not find module {name}")
            return None

    def get_attention(self, name: str) -> List[torch.Tensor]:
        """Retrieve cached attention weights for a layer."""
        return self.attention_weights[name]

    def clear(self):
        """Clear all cached attention weights."""
        self.attention_weights.clear()

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


def register_forward_hook_with_cache(
    model: nn.Module,
    layer_names: List[str],
    cache_type: str = "activation"
) -> ActivationCache:
    """
    Convenience function to register forward hooks and return a cache.

    Args:
        model: Model to hook
        layer_names: List of layer names to hook
        cache_type: Type of cache ('activation' or 'attention')

    Returns:
        Cache object with registered hooks
    """
    if cache_type == "attention":
        cache = AttentionCache()
        cache.register_attention_hooks(model, layer_names)
    else:
        cache = ActivationCache()
        for name in layer_names:
            module = cache._get_module_by_name(model, name) if hasattr(cache, '_get_module_by_name') else None
            if module is None:
                # Simple name lookup
                for n, m in model.named_modules():
                    if n == name:
                        module = m
                        break
            if module is not None:
                hook = module.register_forward_hook(cache.save_hook(name))
                cache.hooks.append(hook)

    return cache


__all__ = [
    'ActivationCache',
    'AttentionCache',
    'register_forward_hook_with_cache'
]
