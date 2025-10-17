"""
Mechanistic Interpretability Module

This module provides tools for mechanistic interpretability experiments
on vision-language models for spatial reasoning tasks.

Submodules:
- attention: Attention extraction, visualization, and analysis
- attribution: Attribution methods (cross-attention, gradients, etc.)
- patching: Activation patching and causal interventions
- probing: Probing classifiers and representation analysis
- utils: Shared utilities (hooks, visualization, metrics)
"""

from .attention import *
from .attribution import *
from .patching import *
from .probing import *
from .utils import *

__version__ = "0.1.0"
