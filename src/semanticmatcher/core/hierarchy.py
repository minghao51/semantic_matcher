"""
Hierarchical entity matching with multi-parent support.

This module provides:
- HierarchyIndex: Graph-based hierarchy representation
- HierarchicalScoring: Depth-aware confidence scoring
- HierarchicalMatcher: User-facing API for hierarchical matching
"""

from typing import Dict, List, Optional, Tuple, Any
import networkx as nx
import numpy as np
from functools import lru_cache


__all__ = [
    "HierarchyIndex",
    "HierarchicalScoring",
    "HierarchicalMatcher",
]
