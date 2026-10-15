"""Resonance: a hybrid audio-feature recommendation engine for Spotify tracks."""

from __future__ import annotations

__version__ = "1.0.0"
__all__ = [
    "config",
    "data",
    "features",
    "stats",
    "RecommendationEngine",
    "Filters",
]

from . import config, data, features, stats  # noqa: E402
from .recommender import Filters, RecommendationEngine  # noqa: E402
