"""Shared fixtures.

Tests that need trained artefacts are skipped (not failed) when the repo has
not been trained yet, so a fresh clone can still run the unit suite.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from resonance import config, data
from resonance.recommender import ArtifactsMissing, RecommendationEngine

ARTIFACTS_READY = config.CATALOG_PATH.exists() and config.EMBEDDING_PATH.exists()
requires_artifacts = pytest.mark.skipif(
    not ARTIFACTS_READY, reason="run `python -m resonance.train` to build model artefacts"
)


@pytest.fixture(scope="session")
def tracks() -> pd.DataFrame:
    return data.load_tracks()


@pytest.fixture(scope="session")
def engine() -> RecommendationEngine:
    try:
        return RecommendationEngine.load()
    except ArtifactsMissing as exc:  # pragma: no cover - CI without artefacts
        pytest.skip(str(exc))


@pytest.fixture
def synthetic_tracks() -> pd.DataFrame:
    """Small, schema-valid frame for tests that must not touch disk."""
    rng = np.random.default_rng(0)
    n = 60
    frame = pd.DataFrame(
        {
            "acousticness": rng.random(n),
            "danceability": rng.random(n),
            "duration_ms": rng.integers(90_000, 400_000, n),
            "energy": rng.random(n),
            "instrumentalness": rng.random(n) ** 3,
            "key": rng.integers(0, 12, n),
            "liveness": rng.random(n) ** 2,
            "loudness": -rng.random(n) * 20,
            "mode": rng.integers(0, 2, n),
            "speechiness": rng.random(n) ** 2,
            "tempo": rng.uniform(60, 190, n),
            "time_signature": np.full(n, 4),
            "valence": rng.random(n),
            "target": rng.integers(0, 2, n),
            "song_title": [f"Track {i}" for i in range(n)],
            "artist": [f"Artist {i % 7}" for i in range(n)],
        }
    )
    frame[config.ID_COLUMN] = [f"id{i:04d}" for i in range(n)]
    return frame
