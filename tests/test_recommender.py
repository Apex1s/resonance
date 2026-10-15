"""Retrieval behaviour: neighbours, blending, filters, diversity, explanations."""

from __future__ import annotations

import numpy as np
import pytest

from resonance import config
from resonance.features import EmbeddingSpace, engineer
from resonance.recommender import Filters, RecommendationEngine


@pytest.fixture
def small_engine(synthetic_tracks) -> RecommendationEngine:
    frame = synthetic_tracks.copy()
    frame["p_like"] = np.linspace(0.05, 0.95, len(frame))
    space = EmbeddingSpace()
    vectors = space.fit_transform(engineer(frame))
    return RecommendationEngine(frame, vectors, space, model=None, model_name="test")


def test_similar_excludes_the_seed(small_engine):
    seed = small_engine.catalog[config.ID_COLUMN].iloc[0]
    hits = small_engine.similar_tracks(seed, k=5)
    assert len(hits) == 5
    assert seed not in set(hits[config.ID_COLUMN])


def test_similarity_is_ordered(small_engine):
    seed = small_engine.catalog[config.ID_COLUMN].iloc[3]
    hits = small_engine.similar_tracks(seed, k=8)
    assert hits["similarity"].is_monotonic_decreasing


def test_taste_weight_one_returns_the_most_likeable(small_engine):
    seed = small_engine.catalog[config.ID_COLUMN].iloc[0]
    hits = small_engine.recommend([seed], k=3, taste_weight=1.0, diversity=0.0)
    assert hits["p_like"].iloc[0] == small_engine.catalog["p_like"].max()


def test_taste_weight_changes_the_ranking(small_engine):
    seed = small_engine.catalog[config.ID_COLUMN].iloc[0]
    pure = small_engine.recommend([seed], k=6, taste_weight=0.0, diversity=0.0)
    taste = small_engine.recommend([seed], k=6, taste_weight=1.0, diversity=0.0)
    assert list(pure[config.ID_COLUMN]) != list(taste[config.ID_COLUMN])


def test_diversity_spreads_the_playlist(small_engine):
    seed = small_engine.catalog[config.ID_COLUMN].iloc[0]
    tight = small_engine.recommend([seed], k=8, diversity=0.0)
    loose = small_engine.recommend([seed], k=8, diversity=0.8)
    assert list(tight[config.ID_COLUMN]) != list(loose[config.ID_COLUMN])
    # A spread playlist sits further from the seed on average.
    assert loose["similarity"].mean() < tight["similarity"].mean()


def test_multiple_seeds_are_all_excluded(small_engine):
    seeds = list(small_engine.catalog[config.ID_COLUMN].iloc[:3])
    hits = small_engine.recommend(seeds, k=5)
    assert not set(seeds) & set(hits[config.ID_COLUMN])


def test_filters_are_hard_constraints(small_engine):
    seed = small_engine.catalog[config.ID_COLUMN].iloc[0]
    hits = small_engine.recommend([seed], k=10, filters=Filters(tempo_min=120, energy_min=0.5))
    assert (hits["tempo"] >= 120).all()
    assert (hits["energy"] >= 0.5).all()


def test_impossible_filters_return_empty_not_error(small_engine):
    seed = small_engine.catalog[config.ID_COLUMN].iloc[0]
    hits = small_engine.recommend([seed], k=5, filters=Filters(tempo_min=249, energy_min=0.99))
    assert len(hits) == 0


def test_exclude_seed_artists(small_engine):
    seed_row = small_engine.catalog.iloc[0]
    hits = small_engine.recommend(
        [seed_row[config.ID_COLUMN]], k=6, exclude_seed_artists=True
    )
    assert seed_row["artist"] not in set(hits["artist"])


def test_unknown_seed_raises(small_engine):
    with pytest.raises(KeyError):
        small_engine.recommend(["does-not-exist"], k=3)


def test_profile_search_respects_the_profile(small_engine):
    loud = small_engine.recommend_profile({"energy": 0.99, "loudness": -2.0}, k=5, taste_weight=0.0)
    quiet = small_engine.recommend_profile({"energy": 0.01, "loudness": -25.0}, k=5, taste_weight=0.0)
    assert loud["energy"].mean() > quiet["energy"].mean()


def test_explain_returns_matches_and_a_contrast(small_engine):
    seed = small_engine.catalog[config.ID_COLUMN].iloc[0]
    hit = small_engine.recommend([seed], k=1).iloc[0]
    reasons = small_engine.explain([seed], hit[config.ID_COLUMN])
    assert reasons
    assert {r["role"] for r in reasons} <= {"match", "contrast"}
    assert all(0 <= r["agreement"] <= 1.5 for r in reasons)


def test_search_matches_title_and_artist(small_engine):
    assert len(small_engine.search("Track 1")) > 0
    assert len(small_engine.search("Artist 3")) > 0
    assert len(small_engine.search("zzzz")) == 0


# --------------------------------------------------------------------------- #
# Against the real trained artefacts
# --------------------------------------------------------------------------- #
def test_real_engine_is_sane(engine):
    assert len(engine) > 1_500
    seed = engine.catalog[config.ID_COLUMN].iloc[0]
    hits = engine.recommend([seed], k=10, taste_weight=0.4, diversity=0.3)
    assert len(hits) == 10
    assert hits["score"].between(0, 1).all()


def test_real_model_scores_a_profile(engine):
    if engine.model is None:
        pytest.skip("taste model artefact not present")
    p = engine.predict_like({"danceability": 0.9, "energy": 0.8, "speechiness": 0.4})
    assert 0.0 <= p <= 1.0
