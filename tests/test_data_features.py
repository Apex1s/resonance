"""Data loading, validation and feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from resonance import config, data, stats
from resonance.data import SchemaError
from resonance.features import EmbeddingSpace, build_preprocessor, engineer, feature_matrix


def test_load_tracks_shape_and_ids(tracks):
    assert len(tracks) > 1_500
    assert tracks[config.ID_COLUMN].is_unique
    assert set(config.AUDIO_FEATURES) <= set(tracks.columns)
    assert tracks.isna().sum().sum() == 0


def test_track_id_is_deterministic():
    first = data.load_tracks()
    second = data.load_tracks()
    pd.testing.assert_series_equal(first[config.ID_COLUMN], second[config.ID_COLUMN])


def test_schema_error_on_missing_column(tmp_path, tracks):
    broken = tmp_path / "broken.csv"
    tracks.drop(columns=["valence"]).to_csv(broken, index=False)
    with pytest.raises(SchemaError, match="valence"):
        data.load_tracks(broken)


def test_schema_error_on_non_binary_target(tmp_path, tracks):
    broken = tmp_path / "broken.csv"
    bad = tracks.copy()
    bad.loc[bad.index[0], "target"] = 7
    bad.to_csv(broken, index=False)
    with pytest.raises(SchemaError, match="binary"):
        data.load_tracks(broken)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_tracks(tmp_path / "nope.csv")


def test_engineer_adds_every_declared_feature(synthetic_tracks):
    out = engineer(synthetic_tracks)
    assert set(config.ENGINEERED_FEATURES) <= set(out.columns)
    assert out["duration_min"].between(1, 7).all()
    assert out["loudness_norm"].between(0, 1).all()
    assert set(out["is_acoustic"].unique()) <= {0, 1}


def test_engineer_is_stateless_on_one_row(synthetic_tracks):
    """Inference path: a single track must produce identical derived values."""
    full = engineer(synthetic_tracks)
    single = engineer(synthetic_tracks.head(1))
    for col in config.ENGINEERED_FEATURES:
        assert single[col].iloc[0] == pytest.approx(full[col].iloc[0])


def test_preprocessor_output_is_finite(synthetic_tracks):
    matrix = build_preprocessor().fit_transform(feature_matrix(synthetic_tracks))
    assert matrix.shape[0] == len(synthetic_tracks)
    assert np.isfinite(matrix).all()


def test_embedding_vectors_are_unit_norm(synthetic_tracks):
    space = EmbeddingSpace()
    vectors = space.fit_transform(engineer(synthetic_tracks))
    assert vectors.shape == (len(synthetic_tracks), len(config.SIMILARITY_WEIGHTS))
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_embedding_state_roundtrip(synthetic_tracks):
    frame = engineer(synthetic_tracks)
    space = EmbeddingSpace().fit(frame)
    restored = EmbeddingSpace.from_state(space.state_dict())
    np.testing.assert_allclose(space.transform(frame), restored.transform(frame), atol=1e-6)


def test_embedding_rejects_unfitted_use(synthetic_tracks):
    with pytest.raises(RuntimeError):
        EmbeddingSpace().transform(engineer(synthetic_tracks))


def test_anova_reports_effect_sizes(tracks):
    table = stats.anova(tracks)
    assert list(table.columns) == [
        "feature", "f_statistic", "p_value", "cohens_d",
        "mean_disliked", "mean_liked", "significant",
    ]
    assert table["p_value"].is_monotonic_increasing
    # The study's headline result: danceability separates the two classes.
    assert table.loc[table["feature"] == "danceability", "significant"].item()


def test_benjamini_hochberg_controls_fdr():
    flags = stats.benjamini_hochberg([1e-9, 0.02, 0.4, 0.9])
    assert flags[0] is True
    assert flags[-1] is False


def test_describe_matches_frame(tracks):
    summary = data.describe(tracks)
    assert summary["n_tracks"] == len(tracks)
    assert 0 < summary["like_rate"] < 1
