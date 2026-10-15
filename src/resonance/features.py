"""Feature engineering and the shared preprocessing pipeline.

Two artefacts are produced from the same raw columns:

* ``build_preprocessor()``: the supervised pipeline feeding the taste model.
* ``build_embeddings()``:  a weighted, L2-normalised vector space used for
  content-based similarity search.

Keeping both in one module guarantees that a track is described the same way
whether it is being scored or being retrieved.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)

from . import config


# --------------------------------------------------------------------------- #
# Derived features
# --------------------------------------------------------------------------- #
def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Add the derived columns listed in :data:`config.ENGINEERED_FEATURES`.

    The transforms are stateless (no fitted statistics), so this function is
    safe to call on a single track at inference time.
    """
    out = df.copy()

    out["duration_min"] = out["duration_ms"] / 60_000.0

    # Normalised loudness: dB is negative and roughly bounded by [-60, 0].
    out["loudness_norm"] = ((out["loudness"] + 60.0) / 60.0).clip(0.0, 1.0)
    out["tempo_norm"] = (out["tempo"] / 250.0).clip(0.0, 1.0)

    # "How happy does it feel" vs "how hard does it hit".
    out["mood_index"] = out["valence"] * out["danceability"]
    out["intensity"] = out["energy"] * out["loudness_norm"]
    out["energy_per_loudness"] = out["energy"] / (out["loudness_norm"] + 1e-3)

    out["is_acoustic"] = (out["acousticness"] > config.ACOUSTIC_THRESHOLD).astype(int)
    out["is_instrumental"] = (out["instrumentalness"] > config.INSTRUMENTAL_THRESHOLD).astype(int)
    out["is_spoken"] = (out["speechiness"] > config.SPEECH_THRESHOLD).astype(int)
    out["is_live"] = (out["liveness"] > config.LIVE_THRESHOLD).astype(int)

    return out


def _log1p(x):
    """Pickle-friendly log transform (lambdas cannot be joblib-dumped)."""
    return np.log1p(np.clip(np.asarray(x, dtype=float), 0.0, None))


# --------------------------------------------------------------------------- #
# Supervised preprocessing
# --------------------------------------------------------------------------- #
def build_preprocessor() -> ColumnTransformer:
    """Column-wise preprocessing matching the EDA conclusions.

    * long-tailed variables  -> ``log1p`` then ``RobustScaler``
    * symmetric variables    -> ``StandardScaler``
    * engineered variables   -> ``StandardScaler``
    * ``key`` / ``mode`` / ``time_signature`` -> one-hot
    """
    log_robust = Pipeline(
        [
            ("log", FunctionTransformer(_log1p, feature_names_out="one-to-one")),
            ("scale", RobustScaler()),
        ]
    )

    engineered = [f for f in config.ENGINEERED_FEATURES if not f.startswith("is_")]
    flags = [f for f in config.ENGINEERED_FEATURES if f.startswith("is_")]

    return ColumnTransformer(
        transformers=[
            ("skewed", log_robust, config.ROBUST_FEATURES),
            ("symmetric", StandardScaler(), config.STANDARD_FEATURES),
            ("derived", StandardScaler(), engineered),
            ("flags", "passthrough", flags),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=10),
                config.CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Columns the preprocessor expects, in a stable order."""
    enriched = engineer(df)
    columns = (
        config.ROBUST_FEATURES
        + config.STANDARD_FEATURES
        + config.ENGINEERED_FEATURES
        + config.CATEGORICAL_FEATURES
    )
    seen: list[str] = []
    for col in columns:
        if col not in seen:
            seen.append(col)
    return enriched[seen]


# --------------------------------------------------------------------------- #
# Similarity space
# --------------------------------------------------------------------------- #
class EmbeddingSpace:
    """Weighted, L2-normalised audio-feature space for nearest-neighbour search.

    Each dimension is quantile-ranked to ``[0, 1]`` before weighting, which
    makes cosine distance meaningful despite the wildly different native scales
    (``duration_ms`` in milliseconds vs ``valence`` in ``[0, 1]``).
    """

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights = dict(weights or config.SIMILARITY_WEIGHTS)
        self.columns = list(self.weights)
        self._quantiles: dict[str, np.ndarray] = {}

    # -- fitting ---------------------------------------------------------- #
    def fit(self, df: pd.DataFrame) -> EmbeddingSpace:
        grid = np.linspace(0, 100, 101)
        for col in self.columns:
            self._quantiles[col] = np.percentile(df[col].astype(float).to_numpy(), grid)
        return self

    def _rank(self, col: str, values: np.ndarray) -> np.ndarray:
        knots = self._quantiles[col]
        return np.interp(values, knots, np.linspace(0.0, 1.0, knots.size))

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self._quantiles:
            raise RuntimeError("EmbeddingSpace must be fitted before transform()")
        cols = []
        for col in self.columns:
            ranked = self._rank(col, df[col].astype(float).to_numpy())
            cols.append(ranked * self.weights[col])
        matrix = np.column_stack(cols).astype(np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.clip(norms, 1e-9, None)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    # -- persistence ------------------------------------------------------ #
    def state_dict(self) -> dict:
        return {
            "columns": self.columns,
            "weights": [self.weights[c] for c in self.columns],
            "quantiles": np.stack([self._quantiles[c] for c in self.columns]),
        }

    @classmethod
    def from_state(cls, state: dict) -> EmbeddingSpace:
        columns = [str(c) for c in state["columns"]]
        weights = {c: float(w) for c, w in zip(columns, state["weights"], strict=True)}
        space = cls(weights)
        space.columns = columns
        quantiles = np.asarray(state["quantiles"])
        space._quantiles = {c: quantiles[i] for i, c in enumerate(columns)}
        return space
