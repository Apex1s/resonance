"""The recommendation engine.

Three retrieval modes share one scoring core:

``similar_tracks``      pure content-based neighbours of a seed track
``recommend``           hybrid: seed similarity blended with predicted taste
``recommend_profile``   same, but seeded by a hand-built audio profile

Ranking happens in two stages: score, then re-rank with Maximal Marginal
Relevance so a playlist does not collapse into ten versions of the same song.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .features import EmbeddingSpace, engineer, feature_matrix


class ArtifactsMissing(RuntimeError):
    """Raised when the engine is used before `python -m resonance.train` ran."""


@dataclass(slots=True)
class Filters:
    """Optional hard constraints applied before ranking."""

    tempo_min: float | None = None
    tempo_max: float | None = None
    duration_max_min: float | None = None
    energy_min: float | None = None
    valence_min: float | None = None
    acoustic_only: bool = False
    instrumental_only: bool = False
    exclude_spoken: bool = False
    exclude_artists: list[str] = field(default_factory=list)

    def mask(self, catalog: pd.DataFrame) -> np.ndarray:
        keep = np.ones(len(catalog), dtype=bool)
        if self.tempo_min is not None:
            keep &= catalog["tempo"].to_numpy() >= self.tempo_min
        if self.tempo_max is not None:
            keep &= catalog["tempo"].to_numpy() <= self.tempo_max
        if self.duration_max_min is not None:
            keep &= catalog["duration_ms"].to_numpy() <= self.duration_max_min * 60_000
        if self.energy_min is not None:
            keep &= catalog["energy"].to_numpy() >= self.energy_min
        if self.valence_min is not None:
            keep &= catalog["valence"].to_numpy() >= self.valence_min
        if self.acoustic_only:
            keep &= catalog["acousticness"].to_numpy() > config.ACOUSTIC_THRESHOLD
        if self.instrumental_only:
            keep &= catalog["instrumentalness"].to_numpy() > config.INSTRUMENTAL_THRESHOLD
        if self.exclude_spoken:
            keep &= catalog["speechiness"].to_numpy() <= config.SPEECH_THRESHOLD
        if self.exclude_artists:
            lowered = {a.lower() for a in self.exclude_artists}
            keep &= ~catalog["artist"].str.lower().isin(lowered).to_numpy()
        return keep


class RecommendationEngine:
    """Loads the trained artefacts and answers retrieval queries."""

    def __init__(
        self,
        catalog: pd.DataFrame,
        vectors: np.ndarray,
        space: EmbeddingSpace,
        model=None,
        model_name: str = "unknown",
    ) -> None:
        self.catalog = catalog.reset_index(drop=True)
        self.vectors = vectors.astype(np.float32)
        self.space = space
        self.model = model
        self.model_name = model_name
        self._index = {tid: i for i, tid in enumerate(self.catalog[config.ID_COLUMN])}
        if "p_like" not in self.catalog.columns:
            self.catalog["p_like"] = 0.5

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def load(cls, models_dir: Path | None = None) -> RecommendationEngine:
        models_dir = Path(models_dir or config.MODELS_DIR)
        catalog_path = models_dir / config.CATALOG_PATH.name
        embedding_path = models_dir / config.EMBEDDING_PATH.name
        model_path = models_dir / config.TASTE_MODEL_PATH.name

        if not catalog_path.exists() or not embedding_path.exists():
            raise ArtifactsMissing(
                "trained artefacts not found, run `python -m resonance.train` first"
            )

        catalog = pd.read_csv(catalog_path)
        blob = np.load(embedding_path, allow_pickle=False)
        space = EmbeddingSpace.from_state(
            {
                "columns": [str(c) for c in blob["columns"]],
                "weights": blob["weights"].tolist(),
                "quantiles": blob["quantiles"],
            }
        )

        model, model_name = None, "unavailable"
        if model_path.exists():
            import joblib

            bundle = joblib.load(model_path)
            model, model_name = bundle["pipeline"], bundle.get("model_name", "unknown")

        return cls(catalog, blob["vectors"], space, model, model_name)

    # ------------------------------------------------------------------ #
    # Lookups
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.catalog)

    def has(self, track_id: str) -> bool:
        return track_id in self._index

    def row(self, track_id: str) -> pd.Series:
        if track_id not in self._index:
            raise KeyError(f"unknown track_id: {track_id}")
        return self.catalog.iloc[self._index[track_id]]

    def search(self, query: str, limit: int = 20) -> pd.DataFrame:
        """Case-insensitive substring search over title and artist."""
        if not query.strip():
            return self.catalog.head(limit)
        q = query.strip().lower()
        title = self.catalog["song_title"].str.lower()
        artist = self.catalog["artist"].str.lower()
        hits = self.catalog[title.str.contains(q, regex=False) | artist.str.contains(q, regex=False)]
        # Prefix matches first, so "dra" surfaces Drake before Hendrix.
        starts = hits["song_title"].str.lower().str.startswith(q) | hits["artist"].str.lower().str.startswith(q)
        return pd.concat([hits[starts], hits[~starts]]).head(limit)

    # ------------------------------------------------------------------ #
    # Scoring core
    # ------------------------------------------------------------------ #
    def _seed_vector(self, track_ids: list[str]) -> np.ndarray:
        idx = [self._index[t] for t in track_ids if t in self._index]
        if not idx:
            raise KeyError("none of the seed track ids are in the catalog")
        centroid = self.vectors[idx].mean(axis=0)
        return centroid / max(float(np.linalg.norm(centroid)), 1e-9)

    def profile_vector(self, features: dict) -> np.ndarray:
        """Embed an arbitrary audio profile into the similarity space."""
        defaults = {col: float(self.catalog[col].median()) for col in config.AUDIO_FEATURES}
        defaults.update({k: float(v) for k, v in features.items() if k in defaults})
        frame = engineer(pd.DataFrame([defaults]))
        vector = self.space.transform(frame)[0]
        return vector / max(float(np.linalg.norm(vector)), 1e-9)

    def predict_like(self, features: dict) -> float:
        """Probability that the listener profile behind the labels likes a track."""
        if self.model is None:
            raise ArtifactsMissing("taste model not loaded")
        defaults = {col: float(self.catalog[col].median()) for col in config.AUDIO_FEATURES}
        defaults.update({c: int(self.catalog[c].mode().iloc[0]) for c in config.CATEGORICAL_FEATURES})
        defaults.update({k: v for k, v in features.items() if k in defaults})
        row = feature_matrix(pd.DataFrame([defaults]))
        return float(self.model.predict_proba(row)[0, 1])

    @staticmethod
    def _mmr(
        similarity_to_query: np.ndarray,
        candidate_vectors: np.ndarray,
        k: int,
        diversity: float,
    ) -> list[int]:
        """Maximal Marginal Relevance re-ranking.

        ``diversity`` in ``[0, 1]``: 0 returns the pure top-k, 1 maximises
        spread across the feature space.
        """
        if diversity <= 0:
            return np.argsort(-similarity_to_query)[:k].tolist()

        selected: list[int] = []
        remaining = set(range(len(similarity_to_query)))
        lam = 1.0 - diversity
        pairwise_cache = np.full(len(similarity_to_query), -1.0, dtype=np.float32)

        first = int(np.argmax(similarity_to_query))
        selected.append(first)
        remaining.discard(first)

        while len(selected) < k and remaining:
            idx = np.fromiter(remaining, dtype=int)
            last = candidate_vectors[selected[-1]]
            sim_to_last = candidate_vectors[idx] @ last
            pairwise_cache[idx] = np.maximum(pairwise_cache[idx], sim_to_last)
            score = lam * similarity_to_query[idx] - diversity * pairwise_cache[idx]
            pick = int(idx[int(np.argmax(score))])
            selected.append(pick)
            remaining.discard(pick)
        return selected

    def _rank(
        self,
        query_vector: np.ndarray,
        *,
        k: int,
        taste_weight: float,
        diversity: float,
        filters: Filters | None,
        exclude: set[str],
    ) -> pd.DataFrame:
        similarity = self.vectors @ query_vector
        p_like = self.catalog["p_like"].to_numpy(dtype=float)

        keep = filters.mask(self.catalog) if filters else np.ones(len(self.catalog), dtype=bool)
        if exclude:
            keep &= ~self.catalog[config.ID_COLUMN].isin(exclude).to_numpy()
        candidates = np.where(keep)[0]
        if candidates.size == 0:
            return self.catalog.head(0).assign(score=[], similarity=[])

        # Similarity is rescaled over the candidate pool so that the blend is
        # not dominated by the narrow absolute range of cosine scores.
        sim = similarity[candidates]
        spread = max(float(sim.max() - sim.min()), 1e-9)
        sim_scaled = (sim - sim.min()) / spread
        score = (1.0 - taste_weight) * sim_scaled + taste_weight * p_like[candidates]

        pool = min(len(candidates), max(k * 6, 60))
        top_local = np.argsort(-score)[:pool]
        chosen_local = self._mmr(score[top_local], self.vectors[candidates[top_local]], k, diversity)
        chosen = candidates[top_local[chosen_local]]

        out = self.catalog.iloc[chosen].copy()
        out["similarity"] = similarity[chosen].round(4)
        out["score"] = score[top_local[chosen_local]].round(4)
        return out.reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def similar_tracks(self, track_id: str, k: int = 10, diversity: float = 0.0) -> pd.DataFrame:
        """Pure content-based neighbours. The taste model is not involved."""
        vector = self._seed_vector([track_id])
        return self._rank(
            vector,
            k=k,
            taste_weight=0.0,
            diversity=diversity,
            filters=None,
            exclude={track_id},
        )

    def recommend(
        self,
        seed_ids: list[str],
        k: int = 10,
        taste_weight: float = 0.35,
        diversity: float = 0.2,
        filters: Filters | None = None,
        exclude_seed_artists: bool = False,
    ) -> pd.DataFrame:
        """Hybrid recommendation from one or more seed tracks."""
        vector = self._seed_vector(seed_ids)
        exclude = set(seed_ids)
        if exclude_seed_artists:
            artists = {self.row(t)["artist"].lower() for t in seed_ids if self.has(t)}
            same = self.catalog["artist"].str.lower().isin(artists)
            exclude |= set(self.catalog.loc[same, config.ID_COLUMN])
        return self._rank(
            vector,
            k=k,
            taste_weight=taste_weight,
            diversity=diversity,
            filters=filters,
            exclude=exclude,
        )

    def recommend_profile(
        self,
        features: dict,
        k: int = 10,
        taste_weight: float = 0.35,
        diversity: float = 0.2,
        filters: Filters | None = None,
    ) -> pd.DataFrame:
        """Recommend from a hand-built audio profile (the dashboard's sliders)."""
        vector = self.profile_vector(features)
        return self._rank(
            vector,
            k=k,
            taste_weight=taste_weight,
            diversity=diversity,
            filters=filters,
            exclude=set(),
        )

    # ------------------------------------------------------------------ #
    # Explanation
    # ------------------------------------------------------------------ #
    def explain(self, seed_ids: list[str], track_id: str, top: int = 4) -> list[dict]:
        """Why this track followed those seeds.

        Compares the candidate to the seed centroid dimension by dimension in
        the weighted similarity space and returns the closest matches plus the
        sharpest contrast, which is usually what makes a playlist worth hearing.
        """
        seed_idx = [self._index[t] for t in seed_ids if t in self._index]
        if not seed_idx or track_id not in self._index:
            return []
        seed_rows = self.catalog.iloc[seed_idx]
        candidate = self.row(track_id)

        rows = []
        for col, weight in self.space.weights.items():
            seed_value = float(seed_rows[col].mean())
            value = float(candidate[col])
            scale = max(float(self.catalog[col].std()), 1e-9)
            delta = abs(value - seed_value) / scale
            rows.append(
                {
                    "feature": col,
                    "seed_value": round(seed_value, 4),
                    "track_value": round(value, 4),
                    "gap": round(delta, 3),
                    "weight": weight,
                    "agreement": round(float(np.exp(-delta)) * weight, 4),
                }
            )
        rows.sort(key=lambda r: r["agreement"], reverse=True)
        matches = rows[:top]
        contrast = max(rows, key=lambda r: r["gap"] * r["weight"])
        if contrast["feature"] not in {m["feature"] for m in matches}:
            contrast = {**contrast, "role": "contrast"}
            return [{**m, "role": "match"} for m in matches] + [contrast]
        return [{**m, "role": "match"} for m in matches]
