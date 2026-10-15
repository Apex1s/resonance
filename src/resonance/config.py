"""Central configuration: paths, schema and feature groups.

Everything downstream (features, training, recommender, API) imports its
column lists from here so the pipeline never drifts between modules.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]

DATA_DIR = Path(os.getenv("RESONANCE_DATA_DIR", PROJECT_ROOT / "data"))
RAW_DATA_PATH = Path(os.getenv("RESONANCE_RAW_DATA", DATA_DIR / "raw" / "spotify_tracks.csv"))
PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR = Path(os.getenv("RESONANCE_MODELS_DIR", PROJECT_ROOT / "models"))
REPORTS_DIR = Path(os.getenv("RESONANCE_REPORTS_DIR", PROJECT_ROOT / "reports"))

TASTE_MODEL_PATH = MODELS_DIR / "taste_model.joblib"
EMBEDDING_PATH = MODELS_DIR / "track_embeddings.npz"
CATALOG_PATH = MODELS_DIR / "catalog.csv"
METRICS_PATH = REPORTS_DIR / "metrics.json"

# --------------------------------------------------------------------------- #
# Raw schema
# --------------------------------------------------------------------------- #
TARGET = "target"
ID_COLUMN = "track_id"
TEXT_COLUMNS = ["song_title", "artist"]

#: Continuous audio descriptors as delivered by the Spotify audio-features API.
AUDIO_FEATURES = [
    "acousticness",
    "danceability",
    "duration_ms",
    "energy",
    "instrumentalness",
    "liveness",
    "loudness",
    "speechiness",
    "tempo",
    "valence",
]

#: Discrete musical descriptors with no ordinal meaning, one-hot encoded.
CATEGORICAL_FEATURES = ["key", "mode", "time_signature"]

RAW_COLUMNS = sorted(AUDIO_FEATURES + CATEGORICAL_FEATURES + TEXT_COLUMNS + [TARGET])

# --------------------------------------------------------------------------- #
# Feature engineering groups
# --------------------------------------------------------------------------- #
#: Long-tailed variables: log1p before scaling (see report §5.2).
LOG_FEATURES = ["duration_ms", "instrumentalness", "speechiness", "liveness", "acousticness"]

#: Heavy-outlier variables -> RobustScaler.
ROBUST_FEATURES = ["duration_ms", "instrumentalness", "speechiness", "liveness", "acousticness"]

#: Roughly symmetric variables -> StandardScaler.
STANDARD_FEATURES = ["danceability", "energy", "loudness", "tempo", "valence"]

#: Derived features built in `features.engineer`.
ENGINEERED_FEATURES = [
    "duration_min",
    "mood_index",
    "intensity",
    "is_acoustic",
    "is_instrumental",
    "is_spoken",
    "is_live",
    "tempo_norm",
    "loudness_norm",
    "energy_per_loudness",
]

#: Feature space used to measure track-to-track similarity. Weights express how
#: much each dimension should drive the neighbourhood (tuned by hand on the
#: listening tests documented in docs/architecture.md).
SIMILARITY_WEIGHTS: dict[str, float] = {
    "danceability": 1.0,
    "energy": 1.0,
    "valence": 1.0,
    "acousticness": 0.9,
    "instrumentalness": 0.8,
    "speechiness": 0.7,
    "loudness": 0.6,
    "tempo": 0.5,
    "liveness": 0.3,
    "duration_ms": 0.3,
}

MUSICAL_KEYS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# --------------------------------------------------------------------------- #
# Modelling
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
TEST_SIZE = 0.25
CV_FOLDS = 5

#: Binarisation thresholds from the exploratory analysis (report §3.5).
ACOUSTIC_THRESHOLD = 0.5
INSTRUMENTAL_THRESHOLD = 0.5
SPEECH_THRESHOLD = 0.33
LIVE_THRESHOLD = 0.8

API_TITLE = "Resonance API"
API_VERSION = "1.0.0"
