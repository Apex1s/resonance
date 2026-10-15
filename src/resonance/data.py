"""Dataset loading, validation and catalog construction."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


class SchemaError(ValueError):
    """Raised when an input CSV does not match the expected Spotify schema."""


def _validate(df: pd.DataFrame, *, require_target: bool = True) -> None:
    expected = set(config.AUDIO_FEATURES + config.CATEGORICAL_FEATURES)
    if require_target:
        expected.add(config.TARGET)
    missing = expected - set(df.columns)
    if missing:
        raise SchemaError(f"missing column(s): {sorted(missing)}")

    numeric = config.AUDIO_FEATURES + config.CATEGORICAL_FEATURES
    bad = [c for c in numeric if not pd.api.types.is_numeric_dtype(df[c])]
    if bad:
        raise SchemaError(f"non-numeric audio column(s): {bad}")

    if require_target:
        values = set(pd.unique(df[config.TARGET].dropna()))
        if not values <= {0, 1}:
            raise SchemaError(f"target must be binary, found {sorted(values)}")


def _make_track_id(row: pd.Series) -> str:
    """Stable, human-free identifier derived from title + artist."""
    raw = f"{row.get('song_title', '')}|{row.get('artist', '')}".lower().strip()
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def load_tracks(path: str | Path | None = None, *, require_target: bool = True) -> pd.DataFrame:
    """Load the raw track table and attach a stable ``track_id``.

    Parameters
    ----------
    path:
        CSV to read. Defaults to :data:`config.RAW_DATA_PATH`.
    require_target:
        When ``False`` the ``target`` column may be absent (inference on a
        catalog that carries no feedback yet).
    """
    path = Path(path or config.RAW_DATA_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"dataset not found at {path}. Place the Spotify CSV there or set RESONANCE_RAW_DATA."
        )

    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    _validate(df, require_target=require_target)

    for col in config.TEXT_COLUMNS:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str).str.strip()

    df = df.drop_duplicates(subset=config.TEXT_COLUMNS + ["duration_ms"]).reset_index(drop=True)
    df[config.ID_COLUMN] = df.apply(_make_track_id, axis=1)

    # Collisions are vanishingly unlikely but would silently break lookups.
    dupes = df[config.ID_COLUMN].duplicated()
    if dupes.any():
        suffix = dupes.groupby(df[config.ID_COLUMN]).cumsum().astype(str)
        df.loc[dupes, config.ID_COLUMN] += "-" + suffix[dupes]

    return df


def describe(df: pd.DataFrame) -> dict:
    """Dataset-level summary consumed by the dashboard."""
    liked = int(df[config.TARGET].sum()) if config.TARGET in df else 0
    return {
        "n_tracks": int(len(df)),
        "n_artists": int(df["artist"].nunique()),
        "n_liked": liked,
        "n_disliked": int(len(df) - liked),
        "like_rate": round(float(liked / len(df)), 4) if len(df) else 0.0,
        "median_duration_min": round(float(df["duration_ms"].median() / 60000), 2),
        "median_tempo": round(float(df["tempo"].median()), 1),
        "top_artists": [
            {"artist": a, "n_tracks": int(n)}
            for a, n in df["artist"].value_counts().head(12).items()
            if a
        ],
    }


def feature_distributions(df: pd.DataFrame, bins: int = 24) -> dict:
    """Per-feature histograms split by class, pre-computed for the dashboard."""
    out: dict[str, dict] = {}
    for feature in config.AUDIO_FEATURES:
        series = df[feature].astype(float)
        lo, hi = float(series.min()), float(series.max())
        edges = np.linspace(lo, hi, bins + 1)
        entry: dict[str, object] = {
            "edges": [round(e, 4) for e in edges.tolist()],
            "mean": round(float(series.mean()), 4),
            "std": round(float(series.std()), 4),
        }
        if config.TARGET in df.columns:
            for label in (0, 1):
                counts, _ = np.histogram(series[df[config.TARGET] == label], bins=edges)
                entry[f"class_{label}"] = counts.tolist()
                entry[f"mean_{label}"] = round(float(series[df[config.TARGET] == label].mean()), 4)
        out[feature] = entry
    return out
