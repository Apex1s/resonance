"""Export a self-contained snapshot the dashboard can run on without a server.

The dashboard talks to the FastAPI service when one is reachable. When it is
not (a static host, a GitHub Pages demo, a shared link), it falls back to this
snapshot and reproduces the same ranking maths in the browser: cosine
similarity over the identical embedding vectors, the identical blend with
``p_like``, the identical MMR re-ranking.

The only thing that cannot cross over is the scikit-learn model itself, so in
offline mode the "taste" score of a hand-built profile is estimated from its
nearest catalog neighbours. The dashboard labels it as such.

    python -m resonance.export_snapshot
"""

from __future__ import annotations

import json
import logging

from . import config
from .recommender import RecommendationEngine

LOGGER = logging.getLogger("resonance.snapshot")

OUTPUT_PATH = config.PACKAGE_ROOT / "api" / "static" / "snapshot.json"

CATALOG_FIELDS = [
    "acousticness",
    "danceability",
    "energy",
    "instrumentalness",
    "liveness",
    "loudness",
    "speechiness",
    "tempo",
    "valence",
    "duration_ms",
]


def build(include_metrics: bool = True) -> dict:
    engine = RecommendationEngine.load()
    catalog = engine.catalog

    tracks = []
    for _, row in catalog.iterrows():
        tracks.append(
            {
                "id": row[config.ID_COLUMN],
                "t": row["song_title"],
                "a": row["artist"],
                "f": [round(float(row[c]), 4) for c in CATALOG_FIELDS],
                "k": int(row["key"]),
                "m": int(row["mode"]),
                "s": int(row["time_signature"]),
                "p": round(float(row["p_like"]), 4),
                "y": int(row[config.TARGET]) if config.TARGET in row else None,
            }
        )

    state = engine.space.state_dict()
    snapshot = {
        "version": config.API_VERSION,
        "fields": CATALOG_FIELDS,
        "tracks": tracks,
        "vectors": [[round(float(v), 4) for v in vec] for vec in engine.vectors],
        "space": {
            "columns": state["columns"],
            "weights": [float(w) for w in state["weights"]],
            "quantiles": [[round(float(q), 4) for q in row] for row in state["quantiles"]],
        },
        "model_name": engine.model_name,
    }

    if include_metrics and config.METRICS_PATH.exists():
        snapshot["metrics"] = json.loads(config.METRICS_PATH.read_text())

    return snapshot


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    snapshot = build()
    OUTPUT_PATH.write_text(json.dumps(snapshot, separators=(",", ":")))
    size_kb = OUTPUT_PATH.stat().st_size / 1024
    LOGGER.info("wrote %s (%d tracks, %.0f KB)", OUTPUT_PATH, len(snapshot["tracks"]), size_kb)


if __name__ == "__main__":
    main()
