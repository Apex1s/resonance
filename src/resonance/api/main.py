"""FastAPI service exposing the recommendation engine and the dashboard.

Start locally::

    uvicorn resonance.api.main:app --reload

The dashboard is served at ``/`` and talks to the same JSON endpoints that are
documented at ``/docs``.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .. import config
from ..recommender import ArtifactsMissing, Filters, RecommendationEngine
from .schemas import (
    PredictResponse,
    ProfileRequest,
    RecommendRequest,
    RecommendResponse,
    Track,
)

LOGGER = logging.getLogger("resonance.api")
STATIC_DIR = Path(__file__).parent / "static"

state: dict = {"engine": None, "metrics": None, "error": None}


# --------------------------------------------------------------------------- #
# Lifespan
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        state["engine"] = RecommendationEngine.load()
        LOGGER.info("engine ready: %d tracks, model=%s", len(state["engine"]), state["engine"].model_name)
    except (ArtifactsMissing, FileNotFoundError) as exc:
        state["error"] = str(exc)
        LOGGER.warning("engine unavailable: %s", exc)

    if config.METRICS_PATH.exists():
        state["metrics"] = json.loads(config.METRICS_PATH.read_text())
    yield
    state.clear()


app = FastAPI(
    title=config.API_TITLE,
    version=config.API_VERSION,
    description=(
        "Hybrid recommendation engine over Spotify audio features: content-based "
        "retrieval blended with a calibrated taste model."
    ),
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def engine() -> RecommendationEngine:
    if state["engine"] is None:
        raise HTTPException(
            status_code=503,
            detail=state["error"] or "engine not loaded, run `python -m resonance.train`",
        )
    return state["engine"]


# --------------------------------------------------------------------------- #
# Serialisation
# --------------------------------------------------------------------------- #
def _track(row: pd.Series) -> dict:
    key = int(row["key"])
    payload = {
        "track_id": row[config.ID_COLUMN],
        "song_title": row["song_title"],
        "artist": row["artist"],
        "duration_ms": int(row["duration_ms"]),
        "tempo": round(float(row["tempo"]), 1),
        "key": key,
        "key_name": config.MUSICAL_KEYS[key % 12] + (" major" if int(row["mode"]) == 1 else " minor"),
        "mode": int(row["mode"]),
        "time_signature": int(row["time_signature"]),
        **{f: round(float(row[f]), 4) for f in
           ["acousticness", "danceability", "energy", "instrumentalness",
            "liveness", "loudness", "speechiness", "valence"]},
    }
    if "p_like" in row:
        payload["p_like"] = round(float(row["p_like"]), 4)
    if config.TARGET in row and not pd.isna(row[config.TARGET]):
        payload["liked"] = int(row[config.TARGET])
    return payload


def _filters(spec) -> Filters | None:
    return Filters(**spec.model_dump()) if spec else None


# --------------------------------------------------------------------------- #
# Meta
# --------------------------------------------------------------------------- #
@app.get("/api/health", tags=["meta"])
def health() -> dict:
    eng = state["engine"]
    return {
        "status": "ok" if eng else "degraded",
        "version": config.API_VERSION,
        "tracks": len(eng) if eng else 0,
        "model": eng.model_name if eng else None,
        "metrics_loaded": state["metrics"] is not None,
    }


@app.get("/api/stats", tags=["meta"])
def stats() -> JSONResponse:
    """Headline numbers for the dashboard: dataset, leaderboard, test metrics."""
    metrics = state["metrics"]
    if metrics is None:
        raise HTTPException(status_code=503, detail="metrics.json not found, run training first")
    return JSONResponse(
        {
            "generated_at": metrics["generated_at"],
            "dataset": metrics["dataset"],
            "split": metrics["split"],
            "best_model": metrics["best_model"],
            "test_metrics": metrics["test_metrics"],
            "leaderboard": metrics["leaderboard"],
        }
    )


@app.get("/api/insights", tags=["meta"])
def insights() -> JSONResponse:
    """Statistical evidence behind the model: ANOVA, chi², importance, curves."""
    metrics = state["metrics"]
    if metrics is None:
        raise HTTPException(status_code=503, detail="metrics.json not found, run training first")
    return JSONResponse(
        {
            "permutation_importance": metrics["permutation_importance"],
            "anova": metrics["anova"],
            "chi_square": metrics["chi_square"],
            "correlations": metrics["correlations"],
            "distributions": metrics["distributions"],
        }
    )


# --------------------------------------------------------------------------- #
# Catalog
# --------------------------------------------------------------------------- #
@app.get("/api/tracks", response_model=list[Track], tags=["catalog"])
def list_tracks(
    q: str = Query("", description="search title or artist"),
    limit: int = Query(25, ge=1, le=200),
    sort: str = Query("p_like", pattern="^(p_like|tempo|energy|danceability|valence|duration_ms)$"),
) -> list[dict]:
    eng = engine()
    hits = eng.search(q, limit=limit if q else max(limit, 50))
    if not q:
        hits = hits.sort_values(sort, ascending=False)
    return [_track(r) for _, r in hits.head(limit).iterrows()]


@app.get("/api/tracks/{track_id}", response_model=Track, tags=["catalog"])
def get_track(track_id: str) -> dict:
    try:
        return _track(engine().row(track_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"unknown track_id: {track_id}") from exc


@app.get("/api/tracks/{track_id}/similar", tags=["recommend"])
def similar(
    track_id: str,
    k: int = Query(10, ge=1, le=50),
    diversity: float = Query(0.0, ge=0, le=1),
) -> dict:
    eng = engine()
    if not eng.has(track_id):
        raise HTTPException(status_code=404, detail=f"unknown track_id: {track_id}")
    started = time.perf_counter()
    hits = eng.similar_tracks(track_id, k=k, diversity=diversity)
    return {
        "seed": _track(eng.row(track_id)),
        "results": [
            {**_track(r), "score": float(r["score"]), "similarity": float(r["similarity"])}
            for _, r in hits.iterrows()
        ],
        "took_ms": round((time.perf_counter() - started) * 1000, 2),
    }


# --------------------------------------------------------------------------- #
# Recommendation
# --------------------------------------------------------------------------- #
@app.post("/api/recommend", response_model=RecommendResponse, tags=["recommend"])
def recommend(request: RecommendRequest) -> dict:
    eng = engine()
    unknown = [t for t in request.seed_ids if not eng.has(t)]
    if len(unknown) == len(request.seed_ids):
        raise HTTPException(status_code=404, detail=f"no known seed track: {unknown}")

    started = time.perf_counter()
    hits = eng.recommend(
        seed_ids=[t for t in request.seed_ids if eng.has(t)],
        k=request.k,
        taste_weight=request.taste_weight,
        diversity=request.diversity,
        filters=_filters(request.filters),
        exclude_seed_artists=request.exclude_seed_artists,
    )
    results = []
    for _, row in hits.iterrows():
        item = {**_track(row), "score": float(row["score"]), "similarity": float(row["similarity"])}
        if request.explain:
            item["why"] = eng.explain(request.seed_ids, row[config.ID_COLUMN])
        results.append(item)

    return {
        "seeds": [_track(eng.row(t)) for t in request.seed_ids if eng.has(t)],
        "results": results,
        "settings": {
            "k": request.k,
            "taste_weight": request.taste_weight,
            "diversity": request.diversity,
            "filtered": request.filters is not None,
            "unknown_seeds": unknown,
        },
        "took_ms": round((time.perf_counter() - started) * 1000, 2),
    }


@app.post("/api/recommend/profile", tags=["recommend"])
def recommend_from_profile(request: ProfileRequest) -> dict:
    eng = engine()
    started = time.perf_counter()
    hits = eng.recommend_profile(
        features=request.profile.as_dict(),
        k=request.k,
        taste_weight=request.taste_weight,
        diversity=request.diversity,
        filters=_filters(request.filters),
    )
    return {
        "profile": request.profile.as_dict(),
        "results": [
            {**_track(r), "score": float(r["score"]), "similarity": float(r["similarity"])}
            for _, r in hits.iterrows()
        ],
        "took_ms": round((time.perf_counter() - started) * 1000, 2),
    }


@app.post("/api/predict", response_model=PredictResponse, tags=["recommend"])
def predict(profile: ProfileRequest) -> dict:
    """Score an arbitrary audio profile with the taste model."""
    eng = engine()
    features = profile.profile.as_dict()
    try:
        p = eng.predict_like(features)
    except ArtifactsMissing as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    population = eng.catalog["p_like"].to_numpy(dtype=float)
    percentile = float((population < p).mean() * 100)
    nearest = eng.recommend_profile(features, k=5, taste_weight=0.0, diversity=0.0)
    verdict = (
        "strong match" if p >= 0.7 else
        "likely" if p >= 0.55 else
        "borderline" if p >= 0.45 else
        "unlikely"
    )
    return {
        "p_like": round(p, 4),
        "verdict": verdict,
        "percentile": round(percentile, 1),
        "nearest": [_track(r) for _, r in nearest.iterrows()],
    }


# --------------------------------------------------------------------------- #
# Dashboard
# --------------------------------------------------------------------------- #
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def dashboard():
    index = STATIC_DIR / "index.html"
    if not index.exists():
        return JSONResponse({"detail": "dashboard not bundled", "docs": "/docs"})
    return FileResponse(index)
