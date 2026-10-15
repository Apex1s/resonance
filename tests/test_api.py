"""HTTP contract of the FastAPI service."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from resonance.api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def seed_id(client):
    response = client.get("/api/tracks", params={"limit": 1})
    if response.status_code == 503:
        pytest.skip("engine artefacts not built")
    return response.json()[0]["track_id"]


def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] in {"ok", "degraded"}
    assert body["version"]


def test_openapi_is_served(client):
    schema = client.get("/openapi.json").json()
    assert "/api/recommend" in schema["paths"]


def test_dashboard_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Resonance" in response.text


def test_track_payload_shape(client, seed_id):
    track = client.get(f"/api/tracks/{seed_id}").json()
    for field in ("song_title", "artist", "tempo", "key_name", "p_like"):
        assert field in track
    assert track["key_name"].split()[-1] in {"major", "minor"}


def test_unknown_track_is_404(client, seed_id):
    assert client.get("/api/tracks/nope").status_code == 404


def test_search_filters_the_catalog(client, seed_id):
    hits = client.get("/api/tracks", params={"q": "drake", "limit": 5}).json()
    assert hits
    assert all("drake" in (h["artist"] + h["song_title"]).lower() for h in hits)


def test_similar_endpoint(client, seed_id):
    body = client.get(f"/api/tracks/{seed_id}/similar", params={"k": 5}).json()
    assert body["seed"]["track_id"] == seed_id
    assert len(body["results"]) == 5
    assert body["results"][0]["similarity"] >= body["results"][-1]["similarity"]


def test_recommend_with_filters_and_explanations(client, seed_id):
    payload = {
        "seed_ids": [seed_id],
        "k": 6,
        "taste_weight": 0.4,
        "diversity": 0.3,
        "filters": {"tempo_min": 90, "exclude_spoken": True},
        "explain": True,
    }
    body = client.post("/api/recommend", json=payload).json()
    assert len(body["results"]) == 6
    assert all(r["tempo"] >= 90 for r in body["results"])
    assert all(r["speechiness"] <= 0.33 for r in body["results"])
    assert body["results"][0]["why"]
    assert body["took_ms"] >= 0


def test_recommend_rejects_bad_parameters(client, seed_id):
    assert client.post("/api/recommend", json={"seed_ids": [], "k": 5}).status_code == 422
    assert client.post(
        "/api/recommend", json={"seed_ids": [seed_id], "taste_weight": 3}
    ).status_code == 422


def test_recommend_unknown_seed_is_404(client):
    response = client.post("/api/recommend", json={"seed_ids": ["nope"], "k": 3})
    assert response.status_code in {404, 503}


def test_predict_scores_a_profile(client, seed_id):
    body = client.post(
        "/api/predict",
        json={"profile": {"danceability": 0.9, "energy": 0.8, "speechiness": 0.35}},
    ).json()
    assert 0 <= body["p_like"] <= 1
    assert body["verdict"] in {"strong match", "likely", "borderline", "unlikely"}
    assert len(body["nearest"]) == 5


def test_profile_recommendation(client, seed_id):
    body = client.post(
        "/api/recommend/profile",
        json={"profile": {"energy": 0.95, "tempo": 128}, "k": 4, "taste_weight": 0.2},
    ).json()
    assert len(body["results"]) == 4


def test_stats_and_insights(client):
    stats = client.get("/api/stats")
    if stats.status_code == 503:
        pytest.skip("metrics.json not built")
    body = stats.json()
    assert 0.5 < body["test_metrics"]["roc_auc"] <= 1.0
    assert body["leaderboard"]

    insights = client.get("/api/insights").json()
    assert insights["permutation_importance"]
    assert insights["anova"]
