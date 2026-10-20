"""Security and robustness behaviours of the API, the model loader and the data layer."""

from __future__ import annotations

import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from resonance import config, data
from resonance.api import main
from resonance.api.main import app
from resonance.recommender import UntrustedModel, verify_model

requires_artifacts = pytest.mark.skipif(
    not (config.CATALOG_PATH.exists() and config.MODEL_MANIFEST_PATH.exists()),
    reason="run `python -m resonance.train` to build model artefacts",
)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
def test_dashboard_sends_security_headers(client):
    headers = client.get("/").headers
    assert headers["x-content-type-options"] == "nosniff"
    assert headers["x-frame-options"] == "DENY"
    csp = headers["content-security-policy"]
    assert "frame-ancestors 'none'" in csp
    assert "'sha256-" in csp and "'unsafe-inline'" not in csp.split("script-src")[1].split(";")[0]


def test_docs_are_not_blocked_by_the_dashboard_csp(client):
    response = client.get("/docs")
    assert response.status_code == 200
    assert "content-security-policy" not in response.headers


def test_search_query_length_is_capped(client):
    assert client.get("/api/tracks", params={"q": "a" * 101}).status_code == 422


def test_inverted_tempo_range_is_rejected(client):
    payload = {"seed_ids": ["x"], "filters": {"tempo_min": 180, "tempo_max": 90}}
    assert client.post("/api/recommend", json=payload).status_code == 422


def test_health_after_shutdown_reports_degraded():
    with TestClient(app):
        pass
    assert main.health()["status"] == "degraded"


def test_corrupt_metrics_do_not_break_startup(tmp_path, monkeypatch):
    broken = tmp_path / "metrics.json"
    broken.write_text("{not json")
    monkeypatch.setattr(config, "METRICS_PATH", broken)
    with TestClient(app) as c:
        assert c.get("/api/health").json()["metrics_loaded"] is False
        assert c.get("/api/stats").status_code == 503


# --------------------------------------------------------------------------- #
# Model integrity
# --------------------------------------------------------------------------- #
@requires_artifacts
def test_shipped_model_matches_its_manifest():
    manifest = json.loads(config.MODEL_MANIFEST_PATH.read_text())
    assert manifest["file"] == config.TASTE_MODEL_PATH.name
    verify_model(config.TASTE_MODEL_PATH)


def test_tampered_model_is_refused(tmp_path):
    fake = tmp_path / "taste_model.joblib"
    fake.write_bytes(b"not the trained model")
    with pytest.raises(UntrustedModel):
        verify_model(fake)


def test_model_without_manifest_is_refused(tmp_path, monkeypatch):
    model = tmp_path / "taste_model.joblib"
    model.write_bytes(b"anything")
    monkeypatch.setattr(config, "MODEL_MANIFEST_PATH", tmp_path / "missing.json")
    monkeypatch.setattr(config, "ALLOW_UNVERIFIED_MODEL", False)
    with pytest.raises(UntrustedModel):
        verify_model(model)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def test_one_row_per_song(tracks):
    key = tracks["song_title"].str.lower() + "|" + tracks["artist"].str.lower()
    assert key.is_unique
    assert not tracks[config.ID_COLUMN].str.contains("-").any()


def test_duplicate_songs_are_collapsed(tmp_path, tracks):
    sample = tracks.drop(columns=[config.ID_COLUMN]).head(5)
    repeated = pd.concat([sample, sample.assign(duration_ms=sample["duration_ms"] + 1000)])
    path = tmp_path / "dupes.csv"
    repeated.to_csv(path, index=False)
    assert len(data.load_tracks(path)) == 5


@requires_artifacts
def test_catalog_scores_are_out_of_sample():
    """Training rows must not get in-sample probabilities (near-perfect AUC)."""
    from sklearn.metrics import roc_auc_score

    catalog = pd.read_csv(config.CATALOG_PATH)
    train = catalog[~catalog["in_test_split"]]
    test = catalog[catalog["in_test_split"]]
    auc_train = roc_auc_score(train[config.TARGET], train["p_like"])
    auc_test = roc_auc_score(test[config.TARGET], test["p_like"])
    assert auc_train < 0.95
    assert abs(auc_train - auc_test) < 0.08
