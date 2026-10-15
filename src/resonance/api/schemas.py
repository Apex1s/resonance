"""Request/response models for the Resonance API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AudioProfile(BaseModel):
    """A hand-built audio fingerprint. Every field is optional; anything left
    out falls back to the catalog median."""

    acousticness: float | None = Field(None, ge=0, le=1)
    danceability: float | None = Field(None, ge=0, le=1)
    energy: float | None = Field(None, ge=0, le=1)
    instrumentalness: float | None = Field(None, ge=0, le=1)
    liveness: float | None = Field(None, ge=0, le=1)
    speechiness: float | None = Field(None, ge=0, le=1)
    valence: float | None = Field(None, ge=0, le=1)
    loudness: float | None = Field(None, ge=-60, le=5)
    tempo: float | None = Field(None, ge=30, le=250)
    duration_ms: int | None = Field(None, ge=10_000, le=1_500_000)

    def as_dict(self) -> dict:
        return {k: v for k, v in self.model_dump().items() if v is not None}


class FilterSpec(BaseModel):
    """Hard constraints applied before ranking."""

    tempo_min: float | None = Field(None, ge=0, le=250)
    tempo_max: float | None = Field(None, ge=0, le=250)
    duration_max_min: float | None = Field(None, gt=0, le=30, description="max duration in minutes")
    energy_min: float | None = Field(None, ge=0, le=1)
    valence_min: float | None = Field(None, ge=0, le=1)
    acoustic_only: bool = False
    instrumental_only: bool = False
    exclude_spoken: bool = False
    exclude_artists: list[str] = Field(default_factory=list, max_length=50)


class RecommendRequest(BaseModel):
    seed_ids: list[str] = Field(..., min_length=1, max_length=20)
    k: int = Field(10, ge=1, le=50)
    taste_weight: float = Field(0.35, ge=0, le=1, description="0 = pure similarity, 1 = pure taste model")
    diversity: float = Field(0.2, ge=0, le=1, description="MMR trade-off; 0 = top-k, 1 = max spread")
    exclude_seed_artists: bool = False
    filters: FilterSpec | None = None
    explain: bool = True


class ProfileRequest(BaseModel):
    profile: AudioProfile
    k: int = Field(10, ge=1, le=50)
    taste_weight: float = Field(0.35, ge=0, le=1)
    diversity: float = Field(0.2, ge=0, le=1)
    filters: FilterSpec | None = None


class Track(BaseModel):
    track_id: str
    song_title: str
    artist: str
    duration_ms: int
    tempo: float
    key: int
    key_name: str
    mode: int
    time_signature: int
    acousticness: float
    danceability: float
    energy: float
    instrumentalness: float
    liveness: float
    loudness: float
    speechiness: float
    valence: float
    p_like: float | None = None
    liked: int | None = None


class Recommendation(Track):
    score: float
    similarity: float
    why: list[dict] = Field(default_factory=list)


class RecommendResponse(BaseModel):
    seeds: list[Track]
    results: list[Recommendation]
    settings: dict
    took_ms: float


class PredictResponse(BaseModel):
    p_like: float
    verdict: str
    percentile: float
    nearest: list[Track]
