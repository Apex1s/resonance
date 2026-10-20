# HTTP API

Base URL `http://localhost:8000`. Interactive docs at `/docs` (Swagger) and
`/redoc`. The dashboard at `/` uses nothing but these endpoints.

Every endpoint returns `503` with an explanatory `detail` when the trained
artefacts are missing, so run `python -m resonance.train` first.

---

## `GET /api/health`

```json
{"status":"ok","version":"1.0.0","tracks":1981,"model":"Hist Gradient Boosting","metrics_loaded":true}
```

## `GET /api/tracks`

| Query | Default | Notes |
|---|---|---|
| `q` | `""` | substring match on title **or** artist; prefix matches rank first |
| `limit` | 25 | 1–200 |
| `sort` | `p_like` | also `tempo`, `energy`, `danceability`, `valence`, `duration_ms` (ignored when `q` is set) |

```bash
curl "localhost:8000/api/tracks?q=drake&limit=3"
```

## `GET /api/tracks/{track_id}`

One track, including `key_name` (`"G# major"`), `p_like` and the ground-truth
`liked` label. `404` when the id is unknown.

## `GET /api/tracks/{track_id}/similar`

Pure content-based neighbours. The taste model is not consulted.

| Query | Default | Range |
|---|---|---|
| `k` | 10 | 1–50 |
| `diversity` | 0.0 | 0–1 |

```bash
curl "localhost:8000/api/tracks/08cb6b5d2e9d/similar?k=5"
```

## `POST /api/recommend`

The main endpoint: hybrid ranking from one or more seeds.

```jsonc
{
  "seed_ids": ["08cb6b5d2e9d"],   // 1–20, required
  "k": 12,                        // 1–50
  "taste_weight": 0.35,           // 0 = similarity only, 1 = taste only
  "diversity": 0.25,              // MMR trade-off
  "exclude_seed_artists": false,
  "explain": true,
  "filters": {
    "tempo_min": 110,
    "tempo_max": null,
    "duration_max_min": 5.0,
    "energy_min": 0.4,
    "valence_min": null,
    "acoustic_only": false,
    "instrumental_only": false,
    "exclude_spoken": true,
    "exclude_artists": ["Drake"]
  }
}
```

Response (abridged):

```jsonc
{
  "seeds": [ { "track_id": "...", "song_title": "Redbone", ... } ],
  "results": [
    {
      "track_id": "be6781d755a0",
      "song_title": "Murderer",
      "artist": "Barrington Levy",
      "tempo": 143.0, "key_name": "C major", "duration_ms": 233000,
      "p_like": 0.9711,
      "similarity": 0.9731,
      "score": 0.9845,
      "why": [
        {"feature": "valence", "role": "match", "seed_value": 0.588,
         "track_value": 0.616, "gap": 0.113, "agreement": 0.8928},
        {"feature": "speechiness", "role": "contrast", "gap": 2.41, ...}
      ]
    }
  ],
  "settings": {"k": 12, "taste_weight": 0.35, "diversity": 0.25,
               "filtered": true, "unknown_seeds": []},
  "took_ms": 11.08
}
```

Notes

* Unknown seed ids are reported in `settings.unknown_seeds` and skipped; `404`
  only when *none* of the seeds exist.
* Filters are hard constraints applied before ranking. When nothing satisfies
  them, `results` comes back empty rather than silently relaxing one.
* Results come back in playlist order, which is MMR order, not sorted by
  `score` once `diversity > 0`.

## `POST /api/recommend/profile`

Same ranking, seeded by a hand-built audio profile instead of real tracks. Any
field left out falls back to the catalog median.

```bash
curl -X POST localhost:8000/api/recommend/profile \
  -H 'Content-Type: application/json' \
  -d '{"profile":{"energy":0.95,"tempo":128,"valence":0.7},"k":8,"taste_weight":0.3}'
```

## `POST /api/predict`

Scores a profile with the calibrated taste model.

```json
{
  "p_like": 0.7888,
  "verdict": "strong match",
  "percentile": 68.1,
  "nearest": [ { "song_title": "Dope", "artist": "Tyga", ... } ]
}
```

`verdict` thresholds: ≥0.70 *strong match*, ≥0.55 *likely*, ≥0.45 *borderline*,
below that *unlikely*.

## `GET /api/stats`

Dataset summary, split sizes, the winning model with its hyper-parameters, the
held-out metrics and the full CV leaderboard.

## `GET /api/insights`

Permutation importance, ANOVA (with Cohen's *d*), chi-squared (with Cramér's
*V*), the correlation matrix, and per-feature histograms split by class. This is
the payload behind the dashboard's third tab.

---

## Status codes

| Code | When |
|---|---|
| 200 | fine |
| 404 | unknown `track_id`, or no known seed |
| 422 | pydantic validation (`k > 50`, `taste_weight > 1`, empty `seed_ids`) |
| 503 | artefacts or `metrics.json` missing, train first |

## Python client

The HTTP layer is thin; in-process use is identical.

```python
from resonance import RecommendationEngine, Filters

engine = RecommendationEngine.load()
seed = engine.search("Redbone").iloc[0]["track_id"]

playlist = engine.recommend(
    [seed], k=12, taste_weight=0.35, diversity=0.25,
    filters=Filters(tempo_min=100, exclude_spoken=True),
)
print(playlist[["song_title", "artist", "p_like", "score"]])
print(engine.explain([seed], playlist.iloc[0]["track_id"]))
```
