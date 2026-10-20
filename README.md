# Resonance

A music recommender built on Spotify audio features. You give it a track you
like, it gives you a playlist, and it tells you why each song is in there.

[![CI](https://github.com/Apex1s/resonance/actions/workflows/ci.yml/badge.svg)](https://github.com/aApex1s/resonance/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/license-MIT-black)](LICENSE)

![The playlist mixer](docs/assets/dashboard-mixer.png)

## The idea

The dataset is 2,017 tracks (1,981 distinct songs) that one listener labelled
as liked or skipped.
The usual exercise is to train a classifier on it, which works fine (about
0.85 AUC), but a classifier on its own cannot build a playlist. Rank every
track by `P(like)` and you get the same list every time, no matter what you
were listening to.

So Resonance scores candidates two ways and lets you decide how much each one
counts:

```
score = (1 - w) * similarity(candidate, seeds) + w * P(like | candidate)
```

At `w = 0` you get the nearest neighbours of your seed, which tends to mean ten
versions of the same song. At `w = 1` you get the listener's favourites and the
seed barely matters. Around 0.35 is where it gets interesting. A second pass
(Maximal Marginal Relevance) spreads the picks out so the playlist doesn't
collapse into near-duplicates.

More on the design in [docs/architecture.md](docs/architecture.md).

## Getting started

```bash
git clone https://github.com/Apex1s/resonance.git
cd resonance
pip install -r requirements.lock   # Python 3.11+, exact versions the model was trained with
pip install -e ".[dev]"

make serve      # dashboard and API on http://localhost:8000
```

The trained model ships with the repo, so there is nothing to train before the
first run. If you want to rebuild it from the CSV:

```bash
make train      # ~4 minutes
make snapshot   # refresh the dashboard's offline data
make test
```

There is a Dockerfile too: `make docker && make docker-run`.

From Python:

```python
from resonance import RecommendationEngine, Filters

engine = RecommendationEngine.load()
seed = engine.search("Redbone").iloc[0]["track_id"]

playlist = engine.recommend(
    [seed], k=12,
    taste_weight=0.35,
    diversity=0.25,
    filters=Filters(tempo_min=100, exclude_spoken=True),
)
print(playlist[["song_title", "artist", "tempo", "p_like", "score"]])
```

## The dashboard

Three tabs on top of the same engine:

**Mixer** picks seeds, moves the taste and diversity dials, applies filters
(tempo floor, length cap, no spoken word, instrumental only) and shows what
each recommendation matched on.

**Taste lab** has ten sliders for building a track that doesn't exist, and the
model scores it.

**Insights** shows the model comparison, the ROC curve, feature importance and
the statistical tests.

It calls the API when one is running. When there isn't one, it loads a 500 KB
snapshot and does the same ranking in the browser, which is why it can also be
hosted as a plain static page (see `.github/workflows/pages.yml`).

## Results

Eight models, same preprocessing, 5-fold stratified cross-validation:

| Model | CV ROC-AUC | Accuracy | F1 |
|---|---|---|---|
| Random Forest | 0.857 ± 0.020 | 0.776 | 0.771 |
| **Hist Gradient Boosting** | **0.852 ± 0.016** | 0.766 | 0.763 |
| Gradient Boosting | 0.849 ± 0.023 | 0.762 | 0.756 |
| Extra Trees | 0.846 ± 0.020 | 0.762 | 0.755 |
| SVM (RBF) | 0.838 ± 0.020 | 0.758 | 0.746 |
| k-Nearest Neighbours | 0.823 ± 0.019 | 0.729 | 0.684 |
| Logistic Regression | 0.786 ± 0.028 | 0.731 | 0.721 |
| Decision Tree | 0.664 ± 0.035 | 0.664 | 0.663 |

The top three get a grid search; tuned, Hist Gradient Boosting edges ahead
(CV ROC-AUC 0.860) and ships. It is wrapped in isotonic calibration if that
improves the cross-validated Brier score on the training split (here it does,
0.155 to 0.151). That step matters because the probability is added to a
similarity term, so it has to mean something on its own. The test split plays
no part in either choice: it is only used to report the final model.

On the 496 held-out tracks: **ROC-AUC 0.830**, accuracy 76.0%, F1 0.767,
Brier 0.169. Confusion matrix 181 / 67 / 52 / 196.

Precision@10 is 0.90 and precision@50 is 0.94, against a 50.1% base rate. That
is the number worth quoting for a recommender: AUC describes the whole ranking,
but a user only ever sees the top of it.

### What drives the predictions

Permutation importance on the test split, next to the ANOVA effect sizes:

| Feature | Importance | Cohen's *d* |
|---|---|---|
| `instrumentalness` | 0.112 | +0.31 |
| `speechiness` | 0.036 | +0.33 |
| `loudness` | 0.036 | -0.13 |
| `danceability` | 0.029 | +0.37 |
| `duration_ms` | 0.020 | +0.29 |

Vocals or no vocals is the strongest signal by a wide margin. With n ≈ 2,000
nearly everything clears p < 0.05, so the reports carry Cohen's *d* and a
Benjamini-Hochberg flag next to every p-value.

![Model insights](docs/assets/dashboard-insights.png)

## API

`make serve`, then open `/docs` for Swagger. Full reference in
[docs/api.md](docs/api.md).

| Method | Path | Does |
|---|---|---|
| `GET` | `/api/tracks?q=` | search the catalog |
| `GET` | `/api/tracks/{id}/similar` | content-based neighbours |
| `POST` | `/api/recommend` | hybrid playlist from seeds, with filters |
| `POST` | `/api/predict` | score an audio profile |
| `GET` | `/api/stats`, `/api/insights` | metrics and statistical tests |

```bash
curl -X POST localhost:8000/api/recommend \
  -H 'Content-Type: application/json' \
  -d '{"seed_ids":["08cb6b5d2e9d"],"k":10,"taste_weight":0.35,
       "diversity":0.25,"filters":{"tempo_min":110,"exclude_spoken":true}}'
```

A request takes about 11 ms.

## A few notes on the build

Audio features get described twice, on purpose. The classifier sees 38
dimensions with one-hot keys and engineered ratios. Similarity search sees 10
dimensions, quantile-ranked so that `duration_ms` in the tens of thousands
doesn't drown out `valence` between 0 and 1, then weighted by how much each
attribute should actually pull a neighbourhood.

Other things worth knowing:

- The CSV is validated on load, so a wrong file fails with a named column
  instead of a `KeyError` further down.
- `track_id` is a hash of title and artist, so ids survive a re-export.
- Scalers and encoders live inside the `Pipeline`, so cross-validation fits
  them fold by fold and nothing leaks.
- CI retrains, boots the API, calls it, and fails if AUC drops below 0.80.
- 54 tests cover the pipeline, the ranking, the HTTP contract and the
  hardening below.

Deployment notes:

- The model is a pickle, and unpickling runs code. Training writes its SHA-256
  to `src/resonance/model_manifest.json`, inside the package, and the API
  refuses to load a `taste_model.joblib` that does not match. Retrieval keeps
  working without it; only profile scoring needs the model. For local
  experiments only, `RESONANCE_ALLOW_UNVERIFIED_MODEL=1` skips the check when
  there is no manifest.
- `requirements.lock` pins every dependency (`make lock` regenerates it). The
  Docker image and the CI pipeline job install from it, so the model is loaded
  by the scikit-learn version it was trained with.
- CORS defaults to `*` (read-only API, no cookies). Set
  `RESONANCE_CORS_ORIGINS=https://example.org,...` when deploying. There is no
  authentication or rate limiting in the app: put it behind a reverse proxy
  that rate-limits if it is exposed publicly.
- The dashboard is served with a Content-Security-Policy that allows only its
  own inline script (by hash), plus `nosniff`, `X-Frame-Options: DENY` and
  `Referrer-Policy: no-referrer`.

## Dataset

2,017 tracks from the Spotify Web API audio-features endpoint, 1,981 after
de-duplication on title and artist (a song repeated with a different duration
would otherwise sit on both sides of the train/test split), with one listener's
like/skip label. Columns and provenance in
[data/README.md](data/README.md).

Worth being precise about: `target` is one person's taste, not chart
popularity. The model learns a listener, and the recommender ranks for that
listener.

## Where it came from

This started as a statistical study of Spotify audio features
([docs/rapport-analyse-statistique.pdf](docs/rapport-analyse-statistique.pdf),
EMINES, May 2025) by Ammar Haddad, Anas Sabbahi and Othmane Tahfi, supervised
by Lahcen Douge and Mohammed El Machkouri. The original notebook is kept as-is
in `notebooks/`.

The study ended at the classifier. This repo picks up its conclusions about
which features matter and builds the recommender they were pointing at.

## What's next

- Thumbs up/down during a session, refitting the taste model as you go
- Playlist order as a transition model rather than a ranked set
- Spotify OAuth so anyone can seed from their own library
- Approximate nearest neighbours once the catalog passes ~100k tracks

## License

MIT, see [LICENSE](LICENSE). The dataset is included for reproducibility, the
audio is not.
