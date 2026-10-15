# Data

## `raw/spotify_tracks.csv`

2,017 tracks pulled from the Spotify Web API's **audio-features** endpoint, each
labelled with one listener's verdict. 1,999 rows survive de-duplication (same
title, artist and duration).

| Column | Meaning | Range |
|---|---|---|
| `acousticness` | confidence the track is acoustic | 0–1 |
| `danceability` | tempo, rhythm stability, beat strength combined | 0–1 |
| `duration_ms` | length in milliseconds | 16 s to 16 min |
| `energy` | perceptual intensity | 0–1 |
| `instrumentalness` | confidence the track has no vocals | 0–1 |
| `key` | pitch class, `0 = C … 11 = B` | 0–11 |
| `liveness` | confidence of an audience in the recording | 0–1 |
| `loudness` | average loudness | -33 to -0.3 dB |
| `mode` | `0 = minor`, `1 = major` | {0, 1} |
| `speechiness` | presence of spoken words | 0–1 |
| `tempo` | estimated beats per minute | 48–219 |
| `time_signature` | beats per bar | 3–5 |
| `valence` | musical positivity | 0–1 |
| `target` | **1 = the listener liked it**, 0 = they did not | {0, 1} |
| `song_title`, `artist` | free text | n/a |

### What `target` actually is

It is one person's like/skip history, not a global popularity score. Everything
this project predicts is therefore *taste*, not chart performance. The model
learns one listener's preferences and the recommender ranks for that listener.
The original study framed it as popularity; the framing here is narrower and
matches what the label can support.

### Class balance

50.6% liked and 49.4% skipped, so accuracy is a meaningful
metric and no resampling is needed.

### Provenance

Public Kaggle dataset *Spotify Song Attributes* (George McIntire), collected
through the Spotify Web API. Audio features are described in the
[API reference](https://developer.spotify.com/documentation/web-api/reference/get-audio-features).
The CSV is redistributed here for reproducibility; no audio is included.

## `processed/`

Scratch space for intermediate frames. Not tracked.
