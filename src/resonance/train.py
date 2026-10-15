"""Training pipeline: model selection, tuning, evaluation and artefact export.

Run with::

    python -m resonance.train              # full run (tuning included)
    python -m resonance.train --fast       # skip the grid search

Outputs
-------
models/taste_model.joblib    fitted end-to-end pipeline (preprocessing + model)
models/track_embeddings.npz  similarity space + track vectors
models/catalog.csv           catalog enriched with the predicted like-probability
reports/metrics.json         everything the dashboard renders
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from . import config, data, stats
from .features import EmbeddingSpace, build_preprocessor, engineer, feature_matrix

LOGGER = logging.getLogger("resonance.train")


# --------------------------------------------------------------------------- #
# Candidate models
# --------------------------------------------------------------------------- #
def candidate_models() -> dict[str, object]:
    rs = config.RANDOM_STATE
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=rs),
        "Decision Tree": DecisionTreeClassifier(random_state=rs),
        "k-Nearest Neighbours": KNeighborsClassifier(n_neighbors=25, weights="distance"),
        "SVM (RBF)": CalibratedClassifierCV(SVC(random_state=rs), ensemble=False),
        "Random Forest": RandomForestClassifier(n_estimators=400, random_state=rs, n_jobs=-1),
        "Extra Trees": ExtraTreesClassifier(n_estimators=400, random_state=rs, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=rs),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(random_state=rs),
    }


PARAM_GRIDS: dict[str, dict] = {
    "Random Forest": {
        "model__n_estimators": [300, 600],
        "model__max_depth": [None, 12, 20],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", 0.4],
    },
    "Extra Trees": {
        "model__n_estimators": [400, 800],
        "model__max_depth": [None, 16],
        "model__min_samples_leaf": [1, 2],
    },
    "Hist Gradient Boosting": {
        "model__learning_rate": [0.05, 0.1],
        "model__max_leaf_nodes": [15, 31, 63],
        "model__min_samples_leaf": [10, 20],
        "model__l2_regularization": [0.0, 1.0],
    },
    "Gradient Boosting": {
        "model__n_estimators": [200, 400],
        "model__learning_rate": [0.05, 0.1],
        "model__max_depth": [2, 3],
    },
}


def make_pipeline(model) -> Pipeline:
    return Pipeline([("prep", build_preprocessor()), ("model", model)])


def _curve(x: np.ndarray, y: np.ndarray, n: int = 120) -> list[list[float]]:
    """Down-sample a curve so metrics.json stays small."""
    idx = np.arange(len(x)) if len(x) <= n else np.unique(np.linspace(0, len(x) - 1, n).astype(int))
    return [[round(float(x[i]), 4), round(float(y[i]), 4)] for i in idx]


# --------------------------------------------------------------------------- #
# Steps
# --------------------------------------------------------------------------- #
def compare_models(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Stratified cross-validation of every candidate on the training split."""
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    scoring = ["roc_auc", "accuracy", "f1", "average_precision"]
    rows = []
    for name, model in candidate_models().items():
        started = time.perf_counter()
        scores = cross_validate(make_pipeline(model), X, y, cv=cv, scoring=scoring, n_jobs=1)
        rows.append(
            {
                "model": name,
                "roc_auc": round(float(scores["test_roc_auc"].mean()), 4),
                "roc_auc_std": round(float(scores["test_roc_auc"].std()), 4),
                "accuracy": round(float(scores["test_accuracy"].mean()), 4),
                "f1": round(float(scores["test_f1"].mean()), 4),
                "avg_precision": round(float(scores["test_average_precision"].mean()), 4),
                "fit_seconds": round(time.perf_counter() - started, 2),
            }
        )
        LOGGER.info("%-24s AUC=%.4f (+/-%.4f)", name, rows[-1]["roc_auc"], rows[-1]["roc_auc_std"])
    return pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)


def tune(name: str, X: pd.DataFrame, y: pd.Series) -> tuple[Pipeline, dict, float]:
    """Grid-search one candidate, scoring on ROC-AUC."""
    model = candidate_models()[name]
    grid = PARAM_GRIDS.get(name)
    pipe = make_pipeline(model)
    if not grid:
        pipe.fit(X, y)
        return pipe, {}, float("nan")
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=cv, n_jobs=-1, refit=True)
    search.fit(X, y)
    LOGGER.info("%-24s tuned AUC=%.4f %s", name, search.best_score_, search.best_params_)
    return search.best_estimator_, search.best_params_, float(search.best_score_)


def evaluate(pipe: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    fpr, tpr, _ = roc_curve(y_test, proba)
    precision, recall, _ = precision_recall_curve(y_test, proba)
    prob_true, prob_pred = calibration_curve(y_test, proba, n_bins=10, strategy="quantile")
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    return {
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "average_precision": round(float(average_precision_score(y_test, proba)), 4),
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "precision": round(float(precision_score(y_test, pred)), 4),
        "recall": round(float(recall_score(y_test, pred)), 4),
        "f1": round(float(f1_score(y_test, pred)), 4),
        "brier": round(float(brier_score_loss(y_test, proba)), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "roc_curve": _curve(fpr, tpr),
        "pr_curve": _curve(recall, precision),
        "calibration_curve": [
            [round(float(p), 4), round(float(t), 4)] for p, t in zip(prob_pred, prob_true, strict=True)
        ],
        "n_test": int(len(y_test)),
    }


def importances(pipe: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, top: int = 18) -> list[dict]:
    """Permutation importance on the held-out split (model-agnostic, honest)."""
    result = permutation_importance(
        pipe,
        X_test,
        y_test,
        n_repeats=12,
        random_state=config.RANDOM_STATE,
        scoring="roc_auc",
        n_jobs=1,
    )
    rows = [
        {
            "feature": col,
            "importance": round(float(result.importances_mean[i]), 5),
            "std": round(float(result.importances_std[i]), 5),
        }
        for i, col in enumerate(X_test.columns)
    ]
    rows.sort(key=lambda r: r["importance"], reverse=True)
    return rows[:top]


def ranking_metrics(proba: np.ndarray, y_true: np.ndarray, ks=(10, 20, 50)) -> dict:
    """Top-K precision, which is how a recommender is judged in production."""
    order = np.argsort(-proba)
    out = {}
    base = float(np.mean(y_true))
    for k in ks:
        k = min(k, len(order))
        hits = float(np.mean(y_true[order[:k]]))
        out[f"precision_at_{k}"] = round(hits, 4)
        out[f"lift_at_{k}"] = round(hits / base, 3) if base else 0.0
    return out


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run(fast: bool = False) -> dict:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    started = time.perf_counter()

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = data.load_tracks()
    LOGGER.info("loaded %d tracks / %d artists", len(df), df["artist"].nunique())

    X = feature_matrix(df)
    y = df[config.TARGET].astype(int)

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X,
        y,
        np.arange(len(df)),
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )

    leaderboard = compare_models(X_train, y_train)

    if fast:
        best_name = str(leaderboard.iloc[0]["model"])
        best_pipe = make_pipeline(candidate_models()[best_name]).fit(X_train, y_train)
        best_params, tuned_cv = {}, float(leaderboard.iloc[0]["roc_auc"])
        tuning_table = []
    else:
        shortlist = [n for n in leaderboard["model"].head(3) if n in PARAM_GRIDS]
        tuning_table, tuned = [], {}
        for name in shortlist:
            pipe, params, score = tune(name, X_train, y_train)
            tuned[name] = (pipe, params, score)
            tuning_table.append({"model": name, "cv_roc_auc": round(score, 4), "best_params": params})
        best_name = max(tuned, key=lambda n: tuned[n][2])
        best_pipe, best_params, tuned_cv = tuned[best_name]

    LOGGER.info("winner: %s", best_name)

    # Probability calibration. Recommendation scores are blended with
    # similarity, so they have to mean something on an absolute scale.
    calibrated = CalibratedClassifierCV(best_pipe, method="isotonic", cv=5)
    calibrated.fit(X_train, y_train)

    raw_eval = evaluate(best_pipe, X_test, y_test)
    cal_eval = evaluate(calibrated, X_test, y_test)
    use_calibrated = cal_eval["brier"] <= raw_eval["brier"]
    final_pipe = calibrated if use_calibrated else best_pipe
    final_eval = cal_eval if use_calibrated else raw_eval
    LOGGER.info(
        "calibration: brier %.4f -> %.4f (%s)",
        raw_eval["brier"],
        cal_eval["brier"],
        "kept" if use_calibrated else "discarded",
    )

    perm = importances(best_pipe, X_test, y_test)
    final_eval["ranking"] = ranking_metrics(
        final_pipe.predict_proba(X_test)[:, 1], y_test.to_numpy()
    )

    # ---------------- artefacts ---------------- #
    joblib.dump(
        {"pipeline": final_pipe, "model_name": best_name, "calibrated": use_calibrated},
        config.TASTE_MODEL_PATH,
    )

    space = EmbeddingSpace().fit(engineer(df))
    vectors = space.transform(engineer(df))
    state = space.state_dict()
    np.savez_compressed(
        config.EMBEDDING_PATH,
        vectors=vectors,
        track_ids=df[config.ID_COLUMN].to_numpy().astype("U32"),
        columns=np.array(state["columns"], dtype="U32"),
        weights=np.array(state["weights"], dtype=np.float32),
        quantiles=state["quantiles"].astype(np.float32),
    )

    catalog = df.copy()
    catalog["p_like"] = final_pipe.predict_proba(X)[:, 1].round(4)
    catalog["in_test_split"] = False
    catalog.loc[idx_test, "in_test_split"] = True
    catalog.to_csv(config.CATALOG_PATH, index=False)

    # ---------------- report ---------------- #
    anova_df = stats.anova(df)
    anova_df["significant_fdr"] = stats.benjamini_hochberg(anova_df["p_value"].tolist())
    chi_df = stats.chi_square(df)

    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": data.describe(df),
        "split": {
            "train": int(len(X_train)),
            "test": int(len(X_test)),
            "test_size": config.TEST_SIZE,
            "stratified": True,
        },
        "leaderboard": leaderboard.to_dict(orient="records"),
        "tuning": tuning_table,
        "best_model": {
            "name": best_name,
            "params": {k.replace("model__", ""): v for k, v in (best_params or {}).items()},
            "cv_roc_auc": round(tuned_cv, 4) if tuned_cv == tuned_cv else None,
            "calibrated": use_calibrated,
            "calibration_gain_brier": round(raw_eval["brier"] - cal_eval["brier"], 4),
        },
        "test_metrics": final_eval,
        "permutation_importance": perm,
        "anova": anova_df.to_dict(orient="records"),
        "chi_square": chi_df.to_dict(orient="records"),
        "correlations": {
            "columns": config.AUDIO_FEATURES + [config.TARGET],
            "matrix": stats.correlations(df).to_numpy().tolist(),
        },
        "distributions": data.feature_distributions(df),
        "runtime_seconds": round(time.perf_counter() - started, 1),
    }

    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    LOGGER.info(
        "done in %.1fs | test AUC=%.4f acc=%.4f -> %s",
        metrics["runtime_seconds"],
        final_eval["roc_auc"],
        final_eval["accuracy"],
        config.METRICS_PATH,
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Resonance taste model.")
    parser.add_argument("--fast", action="store_true", help="skip hyper-parameter search")
    run(fast=parser.parse_args().fast)


if __name__ == "__main__":
    main()
