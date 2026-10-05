"""
src/models/train.py
----------------------
Leak-free training pipeline — v2.

Methodology (the two things v1 got wrong, fixed here):

  1. PREPROCESSING IS NOW INSIDE THE CV PIPELINE. Classical models are
     wrapped as sklearn Pipeline([("preprocessing", ...), ("classifier", ...)])
     and GridSearchCV'd directly on the RAW training split, so every CV fold
     fits its own preprocessing on only that fold's training rows. (v1 fit
     preprocessing once on the whole training split before CV — not a
     test-set leak, but not as rigorous as it should be either.)

  2. THE TEST SET IS NEVER USED TO SELECT THE BEST MODEL. v1 picked the
     overall winner by comparing all 5 candidates' TEST ROC-AUC — a real
     leak into model selection. Here, the winner is chosen purely from
     K-fold CV performance (same StratifiedKFold splitter for every
     candidate, classical AND the DL model, for a fair comparison). The
     test set is touched exactly once, at the very end, to evaluate ONLY
     the already-selected, already-calibrated winner.

Full order of operations:

    1. Load + validate raw dataset
    2. Train/test split (stratified)                    <- split before anything is fit
    3. K-fold CV on the TRAINING split only:
         - classical models: Pipeline(preprocessing, classifier) + GridSearchCV
         - DL model: manual StratifiedKFold loop, preprocessing refit per fold,
           an inner sub-split used only for early-stopping (never the scored fold)
    4. Select best_model_type by mean CV ROC-AUC ONLY
    5. Refit preprocessing on the FULL training split; refit the winning
       model type on the fully-transformed training split
    6. If the winner is a classical model, wrap it in CalibratedClassifierCV
       (sigmoid/Platt scaling, its own internal 5-fold CV on training data
       only) to improve probability calibration. DL models are not
       calibrated this way — documented limitation, not silently skipped.
    7. Evaluate the final (calibrated, if applicable) model EXACTLY ONCE on
       the untouched test set
    8. Save model + preprocessing pipeline + full metadata (CV results for
       all 5 candidates, calibration before/after Brier score, dataset hash,
       timestamp, seed, model version)

Usage:
    python -m src.models.train --data data/heart.csv
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.data.preprocessing import build_pipeline, save_pipeline, transform_to_df
from src.data.validation import validate_dataset
from src.models.evaluate import compute_metrics, save_evaluation_plots
from src.models.model_registry import build_metadata, dataset_hash, save_metadata
from src.utils.config import CONFIG
from src.utils.logging_config import get_logger
from src.utils.reproducibility import set_global_seed

logger = get_logger(__name__)

N_CV_FOLDS = 5

CLASSICAL_MODEL_GRID = {
    "logistic_regression": (
        LogisticRegression(max_iter=2000, random_state=CONFIG.random_seed),
        {"C": [0.01, 0.1, 1, 10]},
    ),
    "decision_tree": (
        DecisionTreeClassifier(random_state=CONFIG.random_seed),
        {"max_depth": [None, 5, 10, 15], "min_samples_leaf": [1, 5, 10]},
    ),
    "random_forest": (
        RandomForestClassifier(random_state=CONFIG.random_seed),
        {"n_estimators": [100, 200], "max_depth": [None, 5, 10], "min_samples_leaf": [1, 2, 4]},
    ),
    "svm": (
        SVC(probability=True, random_state=CONFIG.random_seed),
        {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"]},
    ),
}


def build_dl_model(input_dim: int):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    tf.random.set_seed(CONFIG.random_seed)
    model = keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(32, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy", keras.metrics.AUC(name="auc")])
    return model


def _train_dl_for_scoring(X_sub_train, y_sub_train, X_sub_val, y_sub_val):
    """Train a DL model for CV scoring only: early-stops on an inner
    sub-validation split, never on the fold being scored."""
    from tensorflow.keras.callbacks import EarlyStopping

    model = build_dl_model(X_sub_train.shape[1])
    model.fit(
        X_sub_train.values, y_sub_train.values,
        validation_data=(X_sub_val.values, y_sub_val.values),
        epochs=150, batch_size=16, verbose=0,
        callbacks=[EarlyStopping(monitor="val_auc", mode="max", patience=15, restore_best_weights=True)],
    )
    return model


def cv_score_dl(X_train_raw: pd.DataFrame, y_train: pd.Series, skf: StratifiedKFold) -> dict:
    """Manual K-fold CV for the DL model, with preprocessing refit per fold
    (matching what Pipeline+GridSearchCV does automatically for classical
    models) and early stopping on an inner sub-split that is never the
    scored fold."""
    fold_scores = []
    for fold_i, (train_idx, val_idx) in enumerate(skf.split(X_train_raw, y_train)):
        fold_train_raw = X_train_raw.iloc[train_idx]
        fold_val_raw = X_train_raw.iloc[val_idx]
        fold_y_train = y_train.iloc[train_idx]
        fold_y_val = y_train.iloc[val_idx]

        pipeline = build_pipeline()
        pipeline.fit(fold_train_raw)
        fold_train_processed = transform_to_df(pipeline, fold_train_raw)
        fold_val_processed = transform_to_df(pipeline, fold_val_raw)

        # Inner split of the fold's training portion, ONLY for early stopping.
        sub_train_X, sub_val_X, sub_train_y, sub_val_y = train_test_split(
            fold_train_processed, fold_y_train, test_size=0.2,
            random_state=CONFIG.random_seed, stratify=fold_y_train,
        )
        model = _train_dl_for_scoring(sub_train_X, sub_train_y, sub_val_X, sub_val_y)

        # Score on the fold's held-out validation portion — never used for
        # early stopping, so this is a clean out-of-fold score.
        y_proba = model.predict(fold_val_processed.values, verbose=0).ravel()
        score = roc_auc_score(fold_y_val, y_proba)
        fold_scores.append(float(score))
        logger.info(f"    DL fold {fold_i + 1}/{skf.get_n_splits()}: ROC-AUC={score:.3f}")

    return {"mean_roc_auc": float(np.mean(fold_scores)), "std_roc_auc": float(np.std(fold_scores)), "fold_scores": fold_scores}


def cv_score_classical(X_train_raw: pd.DataFrame, y_train: pd.Series, skf: StratifiedKFold):
    """Pipeline(preprocessing, classifier) + GridSearchCV for every classical
    model type — preprocessing is refit inside every fold automatically."""
    cv_results = {}
    best_params_by_type = {}
    best_estimator_by_type = {}

    for model_type, (estimator, grid) in CLASSICAL_MODEL_GRID.items():
        logger.info(f"Cross-validating {model_type} (preprocessing + classifier fit per fold)...")
        pipeline = Pipeline([("preprocessing", build_pipeline()), ("classifier", estimator)])
        prefixed_grid = {f"classifier__{k}": v for k, v in grid.items()}

        search = GridSearchCV(pipeline, prefixed_grid, cv=skf, scoring="roc_auc", n_jobs=-1)
        search.fit(X_train_raw, y_train)

        fold_scores = [
            search.cv_results_[f"split{i}_test_score"][search.best_index_]
            for i in range(skf.get_n_splits())
        ]
        cv_results[model_type] = {
            "mean_roc_auc": float(search.best_score_),
            "std_roc_auc": float(np.std(fold_scores)),
            "fold_scores": [float(s) for s in fold_scores],
            "best_params": search.best_params_,
        }
        best_params_by_type[model_type] = {
            k.replace("classifier__", ""): v for k, v in search.best_params_.items()
        }
        best_estimator_by_type[model_type] = search.best_estimator_.named_steps["classifier"]
        logger.info(f"  {model_type}: mean CV ROC-AUC={search.best_score_:.3f} (+/- {cv_results[model_type]['std_roc_auc']:.3f})")

    return cv_results, best_params_by_type, best_estimator_by_type


def train(data_path: str) -> None:
    set_global_seed(CONFIG.random_seed)
    models_dir = CONFIG.models_dir
    reports_dir = CONFIG.reports_dir
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} rows from {data_path}")

    validation = validate_dataset(df)
    for w in validation.warnings:
        logger.warning(w)
    validation.raise_if_invalid()

    fp = dataset_hash(df)
    logger.info(f"Dataset fingerprint: {fp}")

    X_raw = df.drop(columns=["target"])
    y = df["target"]

    # 1. SPLIT FIRST. Test set is not referenced again until the final block.
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=CONFIG.random_seed, stratify=y
    )

    # 2. K-fold CV on the training split only — same splitter for every
    #    candidate so the comparison is apples-to-apples.
    skf = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=CONFIG.random_seed)

    cv_results, best_params_by_type, best_estimator_by_type = cv_score_classical(X_train_raw, y_train, skf)

    logger.info("Cross-validating neural_network_dl (preprocessing + model fit per fold)...")
    cv_results["neural_network_dl"] = cv_score_dl(X_train_raw, y_train, skf)
    logger.info(f"  neural_network_dl: mean CV ROC-AUC={cv_results['neural_network_dl']['mean_roc_auc']:.3f} "
                f"(+/- {cv_results['neural_network_dl']['std_roc_auc']:.3f})")

    # 3. Select the winner from CV ONLY. The test set has not been touched yet.
    best_model_type = max(cv_results, key=lambda k: cv_results[k]["mean_roc_auc"])
    logger.info(f"\nSelected by CV (test set not used): {best_model_type} "
                f"(mean CV ROC-AUC={cv_results[best_model_type]['mean_roc_auc']:.3f})")

    # 4. Refit preprocessing + the winning model type on the FULL training split.
    final_pipeline = build_pipeline()
    final_pipeline.fit(X_train_raw)
    save_pipeline(final_pipeline, models_dir)

    X_train_processed = transform_to_df(final_pipeline, X_train_raw)
    X_test_processed = transform_to_df(final_pipeline, X_test_raw)  # transform-only

    X_train_processed.sample(min(100, len(X_train_processed)), random_state=CONFIG.random_seed).to_csv(
        models_dir / "background_sample.csv", index=False
    )

    calibration_applied = False
    brier_before = None
    brier_after = None

    if best_model_type == "neural_network_dl":
        logger.info("Training final DL model on full training split...")
        final_model = train_dl_model_full(X_train_processed, y_train, models_dir)
        y_proba = final_model.predict(X_test_processed.values, verbose=0).ravel()
        brier_before = brier_after = float(brier_score_loss(y_test, y_proba))
    else:
        base_estimator = clone(best_estimator_by_type[best_model_type])
        base_estimator.set_params(**best_params_by_type[best_model_type])

        # Uncalibrated fit, purely to report the before/after Brier comparison
        # as part of this single final evaluation pass (not model selection).
        uncal_fit = clone(base_estimator)
        uncal_fit.fit(X_train_processed, y_train)
        y_proba_uncal = uncal_fit.predict_proba(X_test_processed)[:, 1]
        brier_before = float(brier_score_loss(y_test, y_proba_uncal))

        logger.info(f"Calibrating {best_model_type} with sigmoid (Platt) scaling...")
        calibrated = CalibratedClassifierCV(estimator=clone(base_estimator), method="sigmoid", cv=5)
        calibrated.fit(X_train_processed, y_train)
        final_model = calibrated
        calibration_applied = True
        y_proba = calibrated.predict_proba(X_test_processed)[:, 1]
        brier_after = float(brier_score_loss(y_test, y_proba))
        logger.info(f"  Brier score: {brier_before:.4f} (uncalibrated) -> {brier_after:.4f} (calibrated)")

    # 5. Single final evaluation on the untouched test set.
    y_pred = (y_proba >= 0.5).astype(int)
    test_metrics = compute_metrics(y_test, y_pred, y_proba)
    save_evaluation_plots(best_model_type, y_test, y_proba, y_pred, reports_dir)
    logger.info(f"[FINAL TEST, computed once] {best_model_type}: "
                f"ROC-AUC={test_metrics['roc_auc']:.3f} Brier={test_metrics['brier_score']:.4f}")

    if best_model_type == "neural_network_dl":
        final_model.save(models_dir / "best_model.keras")
    else:
        joblib.dump(final_model, models_dir / "best_model.pkl")

    metadata = build_metadata(
        best_model_type=best_model_type,
        cv_results=cv_results,
        test_metrics=test_metrics,
        dataset_fingerprint=fp,
        n_rows=len(df),
        n_features=X_train_processed.shape[1],
        random_seed=CONFIG.random_seed,
        cv_folds=N_CV_FOLDS,
        selection_method=(
            "Best mean ROC-AUC across StratifiedKFold(n_splits=5) cross-validation on the "
            "training split only. Test set was never used for model selection."
        ),
        calibration_applied=calibration_applied,
        brier_before=brier_before,
        brier_after=brier_after,
    )
    save_metadata(metadata, models_dir)

    comparison_table = build_comparison_table_cv(cv_results, best_model_type, test_metrics)
    (reports_dir / "model_comparison.md").write_text(comparison_table)
    logger.info("\n" + comparison_table)
    logger.info(f"Saved all artifacts to {models_dir.resolve()}/ and reports to {reports_dir.resolve()}/")


def train_dl_model_full(X_train_processed: pd.DataFrame, y_train: pd.Series, out_dir: Path):
    """Final production DL model, trained on the full training split with an
    internal validation carve-out used only for early stopping/checkpointing
    (not for model selection — that already happened via cv_score_dl)."""
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = build_dl_model(X_train_processed.shape[1])

    checkpoint_path = out_dir / "dl_checkpoint.keras"
    callbacks = [
        EarlyStopping(monitor="val_auc", mode="max", patience=15, restore_best_weights=True),
        ModelCheckpoint(str(checkpoint_path), monitor="val_auc", mode="max", save_best_only=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-5),
    ]
    history = model.fit(
        X_train_processed.values, y_train.values,
        validation_split=0.2, epochs=200, batch_size=16, verbose=0, callbacks=callbacks,
    )
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(out_dir / "dl_training_history.json", "w") as f:
        json.dump(history_dict, f, indent=2)
    _plot_training_curves(history_dict, out_dir)
    return model


def _plot_training_curves(history: dict, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].plot(history["loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Loss"); axes[0].set_xlabel("epoch"); axes[0].legend()

    auc_key = "auc" if "auc" in history else "accuracy"
    axes[1].plot(history[auc_key], label="train")
    axes[1].plot(history[f"val_{auc_key}"], label="val")
    axes[1].set_title(auc_key.upper()); axes[1].set_xlabel("epoch"); axes[1].legend()

    fig.tight_layout()
    fig.savefig(Path(out_dir) / "dl_training_curves.png", dpi=120)
    plt.close(fig)


def build_comparison_table_cv(cv_results: dict, best_model_type: str, test_metrics: dict) -> str:
    header = "| Model | CV ROC-AUC (mean +/- std) | Selected |"
    sep = "|---|---|---|"
    rows = [header, sep]
    for name, r in cv_results.items():
        marker = "**YES**" if name == best_model_type else ""
        rows.append(f"| {name} | {r['mean_roc_auc']:.3f} +/- {r['std_roc_auc']:.3f} | {marker} |")
    rows.append("")
    rows.append(f"**Final test-set evaluation (computed once, winner only: `{best_model_type}`):**")
    rows.append("")
    rows.append("| Metric | Value |")
    rows.append("|---|---|")
    for k in ["accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc", "brier_score"]:
        rows.append(f"| {k} | {test_metrics[k]:.4f} |")
    return "\n".join(rows)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to heart.csv (UCI-style schema)")
    args = parser.parse_args()
    train(args.data)
