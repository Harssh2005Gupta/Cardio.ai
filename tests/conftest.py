"""
tests/conftest.py
--------------------
Shared fixtures. Tests never require a real dataset or a real OPENAI_API_KEY:
a small synthetic dataset (same schema as the real one) is generated once
per test session and used to actually train artifacts that other tests
exercise.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def synthetic_df() -> pd.DataFrame:
    rng = np.random.RandomState(0)
    n = 200
    df = pd.DataFrame(
        {
            "age": rng.randint(29, 78, n),
            "sex": rng.randint(0, 2, n),
            "cp": rng.randint(0, 4, n),
            "trestbps": rng.randint(94, 201, n),
            "chol": rng.randint(126, 565, n),
            "fbs": rng.randint(0, 2, n),
            "restecg": rng.randint(0, 3, n),
            "thalach": rng.randint(71, 203, n),
            "exang": rng.randint(0, 2, n),
            "oldpeak": np.round(rng.uniform(0, 6.2, n), 1),
            "slope": rng.randint(0, 3, n),
            "ca": rng.randint(0, 5, n),
            "thal": rng.randint(1, 4, n),
        }
    )
    score = df["cp"] * 0.5 + df["thalach"] * 0.01 - df["age"] * 0.03 + df["exang"] * 1.2 + rng.normal(0, 1, n)
    df["target"] = (score > np.median(score)).astype(int)
    return df


@pytest.fixture(scope="session")
def trained_models_dir(tmp_path_factory, synthetic_df) -> Path:
    """Actually runs the real training pipeline once against synthetic data
    and points CONFIG at the resulting artifacts for the rest of the session."""
    from src.utils.config import CONFIG
    import src.models.train as train_mod

    tmp_dir = tmp_path_factory.mktemp("cardio_ai_test_artifacts")
    data_path = tmp_dir / "heart.csv"
    synthetic_df.to_csv(data_path, index=False)

    models_dir = tmp_dir / "models"
    reports_dir = tmp_dir / "reports"

    object.__setattr__(CONFIG, "models_dir", models_dir)
    object.__setattr__(CONFIG, "reports_dir", reports_dir)

    train_mod.train(str(data_path))
    return models_dir
