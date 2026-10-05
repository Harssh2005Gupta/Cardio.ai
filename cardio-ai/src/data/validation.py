"""
src/data/validation.py
------------------------
Explicit, fail-loud validation for both the training dataset (batch) and a
single patient record submitted at inference time. Nothing here silently
accepts bad data — every failure returns a clear, user-facing message.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import pandas as pd

REQUIRED_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
]

# (min, max) plausible physiological ranges — generous bounds, not clinical
# thresholds, just enough to catch obviously corrupted data (e.g. age=-5).
NUMERIC_RANGES = {
    "age": (0, 120),
    "trestbps": (0, 300),
    "chol": (0, 800),
    "thalach": (0, 260),
    "oldpeak": (-3, 10),
}

CATEGORICAL_ALLOWED = {
    "sex": {0, 1},
    "cp": {0, 1, 2, 3},
    "fbs": {0, 1},
    "restecg": {0, 1, 2},
    "exang": {0, 1},
    "slope": {0, 1, 2},
    "ca": {0, 1, 2, 3, 4},
    "thal": {1, 2, 3},
}


@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def raise_if_invalid(self) -> None:
        if not self.is_valid:
            raise ValueError("Validation failed:\n" + "\n".join(f"- {e}" for e in self.errors))


def validate_dataset(df: pd.DataFrame) -> ValidationResult:
    """Validate a full training dataframe before it touches any pipeline."""
    errors: List[str] = []
    warnings: List[str] = []

    if df is None or len(df) == 0:
        return ValidationResult(False, ["Dataset is empty."])

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")

    unexpected_cols = [c for c in df.columns if c not in REQUIRED_COLUMNS]
    if unexpected_cols:
        warnings.append(f"Unexpected extra columns will be ignored: {unexpected_cols}")

    if errors:
        # Can't safely check dtypes/ranges if columns are missing.
        return ValidationResult(False, errors, warnings)

    if "target" not in df.columns:
        errors.append("Target column 'target' is missing.")
    elif df["target"].isnull().any():
        errors.append("Target column contains missing values.")
    elif not set(df["target"].dropna().unique()).issubset({0, 1}):
        errors.append("Target column must be binary (0/1).")

    for col, (lo, hi) in NUMERIC_RANGES.items():
        if col not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(f"Column '{col}' must be numeric.")
            continue
        out_of_range = df[(df[col] < lo) | (df[col] > hi)]
        if len(out_of_range) > 0:
            warnings.append(f"{len(out_of_range)} row(s) have '{col}' outside plausible range [{lo}, {hi}].")

    for col, allowed in CATEGORICAL_ALLOWED.items():
        if col not in df.columns:
            continue
        bad_values = set(df[col].dropna().unique()) - allowed
        if bad_values:
            errors.append(f"Column '{col}' has values outside {sorted(allowed)}: {sorted(bad_values)}")

    n_dupes = df.duplicated().sum()
    if n_dupes > 0:
        warnings.append(f"{n_dupes} duplicate row(s) found.")

    missing_counts = df[REQUIRED_COLUMNS].isnull().sum()
    cols_with_missing = missing_counts[missing_counts > 0]
    if len(cols_with_missing) > 0:
        warnings.append(f"Missing values present: {cols_with_missing.to_dict()}")

    return ValidationResult(is_valid=(len(errors) == 0), errors=errors, warnings=warnings)


def validate_patient_record(record: dict) -> ValidationResult:
    """Validate a single patient record submitted for inference (no 'target')."""
    errors: List[str] = []
    required = [c for c in REQUIRED_COLUMNS if c != "target"]

    missing = [c for c in required if c not in record or record[c] is None]
    if missing:
        errors.append(f"Missing required fields: {missing}")
        return ValidationResult(False, errors)

    for col, (lo, hi) in NUMERIC_RANGES.items():
        val = record.get(col)
        try:
            val = float(val)
        except (TypeError, ValueError):
            errors.append(f"Field '{col}' must be numeric, got: {record.get(col)!r}")
            continue
        if not (lo <= val <= hi):
            errors.append(f"Field '{col}'={val} is outside plausible range [{lo}, {hi}].")

    for col, allowed in CATEGORICAL_ALLOWED.items():
        val = record.get(col)
        try:
            val = int(val)
        except (TypeError, ValueError):
            errors.append(f"Field '{col}' must be an integer code, got: {record.get(col)!r}")
            continue
        if val not in allowed:
            errors.append(f"Field '{col}'={val} not in allowed values {sorted(allowed)}.")

    return ValidationResult(is_valid=(len(errors) == 0), errors=errors)
