"""
src/utils/config.py
--------------------
Central, single-source configuration. Everything that varies between
environments (paths, model choice, GenAI provider settings) lives here and
is read from environment variables so nothing is hardcoded in source files.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Config:
    # Paths
    project_root: Path = PROJECT_ROOT
    models_dir: Path = PROJECT_ROOT / "models"
    reports_dir: Path = PROJECT_ROOT / "reports"
    data_dir: Path = PROJECT_ROOT / "data"

    # Reproducibility
    random_seed: int = int(os.environ.get("RANDOM_SEED", "42"))

    # Model artifacts
    pipeline_filename: str = "preprocessing_pipeline.pkl"
    model_metadata_filename: str = "metadata.json"

    # GenAI
    openai_api_key: str | None = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY"))
    openai_model: str = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    genai_timeout_seconds: float = float(os.environ.get("GENAI_TIMEOUT_SECONDS", "20"))

    # API
    api_host: str = os.environ.get("API_HOST", "0.0.0.0")
    api_port: int = int(os.environ.get("API_PORT", "8000"))


CONFIG = Config()
