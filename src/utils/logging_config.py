"""
src/utils/logging_config.py
----------------------------
Standard library logging setup. Deliberately does NOT log request bodies or
environment variables wholesale, to avoid accidentally leaking API keys or
patient data into logs.
"""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            stream=sys.stdout,
        )
        _CONFIGURED = True
    return logging.getLogger(name)


def redact(value: str, keep: int = 4) -> str:
    """Redact a secret for safe logging, e.g. 'sk-abc...xyz1' -> 'sk-a***xyz1'."""
    if not value or len(value) <= keep * 2:
        return "***"
    return f"{value[:keep]}***{value[-keep:]}"
