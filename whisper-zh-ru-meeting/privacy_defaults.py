#!/usr/bin/env python3
"""Privacy-first process defaults for optional third-party model runtimes.

Meeting audio and transcripts are processed locally. Model downloads are the
only supported runtime network operation and must be explicitly enabled.
"""
from __future__ import annotations

import os


_TRUE = {"1", "true", "yes", "on"}
DOWNLOAD_OPT_IN = "WHISPER_MEETING_ALLOW_MODEL_DOWNLOADS"


def model_downloads_allowed() -> bool:
    """Return whether this process has explicit permission to fetch models."""
    return os.environ.get(DOWNLOAD_OPT_IN, "").strip().lower() in _TRUE


def configure_local_privacy() -> bool:
    """Disable supported telemetry and default model hubs to offline mode."""
    os.environ["DO_NOT_TRACK"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["HF_HUB_DISABLE_UPDATE_CHECK"] = "1"
    os.environ["PYANNOTE_METRICS_ENABLED"] = "0"
    allowed = model_downloads_allowed()
    if not allowed:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    return allowed


MODEL_DOWNLOADS_ALLOWED = configure_local_privacy()
