"""
Pytest configuration — makes the project root importable and marks
tests that need live API keys so they're skipped in CI by default.
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def pytest_collection_modifyitems(config, items):
    """Auto-skip tests marked `live` when the API key is missing."""
    if os.getenv("GOOGLE_API_KEY"):
        return
    skip_live = pytest.mark.skip(reason="GOOGLE_API_KEY required for live tests")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)


def pytest_configure(config):
    config.addinivalue_line("markers", "live: requires live API key (Google Gemini)")
