"""Shared pytest fixtures for taskbench tests."""

import pytest


def pytest_collection_modifyitems(config, items):
    """Mark tests in test_*_smoke.py as slow so they're easy to skip."""
    for item in items:
        if "smoke" in item.nodeid:
            item.add_marker(pytest.mark.slow)
