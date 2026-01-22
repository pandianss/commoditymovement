import pytest
import os
import sys

def test_imports():
    """
    Tier 4: System Tests (Smoke Test)
    Verify that all key modules can be imported without error.
    """
    try:
        from src.api.main import app
        from src.intelligence.event_alignment import EventAligner
        from src.contracts.dataset_spec import DatasetSpec
        from src.run_daily_update import main as run_daily_update_main
        assert True
    except ImportError as e:
        pytest.fail(f"System import failed: {e}")
