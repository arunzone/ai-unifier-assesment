# tests/test_app.py
# Unit tests for the application

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from app import main


def test_main(capsys):
    """Test that main runs and prints output."""
    main()
    captured = capsys.readouterr()
    assert "App is running!" in captured.out
