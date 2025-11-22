import os
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def mock_env_vars():
    env_vars = {
        "OPENAI_BASE_URL": "https://test.example.com",
        "OPENAI_API_KEY": "sk-test-key",
    }
    with patch.dict(os.environ, env_vars, clear=False):
        yield
