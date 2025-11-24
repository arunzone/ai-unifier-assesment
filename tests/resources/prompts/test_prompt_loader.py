from unittest.mock import patch, MagicMock

import pytest
from assertpy import assert_that

from ai_unifier_assesment.resources.prompts.prompt_loader import PromptLoader


def test_load_returns_prompt_content_when_file_exists():
    loader = PromptLoader()

    with patch.object(loader, "_prompts_dir") as mock_dir:
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_path.read_text.return_value = "  Test prompt content  "
        mock_dir.__truediv__.return_value = mock_path

        result = loader.load("test_prompt")

        assert_that(result).is_equal_to("Test prompt content")


def test_load_raises_file_not_found_when_prompt_missing():
    loader = PromptLoader()

    with patch.object(loader, "_prompts_dir") as mock_dir:
        mock_path = MagicMock()
        mock_path.exists.return_value = False
        mock_dir.__truediv__.return_value = mock_path

        with pytest.raises(FileNotFoundError) as exc_info:
            loader.load("nonexistent_prompt")

        assert_that(str(exc_info.value)).contains("Prompt file not found")
