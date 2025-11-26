import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest
from assertpy import assert_that

from ai_unifier_assesment.agent.code_writer_service import CodeWriterService
from ai_unifier_assesment.agent.language import Language
from ai_unifier_assesment.agent.state import CodeHealingState
from ai_unifier_assesment.agent.tools.code_writer_tool import CodeWriterOutput
from ai_unifier_assesment.config import Settings

PYTHOM_MAIN_FILE_CONTENT = """
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

if __name__ == "__main__":
    print(quicksort([3, 6, 8, 10, 1, 2, 1]))
"""
PYTOHN_TEST_FILE_CONTENT = """
import pytest
from quicksort import quicksort

def test_quicksort():
    assert quicksort([]) == []
"""

PYTHON_CODE_BLOCK = f"""
FILE: quicksort.py
```python
{PYTHOM_MAIN_FILE_CONTENT}
```
FILE: test_quicksort.py
```python
{PYTOHN_TEST_FILE_CONTENT}
```
"""


@pytest.fixture
def mock_code_writer():
    """Create a mock code writer."""
    mock_writer = Mock()
    mock_writer.write.return_value = CodeWriterOutput(
        success=True, file_path="/tmp/test.py", message="Written successfully"
    )
    return mock_writer


@pytest.fixture
def code_writer_service(mock_code_writer):
    """Create CodeWriterService with mocked dependencies."""
    settings = Settings()
    return CodeWriterService(code_writer=mock_code_writer, settings=settings)


def test_should_write_main_python_file_with_file_markers(code_writer_service, mock_code_writer):
    """Test writing Python files to disk using FILE: markers."""
    with tempfile.TemporaryDirectory() as temp_dir:
        state = CodeHealingState(
            task_description="Write quicksort in Python",
            language=Language.PYTHON,
            working_directory=temp_dir,
            current_code=PYTHON_CODE_BLOCK,
        )

        code_writer_service.write_code_to_disk(state)

        calls = mock_code_writer.write.call_args_list
        file_paths = [call[0][0].file_path for call in calls]

        assert_that(file_paths).contains(str(Path(temp_dir) / "quicksort.py"))


def test_should_write_test_python_file_with_file_markers(code_writer_service, mock_code_writer):
    """Test writing Python files to disk using FILE: markers."""
    with tempfile.TemporaryDirectory() as temp_dir:
        state = CodeHealingState(
            task_description="Write quicksort in Python",
            language=Language.PYTHON,
            working_directory=temp_dir,
            current_code=PYTHON_CODE_BLOCK,
        )

        code_writer_service.write_code_to_disk(state)

        calls = mock_code_writer.write.call_args_list
        file_paths = [call[0][0].file_path for call in calls]

        assert_that(file_paths).contains(str(Path(temp_dir) / "test_quicksort.py"))


def test_should_write_main_python_file_content(code_writer_service, mock_code_writer):
    """Test writing Python files to disk using FILE: markers."""
    with tempfile.TemporaryDirectory() as temp_dir:
        state = CodeHealingState(
            task_description="Write quicksort in Python",
            language=Language.PYTHON,
            working_directory=temp_dir,
            current_code=PYTHON_CODE_BLOCK,
        )

        code_writer_service.write_code_to_disk(state)

        calls = mock_code_writer.write.call_args_list
        quicksort_call = [call for call in calls if "quicksort.py" in call[0][0].file_path][0]
        assert_that(quicksort_call[0][0].code.strip()).contains(PYTHOM_MAIN_FILE_CONTENT.strip())


def test_should_write_test_python_file_content(code_writer_service, mock_code_writer):
    """Test writing Python files to disk using FILE: markers."""
    with tempfile.TemporaryDirectory() as temp_dir:
        state = CodeHealingState(
            task_description="Write quicksort in Python",
            language=Language.PYTHON,
            working_directory=temp_dir,
            current_code=PYTHON_CODE_BLOCK,
        )

        code_writer_service.write_code_to_disk(state)

        calls = mock_code_writer.write.call_args_list
        test_call = [call for call in calls if "test_quicksort.py" in call[0][0].file_path][0]
        assert_that(test_call[0][0].code.strip()).contains(PYTOHN_TEST_FILE_CONTENT.strip())
