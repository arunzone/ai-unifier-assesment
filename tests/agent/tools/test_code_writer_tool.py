"""Tests for CodeWriterTool."""

import tempfile
from pathlib import Path

from assertpy import assert_that

from ai_unifier_assesment.agent.tools.code_writer_tool import (
    CodeWriterInput,
    CodeWriterTool,
)


def test_write_python_file():
    """Test writing a Python file."""
    tool = CodeWriterTool()

    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "test.py"

        input_data = CodeWriterInput(
            code='print("hello world")',
            file_path=str(file_path),
            language="python",
        )

        result = tool.write(input_data)

        assert_that(result.success).is_true()
        assert_that(result.file_path).is_equal_to(str(file_path))
        assert_that(file_path.exists()).is_true()
        assert_that(file_path.read_text()).is_equal_to('print("hello world")')


def test_write_rust_file():
    """Test writing a Rust file."""
    tool = CodeWriterTool()

    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "lib.rs"

        input_data = CodeWriterInput(
            code='fn main() { println!("hello"); }',
            file_path=str(file_path),
            language="rust",
        )

        result = tool.write(input_data)

        assert_that(result.success).is_true()
        assert_that(result.file_path).is_equal_to(str(file_path))
        assert_that(file_path.exists()).is_true()
        assert_that(file_path.read_text()).contains("println!")


def test_write_creates_parent_directories():
    """Test that parent directories are created if they don't exist."""
    tool = CodeWriterTool()

    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "subdir" / "nested" / "test.py"

        input_data = CodeWriterInput(
            code="# test",
            file_path=str(file_path),
            language="python",
        )

        result = tool.write(input_data)

        assert_that(result.success).is_true()
        assert_that(file_path.exists()).is_true()
        assert_that(file_path.parent.exists()).is_true()


def test_write_fails_with_wrong_extension():
    """Test that writing fails if extension doesn't match language."""
    tool = CodeWriterTool()

    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "test.py"

        input_data = CodeWriterInput(
            code="fn main() {}",
            file_path=str(file_path),
            language="rust",  # Language is rust but extension is .py
        )

        result = tool.write(input_data)

        assert_that(result.success).is_false()
        assert_that(result.message).contains("does not match language")


def test_write_overwrites_existing_file():
    """Test that existing files are overwritten."""
    tool = CodeWriterTool()

    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "test.py"

        # Write initial content
        file_path.write_text("old content")

        # Overwrite with new content
        input_data = CodeWriterInput(
            code="new content",
            file_path=str(file_path),
            language="python",
        )

        result = tool.write(input_data)

        assert_that(result.success).is_true()
        assert_that(file_path.read_text()).is_equal_to("new content")


def test_write_handles_invalid_path():
    """Test handling of invalid file paths."""
    tool = CodeWriterTool()

    # Try to write to an invalid location (contains null byte)
    input_data = CodeWriterInput(
        code="test",
        file_path="/tmp/test\x00.py",  # Invalid path with null byte
        language="python",
    )

    result = tool.write(input_data)

    # Should fail gracefully
    assert_that(result.success).is_false()
    assert_that(result.message).is_not_empty()
