"""Tests for CodeTesterTool."""

import shutil
import tempfile
from pathlib import Path

import pytest
from assertpy import assert_that

from ai_unifier_assesment.agent.tools.code_tester_tool import (
    CodeTesterInput,
    CodeTesterTool,
)

# Check if cargo is available
CARGO_AVAILABLE = shutil.which("cargo") is not None


def test_python_passing_tests():
    """Test running passing Python tests."""
    tool = CodeTesterTool()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create a simple passing test
        test_file = temp_path / "test_sample.py"
        test_file.write_text(
            """
def test_passing():
    assert 1 + 1 == 2
"""
        )

        input_data = CodeTesterInput(
            working_directory=str(temp_path),
            language="python",
        )

        result = tool.test(input_data)

        assert_that(result.success).is_true()
        assert_that(result.exit_code).is_equal_to(0)
        assert_that(result.command).contains("pytest")


def test_python_failing_tests():
    """Test running failing Python tests."""
    tool = CodeTesterTool()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create a failing test
        test_file = temp_path / "test_sample.py"
        test_file.write_text(
            """
def test_failing():
    assert 1 + 1 == 3
"""
        )

        input_data = CodeTesterInput(
            working_directory=str(temp_path),
            language="python",
        )

        result = tool.test(input_data)

        assert_that(result.success).is_false()
        assert_that(result.exit_code).is_not_equal_to(0)
        assert_that(result.stdout).contains("FAILED")


@pytest.mark.skipif(not CARGO_AVAILABLE, reason="Cargo not installed")
def test_rust_passing_tests():
    """Test running passing Rust tests."""
    tool = CodeTesterTool()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create Cargo.toml
        cargo_toml = temp_path / "Cargo.toml"
        cargo_toml.write_text(
            """
[package]
name = "test_project"
version = "0.1.0"
edition = "2021"
"""
        )

        # Create src directory and lib.rs with passing test
        src_dir = temp_path / "src"
        src_dir.mkdir()
        lib_file = src_dir / "lib.rs"
        lib_file.write_text(
            """
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        assert_eq!(add(1, 2), 3);
    }
}
"""
        )

        input_data = CodeTesterInput(
            working_directory=str(temp_path),
            language="rust",
            timeout=60,  # Rust compilation can take longer
        )

        result = tool.test(input_data)

        assert_that(result.success).is_true()
        assert_that(result.exit_code).is_equal_to(0)
        assert_that(result.command).contains("cargo")


@pytest.mark.skipif(not CARGO_AVAILABLE, reason="Cargo not installed")
def test_rust_failing_tests():
    """Test running failing Rust tests."""
    tool = CodeTesterTool()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create Cargo.toml
        cargo_toml = temp_path / "Cargo.toml"
        cargo_toml.write_text(
            """
[package]
name = "test_project"
version = "0.1.0"
edition = "2021"
"""
        )

        # Create src directory and lib.rs with failing test
        src_dir = temp_path / "src"
        src_dir.mkdir()
        lib_file = src_dir / "lib.rs"
        lib_file.write_text(
            """
#[cfg(test)]
mod tests {
    #[test]
    fn test_failing() {
        assert_eq!(1 + 1, 3);
    }
}
"""
        )

        input_data = CodeTesterInput(
            working_directory=str(temp_path),
            language="rust",
            timeout=60,
        )

        result = tool.test(input_data)

        assert_that(result.success).is_false()
        assert_that(result.exit_code).is_not_equal_to(0)


def test_nonexistent_directory():
    """Test handling of nonexistent working directory."""
    tool = CodeTesterTool()

    input_data = CodeTesterInput(
        working_directory="/nonexistent/directory",
        language="python",
    )

    result = tool.test(input_data)

    assert_that(result.success).is_false()
    assert_that(result.exit_code).is_equal_to(-1)
    assert_that(result.stderr).contains("does not exist")


def test_python_syntax_error():
    """Test handling of Python syntax errors."""
    tool = CodeTesterTool()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test file with syntax error
        test_file = temp_path / "test_sample.py"
        test_file.write_text(
            """
def test_syntax_error():
    x = (  # Unclosed parenthesis
"""
        )

        input_data = CodeTesterInput(
            working_directory=str(temp_path),
            language="python",
        )

        result = tool.test(input_data)

        assert_that(result.success).is_false()
        # Syntax errors are captured in the test collection phase
        assert_that(result.stdout or result.stderr).is_not_empty()


@pytest.mark.skipif(not CARGO_AVAILABLE, reason="Cargo not installed")
def test_rust_compilation_error():
    """Test handling of Rust compilation errors."""
    tool = CodeTesterTool()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create Cargo.toml
        cargo_toml = temp_path / "Cargo.toml"
        cargo_toml.write_text(
            """
[package]
name = "test_project"
version = "0.1.0"
edition = "2021"
"""
        )

        # Create src directory with compilation error
        src_dir = temp_path / "src"
        src_dir.mkdir()
        lib_file = src_dir / "lib.rs"
        lib_file.write_text(
            """
fn broken_function() -> i32 {
    "not an integer"  // Type error
}
"""
        )

        input_data = CodeTesterInput(
            working_directory=str(temp_path),
            language="rust",
            timeout=60,
        )

        result = tool.test(input_data)

        assert_that(result.success).is_false()
        assert_that(result.exit_code).is_not_equal_to(0)


def test_timeout():
    """Test timeout handling."""
    tool = CodeTesterTool()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create a test that would run forever
        test_file = temp_path / "test_infinite.py"
        test_file.write_text(
            """
def test_infinite():
    import time
    time.sleep(999)
"""
        )

        input_data = CodeTesterInput(
            working_directory=str(temp_path),
            language="python",
            timeout=1,  # 1 second timeout
        )

        result = tool.test(input_data)

        assert_that(result.success).is_false()
        assert_that(result.exit_code).is_equal_to(-1)
        assert_that(result.stderr).contains("timed out")
