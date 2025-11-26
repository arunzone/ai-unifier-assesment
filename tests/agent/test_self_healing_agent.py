"""Tests for CodingAgent - one assert per test, clear and minimal."""

from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
from assertpy import assert_that
from langchain_core.messages import AIMessage

from ai_unifier_assesment.agent.coding_agent import CodingAgent
from ai_unifier_assesment.agent.state import CodeHealingState
from ai_unifier_assesment.agent.tools.code_writer_tool import CodeWriterOutput
from ai_unifier_assesment.agent.tools.tester_models import CodeTesterOutput


@pytest.fixture
def agent():
    """Create CodingAgent with mocked dependencies."""
    mock_model = Mock()
    mock_model.simple_model.return_value = AsyncMock()

    mock_prompt_loader = Mock()
    mock_prompt_loader.load.return_value = "System prompt"

    mock_code_writer = Mock()
    mock_code_writer.write.return_value = CodeWriterOutput(success=True, file_path="/tmp/test.py", message="Written")

    mock_code_tester = Mock()
    mock_code_tester.test.return_value = CodeTesterOutput(
        success=True, stdout="passed", stderr="", exit_code=0, command="pytest"
    )

    mock_event_processor = Mock()
    mock_settings = Mock()
    mock_language_detector = AsyncMock()
    mock_initial_code_generator = AsyncMock()

    return CodingAgent(
        model=mock_model,
        prompt_loader=mock_prompt_loader,
        code_writer=mock_code_writer,
        code_tester=mock_code_tester,
        event_processor=mock_event_processor,
        settings=mock_settings,
        language_detector=mock_language_detector,
        initial_code_generator=mock_initial_code_generator,
    )


def test_should_create_temp_directory_with_language_prefix(agent):
    state = CodeHealingState(task_description="Test", language="python", working_directory="")

    result = agent._setup_working_directory_node(state)

    assert_that(result["working_directory"]).contains("code_healing_python")


def test_should_create_existing_temp_directory(agent):
    state = CodeHealingState(task_description="Test", language="rust", working_directory="")

    result = agent._setup_working_directory_node(state)
    working_dir = Path(result["working_directory"])

    assert_that(working_dir.exists()).is_true()


@pytest.mark.asyncio
async def test_should_fix_code_using_test_output(agent):
    agent._prompt_loader.load.return_value = "Fix: {previous_code}\nError: {test_output}"
    agent._model.simple_model.return_value.ainvoke.return_value = AIMessage(
        content="def fib(n): return n if n <= 1 else fib(n-1) + fib(n-2)"
    )
    state = CodeHealingState(
        task_description="Write fibonacci",
        language="python",
        working_directory="/tmp",
        current_code="def fib(n): return n",
        test_output="AssertionError: fib(5) != 5",
    )

    result = await agent._fix_code(state)

    assert_that(result.current_code).contains("fib(n-1) + fib(n-2)")


# Code Parsing Tests


def test_should_parse_files_with_file_markers(agent):
    code_content = (
        "FILE: main.py\n```python\nprint('hello')\n```\n\nFILE: test_main.py\n```python\ndef test(): pass\n```"
    )

    files = agent._parse_code_files(code_content, "python")

    assert files == {"main.py": "print('hello')", "test_main.py": "def test(): pass"}


def test_should_extract_filename_from_file_marker(agent):
    code_content = "FILE: my_module.py\n```python\nx = 1\n```"

    files = agent._parse_code_files(code_content, "python")

    assert "my_module.py" in files


def test_should_fallback_parse_python_test_code(agent):
    code_content = "```python\nimport pytest\ndef test_add(): pass\n```"

    files = agent._fallback_parse(code_content, "python")

    assert files == {"test_main.py": "import pytest\ndef test_add(): pass"}


def test_should_fallback_parse_python_main_code(agent):
    code_content = "```python\ndef add(a, b): return a + b\n```"

    files = agent._fallback_parse(code_content, "python")

    assert files == {"main.py": "def add(a, b): return a + b"}


def test_should_fallback_parse_rust_code_to_lib_rs(agent):
    code_content = "```rust\nfn add(a: i32, b: i32) -> i32 { a + b }\n```"

    files = agent._fallback_parse(code_content, "rust")

    assert files == {"lib.rs": "fn add(a: i32, b: i32) -> i32 { a + b }"}


def test_should_return_empty_dict_when_no_code_blocks_found(agent):
    code_content = "Just plain text with no code blocks"

    files = agent._parse_code_files(code_content, "python")

    assert files == {}


# Test Execution Tests


def test_should_mark_success_when_tests_pass(agent):
    agent._code_tester.test.return_value = CodeTesterOutput(
        success=True, stdout="All passed", stderr="", exit_code=0, command="pytest"
    )
    state = CodeHealingState(task_description="Test", language="python", working_directory="/tmp")

    result = agent._run_tests(state)

    assert result.success is True


def test_should_mark_failure_when_tests_fail(agent):
    agent._code_tester.test.return_value = CodeTesterOutput(
        success=False, stdout="", stderr="AssertionError", exit_code=1, command="pytest"
    )
    state = CodeHealingState(task_description="Test", language="python", working_directory="/tmp")

    result = agent._run_tests(state)

    assert result.success is False


def test_should_capture_test_output_from_stdout_and_stderr(agent):
    agent._code_tester.test.return_value = CodeTesterOutput(
        success=False, stdout="Test output", stderr="Error output", exit_code=1, command="pytest"
    )
    state = CodeHealingState(task_description="Test", language="python", working_directory="/tmp")

    result = agent._run_tests(state)

    assert "STDOUT:\nTest output" in result.test_output and "STDERR:\nError output" in result.test_output


# Decision Logic Tests


def test_should_decide_success_when_tests_pass(agent):
    state = CodeHealingState(
        task_description="Test", language="python", working_directory="/tmp", success=True, attempt_number=0
    )

    decision = agent._decide_next_step(state)

    assert decision == "success"


def test_should_decide_retry_when_tests_fail_and_attempts_remain(agent):
    state = CodeHealingState(
        task_description="Test", language="python", working_directory="/tmp", success=False, attempt_number=0
    )

    decision = agent._decide_next_step(state)

    assert decision == "retry"


def test_should_decide_failure_when_max_attempts_reached(agent):
    state = CodeHealingState(
        task_description="Test", language="python", working_directory="/tmp", success=False, attempt_number=2
    )

    decision = agent._decide_next_step(state)

    assert decision == "failure"


# Attempt Counter Tests


def test_should_increment_attempt_from_zero_to_one(agent):
    state = CodeHealingState(task_description="Test", language="python", working_directory="/tmp", attempt_number=0)

    result = agent._increment_attempt_node(state)

    assert result == {"attempt_number": 1}


def test_should_increment_attempt_from_one_to_two(agent):
    state = CodeHealingState(task_description="Test", language="python", working_directory="/tmp", attempt_number=1)

    result = agent._increment_attempt_node(state)

    assert result == {"attempt_number": 2}


# Finalization Tests


def test_should_return_success_message_on_successful_completion(agent):
    state = CodeHealingState(
        task_description="Test",
        language="python",
        working_directory="/tmp",
        success=True,
        final_message="Success! All tests passed on attempt 1",
        current_code="def fib(n): return n",
        attempt_number=0,
    )

    result = agent._finalize_node(state)

    assert result["final_message"] == "Success! All tests passed on attempt 1"


def test_should_return_failure_message_with_error_details(agent):
    state = CodeHealingState(
        task_description="Test",
        language="python",
        working_directory="/tmp",
        success=False,
        test_output="AssertionError: test failed",
        attempt_number=2,
    )

    result = agent._finalize_node(state)

    assert (
        "Failed after 3 attempts" in result["final_message"]
        and "AssertionError: test failed" in result["final_message"]
    )


def test_should_return_working_directory_in_finalize(agent):
    state = CodeHealingState(
        task_description="Test",
        language="python",
        working_directory="/tmp/test_dir",
        success=True,
        final_message="Success",
        current_code="code",
        attempt_number=0,
    )

    result = agent._finalize_node(state)

    assert result["working_directory"] == "/tmp/test_dir"


def test_should_return_final_code_in_finalize(agent):
    state = CodeHealingState(
        task_description="Test",
        language="python",
        working_directory="/tmp",
        success=True,
        final_message="Success",
        current_code="def fib(n): return n",
        attempt_number=0,
    )

    result = agent._finalize_node(state)

    assert result["final_code"] == "def fib(n): return n"


def test_should_return_total_attempts_in_finalize(agent):
    state = CodeHealingState(
        task_description="Test",
        language="python",
        working_directory="/tmp",
        success=True,
        final_message="Success",
        current_code="code",
        attempt_number=2,
    )

    result = agent._finalize_node(state)

    assert result["attempts"] == 3
