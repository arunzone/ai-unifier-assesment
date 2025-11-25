"""Tests for SelfHealingAgent."""

from unittest.mock import AsyncMock, Mock

import pytest
from assertpy import assert_that

from ai_unifier_assesment.agent.self_healing_agent import SelfHealingAgent
from ai_unifier_assesment.agent.tools.code_tester_tool import (
    CodeTesterOutput,
    CodeTesterTool,
)
from ai_unifier_assesment.agent.tools.code_writer_tool import (
    CodeWriterOutput,
    CodeWriterTool,
)


@pytest.fixture
def mock_model():
    """Mock LLM model."""
    model = Mock()
    simple_model = AsyncMock()
    simple_model.ainvoke = AsyncMock()
    model.simple_model.return_value = simple_model
    return model


@pytest.fixture
def mock_prompt_loader():
    """Mock prompt loader."""
    loader = Mock()
    loader.load = Mock(
        side_effect=lambda name: (
            "System prompt content" if name == "code_healing_system" else "Fix prompt: {previous_code} {test_output}"
        )
    )
    return loader


@pytest.fixture
def mock_code_writer():
    """Mock code writer tool."""
    writer = Mock(spec=CodeWriterTool)
    writer.write = Mock(
        return_value=CodeWriterOutput(
            success=True,
            file_path="/tmp/test.py",
            message="Success",
        )
    )
    return writer


@pytest.fixture
def mock_code_tester():
    """Mock code tester tool."""
    tester = Mock(spec=CodeTesterTool)
    return tester


@pytest.fixture
def mock_settings():
    """Mock settings."""
    return Mock()


@pytest.fixture
def agent(mock_model, mock_prompt_loader, mock_code_writer, mock_code_tester, mock_settings):
    """Create agent with mocked dependencies."""
    return SelfHealingAgent(
        model=mock_model,
        prompt_loader=mock_prompt_loader,
        code_writer=mock_code_writer,
        code_tester=mock_code_tester,
        settings=mock_settings,
    )


@pytest.mark.asyncio
async def test_successful_first_attempt(agent, mock_model, mock_code_tester):
    """Test successful code generation on first attempt."""
    # Mock LLM responses: first for language detection, then for code generation
    language_response = Mock()
    language_response.content = "python"

    code_response = Mock()
    code_response.content = """
FILE: main.py
```python
def add(a, b):
    return a + b
```

FILE: test_main.py
```python
from main import add

def test_add():
    assert add(1, 2) == 3
```
"""
    mock_model.simple_model.return_value.ainvoke.side_effect = [
        language_response,
        code_response,
    ]

    # Mock successful test execution
    mock_code_tester.test = Mock(
        return_value=CodeTesterOutput(
            success=True,
            stdout="All tests passed",
            stderr="",
            exit_code=0,
            command="pytest",
        )
    )

    result = await agent.heal("write an add function in Python")

    assert_that(result.success).is_true()
    assert_that(result.attempt_number).is_equal_to(0)
    assert_that(result.final_message).contains("Success")
    assert_that(result.language).is_equal_to("python")


@pytest.mark.asyncio
async def test_failure_after_max_attempts(agent, mock_model, mock_code_tester):
    """Test failure after exhausting max attempts."""
    # Mock LLM responses: language detection + 3 code generation attempts
    language_response = Mock()
    language_response.content = "rust"

    code_response = Mock()
    code_response.content = """
FILE: lib.rs
```rust
fn broken() {}
```
"""
    mock_model.simple_model.return_value.ainvoke.side_effect = [
        language_response,
        code_response,
        code_response,  # Retry 1
        code_response,  # Retry 2
    ]

    # Mock failing test execution
    mock_code_tester.test = Mock(
        return_value=CodeTesterOutput(
            success=False,
            stdout="",
            stderr="Compilation error",
            exit_code=1,
            command="cargo test",
        )
    )

    result = await agent.heal("write broken rust code")

    assert_that(result.success).is_false()
    assert_that(result.attempt_number).is_equal_to(2)  # 0, 1, 2 = 3 attempts
    assert_that(result.final_message).contains("Failed after 3 attempts")
    assert_that(result.language).is_equal_to("rust")


@pytest.mark.asyncio
async def test_successful_second_attempt(agent, mock_model, mock_code_tester):
    """Test success on second attempt after initial failure."""
    # Mock LLM responses: language detection + 2 code generation attempts
    language_response = Mock()
    language_response.content = "python"

    first_code_response = Mock()
    first_code_response.content = """
FILE: test.py
```python
def broken():
    return 1 / 0
```
"""

    second_code_response = Mock()
    second_code_response.content = """
FILE: test.py
```python
def fixed():
    return 42

def test_fixed():
    assert fixed() == 42
```
"""

    mock_model.simple_model.return_value.ainvoke.side_effect = [
        language_response,
        first_code_response,
        second_code_response,
    ]

    # Mock test results: fail then succeed
    mock_code_tester.test = Mock(
        side_effect=[
            CodeTesterOutput(
                success=False,
                stdout="",
                stderr="ZeroDivisionError",
                exit_code=1,
                command="pytest",
            ),
            CodeTesterOutput(
                success=True,
                stdout="1 passed",
                stderr="",
                exit_code=0,
                command="pytest",
            ),
        ]
    )

    result = await agent.heal("write a function in Python")

    assert_that(result.success).is_true()
    assert_that(result.attempt_number).is_equal_to(1)  # Success on second attempt
    assert_that(result.final_message).contains("Success")
    assert_that(result.language).is_equal_to("python")


@pytest.mark.asyncio
async def test_parse_code_files_with_file_markers(agent):
    """Test parsing code with FILE: markers."""
    code_content = """
FILE: main.py
```python
print("hello")
```

FILE: test_main.py
```python
def test_main():
    pass
```
"""

    files = agent._parse_code_files(code_content, "python")

    assert_that(files).contains_key("main.py", "test_main.py")
    assert_that(files["main.py"]).contains('print("hello")')
    assert_that(files["test_main.py"]).contains("def test_main")


@pytest.mark.asyncio
async def test_parse_code_files_fallback(agent):
    """Test fallback parsing when FILE: markers are missing."""
    code_content = """
```python
def hello():
    return "world"
```

```python
def test_hello():
    assert hello() == "world"
```
"""

    files = agent._parse_code_files(code_content, "python")

    # Should create default filenames
    assert_that(files).is_not_empty()
    assert_that(len(files)).is_greater_than(0)


@pytest.mark.asyncio
async def test_format_test_output(agent):
    """Test formatting of test output."""
    formatted = agent._format_test_output("stdout content", "stderr content")

    assert_that(formatted).contains("STDERR")
    assert_that(formatted).contains("STDOUT")
    assert_that(formatted).contains("stderr content")
    assert_that(formatted).contains("stdout content")


@pytest.mark.asyncio
async def test_format_test_output_empty(agent):
    """Test formatting when output is empty."""
    formatted = agent._format_test_output("", "")

    assert_that(formatted).contains("No output captured")


@pytest.mark.asyncio
async def test_rust_code_structure(agent):
    """Test that Rust code goes to lib.rs."""
    code_content = """
```rust
fn main() {}
```
"""

    files = agent._parse_code_files(code_content, "rust")

    assert_that(files).contains_key("lib.rs")


@pytest.mark.asyncio
async def test_agent_loads_correct_system_prompts(agent, mock_prompt_loader):
    """Test that agent loads the correct prompt files."""
    language_response = Mock()
    language_response.content = "python"

    llm_response = Mock()
    llm_response.content = "FILE: test.py\n```python\npass\n```"
    agent._model.simple_model.return_value.ainvoke.side_effect = [
        language_response,
        llm_response,
    ]

    agent._code_tester.test = Mock(
        return_value=CodeTesterOutput(success=True, stdout="", stderr="", exit_code=0, command="pytest")
    )

    await agent.heal("test task")

    # Should be called twice: once for language detection, once for system prompt
    assert_that(mock_prompt_loader.load.call_count).is_equal_to(2)
    mock_prompt_loader.load.assert_any_call("language_detection")
    mock_prompt_loader.load.assert_any_call("code_healing_system")


@pytest.mark.asyncio
async def test_agent_uses_fix_prompt_on_retry(agent, mock_prompt_loader, mock_model):
    """Test that agent uses fix prompt on retry attempts."""
    language_response = Mock()
    language_response.content = "python"

    first_response = Mock()
    first_response.content = "FILE: test.py\n```python\npass\n```"

    second_response = Mock()
    second_response.content = "FILE: test.py\n```python\nfixed\n```"

    mock_model.simple_model.return_value.ainvoke.side_effect = [
        language_response,
        first_response,
        second_response,
    ]

    agent._code_tester.test = Mock(
        side_effect=[
            CodeTesterOutput(success=False, stdout="", stderr="error", exit_code=1, command="pytest"),
            CodeTesterOutput(success=True, stdout="", stderr="", exit_code=0, command="pytest"),
        ]
    )

    await agent.heal("test task")

    mock_prompt_loader.load.assert_any_call("code_healing_fix")
