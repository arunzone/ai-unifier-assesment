"""Tests for code healing API endpoints."""

from unittest.mock import Mock

import pytest
from assertpy import assert_that
from fastapi.testclient import TestClient

from ai_unifier_assesment.agent.self_healing_agent import SelfHealingAgent
from ai_unifier_assesment.agent.state import CodeHealingState
from ai_unifier_assesment.app import app


@pytest.fixture
def mock_agent():
    """Mock self-healing agent."""
    agent = Mock(spec=SelfHealingAgent)
    return agent


@pytest.fixture
def client(mock_agent):
    """Test client with mocked agent."""
    app.dependency_overrides[SelfHealingAgent] = lambda: mock_agent
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_heal_code_success(client, mock_agent):
    """Test successful code healing endpoint."""
    # Mock successful healing
    mock_state = CodeHealingState(
        task_description="write quicksort",
        language="python",
        working_directory="/tmp/test",
        current_code="def quicksort(arr): pass",
        test_output="All tests passed",
        attempt_number=0,
        success=True,
        final_message="Success! All tests passed on attempt 1",
    )

    mock_agent.heal.return_value = mock_state

    response = client.post(
        "/api/heal-code",
        json={
            "task_description": "write quicksort in Python",
            "language": "python",
        },
    )

    assert_that(response.status_code).is_equal_to(200)
    data = response.json()
    assert_that(data["success"]).is_true()
    assert_that(data["attempts"]).is_equal_to(1)
    assert_that(data["final_code"]).contains("quicksort")
    assert_that(data["message"]).contains("Success")


@pytest.mark.asyncio
async def test_heal_code_failure(client, mock_agent):
    """Test failed code healing endpoint."""
    # Mock failed healing
    mock_state = CodeHealingState(
        task_description="write broken code",
        language="rust",
        working_directory="/tmp/test",
        current_code="fn broken() {}",
        test_output="Compilation failed",
        attempt_number=2,
        success=False,
        final_message="Failed after 3 attempts",
    )

    mock_agent.heal.return_value = mock_state

    response = client.post(
        "/api/heal-code",
        json={
            "task_description": "write broken code",
            "language": "rust",
        },
    )

    assert_that(response.status_code).is_equal_to(200)
    data = response.json()
    assert_that(data["success"]).is_false()
    assert_that(data["attempts"]).is_equal_to(3)
    assert_that(data["message"]).contains("Failed")


@pytest.mark.asyncio
async def test_heal_code_invalid_language(client):
    """Test with invalid language."""
    response = client.post(
        "/api/heal-code",
        json={
            "task_description": "write code",
            "language": "javascript",  # Not supported
        },
    )

    # Should fail validation
    assert_that(response.status_code).is_equal_to(422)


@pytest.mark.asyncio
async def test_heal_code_missing_fields(client):
    """Test with missing required fields."""
    response = client.post(
        "/api/heal-code",
        json={
            "task_description": "write code",
            # Missing language field
        },
    )

    assert_that(response.status_code).is_equal_to(422)


@pytest.mark.asyncio
async def test_heal_code_python_language(client, mock_agent):
    """Test with Python language."""
    mock_state = CodeHealingState(
        task_description="test",
        language="python",
        working_directory="/tmp/test",
        success=True,
        final_message="Success",
    )

    mock_agent.heal.return_value = mock_state

    response = client.post(
        "/api/heal-code",
        json={
            "task_description": "write binary search",
            "language": "python",
        },
    )

    assert_that(response.status_code).is_equal_to(200)
    # Verify agent was called with correct parameters
    mock_agent.heal.assert_called_once()
    call_args = mock_agent.heal.call_args[0]
    assert_that(call_args[0]).is_equal_to("write binary search")
    assert_that(call_args[1]).is_equal_to("python")


@pytest.mark.asyncio
async def test_heal_code_rust_language(client, mock_agent):
    """Test with Rust language."""
    mock_state = CodeHealingState(
        task_description="test",
        language="rust",
        working_directory="/tmp/test",
        success=True,
        final_message="Success",
    )

    mock_agent.heal.return_value = mock_state

    response = client.post(
        "/api/heal-code",
        json={
            "task_description": "write quicksort in Rust",
            "language": "rust",
        },
    )

    assert_that(response.status_code).is_equal_to(200)
    mock_agent.heal.assert_called_once()
    call_args = mock_agent.heal.call_args[0]
    assert_that(call_args[1]).is_equal_to("rust")


@pytest.mark.asyncio
async def test_heal_code_includes_working_directory(client, mock_agent):
    """Test that response includes working directory."""
    mock_state = CodeHealingState(
        task_description="test",
        language="python",
        working_directory="/tmp/code_healing_12345",
        success=True,
        final_message="Success",
    )

    mock_agent.heal.return_value = mock_state

    response = client.post(
        "/api/heal-code",
        json={
            "task_description": "test task",
            "language": "python",
        },
    )

    assert_that(response.status_code).is_equal_to(200)
    data = response.json()
    assert_that(data["working_directory"]).is_equal_to("/tmp/code_healing_12345")


@pytest.mark.asyncio
async def test_heal_code_agent_raises_value_error(client, mock_agent):
    """Test handling of ValueError from agent."""
    mock_agent.heal.side_effect = ValueError("Invalid input")

    response = client.post(
        "/api/heal-code",
        json={
            "task_description": "test",
            "language": "python",
        },
    )

    assert_that(response.status_code).is_equal_to(400)
    assert_that(response.json()["detail"]).contains("Invalid input")


@pytest.mark.asyncio
async def test_heal_code_agent_raises_unexpected_error(client, mock_agent):
    """Test handling of unexpected errors from agent."""
    mock_agent.heal.side_effect = RuntimeError("Unexpected error")

    response = client.post(
        "/api/heal-code",
        json={
            "task_description": "test",
            "language": "python",
        },
    )

    assert_that(response.status_code).is_equal_to(500)
    assert_that(response.json()["detail"]).contains("Internal server error")
