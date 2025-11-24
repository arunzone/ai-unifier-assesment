from unittest.mock import AsyncMock, Mock, patch

import pytest
from assertpy import assert_that

from ai_unifier_assesment.agent.trip_planner_agent import TripPlannerAgent
from ai_unifier_assesment.agent.state import TripItinerary


@pytest.fixture
def mock_model():
    model = Mock()
    chat_model = Mock()
    model.get_chat_model.return_value = chat_model
    return model


def test_agent_creates_expected_tools(mock_model):
    agent = TripPlannerAgent(mock_model)

    tool_names = [t.name for t in agent._tools]

    assert_that(tool_names).is_equal_to(
        [
            "search_flights",
            "get_weather_forecast",
            "search_attractions",
        ]
    )


@pytest.mark.asyncio
async def test_plan_trip_returns_expected_result_structure(mock_model):
    agent = TripPlannerAgent(mock_model)
    mock_itinerary = TripItinerary(
        destination="Auckland",
        duration_days=2,
        total_budget=500,
        actual_cost=450,
        flights=[],
        days=[],
        summary="A wonderful 2-day trip to Auckland",
    )

    with patch.object(agent, "_build_graph") as mock_build:
        mock_graph = Mock()
        mock_compiled = AsyncMock()
        mock_compiled.ainvoke.return_value = {
            "messages": [],
            "scratchpad": ["[User] Plan a 2-day trip to Auckland for under NZ$500"],
            "itinerary": mock_itinerary,
        }
        mock_graph.compile.return_value = mock_compiled
        mock_build.return_value = mock_graph

        result = await agent.plan_trip("Plan a 2-day trip to Auckland for under NZ$500")

        assert_that(result).is_equal_to(
            {
                "itinerary": {
                    "destination": "Auckland",
                    "duration_days": 2,
                    "total_budget": 500,
                    "actual_cost": 450,
                    "flights": [],
                    "days": [],
                    "summary": "A wonderful 2-day trip to Auckland",
                },
                "scratchpad": ["[User] Plan a 2-day trip to Auckland for under NZ$500"],
            }
        )


@pytest.mark.asyncio
async def test_plan_trip_returns_none_itinerary_when_not_generated(mock_model):
    agent = TripPlannerAgent(mock_model)

    with patch.object(agent, "_build_graph") as mock_build:
        mock_graph = Mock()
        mock_compiled = AsyncMock()
        mock_compiled.ainvoke.return_value = {
            "messages": [],
            "scratchpad": ["[User] Invalid request"],
            "itinerary": None,
        }
        mock_graph.compile.return_value = mock_compiled
        mock_build.return_value = mock_graph

        result = await agent.plan_trip("Invalid request")

        assert_that(result).is_equal_to(
            {
                "itinerary": None,
                "scratchpad": ["[User] Invalid request"],
            }
        )
