from unittest.mock import AsyncMock, Mock, patch

import pytest
from assertpy import assert_that

from ai_unifier_assesment.agent.trip_planner_agent import TripPlannerAgent
from ai_unifier_assesment.agent.state import TripItinerary
from ai_unifier_assesment.agent.tools.flight_tool import FlightTool
from ai_unifier_assesment.agent.tools.weather_tool import WeatherTool
from ai_unifier_assesment.agent.tools.attractions_tool import AttractionsTool


@pytest.fixture
def mock_model():
    model = Mock()
    model.simple_model.return_value.bind_tools.return_value = Mock()
    model.get_chat_model_for_evaluation.return_value.with_structured_output.return_value = Mock()
    return model


@pytest.fixture
def flight_tool():
    return FlightTool()


@pytest.fixture
def weather_tool():
    return WeatherTool()


@pytest.fixture
def attractions_tool():
    return AttractionsTool()


def test_agent_creates_expected_tools(mock_model, flight_tool, weather_tool, attractions_tool):
    agent = TripPlannerAgent(mock_model, flight_tool, weather_tool, attractions_tool)

    tool_names = [t.name for t in agent._tools]

    assert_that(tool_names).is_equal_to(
        [
            "search_flights",
            "get_weather_forecast",
            "search_attractions",
        ]
    )


@pytest.mark.asyncio
async def test_plan_trip_returns_expected_result_structure(mock_model, flight_tool, weather_tool, attractions_tool):
    agent = TripPlannerAgent(mock_model, flight_tool, weather_tool, attractions_tool)
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
            }
        )


@pytest.mark.asyncio
async def test_plan_trip_returns_none_itinerary_when_not_generated(
    mock_model, flight_tool, weather_tool, attractions_tool
):
    agent = TripPlannerAgent(mock_model, flight_tool, weather_tool, attractions_tool)

    with patch.object(agent, "_build_graph") as mock_build:
        mock_graph = Mock()
        mock_compiled = AsyncMock()
        mock_compiled.ainvoke.return_value = {
            "messages": [],
            "itinerary": None,
        }
        mock_graph.compile.return_value = mock_compiled
        mock_build.return_value = mock_graph

        result = await agent.plan_trip("Invalid request")

        assert_that(result).is_equal_to(
            {
                "itinerary": None,
            }
        )
