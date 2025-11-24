from unittest.mock import AsyncMock

import pytest
from assertpy import assert_that
from fastapi.testclient import TestClient

from ai_unifier_assesment.app import app
from ai_unifier_assesment.agent.trip_planner_agent import TripPlannerAgent


@pytest.fixture
def mock_agent():
    return AsyncMock(spec=TripPlannerAgent)


@pytest.fixture
def client(mock_agent):
    app.dependency_overrides[TripPlannerAgent] = lambda: mock_agent
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_plan_trip_returns_expected_response(client, mock_agent):
    mock_agent.plan_trip.return_value = {
        "itinerary": {
            "destination": "Auckland",
            "duration_days": 2,
            "total_budget": 500,
            "actual_cost": 420,
            "flights": [
                {
                    "airline": "Air New Zealand",
                    "flight_number": "NZ123",
                    "departure_time": "06:00",
                    "arrival_time": "08:30",
                    "price": 100,
                }
            ],
            "days": [
                {
                    "day": 1,
                    "date": "2024-03-15",
                    "weather": {
                        "date": "2024-03-15",
                        "condition": "Sunny",
                        "temperature_high": 22,
                        "temperature_low": 15,
                        "precipitation_chance": 10,
                    },
                    "activities": [
                        {
                            "time": "10:00",
                            "name": "Sky Tower",
                            "description": "Visit iconic tower",
                            "cost": 32,
                            "duration_hours": 2,
                        }
                    ],
                    "daily_cost": 32,
                }
            ],
            "summary": "A budget-friendly 2-day Auckland adventure",
        },
        "scratchpad": [
            "[User] Plan a 2-day trip to Auckland for under NZ$500",
            "[Agent] Searching for flights...",
            "[Agent] Generated final itinerary with total cost: $420",
        ],
    }

    response = client.post(
        "/api/plan-trip",
        json={"prompt": "Plan a 2-day trip to Auckland for under NZ$500"},
    )

    assert_that(response.json()).is_equal_to(
        {
            "itinerary": {
                "destination": "Auckland",
                "duration_days": 2,
                "total_budget": 500,
                "actual_cost": 420,
                "flights": [
                    {
                        "airline": "Air New Zealand",
                        "flight_number": "NZ123",
                        "departure_time": "06:00",
                        "arrival_time": "08:30",
                        "price": 100,
                    }
                ],
                "days": [
                    {
                        "day": 1,
                        "date": "2024-03-15",
                        "weather": {
                            "date": "2024-03-15",
                            "condition": "Sunny",
                            "temperature_high": 22,
                            "temperature_low": 15,
                            "precipitation_chance": 10,
                        },
                        "activities": [
                            {
                                "time": "10:00",
                                "name": "Sky Tower",
                                "description": "Visit iconic tower",
                                "cost": 32,
                                "duration_hours": 2,
                            }
                        ],
                        "daily_cost": 32,
                    }
                ],
                "summary": "A budget-friendly 2-day Auckland adventure",
            },
            "scratchpad": [
                "[User] Plan a 2-day trip to Auckland for under NZ$500",
                "[Agent] Searching for flights...",
                "[Agent] Generated final itinerary with total cost: $420",
            ],
        }
    )


def test_plan_trip_returns_null_itinerary_on_failure(client, mock_agent):
    mock_agent.plan_trip.return_value = {
        "itinerary": None,
        "scratchpad": ["[User] Invalid request", "[Agent] Error parsing itinerary"],
    }

    response = client.post(
        "/api/plan-trip",
        json={"prompt": "Invalid request"},
    )

    assert_that(response.json()).is_equal_to(
        {
            "itinerary": None,
            "scratchpad": ["[User] Invalid request", "[Agent] Error parsing itinerary"],
        }
    )
