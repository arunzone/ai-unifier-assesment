from assertpy import assert_that

from ai_unifier_assesment.agent.tools.flight_tool import (
    FlightTool,
    FlightSearchInput,
    FlightSearchOutput,
    FlightResult,
)


def test_search_returns_expected_flights():
    tool = FlightTool()
    input = FlightSearchInput(origin="Wellington", destination="Auckland", date="2024-03-15")

    result = tool.search(input)

    assert_that(result).is_equal_to(
        FlightSearchOutput(
            origin="Wellington",
            destination="Auckland",
            date="2024-03-15",
            flights=[
                FlightResult(
                    airline="Air New Zealand",
                    flight_number="NZ123",
                    departure_time="06:00",
                    arrival_time="08:30",
                    price=120.0,
                ),
                FlightResult(
                    airline="Jetstar",
                    flight_number="JQ201",
                    departure_time="10:30",
                    arrival_time="13:00",
                    price=85.0,
                ),
                FlightResult(
                    airline="Air New Zealand",
                    flight_number="NZ456",
                    departure_time="14:00",
                    arrival_time="16:30",
                    price=135.0,
                ),
                FlightResult(
                    airline="Qantas",
                    flight_number="QF45",
                    departure_time="18:00",
                    arrival_time="20:30",
                    price=150.0,
                ),
            ],
        )
    )
