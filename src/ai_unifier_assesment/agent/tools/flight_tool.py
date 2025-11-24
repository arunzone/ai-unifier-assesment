from pydantic import BaseModel


class FlightSearchInput(BaseModel):
    origin: str
    destination: str
    date: str


class FlightResult(BaseModel):
    airline: str
    flight_number: str
    departure_time: str
    arrival_time: str
    price: float


class FlightSearchOutput(BaseModel):
    origin: str
    destination: str
    date: str
    flights: list[FlightResult]


class FlightTool:
    """Tool for searching flights between cities."""

    def search(self, input: FlightSearchInput) -> FlightSearchOutput:
        """Search for available flights.

        Args:
            input: Search parameters including origin, destination, and date.

        Returns:
            FlightSearchOutput with available flight options.
        """
        # Mock flight data with fixed prices for deterministic results
        mock_flights = [
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
        ]

        return FlightSearchOutput(
            origin=input.origin,
            destination=input.destination,
            date=input.date,
            flights=mock_flights,
        )
