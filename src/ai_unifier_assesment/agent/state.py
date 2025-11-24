import operator
from datetime import date
from typing import Annotated, Optional

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class TripConstraints(BaseModel):
    """Constraints extracted from user prompt."""

    destination: str
    duration_days: int
    budget: float
    start_date: Optional[date] = None
    preferences: list[str] = Field(default_factory=list)


class FlightOption(BaseModel):
    """Flight search result."""

    airline: str
    departure_time: str
    arrival_time: str
    price: float
    flight_number: str


class WeatherInfo(BaseModel):
    """Weather forecast for a day."""

    date: str
    condition: str
    temperature_high: float
    temperature_low: float
    precipitation_chance: int


class Attraction(BaseModel):
    """Attraction or activity."""

    name: str
    description: str
    category: str
    estimated_cost: float
    duration_hours: float
    location: str


class Activity(BaseModel):
    """Planned activity in itinerary."""

    time: str
    name: str
    description: str
    cost: float
    duration_hours: float


class DayPlan(BaseModel):
    """Plan for a single day."""

    day: int
    date: str
    weather: WeatherInfo
    activities: list[Activity]
    daily_cost: float


class TripItinerary(BaseModel):
    """Final trip itinerary output."""

    destination: str
    duration_days: int
    total_budget: float
    actual_cost: float
    flights: list[FlightOption]
    days: list[DayPlan]
    summary: str


class TripPlannerState(BaseModel):
    """State for the trip planner agent."""

    messages: Annotated[list[AnyMessage], add_messages]
    constraints: Optional[TripConstraints] = None
    scratchpad: Annotated[list[str], operator.add] = Field(default_factory=list)
    flight_options: list[FlightOption] = Field(default_factory=list)
    weather_forecast: list[WeatherInfo] = Field(default_factory=list)
    attractions: list[Attraction] = Field(default_factory=list)
    itinerary: Optional[TripItinerary] = None

    class Config:
        arbitrary_types_allowed = True
