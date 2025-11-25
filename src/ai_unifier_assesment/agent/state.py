from datetime import date
from typing import Annotated, Optional

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class TripConstraints(BaseModel):
    destination: str
    duration_days: int
    budget: float
    start_date: Optional[date] = None
    preferences: list[str] = Field(default_factory=list)


class FlightOption(BaseModel):
    airline: str
    departure_time: str
    arrival_time: str
    price: float
    flight_number: str


class WeatherInfo(BaseModel):
    date: str
    condition: str
    temperature_high: float
    temperature_low: float
    precipitation_chance: int


class Attraction(BaseModel):
    name: str
    description: str
    category: str
    estimated_cost: float
    duration_hours: float
    location: str


class Activity(BaseModel):
    time: str
    name: str
    description: str
    cost: float
    duration_hours: float


class DayPlan(BaseModel):
    day: int
    date: str
    weather: WeatherInfo
    activities: list[Activity]
    daily_cost: float


class TripItinerary(BaseModel):
    destination: str
    duration_days: int
    total_budget: float
    actual_cost: float
    flights: list[FlightOption]
    days: list[DayPlan]
    summary: str


class TripPlannerState(BaseModel):
    messages: Annotated[list[AnyMessage], add_messages]
    constraints: Optional[TripConstraints] = None
    flight_options: list[FlightOption] = Field(default_factory=list)
    weather_forecast: list[WeatherInfo] = Field(default_factory=list)
    attractions: list[Attraction] = Field(default_factory=list)
    itinerary: Optional[TripItinerary] = None

    class Config:
        arbitrary_types_allowed = True


class CodeHealingState(BaseModel):
    task_description: str = Field(description="Natural language coding task")
    language: str = Field(description="Programming language (python or rust)")
    working_directory: str = Field(description="Directory for code and tests")
    current_code: Optional[str] = Field(default=None, description="Current version of the code")
    test_output: Optional[str] = Field(default=None, description="Output from test execution")
    attempt_number: int = Field(default=0, description="Current attempt number (0-2)")
    success: bool = Field(default=False, description="Whether tests passed")
    final_message: str = Field(default="", description="Final status message")

    class Config:
        arbitrary_types_allowed = True
