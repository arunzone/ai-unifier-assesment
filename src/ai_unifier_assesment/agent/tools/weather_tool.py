from datetime import datetime, timedelta

from pydantic import BaseModel


class WeatherInput(BaseModel):
    location: str
    start_date: str
    days: int


class DayWeather(BaseModel):
    date: str
    condition: str
    temperature_high: float
    temperature_low: float
    precipitation_chance: int


class WeatherOutput(BaseModel):
    location: str
    forecast: list[DayWeather]


class WeatherTool:
    """Tool for getting weather forecasts."""

    # Fixed weather patterns for deterministic results
    _weather_patterns = [
        ("Sunny", 22.0, 15.0, 10),
        ("Partly Cloudy", 20.0, 14.0, 25),
        ("Clear", 23.0, 16.0, 5),
        ("Cloudy", 18.0, 13.0, 35),
        ("Light Rain", 17.0, 12.0, 60),
    ]

    def get_forecast(self, input: WeatherInput) -> WeatherOutput:
        """Get weather forecast for a location.

        Args:
            input: Location and date range for forecast.

        Returns:
            WeatherOutput with daily forecasts.
        """
        forecast = []
        start = datetime.strptime(input.start_date, "%Y-%m-%d")

        for i in range(input.days):
            current_date = start + timedelta(days=i)
            pattern = self._weather_patterns[i % len(self._weather_patterns)]

            forecast.append(
                DayWeather(
                    date=current_date.strftime("%Y-%m-%d"),
                    condition=pattern[0],
                    temperature_high=pattern[1],
                    temperature_low=pattern[2],
                    precipitation_chance=pattern[3],
                )
            )

        return WeatherOutput(
            location=input.location,
            forecast=forecast,
        )
