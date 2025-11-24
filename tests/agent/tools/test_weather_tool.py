from assertpy import assert_that

from ai_unifier_assesment.agent.tools.weather_tool import (
    WeatherTool,
    WeatherInput,
    WeatherOutput,
    DayWeather,
)


def test_get_forecast_returns_expected_weather():
    tool = WeatherTool()
    input = WeatherInput(location="Auckland", start_date="2024-03-15", days=2)

    result = tool.get_forecast(input)

    assert_that(result).is_equal_to(
        WeatherOutput(
            location="Auckland",
            forecast=[
                DayWeather(
                    date="2024-03-15",
                    condition="Sunny",
                    temperature_high=22.0,
                    temperature_low=15.0,
                    precipitation_chance=10,
                ),
                DayWeather(
                    date="2024-03-16",
                    condition="Partly Cloudy",
                    temperature_high=20.0,
                    temperature_low=14.0,
                    precipitation_chance=25,
                ),
            ],
        )
    )
