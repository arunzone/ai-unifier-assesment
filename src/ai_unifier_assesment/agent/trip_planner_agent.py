import logging
from typing import Annotated

from fastapi import Depends
from langchain_core.globals import set_verbose, set_debug

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from ai_unifier_assesment.agent.state import TripPlannerState, TripItinerary
from ai_unifier_assesment.agent.tools.flight_tool import FlightTool, FlightSearchInput
from ai_unifier_assesment.agent.tools.weather_tool import WeatherTool, WeatherInput
from ai_unifier_assesment.agent.tools.attractions_tool import AttractionsTool, AttractionsInput
from ai_unifier_assesment.large_language_model.model import Model
from ai_unifier_assesment.resources.prompts.prompt_loader import prompt_loader

logger = logging.getLogger(__name__)


class TripPlannerAgent:
    def __init__(self, model: Annotated[Model, Depends(Model)]):
        self._model = model
        self._flight_tool = FlightTool()
        self._weather_tool = WeatherTool()
        self._attractions_tool = AttractionsTool()
        self._tools = self._create_tools()

    def _create_tools(self) -> list[StructuredTool]:
        return [
            StructuredTool.from_function(
                func=lambda origin, destination, date: self._flight_tool.search(
                    FlightSearchInput(origin=origin, destination=destination, date=date)
                ).model_dump_json(),
                name="search_flights",
                description="Search for flights between cities. Returns flight options with prices.",
                args_schema=FlightSearchInput,
            ),
            StructuredTool.from_function(
                func=lambda location, start_date, days: self._weather_tool.get_forecast(
                    WeatherInput(location=location, start_date=start_date, days=days)
                ).model_dump_json(),
                name="get_weather_forecast",
                description="Get weather forecast for a location. Returns daily forecasts.",
                args_schema=WeatherInput,
            ),
            StructuredTool.from_function(
                func=lambda location, categories=None: self._attractions_tool.search(
                    AttractionsInput(location=location, categories=categories or [])
                ).model_dump_json(),
                name="search_attractions",
                description="Search for attractions and activities in a location. Returns attractions with costs and durations.",
                args_schema=AttractionsInput,
            ),
        ]

    def _build_graph(self) -> StateGraph:
        set_verbose(True)
        set_debug(True)

        llm = self._model.get_chat_model()
        llm_with_tools = llm.bind_tools(self._tools)

        def call_model(state: TripPlannerState) -> dict:
            system_prompt = prompt_loader.load("trip_planner_system")
            messages = [SystemMessage(content=system_prompt)] + state.messages
            response = llm_with_tools.invoke(messages)

            # Log reasoning to scratchpad
            scratchpad_entry = (
                f"[Agent] {response.content[:200]}..." if len(response.content) > 200 else f"[Agent] {response.content}"
            )
            logger.info(scratchpad_entry)

            return {
                "messages": [response],
                "scratchpad": [scratchpad_entry] if response.content else [],
            }

        def should_continue(state: TripPlannerState) -> str:
            """Determine if we should continue with tools or end."""
            last_message = state.messages[-1]
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "tools"
            return "generate_itinerary"

        def generate_itinerary(state: TripPlannerState) -> dict:
            llm = self._model.get_chat_model().with_structured_output(TripItinerary)
            itinerary_prompt = prompt_loader.load("trip_planner_itinerary")
            messages = state.messages + [HumanMessage(content=itinerary_prompt)]

            try:
                itinerary = llm.invoke(messages)
                logger.info(f"[Agent] Generated itinerary: {itinerary.summary}")
                return {
                    "messages": messages,
                    "itinerary": itinerary,
                    "scratchpad": [f"[Agent] Generated final itinerary with total cost: ${itinerary.actual_cost}"],
                }
            except Exception as e:
                logger.error(f"Failed to generate itinerary: {e}")
                return {
                    "messages": messages,
                    "scratchpad": [f"[Agent] Error generating itinerary: {e}"],
                }

        # Build the graph
        graph = StateGraph(TripPlannerState)

        # Add nodes
        graph.add_node("agent", call_model)
        graph.add_node("tools", ToolNode(self._tools))
        graph.add_node("generate_itinerary", generate_itinerary)

        # Set entry point
        graph.set_entry_point("agent")

        # Add edges
        graph.add_conditional_edges(
            "agent",
            should_continue,
            {
                "tools": "tools",
                "generate_itinerary": "generate_itinerary",
            },
        )
        graph.add_edge("tools", "agent")
        graph.add_edge("generate_itinerary", END)

        return graph

    async def plan_trip(self, user_prompt: str) -> dict:
        """Plan a trip based on user prompt.

        Args:
            user_prompt: Natural language trip request.

        Returns:
            Dictionary with itinerary and scratchpad reasoning.
        """
        graph = self._build_graph().compile()

        initial_state: dict = {
            "messages": [HumanMessage(content=user_prompt)],
            "scratchpad": [f"[User] {user_prompt}"],
        }

        # Run the graph
        result = await graph.ainvoke(initial_state)  # type: ignore[arg-type]

        itinerary = result.get("itinerary")
        return {
            "itinerary": itinerary.model_dump() if itinerary else None,
            "scratchpad": result.get("scratchpad", []),
        }
