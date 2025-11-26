import logging
from typing import Annotated, Any

from fastapi import Depends
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import StructuredTool
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from ai_unifier_assesment.agent.state import TripItinerary, TripPlannerState
from ai_unifier_assesment.agent.tools.attractions_tool import AttractionsInput, AttractionsTool
from ai_unifier_assesment.agent.tools.flight_tool import FlightSearchInput, FlightTool
from ai_unifier_assesment.agent.tools.weather_tool import WeatherInput, WeatherTool
from ai_unifier_assesment.large_language_model.model import Model
from ai_unifier_assesment.resources.prompts.prompt_loader import prompt_loader

logger = logging.getLogger(__name__)


class TripPlannerAgent:
    def __init__(
        self,
        model: Annotated[Model, Depends(Model)],
        flight_tool: Annotated[FlightTool, Depends(FlightTool)],
        weather_tool: Annotated[WeatherTool, Depends(WeatherTool)],
        attractions_tool: Annotated[AttractionsTool, Depends(AttractionsTool)],
    ):
        self._model = model
        self._flight_tool = flight_tool
        self._weather_tool = weather_tool
        self._attractions_tool = attractions_tool
        self._tools = self._create_tools()
        self._simple_llm = self._model.simple_model()
        self._llm_with_tools = self._simple_llm.bind_tools(self._tools)
        self._llm_structured_ouput: Runnable[Any, Any] = (
            self._model.get_chat_model_for_evaluation().with_structured_output(TripItinerary)
        )
        self._itinerary_prompt = prompt_loader.load("trip_planner_itinerary")

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

    def _log_pre_tool_call_scratchpad(self, state: TripPlannerState):
        logger.info("--- SCRATCH-PAD: LLM Reasoning STARTED ---")
        logger.info(f"Incoming Messages: {[m.type for m in state.messages]}")

    def _log_post_tool_call_scratchpad(self, state: TripPlannerState, response):
        if response.tool_calls:
            logger.info(f"LLM Decision: Requesting Tool Calls (Count: {len(response.tool_calls)})")
            for tc in response.tool_calls:
                logger.info(f"  -> Tool Call: {tc.name} with Args: {tc.args}")
        else:
            # If the LLM returns a text response, it's often a thinking step before final response
            logger.info(f"LLM Decision: Intermediate Text Response/Reasoning. Content: {response.content[:100]}...")

        logger.info("--- SCRATCH-PAD: LLM Reasoning ENDED ---")

    def _call_model(self, state: TripPlannerState) -> dict:
        self._log_pre_tool_call_scratchpad(state)
        response = self._llm_with_tools.invoke(state.messages)
        self._log_post_tool_call_scratchpad(state, response)

        return {"messages": [response]}

    def _should_continue(self, state: TripPlannerState) -> str:
        last_message = state.messages[-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            logger.info("--- SCRATCH-PAD: Decision making: Making Tool call ---")
            return "tools"
        logger.info("--- SCRATCH-PAD: Decision making: Generate intinerary ---")
        return "generate_itinerary"

    def _generate_itinerary(self, state: TripPlannerState) -> dict:
        logger.info("--- SCRATCH-PAD: Final Itinerary Synthesis ---")
        logger.info("Decision: All information gathered. Generating final structured itinerary (JSON).")
        messages = state.messages + [HumanMessage(content=self._itinerary_prompt)]
        response = self._llm_structured_ouput.invoke(messages)
        logger.info(f"Generated final itinerary with total cost: ${response.actual_cost}")

        return {
            "messages": [AIMessage(content=response.model_dump_json())],
            "itinerary": response,
        }

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(TripPlannerState)

        graph.add_node("agent", self._call_model)
        graph.add_node("tools", ToolNode(self._tools))
        graph.add_node("generate_itinerary", self._generate_itinerary)

        graph.set_entry_point("agent")
        graph.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "tools": "tools",
                "generate_itinerary": "generate_itinerary",
            },
        )
        graph.add_edge("tools", "agent")
        graph.add_edge("generate_itinerary", END)

        return graph

    async def plan_trip(self, user_prompt: str) -> dict:
        graph = self._build_graph().compile()
        system_prompt = prompt_loader.load("trip_planner_system")

        initial_state: dict = {
            "messages": [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ],
        }

        result = await graph.ainvoke(initial_state)  # type: ignore[arg-type]

        itinerary = result.get("itinerary")
        return {
            "itinerary": itinerary.model_dump() if itinerary else None,
        }
