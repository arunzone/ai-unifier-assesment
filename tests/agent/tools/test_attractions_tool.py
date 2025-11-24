from assertpy import assert_that

from ai_unifier_assesment.agent.tools.attractions_tool import (
    AttractionsTool,
    AttractionsInput,
    AttractionsOutput,
    AttractionResult,
)


def test_search_returns_all_attractions_when_no_category_filter():
    tool = AttractionsTool()
    input = AttractionsInput(location="Auckland")

    result = tool.search(input)

    assert_that(result.location).is_equal_to("Auckland")
    assert_that(result.attractions).is_length(8)


def test_search_filters_by_category():
    tool = AttractionsTool()
    input = AttractionsInput(location="Auckland", categories=["museum"])

    result = tool.search(input)

    assert_that(result).is_equal_to(
        AttractionsOutput(
            location="Auckland",
            attractions=[
                AttractionResult(
                    name="Auckland Museum",
                    description="War memorial museum with Maori artifacts and natural history",
                    category="museum",
                    estimated_cost=25.0,
                    duration_hours=3.0,
                    location="Auckland Domain",
                ),
                AttractionResult(
                    name="Auckland Art Gallery",
                    description="New Zealand's largest art gallery",
                    category="museum",
                    estimated_cost=0.0,
                    duration_hours=2.0,
                    location="Auckland CBD",
                ),
            ],
        )
    )


def test_search_returns_empty_for_nonexistent_category():
    tool = AttractionsTool()
    input = AttractionsInput(location="Auckland", categories=["nightlife"])

    result = tool.search(input)

    assert_that(result).is_equal_to(
        AttractionsOutput(
            location="Auckland",
            attractions=[],
        )
    )
