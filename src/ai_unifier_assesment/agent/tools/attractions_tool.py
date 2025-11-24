from pydantic import BaseModel


class AttractionsInput(BaseModel):
    location: str
    categories: list[str] = []


class AttractionResult(BaseModel):
    name: str
    description: str
    category: str
    estimated_cost: float
    duration_hours: float
    location: str


class AttractionsOutput(BaseModel):
    location: str
    attractions: list[AttractionResult]


class AttractionsTool:
    """Tool for searching attractions and activities."""

    def search(self, input: AttractionsInput) -> AttractionsOutput:
        """Search for attractions in a location.

        Args:
            input: Location and optional category filters.

        Returns:
            AttractionsOutput with available attractions.
        """
        # Mock Auckland attractions
        all_attractions = [
            AttractionResult(
                name="Sky Tower",
                description="Iconic 328m tower with observation deck and restaurants",
                category="landmark",
                estimated_cost=32.0,
                duration_hours=2.0,
                location="Auckland CBD",
            ),
            AttractionResult(
                name="Auckland Museum",
                description="War memorial museum with Maori artifacts and natural history",
                category="museum",
                estimated_cost=25.0,
                duration_hours=3.0,
                location="Auckland Domain",
            ),
            AttractionResult(
                name="Waiheke Island Ferry & Wine Tour",
                description="Scenic ferry ride and vineyard visits",
                category="tour",
                estimated_cost=85.0,
                duration_hours=6.0,
                location="Waiheke Island",
            ),
            AttractionResult(
                name="Rangitoto Island Hike",
                description="Volcanic island hike with panoramic views",
                category="outdoor",
                estimated_cost=40.0,
                duration_hours=5.0,
                location="Rangitoto Island",
            ),
            AttractionResult(
                name="Viaduct Harbour",
                description="Waterfront dining and entertainment precinct",
                category="dining",
                estimated_cost=50.0,
                duration_hours=2.0,
                location="Viaduct Harbour",
            ),
            AttractionResult(
                name="Auckland Art Gallery",
                description="New Zealand's largest art gallery",
                category="museum",
                estimated_cost=0.0,
                duration_hours=2.0,
                location="Auckland CBD",
            ),
            AttractionResult(
                name="Kelly Tarlton's Sea Life Aquarium",
                description="Underground aquarium with sharks and penguins",
                category="attraction",
                estimated_cost=44.0,
                duration_hours=2.5,
                location="Orakei",
            ),
            AttractionResult(
                name="Mount Eden",
                description="Volcanic cone with city views - free to visit",
                category="outdoor",
                estimated_cost=0.0,
                duration_hours=1.5,
                location="Mount Eden",
            ),
        ]

        # Filter by categories if specified
        if input.categories:
            filtered = [a for a in all_attractions if a.category in input.categories]
        else:
            filtered = all_attractions

        return AttractionsOutput(
            location=input.location,
            attractions=filtered,
        )
