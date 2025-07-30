import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

import logging
from typing import Dict, Any, List, Optional
from enum import Enum
from pydantic import BaseModel, Field
from agents import Agent, Runner

logger = logging.getLogger(__name__)

class NarrativeArc(str, Enum):
    HERO_JOURNEY = "hero_journey"
    DAVID_VS_GOLIATH = "david_vs_goliath"
    REDEMPTION = "redemption"
    RISE_AND_FALL = "rise_and_fall"
    THRILLER = "thriller"
    TACTICAL_CHESS = "tactical_chess"
    EMOTIONAL_ROLLER_COASTER = "emotional_roller_coaster"

class StoryTension(int, Enum):
    LOW = 1
    MEDIUM = 5
    HIGH = 8
    EXTREME = 10

class CharacterRole(BaseModel):
    name: str
    role_type: str  # protagonist, antagonist, mentor, etc.
    motivation: str
    stakes: str
    arc_description: str
    key_moments: List[str]
    emotional_journey: str

class PlotPoint(BaseModel):
    sequence: int
    narrative_element: str
    time_in_match: str
    event_description: str
    emotional_impact: int
    tension_level: int
    characters_involved: List[str]
    significance: str
    transitions_to: str

class NarrativePlan:
    narrative_arc: NarrativeArc
    central_theme: str
    emotional_journey: str
    overall_tension: StoryTension

    main_characters: List[CharacterRole]
    supporting_characters: List[CharacterRole]

    plot_points: List[PlotPoint]
    story_beats: Dict[str, str]

    opening_hook: str
    narrative_flow: List[str]
    climactic_moment: str
    resolution_message: str

    tension_builders: List[str]
    emotional_peaks: List[str]
    surprise_elements: List[str]
    human_interest_angles: List[str]

    tone_direction: str
    pacing_notes: str
    statistical_integration: List[str]
    quote_placement: Dict[str, str]

    @classmethod
    def fallback(cls):
        return cls(
            narrative_arc=NarrativeArc.THRILLER,
            central_theme="Compelling match narrative",
            emotional_journey="Tension builds to climactic finish",
            overall_tension=StoryTension.MEDIUM,
            main_characters=[],
            supporting_characters=[],
            plot_points=[],
            story_beats={"setup": "Match begins", "climax": "Decisive moment", "resolution": "Final outcome"},
            opening_hook="Match begins with high stakes",
            narrative_flow=["setup", "rising_action", "climax", "resolution"],
            climactic_moment="Key match moment",
            resolution_message="Match concludes with significance",
            tension_builders=["Key match moments"],
            emotional_peaks=["Dramatic highlights"],
            surprise_elements=["Unexpected developments"],
            human_interest_angles=["Player stories"],
            tone_direction="Engaging and informative",
            pacing_notes="Build tension throughout",
            statistical_integration=["Use stats to support narrative"],
            quote_placement={"climax": "Player reactions"}
        )

class NarrativePlanner:
    """
    Plug-in narrative planner used between ResearchAgent and WriterAgent
    Generates NarrativePlan to enhance style, emotion, and storytelling.
    """
    def __init__(self, agent):
        self.agent = agent

    async def plan_from_research(self, research_data: Dict[str, Any]) -> NarrativePlan:
        """
        Create narrative plan based on structured research data
        Input: Output from ResearchAgent (game_analysis, player_performance, historical_context)
        Output: NarrativePlan (used by WriterAgent to stylize content)
        """
        logger.info("Planning narrative structure from research insights")

        prompt = f"""
        Given the following football match research data, create a comprehensive narrative plan for writing a vivid and emotionally engaging recap article.

        RESEARCH DATA:
        - Game Analysis: {research_data.get("game_analysis", [])}
        - Player Performance: {research_data.get("player_performance", [])}
        - Historical Context: {research_data.get("historical_context", [])}

        TASKS:
        1. Choose the best narrative arc (hero_journey, redemption, thriller, etc.)
        2. Identify key characters, their roles, and arcs
        3. Highlight major plot points (setup, climax, resolution)
        4. Suggest tone, pacing, and structure
        5. Output structured JSON using NarrativePlan schema

        RESPONSE FORMAT:
        (Follow the fields in the NarrativePlan definition)
        """

        try:
            result = await Runner.run(self.agent, prompt)
            return NarrativePlan.parse_raw(result.final_output)  # assumes pydantic model
        except Exception as e:
            logger.warning(f"Failed to create narrative plan, fallback used: {e}")
            return NarrativePlan.fallback()

    def inject_into_writer_prompt(self, plan: NarrativePlan) -> str:
        """
        Merge base writing prompt with narrative plan to create stylized article guidance
        """
        outline = "\n".join([
            f"{i+1}. {beat}" for i, beat in enumerate(plan.narrative_flow)
        ])

        return f"""

        ---
        ENHANCED STORY OUTLINE:
        Narrative Arc: {plan.narrative_arc.value}
        Central Theme: {plan.central_theme}
        Main Characters: {[c.name for c in plan.main_characters]}
        Tone: {plan.tone_direction}
        Structure:
        {outline}
        Climax: {plan.climactic_moment}
        Resolution: {plan.resolution_message}
        ---
        """

    def process_live_commentary(self, commentary: str):
        pass

    def process_history_data(self, history_data: str):
        pass

    def process_interview_quotes(self, interview_quotes: str):
        pass


        