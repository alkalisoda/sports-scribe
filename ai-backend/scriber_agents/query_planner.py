"""
Coarse-to-Fine Query Planner for Sports Intelligence Layer Integration.

This module implements a two-stage query planning system:
1. Coarse Stage: Generate broad analytical angles and exploratory queries
2. Fine Stage: Refine focus based on retrieval results and generate specific queries
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


class AnalysisAngle(Enum):
    """Analysis angles for coarse query generation"""
    PERFORMANCE_SPOTLIGHT = "performance_spotlight"
    TACTICAL_DYNAMICS = "tactical_dynamics"
    HISTORICAL_CONTEXT = "historical_context"
    NARRATIVE_DRAMA = "narrative_drama"
    STATISTICAL_SIGNIFICANCE = "statistical_significance"
    TEAM_FORM_ANALYSIS = "team_form_analysis"
    PLAYER_MILESTONES = "player_milestones"


@dataclass
class CoarseAngle:
    """Represents a coarse analysis angle"""
    angle: AnalysisAngle
    priority: float  # 0.0 - 1.0
    rationale: str
    broad_questions: List[str]


@dataclass
class CoarseRetrievalResult:
    """Results from coarse retrieval stage"""
    angle: AnalysisAngle
    questions: List[str]
    results: List[Dict[str, Any]]
    relevance_score: float
    data_richness: float


@dataclass
class FineAngle:
    """Refined analysis angle for fine queries"""
    original_angle: AnalysisAngle
    refined_focus: str
    specific_questions: List[str]
    expected_insights: List[str]


@dataclass
class QueryPlanningResult:
    """Complete query planning result"""
    coarse_angles: List[CoarseAngle]
    coarse_results: List[CoarseRetrievalResult]
    selected_fine_angles: List[FineAngle]
    fine_results: List[Dict[str, Any]]
    processing_metadata: Dict[str, Any]


class QueryPlanner:
    """
    Coarse-to-Fine Query Planner for intelligent sports data retrieval.

    Workflow:
    1. Analyze game data to generate coarse analysis angles
    2. Generate broad exploratory questions for each angle
    3. Execute coarse queries against Sports Intelligence Layer
    4. Analyze retrieval results to select promising angles
    5. Generate refined, specific questions for selected angles
    6. Execute fine queries for detailed insights
    """

    def __init__(self, sports_intel_client, config: Dict[str, Any] = None):
        """Initialize the Query Planner"""
        self.sports_intel = sports_intel_client
        self.config = config or {}

        # Initialize LLM for planning
        self.planner_llm = ChatOpenAI(
            model=self.config.get("planning_model", "gpt-4o"),
            temperature=self.config.get("planning_temperature", 0.8),
            max_tokens=self.config.get("planning_max_tokens", 1500),
        )

        # Configuration
        self.max_coarse_angles = self.config.get("max_coarse_angles", 5)
        self.max_fine_angles = self.config.get("max_fine_angles", 3)
        self.coarse_questions_per_angle = self.config.get("coarse_questions_per_angle", 3)
        self.fine_questions_per_angle = self.config.get("fine_questions_per_angle", 4)

        logger.info("Query Planner initialized with coarse-to-fine strategy")

    async def plan_and_execute_queries(self, game_data: Dict[str, Any]) -> QueryPlanningResult:
        """
        Execute complete coarse-to-fine query planning and retrieval.

        Args:
            game_data: Compact game data from pipeline

        Returns:
            QueryPlanningResult with both coarse and fine retrieval results
        """
        import time
        start_time = time.time()

        logger.info("Starting coarse-to-fine query planning")

        try:
            # Stage 1: Generate coarse analysis angles
            logger.info("Stage 1: Generating coarse analysis angles")
            coarse_angles = await self._generate_coarse_angles(game_data)

            # Stage 2: Execute coarse queries
            logger.info("Stage 2: Executing coarse queries")
            coarse_results = await self._execute_coarse_queries(coarse_angles, game_data)

            # Stage 3: Analyze results and select fine angles
            logger.info("Stage 3: Selecting fine angles based on coarse results")
            fine_angles = await self._select_fine_angles(coarse_results, game_data)

            # Stage 4: Execute fine queries
            logger.info("Stage 4: Executing fine queries")
            fine_results = await self._execute_fine_queries(fine_angles, game_data)

            # Create result with metadata
            processing_time = time.time() - start_time
            metadata = {
                "processing_time_seconds": processing_time,
                "coarse_angles_generated": len(coarse_angles),
                "coarse_queries_executed": sum(len(angle.broad_questions) for angle in coarse_angles),
                "fine_angles_selected": len(fine_angles),
                "fine_queries_executed": sum(len(angle.specific_questions) for angle in fine_angles),
                "total_results_retrieved": len(fine_results),
                "query_planning_strategy": "coarse_to_fine"
            }

            result = QueryPlanningResult(
                coarse_angles=coarse_angles,
                coarse_results=coarse_results,
                selected_fine_angles=fine_angles,
                fine_results=fine_results,
                processing_metadata=metadata
            )

            logger.info(f"Query planning completed in {processing_time:.3f}s")
            return result

        except Exception as e:
            logger.error(f"Error in query planning: {e}")
            raise

    async def _generate_coarse_angles(self, game_data: Dict[str, Any]) -> List[CoarseAngle]:
        """Generate coarse analysis angles based on game data"""

        # Extract key information for angle generation
        match_info = game_data.get("match_info", {})
        events = game_data.get("events", [])
        players = game_data.get("players", [])

        home_team = match_info.get("teams", {}).get("home", {}).get("name", "Home Team")
        away_team = match_info.get("teams", {}).get("away", {}).get("name", "Away Team")

        coarse_planning_prompt = f"""
        As a sports analysis strategist, analyze this game data and generate coarse analysis angles for in-depth research.

        GAME CONTEXT:
        - Match: {home_team} vs {away_team}
        - Events: {len(events)} key events
        - Key Players: {len(players)} players identified
        - League: {match_info.get("league", {}).get("name", "Unknown")}

        AVAILABLE ANALYSIS ANGLES:
        1. PERFORMANCE_SPOTLIGHT - Focus on standout individual performances
        2. TACTICAL_DYNAMICS - Analyze tactical setup and strategic decisions
        3. HISTORICAL_CONTEXT - Explore historical significance and patterns
        4. NARRATIVE_DRAMA - Identify dramatic moments and storylines
        5. STATISTICAL_SIGNIFICANCE - Focus on statistical achievements and records
        6. TEAM_FORM_ANALYSIS - Analyze team form and momentum
        7. PLAYER_MILESTONES - Track milestone achievements and career moments

        For each promising angle, generate:
        1. Priority score (0.0-1.0) based on data richness and story potential
        2. Rationale for why this angle is worth exploring
        3. 3 broad exploratory questions for coarse retrieval

        Return JSON format:
        {{
            "angles": [
                {{
                    "angle": "PERFORMANCE_SPOTLIGHT",
                    "priority": 0.85,
                    "rationale": "Strong individual performances evident in match data",
                    "broad_questions": [
                        "Which players had standout performances in this match?",
                        "What notable statistical achievements occurred?",
                        "How do these performances compare to season averages?"
                    ]
                }}
            ]
        }}

        Generate {self.max_coarse_angles} most promising angles.
        """

        result = await self.planner_llm.ainvoke([
            SystemMessage(content="You are a sports analysis strategist specializing in identifying promising research angles."),
            HumanMessage(content=coarse_planning_prompt)
        ])

        # Parse the result
        coarse_angles = self._parse_coarse_angles_response(result.content)

        logger.info(f"Generated {len(coarse_angles)} coarse analysis angles")
        return coarse_angles

    async def _execute_coarse_queries(self, coarse_angles: List[CoarseAngle],
                                    game_data: Dict[str, Any]) -> List[CoarseRetrievalResult]:
        """Execute broad queries for each coarse angle"""

        coarse_results = []

        for angle in coarse_angles:
            logger.info(f"Executing coarse queries for angle: {angle.angle.value}")

            # Execute all questions for this angle in parallel
            query_tasks = [
                self.sports_intel.ask(question, context=game_data)
                for question in angle.broad_questions
            ]

            try:
                query_results = await asyncio.gather(*query_tasks, return_exceptions=True)

                # Process results and calculate relevance scores
                valid_results = []
                for result in query_results:
                    if not isinstance(result, Exception) and result:
                        valid_results.append(result.supporting_context)

                # Calculate relevance and data richness scores
                relevance_score = self._calculate_relevance_score(valid_results, angle)
                data_richness = self._calculate_data_richness(valid_results)

                coarse_result = CoarseRetrievalResult(
                    angle=angle.angle,
                    questions=angle.broad_questions,
                    results=valid_results,
                    relevance_score=relevance_score,
                    data_richness=data_richness
                )

                coarse_results.append(coarse_result)

                logger.info(f"Coarse retrieval for {angle.angle.value}: "
                          f"{len(valid_results)} results, relevance: {relevance_score:.3f}")

            except Exception as e:
                logger.warning(f"Error executing coarse queries for {angle.angle.value}: {e}")
                # Add empty result to maintain structure
                coarse_results.append(CoarseRetrievalResult(
                    angle=angle.angle,
                    questions=angle.broad_questions,
                    results=[],
                    relevance_score=0.0,
                    data_richness=0.0
                ))

        return coarse_results

    async def _select_fine_angles(self, coarse_results: List[CoarseRetrievalResult],
                                game_data: Dict[str, Any]) -> List[FineAngle]:
        """Analyze coarse results and select angles for fine-grained exploration"""

        # Sort by combined score (relevance + data richness)
        scored_results = []
        for result in coarse_results:
            combined_score = (result.relevance_score * 0.6) + (result.data_richness * 0.4)
            scored_results.append((combined_score, result))

        scored_results.sort(key=lambda x: x[0], reverse=True)

        # Select top angles for fine exploration
        top_results = scored_results[:self.max_fine_angles]

        fine_angles = []
        for score, coarse_result in top_results:
            logger.info(f"Refining angle {coarse_result.angle.value} (score: {score:.3f})")

            # Generate refined focus and specific questions
            fine_angle = await self._refine_angle(coarse_result, game_data)
            fine_angles.append(fine_angle)

        return fine_angles

    async def _refine_angle(self, coarse_result: CoarseRetrievalResult,
                          game_data: Dict[str, Any]) -> FineAngle:
        """Refine a coarse angle into specific focused queries"""

        # Analyze coarse results to determine specific focus
        results_summary = self._summarize_coarse_results(coarse_result.results)

        refinement_prompt = f"""
        Based on the coarse retrieval results, refine the analysis angle for focused exploration.

        ORIGINAL ANGLE: {coarse_result.angle.value}

        COARSE QUERIES EXECUTED:
        {chr(10).join(f"- {q}" for q in coarse_result.questions)}

        RETRIEVAL RESULTS SUMMARY:
        {results_summary}

        DATA RICHNESS: {coarse_result.data_richness:.3f}
        RELEVANCE SCORE: {coarse_result.relevance_score:.3f}

        Based on these results, generate:
        1. A refined focus statement (specific aspect to explore)
        2. {self.fine_questions_per_angle} specific, targeted questions for detailed retrieval
        3. Expected insights from this refined exploration

        Return JSON format:
        {{
            "refined_focus": "Specific aspect to explore in detail",
            "specific_questions": [
                "Targeted question 1",
                "Targeted question 2",
                "Targeted question 3",
                "Targeted question 4"
            ],
            "expected_insights": [
                "Expected insight 1",
                "Expected insight 2"
            ]
        }}
        """

        result = await self.planner_llm.ainvoke([
            SystemMessage(content="You are a sports research specialist who refines broad analysis into focused investigations."),
            HumanMessage(content=refinement_prompt)
        ])

        # Parse the refinement result
        fine_angle_data = self._parse_fine_angle_response(result.content)

        fine_angle = FineAngle(
            original_angle=coarse_result.angle,
            refined_focus=fine_angle_data.get("refined_focus", "Detailed analysis"),
            specific_questions=fine_angle_data.get("specific_questions", []),
            expected_insights=fine_angle_data.get("expected_insights", [])
        )

        logger.info(f"Refined {coarse_result.angle.value} → {fine_angle.refined_focus}")
        return fine_angle

    async def _execute_fine_queries(self, fine_angles: List[FineAngle],
                                  game_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute specific fine-grained queries"""

        all_fine_results = []

        for fine_angle in fine_angles:
            logger.info(f"Executing fine queries for: {fine_angle.refined_focus}")

            # Execute specific questions for this refined angle
            query_tasks = [
                self.sports_intel.ask(question, context=game_data)
                for question in fine_angle.specific_questions
            ]

            try:
                query_results = await asyncio.gather(*query_tasks, return_exceptions=True)

                # Process and structure the results
                angle_results = []
                for i, result in enumerate(query_results):
                    if not isinstance(result, Exception) and result:
                        angle_results.append({
                            "question": fine_angle.specific_questions[i],
                            "answer": result.main_insight,
                            "confidence": result.confidence_score,
                            "supporting_data": result.supporting_context,
                            "refined_focus": fine_angle.refined_focus,
                            "original_angle": fine_angle.original_angle.value
                        })

                all_fine_results.extend(angle_results)

                logger.info(f"Fine retrieval for '{fine_angle.refined_focus}': "
                          f"{len(angle_results)} detailed results")

            except Exception as e:
                logger.warning(f"Error executing fine queries for '{fine_angle.refined_focus}': {e}")

        return all_fine_results

    def _parse_coarse_angles_response(self, response_text: str) -> List[CoarseAngle]:
        """Parse LLM response for coarse angles"""
        try:
            import json
            import re

            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON found in response")

            data = json.loads(json_match.group())
            angles_data = data.get("angles", [])

            coarse_angles = []
            for angle_data in angles_data:
                try:
                    angle_enum = AnalysisAngle(angle_data.get("angle", "").lower())
                    coarse_angle = CoarseAngle(
                        angle=angle_enum,
                        priority=float(angle_data.get("priority", 0.5)),
                        rationale=angle_data.get("rationale", ""),
                        broad_questions=angle_data.get("broad_questions", [])
                    )
                    coarse_angles.append(coarse_angle)
                except (ValueError, KeyError) as e:
                    logger.warning(f"Error parsing angle data: {e}")
                    continue

            return coarse_angles

        except Exception as e:
            logger.error(f"Error parsing coarse angles response: {e}")
            # Return fallback angles
            return self._get_fallback_coarse_angles()

    def _parse_fine_angle_response(self, response_text: str) -> Dict[str, Any]:
        """Parse LLM response for fine angle refinement"""
        try:
            import json
            import re

            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON found in response")

            return json.loads(json_match.group())

        except Exception as e:
            logger.error(f"Error parsing fine angle response: {e}")
            return {
                "refined_focus": "Detailed analysis",
                "specific_questions": ["What are the key insights from this angle?"],
                "expected_insights": ["Comprehensive analysis"]
            }

    def _calculate_relevance_score(self, results: List[Dict[str, Any]],
                                 angle: CoarseAngle) -> float:
        """Calculate relevance score based on result quality and angle alignment"""
        if not results:
            return 0.0

        # Simple heuristic based on result count and content
        base_score = min(len(results) / len(angle.broad_questions), 1.0)

        # Boost score based on result richness
        content_score = 0.0
        for result in results:
            if isinstance(result, dict) and result:
                content_score += 0.2

        return min(base_score + content_score, 1.0)

    def _calculate_data_richness(self, results: List[Dict[str, Any]]) -> float:
        """Calculate data richness score"""
        if not results:
            return 0.0

        richness_indicators = 0
        for result in results:
            if isinstance(result, dict):
                # Check for various data indicators
                if 'value' in result:
                    richness_indicators += 1
                if 'statistics' in result:
                    richness_indicators += 1
                if 'performance' in result:
                    richness_indicators += 1
                if len(str(result)) > 100:  # Non-empty content
                    richness_indicators += 1

        return min(richness_indicators / (len(results) * 2), 1.0)

    def _summarize_coarse_results(self, results: List[Dict[str, Any]]) -> str:
        """Create a summary of coarse retrieval results"""
        if not results:
            return "No results retrieved"

        summary_parts = []
        for i, result in enumerate(results, 1):
            if isinstance(result, dict):
                result_type = "data found" if result else "no data"
                summary_parts.append(f"Query {i}: {result_type}")
            else:
                summary_parts.append(f"Query {i}: {str(result)[:100]}...")

        return "; ".join(summary_parts)

    def _get_fallback_coarse_angles(self) -> List[CoarseAngle]:
        """Return fallback coarse angles if parsing fails"""
        return [
            CoarseAngle(
                angle=AnalysisAngle.PERFORMANCE_SPOTLIGHT,
                priority=0.8,
                rationale="Fallback performance analysis",
                broad_questions=[
                    "Which players had notable performances?",
                    "What key statistics stand out?",
                    "How do performances compare to averages?"
                ]
            ),
            CoarseAngle(
                angle=AnalysisAngle.HISTORICAL_CONTEXT,
                priority=0.7,
                rationale="Fallback historical context",
                broad_questions=[
                    "What is the historical significance?",
                    "How do teams historically perform?",
                    "What patterns are relevant?"
                ]
            )
        ]