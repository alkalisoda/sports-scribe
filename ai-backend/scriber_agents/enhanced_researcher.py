"""
Enhanced Research Agent with Coarse-to-Fine Query Planning.

This agent integrates the existing ResearchAgent with the new QueryPlanner
to implement intelligent, two-stage data retrieval from the Sports Intelligence Layer.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import time

from .researcher import ResearchAgent, EnhancedResearchResult, AnalysisResult, NarrativePlan
from .query_planner import QueryPlanner, QueryPlanningResult

logger = logging.getLogger(__name__)


@dataclass
class IntelligentResearchResult:
    """Enhanced research result with intelligent query planning metadata"""
    traditional_analysis: AnalysisResult
    narrative_plan: NarrativePlan
    intelligent_insights: List[Dict[str, Any]]
    query_planning_metadata: Dict[str, Any]
    processing_metadata: Dict[str, Any]


class EnhancedResearchAgent(ResearchAgent):
    """
    Enhanced Research Agent that combines traditional storyline analysis
    with intelligent, coarse-to-fine query planning against the Sports Intelligence Layer.

    Workflow:
    1. Execute traditional storyline analysis (existing functionality)
    2. Generate coarse analysis angles based on game data
    3. Execute broad queries for initial data exploration
    4. Refine angles based on retrieval results
    5. Execute fine-grained queries for detailed insights
    6. Synthesize traditional analysis with intelligent insights
    """

    def __init__(self, config: Dict[str, Any], sports_intel_client):
        """Initialize Enhanced Research Agent"""
        super().__init__(config)

        # Initialize Query Planner with sports intelligence client
        self.query_planner = QueryPlanner(
            sports_intel_client,
            config.get('query_planning', {})
        )

        # Enhanced configuration
        self.enable_traditional_analysis = config.get('enable_traditional_analysis', True)
        self.enable_intelligent_planning = config.get('enable_intelligent_planning', True)
        self.synthesis_approach = config.get('synthesis_approach', 'hybrid')  # 'hybrid', 'intelligence_first', 'traditional_first'

        logger.info("Enhanced Research Agent initialized with coarse-to-fine query planning")

    async def get_intelligent_research(self, game_data: Dict[str, Any]) -> IntelligentResearchResult:
        """
        Get comprehensive research using both traditional analysis and intelligent query planning.

        Args:
            game_data: Compact game data from pipeline

        Returns:
            IntelligentResearchResult: Combined traditional and intelligent analysis
        """
        start_time = time.time()
        logger.info("Starting intelligent research with coarse-to-fine planning")

        try:
            # Execute both approaches in parallel if enabled
            tasks = []

            # Traditional analysis task
            if self.enable_traditional_analysis:
                traditional_task = self.get_enhanced_research_with_narrative(game_data)
                tasks.append(("traditional", traditional_task))

            # Intelligent query planning task
            if self.enable_intelligent_planning:
                intelligent_task = self.query_planner.plan_and_execute_queries(game_data)
                tasks.append(("intelligent", intelligent_task))

            # Execute tasks
            if len(tasks) == 2:
                # Parallel execution
                logger.info("Executing traditional analysis and intelligent planning in parallel")
                traditional_result, intelligent_result = await asyncio.gather(
                    tasks[0][1], tasks[1][1]
                )
            elif len(tasks) == 1:
                # Single execution
                if tasks[0][0] == "traditional":
                    logger.info("Executing traditional analysis only")
                    traditional_result = await tasks[0][1]
                    intelligent_result = None
                else:
                    logger.info("Executing intelligent planning only")
                    traditional_result = None
                    intelligent_result = await tasks[0][1]
            else:
                raise ValueError("No analysis method enabled")

            # Synthesize results
            synthesis_result = await self._synthesize_research_results(
                traditional_result, intelligent_result, game_data
            )

            processing_time = time.time() - start_time
            logger.info(f"Intelligent research completed in {processing_time:.3f}s")

            return synthesis_result

        except Exception as e:
            logger.error(f"Error in intelligent research: {e}")
            # Return fallback result
            return await self._create_fallback_result(game_data, str(e))

    async def _synthesize_research_results(self,
                                         traditional_result: Optional[EnhancedResearchResult],
                                         intelligent_result: Optional[QueryPlanningResult],
                                         game_data: Dict[str, Any]) -> IntelligentResearchResult:
        """Synthesize traditional and intelligent research results"""

        logger.info("Synthesizing traditional analysis with intelligent insights")

        # Extract components
        if traditional_result:
            traditional_analysis = traditional_result.analysis
            narrative_plan = traditional_result.narrative_plan
        else:
            # Create minimal traditional components
            traditional_analysis = AnalysisResult(
                storylines=["Game analysis based on available data"],
                confidence=0.7,
                analysis_type="minimal_traditional"
            )
            narrative_plan = self._create_fallback_narrative_plan(traditional_analysis.storylines)

        # Extract intelligent insights
        intelligent_insights = []
        query_planning_metadata = {}

        if intelligent_result:
            # Process fine query results into insights
            for fine_result in intelligent_result.fine_results:
                insight = {
                    "type": "intelligent_insight",
                    "original_angle": fine_result.get("original_angle"),
                    "refined_focus": fine_result.get("refined_focus"),
                    "question": fine_result.get("question"),
                    "answer": fine_result.get("answer"),
                    "confidence": fine_result.get("confidence", 0.0),
                    "supporting_data": fine_result.get("supporting_data", {}),
                    "source": "sports_intelligence_layer"
                }
                intelligent_insights.append(insight)

            query_planning_metadata = intelligent_result.processing_metadata
        else:
            query_planning_metadata = {
                "intelligent_planning_enabled": False,
                "reason": "Intelligent planning disabled or failed"
            }

        # Apply synthesis approach
        if self.synthesis_approach == "hybrid":
            # Merge traditional storylines with intelligent insights
            enhanced_storylines = await self._merge_storylines_with_insights(
                traditional_analysis.storylines, intelligent_insights
            )
            traditional_analysis.storylines = enhanced_storylines
        elif self.synthesis_approach == "intelligence_first":
            # Prioritize intelligent insights, supplement with traditional
            if intelligent_insights:
                insight_storylines = [
                    f"{insight['refined_focus']}: {insight['answer']}"
                    for insight in intelligent_insights[:5]
                ]
                traditional_analysis.storylines = insight_storylines + traditional_analysis.storylines[:3]
        # For 'traditional_first', keep original storylines as primary

        # Create processing metadata
        processing_metadata = {
            "synthesis_approach": self.synthesis_approach,
            "traditional_enabled": self.enable_traditional_analysis,
            "intelligent_enabled": self.enable_intelligent_planning,
            "traditional_storylines": len(traditional_analysis.storylines) if traditional_result else 0,
            "intelligent_insights": len(intelligent_insights),
            "synthesis_method": "parallel" if traditional_result and intelligent_result else "single",
            "processing_timestamp": time.time()
        }

        # Combine query planning metadata
        if traditional_result:
            processing_metadata.update({
                "traditional_processing_time": traditional_result.processing_metadata.get("processing_time_seconds", 0),
                "traditional_confidence": traditional_result.analysis.confidence
            })

        return IntelligentResearchResult(
            traditional_analysis=traditional_analysis,
            narrative_plan=narrative_plan,
            intelligent_insights=intelligent_insights,
            query_planning_metadata=query_planning_metadata,
            processing_metadata=processing_metadata
        )

    async def _merge_storylines_with_insights(self,
                                            traditional_storylines: List[str],
                                            intelligent_insights: List[Dict[str, Any]]) -> List[str]:
        """Merge traditional storylines with intelligent insights"""

        if not intelligent_insights:
            return traditional_storylines

        logger.info(f"Merging {len(traditional_storylines)} traditional storylines with {len(intelligent_insights)} intelligent insights")

        # Convert insights to storylines
        insight_storylines = []
        for insight in intelligent_insights:
            if insight.get("confidence", 0) > 0.7:  # High confidence insights
                storyline = f"{insight.get('refined_focus', 'Analysis')}: {insight.get('answer', '')}"
                insight_storylines.append(storyline)

        # Interleave traditional and intelligent storylines
        merged_storylines = []
        max_len = max(len(traditional_storylines), len(insight_storylines))

        for i in range(max_len):
            # Add intelligent insight first (higher priority)
            if i < len(insight_storylines):
                merged_storylines.append(insight_storylines[i])

            # Add traditional storyline
            if i < len(traditional_storylines):
                merged_storylines.append(traditional_storylines[i])

        # Limit to reasonable number
        return merged_storylines[:10]

    async def _create_fallback_result(self, game_data: Dict[str, Any], error_msg: str) -> IntelligentResearchResult:
        """Create fallback result when intelligent research fails"""

        logger.warning(f"Creating fallback research result due to error: {error_msg}")

        # Create basic traditional analysis
        fallback_storylines = [
            "Game analysis based on available match data",
            "Key events and player performances from the match",
            "Statistical highlights and notable moments"
        ]

        traditional_analysis = AnalysisResult(
            storylines=fallback_storylines,
            confidence=0.6,
            analysis_type="fallback_analysis"
        )

        narrative_plan = self._create_fallback_narrative_plan(fallback_storylines)

        processing_metadata = {
            "fallback_used": True,
            "error_message": error_msg,
            "synthesis_approach": "fallback",
            "traditional_enabled": self.enable_traditional_analysis,
            "intelligent_enabled": self.enable_intelligent_planning,
            "processing_timestamp": time.time()
        }

        return IntelligentResearchResult(
            traditional_analysis=traditional_analysis,
            narrative_plan=narrative_plan,
            intelligent_insights=[],
            query_planning_metadata={"fallback": True, "error": error_msg},
            processing_metadata=processing_metadata
        )

    # Legacy compatibility methods

    async def get_enhanced_research_with_narrative(self, game_data: Dict[str, Any]) -> EnhancedResearchResult:
        """Backward compatibility wrapper for enhanced research"""
        logger.info("Executing enhanced research (legacy compatibility)")
        return await super().get_enhanced_research_with_narrative(game_data)

    async def get_storyline_from_game_data(self, game_data: dict) -> list[str]:
        """Backward compatibility wrapper for storyline generation"""
        logger.info("Executing storyline generation (legacy compatibility)")
        return await super().get_storyline_from_game_data(game_data)

    async def get_history_from_team_data(self, team_data: dict) -> list[str]:
        """Backward compatibility wrapper for historical context"""
        logger.info("Executing historical context analysis (legacy compatibility)")
        return await super().get_history_from_team_data(team_data)

    async def get_performance_from_player_game_data(self, player_data: dict, game_data: dict) -> list[str]:
        """Backward compatibility wrapper for player performance analysis"""
        logger.info("Executing player performance analysis (legacy compatibility)")
        return await super().get_performance_from_player_game_data(player_data, game_data)


class IntelligentResearchOrchestrator:
    """
    Orchestrator for different research strategies based on configuration and requirements.

    This class helps manage the transition from traditional to intelligent research
    and provides a unified interface for the pipeline.
    """

    def __init__(self, config: Dict[str, Any], sports_intel_client):
        """Initialize the research orchestrator"""
        self.config = config
        self.research_strategy = config.get('research_strategy', 'intelligent')  # 'traditional', 'intelligent', 'adaptive'

        # Initialize appropriate research agent
        if self.research_strategy in ['intelligent', 'adaptive']:
            self.research_agent = EnhancedResearchAgent(config, sports_intel_client)
        else:
            # Traditional research agent
            from .researcher import ResearchAgent
            self.research_agent = ResearchAgent(config)

        logger.info(f"Research orchestrator initialized with strategy: {self.research_strategy}")

    async def conduct_research(self, game_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Conduct research using the configured strategy.

        Returns standardized research result format regardless of strategy.
        """

        if self.research_strategy == 'intelligent':
            # Use intelligent research
            result = await self.research_agent.get_intelligent_research(game_data)
            return self._format_intelligent_result(result)

        elif self.research_strategy == 'adaptive':
            # Decide strategy based on data characteristics
            if self._should_use_intelligent_research(game_data):
                result = await self.research_agent.get_intelligent_research(game_data)
                return self._format_intelligent_result(result)
            else:
                # Fall back to traditional
                result = await self.research_agent.get_enhanced_research_with_narrative(game_data)
                return self._format_traditional_result(result)

        else:  # traditional
            # Use traditional research
            result = await self.research_agent.get_enhanced_research_with_narrative(game_data)
            return self._format_traditional_result(result)

    def _should_use_intelligent_research(self, game_data: Dict[str, Any]) -> bool:
        """Determine if intelligent research should be used based on data characteristics"""

        # Check data richness
        events_count = len(game_data.get("events", []))
        players_count = len(game_data.get("players", []))

        # Use intelligent research for richer datasets
        if events_count >= 5 and players_count >= 3:
            return True

        # Check for complex scenarios
        match_info = game_data.get("match_info", {})
        is_important_match = match_info.get("league", {}).get("name", "").lower() in ["premier league", "champions league"]

        return is_important_match

    def _format_intelligent_result(self, result: IntelligentResearchResult) -> Dict[str, Any]:
        """Format intelligent research result for pipeline consumption"""
        return {
            "research_type": "intelligent",
            "storylines": result.traditional_analysis.storylines,
            "narrative_plan": result.narrative_plan,
            "intelligent_insights": result.intelligent_insights,
            "confidence": result.traditional_analysis.confidence,
            "processing_metadata": result.processing_metadata,
            "query_planning_metadata": result.query_planning_metadata
        }

    def _format_traditional_result(self, result: EnhancedResearchResult) -> Dict[str, Any]:
        """Format traditional research result for pipeline consumption"""
        return {
            "research_type": "traditional",
            "storylines": result.analysis.storylines,
            "narrative_plan": result.narrative_plan,
            "intelligent_insights": [],
            "confidence": result.analysis.confidence,
            "processing_metadata": result.processing_metadata,
            "query_planning_metadata": {}
        }