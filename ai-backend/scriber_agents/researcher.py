"""Research Agent.

This agent provides contextual background and analysis for sports articles using
LangChain framework with Chain of Thought reasoning and Agent + Tools architecture.
It researches historical data, team/player statistics, and relevant context
to enrich the content generation process.
"""

import logging
from typing import Any, List, Dict, Optional
from dotenv import load_dotenv
import json

# LangChain imports
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

load_dotenv()
logger = logging.getLogger(__name__)


class AnalysisResult(BaseModel):
    """Schema for analysis results."""
    storylines: List[str] = Field(description="List of storylines generated from analysis")
    confidence: float = Field(description="Confidence score of the analysis", ge=0.0, le=1.0)
    analysis_type: str = Field(description="Type of analysis performed")


class StorylinePriority(BaseModel):
    """Schema for storyline with priority and context."""
    content: str = Field(description="The storyline content")
    priority: int = Field(description="Priority ranking (1=highest)", ge=1)
    narrative_angle: str = Field(description="Narrative angle (drama, analysis, performance, tactical)")
    audience_appeal: float = Field(description="Estimated audience appeal score", ge=0.0, le=1.0)
    story_type: str = Field(description="Type of story (match_decisive, player_spotlight, tactical_insight, historical_context)")


class NarrativePlan(BaseModel):
    """Schema for narrative planning results."""
    primary_narrative: str = Field(description="Main narrative focus of the article")
    storytelling_focus: str = Field(description="Primary storytelling approach")
    prioritized_storylines: List[StorylinePriority] = Field(description="Storylines ranked by importance and appeal")
    narrative_style: str = Field(description="Recommended narrative style (dramatic, analytical, balanced)")
    target_audience: str = Field(description="Primary target audience (general_fans, tactical_enthusiasts, club_supporters)")
    confidence: float = Field(description="Confidence in narrative selection", ge=0.0, le=1.0)


class EnhancedResearchResult(BaseModel):
    """Enhanced schema combining analysis and narrative planning."""
    analysis: AnalysisResult = Field(description="Raw analysis results")
    narrative_plan: NarrativePlan = Field(description="Narrative planning results")
    processing_metadata: Dict[str, Any] = Field(description="Processing metadata and timing info")


class MatchInfoAnalysisTool(BaseTool):
    """Tool for analyzing match information."""
    
    name: str = "match_info_analyzer"
    description: str = "Analyze basic match information for storylines including match context, teams, venue, league, and final score"
    
    def _run(self, match_info: str) -> str:
        """Run the match info analysis."""
        return f"Analyzing match information: {match_info}"
    
    async def _arun(self, match_info: str) -> str:
        """Async version of the run method."""
        return self._run(match_info)


class EventsAnalysisTool(BaseTool):
    """Tool for analyzing key match events."""
    
    name: str = "events_analyzer"
    description: str = "Analyze key match events (goals, cards, substitutions) for storylines"
    
    def _run(self, events: str) -> str:
        """Run the events analysis."""
        return f"Analyzing match events: {events}"
    
    async def _arun(self, events: str) -> str:
        """Async version of the run method."""
        return self._run(events)


class PlayerPerformanceAnalysisTool(BaseTool):
    """Tool for analyzing player performances."""
    
    name: str = "player_performance_analyzer"
    description: str = "Analyze individual player performances focusing on high-rated players and meaningful contributions"
    
    def _run(self, players: str) -> str:
        """Run the player performance analysis."""
        return f"Analyzing player performances: {players}"
    
    async def _arun(self, players: str) -> str:
        """Async version of the run method."""
        return self._run(players)


class TeamStatisticsAnalysisTool(BaseTool):
    """Tool for analyzing team statistics."""
    
    name: str = "team_statistics_analyzer"
    description: str = "Analyze team-wide statistics including possession, shots, corners, fouls"
    
    def _run(self, statistics: str) -> str:
        """Run the team statistics analysis."""
        return f"Analyzing team statistics: {statistics}"
    
    async def _arun(self, statistics: str) -> str:
        """Async version of the run method."""
        return self._run(statistics)


class LineupAnalysisTool(BaseTool):
    """Tool for analyzing lineups and formations."""
    
    name: str = "lineup_analyzer"
    description: str = "Analyze lineups, formations, and tactical setup"
    
    def _run(self, lineups: str) -> str:
        """Run the lineup analysis."""
        return f"Analyzing lineups and formations: {lineups}"
    
    async def _arun(self, lineups: str) -> str:
        """Async version of the run method."""
        return self._run(lineups)


class ResearchAgent:
    """LangChain-based Research Agent with Chain of Thought reasoning."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the LangChain Research Agent with configuration."""
        self.config = config or {}
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=self.config.get("model", "gpt-4-1106-preview"),
            temperature=self.config.get("temperature", 0.7),
            max_tokens=self.config.get("max_tokens", 2000),
        )
        
        # Initialize tools (currently placeholder tools that don't call external APIs)
        self.tools = [
            MatchInfoAnalysisTool(),
            EventsAnalysisTool(),
            PlayerPerformanceAnalysisTool(),
            TeamStatisticsAnalysisTool(),
            LineupAnalysisTool(),
        ]
        
        # Create the main system prompt with Chain of Thought reasoning
        self.system_prompt = """You are a sports research agent with Chain of Thought reasoning capabilities. 
        Provide clear, factual analysis based ONLY on provided data.

            CORE PRINCIPLES:
            - ONLY use information explicitly provided in the data
            - When in doubt, exclude rather than include
            - Clearly distinguish between THIS MATCH events and background information
        - Use Chain of Thought reasoning to break down complex analysis step by step

        CHAIN OF THOUGHT PROCESS:
        1. First, identify what data is available
        2. Then, determine what analysis can be performed
        3. Next, apply relevant validation rules
        4. Finally, generate structured storylines

            DATA VERIFICATION RULES:
            - Use EXACT names, numbers, and times from the data
            - Use "elapsed" + "extra" format for times (e.g., 90+1 for elapsed:90, extra:1)
            - Verify every detail against the original data
            - If goalkeeper data is not explicitly provided, DO NOT mention saves

            EVENT TYPE ISOLATION RULES:
            - Each event type has its own specific data - DO NOT mix them
            - Goal time cannot be used as substitution time
            - Substitution time cannot be used as card time
            - Card time cannot be used as goal time
            - Both players in substitution must appear in SAME substitution event

            GENERAL EXCLUSION PRINCIPLE:
            - Only describe events that explicitly appear in the data
            - Exclude anything uncertain, unverified, or not clearly listed
            - Do not fabricate, assume, or infer events not present

        Always return clear, structured analysis based solely on the provided data.
        Use the available tools to help with specific analysis tasks, but remember the tools are for organization - the actual analysis logic remains with you.
        """
        
        # Create the prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Create the agent
        self.agent = create_openai_tools_agent(self.llm, self.tools, self.prompt)
        
        # Create the agent executor
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            max_iterations=3,
            early_stopping_method="generate"
        )
        
        # Initialize JSON output parser
        self.json_parser = JsonOutputParser(pydantic_object=AnalysisResult)
        
        # Initialize narrative planner
        self.narrative_llm = ChatOpenAI(
            model=self.config.get("narrative_model", "gpt-4o"),
            temperature=self.config.get("narrative_temperature", 0.6),
            max_tokens=self.config.get("narrative_max_tokens", 1500),
        )
        
        logger.info("LangChain Research Agent with Narrative Planner initialized successfully")


    async def get_storyline_from_game_data(self, game_data: dict) -> list[str]:
        """Get comprehensive storylines from game data using Chain of Thought reasoning.
        
        Args:
            game_data: Compact game data from pipeline (contains match_info, events, players, statistics, lineups)
            
        Returns:
            list[str]: Comprehensive list of storylines including analysis
        """
        logger.info("Generating comprehensive storylines from compact game data using Chain of Thought reasoning")
        
        try:
            # Extract different components from compact data
            match_info = game_data.get("match_info", {})
            events = game_data.get("events", [])
            players = game_data.get("players", [])
            statistics = game_data.get("statistics", [])
            lineups = game_data.get("lineups", [])
            
            # Use Chain of Thought reasoning for comprehensive analysis
            cot_prompt = f"""
            Using Chain of Thought reasoning, analyze the following game data comprehensively:

            STEP 1 - DATA INVENTORY:
            Let me first identify what data is available:
            - Match Info: {bool(match_info)}
            - Events: {len(events)} events available
            - Players: {len(players)} players available  
            - Statistics: {len(statistics)} team stats available
            - Lineups: {len(lineups)} lineup records available

            STEP 2 - ANALYSIS PLANNING:
            Based on available data, I will analyze each component separately to ensure accuracy:

            GAME DATA TO ANALYZE:
            Match Info: {match_info}
            Events: {events}
            Players: {players}
            Statistics: {statistics}
            Lineups: {lineups}

            STEP 3 - COMPONENT ANALYSIS:
            Now I will analyze each component following the strict validation rules:

            STEP 4 - STORYLINE GENERATION:
            Generate storylines in JSON format as a list of strings. Each storyline should be factual and based only on the provided data.

            Return the result as a JSON object with this structure:
            {{
                "storylines": ["storyline1", "storyline2", ...],
                "confidence": 0.9,
                "analysis_type": "comprehensive_game_analysis"
            }}
            """
            
            # Execute the analysis using the agent
            result = await self.agent_executor.ainvoke({
                "input": cot_prompt
            })
            
            # Parse the output
            output_text = result.get("output", "")
            storylines = self._parse_storylines_from_output(output_text)
            
            if not storylines:
                # Fallback to component-by-component analysis
                storylines = await self._analyze_components_separately(
                    match_info, events, players, statistics, lineups
                )
            
            logger.info(f"Generated {len(storylines)} storylines using Chain of Thought reasoning")
            return storylines
            
        except Exception as e:
            logger.error(f"Error generating comprehensive storylines: {e}")
            return ["Comprehensive match analysis based on available game data", "Key moments and turning points from the match"]

    async def get_enhanced_research_with_narrative(self, game_data: dict) -> EnhancedResearchResult:
        """Get comprehensive research analysis with narrative planning.
        
        This method combines traditional storyline analysis with intelligent narrative planning
        to provide structured guidance for article writing.
        
        Args:
            game_data: Compact game data from pipeline
            
        Returns:
            EnhancedResearchResult: Analysis + narrative planning results
        """
        import time
        start_time = time.time()
        
        logger.info("Generating enhanced research with narrative planning")
        
        try:
            # Step 1: Generate traditional storylines
            storylines = await self.get_storyline_from_game_data(game_data)
            
            # Step 2: Create basic analysis result
            analysis_result = AnalysisResult(
                storylines=storylines,
                confidence=0.85,
                analysis_type="comprehensive_with_narrative"
            )
            
            # Step 3: Generate narrative plan
            narrative_plan = await self._create_narrative_plan(storylines, game_data)
            
            # Step 4: Create processing metadata
            processing_time = time.time() - start_time
            metadata = {
                "processing_time_seconds": round(processing_time, 3),
                "storylines_count": len(storylines),
                "narrative_angles_identified": len(set(sl.narrative_angle for sl in narrative_plan.prioritized_storylines)),
                "primary_focus": narrative_plan.primary_narrative,
                "timestamp": time.time()
            }
            
            # Step 5: Combine everything
            enhanced_result = EnhancedResearchResult(
                analysis=analysis_result,
                narrative_plan=narrative_plan,
                processing_metadata=metadata
            )
            
            logger.info(f"Enhanced research completed in {processing_time:.3f}s with {len(storylines)} storylines")
            return enhanced_result
            
        except Exception as e:
            logger.error(f"Error in enhanced research generation: {e}")
            # Return fallback result
            fallback_storylines = ["Match analysis based on available data", "Key events and performances"]
            return self._create_fallback_enhanced_result(fallback_storylines, str(e))

    async def _create_narrative_plan(self, storylines: List[str], game_data: dict) -> NarrativePlan:
        """Create narrative plan based on storylines and game data."""
        logger.info("Creating narrative plan from storylines")
        
        try:
            # Extract key game context for narrative planning
            match_info = game_data.get("match_info", {})
            teams = [match_info.get("home_team", "Team A"), match_info.get("away_team", "Team B")]
            score = match_info.get("score", "Unknown")
            
            # Create narrative planning prompt
            narrative_prompt = f"""
            As a sports narrative expert, analyze these storylines and create a narrative plan:
            
            GAME CONTEXT:
            - Teams: {teams[0]} vs {teams[1]}
            - Score: {score}
            - Competition: {match_info.get('competition', 'Unknown')}
            
            STORYLINES TO ANALYZE:
            {chr(10).join(f'{i+1}. {storyline}' for i, storyline in enumerate(storylines))}
            
            Create a narrative plan that:
            1. Identifies the PRIMARY NARRATIVE (main story focus)
            2. Selects STORYTELLING FOCUS (dramatic, analytical, performance-based, tactical)
            3. Prioritizes storylines by importance and audience appeal
            4. Assigns narrative angles to each storyline
            5. Recommends narrative style and target audience
            
            Return JSON with this structure:
            {{
                "primary_narrative": "Main story focus",
                "storytelling_focus": "Primary approach",
                "narrative_style": "dramatic/analytical/balanced",
                "target_audience": "general_fans/tactical_enthusiasts/club_supporters",
                "confidence": 0.9,
                "prioritized_storylines": [
                    {{
                        "content": "storyline text",
                        "priority": 1,
                        "narrative_angle": "drama/analysis/performance/tactical",
                        "audience_appeal": 0.8,
                        "story_type": "match_decisive/player_spotlight/tactical_insight/historical_context"
                    }}
                ]
            }}
            """
            
            # Execute narrative planning
            result = await self._safe_llm_call(
                narrative_prompt, 
                "narrative_planning",
                max_retries=2
            )
            
            # Parse and validate narrative plan
            narrative_data = self._parse_narrative_plan(result)
            return self._create_narrative_plan_object(narrative_data, storylines)
            
        except Exception as e:
            logger.error(f"Error creating narrative plan: {e}")
            return self._create_fallback_narrative_plan(storylines)

    async def _analyze_components_separately(self, match_info, events, players, statistics, lineups) -> List[str]:
        """Analyze components separately using Chain of Thought reasoning."""
        all_storylines = []
        
        # 1. Analyze match information
        if match_info:
            logger.info("Analyzing match information with CoT...")
            match_storylines = await self._analyze_match_info_cot(match_info)
            all_storylines.extend(match_storylines)
            
        # 2. Analyze key events
        if events:
            logger.info("Analyzing key events with CoT...")
            event_storylines = await self._analyze_events_cot(events)
            all_storylines.extend(event_storylines)
            
        # 3. Analyze player performances
        if players:
            logger.info("Analyzing player performances with CoT...")
            player_storylines = await self._analyze_player_performances_cot(players)
            all_storylines.extend(player_storylines)
            
        # 4. Analyze team statistics
        if statistics:
            logger.info("Analyzing team statistics with CoT...")
            stats_storylines = await self._analyze_team_statistics_cot(statistics)
            all_storylines.extend(stats_storylines)
            
        # 5. Analyze lineups and formations
        if lineups:
            logger.info("Analyzing lineups with CoT...")
            lineup_storylines = await self._analyze_lineups_cot(lineups)
            all_storylines.extend(lineup_storylines)
        
        return all_storylines

    async def _safe_llm_call(self, prompt: str, operation_name: str, max_retries: int = 3, timeout: float = 30.0) -> str:
        """Make a safe LLM call with timeout and retry mechanism."""
        import asyncio
        base_delay = 1
        
        for attempt in range(max_retries):
            try:
                result = await asyncio.wait_for(
                    self.llm.ainvoke([HumanMessage(content=prompt)]),
                    timeout=timeout
                )
                return result.content
                
            except asyncio.TimeoutError:
                logger.warning(f"Timeout on attempt {attempt + 1}/{max_retries} for {operation_name}")
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    logger.info(f"Retrying {operation_name} in {delay} seconds...")
                    await asyncio.sleep(delay)
                    continue
                else:
                    logger.error(f"All retry attempts failed due to timeout for {operation_name}")
                    raise asyncio.TimeoutError(f"{operation_name} timed out after {max_retries} attempts")
                    
            except Exception as e:
                logger.error(f"Error in {operation_name} on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.info(f"Retrying {operation_name} in {delay} seconds...")
                    await asyncio.sleep(delay)
                    continue
                else:
                    raise e
        
        raise Exception(f"{operation_name} failed after {max_retries} attempts")

    def _parse_storylines_from_output(self, output_text: str) -> List[str]:
        """Parse storylines from LLM output text."""
        try:
            # Try to parse as JSON first
            if output_text.strip().startswith('['):
                storylines = json.loads(output_text)
                if isinstance(storylines, list):
                    return [str(s).strip() for s in storylines if s]
            
            # Try to find JSON array in the text
            import re
            json_pattern = r'\[.*?\]'
            matches = re.findall(json_pattern, output_text, re.DOTALL)
            for match in matches:
                try:
                    storylines = json.loads(match)
                    if isinstance(storylines, list):
                        return [str(s).strip() for s in storylines if s]
                except:
                    continue
            
            # Fallback: split by lines and clean
            lines = [line.strip() for line in output_text.split('\n') if line.strip()]
            # Filter out non-storyline content
            storylines = []
            for line in lines:
                if any(keyword in line.lower() for keyword in ['step', 'analysis', 'examination', 'validation']):
                    continue
                if line.startswith('"') and line.endswith('"'):
                    storylines.append(line[1:-1])
                elif len(line) > 10:  # Reasonable storyline length
                    storylines.append(line)
            
            return storylines[:10]  # Limit to reasonable number
            
        except Exception as e:
            logger.error(f"Error parsing storylines: {e}")
            return []

    async def _analyze_match_info_cot(self, match_info: dict) -> list[str]:
        """Analyze basic match information using Chain of Thought reasoning."""
        try:
            cot_prompt = f"""
            CHAIN OF THOUGHT ANALYSIS - MATCH INFORMATION:

            STEP 1 - DATA EXAMINATION:
            Let me examine the match information data:
            {match_info}

            STEP 2 - VALIDATION:
            I need to verify what information is explicitly available:
            - Team names: Check for exact team names
            - Venue: Look for venue information
            - League: Identify league context
            - Final score: Determine match result
            - Match date/time: Extract timing information

            STEP 3 - STORYLINE RULES APPLICATION:
            Applying the rules:
            - Focus on match context, teams, venue, league, and final score
            - Use exact team names, venue, and league information
            - Describe the match result clearly
            - NO historical data or assumptions

            STEP 4 - STORYLINE GENERATION:
            Based on the validated data, generate storylines.

            OUTPUT FORMAT: Return ONLY a JSON array of simple strings.
            Example: ["Team A defeated Team B 1-0 at Venue X", "The match was the opening/mid-season/closing fixture of the 2024 Premier League season"]
            """
            
            result = await self.llm.ainvoke([HumanMessage(content=cot_prompt)])
            storylines = self._parse_storylines_from_output(result.content)
            return storylines
            
        except Exception as e:
            logger.error(f"Error analyzing match info with CoT: {e}")
            return []

    async def _analyze_events_cot(self, events: list) -> list[str]:
        """Analyze key events using Chain of Thought reasoning."""
        try:
            cot_prompt = f"""
            CHAIN OF THOUGHT ANALYSIS - MATCH EVENTS:

            STEP 1 - DATA EXAMINATION:
            Let me examine the events data:
            {events}

            STEP 2 - EVENT CATEGORIZATION:
            I need to categorize and validate each event type:
            - Goals: Identify scorer, assist, time, team
            - Cards: Identify player, card type, time, team  
            - Substitutions: Identify players in/out, time, team
            - VAR events: Identify type and impact

            STEP 3 - VALIDATION RULES APPLICATION:
            Applying strict validation rules:
            - Each event must contain its own player and time data - DO NOT mix between events
            - Goal event player = only the player listed in that Goal event
            - Card event player = only the player listed in that Card event  
            - Substitution event players = only the players listed in that Substitution event

            STEP 4 - STORYLINE GENERATION:
            Generate factual storylines based on validated events.

            OUTPUT FORMAT: Return ONLY a JSON array of simple strings.
            Example: ["Player A scored the winning goal in the nth minute", "Player B was substituted in at n minutes, replacing Player C"]
            """
            
            result = await self.llm.ainvoke([HumanMessage(content=cot_prompt)])
            storylines = self._parse_storylines_from_output(result.content)
            return storylines
            
        except Exception as e:
            logger.error(f"Error analyzing events with CoT: {e}")
            return []

    async def _analyze_player_performances_cot(self, players: list) -> list[str]:
        """Analyze individual player performances using Chain of Thought reasoning."""
        try:
            cot_prompt = f"""
            CHAIN OF THOUGHT ANALYSIS - PLAYER PERFORMANCES:

            STEP 1 - DATA EXAMINATION:
            Let me examine the player performance data:
            {players}

            STEP 2 - PERFORMANCE CRITERIA IDENTIFICATION:
            I need to identify meaningful performance indicators:
            - Playing time: 60+ minutes
            - Pass accuracy: ≥ 80% with ≥ 35+ total passes
            - Defensive actions: ≥ 2 tackles, interceptions, or clearances
            - Duels: ≥ 4 duels won
            - Direct contributions: ≥ 1 goal or assist

            STEP 3 - STORYLINE GENERATION:
            Generate performance storylines based on validated data.

            OUTPUT FORMAT: Return ONLY a JSON array of simple strings describing player actions.
            Example: ["Player A completed 85% of passes with 45 total passes", "Player B won 8 out of 12 duels"]
            """
            
            result = await self.llm.ainvoke([HumanMessage(content=cot_prompt)])
            storylines = self._parse_storylines_from_output(result.content)
            return storylines
            
        except Exception as e:
            logger.error(f"Error analyzing player performances with CoT: {e}")
            return []

    async def _analyze_team_statistics_cot(self, statistics: list) -> list[str]:
        """Analyze team statistics using Chain of Thought reasoning."""
        try:
            cot_prompt = f"""
            CHAIN OF THOUGHT ANALYSIS - TEAM STATISTICS:

            STEP 1 - DATA EXAMINATION:
            Let me examine the team statistics data:
            {statistics}

            STEP 2 - STATISTIC CATEGORIZATION:
            I need to categorize the available team statistics:
            - Possession: Ball possession percentages
            - Shooting: Shots, shots on target, shots inside/outside box
            - Set pieces: Corners, free kicks
            - Discipline: Fouls, cards

            STEP 3 - STORYLINE GENERATION:
            Generate comparative team statistics storylines.

            OUTPUT FORMAT: Return ONLY a JSON array of simple strings.
            Example: ["Manchester United dominated possession with 55% compared to Fulham's 45%", "Both teams received 3 yellow cards each"]
            """
            
            result = await self.llm.ainvoke([HumanMessage(content=cot_prompt)])
            storylines = self._parse_storylines_from_output(result.content)
            return storylines
            
        except Exception as e:
            logger.error(f"Error analyzing team statistics with CoT: {e}")
            return []

    async def _analyze_lineups_cot(self, lineups: list) -> list[str]:
        """Analyze lineups and formations using Chain of Thought reasoning."""
        try:
            cot_prompt = f"""
            CHAIN OF THOUGHT ANALYSIS - LINEUPS AND FORMATIONS:

            STEP 1 - DATA EXAMINATION:
            Let me examine the lineup data:
            {lineups}

            STEP 2 - TACTICAL INFORMATION EXTRACTION:
            I need to extract tactical information:
            - Formations: Team formations (e.g., 4-2-3-1, 3-5-2)
            - Starting XI: Key players in starting lineup
            - Tactical setup: Defensive/attacking approach if evident

            STEP 3 - STORYLINE GENERATION:
            Generate lineup and formation storylines.

            OUTPUT FORMAT: Return ONLY a JSON array of simple strings.
            Example: ["Both teams employed a 4-2-3-1 formation", "Manchester United's starting XI featured key players like Bruno Fernandes"]
            """
            
            result = await self.llm.ainvoke([HumanMessage(content=cot_prompt)])
            storylines = self._parse_storylines_from_output(result.content)
            return storylines
            
        except Exception as e:
            logger.error(f"Error analyzing lineups with CoT: {e}")
            return []




    # All old methods using Runner have been removed and replaced with 
    # LangChain-based methods with Chain of Thought reasoning above
        
    async def get_history_from_team_data(self, team_data: dict) -> list[str]:
        """Get historical context from team data using Chain of Thought reasoning.
        
        Args:
            team_data: Team information including enhanced data (background/historical only)
            
        Returns:
            list[str]: Historical context and background information
        """
        logger.info("Analyzing historical context from team data using Chain of Thought reasoning")
        
        try:
            cot_prompt = f"""
            CHAIN OF THOUGHT ANALYSIS - TEAM HISTORICAL CONTEXT:

            STEP 1 - DATA EXAMINATION:
            Let me examine the team data for background information:
            {team_data}

            STEP 2 - CONTEXT IDENTIFICATION:
            I need to identify historical/background information:
            - Team history and achievements
            - Recent form or season performance
            - Head-to-head records
            - Notable players or transfers
            - League position or standings

            STEP 3 - VALIDATION RULES:
            Applying validation rules:
            - Use only background/historical information
            - Do NOT mention current match events
            - Only include facts explicitly in the data
            - No assumptions or inferences

            STEP 4 - STORYLINE GENERATION:
            Generate 3-5 background statements based on validated data.

            OUTPUT: JSON array of background statements.
            """
            
            # Use safe LLM call with timeout and retry
            try:
                content = await self._safe_llm_call(cot_prompt, "historical context analysis")
                storylines = self._parse_storylines_from_output(content)
                
                if not storylines:
                    return ["Historical context based on available team data", "Team performance analysis from provided data"]
                
                return storylines[:5]  # Limit to 5 background statements
                
            except Exception as e:
                logger.error(f"Safe LLM call failed for historical context: {e}")
                return ["Historical context analysis failed - using fallback insights", "Team performance analysis from provided data"]
            
        except Exception as e:
            logger.error(f"Error analyzing historical context with CoT: {e}")
            return ["Historical context based on available team data", "Team performance analysis from provided data"]

    async def get_performance_from_player_game_data(self, player_data: dict, game_data: dict) -> list[str]:
        """Analyze individual player performance using Chain of Thought reasoning.
        
        Args:
            player_data: Player information including enhanced data
            game_data: Compact game data for context (current match events only)
            
        Returns:
            list[str]: Player performance analysis based ONLY on current match events
        """
        logger.info("Analyzing individual player performance using Chain of Thought reasoning")
        
        try:
            cot_prompt = f"""
            CHAIN OF THOUGHT ANALYSIS - INDIVIDUAL PLAYER PERFORMANCE:

            STEP 1 - DATA EXAMINATION:
            Let me examine the player and game data:
            Player Data: {player_data}
            Game Data Events: {game_data.get("events", [])}
            Game Data Players: {game_data.get("players", [])}

            STEP 2 - PERFORMANCE COMPONENT IDENTIFICATION:
            I need to identify performance components:
            - Player events: Goals, assists, cards, substitutions
            - Player statistics: Passes, tackles, duels, ratings
            - Match involvement: Minutes played, key actions

            STEP 3 - VALIDATION RULES APPLICATION:
            Applying validation rules:
            - Only use current match events and statistics
            - Each event must contain its own player and time data
            - Do not mix events or assume connections
            - Verify exact numbers and statistics

            STEP 4 - CONTRIBUTION ASSESSMENT:
            Assess meaningful contributions:
            - Goals and assists
            - High pass accuracy with significant volume
            - Defensive actions (tackles, interceptions)
            - Duel success rate
            - Overall match impact

            STEP 5 - STORYLINE GENERATION:
            Generate player performance storylines based on current match data only.

            OUTPUT: JSON array of player performance statements.
            """
            
            result = await self.llm.ainvoke([HumanMessage(content=cot_prompt)])
            storylines = self._parse_storylines_from_output(result.content)
            
            if not storylines:
                return ["Player performance analysis based on available data", "Individual contributions from the match data"]
            
            return storylines
            
        except Exception as e:
            logger.error(f"Error analyzing player performance with CoT: {e}")
            return ["Player performance analysis based on available data", "Individual contributions from the match data"]

    def _parse_narrative_plan(self, result_text: str) -> dict:
        """Parse narrative plan from LLM response."""
        try:
            # Try to extract JSON from the result
            import re
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
            
            # If no JSON found, return None to trigger fallback
            return None
            
        except Exception as e:
            logger.error(f"Error parsing narrative plan: {e}")
            return None
    
    def _create_narrative_plan_object(self, narrative_data: dict, storylines: List[str]) -> NarrativePlan:
        """Create NarrativePlan object from parsed data."""
        if not narrative_data:
            return self._create_fallback_narrative_plan(storylines)
        
        try:
            # Extract prioritized storylines
            prioritized_storylines = []
            storylines_data = narrative_data.get("prioritized_storylines", [])
            
            for i, sl_data in enumerate(storylines_data):
                prioritized_storylines.append(StorylinePriority(
                    content=sl_data.get("content", storylines[i] if i < len(storylines) else "Story content"),
                    priority=sl_data.get("priority", i + 1),
                    narrative_angle=sl_data.get("narrative_angle", "analysis"),
                    audience_appeal=sl_data.get("audience_appeal", 0.7),
                    story_type=sl_data.get("story_type", "match_decisive")
                ))
            
            # If no prioritized storylines from data, create from original storylines
            if not prioritized_storylines:
                prioritized_storylines = self._create_default_prioritized_storylines(storylines)
            
            return NarrativePlan(
                primary_narrative=narrative_data.get("primary_narrative", "Match analysis and key moments"),
                storytelling_focus=narrative_data.get("storytelling_focus", "balanced"),
                prioritized_storylines=prioritized_storylines,
                narrative_style=narrative_data.get("narrative_style", "balanced"),
                target_audience=narrative_data.get("target_audience", "general_fans"),
                confidence=narrative_data.get("confidence", 0.8)
            )
            
        except Exception as e:
            logger.error(f"Error creating narrative plan object: {e}")
            return self._create_fallback_narrative_plan(storylines)
    
    def _create_fallback_narrative_plan(self, storylines: List[str]) -> NarrativePlan:
        """Create fallback narrative plan when AI planning fails."""
        logger.info("Creating fallback narrative plan")
        
        # Create default prioritized storylines
        prioritized_storylines = self._create_default_prioritized_storylines(storylines)
        
        return NarrativePlan(
            primary_narrative="Match recap with key highlights and analysis",
            storytelling_focus="balanced",
            prioritized_storylines=prioritized_storylines,
            narrative_style="analytical",
            target_audience="general_fans",
            confidence=0.7
        )
    
    def _create_default_prioritized_storylines(self, storylines: List[str]) -> List[StorylinePriority]:
        """Create default prioritized storylines from raw storylines."""
        prioritized = []
        
        for i, storyline in enumerate(storylines[:5]):  # Limit to top 5
            # Simple heuristic-based categorization
            narrative_angle = self._determine_narrative_angle(storyline)
            story_type = self._determine_story_type(storyline)
            audience_appeal = self._estimate_audience_appeal(storyline)
            
            prioritized.append(StorylinePriority(
                content=storyline,
                priority=i + 1,
                narrative_angle=narrative_angle,
                audience_appeal=audience_appeal,
                story_type=story_type
            ))
        
        return prioritized
    
    def _determine_narrative_angle(self, storyline: str) -> str:
        """Determine narrative angle based on storyline content."""
        storyline_lower = storyline.lower()
        
        if any(word in storyline_lower for word in ["dramatic", "winner", "last-minute", "comeback"]):
            return "drama"
        elif any(word in storyline_lower for word in ["tactics", "formation", "strategy", "system"]):
            return "tactical"
        elif any(word in storyline_lower for word in ["performance", "rating", "stats", "contributions"]):
            return "performance"
        else:
            return "analysis"
    
    def _determine_story_type(self, storyline: str) -> str:
        """Determine story type based on storyline content."""
        storyline_lower = storyline.lower()
        
        if any(word in storyline_lower for word in ["goal", "winner", "decisive", "crucial"]):
            return "match_decisive"
        elif any(word in storyline_lower for word in ["player", "performance", "standout", "individual"]):
            return "player_spotlight"
        elif any(word in storyline_lower for word in ["tactics", "formation", "tactical"]):
            return "tactical_insight"
        else:
            return "historical_context"
    
    def _estimate_audience_appeal(self, storyline: str) -> float:
        """Estimate audience appeal based on storyline content."""
        storyline_lower = storyline.lower()
        
        # High appeal keywords
        high_appeal_words = ["goal", "winner", "dramatic", "comeback", "historic", "record"]
        medium_appeal_words = ["performance", "key", "important", "significant"]
        
        if any(word in storyline_lower for word in high_appeal_words):
            return 0.9
        elif any(word in storyline_lower for word in medium_appeal_words):
            return 0.7
        else:
            return 0.6
    
    def _create_fallback_enhanced_result(self, storylines: List[str], error_msg: str) -> EnhancedResearchResult:
        """Create fallback enhanced result when processing fails."""
        import time
        
        # Create basic analysis
        analysis = AnalysisResult(
            storylines=storylines,
            confidence=0.6,
            analysis_type="fallback_analysis"
        )
        
        # Create fallback narrative plan
        narrative_plan = self._create_fallback_narrative_plan(storylines)
        
        # Create metadata
        metadata = {
            "processing_time_seconds": 0.1,
            "storylines_count": len(storylines),
            "narrative_angles_identified": 1,
            "primary_focus": "fallback_analysis",
            "timestamp": time.time(),
            "error": error_msg,
            "fallback_used": True
        }
        
        return EnhancedResearchResult(
            analysis=analysis,
            narrative_plan=narrative_plan,
            processing_metadata=metadata
        )