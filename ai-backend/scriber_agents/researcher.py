"""Research Agent.

This agent provides contextual background and analysis for sports articles.
It researches historical data, team/player statistics, and relevant context
to enrich the content generation process.
"""

import os
import logging
from typing import Any, List, Dict, Optional
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from agents import Agent, Runner
from utils.security import sanitize_log_input, sanitize_multiple_log_inputs

load_dotenv()
logger = logging.getLogger(__name__)


class TeamHistory(BaseModel):
    """Team historical matchup data."""
    
    total_matches: int = Field(description="Total number of matches between teams")
    team_wins: int = Field(description="Number of wins for primary team")
    opponent_wins: int = Field(description="Number of wins for opponent team")
    draws: int = Field(description="Number of draws")
    recent_results: List[str] = Field(description="Last 5 match results (W/L/D)")
    team_last_5: List[str] = Field(description="Primary team's last 5 results")
    opponent_last_5: List[str] = Field(description="Opponent team's last 5 results")
    storylines: List[str] = Field(description="Key storylines from historical data")


class PlayerPerformance(BaseModel):
    """Player performance analysis data."""
    
    season_stats: Dict[str, Any] = Field(description="Season statistics (goals, assists, etc.)")
    recent_form: Dict[str, Any] = Field(description="Recent form data")
    key_moments: List[str] = Field(description="Key moments and achievements")
    storylines: List[str] = Field(description="Player-related storylines")


class SeasonTrends(BaseModel):
    """Season trends and statistics."""
    
    league_standings: Dict[str, Any] = Field(description="Current league standings")
    season_stats: Dict[str, Any] = Field(description="Season statistics")
    trends: List[str] = Field(description="Current season trends")
    storylines: List[str] = Field(description="Season-related storylines")


class GameAnalysis(BaseModel):
    """Game analysis and storylines."""
    
    fixture_summary: Dict[str, Any] = Field(description="Fixture information")
    key_events: List[Dict[str, Any]] = Field(description="Key match events")
    storylines: List[str] = Field(description="Game-specific storylines")
    match_highlights: List[str] = Field(description="Match highlights")


class ResearchData(BaseModel):
    """Complete research data structure."""
    
    team_history: Optional[TeamHistory] = Field(description="Team historical data")
    player_performance: Optional[PlayerPerformance] = Field(description="Player performance data")
    season_trends: Optional[SeasonTrends] = Field(description="Season trends data")
    game_analysis: Optional[GameAnalysis] = Field(description="Game analysis data")
    top_storylines: List[str] = Field(description="Top 3-5 most important storylines", max_items=5)


# Agent prompts
TEAM_HISTORY_PROMPT = """
You are a sports research agent specializing in team historical analysis. Your task is to analyze the historical matchup data between two teams and extract key storylines.

Focus on:
1. Head-to-head record and recent form
2. Key historical moments between the teams
3. Current form of both teams
4. Tactical patterns and playing styles
5. Most compelling storylines for writers

Guidelines:
- Keep analysis simple and accessible for junior writers
- Focus on the most important 3-5 storylines
- Provide factual, objective analysis
- Highlight trends and patterns
- Consider recent form and momentum

Your output should include:
- Total matches, wins, draws between teams
- Recent results (last 5 meetings)
- Current form of both teams
- Key storylines that would interest readers
"""

PLAYER_PERFORMANCE_PROMPT = """
You are a sports research agent specializing in player performance analysis. Your task is to analyze a player's performance data and extract key storylines.

Focus on:
1. Season statistics and achievements
2. Recent form and momentum
3. Key moments and highlights
4. Player's role and impact on team
5. Most compelling storylines for writers

Guidelines:
- Keep analysis simple and accessible for junior writers
- Focus on the most important 3-5 storylines
- Provide factual, objective analysis
- Highlight exceptional performances
- Consider context and opposition quality

Your output should include:
- Season statistics (goals, assists, appearances, etc.)
- Recent form data (last 5 games)
- Key moments and achievements
- Player-related storylines
"""

SEASON_TRENDS_PROMPT = """
You are a sports research agent specializing in season trends analysis. Your task is to analyze current season data and extract key storylines.

Focus on:
1. League standings and title race
2. Season statistics and records
3. Current trends and patterns
4. Relegation battles and key races
5. Most compelling storylines for writers

Guidelines:
- Keep analysis simple and accessible for junior writers
- Focus on the most important 3-5 storylines
- Provide factual, objective analysis
- Highlight significant trends
- Consider the broader context

Your output should include:
- Current league standings
- Season statistics and records
- Key trends and patterns
- Season-related storylines
"""

GAME_ANALYSIS_PROMPT = """
You are a sports research agent specializing in game analysis. Your task is to analyze match data and extract key storylines.

Focus on:
1. Match result and scoreline
2. Key events and moments
3. Individual performances
4. Tactical aspects
5. Most compelling storylines for writers

Guidelines:
- Keep analysis simple and accessible for junior writers
- Focus on the most important 3-5 storylines
- Provide factual, objective analysis
- Highlight dramatic moments
- Consider the match context

Your output should include:
- Fixture summary and result
- Key match events
- Game-specific storylines
- Match highlights
"""

STORYLINE_GENERATION_PROMPT = """
You are a sports research agent specializing in storyline generation. Your task is to analyze multiple data sources and identify the top 3-5 most compelling storylines for sports articles.

Focus on:
1. Most newsworthy and interesting angles
2. Stories that would engage readers
3. Context and background information
4. Human interest elements
5. Tactical and statistical insights

Guidelines:
- Select only the most important 3-5 storylines
- Keep storylines simple and accessible for junior writers
- Focus on what makes this match/player/team interesting
- Consider historical context and current form
- Avoid overly complex analysis

Your output should be a list of 3-5 compelling storylines that writers can use as the foundation for their articles.
"""


# Agent instances
team_history_agent = Agent(
    name="TeamHistoryAgent",
    instructions=TEAM_HISTORY_PROMPT,
    output_type=TeamHistory,
    tools=[],
    model=os.getenv("OPENAI_MODEL", "gpt-4o"),
)

player_performance_agent = Agent(
    name="PlayerPerformanceAgent",
    instructions=PLAYER_PERFORMANCE_PROMPT,
    output_type=PlayerPerformance,
    tools=[],
    model=os.getenv("OPENAI_MODEL", "gpt-4o"),
)

season_trends_agent = Agent(
    name="SeasonTrendsAgent",
    instructions=SEASON_TRENDS_PROMPT,
    output_type=SeasonTrends,
    tools=[],
    model=os.getenv("OPENAI_MODEL", "gpt-4o"),
)

game_analysis_agent = Agent(
    name="GameAnalysisAgent",
    instructions=GAME_ANALYSIS_PROMPT,
    output_type=GameAnalysis,
    tools=[],
    model=os.getenv("OPENAI_MODEL", "gpt-4o"),
)

storyline_generation_agent = Agent(
    name="StorylineGenerationAgent",
    instructions=STORYLINE_GENERATION_PROMPT,
    output_type=List[str],
    tools=[],
    model=os.getenv("OPENAI_MODEL", "gpt-4o"),
)


class ResearchAgent:
    """Agent responsible for researching contextual information and analysis."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the Research Agent with configuration."""
        self.config = config or {}
        logger.info("Research Agent initialized")

    async def research_team_history(self, team_id: str, opponent_id: str) -> TeamHistory:
        """Research historical matchups between teams.

        Args:
            team_id: Primary team identifier
            opponent_id: Opponent team identifier

        Returns:
            TeamHistory: Historical context and storylines
        """
        team_safe, opponent_safe = sanitize_multiple_log_inputs(team_id, opponent_id)
        logger.info("Researching history between teams: %s vs %s", team_safe, opponent_safe)
        
        prompt = f"""
        Analyze the historical matchup between Team ID {team_id} and Team ID {opponent_id}.
        
        Provide historical context including:
        - Head-to-head record
        - Recent form of both teams
        - Key storylines from their meetings
        - Current form and momentum
        
        Focus on the most compelling 3-5 storylines that would interest readers.
        """
        
        try:
            result = await Runner.run(team_history_agent, prompt)
            return result.final_output_as(TeamHistory)
        except Exception as e:
            logger.error(f"Error researching team history: {e}")
            # Return default structure if agent fails
            return TeamHistory(
                total_matches=0,
                team_wins=0,
                opponent_wins=0,
                draws=0,
                recent_results=[],
                team_last_5=[],
                opponent_last_5=[],
                storylines=["Teams have limited historical data", "Both teams in good form this season"]
            )

    async def research_player_performance(self, player_id: str, context: Dict[str, Any]) -> PlayerPerformance:
        """Research player performance trends and statistics.

        Args:
            player_id: Player identifier
            context: Game/season context

        Returns:
            PlayerPerformance: Player analysis and storylines
        """
        logger.info("Researching player performance: %s", sanitize_log_input(player_id))
        
        prompt = f"""
        Analyze the performance of Player ID {player_id} in the context of {context}.
        
        Provide performance analysis including:
        - Season statistics and achievements
        - Recent form and momentum
        - Key moments and highlights
        - Player's role and impact
        
        Focus on the most compelling 3-5 storylines that would interest readers.
        """
        
        try:
            result = await Runner.run(player_performance_agent, prompt)
            return result.final_output_as(PlayerPerformance)
        except Exception as e:
            logger.error(f"Error researching player performance: {e}")
            # Return default structure if agent fails
            return PlayerPerformance(
                season_stats={"goals": 0, "assists": 0, "appearances": 0},
                recent_form={"last_5_games": [], "goals_in_last_5": 0, "assists_in_last_5": 0},
                key_moments=["Player has been consistent this season"],
                storylines=["Player in good form", "Key contributor to team success"]
            )

    async def research_season_trends(self, league: str, season: str) -> SeasonTrends:
        """Research current season trends and statistics.

        Args:
            league: League identifier
            season: Season identifier

        Returns:
            SeasonTrends: Season trends and storylines
        """
        league_safe, season_safe = sanitize_multiple_log_inputs(league, season)
        logger.info("Researching season trends for %s - %s", league_safe, season_safe)
        
        prompt = f"""
        Analyze the current season trends for League {league} in Season {season}.
        
        Provide season analysis including:
        - Current league standings
        - Season statistics and records
        - Key trends and patterns
        - Title race and relegation battles
        
        Focus on the most compelling 3-5 storylines that would interest readers.
        """
        
        try:
            result = await Runner.run(season_trends_agent, prompt)
            return result.final_output_as(SeasonTrends)
        except Exception as e:
            logger.error(f"Error researching season trends: {e}")
            # Return default structure if agent fails
            return SeasonTrends(
                league_standings={"top_3": [], "relegation_zone": [], "title_race": "Competitive season"},
                season_stats={"total_goals": 0, "avg_goals_per_game": 0},
                trends=["Competitive season with close title race"],
                storylines=["Exciting season with multiple contenders", "Close battles throughout the table"]
            )

    async def analyze_game_data(self, game_data: Dict[str, Any]) -> GameAnalysis:
        """Analyze game data and extract key storylines.

        Args:
            game_data: Raw game data from Data Collector

        Returns:
            GameAnalysis: Game analysis and storylines
        """
        logger.info("Analyzing game data for storylines")
        
        prompt = f"""
        Analyze the following game data and extract key storylines:
        
        {game_data}
        
        Provide game analysis including:
        - Match result and scoreline
        - Key events and moments
        - Individual performances
        - Tactical aspects
        
        Focus on the most compelling 3-5 storylines that would interest readers.
        """
        
        try:
            result = await Runner.run(game_analysis_agent, prompt)
            return result.final_output_as(GameAnalysis)
        except Exception as e:
            logger.error(f"Error analyzing game data: {e}")
            # Return default structure if agent fails
            return GameAnalysis(
                fixture_summary={},
                key_events=[],
                storylines=["Exciting match with plenty of action", "Key players making the difference"],
                match_highlights=["Dramatic finish", "Outstanding individual performances"]
            )

    async def generate_storylines(self, data_list: List[Dict[str, Any]]) -> List[str]:
        """Generate storylines from collected data.

        Args:
            data_list: List of data dictionaries from Data Collector

        Returns:
            List[str]: Top 3-5 most important storylines
        """
        logger.info("Generating storylines from %d data sources", len(data_list))
        
        prompt = f"""
        Analyze the following sports data and identify the top 3-5 most compelling storylines:
        
        {data_list}
        
        Focus on:
        - Most newsworthy and interesting angles
        - Stories that would engage readers
        - Context and background information
        - Human interest elements
        
        Return only the top 3-5 most important storylines that writers can use as the foundation for their articles.
        """
        
        try:
            result = await Runner.run(storyline_generation_agent, prompt)
            storylines = result.final_output_as(List[str])
            return storylines[:5]  # Ensure we only return max 5 storylines
        except Exception as e:
            logger.error(f"Error generating storylines: {e}")
            # Return default storylines if agent fails
            return [
                "Exciting match with plenty of action",
                "Key players making the difference",
                "Tactical battle between managers"
            ]

    async def execute(self, task: Dict[str, Any]) -> ResearchData:
        """Execute research task and return comprehensive research data.

        Args:
            task: Task dictionary containing research parameters

        Returns:
            ResearchData: Complete research data with storylines
        """
        logger.info("Executing research task")
        
        research_data = ResearchData(top_storylines=[])
        
        try:
            # Extract task parameters
            team_id = task.get("team_id")
            opponent_id = task.get("opponent_id")
            player_id = task.get("player_id")
            league = task.get("league")
            season = task.get("season")
            game_data = task.get("game_data")
            
            # Perform research based on available data
            if team_id and opponent_id:
                research_data.team_history = await self.research_team_history(team_id, opponent_id)
            
            if player_id:
                context = {"league": league, "season": season}
                research_data.player_performance = await self.research_player_performance(player_id, context)
            
            if league and season:
                research_data.season_trends = await self.research_season_trends(league, season)
            
            if game_data:
                research_data.game_analysis = await self.analyze_game_data(game_data)
            
            # Generate top storylines from all collected data
            data_list = []
            if research_data.team_history:
                data_list.append({"type": "team_history", "data": research_data.team_history.dict()})
            if research_data.player_performance:
                data_list.append({"type": "player_performance", "data": research_data.player_performance.dict()})
            if research_data.season_trends:
                data_list.append({"type": "season_trends", "data": research_data.season_trends.dict()})
            if research_data.game_analysis:
                data_list.append({"type": "game_analysis", "data": research_data.game_analysis.dict()})
            
            if data_list:
                research_data.top_storylines = await self.generate_storylines(data_list)
            
            logger.info("Research task completed successfully")
            return research_data
            
        except Exception as e:
            logger.error(f"Error executing research task: {e}")
            # Return basic structure with default storylines
            research_data.top_storylines = [
                "Exciting match with plenty of action",
                "Key players making the difference",
                "Tactical battle between managers"
            ]
            return research_data
