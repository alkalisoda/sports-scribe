"""Research Agent.

This agent provides contextual background and analysis for sports articles.
It researches historical data, team/player statistics, and relevant context
to enrich the content generation process.
"""

import logging
from typing import Any, List, Dict
from dotenv import load_dotenv
import json

from agents import Agent, Runner

load_dotenv()
logger = logging.getLogger(__name__)


class ResearchAgent:
    """Agent responsible for researching contextual information and analysis."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the Research Agent with configuration."""
        self.config = config or {}
        
        # Initialize the research agent without web search capability
        self.agent = Agent(
            instructions="""You are a sports research agent specializing in analyzing game data, team history, and player performance. 
            Your task is to provide clear, engaging storylines and analysis that junior writers can easily understand and use.
            
            CRITICAL REQUIREMENTS:
            - ONLY use information that is explicitly provided in the data
            - DO NOT invent, assume, or speculate about any facts not present in the data
            - If data is missing or incomplete, acknowledge this limitation
            - Base all analysis strictly on the factual data provided
            - Do not add external knowledge or assumptions
            
            Focus on:
            1. Most important 3-5 storylines only (based on provided data)
            2. Historical context between teams (from provided data only)
            3. Individual player performances and impact (from provided data only)
            4. Key moments and turning points (from provided data only)
            5. Tactical and strategic insights (from provided data only)
            
            Guidelines:
            - Keep analysis simple and accessible for junior writers
            - Focus on what makes this match/player/team interesting based on actual data
            - Provide factual, objective analysis using only provided information
            - Highlight human interest elements that are supported by the data
            - Consider broader context and significance only if supported by the data
            - If data is insufficient, state what information is missing rather than making assumptions
            
            Always return clear, structured analysis that writers can immediately use, based solely on the provided data.""",
            name="ResearchAgent",
            output_type=str,
            model=self.config.get("model", "gpt-4o-mini"),
        )
        
        logger.info("Research Agent initialized successfully")

    async def get_storyline_from_game_data(self, game_data: dict) -> list[str]:
        """Get storylines from game data ONLY (current match events).
        
        Args:
            game_data: Game data from Data Collector (ONLY current match events)
            
        Returns:
            list[str]: List of storylines based ONLY on current match events
        """
        logger.info("Generating storylines from game data (current match events only)")
        
        try:
            prompt = f"""
            You are analyzing game data for THIS SPECIFIC MATCH ONLY. Your task is to extract factual storylines that actually happened in this game.

            GAME DATA (CURRENT MATCH EVENTS ONLY):
            {game_data}

            CRITICAL MATCHING RULES:
            1. ONLY use information that explicitly appears in the game data above
            2. ONLY describe events that actually occurred in THIS match
            3. DO NOT make assumptions, inferences, or interpretations
            4. DO NOT include any historical context or background information
            5. DO NOT mention player or team statistics unless they appear in the match events
            6. If information is not clearly present in the data, DO NOT include it
            7. Focus ONLY on: goals, cards, substitutions, final score, venue, date, teams
            8. CRITICAL: When mentioning players, teams, or events, use EXACTLY the names and details from the data
            9. CRITICAL: Do not mix up player names, team names, or event times
            10. CRITICAL: If a player name is unclear or incomplete in the data, do not guess or complete it
            11. CRITICAL: Verify that each player mentioned actually participated in the specific event described

            REQUIRED FORMAT:
            Output ONLY a JSON array of 3-5 factual statements about THIS match.
            Each statement must be directly supported by the game data.
            Example format: ["Fact 1 about this match", "Fact 2 about this match", "Fact 3 about this match"]

            VALID TOPICS (only if data supports them):
            - Goals scored in this match (player, time, team)
            - Cards shown in this match (player, time, type)
            - Substitutions made in this match (player, time)
            - Final score of this match
            - Teams that played in this match
            - Venue where this match was played
            - Date when this match was played

            INVALID TOPICS (do not include):
            - Player historical statistics
            - Team historical performance
            - Previous meetings between teams
            - Season-long statistics
            - Background information not in the match data
            - Any player or team information not explicitly in the match events

            Instructions:
            - Output only a JSON array of strings
            - No explanations, no markdown, no extra text
            - Each statement must be a fact from THIS match only
            - If you cannot find clear facts, output fewer statements
            - Be extremely conservative - only include what is clearly stated in the data
            - Double-check all player names, team names, and event details against the provided data
            """
            
            result = await Runner.run(self.agent, prompt)
            try:
                storylines = json.loads(result.final_output)
                if isinstance(storylines, list):
                    if all(isinstance(s, dict) and len(s) == 1 for s in storylines):
                        return [list(s.values())[0] for s in storylines]
                    return [str(s).strip() for s in storylines if s]
            except Exception:
                return [line.strip() for line in result.final_output.splitlines() if line.strip()]
            
        except Exception as e:
            logger.error(f"Error generating storylines from game data: {e}")
            return ["Match analysis based on available game data", "Key moments and player performances from the data"]

    async def get_history_from_team_data(self, team_data: dict) -> list[str]:
        """Get historical context from team data ONLY (background information).
        
        Args:
            team_data: Team information including enhanced data (background/historical only)
            
        Returns:
            list[str]: Historical context and background information
        """
        logger.info("Analyzing historical context from team data (background information only)")
        
        try:
            prompt = f"""
            You are analyzing BACKGROUND and HISTORICAL information about teams. This is NOT about the current match.

            TEAM DATA (BACKGROUND/HISTORICAL INFORMATION ONLY):
            {team_data}

            STRICT RULES:
            1. This data is for BACKGROUND CONTEXT only, not current match events
            2. ONLY use information that explicitly appears in the team data above
            3. DO NOT mention any events from the current match
            4. DO NOT make assumptions about current match performance
            5. Focus on historical facts, team information, and background context
            6. If information is not clearly present in the data, DO NOT include it

            REQUIRED FORMAT:
            Output ONLY a JSON array of 3-5 background context statements.
            Each statement must be directly supported by the team data.
            Example format: ["Background fact 1", "Background fact 2", "Background fact 3"]

            VALID TOPICS (only if data supports them):
            - Team founding dates and history
            - Stadium information and capacity
            - League and competition information
            - Team codes and country information
            - Historical team achievements (if mentioned in data)
            - Background information about teams

            INVALID TOPICS (do not include):
            - Current match events
            - Current match scores
            - Current match players
            - Current match statistics
            - Any information not in the provided team data

            Instructions:
            - Output only a JSON array of strings
            - No explanations, no markdown, no extra text
            - Each statement must be background information only
            - If you cannot find clear background facts, output fewer statements
            - Be extremely conservative - only include what is clearly stated in the data
            - Remember: This is BACKGROUND context, not current match information
            """
            
            result = await Runner.run(self.agent, prompt)
            try:
                storylines = json.loads(result.final_output)
                if isinstance(storylines, list):
                    return [str(s).strip() for s in storylines if s]
            except Exception:
                return [line.strip() for line in result.final_output.splitlines() if line.strip()]
            
        except Exception as e:
            logger.error(f"Error analyzing historical context: {e}")
            return ["Historical context based on available team data", "Team performance analysis from provided data"]

    async def get_performance_from_player_game_data(self, player_data: dict, game_data: dict) -> list[str]:
        """Analyze individual player performance from game data ONLY (current match events).
        
        Args:
            player_data: Player information including enhanced data
            game_data: Game data for context (current match events only)
            
        Returns:
            list[str]: Player performance analysis based ONLY on current match events
        """
        logger.info("Analyzing individual player performance from game data (current match events only)")
        
        try:
            prompt = f"""
            You are analyzing player performance from THIS SPECIFIC MATCH. Focus on what players actually did in this game.

            GAME CONTEXT (CURRENT MATCH EVENTS ONLY):
            {game_data}

            PLAYER DATA (CURRENT MATCH + HISTORICAL BACKGROUND):
            {player_data}

            CRITICAL MATCHING RULES:
            1. ONLY describe what players did in THIS match (goals, cards, substitutions, etc.)
            2. ONLY use information that explicitly appears in the game data above
            3. DO NOT make assumptions about player performance
            4. DO NOT confuse historical statistics with current match events
            5. If a player did nothing notable in this match, DO NOT mention them
            6. Historical data is for background context only, not current performance
            7. CRITICAL: When mentioning players, use EXACTLY the names from the match events data
            8. CRITICAL: Do not mix up player names, event times, or team affiliations
            9. CRITICAL: If a player name is unclear or incomplete in the data, do not guess or complete it
            10. CRITICAL: Verify that each player mentioned actually participated in the specific event described
            11. CRITICAL: Only mention players who have clear, verifiable actions in the match events

            REQUIRED FORMAT:
            Output ONLY a JSON array of 3-5 factual statements about player performance in THIS match.
            Each statement must be directly supported by the game data.
            Example format: ["Player X scored in this match", "Player Y received a card in this match"]

            VALID TOPICS (only if data supports them):
            - Goals scored by players in this match
            - Cards received by players in this match
            - Substitutions made by players in this match
            - Players who started the match
            - Players who were on the bench
            - Specific match events involving players

            INVALID TOPICS (do not include):
            - Player historical statistics
            - Player season-long performance
            - Player background information not relevant to this match
            - Assumptions about player performance
            - Any information not clearly stated in the match data
            - Any player not explicitly mentioned in the match events

            Instructions:
            - Output only a JSON array of strings
            - No explanations, no markdown, no extra text
            - Each statement must be about THIS match only
            - If you cannot find clear player facts from this match, output fewer statements
            - Be extremely conservative - only include what is clearly stated in the match data
            - Focus on actual events, not interpretations or background
            - Double-check all player names and event details against the provided match data
            """
            
            result = await Runner.run(self.agent, prompt)
            try:
                storylines = json.loads(result.final_output)
                if isinstance(storylines, list):
                    return [str(s).strip() for s in storylines if s]
            except Exception:
                return [line.strip() for line in result.final_output.splitlines() if line.strip()]
            
        except Exception as e:
            logger.error(f"Error analyzing player performance: {e}")
            return ["Player performance analysis based on available data", "Individual contributions from the match data"]
