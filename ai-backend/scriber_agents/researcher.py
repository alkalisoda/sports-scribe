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
            
            CORE PRINCIPLES:
            - ONLY use information explicitly provided in the data
            - DO NOT invent, assume, or speculate about facts not present in the data
            - When in doubt, exclude rather than include
            - Base all analysis strictly on factual data provided
            - CRITICAL: Clearly distinguish between THIS MATCH events and other matches/background
            - CRITICAL: Only describe events that actually occurred in THIS specific match
            - CRITICAL: If an event did not happen in THIS match, DO NOT include it
            
            DATA VERIFICATION RULES:
            - Double-check every player name spelling exactly as in the data
            - Use precise minute times: "elapsed" + "extra" format (e.g., 90+1 for elapsed:90, extra:1)
            - Cross-reference each event with the correct player
            - Use season format like "2021/22 season" not just "2021 season"
            
            TIME FORMAT RULES:
            - "elapsed": main referee time (e.g., 90 = 90th minute)
            - "extra": stoppage time (e.g., 1 = 1st minute of stoppage time)
            - Combined format: "elapsed" + "extra" (e.g., 90+1 for elapsed:90, extra:1)
            - Always use the combined format in outputs
            
            SUBSTITUTION LOGIC:
            - "startXI" array = players who started the match
            - "substitutes" array = players who were on the bench
            - In substitution events: "player" field = who went off, "assist" field = who came on
            - Players cannot participate in events after being substituted off
            - Substitute players cannot participate in events before coming on
            - Be explicit about substitution direction (off vs on)
            
            EXCLUSION RULES:
            - Do not describe actions by players who were already substituted off
            - Do not describe actions by players before they came on as substitutes
            - Do not use vague time descriptions like "shortly after" without specific minutes
            - Do not mix up player names (e.g., Mount vs Maguire)
            - Do not use approximate times when exact times are available (e.g., 90 vs 90+1 for elapsed:90, extra:1)
            - Do not use ambiguous substitution descriptions
            - CRITICAL: Do not include events that did not happen in THIS match (e.g., Mount receiving a card when he didn't)
            - CRITICAL: Do not fabricate events like goals, cards, or other actions not in the data
            - CRITICAL: Do not include background/historical events as if they happened in THIS match
            
            Focus on:
            1. Most important 3-5 storylines only (from THIS MATCH data only)
            2. Historical context between teams (background information only, not THIS MATCH events)
            3. Individual player performances and impact (from THIS MATCH events only)
            4. Key moments and turning points (from THIS MATCH events only)
            5. Tactical and strategic insights (from THIS MATCH data only)
            
            Guidelines:
            - Keep analysis simple and accessible for junior writers
            - Focus on what makes THIS MATCH interesting based on actual THIS MATCH data
            - Provide factual, objective analysis using only THIS MATCH information
            - If data is insufficient, state what information is missing rather than making assumptions
            - CRITICAL: Always specify when describing events - "in this match", "during this game", etc.
            - CRITICAL: Never mix THIS MATCH events with background/historical information
            
            Always return clear, structured analysis that writers can immediately use, based solely on the provided data.""",
            name="ResearchAgent",
            output_type=str,
            model=self.config.get("model", "gpt-4o-mini"),
        )
        
        logger.info("Research Agent initialized successfully")

    async def get_substitution_analysis(self, game_data: dict) -> list[str]:
        """Analyze substitution events with precise verification of who came on vs who went off.
        
        Args:
            game_data: Game data containing events and lineup information
            
        Returns:
            list[str]: Accurate substitution statements
        """
        logger.info("Analyzing substitution events with precise verification")
        
        try:
            prompt = f"""
            You are analyzing substitution events from THIS SPECIFIC MATCH ONLY.

            GAME DATA (THIS MATCH ONLY):
            {game_data}

            CRITICAL RULES:
            - ONLY analyze substitutions that actually occurred in THIS MATCH
            - Cross-reference with lineup data: "startXI" = starters, "substitutes" = bench
            - "player" field = who went OFF, "assist" field = who came ON
            - Verify chronological logic: players cannot act after being substituted off
            - Use precise minute times: "elapsed" + "extra" format (e.g., 90+1 for elapsed:90, extra:1)
            - Always specify "in this match" or "during this game" when describing events

            VALID STATEMENTS (only if explicitly supported by data):
            - "Player A was substituted off in the Xth minute of this match"
            - "Player B came on as a substitute in the Xth minute of this match"
            - "Player B replaced Player A in the Xth minute of this match"

            STRICTLY FORBIDDEN:
            - Substitutions not explicitly recorded in THIS MATCH data
            - Incorrect substitution direction
            - Players not mentioned in lineup data
            - Actions by players after being substituted off
            - Actions by substitutes before coming on
            - Vague time descriptions like "shortly after" - use "elapsed" + "extra" format instead
            - Events from other matches or background information

            REQUIRED FORMAT:
            Output ONLY a JSON array of accurate substitution statements.
            Example format: ["Substitution statement 1", "Substitution statement 2"]

            Instructions:
            - Output only a JSON array of strings
            - No explanations, no markdown, no extra text
            - Be extremely conservative - only include what is clearly stated in THIS MATCH data
            - When uncertain, exclude rather than include
            - Always specify that events happened "in this match"
            """
            
            result = await Runner.run(self.agent, prompt)
            try:
                substitutions = json.loads(result.final_output)
                if isinstance(substitutions, list):
                    return [str(s).strip() for s in substitutions if s]
            except Exception:
                return [line.strip() for line in result.final_output.splitlines() if line.strip()]
            
        except Exception as e:
            logger.error(f"Error analyzing substitutions: {e}")
            return ["Substitution analysis based on available data"]

    async def get_storyline_from_game_data(self, game_data: dict) -> list[str]:
        """Get comprehensive storylines from game data including turning points, timeline, stats, and analysis.
        
        Args:
            game_data: Game data from Data Collector (current match events)
            
        Returns:
            list[str]: Comprehensive list of storylines including analysis
        """
        logger.info("Generating comprehensive storylines from game data with enhanced analysis")
        
        try:
            # Get additional analysis components from game_data
            turning_points = await self.get_turning_points(game_data)
            best_worst_moments = await self.get_best_and_worst_moments(game_data)
            missed_chances = await self.get_missed_chances(game_data)
            substitution_analysis = await self.get_substitution_analysis(game_data)
            
            # Get timeline and stats if available from game_data
            event_timeline = []
            stat_summary = []
            formations = []
            
            try:
                event_timeline = await self.get_event_timeline(game_data)
            except Exception as e:
                logger.warning(f"Could not generate event timeline: {e}")
            
            try:
                stat_summary = await self.get_stat_summary(game_data)
            except Exception as e:
                logger.warning(f"Could not generate stat summary: {e}")
            
            try:
                formations = await self.get_formations_from_lineup_data(game_data)
            except Exception as e:
                logger.warning(f"Could not generate formations: {e}")
            
            prompt = f"""
            You are analyzing game data for THIS SPECIFIC MATCH ONLY.

            GAME DATA (CURRENT MATCH EVENTS ONLY):
            {game_data}

            ADDITIONAL ANALYSIS DATA:
            - Turning Points: {turning_points}
            - Best/Worst Moments: {best_worst_moments}
            - Missed Chances: {missed_chances}
            - Substitution Analysis: {substitution_analysis}
            - Event Timeline: {event_timeline}
            - Statistical Summary: {stat_summary}
            - Team Formations: {formations}

            CRITICAL RULES:
            - ONLY use information explicitly provided in THIS MATCH data
            - ONLY describe events that actually occurred in THIS match
            - Use EXACTLY the names and details from THIS MATCH data
            - Verify chronological logic - players cannot act after being substituted off
            - Use specific minute times: "elapsed" + "extra" format (e.g., 90+1 for elapsed:90, extra:1)
            - Double-check every player name against the exact spelling in the data
            - Be precise about substitution direction (off vs on)
            - When in doubt, exclude rather than include
            - CRITICAL: Always specify "in this match", "during this game", or "of this match" when describing events
            - CRITICAL: Do not fabricate events that did not happen in THIS match (e.g., Mount receiving a card when he didn't)

            REQUIRED FORMAT:
            Output ONLY a JSON array of 5-8 comprehensive storylines.
            Example format: ["Storyline 1", "Storyline 2", "Storyline 3"]

            STORYLINE COMPONENTS (when data supports them):
            - Key match events (goals, cards, substitutions, final score) from THIS MATCH
            - Turning points that changed the game's momentum in THIS MATCH
            - Best and worst moments that defined THIS MATCH
            - Missed opportunities that could have changed the outcome of THIS MATCH
            - Statistical insights (possession, shots, cards, etc.) from THIS MATCH
            - Teams and venue information for THIS MATCH

            INVALID TOPICS (do not include):
            - Player historical statistics from other matches
            - Team historical performance from other matches
            - Previous meetings between teams
            - Season-long statistics
            - Background information not in THIS MATCH data
            - Events that did not happen in THIS MATCH

            Instructions:
            - Output only a JSON array of strings
            - No explanations, no markdown, no extra text
            - Be extremely conservative - only include what is clearly stated in THIS MATCH data
            - Make storylines interesting and narrative-driven while staying factual
            - When uncertain, exclude rather than include
            - Always specify that events happened "in this match" or "during this game"
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
            logger.error(f"Error generating comprehensive storylines from game data: {e}")
            return ["Comprehensive match analysis based on available game data", "Key moments and turning points from the match"]
        
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
            You are analyzing player performance from THIS SPECIFIC MATCH.

            GAME CONTEXT (CURRENT MATCH EVENTS ONLY):
            {game_data}

            PLAYER DATA (CURRENT MATCH + HISTORICAL BACKGROUND):
            {player_data}

            CRITICAL RULES:
            - ONLY describe what players did in THIS match (goals, cards, substitutions, etc.)
            - ONLY use information explicitly provided in THIS MATCH game data
            - Use EXACTLY the names from THIS MATCH events data
            - Verify chronological logic - players cannot act after being substituted off
            - Use specific minute times: "elapsed" + "extra" format (e.g., 90+1 for elapsed:90, extra:1)
            - Double-check every player name against the exact spelling in the data
            - Be precise about substitution direction (off vs on)
            - When in doubt, exclude rather than include
            - CRITICAL: Do not include events that did not happen in THIS match (e.g., Mount receiving a card when he didn't)
            - CRITICAL: Always specify "in this match", "during this game", or "of this match" when describing events

            REQUIRED FORMAT:
            Output ONLY a JSON array of 3-5 factual statements about player performance.
            Example format: ["Player X scored in this match", "Player Y received a card in this match"]

            VALID TOPICS (only if data supports them):
            - Goals scored by players in THIS match
            - Cards received by players in THIS match
            - Substitutions made by players in THIS match
            - Players who started THIS match
            - Players who were on the bench in THIS match
            - Specific match events involving players in THIS match

            INVALID TOPICS (do not include):
            - Player historical statistics from other matches
            - Player season-long performance from other matches
            - Player background information not relevant to THIS match
            - Assumptions about player performance
            - Any information not clearly stated in THIS MATCH data
            - Events that did not happen in THIS match

            Instructions:
            - Output only a JSON array of strings
            - No explanations, no markdown, no extra text
            - Be extremely conservative - only include what is clearly stated in THIS MATCH data
            - Focus on actual events from THIS match, not interpretations or background
            - When uncertain, exclude rather than include
            - Always specify that events happened "in this match" or "during this game"
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
    
    async def get_turning_points(self, game_data: dict) -> list[str]:
        """
        Analyze the match and return key turning points that shaped the result.
        Focus on dramatic shifts in momentum (e.g. red cards, equalizers, late goals).
        Args:
            game_data: Match event data (goals, cards, substitutions, etc.)
        Returns:
            list[str]: 2-3 turning point statements from the match
        """
        logger.info("Analyzing match for turning points (game-changing moments)")
        try:
            prompt = f"""
            You are analyzing THIS SPECIFIC MATCH ONLY to extract the 2-3 most significant turning points that shaped the outcome.

            GAME DATA (THIS MATCH ONLY - ALL INFORMATION MUST COME FROM HERE):
            {game_data}

            ABSOLUTE RULES - YOU MUST FOLLOW THESE EXACTLY:
            1. ONLY use information that explicitly appears in the game data above
            2. ONLY identify turning points that actually occurred in THIS specific match
            3. DO NOT make any assumptions, inferences, or interpretations beyond what is stated in the data
            4. DO NOT include any background or historical data
            5. DO NOT add any external knowledge or context
            6. CRITICAL: Every turning point must be a clear, specific match event with verifiable impact
            7. CRITICAL: Be extremely conservative - only mention what clearly happened in this match
            8. CRITICAL: If information is unclear or missing, do not speculate or assume
            9. CRITICAL: If an event did not explicitly happen, DO NOT include it as a turning point
            10. CRITICAL: Only include events that are clearly documented in the data
            11. CRITICAL: When in doubt about whether something was a turning point, exclude it

            VALID TURNING POINTS (only if explicitly supported by game data):
            - Red cards that changed momentum and team dynamics
            - Equalizing goals that brought teams level
            - Go-ahead goals that gave a team the lead
            - Goals scored late in the match (85+ minutes)
            - Penalties awarded, scored, or missed
            - Back-to-back goals that shifted control dramatically
            - Impactful substitutions where a player scores shortly after entering
            - Own goals that changed the course of the match
            - Goals that broke deadlocks or extended leads significantly

            STRICTLY FORBIDDEN (DO NOT INCLUDE):
            - Any background or historical data about teams or players
            - Anything not explicitly shown in the match events
            - Vague or speculative statements about momentum
            - Assumptions about psychological impact
            - External commentary or analysis
            - Events from other matches or seasons
            - Player or team statistics not from this match

            DATA VALIDATION REQUIREMENTS:
            - Verify that each turning point actually occurred in this match
            - Confirm that the timing and details match the game data exactly
            - Ensure that the impact described is supported by the data
            - Cross-reference all player names and team names with the data
            - Validate that the sequence of events is accurate
            - Verify that each player mentioned actually participated in the specific event described

            REQUIRED FORMAT:
            Output ONLY a JSON array of 2-3 factual turning point statements.
            Each must be a clear, specific match event with demonstrable impact.
            No extra commentary, no markdown, no explanations.
            Example format: ["Turning point 1", "Turning point 2", "Turning point 3"]

            Instructions:
            - Output only a JSON array of strings
            - No explanations, no markdown, no extra text
            - Each turning point must be a specific event from this match
            - If you cannot find clear turning points, output fewer statements
            - Be extremely conservative - only include what is clearly stated in the data
            - Focus on actual events with clear impact, not interpretations
            - If data is insufficient, acknowledge the limitation rather than making assumptions
            - Only mention players with clear, verifiable actions in match events
            - EXCLUSION PRINCIPLE: If an event did not happen, DO NOT include it as a turning point
            - EXCLUSION PRINCIPLE: When uncertain, exclude rather than include
            - EXCLUSION PRINCIPLE: Only include events that are clearly documented in the data
            """
            result = await Runner.run(self.agent, prompt)
            try:
                points = json.loads(result.final_output)
                if isinstance(points, list):
                    return [str(p).strip() for p in points if p]
            except Exception:
                return [line.strip() for line in result.final_output.splitlines() if line.strip()]
        except Exception as e:
            logger.error(f"Error analyzing turning points: {e}")
            return ["Turning point analysis based on available data"]
        
    async def get_event_timeline(self, game_data: dict) -> list[str]:
        logger.info("Generating minute-by-minute event timeline")
        try:
            prompt = f"""
            You are creating a chronological timeline of events from THIS SPECIFIC MATCH ONLY.

            GAME DATA (THIS MATCH ONLY - ALL INFORMATION MUST COME FROM HERE):
            {game_data}

            ABSOLUTE RULES - YOU MUST FOLLOW THESE EXACTLY:
            1. ONLY use information that explicitly appears in the game data above
            2. ONLY include events that actually occurred in THIS specific match
            3. DO NOT make any assumptions, inferences, or interpretations beyond what is stated in the data
            4. DO NOT include any background or historical data
            5. DO NOT add any external knowledge or context
            6. CRITICAL: Every event must be traceable to the game data
            7. CRITICAL: Use exact timestamps and details from the data
            8. CRITICAL: If timing information is unclear, do not guess or assume
            9. CRITICAL: If an event did not explicitly happen, DO NOT include it in the timeline
            10. CRITICAL: Only include events that are clearly documented in the data
            11. CRITICAL: When in doubt about whether an event occurred, exclude it

            VALID EVENTS TO INCLUDE (only if explicitly supported by game data):
            - Goals scored (with player, time, team)
            - Cards shown (yellow/red cards with player, time, type)
            - Substitutions made (player in/out, time)
            - Penalties awarded or missed
            - Match start and end times
            - Halftime break
            - Any other significant match events with timestamps

            STRICTLY FORBIDDEN (DO NOT INCLUDE):
            - Any background or historical data about teams or players
            - Events not explicitly shown in the match data
            - Assumptions about event timing or sequence
            - External commentary or analysis
            - Events from other matches or seasons
            - Player or team statistics not from this match

            DATA VALIDATION REQUIREMENTS:
            - Verify that each event actually occurred in this match
            - Confirm that all timestamps match the game data exactly
            - Ensure that all player names and team names are accurate
            - Cross-reference event details with the provided data
            - Validate that the chronological order is correct
            - Verify that each player mentioned actually participated in the specific event described

            REQUIRED FORMAT:
            Output ONLY a JSON array of chronological event statements.
            Each statement should include the time and specific details from the data.
            No extra commentary, no markdown, no explanations.
            Example format: ["Event 1 with time", "Event 2 with time", "Event 3 with time"]

            Instructions:
            - Output only a JSON array of strings
            - No explanations, no markdown, no extra text
            - Each event must be from this match with accurate timing
            - If you cannot find clear events with timestamps, output fewer statements
            - Be extremely conservative - only include what is clearly stated in the data
            - Focus on actual events with timestamps, not interpretations
            - If timing data is insufficient, acknowledge the limitation rather than making assumptions
            - Only mention players with clear, verifiable actions in match events
            - EXCLUSION PRINCIPLE: If an event did not happen, DO NOT include it in the timeline
            - EXCLUSION PRINCIPLE: When uncertain, exclude rather than include
            - EXCLUSION PRINCIPLE: Only include events that are clearly documented in the data
            """
            result = await Runner.run(self.agent, prompt)
            try:
                timeline = json.loads(result.final_output)
                if isinstance(timeline, list):
                    return [str(t).strip() for t in timeline if t]
            except Exception:
                return [line.strip() for line in result.final_output.splitlines() if line.strip()]
        except Exception as e:
            logger.error(f"Error generating event timeline: {e}")
            return ["Event timeline based on available data"]

    async def get_stat_summary(self, stat_data: dict) -> list[str]:
        logger.info("Extracting statistical summary from match data")
        try:
            prompt = f"""
            You are summarizing statistical data from THIS SPECIFIC MATCH ONLY.

            STATISTICAL DATA (THIS MATCH ONLY - ALL INFORMATION MUST COME FROM HERE):
            {stat_data}

            ABSOLUTE RULES - YOU MUST FOLLOW THESE EXACTLY:
            1. ONLY use information that explicitly appears in the statistical data above
            2. ONLY summarize statistics from THIS specific match
            3. DO NOT make any assumptions, inferences, or interpretations beyond what is stated in the data
            4. DO NOT include any background or historical data
            5. DO NOT add any external knowledge or context
            6. CRITICAL: Every statistic must be traceable to the provided data
            7. CRITICAL: Use exact numbers and percentages from the data
            8. CRITICAL: If statistical information is unclear, do not guess or assume

            VALID STATISTICS TO INCLUDE (only if explicitly supported by data):
            - Possession percentages for each team
            - Shots on target and total shots
            - Yellow and red cards
            - Corner kicks
            - Fouls committed
            - Offsides
            - Passes completed and accuracy
            - Tackles and interceptions
            - Any other numerical match statistics

            STRICTLY FORBIDDEN (DO NOT INCLUDE):
            - Any background or historical data about teams or players
            - Statistics not explicitly shown in the match data
            - Assumptions about statistical significance
            - External commentary or analysis
            - Statistics from other matches or seasons
            - Player or team statistics not from this match

            DATA VALIDATION REQUIREMENTS:
            - Verify that each statistic actually comes from this match
            - Confirm that all numbers match the data exactly
            - Ensure that all team names are accurate
            - Cross-reference statistics with the provided data
            - Validate that percentages and totals are consistent

            REQUIRED FORMAT:
            Output ONLY a JSON array of statistical summary statements.
            Each statement should include specific numbers and details from the data.
            No extra commentary, no markdown, no explanations.
            Example format: ["Stat summary 1", "Stat summary 2", "Stat summary 3"]

            Instructions:
            - Output only a JSON array of strings
            - No explanations, no markdown, no extra text
            - Each statistic must be from this match with accurate numbers
            - If you cannot find clear statistics, output fewer statements
            - Be extremely conservative - only include what is clearly stated in the data
            - Focus on actual numbers and percentages, not interpretations
            - If statistical data is insufficient, acknowledge the limitation rather than making assumptions
            """
            result = await Runner.run(self.agent, prompt)
            try:
                stats = json.loads(result.final_output)
                if isinstance(stats, list):
                    return [str(s).strip() for s in stats if s]
            except Exception:
                return [line.strip() for line in result.final_output.splitlines() if line.strip()]
        except Exception as e:
            logger.error(f"Error extracting statistical summary: {e}")
            return ["Statistical summary based on available data"]

    async def get_best_and_worst_moments(self, game_data: dict) -> Dict[str, str]:
        logger.info("Finding best and worst moments in match")
        try:
            prompt = f"""
            You are identifying the best and worst moments from THIS SPECIFIC MATCH ONLY.

            GAME DATA (THIS MATCH ONLY - ALL INFORMATION MUST COME FROM HERE):
            {game_data}

            ABSOLUTE RULES - YOU MUST FOLLOW THESE EXACTLY:
            1. ONLY use information that explicitly appears in the game data above
            2. ONLY identify moments that actually occurred in THIS specific match
            3. DO NOT make any assumptions, inferences, or interpretations beyond what is stated in the data
            4. DO NOT include any background or historical data
            5. DO NOT add any external knowledge or context
            6. CRITICAL: Every moment must be traceable to the game data
            7. CRITICAL: Be extremely conservative - only mention what clearly happened in this match
            8. CRITICAL: If information is unclear or missing, do not speculate or assume
            9. CRITICAL: If a moment did not explicitly happen, DO NOT include it
            10. CRITICAL: Only include moments that are clearly documented in the data
            11. CRITICAL: When in doubt about whether a moment occurred, exclude it

            VALID MOMENTS TO IDENTIFY (only if explicitly supported by game data):
            - Best moment: The most decisive goal or action that determined the outcome
            - Worst moment: The most significant missed opportunity or mistake
            - Examples: decisive goals, missed penalties, own goals, red cards, etc.

            STRICTLY FORBIDDEN (DO NOT INCLUDE):
            - Any background or historical data about teams or players
            - Moments not explicitly shown in the match data
            - Assumptions about psychological impact or significance
            - External commentary or analysis
            - Moments from other matches or seasons
            - Player or team statistics not from this match

            DATA VALIDATION REQUIREMENTS:
            - Verify that each moment actually occurred in this match
            - Confirm that the details match the game data exactly
            - Ensure that all player names and team names are accurate
            - Cross-reference moment details with the provided data
            - Validate that the impact described is supported by the data
            - Verify that each player mentioned actually participated in the specific event described

            REQUIRED FORMAT:
            Output ONLY a JSON object with 'best_moment' and 'worst_moment' keys.
            Each value should be a clear, specific moment from this match.
            No extra commentary, no markdown, no explanations.
            Example format: {{"best_moment": "Specific moment 1", "worst_moment": "Specific moment 2"}}

            Instructions:
            - Output only a JSON object with the specified keys
            - No explanations, no markdown, no extra text
            - Each moment must be from this match with accurate details
            - If you cannot find clear moments, use "Unavailable" for that key
            - Be extremely conservative - only include what is clearly stated in the data
            - Focus on actual events with clear impact, not interpretations
            - If data is insufficient, acknowledge the limitation rather than making assumptions
            - Only mention players with clear, verifiable actions in match events
            - EXCLUSION PRINCIPLE: If a moment did not happen, DO NOT include it
            - EXCLUSION PRINCIPLE: When uncertain, exclude rather than include
            - EXCLUSION PRINCIPLE: Only include moments that are clearly documented in the data
            """
            result = await Runner.run(self.agent, prompt)
            try:
                moments = json.loads(result.final_output)
                if isinstance(moments, dict):
                    return {
                        "best_moment": moments.get("best_moment", "Unavailable"),
                        "worst_moment": moments.get("worst_moment", "Unavailable")
                    }
            except Exception:
                return {"best_moment": "Unavailable", "worst_moment": "Unavailable"}
        except Exception as e:
            logger.error(f"Error generating best/worst moments: {e}")
            return {"best_moment": "Unavailable", "worst_moment": "Unavailable"}

    async def get_missed_chances(self, game_data: dict) -> list[str]:
        logger.info("Identifying missed chances from match data")
        try:
            prompt = f"""
            You are identifying missed chances from THIS SPECIFIC MATCH ONLY.

            GAME DATA (THIS MATCH ONLY - ALL INFORMATION MUST COME FROM HERE):
            {game_data}

            ABSOLUTE RULES - YOU MUST FOLLOW THESE EXACTLY:
            1. ONLY use information that explicitly appears in the game data above
            2. ONLY identify missed chances that actually occurred in THIS specific match
            3. DO NOT make any assumptions, inferences, or interpretations beyond what is stated in the data
            4. DO NOT include any background or historical data
            5. DO NOT add any external knowledge or context
            6. CRITICAL: Every missed chance must be traceable to the game data
            7. CRITICAL: Be extremely conservative - only mention what clearly happened in this match
            8. CRITICAL: If information is unclear or missing, do not speculate or assume
            9. CRITICAL: If a missed chance did not explicitly happen, DO NOT include it
            10. CRITICAL: Only include missed chances that are clearly documented in the data
            11. CRITICAL: When in doubt about whether a missed chance occurred, exclude it

            VALID MISSED CHANCES TO IDENTIFY (only if explicitly supported by game data):
            - Missed penalties
            - Clear goal-scoring opportunities that were not converted
            - Near-miss shots that hit the post or crossbar
            - One-on-one chances that were not scored
            - Open goal opportunities that were missed
            - Any other significant missed opportunities with potential impact

            STRICTLY FORBIDDEN (DO NOT INCLUDE):
            - Any background or historical data about teams or players
            - Missed chances not explicitly shown in the match data
            - Assumptions about what might have happened
            - External commentary or analysis
            - Missed chances from other matches or seasons
            - Player or team statistics not from this match

            DATA VALIDATION REQUIREMENTS:
            - Verify that each missed chance actually occurred in this match
            - Confirm that the details match the game data exactly
            - Ensure that all player names and team names are accurate
            - Cross-reference missed chance details with the provided data
            - Validate that the potential impact described is supported by the data
            - Verify that each player mentioned actually participated in the specific event described

            REQUIRED FORMAT:
            Output ONLY a JSON array of missed chance statements.
            Each statement should describe a specific missed opportunity from this match.
            No extra commentary, no markdown, no explanations.
            Example format: ["Missed chance 1", "Missed chance 2", "Missed chance 3"]

            Instructions:
            - Output only a JSON array of strings
            - No explanations, no markdown, no extra text
            - Each missed chance must be from this match with accurate details
            - If you cannot find clear missed chances, output fewer statements
            - Be extremely conservative - only include what is clearly stated in the data
            - Focus on actual missed opportunities, not interpretations
            - If data is insufficient, acknowledge the limitation rather than making assumptions
            - Only mention players with clear, verifiable actions in match events
            - EXCLUSION PRINCIPLE: If a missed chance did not happen, DO NOT include it
            - EXCLUSION PRINCIPLE: When uncertain, exclude rather than include
            - EXCLUSION PRINCIPLE: Only include missed chances that are clearly documented in the data
            """
            result = await Runner.run(self.agent, prompt)
            try:
                chances = json.loads(result.final_output)
                if isinstance(chances, list):
                    return [str(c).strip() for c in chances if c]
            except Exception:
                return [line.strip() for line in result.final_output.splitlines() if line.strip()]
        except Exception as e:
            logger.error(f"Error identifying missed chances: {e}")
            return ["Missed chances based on available data"]

    async def get_formations_from_lineup_data(self, lineup_data: dict) -> list[str]:
        logger.info("Extracting team formations from lineup data")
        try:
            prompt = f"""
            You are identifying team formations from THIS SPECIFIC MATCH ONLY.

            LINEUP DATA (THIS MATCH ONLY - ALL INFORMATION MUST COME FROM HERE):
            {lineup_data}

            ABSOLUTE RULES - YOU MUST FOLLOW THESE EXACTLY:
            1. ONLY use information that explicitly appears in the lineup data above
            2. ONLY identify formations that were used in THIS specific match
            3. DO NOT make any assumptions, inferences, or interpretations beyond what is stated in the data
            4. DO NOT include any background or historical data
            5. DO NOT add any external knowledge or context
            6. CRITICAL: Every formation must be traceable to the lineup data
            7. CRITICAL: Be extremely conservative - only mention what clearly appears in the data
            8. CRITICAL: If formation information is unclear, do not guess or assume
            9. CRITICAL: If a formation is not clearly documented, DO NOT include it
            10. CRITICAL: Only include formations that are explicitly stated in the data
            11. CRITICAL: When in doubt about formation details, exclude rather than include

            VALID FORMATIONS TO IDENTIFY (only if explicitly supported by lineup data):
            - Starting formations for both teams (e.g., 4-3-3, 3-5-2, 4-4-2)
            - Formation changes during the match (if substitution data shows tactical changes)
            - Player positions and their arrangement
            - Any tactical setup information clearly stated in the data

            STRICTLY FORBIDDEN (DO NOT INCLUDE):
            - Any background or historical data about teams or players
            - Formations not explicitly shown in the lineup data
            - Assumptions about tactical preferences or playing styles
            - External commentary or analysis
            - Formations from other matches or seasons
            - Player or team statistics not from this match

            DATA VALIDATION REQUIREMENTS:
            - Verify that each formation actually comes from this match
            - Confirm that the formation details match the lineup data exactly
            - Ensure that all team names and player positions are accurate
            - Cross-reference formation details with the provided data
            - Validate that the tactical setup described is supported by the data

            REQUIRED FORMAT:
            Output ONLY a JSON array of formation statements.
            Each statement should describe a specific formation from this match.
            No extra commentary, no markdown, no explanations.
            Example format: ["Formation 1", "Formation 2", "Formation 3"]

            Instructions:
            - Output only a JSON array of strings
            - No explanations, no markdown, no extra text
            - Each formation must be from this match with accurate details
            - If you cannot find clear formations, output fewer statements
            - Be extremely conservative - only include what is clearly stated in the data
            - Focus on actual tactical setups, not interpretations
            - If formation data is insufficient, acknowledge the limitation rather than making assumptions
            - EXCLUSION PRINCIPLE: If a formation is not documented, DO NOT include it
            - EXCLUSION PRINCIPLE: When uncertain, exclude rather than include
            - EXCLUSION PRINCIPLE: Only include formations that are clearly documented in the data
            """
            result = await Runner.run(self.agent, prompt)
            try:
                formations = json.loads(result.final_output)
                if isinstance(formations, list):
                    return [str(f).strip() for f in formations if f]
            except Exception:
                return [line.strip() for line in result.final_output.splitlines() if line.strip()]
        except Exception as e:
            logger.error(f"Error identifying formations: {e}")
            return ["Formations based on available data"]
