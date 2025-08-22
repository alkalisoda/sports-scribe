"""Research Agent.

This agent provides contextual background and analysis for sports articles.
It researches historical data, team/player statistics, and relevant context
to enrich the content generation process.
"""

import json
import logging
from typing import Any

from agents import Agent, Runner
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class ResearchAgent:
    """Agent responsible for researching contextual information and analysis."""

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize the Research Agent with configuration."""
        self.config = config or {}

        # Initialize the research agent without web search capability
        self.agent = Agent(
            instructions="""You are a sports research agent. Provide clear, factual analysis based ONLY on provided data.

            CORE PRINCIPLES:
            - ONLY use information explicitly provided in the data
            - When in doubt, exclude rather than include
            - Clearly distinguish between THIS MATCH events and background information

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

            Always return clear, structured analysis based solely on the provided data.""",
            name="ResearchAgent",
            output_type=str,
            model=self.config.get("model", "gpt-4.1-nano"),
        )

        logger.info("Research Agent initialized successfully")

    async def get_storyline_from_game_data(self, game_data: dict) -> list[str]:
        """Get comprehensive storylines from game data by analyzing different components separately.

        Args:
            game_data: Compact game data from pipeline (contains match_info, events, players, statistics, lineups)

        Returns:
            list[str]: Comprehensive list of storylines including analysis
        """
        logger.info(
            "Generating comprehensive storylines from compact game data by analyzing components separately"
        )

        try:
            # Extract different components from compact data
            match_info = game_data.get("match_info", {})
            events = game_data.get("events", [])
            players = game_data.get("players", [])
            statistics = game_data.get("statistics", [])
            lineups = game_data.get("lineups", [])

            all_storylines = []

            # 1. Analyze match information (basic game context)
            if match_info:
                logger.info("Analyzing match information...")
                match_storylines = await self._analyze_match_info(match_info)
                all_storylines.extend(match_storylines)

            # 2. Analyze key events (goals, cards, substitutions)
            if events:
                logger.info("Analyzing key events...")
                event_storylines = await self._analyze_events(events)
                all_storylines.extend(event_storylines)

            # 3. Analyze player performances (focus on high-rated players)
            if players:
                logger.info("Analyzing player performances...")
                player_storylines = await self._analyze_player_performances(players)
                all_storylines.extend(player_storylines)

            # 4. Analyze team statistics
            if statistics:
                logger.info("Analyzing team statistics...")
                stats_storylines = await self._analyze_team_statistics(statistics)
                all_storylines.extend(stats_storylines)

            # 5. Analyze lineups and formations
            if lineups:
                logger.info("Analyzing lineups and formations...")
                lineup_storylines = await self._analyze_lineups(lineups)
                all_storylines.extend(lineup_storylines)

            logger.info(
                f"Generated {len(all_storylines)} storylines from separate component analysis"
            )
            return all_storylines

        except Exception as e:
            logger.error(
                f"Error generating comprehensive storylines from game data: {e}"
            )
            return [
                "Comprehensive match analysis based on available game data",
                "Key moments and turning points from the match",
            ]

    async def _analyze_match_info(self, match_info: dict) -> list[str]:
        """Analyze basic match information."""
        try:
            match_info_str = str(match_info)
            prompt = f"""
            Analyze basic match information for storylines.

            MATCH INFO:
            {match_info_str}

            RULES:
            - Focus on match context, teams, venue, league, and final score
            - Use exact team names, venue, and league information
            - Describe the match result clearly
            - NO historical data or assumptions

            OUTPUT FORMAT: Return ONLY a JSON array of simple strings.
            Example: ["Team A defeated Team B 1-0 at Venue X", "The match was the opening/mid-season/closing fixture of the 2024 Premier League season"]
            """

            result = await Runner.run(self.agent, prompt)
            try:
                storylines = json.loads(result.final_output)
                if isinstance(storylines, list):
                    # Handle both string and dict formats
                    processed_storylines = []
                    for s in storylines:
                        if isinstance(s, str):
                            processed_storylines.append(s.strip())
                        elif isinstance(s, dict):
                            # Extract storyline from dict if present
                            if "storyline" in s:
                                processed_storylines.append(str(s["storyline"]).strip())
                            elif "details" in s:
                                processed_storylines.append(str(s["details"]).strip())
                            else:
                                processed_storylines.append(str(s).strip())
                    return processed_storylines
            except Exception:
                return [
                    line.strip()
                    for line in result.final_output.splitlines()
                    if line.strip()
                ]

        except Exception as e:
            logger.error(f"Error analyzing match info: {e}")
            return []

    async def _analyze_events(self, events: list) -> list[str]:
        """Analyze key events (goals, cards, substitutions)."""
        try:
            events_str = str(events)
            prompt = f"""
            Analyze key match events for storylines.

            EVENTS:
            {events_str}

            EVENT-PLAYER CORRESPONDENCE RULES:
            - Each event must contain its own player and time data - DO NOT mix between events
            - Goal event player = only the player listed in that Goal event
            - Card event player = only the player listed in that Card event
            - Substitution event players = only the players listed in that Substitution event
            - Goal time cannot be used as substitution time
            - Card time cannot be used as goal time

            GOAL & ASSIST VALIDATION RULES:
            - Only describe goals from "Goal" events (type="Goal")
            - "player" = who scored, "assist" = who assisted
            - NEVER attribute a goal to a player who only assisted
            - NEVER attribute an assist to a player who only scored

            GOAL COUNT VALIDATION RULES:
            - Use only "Goal" events (type == "Goal") to determine how many goals each player scored.
            - If a player appears only ONCE as the scorer, do NOT say “scored again”, “second goal”, “brace”, “double”, etc.
            - These terms may ONLY be used if the same player appears MULTIPLE times as scorer.
            - If the player scored once, use phrases like “scored a goal” or “found the net”.
            - NEVER assume a player scored more than once unless it's explicitly recorded.

            SUBSTITUTION IDENTITY LOGIC:
            - In substitution events: "in" = player being substituted ON, "out" = player being substituted OFF
            - Only call a player "substituted in" if they appear as the "in" field in a substitution event
            - Only call a player "substituted out" if they appear as the "out" field in the same event
            - Use clear language: "Player X was substituted in, replacing Player Y" or "Player Y was replaced by Player X"
            - Never reverse the order of the players in the substitution event.

            TEAM VERIFICATION FOR EVENTS:
            - Each event (goal, card, substitution) contains a "team" field indicating which team made the event
            - All involved players ("in", "out", "player", "assist") MUST belong to the same team as specified in the "team" field
            - DO NOT list players under the wrong team
            - DO NOT describe players from the opposing team as involved in the current team's event
            - Mention the team name in the storylines
            Example: If team = "Southampton", then both "player" and "assist" must be Southampton players

            VAR EVENTS:
            - If an event has `type = Var` and `detail = Goal cancelled`, do NOT assume the `player` listed scored the goal unless there is a separate `goal` event with the same player.
            - A VAR event involving a player only means the player was affected by or related to the decision — not necessarily the scorer.
            - Only describe a player as scoring a goal if there is an explicit `event_type = goal` with `scorer = player`.
            - Use safe phrasing like "A goal was cancelled by VAR involving [player]" if no scorer is confirmed.

            GOAL TIMING LOGIC:
            - Do NOT describe a goal as "early lead" unless it happens in first half (≤ 45 minutes)
            - If goal occurs after 75th minute, describe as "late winner" or "decisive goal"

            OUTPUT FORMAT: Return ONLY a JSON array of simple strings.
            Example: ["Player A scored the winning goal in the nth minute", "Player B was substituted in at n minutes, replacing Player C", "VAR cancelled a potential goal of Team A for offside, involving Player D", "Half time was reached"]

            SUBSTITUTION IMPACT RULES:
            - When analyzing substitutions, evaluate their impact based on subsequent events.
            - If a substituted-in player scored a goal, made an assist, or received a card, describe the substitution as impactful.
            - Highlight linkages: e.g., "Substitute Player A scored the winner after coming on in the nth minute after replacing Player B"
            - If a substitution was followed by no key contribution or came in very late, it should be noted as such.
            - Do not describe substitutions as meaningful unless supported by data (e.g., goal, assist, card).
            """

            result = await Runner.run(self.agent, prompt)
            try:
                storylines = json.loads(result.final_output)
                if isinstance(storylines, list):
                    # Handle both string and dict formats
                    processed_storylines = []
                    for s in storylines:
                        if isinstance(s, str):
                            processed_storylines.append(s.strip())
                        elif isinstance(s, dict):
                            # Extract storyline from dict if present
                            if "storyline" in s:
                                processed_storylines.append(str(s["storyline"]).strip())
                            elif "details" in s:
                                processed_storylines.append(str(s["details"]).strip())
                            else:
                                processed_storylines.append(str(s).strip())
                    return processed_storylines
            except Exception:
                return [
                    line.strip()
                    for line in result.final_output.splitlines()
                    if line.strip()
                ]

        except Exception as e:
            logger.error(f"Error analyzing events: {e}")
            return []

    async def _analyze_player_performances(self, players: list) -> list[str]:
        """Analyze individual player performances (focus on high-rated players)."""
        try:
            players_str = str(players)
            prompt = f"""
            Analyze individual player performances for storylines.

            PLAYERS:
            {players_str}

            STATISTICS VALIDATION RULES:
            - Only use statistics explicitly provided in the data
            - Distinguish between individual player stats and team stats
            - Verify exact numbers from source data - DO NOT approximate or round
            - Individual stats (e.g., "player won 10/14 duels") ≠ Team stats

            PLAYER STATISTICS STORYLINE RULES:
            - Use player statistics and match contribution to determine inclusion
            - DO NOT rely solely on rating for filtering
            - Describe any player who showed meaningful involvement, such as:
              - Playing 60+ minutes with ≥ 80% pass accuracy or ≥ 35+ total passes
              - ≥ 2 tackles, interceptions, or clearances
              - ≥ 4 duels won
              - ≥ 1 goal or assist
            - You may still mention high-rated players (rating ≥ 7.0), but it is not mandatory
            - DO NOT describe players who had zero minutes or no stats
            - DO NOT include yellow or red cards in player performance. Only analyze goals, assists, passes, tackles, duels, etc.
            - In substitution events: "in" = player being substituted ON, "out" = player being substituted OFF
            - For VAR or canceled goals, do NOT assume the player scored unless explicitly stated; only mention the player's involvement and the event. Example: "A goal initially scored by Player A was canceled by VAR at the nth minute." or "A goal was canceled by VAR involving Player A."

            GOAL COUNT VALIDATION (MANDATORY):
            - If a player is described as having scored "a brace", "twice", "two goals", or "a second goal", you MUST verify that the player appears more than once as a scorer in the 'events' section where type == "Goal".
            - If the player appears only once, this is a factual error.
            - Correct any instance of "brace" or "second goal" to reflect the accurate number of goals scored.
            - DO NOT rely on `player_performance` or inferred phrasing. Use `goal` events only.

            OUTPUT FORMAT: Return ONLY a JSON array of simple strings, each describing the player's own actions and involvement, with no ambiguity.
            Example: ["Player A was substituted in for Player B at the nth minute.", "A potential goal was canceled by VAR at the nth minute, involving Player C."]
            """

            result = await Runner.run(self.agent, prompt)
            try:
                storylines = json.loads(result.final_output)
                if isinstance(storylines, list):
                    # Handle both string and dict formats
                    processed_storylines = []
                    for s in storylines:
                        if isinstance(s, str):
                            processed_storylines.append(s.strip())
                        elif isinstance(s, dict):
                            # Extract storyline from dict if present
                            if "storyline" in s:
                                processed_storylines.append(str(s["storyline"]).strip())
                            elif "details" in s:
                                processed_storylines.append(str(s["details"]).strip())
                            else:
                                processed_storylines.append(str(s).strip())
                    return processed_storylines
            except Exception:
                return [
                    line.strip()
                    for line in result.final_output.splitlines()
                    if line.strip()
                ]

        except Exception as e:
            logger.error(f"Error analyzing player performances: {e}")
            return []

    async def _analyze_player_events(self, events: list) -> list[str]:
        """Analyze player events (goals, assists, cards, substitutions)."""
        try:
            events_str = str(events)
            prompt = f"""
            Analyze player events for performance storylines.

            EVENTS:
            {events_str}

            EVENT-PLAYER CORRESPONDENCE RULES:
            - Each event must contain its own player and time data - DO NOT mix between events
            - Goal event player = only the player listed in that Goal event
            - Card event player = only the player listed in that Card event
            - Substitution event players = only the players listed in that Substitution event

            GOAL & ASSIST VALIDATION RULES:
            - Only describe goals from "Goal" events (type="Goal")
            - "player" = who scored, "assist" = who assisted
            - NEVER attribute a goal to a player who only assisted
            - NEVER attribute an assist to a player who only scored

            GOAL COUNT VALIDATION RULES:
            - Use only "Goal" events (type == "Goal") to determine how many goals each player scored.
            - If a player appears only ONCE as the scorer, do NOT say “scored again”, “second goal”, “brace”, “double”, etc.
            - These terms may ONLY be used if the same player appears MULTIPLE times as scorer.
            - If the player scored once, use phrases like “scored a goal” or “found the net”.
            - NEVER assume a player scored more than once unless it's explicitly recorded.

            SUBSTITUTION IDENTITY RULE:
            - In substitution events: "in" = player being substituted ON, "out" = player being substituted OFF
            - Only call a player "substituted in" if they appear as the "in" field in a substitution event
            - Only call a player "substituted out" if they appear as the "out" field in the same event
            - Use clear language: "Player X was substituted in, replacing Player Y"
            - The structure is now unambiguous: "in" = coming on, "out" = going off
            - Don't use the same player for both "in" and "out" in the same substitution event
            - Don't use "assist" for substitution events, use "replace" instead

            ASSIST VALIDATION RULE:
            - Only mention an assist if the player is listed as "assist" in a Goal event

            CARD VALIDATION RULES:
            - Only describe cards shown in "Card" events (type="Card")
            - Card time must come from Card event time, not other events
            - DO NOT include yellow or red cards in player performance. Only analyze goals, assists, passes, tackles, duels, etc.

            CONTRIBUTION FILTERING RULE:
            - Only include players who made notable contributions
            - Focus on players with goals, assists, or substitutions
            - Only mention cards if they lead to red cards or cause significant incidents
            - Avoid listing players with no meaningful involvement
            - DO NOT duplicate information that appears in game_analysis

            OUTPUT FORMAT: Return ONLY a JSON array of simple strings.
            Example: ["J. Zirkzee scored the winning goal in the 87th minute", "A. Diallo was substituted in at 61 minutes, replacing A. Garnacho"]

            SUBSTITUTION IMPACT RULES:
            - When analyzing substitutions, evaluate their impact based on subsequent events.
            - If a substituted-in player scored a goal, made an replacement, or received a card, describe the substitution as impactful.
            - Highlight linkages: e.g., "Substitute J. Zirkzee scored the winner after coming on in the 61st minute after replacing M. Mount"
            - If a substitution was followed by no key contribution or came in very late, it should be noted as such.
            - Do not describe substitutions as meaningful unless supported by data (e.g., goal, assist, card).
            - DO NOT infer substitution time from goal/card event.
            - Example (valid): "Player A, who came on in the 46th minute, was booked in the 90th minute"
            """

            result = await Runner.run(self.agent, prompt)
            try:
                storylines = json.loads(result.final_output)
                if isinstance(storylines, list):
                    # Handle both string and dict formats
                    processed_storylines = []
                    for s in storylines:
                        if isinstance(s, str):
                            processed_storylines.append(s.strip())
                        elif isinstance(s, dict):
                            # Extract storyline from dict if present
                            if "storyline" in s:
                                processed_storylines.append(str(s["storyline"]).strip())
                            elif "details" in s:
                                processed_storylines.append(str(s["details"]).strip())
                            else:
                                processed_storylines.append(str(s).strip())
                    return processed_storylines
            except Exception:
                return [
                    line.strip()
                    for line in result.final_output.splitlines()
                    if line.strip()
                ]

        except Exception as e:
            logger.error(f"Error analyzing player events: {e}")
            return []

    async def _analyze_player_statistics(self, players: list) -> list[str]:
        """Analyze player statistics for performance storylines (focus on high-rated players)."""
        try:
            players_str = str(players)
            prompt = f"""
            Analyze player statistics for performance storylines.

            PLAYERS:
            {players_str}

            STATISTICS VALIDATION RULES:
            - Only use statistics explicitly provided in the data
            - Distinguish between individual player stats and team stats
            - Verify exact numbers from source data - DO NOT approximate or round
            - Individual stats (e.g., "player won 10/14 duels") ≠ Team stats

            PLAYER STATISTICS STORYLINE RULES:
            - Use player statistics and match contribution to determine inclusion
            - DO NOT rely solely on rating for filtering
            - Describe any player who showed meaningful involvement, such as:
              - Playing 60+ minutes with ≥ 80% pass accuracy or ≥ 35+ total passes
              - ≥ 2 tackles, interceptions, or clearances
              - ≥ 4 duels won
              - ≥ 1 goal or assist
            - You may still mention high-rated players (rating ≥ 7.0), but it is not mandatory
            - DO NOT describe players who had zero minutes or no stats

            OUTPUT FORMAT: Return ONLY a JSON array of simple strings.
            Example: ["Casemiro completed 53 passes with 43% accuracy in 90 minutes", "Player X made 4 tackles and won 7 out of 13 duels"]
            """

            result = await Runner.run(self.agent, prompt)
            try:
                storylines = json.loads(result.final_output)
                if isinstance(storylines, list):
                    # Handle both string and dict formats
                    processed_storylines = []
                    for s in storylines:
                        if isinstance(s, str):
                            processed_storylines.append(s.strip())
                        elif isinstance(s, dict):
                            # Extract storyline from dict if present
                            if "storyline" in s:
                                processed_storylines.append(str(s["storyline"]).strip())
                            elif "details" in s:
                                processed_storylines.append(str(s["details"]).strip())
                            else:
                                processed_storylines.append(str(s).strip())
                    return processed_storylines
            except Exception:
                return [
                    line.strip()
                    for line in result.final_output.splitlines()
                    if line.strip()
                ]

        except Exception as e:
            logger.error(f"Error analyzing player statistics: {e}")
            return []

    async def _analyze_team_statistics(self, statistics: list) -> list[str]:
        """Analyze team statistics."""
        try:
            statistics_str = str(statistics)
            prompt = f"""
            Analyze team statistics for storylines.

            STATISTICS:
            {statistics_str}

            TEAM-LEVEL STATS RULES:
            - Only use team-wide statistics from the "statistics" section
            - Compare statistics between teams
            - Focus on key metrics like possession, shots, corners, fouls

            - Include detailed shooting breakdown:
                - "Shots insidebox"
                - "Shots outsidebox"
                - "Blocked shots"
            - Always quote the exact number from the statistics data
            - Never assume or simplify; do not equate “shots on target” with “inside the box”

            OUTPUT FORMAT: Return ONLY a JSON array of simple strings.
            Example: ["Manchester United dominated possession with 55% compared to Fulham's 45%", "Both teams received 3 yellow cards each"]
            """

            result = await Runner.run(self.agent, prompt)
            try:
                storylines = json.loads(result.final_output)
                if isinstance(storylines, list):
                    # Handle both string and dict formats
                    processed_storylines = []
                    for s in storylines:
                        if isinstance(s, str):
                            processed_storylines.append(s.strip())
                        elif isinstance(s, dict):
                            # Extract storyline from dict if present
                            if "storyline" in s:
                                processed_storylines.append(str(s["storyline"]).strip())
                            elif "details" in s:
                                processed_storylines.append(str(s["details"]).strip())
                            else:
                                processed_storylines.append(str(s).strip())
                    return processed_storylines
            except Exception:
                return [
                    line.strip()
                    for line in result.final_output.splitlines()
                    if line.strip()
                ]

        except Exception as e:
            logger.error(f"Error analyzing team statistics: {e}")
            return []

    async def _analyze_lineups(self, lineups: list) -> list[str]:
        """Analyze lineups and formations."""
        try:
            lineups_str = str(lineups)
            prompt = f"""
            Analyze lineups and formations for storylines.

            LINEUPS:
            {lineups_str}

            RULES:
            - Focus on formations, key players, and tactical setup
            - Use exact formation information
            - Mention notable players in starting XI
            - NO assumptions about player performance

            OUTPUT FORMAT: Return ONLY a JSON array of simple strings.
            Example: ["Both teams employed a 4-2-3-1 formation", "Manchester United's starting XI featured key players like Bruno Fernandes"]
            """

            result = await Runner.run(self.agent, prompt)
            try:
                storylines = json.loads(result.final_output)
                if isinstance(storylines, list):
                    # Handle both string and dict formats
                    processed_storylines = []
                    for s in storylines:
                        if isinstance(s, str):
                            processed_storylines.append(s.strip())
                        elif isinstance(s, dict):
                            # Extract storyline from dict if present
                            if "storyline" in s:
                                processed_storylines.append(str(s["storyline"]).strip())
                            elif "details" in s:
                                processed_storylines.append(str(s["details"]).strip())
                            else:
                                processed_storylines.append(str(s).strip())
                    return processed_storylines
            except Exception:
                return [
                    line.strip()
                    for line in result.final_output.splitlines()
                    if line.strip()
                ]

        except Exception as e:
            logger.error(f"Error analyzing lineups: {e}")
            return []

    async def get_history_from_team_data(self, team_data: dict) -> list[str]:
        """Get historical context from team data ONLY (background information).

        Args:
            team_data: Team information including enhanced data (background/historical only)

        Returns:
            list[str]: Historical context and background information
        """
        logger.info(
            "Analyzing historical context from team data (background information only)"
        )

        try:
            team_data_str = str(team_data)
            prompt = f"""
            Analyze BACKGROUND information about teams.

            TEAM DATA:
            {team_data_str}

            RULES:
            - Use only background/historical information
            - Do NOT mention current match events
            - Only include facts explicitly in the data

            OUTPUT: JSON array of 3-5 background statements.
            """

            result = await Runner.run(self.agent, prompt)
            try:
                storylines = json.loads(result.final_output)
                if isinstance(storylines, list):
                    return [str(s).strip() for s in storylines if s]
            except Exception:
                return [
                    line.strip()
                    for line in result.final_output.splitlines()
                    if line.strip()
                ]

        except Exception as e:
            logger.error(f"Error analyzing historical context: {e}")
            return [
                "Historical context based on available team data",
                "Team performance analysis from provided data",
            ]

    async def get_performance_from_player_game_data(
        self, player_data: dict, game_data: dict
    ) -> list[str]:
        """Analyze individual player performance from game data by analyzing components separately.

        Args:
            player_data: Player information including enhanced data
            game_data: Compact game data for context (current match events only)

        Returns:
            list[str]: Player performance analysis based ONLY on current match events
        """
        logger.info(
            "Analyzing individual player performance from compact game data by analyzing components separately"
        )

        try:
            all_storylines = []

            # Extract different components from compact data
            events = game_data.get("events", [])
            players = game_data.get("players", [])

            # 1. Analyze player events (goals, assists, cards, substitutions)
            if events:
                logger.info("Analyzing player events...")
                event_storylines = await self._analyze_player_events(events)
                all_storylines.extend(event_storylines)

            # 2. Analyze player statistics (focus on high-rated players)
            if players:
                logger.info("Analyzing player statistics...")
                stats_storylines = await self._analyze_player_statistics(players)
                all_storylines.extend(stats_storylines)

            logger.info(
                f"Generated {len(all_storylines)} player performance storylines from separate component analysis"
            )
            return all_storylines

        except Exception as e:
            logger.error(f"Error analyzing player performance: {e}")
            return [
                "Player performance analysis based on available data",
                "Individual contributions from the match data",
            ]
