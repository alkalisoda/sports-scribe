"""Data Collector Agent.

This agent is responsible for gathering game data from various sports APIs.
It collects real-time and historical sports data to feed into the content generation pipeline.
"""

import logging
from typing import Any, List, Dict
import aiohttp
import os

from utils.security import sanitize_log_input

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

class DataCollectorAgent:
    """Agent responsible for collecting sports data from various APIs and data sources."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the Data Collector Agent with configuration."""
        self.config = config
        self.api_key = config.get("rapidapi_key") or os.getenv("RAPIDAPI_KEY")
        self.base_url = "https://api-football-v1.p.rapidapi.com/v3"
        self.headers = {
            "X-RapidAPI-Key": self.api_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }
        logger.info("Data Collector Agent initialized")

    async def _make_api_request(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make a request to the API-Football API.
        
        Args:
            endpoint: API endpoint (e.g., "/fixtures", "/teams")
            params: Query parameters
            
        Returns:
            Standardized API response structure
        """
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}{endpoint}"
                async with session.get(url, headers=self.headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "get": endpoint,
                            "parameters": params or {},
                            "errors": [],
                            "results": data.get("results", 0),
                            "paging": data.get("paging", {}),
                            "response": data.get("response", [])
                        }
                    else:
                        logger.error(f"API request failed: {response.status}")
                        return {
                            "get": endpoint,
                            "parameters": params or {},
                            "errors": [f"HTTP {response.status}"],
                            "results": 0,
                            "paging": {},
                            "response": []
                        }
        except Exception as e:
            logger.error(f"API request error: {str(e)}")
            return {
                "get": endpoint,
                "parameters": params or {},
                "errors": [str(e)],
                "results": 0,
                "paging": {},
                "response": []
            }

    async def collect_game_data(self, game_id: str) -> Dict[str, Any]:
        """Collect comprehensive data for a specific game.

        Args:
            game_id: Unique identifier for the game (fixture ID)

        Returns:
            Dictionary containing game data in standardized format
        """
        logger.info("Collecting data for game: %s", sanitize_log_input(game_id))
        
        # Collect fixture data
        fixture_data = await self._make_api_request("/fixtures", {"id": game_id})
        
        # Collect events for the game
        events_data = await self._make_api_request("/fixtures/events", {"fixture": game_id})
        
        # Collect lineups for the game
        lineups_data = await self._make_api_request("/fixtures/lineups", {"fixture": game_id})
        
        # Collect statistics for the game
        stats_data = await self._make_api_request("/fixtures/statistics", {"fixture": game_id})
        
        return {
            "get": "game_data",
            "parameters": {"game_id": game_id},
            "errors": [],
            "results": 1,
            "paging": {},
            "response": [
                {
                    "fixture": fixture_data,
                    "events": events_data,
                    "lineups": lineups_data,
                    "statistics": stats_data
                }
            ]
        }

    async def collect_team_data(self, team_id: str) -> Dict[str, Any]:
        """Collect team statistics and information.

        Args:
            team_id: Unique identifier for the team

        Returns:
            Dictionary containing team data in standardized format
        """
        logger.info("Collecting data for team: %s", sanitize_log_input(team_id))
        
        # Collect team information
        team_info = await self._make_api_request("/teams", {"id": team_id})
        
        # Collect team statistics for current season
        team_stats = await self._make_api_request("/teams/statistics", {
            "team": team_id,
            "league": self.config.get("default_league", "39"),  # Premier League default
            "season": self.config.get("default_season", "2024")
        })
        
        # Collect team fixtures
        team_fixtures = await self._make_api_request("/fixtures", {
            "team": team_id,
            "season": self.config.get("default_season", "2024")
        })
        
        return {
            "get": "team_data",
            "parameters": {"team_id": team_id},
            "errors": [],
            "results": 1,
            "paging": {},
            "response": [
                {
                    "team_info": team_info,
                    "team_stats": team_stats,
                    "team_fixtures": team_fixtures
                }
            ]
        }

    async def collect_player_data(self, player_id: str) -> Dict[str, Any]:
        """Collect player statistics and information.

        Args:
            player_id: Unique identifier for the player

        Returns:
            Dictionary containing player data in standardized format
        """
        logger.info("Collecting data for player: %s", sanitize_log_input(player_id))
        
        # Collect player information
        player_info = await self._make_api_request("/players", {"id": player_id})
        
        # Collect player statistics for current season
        player_stats = await self._make_api_request("/players", {
            "id": player_id,
            "season": self.config.get("default_season", "2024")
        })
        
        # Collect player transfers
        player_transfers = await self._make_api_request("/transfers", {"player": player_id})
        
        return {
            "get": "player_data",
            "parameters": {"player_id": player_id},
            "errors": [],
            "results": 1,
            "paging": {},
            "response": [
                {
                    "player_info": player_info,
                    "player_stats": player_stats,
                    "player_transfers": player_transfers
                }
            ]
        }

    async def collect_league_data(self, league_id: str, season: str = None) -> Dict[str, Any]:
        """Collect league standings and information.

        Args:
            league_id: Unique identifier for the league
            season: Season year (defaults to config default)

        Returns:
            Dictionary containing league data in standardized format
        """
        season = season or self.config.get("default_season", "2024")
        logger.info("Collecting data for league: %s, season: %s", 
                   sanitize_log_input(league_id), sanitize_log_input(season))
        
        # Collect league standings
        standings = await self._make_api_request("/standings", {
            "league": league_id,
            "season": season
        })
        
        # Collect league fixtures
        fixtures = await self._make_api_request("/fixtures", {
            "league": league_id,
            "season": season
        })
        
        # Collect top scorers
        top_scorers = await self._make_api_request("/players/topscorers", {
            "league": league_id,
            "season": season
        })
        
        return {
            "get": "league_data",
            "parameters": {"league_id": league_id, "season": season},
            "errors": [],
            "results": 1,
            "paging": {},
            "response": [
                {
                    "standings": standings,
                    "fixtures": fixtures,
                    "top_scorers": top_scorers
                }
            ]
        }
