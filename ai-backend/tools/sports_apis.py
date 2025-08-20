"""
Sports APIs Module

This module provides interface for API-Football from RapidAPI.
Focus: Football (Soccer) only for MVP.
"""

import logging
import os
from typing import Any

import aiohttp

from utils.security import sanitize_log_input, sanitize_multiple_log_inputs

logger = logging.getLogger(__name__)


class APIFootballClient:
    """
    Client for API-Football integration.

    Documentation: https://api-sports.io/sports/football
    Focus: Football (Soccer) data only for MVP
    """

    def __init__(self,
                 api_key: str | None = None,
                 api_key_header: str | None = None,
                 base_url: str | None = None):
        self.api_key = api_key or os.getenv("API_FOOTBALL_KEY")
        self.api_key_header = api_key_header or os.getenv("API_FOOTBALL_KEY_HEADER")
        self.base_url = base_url or os.getenv("API_FOOTBALL_BASE_URL")
        self.headers = {
            self.api_key_header: self.api_key,
            "accept": "application/json"
        }
        self.session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> "APIFootballClient":
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        if self.session:
            await self.session.close()

    async def get_fixtures(
        self,
        league_id: int | None = None,
        season: int | None = None,
        date: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get football fixtures/matches.

        Args:
            league_id: League ID (e.g., 39 for Premier League)
            season: Season year (e.g., 2024)
            date: Date in YYYY-MM-DD format

        Returns:
            List of fixture data dictionaries
        """
        # TODO: Implement API-Football fixtures endpoint
        league_safe, season_safe = sanitize_multiple_log_inputs(league_id, season)
        logger.info(
            "Fetching fixtures for league %s, season %s", league_safe, season_safe
        )

        headers = {
            self.api_key_header: self.api_key,
            "accept": "application/json"
        }

        params = {}
        if league_safe:
            params["league"] = league_safe
        if season_safe:
            params["season"] = season_safe
        if date:
            params["date"] = date  # Format: YYYY-MM-DD

        url = f"{self.base_url}/fixtures"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, params=params) as response:
                    logger.debug("Status code: %s", response.status)
                    logger.debug("Rate limit remaining: %s/%s",
                             response.headers.get("x-ratelimit-requests-remaining"),
                             response.headers.get("x-ratelimit-requests-limit"))

                    response.raise_for_status()
                    data = await response.json()
                    logger.info("Fixtures fetched successfully!")
                    #print(f"{data}")  # Optional: for live debugging
                    return data.get("response", [])
        except aiohttp.ClientResponseError as http_err:
            logger.error("HTTP error occurred: %s", http_err)
        except Exception as err:
            logger.error("Unexpected error: %s", err)

        return []

    async def get_team(self, team_id: int) -> list[dict[str, Any]]:
        """
        Get teams in a league for a season.

        Args:
            league_id: League ID
            season: Season year

        Returns:
            List of team data dictionaries
        """
        # TODO: Implement API-Football teams endpoint
        team_safe = sanitize_multiple_log_inputs(team_id)
        logger.info("Fetching teams for league %s, season %s", team_safe)

        headers = {
            self.api_key_header: self.api_key,
            "accept": "application/json"
        }

        params = {
            "id": team_safe
        }

        url = f"{self.base_url}/teams"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, params=params) as response:
                    logger.debug("Status code: %s", response.status)
                    logger.debug("Rate limit remaining: %s/%s",
                                 response.headers.get("x-ratelimit-requests-remaining"),
                                 response.headers.get("x-ratelimit-requests-limit"))

                    response.raise_for_status()
                    data = await response.json()
                    logger.info("Teams fetched successfully!")
                    #print(f"{data}")  # Optional for debugging
                    return data.get("response", [])
        except aiohttp.ClientResponseError as http_err:
            logger.error("HTTP error occurred: %s", http_err)
        except Exception as err:
            logger.error("Unexpected error: %s", err)

        return []

    async def get_teams(self, league_id: int, season: int) -> list[dict[str, Any]]:
        """
        Get teams in a league for a season.

        Args:
            league_id: League ID
            season: Season year

        Returns:
            List of team data dictionaries
        """
        # TODO: Implement API-Football teams endpoint
        league_safe, season_safe = sanitize_multiple_log_inputs(league_id, season)
        logger.info("Fetching teams for league %s, season %s", league_safe, season_safe)

        headers = {
            self.api_key_header: self.api_key,
            "accept": "application/json"
        }

        params = {
            "league": league_safe,
            "season": season_safe
        }

        url = f"{self.base_url}/teams"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, params=params) as response:
                    logger.debug("Status code: %s", response.status)
                    logger.debug("Rate limit remaining: %s/%s",
                                 response.headers.get("x-ratelimit-requests-remaining"),
                                 response.headers.get("x-ratelimit-requests-limit"))

                    response.raise_for_status()
                    data = await response.json()
                    logger.info("Teams fetched successfully!")
                    #print(f"{data}")  # Optional for debugging
                    return data.get("response", [])
        except aiohttp.ClientResponseError as http_err:
            logger.error("HTTP error occurred: %s", http_err)
        except Exception as err:
            logger.error("Unexpected error: %s", err)

        return []

    async def get_league_standings(self, league_id: int, season: int) -> dict[str, Any]:
        """
        Get league standings/table.

        Args:
            league_id: League ID
            season: Season year

        Returns:
            Dictionary containing league standings
        """
        # TODO: Implement API-Football standings endpoint
        league_safe, season_safe = sanitize_multiple_log_inputs(league_id, season)
        logger.info(
            "Fetching standings for league %s, season %s", league_safe, season_safe
        )

        headers = {
            self.api_key_header: self.api_key,
            "accept": "application/json"
        }

        params = {
            "league": league_safe,
            "season": season_safe
        }

        url = f"{self.base_url}/standings"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, params=params) as response:
                    logger.debug("Status code: %s", response.status)
                    logger.debug("Rate limit remaining: %s/%s",
                                 response.headers.get("x-ratelimit-requests-remaining"),
                                 response.headers.get("x-ratelimit-requests-limit"))

                    response.raise_for_status()
                    data = await response.json()
                    logger.info("Standings fetched successfully!")
                    #print(f"{data}")  # Optional for debugging
                    return data.get("response", {})
        except aiohttp.ClientResponseError as http_err:
            logger.error("HTTP error occurred: %s", http_err)
        except Exception as err:
            logger.error("Unexpected error: %s", err)

        return {}

    async def get_match_statistics(self, fixture_id: int) -> dict[str, Any]:
        """
        Get detailed match statistics.

        Args:
            fixture_id: Fixture/match ID

        Returns:
            Dictionary containing match statistics
        """
        fixture_safe = sanitize_log_input(fixture_id)
        logger.info("Fetching match statistics for fixture %s", fixture_safe)

        headers = {
            self.api_key_header: self.api_key,
            "accept": "application/json"
        }

        params = {
            "fixture": fixture_safe
        }

        url = f"{self.base_url}/fixtures/statistics"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, params=params) as response:
                    logger.debug("Status code: %s", response.status)
                    logger.debug("Rate limit remaining: %s/%s",
                                 response.headers.get("x-ratelimit-requests-remaining"),
                                 response.headers.get("x-ratelimit-requests-limit"))

                    response.raise_for_status()
                    data = await response.json()
                    logger.info("Match statistics fetched successfully!")
                    return data.get("response", {})
        except aiohttp.ClientResponseError as http_err:
            logger.error("HTTP error occurred: %s", http_err)
        except Exception as err:
            logger.error("Unexpected error: %s", err)

        return {}

    async def get_player(self, player_id: int, season: int) -> dict[str, Any]:
        """
        Get data for a single player in a specific season.

        Args:
            player_id: Unique player ID
            season: Season year

        Returns:
            Dictionary containing player data
        """
        player_safe, season_safe = sanitize_multiple_log_inputs(player_id, season)
        logger.info("Fetching data for player %s, season %s", player_safe, season_safe)

        headers = {
            self.api_key_header: self.api_key,
            "accept": "application/json"
        }

        params = {
            "id": player_safe,
            "season": season_safe
        }

        url = f"{self.base_url}/players"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, params=params) as response:
                    logger.debug("Status code: %s", response.status)
                    logger.debug("Rate limit remaining: %s/%s",
                                response.headers.get("x-ratelimit-requests-remaining"),
                                response.headers.get("x-ratelimit-requests-limit"))

                    response.raise_for_status()
                    data = await response.json()
                    logger.info("Player data fetched successfully!")
                    return data.get("response", [{}])[0]  # return the first dict if found
        except aiohttp.ClientResponseError as http_err:
            logger.error("HTTP error occurred: %s", http_err)
        except Exception as err:
            logger.error("Unexpected error: %s", err)

        return {}

    async def get_players(self, team_id: int, season: int) -> list[dict[str, Any]]:
        """
        Get players from a team for a season.

        Args:
            team_id: Team ID
            season: Season year

        Returns:
            List of player data dictionaries
        """
        team_safe, season_safe = sanitize_multiple_log_inputs(team_id, season)
        logger.info("Fetching players for team %s, season %s", team_safe, season_safe)

        headers = {
            self.api_key_header: self.api_key,
            "accept": "application/json"
        }

        params = {
            "team": team_safe,
            "season": season_safe
        }

        url = f"{self.base_url}/players"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, params=params) as response:
                    logger.debug("Status code: %s", response.status)
                    logger.debug("Rate limit remaining: %s/%s",
                                 response.headers.get("x-ratelimit-requests-remaining"),
                                 response.headers.get("x-ratelimit-requests-limit"))

                    response.raise_for_status()
                    data = await response.json()
                    logger.info("Player data fetched successfully!")
                    return data.get("response", [])
        except aiohttp.ClientResponseError as http_err:
            logger.error("HTTP error occurred: %s", http_err)
        except Exception as err:
            logger.error("Unexpected error: %s", err)

        return []


# Football League IDs for common leagues (API-Football)
FOOTBALL_LEAGUES = {
    "premier_league": 39,
    "la_liga": 140,
    "serie_a": 135,
    "bundesliga": 78,
    "ligue_1": 61,
    "champions_league": 2,
    "europa_league": 3,
    "world_cup": 1,
}
