"""Data Collector Agent.

This agent is responsible for gathering game data from various sports APIs.
It collects real-time and historical sports data to feed into the content generation pipeline.

Key improvements:
- Async HTTP client (httpx) for better performance
- Enhanced error handling and retry logic
- RapidAPI rate limit monitoring
- Request timeout handling
- OpenAI client with automatic retries

Requirements:
    - httpx: pip install httpx
    - openai: pip install openai
"""

import logging
from typing import Any, Dict, List, Optional
from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletion
import asyncio
import os
import time
from dotenv import load_dotenv
from agents import function_tool, trace
from pydantic import BaseModel
import httpx
import json
from dataclasses import dataclass

load_dotenv()

# Initialize OpenAI client with improved configuration
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    max_retries=3,  # Automatic retry configuration
    timeout=30.0,   # 30 second timeout
)

# Async OpenAI client for better performance
async_client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    max_retries=3,
    timeout=30.0,
)

current_model = os.getenv("OPENAI_MODEL")

logger = logging.getLogger(__name__)

# Utility functions for rate limit monitoring  
def _extract_rate_limit_info(headers: Dict[str, str]) -> 'RateLimitInfo':
    """Extract rate limit information from RapidAPI response headers."""
    return RateLimitInfo(
        requests_limit=_safe_int_convert(headers.get('x-ratelimit-requests-limit')),
        requests_remaining=_safe_int_convert(headers.get('x-ratelimit-requests-remaining')),
        requests_reset=_safe_int_convert(headers.get('x-ratelimit-requests-reset'))
    )

def _safe_int_convert(value: Optional[str]) -> Optional[int]:
    """Safely convert string to int, return None if conversion fails."""
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None

def _log_rate_limit_info(rate_limit_info: 'RateLimitInfo', endpoint_name: str) -> None:
    """Log rate limit information for monitoring."""
    if rate_limit_info.requests_remaining is not None:
        logging.info(
            f"RapidAPI {endpoint_name} - Rate limit: {rate_limit_info.requests_remaining}/{rate_limit_info.requests_limit} remaining, resets in {rate_limit_info.requests_reset}s"
        )
        
        # Warning if rate limit is getting low
        if rate_limit_info.requests_remaining < 10:
            logging.warning(
                f"RapidAPI {endpoint_name} - Low rate limit: only {rate_limit_info.requests_remaining} requests remaining!"
            )

@dataclass
class RateLimitInfo:
    """Rate limit information from RapidAPI response headers."""
    requests_limit: Optional[int] = None
    requests_remaining: Optional[int] = None
    requests_reset: Optional[int] = None
    
class DataCollectorResponse(BaseModel):
    get: str
    parameters: Dict[str, int]
    errors: List[str]
    results: int
    paging: Dict[str, int]
    response: List[Dict[str, Any]]
    rate_limit_info: Optional[RateLimitInfo] = None

async def get_player_data(player_id: str, season: str = "2023") -> Dict[str, Any]:
    """Get football/soccer player data from RapidAPI with async HTTP client."""
    logging.info("Getting player data for player: %s in season: %s", player_id, season)
    
    api_key = os.getenv("RAPIDAPI_KEY")
    if not api_key:
        raise ValueError("RAPIDAPI_KEY not found in environment variables")
    
    headers = {
        'x-rapidapi-host': "api-football-v1.p.rapidapi.com",
        'x-rapidapi-key': api_key,
    }
    
    url = f"https://api-football-v1.p.rapidapi.com/v3/players?id={player_id}&season={season}"
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(url, headers=headers)
            response.raise_for_status()  # Raises exception for HTTP errors
            
            # Extract rate limit information
            rate_limit_info = _extract_rate_limit_info(response.headers)
            _log_rate_limit_info(rate_limit_info, "player data")
            
            data = response.json()
            data['rate_limit_info'] = rate_limit_info
            
            logging.info("RapidAPI football player data retrieved successfully")
            return data
            
        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP error fetching player data: {e.response.status_code} - {e.response.text}"
            logging.error(error_msg)
            raise Exception(error_msg)
        except httpx.RequestError as e:
            error_msg = f"Request error fetching player data: {str(e)}"
            logging.error(error_msg)
            raise Exception(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error fetching player data: {str(e)}"
            logging.error(error_msg)
            raise Exception(error_msg)

async def get_game_data(fixture_id: str) -> Dict[str, Any]:
    """Get football game data from RapidAPI with async HTTP client."""
    logging.info("Getting game data for fixture: %s", fixture_id)
    
    api_key = os.getenv("RAPIDAPI_KEY")
    if not api_key:
        raise ValueError("RAPIDAPI_KEY not found in environment variables")
    
    headers = {
        'x-rapidapi-key': api_key,
        'x-rapidapi-host': "api-football-v1.p.rapidapi.com"
    }
    
    url = f"https://api-football-v1.p.rapidapi.com/v3/fixtures?id={fixture_id}"
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            
            # Extract rate limit information
            rate_limit_info = _extract_rate_limit_info(response.headers)
            _log_rate_limit_info(rate_limit_info, "game data")
            
            data = response.json()
            data['rate_limit_info'] = rate_limit_info
            
            logging.info("RapidAPI football game data retrieved successfully")
            return data
            
        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP error fetching game data: {e.response.status_code} - {e.response.text}"
            logging.error(error_msg)
            raise Exception(error_msg)
        except httpx.RequestError as e:
            error_msg = f"Request error fetching game data: {str(e)}"
            logging.error(error_msg)
            raise Exception(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error fetching game data: {str(e)}"
            logging.error(error_msg)
            raise Exception(error_msg)

async def get_team_data(team_id: str) -> Dict[str, Any]:
    """Get football/soccer team data from RapidAPI with async HTTP client."""
    logging.info(f"Getting team data for team: {team_id}")
    
    api_key = os.getenv("RAPIDAPI_KEY")
    if not api_key:
        raise ValueError("RAPIDAPI_KEY not found in environment variables")
    
    headers = {
        'x-rapidapi-host': "api-football-v1.p.rapidapi.com",
        'x-rapidapi-key': api_key,
    }
    
    url = f"https://api-football-v1.p.rapidapi.com/v3/teams?id={team_id}"
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            
            # Extract rate limit information
            rate_limit_info = _extract_rate_limit_info(response.headers)
            _log_rate_limit_info(rate_limit_info, "team data")
            
            data = response.json()
            data['rate_limit_info'] = rate_limit_info
            
            logging.info("RapidAPI football team data retrieved successfully")
            return data
            
        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP error fetching team data: {e.response.status_code} - {e.response.text}"
            logging.error(error_msg)
            raise Exception(error_msg)
        except httpx.RequestError as e:
            error_msg = f"Request error fetching team data: {str(e)}"
            logging.error(error_msg)
            raise Exception(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error fetching team data: {str(e)}"
            logging.error(error_msg)
            raise Exception(error_msg)


async def get_football_data() -> Dict[str, Any]:
    """Get football/soccer team data from RapidAPI (legacy function - consider using get_team_data instead)."""
    logging.info("Getting football data from RapidAPI")
    
    api_key = os.getenv("RAPIDAPI_KEY")
    if not api_key:
        raise ValueError("RAPIDAPI_KEY not found in environment variables")
    
    headers = {
        'x-rapidapi-host': "api-football-v1.p.rapidapi.com",
        'x-rapidapi-key': api_key,
    }
    
    url = "https://api-football-v1.p.rapidapi.com/v3/teams?id=33"
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            
            # Extract rate limit information
            rate_limit_info = _extract_rate_limit_info(response.headers)
            _log_rate_limit_info(rate_limit_info, "football data")
            
            data = response.json()
            data['rate_limit_info'] = rate_limit_info
            
            logging.info("RapidAPI football team data retrieved successfully")
            return data
            
        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP error fetching football data: {e.response.status_code} - {e.response.text}"
            logging.error(error_msg)
            raise Exception(error_msg)
        except httpx.RequestError as e:
            error_msg = f"Request error fetching football data: {str(e)}"
            logging.error(error_msg)
            raise Exception(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error fetching football data: {str(e)}"
            logging.error(error_msg)
            raise Exception(error_msg)


# Validation functions removed - direct API calls don't need them


class DataCollectorAgent():
    """Agent responsible for collecting sports data from various APIs and data sources."""

    def __init__(self, config: dict[str, Any]):
        """Initialize the Data Collector with configuration."""
        self.config = config
        logger.info("Data Collector initialized for direct API calls")

    async def collect_game_data(self, game_id: str) -> Dict[str, Any]:
        """Collect game data for a specific game ID directly from API."""
        try:
            logger.info(f"Collecting game data for game {game_id}")
            
            # Call the async API function directly
            data = await get_game_data(game_id)
            
            if not data:
                raise ValueError("No game data received from API")

            logger.info(f"Successfully collected game data for game {game_id}")
            return data
        except Exception as e:
            logger.error(f"Failed to collect game data for game {game_id}: {e}")
            raise

    async def collect_team_data(self, team_id: str) -> Dict[str, Any]:
        """Collect team data for a specific team ID directly from API."""
        try:
            logger.info(f"Collecting team data for team {team_id}")
            
            # Call the async API function directly
            data = await get_team_data(team_id)
            
            if not data:
                raise ValueError("No team data received from API")

            logger.info(f"Successfully collected team data for team {team_id}")
            return data
        except Exception as e:
            logger.error(f"Failed to collect team data for team {team_id}: {e}")
            raise

    async def collect_player_data(self, player_id: str, season: str) -> Dict[str, Any]:
        """Collect player data for a specific player ID and season directly from API."""
        try:
            logger.info(f"Collecting player data for player {player_id} in season {season}")
            
            # Call the async API function directly
            data = await get_player_data(player_id, season)
            
            if not data:
                raise ValueError("No player data received from API")

            logger.info(f"Successfully collected player data for player {player_id} in season {season}")
            return data
        except Exception as e:
            logger.error(f"Failed to collect player data for player {player_id} in season {season}: {e}")
            raise
    
    async def analyze_data_with_openai(self, data: Dict[str, Any], prompt: str) -> str:
        """Analyze sports data using OpenAI with improved error handling."""
        try:
            logger.info("Analyzing data with OpenAI")
            
            # Use the async client for better performance
            response = await async_client.chat.completions.create(
                model=currentModel or "gpt-4",
                messages=[
                    {"role": "system", "content": "You are a sports data analyst."},
                    {"role": "user", "content": f"{prompt}\n\nData: {json.dumps(data, indent=2)}"}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            
            result = response.choices[0].message.content
            logger.info("OpenAI analysis completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Failed to analyze data with OpenAI: {e}")
            raise


async def main():
    """Main function to test the DataCollectorAgent with performance monitoring."""
    param: Dict[str, Any] = {}
    dc = DataCollectorAgent(param)
    
    with trace("Initialize data collector agent class: "):
        try:
            print(">> Testing Improved Data Collector Agent")
            print("=" * 50)
            
            total_start_time = time.time()
            
            # Test game data collection
            print(">> Testing Game Data Collection...")
            start_time = time.time()
            game_data = await dc.collect_game_data("239625")
            game_time = time.time() - start_time
            print(f"   [OK] Completed in {game_time:.2f}s")
            print(f"   Data keys: {list(game_data.keys()) if game_data else 'No data'}")
            if 'rate_limit_info' in game_data and game_data['rate_limit_info']:
                rl_info = game_data['rate_limit_info']
                print(f"   Rate limit: {rl_info.requests_remaining}/{rl_info.requests_limit} remaining")
            
            # Test team data collection
            print("\n>> Testing Team Data Collection...")
            start_time = time.time()
            team_data = await dc.collect_team_data("33")
            team_time = time.time() - start_time
            print(f"   [OK] Completed in {team_time:.2f}s")
            print(f"   Data keys: {list(team_data.keys()) if team_data else 'No data'}")
            if 'rate_limit_info' in team_data and team_data['rate_limit_info']:
                rl_info = team_data['rate_limit_info']
                print(f"   Rate limit: {rl_info.requests_remaining}/{rl_info.requests_limit} remaining")
            
            # Test player data collection
            print("\n>> Testing Player Data Collection...")
            start_time = time.time()
            player_data = await dc.collect_player_data("276", "2023")
            player_time = time.time() - start_time
            print(f"   [OK] Completed in {player_time:.2f}s")
            print(f"   Data keys: {list(player_data.keys()) if player_data else 'No data'}")
            if 'rate_limit_info' in player_data and player_data['rate_limit_info']:
                rl_info = player_data['rate_limit_info']
                print(f"   Rate limit: {rl_info.requests_remaining}/{rl_info.requests_limit} remaining")
            
            total_time = time.time() - total_start_time
            
            print("\n" + "=" * 50)
            print("PERFORMANCE SUMMARY:")
            print(f"   * Game data: {game_time:.2f}s")
            print(f"   * Team data: {team_time:.2f}s")  
            print(f"   * Player data: {player_time:.2f}s")
            print(f"   * Total time: {total_time:.2f}s")
            print(f"   * Average per request: {total_time/3:.2f}s")
            print("\nIMPROVEMENTS ACTIVE:")
            print("   [OK] Async HTTP client (httpx)")
            print("   [OK] Enhanced error handling")
            print("   [OK] Rate limit monitoring")
            print("   [OK] Request timeout (30s)")
            print("   [OK] OpenAI client with retries")
            print("\n>> All API tests completed successfully!")
            
        except Exception as e:
            error_msg = f"[ERROR] Error in data collection tests: {e}"
            print(error_msg)
            logging.error(error_msg)
            return error_msg
    

if __name__ == "__main__":
    asyncio.run(main())
