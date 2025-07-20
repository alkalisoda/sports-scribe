"""Data Collector Agent.

This agent is responsible for gathering game data from various sports APIs.
It collects real-time and historical sports data to feed into the content generation pipeline.
"""

import logging
from typing import Any, Dict, List
from openai import OpenAI
import asyncio
import os
from dotenv import load_dotenv
from agents import function_tool, trace
from pydantic import BaseModel
import http.client
import json

load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

currentModel = os.getenv("OPENAI_MODEL")

logger = logging.getLogger(__name__)

class DataCollectorResponse(BaseModel):
    get: str
    parameters: Dict[str, int]
    errors: List[str]
    results: int
    paging: Dict[str, int]
    response: List[Dict[str, Any]]

def get_player_data(player_id: str, season: str = "2023") -> str:
    """Get football/soccer player data from RapidAPI."""
    logging.info("Getting player data for player: %s in season: %s", player_id, season)
    try:
        api_key = os.getenv("RAPIDAPI_KEY")
        if not api_key:
            raise ValueError("RAPID_API_KEY not found.")
        
        conn = http.client.HTTPSConnection("api-football-v1.p.rapidapi.com")
        
        headers = {
            'x-rapidapi-host': "api-football-v1.p.rapidapi.com",
            'x-rapidapi-key': api_key,
        }

        conn.request("GET", f"/v3/players?id={player_id}&season={season}", headers=headers)

        response = conn.getresponse()
        data = response.read()
        decoded_data = data.decode("utf8")
        logging.info("Rapid API football player data retrieved successfully")
        return decoded_data
    except Exception as e:
        error_msg = f"Error fetching Rapid API football player data: {e}"
        logging.error(error_msg)
        return error_msg

def get_game_data(fixture_id: str) -> str:
    """Get football game data from RapidAPI."""
    logging.info("Getting game data for fixture: %s", fixture_id)
    try:
        api_key = os.getenv("RAPIDAPI_KEY")
        if not api_key:
            raise ValueError("RAPIDAPI_KEY not found.")
        
        conn = http.client.HTTPSConnection("api-football-v1.p.rapidapi.com")
        
        headers = {
        'x-rapidapi-key': api_key,
        'x-rapidapi-host': "api-football-v1.p.rapidapi.com"
        }

        conn.request("GET", f"/v3/fixtures?id={fixture_id}", headers=headers)

        response = conn.getresponse()
        data = response.read()

        decoded_data = data.decode("utf8")
        logging.info("Rapid API football game data retrieved successfully")
        # logging.info(decoded_data)

        return decoded_data
    except Exception as e:
        error_msg = f"Error fetching Rapid API football game data: {e}"
        logging.error(error_msg)
        return error_msg

def get_team_data(team_id: str) -> str:
    """Get football/soccer team data from RapidAPI."""
    logging.info(f"Getting team data for team: {team_id}")
    try:
        api_key = os.getenv("RAPIDAPI_KEY")
        if not api_key:
            raise ValueError("RAPID_API_KEY not found.")
        
        conn = http.client.HTTPSConnection("api-football-v1.p.rapidapi.com")
        
        headers = {
            'x-rapidapi-host': "api-football-v1.p.rapidapi.com",
            'x-rapidapi-key': api_key,
        }

        conn.request("GET", f"/v3/teams?id={team_id}", headers=headers)

        response = conn.getresponse()
        data = response.read()
        decoded_data = data.decode("utf8")
        logging.info("Rapid API football team data retrieved successfully")
        return decoded_data
    except Exception as e:
        error_msg = f"Error fetching Rapid API football team data: {e}"
        print(error_msg)
        return error_msg


def get_football_data() -> str:
    """Get football/soccer team data from RapidAPI."""
    logging.info("Getting football data from RapidAPI")
    try:
        api_key = os.getenv("RAPIDAPI_KEY")
        if not api_key:
            raise ValueError("RAPID_API_KEY not found.")
        
        conn = http.client.HTTPSConnection("api-football-v1.p.rapidapi.com")
        
        headers = {
            'x-rapidapi-host': "api-football-v1.p.rapidapi.com",
            'x-rapidapi-key': api_key,
        }

        conn.request("GET", "/v3/teams?id=33", headers=headers)

        response = conn.getresponse() #Returns HTTP response object
        data = response.read()

        decoded_data = data.decode("utf8")
        logging.info("Rapid API football team data retrieved successfully")
        return decoded_data
    except Exception as e:
        error_msg = f"Error fetching Rapid API football team data: {e}"
        logging.error(error_msg)
        return error_msg


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
            
            # Call the API function directly
            raw_data = get_game_data(game_id)
            
            if not raw_data:
                raise ValueError("No game data received from API")
            
            # Parse the JSON response
            try:
                data = json.loads(raw_data)
                logger.info("Successfully parsed JSON response")
                logger.info(f"Successfully collected game data for game {game_id}")
                return data
                
            except json.JSONDecodeError as json_error:
                logger.error(f"Invalid JSON response from API: {json_error}")
                logger.error(f"Raw response: {raw_data[:500]}...")  # Log first 500 chars
                raise ValueError(f"Invalid JSON response from API: {json_error}")
            
        except Exception as e:
            logger.error(f"Failed to collect game data for game {game_id}: {e}")
            raise

    async def collect_team_data(self, team_id: str) -> Dict[str, Any]:
        """Collect team data for a specific team ID directly from API."""
        try:
            logger.info(f"Collecting team data for team {team_id}")
            
            # Call the API function directly
            raw_data = get_team_data(team_id)
            
            if not raw_data:
                raise ValueError("No team data received from API")
            
            # Parse the JSON response
            try:
                data = json.loads(raw_data)
                logger.info("Successfully parsed JSON response")
                logger.info(f"Successfully collected team data for team {team_id}")
                return data
                
            except json.JSONDecodeError as json_error:
                logger.error(f"Invalid JSON response from API: {json_error}")
                logger.error(f"Raw response: {raw_data[:500]}...")  # Log first 500 chars
                raise ValueError(f"Invalid JSON response from API: {json_error}")
            
        except Exception as e:
            logger.error(f"Failed to collect team data for team {team_id}: {e}")
            raise

    async def collect_player_data(self, player_id: str, season: str) -> Dict[str, Any]:
        """Collect player data for a specific player ID and season directly from API."""
        try:
            logger.info(f"Collecting player data for player {player_id} in season {season}")
            
            # Call the API function directly
            raw_data = get_player_data(player_id, season)
            
            if not raw_data:
                raise ValueError("No player data received from API")
            
            # Parse the JSON response
            try:
                data = json.loads(raw_data)
                logger.info("Successfully parsed JSON response")
                logger.info(f"Successfully collected player data for player {player_id} in season {season}")
                return data
                
            except json.JSONDecodeError as json_error:
                logger.error(f"Invalid JSON response from API: {json_error}")
                logger.error(f"Raw response: {raw_data[:500]}...")  # Log first 500 chars
                raise ValueError(f"Invalid JSON response from API: {json_error}")
            
        except Exception as e:
            logger.error(f"Failed to collect player data for player {player_id} in season {season}: {e}")
            raise


async def main():
     param = dict[str, Any]
     dc = DataCollectorAgent(param)
    
     with trace("Initialize data collector agent class: "):
        try:
            # Test game data collection
            print("Testing Game Data Collection...")
            game_data = await dc.collect_game_data("239625")
            print("Game Data: ", game_data)
            
            # Test team data collection
            print("\nTesting Team Data Collection...")
            team_data = await dc.collect_team_data("33")
            print("Team Data: ", team_data)
            
            # Test player data collection
            print("\nTesting Player Data Collection...")
            player_data = await dc.collect_player_data("276", "2023")
            print("Player Data: ", player_data)
        
        except Exception as e:
            print(f"Error generating data: {e}")
            return f"Error generating data: {e}"
    

if __name__ == "__main__":
    asyncio.run(main())
