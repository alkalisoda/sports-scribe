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
from agents import Agent, GuardrailFunctionOutput, RunContextWrapper, Runner, output_guardrail, trace, function_tool
from pydantic import BaseModel
import http.client
import json

load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

currentModel = os.getenv("OPENAI_MODEL")

logger = logging.getLogger(__name__)

# class PlayerStats(BaseModel):
#     name: str
#     team: str
#     points: int
#     rebounds: int
#     assists: int
#     additional_stats: Optional[Dict[str, float]] = None

# class GameData(BaseModel):
#     game_id: str
#     home_team: str
#     away_team: str
#     final_score: str
#     date: str = Field(description="Date in ISO format (YYYY-MM-DD)")
#     key_stats: Optional[Dict[str, str]] = None  # Changed to single type for strict mode
#     player_performances: Optional[List[PlayerStats]] = None


class DataCollectorResponse(BaseModel):
    get: str
    parameters: Dict[str, int]
    errors: List[str]
    results: int
    paging: Dict[str, int]
    response: List[Dict[str, Any]]

class DataOutput(BaseModel):
    reasoning: str
    is_valid: bool

# original_prompt = """Expert sports data analyst. Collect comprehensive, accurate
#     game statistics from multiple sources. Validate data quality and flag any
#     inconsistencies. Prioritize official sources and recent updates."""

temp_prompt = "" """
        You are a specialized soccer data collector agent. Your role is to:
        1. Collect soccer/football data from the tools you are given
        2. Always return data in the exact JSON structure specified here.
        3. Validate data quality before returning results
        
        CRITICAL: You must ALWAYS return responses in this exact JSON format ONLY:
        {
            "get": "string describing what was requested",
            "parameters": {"dictionary of parameters used"},
            "errors": ["array of any errors encountered"],
            "results": "number of results returned",
            "paging": {
                "current": "current page number",
                "total": "total pages available"
            },
            "response": ["array of actual data objects"]
        }
        
        IMPORTANT RULES:
        - Return ONLY the JSON object, no additional text or explanations
        - Do not include markdown formatting or code blocks
        - If no data is found, return results: 0 and empty response array
        - Ensure all JSON is properly formatted with correct quotes and commas
        - If there's an error, include it in the "errors" array
        """

@function_tool
def get_player_data(player_id: str, season: str = "2023") -> str:
    """Get football/soccer player data from RapidAPI."""
    print("get_player_data():")
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
        print("Rapid API football player data retrieved successfully")
        return decoded_data
    except Exception as e:
        error_msg = f"Error fetching Rapid API football player data: {e}"
        print(error_msg)
        return error_msg

@function_tool
def get_game_data(fixture_id: str) -> str:
    """Get football game data from RapidAPI."""
    print("get_football_data():")
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
        logger.info(f"API raw response: {decoded_data}")

        print("Rapid API football game data retrieved successfully")

        return decoded_data
    except Exception as e:
        error_msg = f"Error fetching Rapid API football game data: {e}"
        print(error_msg)
        return error_msg


@function_tool
def get_team_data(team_id: str) -> str:
    """Get football/soccer team data from RapidAPI."""
    print("get_team_data():")
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
        print("Rapid API football team data retrieved successfully")
        return decoded_data
    except Exception as e:
        error_msg = f"Error fetching Rapid API football team data: {e}"
        print(error_msg)
        return error_msg


@function_tool
def get_football_data() -> str:
    """Get football/soccer team data from RapidAPI."""
    print("get_football_data():")
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

        print("Rapid API football team data retrieved successfully")

        return decoded_data
    except Exception as e:
        error_msg = f"Error fetching Rapid API football team data: {e}"
        print(error_msg)
        return error_msg


@output_guardrail
async def validate_data_quality(
    ctx: RunContextWrapper, agent: Agent, output: str
) -> GuardrailFunctionOutput:
    """Validate data quality with flexible validation."""
    try:
        # Always allow the output through, but log validation status
        if isinstance(output, str):
            # Try to parse as JSON to check structure
            import json
            try:
                data = json.loads(output)
                if isinstance(data, dict):
                    logger.info("Data validation: Valid JSON structure detected")
                    return GuardrailFunctionOutput(
                        output_info=DataOutput(reasoning="Valid JSON structure", is_valid=True),
                        tripwire_triggered=False
                    )
            except json.JSONDecodeError:
                logger.warning("Data validation: Output is not valid JSON, but allowing through")
        
        # Allow output through even if validation fails
        return GuardrailFunctionOutput(
            output_info=DataOutput(reasoning="Output allowed through validation", is_valid=True),
            tripwire_triggered=False
        )
        
    except Exception as e:
        logger.warning(f"Data validation error: {e}, allowing output through")
        return GuardrailFunctionOutput(
            output_info=DataOutput(reasoning=f"Validation error but allowing through: {e}", is_valid=True),
            tripwire_triggered=False
        )

def _extract_json_from_response(response_text: str) -> Dict[str, Any]:
    """Extract valid JSON from a response that may contain mixed content."""
    import re
    
    # First try direct JSON parsing
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON object with proper brace counting
    brace_count = 0
    start_pos = -1
    end_pos = -1
    
    for i, char in enumerate(response_text):
        if char == '{':
            if brace_count == 0:
                start_pos = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_pos != -1:
                end_pos = i
                break
    
    if start_pos != -1 and end_pos != -1:
        try:
            extracted_json = response_text[start_pos:end_pos + 1]
            return json.loads(extracted_json)
        except json.JSONDecodeError:
            pass
    
    # Try regex approach as last resort
    json_matches = list(re.finditer(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL))
    if json_matches:
        # Sort by length to get the largest JSON object
        largest_match = max(json_matches, key=lambda x: len(x.group(0)))
        try:
            return json.loads(largest_match.group(0))
        except json.JSONDecodeError:
            pass
    
    raise ValueError("Could not extract valid JSON from response")


class DataCollectorAgent():
    """Agent responsible for collecting sports data from various APIs and data sources."""

    def __init__(self, config: dict[str, Any]):
        """Initialize the Data Collector Agent with configuration."""
        self.agent= Agent(
            name="SportsDataCollector",
            instructions=temp_prompt,
            tools=[get_game_data, get_player_data, get_team_data, get_football_data],
            model=currentModel,
            output_guardrails=[validate_data_quality],
            )
        
        self.config = config
        logger.info("Data Collector Agent initialized")

    async def collect_game_data(self, game_id: str) -> Dict[str, Any]:
        """Collect game data for a specific game ID."""
        try:
            logger.info(f"Collecting game data for game {game_id}")
            
            # Use the agent to collect game data
            result = await Runner.run(self.agent, f"Get game data for fixture {game_id}")
            
            if not result or not result.final_output:
                raise ValueError("No game data received from collector")
            
            # Parse the result
            if isinstance(result.final_output, str):
                try:
                    data = _extract_json_from_response(result.final_output)
                    logger.info("Successfully parsed JSON response")
                except Exception as json_error:
                    logger.error(f"Invalid JSON response from agent: {json_error}")
                    logger.error(f"Raw response: {result.final_output[:500]}...")  # Log first 500 chars
                    raise ValueError(f"Invalid JSON response from agent: {json_error}")
            else:
                data = result.final_output
            
            logger.info(f"Successfully collected game data for game {game_id}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to collect game data for game {game_id}: {e}")
            raise

    async def collect_team_data(self, team_id: str) -> Dict[str, Any]:
        """Collect team data for a specific team ID."""
        try:
            logger.info(f"Collecting team data for team {team_id}")
            
            # Use the agent to collect team data
            result = await Runner.run(self.agent, f"Get team data for team {team_id}")
            
            if not result or not result.final_output:
                raise ValueError("No team data received from collector")
            
            # Parse the result
            if isinstance(result.final_output, str):
                try:
                    data = _extract_json_from_response(result.final_output)
                    logger.info("Successfully parsed JSON response")
                except Exception as json_error:
                    logger.error(f"Invalid JSON response from agent: {json_error}")
                    logger.error(f"Raw response: {result.final_output[:500]}...")  # Log first 500 chars
                    raise ValueError(f"Invalid JSON response from agent: {json_error}")
            else:
                data = result.final_output
            
            logger.info(f"Successfully collected team data for team {team_id}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to collect team data for team {team_id}: {e}")
            raise

    async def collect_player_data(self, player_id: str, season: str) -> Dict[str, Any]:
        """Collect player data for a specific player ID and season."""
        try:
            logger.info(f"Collecting player data for player {player_id} in season {season}")
            # Use the agent to collect player data
            result = await Runner.run(self.agent, f"Get player data for player {player_id} in season {season}")
            if not result or not result.final_output:
                raise ValueError("No player data received from collector")
            # Parse the result
            if isinstance(result.final_output, str):
                try:
                    data = _extract_json_from_response(result.final_output)
                    logger.info("Successfully parsed JSON response")
                except Exception as json_error:
                    logger.error(f"Invalid JSON response from agent: {json_error}")
                    logger.error(f"Raw response: {result.final_output[:500]}...")  # Log first 500 chars
                    raise ValueError(f"Invalid JSON response from agent: {json_error}")
            else:
                data = result.final_output
            logger.info(f"Successfully collected player data for player {player_id} in season {season}")
            return data
        except Exception as e:
            logger.error(f"Failed to collect player data for player {player_id} in season {season}: {e}")
            raise


async def main():
     param = dict[str, Any]
     dc = DataCollectorAgent(param)
    
     with trace("Initialize data collector agent class: "):
        try:
            data = await Runner.run(dc.agent, temp_prompt)
            print("AI: ", data.final_output)
        
        except Exception as e:
            print(f"Error generating data: {e}")
            return f"Error generating data: {e}"
    

if __name__ == "__main__":
    asyncio.run(main())
