"""Format Manager Agent.

This agent handles data format conversion between different agents in the pipeline.
It ensures that data from one agent is properly formatted for consumption by another agent.
"""

import logging
import json
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from agents import Agent, Runner, function_tool
import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


# ============================================================================
# Data Collector → Researcher Format Conversion Functions
# ============================================================================

@function_tool
def format_game_data_for_researcher(game_data: str, research_type: str) -> str:
    """Format game data from data collector for researcher agent input.
    
    Args:
        game_data: Raw game data from data collector as JSON string
        research_type: Type of research (team_history, player_performance, season_trends, game_analysis)
        
    Returns:
        Formatted data for researcher agent as JSON string
    """
    try:
        data = json.loads(game_data)
        
        # Extract fixture data
        fixture_data = data.get("response", [{}])[0].get("fixture", {})
        fixture_response = fixture_data.get("response", [])
        
        if not fixture_response:
            return json.dumps({"error": "No fixture data found"}, ensure_ascii=False)
        
        fixture = fixture_response[0]
        teams = fixture.get("teams", {})
        goals = fixture.get("goals", {})
        fixture_info = fixture.get("fixture", {})
        league_info = fixture.get("league", {})
        
        # Base extracted data
        extracted_data = {
            "game_id": str(fixture_info.get("id", "")),
            "home_team": teams.get("home", {}).get("name", ""),
            "away_team": teams.get("away", {}).get("name", ""),
            "home_team_id": str(teams.get("home", {}).get("id", "")),
            "away_team_id": str(teams.get("away", {}).get("id", "")),
            "home_score": goals.get("home", 0),
            "away_score": goals.get("away", 0),
            "date": fixture_info.get("date", ""),
            "venue": fixture_info.get("venue", {}).get("name", ""),
            "status": fixture_info.get("status", {}).get("long", ""),
            "league": league_info.get("name", ""),
            "league_id": str(league_info.get("id", "")),
            "season": str(league_info.get("season", "")),
            "round": league_info.get("round", "")
        }
        
        # Add research-specific data
        if research_type == "team_history":
            result = {
                "research_type": "team_history",
                "home_team_id": extracted_data["home_team_id"],
                "away_team_id": extracted_data["away_team_id"],
                "home_team": extracted_data["home_team"],
                "away_team": extracted_data["away_team"],
                "league_id": extracted_data["league_id"],
                "season": extracted_data["season"]
            }
        elif research_type == "season_trends":
            result = {
                "research_type": "season_trends",
                "league_id": extracted_data["league_id"],
                "season": extracted_data["season"],
                "league": extracted_data["league"]
            }
        elif research_type == "game_analysis":
            result = {
                "research_type": "game_analysis",
                "game_data": extracted_data,
                "fixture_data": fixture
            }
        else:
            result = {
                "research_type": research_type,
                "data": extracted_data
            }
        
        return json.dumps(result, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"Error formatting game data for researcher: {e}")
        return json.dumps({"error": f"Formatting failed: {str(e)}"}, ensure_ascii=False)


@function_tool
def format_player_data_for_researcher(player_data: str, research_type: str) -> str:
    """Format player data from data collector for researcher agent input.
    
    Args:
        player_data: Raw player data from data collector as JSON string
        research_type: Type of research (player_performance, player_history, etc.)
        
    Returns:
        Formatted data for researcher agent as JSON string
    """
    try:
        data = json.loads(player_data)
        
        player_info = data.get("response", [{}])[0].get("player_info", {})
        player_response = player_info.get("response", [])
        
        # Handle empty player data
        if not player_response:
            # Create default player data structure
            extracted_data = {
                "player_id": "unknown",
                "player_name": "Unknown Player",
                "age": None,
                "height": None,
                "weight": None,
                "nationality": None,
                "position": None,
                "team_name": None,
                "team_id": None,
                "league": None,
                "league_id": None,
                "season": None
            }
            
            result = {
                "research_type": research_type,
                "player_data": extracted_data,
                "statistics": {},
                "data_available": False
            }
            
            return json.dumps(result, ensure_ascii=False)
        
        player = player_response[0]
        player_details = player.get("player", {})
        statistics = player.get("statistics", [])
        
        # Extract player data
        extracted_data = {
            "player_id": str(player_details.get("id", "")),
            "player_name": player_details.get("name", ""),
            "age": player_details.get("age", ""),
            "height": player_details.get("height", ""),
            "weight": player_details.get("weight", ""),
            "nationality": player_details.get("nationality", ""),
            "position": statistics[0].get("games", {}).get("position", "") if statistics else "",
            "team_name": statistics[0].get("team", {}).get("name", "") if statistics else "",
            "team_id": str(statistics[0].get("team", {}).get("id", "")) if statistics else "",
            "league": statistics[0].get("league", {}).get("name", "") if statistics else "",
            "league_id": str(statistics[0].get("league", {}).get("id", "")) if statistics else "",
            "season": str(statistics[0].get("league", {}).get("season", "")) if statistics else ""
        }
        
        result = {
            "research_type": research_type,
            "player_data": extracted_data,
            "statistics": statistics[0] if statistics else {},
            "data_available": True
        }
        
        return json.dumps(result, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"Error formatting player data for researcher: {e}")
        return json.dumps({"error": f"Formatting failed: {str(e)}"}, ensure_ascii=False)


@function_tool
def format_team_data_for_researcher(team_data: str, research_type: str) -> str:
    """Format team data from data collector for researcher agent input.
    
    Args:
        team_data: Raw team data from data collector as JSON string
        research_type: Type of research (team_history, team_stats, etc.)
        
    Returns:
        Formatted data for researcher agent as JSON string
    """
    try:
        data = json.loads(team_data)
        
        team_info = data.get("response", [{}])[0].get("team_info", {})
        team_response = team_info.get("response", [])
        
        if not team_response:
            return json.dumps({"error": "No team data found"}, ensure_ascii=False)
        
        team = team_response[0]
        team_details = team.get("team", {})
        venue = team.get("venue", {})
        
        # Extract team data
        extracted_data = {
            "team_id": str(team_details.get("id", "")),
            "team_name": team_details.get("name", ""),
            "country": team_details.get("country", ""),
            "founded": team_details.get("founded", ""),
            "venue_name": venue.get("name", ""),
            "venue_capacity": venue.get("capacity", ""),
            "venue_city": venue.get("city", "")
        }
        
        result = {
            "research_type": research_type,
            "team_data": extracted_data
        }
        
        return json.dumps(result, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"Error formatting team data for researcher: {e}")
        return json.dumps({"error": f"Formatting failed: {str(e)}"}, ensure_ascii=False)


# ============================================================================
# Researcher → Writer Format Conversion Functions
# ============================================================================

@function_tool
def format_research_data_for_writer(research_data: str, article_type: str) -> str:
    """Format research data from researcher for writer agent input.
    
    Args:
        research_data: Research data from researcher agent as JSON string
        article_type: Type of article to generate (game_recap, player_spotlight, preview_article)
        
    Returns:
        Formatted research data for writer agent as JSON string
    """
    try:
        data = json.loads(research_data)
        
        # Extract key research components
        research_result = {
            "context": data.get("context", ""),
            "statistics": data.get("statistics", {}),
            "analysis": data.get("analysis", []),
            "storylines": data.get("storylines", []),
            "key_findings": data.get("key_findings", []),
            "sources": data.get("sources", [])
        }
        
        # Add article-specific formatting
        if article_type == "game_recap":
            result = {
                "article_type": "game_recap",
                "research": research_result,
                "focus_areas": ["match_analysis", "key_moments", "player_performance", "tactical_insights"]
            }
        elif article_type == "player_spotlight":
            result = {
                "article_type": "player_spotlight",
                "research": research_result,
                "focus_areas": ["player_background", "performance_analysis", "career_highlights", "future_prospects"]
            }
        elif article_type == "preview_article":
            result = {
                "article_type": "preview_article",
                "research": research_result,
                "focus_areas": ["team_form", "head_to_head", "key_players", "prediction_factors"]
            }
        else:
            result = {
                "article_type": article_type,
                "research": research_result
            }
        
        return json.dumps(result, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"Error formatting research data for writer: {e}")
        return json.dumps({"error": f"Formatting failed: {str(e)}"}, ensure_ascii=False)


# ============================================================================
# Data Collector + Research → Writer Format Conversion Functions
# ============================================================================

@function_tool
def format_complete_data_for_writer(data_collector_output: str, research_output: str, article_type: str) -> str:
    """Format combined data collector and research output for writer agent input.
    
    Args:
        data_collector_output: Output from data collector agent as JSON string
        research_output: Output from research agent as JSON string
        article_type: Type of article to generate
        
    Returns:
        Formatted input for writer agent as JSON string with required structure: {data, research, storylines}
    """
    try:
        # Parse data collector output
        data = json.loads(data_collector_output)
        
        # Parse research output (handle None/empty cases)
        research = {}
        if research_output and research_output.strip() and research_output.lower() != "none":
            try:
                # Handle both JSON strings and Pydantic models
                if isinstance(research_output, str):
                    research = json.loads(research_output)
                else:
                    # Handle Pydantic models
                    if hasattr(research_output, 'model_dump'):
                        research = research_output.model_dump()
                    elif hasattr(research_output, 'dict'):
                        research = research_output.dict()
                    else:
                        research = {"raw_research": str(research_output)}
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse research_output as JSON: {research_output[:100] if isinstance(research_output, str) else str(research_output)[:100]}...")
                research = {"raw_research": research_output}
        
        # Extract storylines from research data
        storylines = []
        if isinstance(research, dict):
            if "storylines" in research and isinstance(research["storylines"], list):
                storylines = research["storylines"]
            elif "analysis" in research and isinstance(research["analysis"], list):
                storylines = research["analysis"]
            elif "key_findings" in research and isinstance(research["key_findings"], list):
                storylines = research["key_findings"]
        
        # Ensure we have at least some storylines
        if not storylines:
            storylines = ["Default storyline based on available data"]
        
        # Create the required structure for writer agent
        result = {
            "data": data,
            "research": research,
            "storylines": storylines
        }
        
        return json.dumps(result, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"Error formatting complete data for writer: {e}")
        return json.dumps({"error": f"Formatting failed: {str(e)}"}, ensure_ascii=False)


# ============================================================================
# Data Format Conversion Functions (for specific article types)
# ============================================================================

@function_tool
def convert_game_data_to_recap_format(game_data: str) -> str:
    """Convert raw game data to format suitable for game recap generation.
    
    Args:
        game_data: Raw game data from data collector as JSON string
        
    Returns:
        Formatted data for game recap generation as JSON string
    """
    try:
        data = json.loads(game_data)
        
        # Extract fixture data
        fixture_data = data.get("response", [{}])[0].get("fixture", {})
        fixture_response = fixture_data.get("response", [])
        
        if not fixture_response:
            return json.dumps({"error": "No fixture data found"}, ensure_ascii=False)
        
        fixture = fixture_response[0]
        teams = fixture.get("teams", {})
        goals = fixture.get("goals", {})
        fixture_info = fixture.get("fixture", {})
        
        # Format for game recap
        formatted_data = {
            "match_info": {
                "home_team": teams.get("home", {}).get("name", "Unknown"),
                "away_team": teams.get("away", {}).get("name", "Unknown"),
                "home_score": goals.get("home", 0),
                "away_score": goals.get("away", 0),
                "date": fixture_info.get("date", "Unknown"),
                "venue": fixture_info.get("venue", {}).get("name", "Unknown"),
                "status": fixture_info.get("status", {}).get("long", "Unknown")
            },
            "statistics": {
                "home_stats": fixture.get("statistics", []),
                "away_stats": fixture.get("statistics", [])
            },
            "events": fixture.get("events", []),
            "lineups": {
                "home_lineup": fixture.get("lineups", []),
                "away_lineup": fixture.get("lineups", [])
            }
        }
        
        return json.dumps(formatted_data, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"Error converting game data: {e}")
        return json.dumps({"error": f"Conversion failed: {str(e)}"}, ensure_ascii=False)


@function_tool
def convert_player_data_to_spotlight_format(player_data: str) -> str:
    """Convert raw player data to format suitable for player spotlight generation.
    
    Args:
        player_data: Raw player data from data collector as JSON string
        
    Returns:
        Formatted data for player spotlight generation as JSON string
    """
    try:
        data = json.loads(player_data)
        
        player_info = data.get("response", [{}])[0].get("player_info", {})
        player_response = player_info.get("response", [])
        
        if not player_response:
            return json.dumps({"error": "No player data found"}, ensure_ascii=False)
        
        player = player_response[0]
        player_details = player.get("player", {})
        statistics = player.get("statistics", [])
        
        # Format for player spotlight
        formatted_data = {
            "player_info": {
                "name": player_details.get("name", "Unknown"),
                "age": player_details.get("age", "Unknown"),
                "height": player_details.get("height", "Unknown"),
                "weight": player_details.get("weight", "Unknown"),
                "nationality": player_details.get("nationality", "Unknown"),
                "position": statistics[0].get("games", {}).get("position", "Unknown") if statistics else "Unknown"
            },
            "team_info": {
                "team_name": statistics[0].get("team", {}).get("name", "Unknown") if statistics else "Unknown",
                "league": statistics[0].get("league", {}).get("name", "Unknown") if statistics else "Unknown"
            },
            "statistics": {
                "appearances": statistics[0].get("games", {}).get("appearences", 0) if statistics else 0,
                "goals": statistics[0].get("goals", {}).get("total", 0) if statistics else 0,
                "assists": statistics[0].get("goals", {}).get("assists", 0) if statistics else 0,
                "yellow_cards": statistics[0].get("cards", {}).get("yellow", 0) if statistics else 0,
                "red_cards": statistics[0].get("cards", {}).get("red", 0) if statistics else 0
            },
            "performance_data": statistics[0] if statistics else {}
        }
        
        return json.dumps(formatted_data, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"Error converting player data: {e}")
        return json.dumps({"error": f"Conversion failed: {str(e)}"}, ensure_ascii=False)


@function_tool
def convert_team_data_to_preview_format(team_data: str, opponent_data: str) -> str:
    """Convert raw team data to format suitable for preview article generation.
    
    Args:
        team_data: Raw team data for home team as JSON string
        opponent_data: Raw team data for away team as JSON string
        
    Returns:
        Formatted data for preview article generation as JSON string
    """
    try:
        home_data = json.loads(team_data)
        away_data = json.loads(opponent_data)
        
        home_team_info = home_data.get("response", [{}])[0].get("team_info", {})
        away_team_info = away_data.get("response", [{}])[0].get("team_info", {})
        
        home_response = home_team_info.get("response", [])
        away_response = away_team_info.get("response", [])
        
        if not home_response or not away_response:
            return json.dumps({"error": "Missing team data"}, ensure_ascii=False)
        
        home_team = home_response[0]
        away_team = away_response[0]
        
        # Format for preview article
        formatted_data = {
            "home_team": {
                "name": home_team.get("team", {}).get("name", "Unknown"),
                "country": home_team.get("team", {}).get("country", "Unknown"),
                "founded": home_team.get("team", {}).get("founded", "Unknown"),
                "venue": home_team.get("venue", {}).get("name", "Unknown"),
                "capacity": home_team.get("venue", {}).get("capacity", "Unknown")
            },
            "away_team": {
                "name": away_team.get("team", {}).get("name", "Unknown"),
                "country": away_team.get("team", {}).get("country", "Unknown"),
                "founded": away_team.get("team", {}).get("founded", "Unknown")
            },
            "matchup_info": {
                "home_team_form": "Recent form data would be here",
                "away_team_form": "Recent form data would be here",
                "head_to_head": "H2H data would be here"
            }
        }
        
        return json.dumps(formatted_data, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"Error converting team data: {e}")
        return json.dumps({"error": f"Conversion failed: {str(e)}"}, ensure_ascii=False)


@function_tool
def merge_research_data(research_data: str) -> str:
    """Merge and format research data from multiple sources.
    
    Args:
        research_data: List of research data from researcher agent as JSON string
        
    Returns:
        Merged and formatted research data as JSON string
    """
    try:
        data_list = json.loads(research_data)
        
        merged_data = {
            "context": [],
            "statistics": [],
            "quotes": [],
            "background": [],
            "analysis": [],
            "storylines": []
        }
        
        for data in data_list:
            if isinstance(data, dict):
                # Categorize research data
                if "context" in data:
                    merged_data["context"].append(data["context"])
                if "statistics" in data:
                    merged_data["statistics"].append(data["statistics"])
                if "quotes" in data:
                    merged_data["quotes"].append(data["quotes"])
                if "background" in data:
                    merged_data["background"].append(data["background"])
                if "analysis" in data:
                    merged_data["analysis"].append(data["analysis"])
                if "storylines" in data:
                    merged_data["storylines"].append(data["storylines"])
        
        return json.dumps(merged_data, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"Error merging research data: {e}")
        return json.dumps({"error": f"Merge failed: {str(e)}"}, ensure_ascii=False)


class FormatManager:
    """Agent responsible for data format conversion between pipeline agents."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the Format Manager with configuration."""
        self.config = config
        self.model = config.get("model", "gpt-4")
        self.api_key = config.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
        
        # Initialize the format conversion agent
        self.format_agent = Agent(
            name="FormatManager",
            instructions="""You are a data format conversion specialist. Your job is to convert data between different formats 
            to ensure compatibility between different agents in the sports article generation pipeline.
            
            You have access to several conversion functions:
            
            # Data Collector → Researcher Conversions
            1. format_game_data_for_researcher - Formats game data for researcher input
            2. format_player_data_for_researcher - Formats player data for researcher input  
            3. format_team_data_for_researcher - Formats team data for researcher input
            
            # Researcher → Writer Conversions
            4. format_research_data_for_writer - Formats research data for writer input
            
            # Complete Data → Writer Conversions
            5. format_complete_data_for_writer - Formats combined data collector + research output for writer input
            
            # Specific Format Conversions
            6. convert_game_data_to_recap_format - Converts game data for game recap articles
            7. convert_player_data_to_spotlight_format - Converts player data for player spotlight articles
            8. convert_team_data_to_preview_format - Converts team data for preview articles
            9. merge_research_data - Merges research data from multiple sources
            
            Always use the appropriate conversion function based on the requested format and input data type.
            Provide clear, structured text output that describes the converted data format.
            You are just a format manager, you don't need to do any research or analysis. Just convert the data to the target format and provide clear text output.
            """,
            tools=[
                # Data Collector → Researcher
                format_game_data_for_researcher,
                format_player_data_for_researcher,
                format_team_data_for_researcher,
                
                # Researcher → Writer
                format_research_data_for_writer,
                
                # Complete Data → Writer
                format_complete_data_for_writer,
                
                # Specific Format Conversions
                convert_game_data_to_recap_format,
                convert_player_data_to_spotlight_format,
                convert_team_data_to_preview_format,
                merge_research_data
            ],
            model=self.model
        )
        
        logger.info("Format Manager initialized")

    # ============================================================================
    # Public Interface Methods
    # ============================================================================

    async def prepare_data_for_researcher(
        self, 
        data_collector_output: Dict[str, Any], 
        research_type: str
    ) -> Dict[str, Any]:
        """Prepare formatted data specifically for the researcher agent.
        
        Args:
            data_collector_output: Output from data collector
            research_type: Type of research (team_history, player_performance, season_trends, game_analysis)
            
        Returns:
            Formatted data ready for researcher agent
        """
        try:
            # Convert complex data to simplified string format for better API handling
            simplified_data = self._simplify_data_for_api(data_collector_output)
            
            # Use Agent to handle format conversion
            result = await self.convert_data_format(
                json.dumps(simplified_data, ensure_ascii=False), 
                "researcher_input", 
                json.dumps({"research_type": research_type}, ensure_ascii=False)
            )
            
            # Validate result
            if "error" in result:
                raise Exception(f"Format manager error: {result['error']}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in prepare_data_for_researcher: {e}")
            return {"error": f"Formatting failed: {str(e)}"}

    async def prepare_data_for_writer(
        self, 
        data_collector_output: Dict[str, Any], 
        research_output: Dict[str, Any], 
        article_type: str
    ) -> Dict[str, Any]:
        """Prepare formatted data specifically for the writer agent.
        
        Args:
            data_collector_output: Output from data collector
            research_output: Output from research agent
            article_type: Type of article to generate
            
        Returns:
            Formatted data ready for writer agent with required structure: {data, research, storylines}
        """
        try:
            # Convert complex data to simplified string format for better API handling
            simplified_data = self._simplify_data_for_api(data_collector_output)
            simplified_research = self._simplify_data_for_api(research_output) if research_output else None
            
            # Use Agent to handle format conversion
            result = await self.convert_data_format(
                json.dumps(simplified_data, ensure_ascii=False), 
                "writer_input", 
                json.dumps({
                    "research_output": simplified_research,
                    "article_type": article_type
                }, ensure_ascii=False) if simplified_research else json.dumps({
                    "research_output": None,
                    "article_type": article_type
                }, ensure_ascii=False)
            )
            
            # Validate result structure
            if "error" in result:
                raise Exception(f"Format manager error: {result['error']}")
            
            # Ensure required structure for writer
            if "data" not in result or "research" not in result or "storylines" not in result:
                logger.error(f"Invalid formatted data structure: {result.keys()}")
                raise Exception("Formatted data missing required keys: data, research, or storylines")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in prepare_data_for_writer: {e}")
            return {"error": f"Formatting failed: {str(e)}"}

    async def convert_data_format(
        self, 
        input_data: str, 
        target_format: str,
        additional_data: Optional[str] = None
    ) -> Dict[str, Any]:
        """Convert data to the specified format using the format agent.
        
        Args:
            input_data: Data to convert as JSON string
            target_format: Target format (researcher_input, writer_input, game_recap, etc.)
            additional_data: Additional data needed for conversion as JSON string
            
        Returns:
            Converted data in target format
        """
        try:
            # Create conversion prompt based on target format
            if target_format == "researcher_input":
                research_type = "unknown"
                if additional_data:
                    try:
                        additional_dict = json.loads(additional_data)
                        research_type = additional_dict.get("research_type", "unknown")
                    except json.JSONDecodeError:
                        research_type = "unknown"
                prompt = f"Format the following data for researcher agent input. Data collector output: {input_data}, Research type: {research_type}"
                
            elif target_format == "writer_input":
                research_output = "None"
                article_type = "unknown"
                if additional_data:
                    try:
                        additional_dict = json.loads(additional_data)
                        research_output = additional_dict.get("research_output", "None")
                        article_type = additional_dict.get("article_type", "unknown")
                    except json.JSONDecodeError:
                        pass
                prompt = f"Format the following data for writer agent input. Data collector output: {input_data}, Research output: {research_output}, Article type: {article_type}"
                
            elif target_format == "game_recap":
                prompt = f"Convert the following game data to game recap format: {input_data}"
            elif target_format == "player_spotlight":
                prompt = f"Convert the following player data to player spotlight format: {input_data}"
            elif target_format == "preview_article":
                if additional_data:
                    prompt = f"Convert the following team data to preview article format. Home team: {input_data}, Away team: {additional_data}"
                else:
                    prompt = f"Convert the following team data to preview article format: {input_data}"
            else:
                prompt = f"Convert the following data to {target_format} format: {input_data}"
            
            # Run the format agent
            result = await Runner.run(self.format_agent, prompt)
            
            # Extract the converted data from the result
            if hasattr(result, 'final_output_as'):
                try:
                    final_output = result.final_output_as(str)
                    return self._parse_agent_output(final_output, target_format)
                except (AttributeError, TypeError):
                    if hasattr(result, 'content'):
                        return self._parse_agent_output(result.content, target_format)
                    else:
                        return self._parse_agent_output(str(result), target_format)
            elif hasattr(result, 'content'):
                return self._parse_agent_output(result.content, target_format)
            else:
                return self._parse_agent_output(str(result), target_format)
                
        except Exception as e:
            logger.error(f"Error converting data format: {e}")
            return {"error": f"Conversion failed: {str(e)}"}

    # ============================================================================
    # Helper Methods
    # ============================================================================

    def _simplify_data_for_api(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Simplify complex data structures for better API handling.
        
        Args:
            data: Complex data structure from data collector or researcher
            
        Returns:
            Simplified data structure with key information extracted
        """
        if not data:
            return {}
        
        try:
            # Handle Pydantic models (like PlayerPerformance)
            if hasattr(data, 'model_dump'):
                # Convert Pydantic model to dict
                return data.model_dump()
            elif hasattr(data, 'dict'):
                # Convert Pydantic model to dict (older versions)
                return data.dict()
            
            # Handle API response format
            if "response" in data and isinstance(data["response"], list):
                # Extract the actual data from API response
                response_data = data["response"][0] if data["response"] else {}
                
                # Extract key information based on data type
                if "fixture" in response_data:
                    # Game data
                    fixture = response_data["fixture"]
                    if "response" in fixture and fixture["response"]:
                        fixture_data = fixture["response"][0]
                        return {
                            "game_id": fixture_data.get("fixture", {}).get("id"),
                            "home_team": fixture_data.get("teams", {}).get("home", {}).get("name"),
                            "away_team": fixture_data.get("teams", {}).get("away", {}).get("name"),
                            "home_score": fixture_data.get("goals", {}).get("home"),
                            "away_score": fixture_data.get("goals", {}).get("away"),
                            "date": fixture_data.get("fixture", {}).get("date"),
                            "venue": fixture_data.get("fixture", {}).get("venue", {}).get("name"),
                            "status": fixture_data.get("fixture", {}).get("status", {}).get("long"),
                            "league": fixture_data.get("league", {}).get("name"),
                            "season": fixture_data.get("league", {}).get("season")
                        }
                
                elif "player_info" in response_data:
                    # Player data
                    player_info = response_data["player_info"]
                    if "response" in player_info and player_info["response"]:
                        player_data = player_info["response"][0]
                        return {
                            "player_id": player_data.get("player", {}).get("id"),
                            "player_name": player_data.get("player", {}).get("name"),
                            "age": player_data.get("player", {}).get("age"),
                            "nationality": player_data.get("player", {}).get("nationality"),
                            "position": player_data.get("statistics", [{}])[0].get("games", {}).get("position") if player_data.get("statistics") else None,
                            "team_name": player_data.get("statistics", [{}])[0].get("team", {}).get("name") if player_data.get("statistics") else None,
                            "league": player_data.get("statistics", [{}])[0].get("league", {}).get("name") if player_data.get("statistics") else None
                        }
                    else:
                        # Handle empty player data
                        return {
                            "player_id": "unknown",
                            "player_name": "Unknown Player",
                            "age": None,
                            "nationality": None,
                            "position": None,
                            "team_name": None,
                            "league": None
                        }
                
                elif "team_info" in response_data:
                    # Team data
                    team_info = response_data["team_info"]
                    if "response" in team_info and team_info["response"]:
                        team_data = team_info["response"][0]
                        return {
                            "team_id": team_data.get("team", {}).get("id"),
                            "team_name": team_data.get("team", {}).get("name"),
                            "country": team_data.get("team", {}).get("country"),
                            "founded": team_data.get("team", {}).get("founded"),
                            "venue_name": team_data.get("venue", {}).get("name"),
                            "venue_capacity": team_data.get("venue", {}).get("capacity")
                        }
            
            # If it's already a simplified format, return as is
            return data
            
        except Exception as e:
            logger.warning(f"Error simplifying data for API: {e}")
            # Return original data if simplification fails
            return data

    def _parse_agent_output(self, text_output: str, target_format: str) -> Dict[str, Any]:
        """Parse Agent text output to structured format.
        
        Args:
            text_output: Raw text output from Agent
            target_format: Target format type
            
        Returns:
            Structured data
        """
        try:
            # Try to extract JSON from text if it contains JSON
            import re
            json_match = re.search(r'\{.*\}', text_output, re.DOTALL)
            if json_match:
                try:
                    parsed_json = json.loads(json_match.group())
                    # Validate structure for writer input
                    if target_format == "writer_input":
                        if "data" not in parsed_json or "research" not in parsed_json or "storylines" not in parsed_json:
                            logger.warning("Parsed JSON missing required writer structure, creating default")
                            return self._create_default_writer_structure(parsed_json)
                    return parsed_json
                except json.JSONDecodeError:
                    pass
            
            # If no valid JSON found, create structured format based on text content
            if target_format == "writer_input":
                return self._create_default_writer_structure({"content": text_output})
            elif target_format == "researcher_input":
                return self._parse_researcher_text(text_output)
            else:
                # Default: return as simple text structure
                return {
                    "content": text_output.strip(),
                    "format": target_format,
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error parsing agent output: {e}")
            return {
                "error": f"Parsing failed: {str(e)}",
                "original_text": text_output,
                "target_format": target_format
            }

    def _create_default_writer_structure(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create default writer structure with required keys."""
        return {
            "data": data.get("data", data),
            "research": data.get("research", {}),
            "storylines": data.get("storylines", ["Default storyline"])
        }

    def _parse_researcher_text(self, text: str) -> Dict[str, Any]:
        """Parse researcher text output into structured format."""
        lines = text.strip().split('\n')
        research_data = {
            "key_findings": [],
            "statistics": {},
            "context": "",
            "sources": []
        }
        
        current_section = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if "key findings" in line.lower() or "findings" in line.lower():
                current_section = "findings"
            elif "statistics" in line.lower() or "stats" in line.lower():
                current_section = "statistics"
            elif "context" in line.lower() or "background" in line.lower():
                current_section = "context"
            elif "sources" in line.lower() or "references" in line.lower():
                current_section = "sources"
            elif line.startswith('-') or line.startswith('•'):
                if current_section == "findings":
                    research_data["key_findings"].append(line[1:].strip())
                elif current_section == "sources":
                    research_data["sources"].append(line[1:].strip())
            elif current_section == "context":
                research_data["context"] += line + " "
            elif ":" in line and current_section == "statistics":
                key, value = line.split(":", 1)
                research_data["statistics"][key.strip()] = value.strip()
        
        research_data["context"] = research_data["context"].strip()
        return research_data

    def get_supported_formats(self) -> List[str]:
        """Get list of supported data formats."""
        return [
            "researcher_input",
            "writer_input",
            "game_recap",
            "player_spotlight", 
            "preview_article",
            "merged_research"
        ]

    async def validate_conversion(self, original_data: Dict[str, Any], converted_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that the conversion preserved all necessary data.
        
        Args:
            original_data: Original data before conversion
            converted_data: Data after conversion
            
        Returns:
            Validation results
        """
        validation_result = {
            "is_valid": True,
            "missing_fields": [],
            "warnings": []
        }
        
        try:
            # Check for missing critical fields based on data type
            if "match_info" in original_data:
                # Game data validation
                required_fields = ["home_team", "away_team", "home_score", "away_score"]
                for field in required_fields:
                    if field not in converted_data.get("match_info", {}):
                        validation_result["missing_fields"].append(f"match_info.{field}")
                        validation_result["is_valid"] = False
            
            elif "player_info" in original_data:
                # Player data validation
                required_fields = ["name", "position"]
                for field in required_fields:
                    if field not in converted_data.get("player_info", {}):
                        validation_result["missing_fields"].append(f"player_info.{field}")
                        validation_result["is_valid"] = False
            
            # Check for data loss
            original_keys = set(str(k) for k in self._flatten_dict(original_data))
            converted_keys = set(str(k) for k in self._flatten_dict(converted_data))
            
            lost_keys = original_keys - converted_keys
            if lost_keys:
                validation_result["warnings"].append(f"Some data fields may have been lost: {list(lost_keys)[:5]}")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating conversion: {e}")
            return {"is_valid": False, "error": str(e)}

    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        """Flatten a nested dictionary for easier comparison."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items) 