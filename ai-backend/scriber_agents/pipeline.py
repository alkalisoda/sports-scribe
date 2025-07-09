"""
Pipeline Orchestrator.

This module coordinates the flow between different agents in the SportsScribe pipeline:
Data Collector → Researcher → Writer
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional, List

from .data_collector import DataCollectorAgent
from .researcher import ResearchAgent
from .writer import WritingAgent
from openai import AsyncOpenAI

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)


class ArticlePipeline:
    """Orchestrates the complete article generation pipeline."""

    def __init__(self):
        """Initialize the pipeline using environment variables."""
        # Get configuration from environment variables
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.rapidapi_key = os.getenv("RAPIDAPI_KEY")
        self.rapidapi_host = os.getenv("RAPIDAPI_HOST", "api-football-v1.p.rapidapi.com")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4")
        self.temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
        self.max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "2000"))
        
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        if not self.rapidapi_key:
            raise ValueError("RAPIDAPI_KEY environment variable is required")
        
        # Create config dict for agents
        config = {
            "openai_api_key": self.openai_api_key,
            "rapidapi_key": self.rapidapi_key,
            "rapidapi_host": self.rapidapi_host,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        self.openai_client = AsyncOpenAI(api_key=self.openai_api_key)
        
        # Initialize all agents with config only (do not pass openai_client)
        self.collector = DataCollectorAgent(config)
        self.researcher = ResearchAgent(config)
        self.writer = WritingAgent(config)
        
        logger.info("Article Pipeline initialized with environment variables")

    async def generate_game_recap(self, game_id: str) -> Dict[str, Any]:
        """Generate a complete game recap article.
        
        Pipeline: Data Collection → Research → Storyline Generation → Content Writing
        """
        pipeline_start_time = datetime.now()
        logger.info(f"[PIPELINE] Starting game recap generation for game: {game_id}")
        
        try:
            # Step 1: Data Collection
            logger.info(f"[PIPELINE-COLLECTOR] Step 1: Data Collection for game {game_id}")
            game_data = await self._collect_game_data(game_id)
            logger.info(f"[PIPELINE-COLLECTOR] Game data collected successfully for game {game_id}")
            
            team_data = await self._collect_team_data(game_data)
            logger.info(f"[PIPELINE-COLLECTOR] Team data collected successfully for game {game_id}")
            
            # Step 2: Research & Context
            logger.info(f"[PIPELINE-RESEARCHER] Step 2: Research & Context for game {game_id}")
            research_data = await self._research_game_context(game_data, team_data)
            logger.info(f"[PIPELINE-RESEARCHER] Research completed successfully for game {game_id}")
            
            # Step 3: Storyline Generation
            logger.info(f"[PIPELINE-RESEARCHER] Step 3: Storyline Generation for game {game_id}")
            data_list = [game_data]
            if team_data.get("home_team"):
                data_list.append(team_data["home_team"])
                logger.debug(f"[PIPELINE-RESEARCHER] Added home team data to storyline generation")
            if team_data.get("away_team"):
                data_list.append(team_data["away_team"])
                logger.debug(f"[PIPELINE-RESEARCHER] Added away team data to storyline generation")
            
            storylines = await self._generate_storylines(data_list)
            logger.info(f"[PIPELINE-RESEARCHER] Generated {len(storylines)} storylines for game {game_id}")
            logger.info(f"[PIPELINE-RESEARCHER] Storylines for game {game_id}: {storylines}")
            logger.debug(f"[PIPELINE-RESEARCHER] Storylines: {storylines[:3]}...")  # Log first 3 storylines
            
            # Step 4: Content Generation
            logger.info(f"[PIPELINE-WRITER] Step 4: Content Generation for game {game_id}")
            article_content = await self.writer.generate_game_recap(
                game_data, research_data, storylines
            )
            logger.info(f"[PIPELINE-WRITER] Article content generated successfully for game {game_id}")
            logger.debug(f"[PIPELINE-WRITER] Article length: {len(article_content)} characters")
            
            # Step 5: Return Results
            logger.info(f"[PIPELINE] Step 5: Formatting results for game {game_id}")
            result = self._format_result(
                content=article_content,
                metadata={
                    "game_id": game_id,
                    "article_type": "recap",
                    "source_data": game_data,
                    "storylines": storylines,
                    "generated_at": datetime.now().isoformat()
                }
            )
            
            pipeline_duration = (datetime.now() - pipeline_start_time).total_seconds()
            logger.info(f"[PIPELINE] Game recap generation completed successfully for game {game_id} in {pipeline_duration:.2f} seconds")
            
            return result
            
        except Exception as e:
            pipeline_duration = (datetime.now() - pipeline_start_time).total_seconds()
            logger.error(f"[PIPELINE] Error generating game recap for {game_id} after {pipeline_duration:.2f} seconds: {str(e)}")
            raise

    async def generate_preview_article(self, game_id: str) -> Dict[str, Any]:
        """Generate a game preview article.
        
        Pipeline: Data Collection → Research → Storyline Generation → Content Writing
        """
        pipeline_start_time = datetime.now()
        logger.info(f"[PIPELINE] Starting preview generation for game: {game_id}")
        
        try:
            # Step 1: Data Collection
            logger.info(f"[PIPELINE-COLLECTOR] Step 1: Data Collection for preview game {game_id}")
            game_data = await self._collect_game_data(game_id)
            logger.info(f"[PIPELINE-COLLECTOR] Game data collected successfully for preview game {game_id}")
            
            # Step 2: Research & Context
            logger.info(f"[PIPELINE-RESEARCHER] Step 2: Research & Context for preview game {game_id}")
            research_data = await self._research_game_context(game_data)
            logger.info(f"[PIPELINE-RESEARCHER] Research completed successfully for preview game {game_id}")
            
            # Step 3: Storyline Generation
            logger.info(f"[PIPELINE-RESEARCHER] Step 3: Storyline Generation for preview game {game_id}")
            storylines = await self._generate_storylines([game_data])
            logger.info(f"[PIPELINE-RESEARCHER] Generated {len(storylines)} storylines for preview game {game_id}")
            logger.info(f"[PIPELINE-RESEARCHER] Preview storylines for game {game_id}: {storylines}")
            logger.debug(f"[PIPELINE-RESEARCHER] Preview storylines: {storylines[:3]}...")  # Log first 3 storylines
            
            # Step 4: Content Generation
            logger.info(f"[PIPELINE-WRITER] Step 4: Content Generation for preview game {game_id}")
            article_content = await self.writer.generate_preview_article(
                game_data, research_data, storylines
            )
            logger.info(f"[PIPELINE-WRITER] Preview article content generated successfully for game {game_id}")
            logger.debug(f"[PIPELINE-WRITER] Preview article length: {len(article_content)} characters")
            
            # Step 5: Return Results
            logger.info(f"[PIPELINE] Step 5: Formatting preview results for game {game_id}")
            result = self._format_result(
                content=article_content,
                metadata={
                    "game_id": game_id,
                    "article_type": "preview",
                    "source_data": game_data,
                    "storylines": storylines,
                    "generated_at": datetime.now().isoformat()
                }
            )
            
            pipeline_duration = (datetime.now() - pipeline_start_time).total_seconds()
            logger.info(f"[PIPELINE] Preview generation completed successfully for game {game_id} in {pipeline_duration:.2f} seconds")
            
            return result
            
        except Exception as e:
            pipeline_duration = (datetime.now() - pipeline_start_time).total_seconds()
            logger.error(f"[PIPELINE] Error generating preview for {game_id} after {pipeline_duration:.2f} seconds: {str(e)}")
            raise

    async def generate_player_spotlight(self, player_id: str, game_id: Optional[str] = None) -> Dict[str, Any]:
        """Generate a player spotlight article.
        
        Pipeline: Data Collection → Research → Storyline Generation → Content Writing
        """
        pipeline_start_time = datetime.now()
        context_info = f"player {player_id}"
        if game_id:
            context_info += f" in game {game_id}"
        logger.info(f"[PIPELINE] Starting player spotlight generation for {context_info}")
        
        try:
            # Step 1: Data Collection
            logger.info(f"[PIPELINE-COLLECTOR] Step 1: Data Collection for player {player_id}")
            player_data = await self._collect_player_data(player_id)
            logger.info(f"[PIPELINE-COLLECTOR] Player data collected successfully for player {player_id}")
            
            # Step 2: Research & Context
            logger.info(f"[PIPELINE-RESEARCHER] Step 2: Research & Context for player {player_id}")
            performance_data = await self._research_player_performance(player_id, game_id)
            logger.info(f"[PIPELINE-RESEARCHER] Research completed successfully for player {player_id}")
            
            # Step 3: Storyline Generation
            logger.info(f"[PIPELINE-RESEARCHER] Step 3: Storyline Generation for player {player_id}")
            storylines = await self._generate_storylines([player_data])
            logger.info(f"[PIPELINE-RESEARCHER] Generated {len(storylines)} storylines for player {player_id}")
            logger.info(f"[PIPELINE-RESEARCHER] Player storylines for player {player_id}: {storylines}")
            logger.debug(f"[PIPELINE-RESEARCHER] Player storylines: {storylines[:3]}...")  # Log first 3 storylines
            
            # Step 4: Content Generation
            logger.info(f"[PIPELINE-WRITER] Step 4: Content Generation for player {player_id}")
            article_content = await self.writer.generate_player_spotlight(
                player_data, performance_data, storylines
            )
            logger.info(f"[PIPELINE-WRITER] Player spotlight content generated successfully for player {player_id}")
            logger.debug(f"[PIPELINE-WRITER] Player spotlight length: {len(article_content)} characters")
            
            # Step 5: Return Results
            logger.info(f"[PIPELINE] Step 5: Formatting player spotlight results for player {player_id}")
            result = self._format_result(
                content=article_content,
                metadata={
                    "player_id": player_id,
                    "game_id": game_id,
                    "article_type": "spotlight",
                    "source_data": player_data,
                    "storylines": storylines,
                    "generated_at": datetime.now().isoformat()
                }
            )
            
            pipeline_duration = (datetime.now() - pipeline_start_time).total_seconds()
            logger.info(f"[PIPELINE] Player spotlight generation completed successfully for {context_info} in {pipeline_duration:.2f} seconds")
            
            return result
            
        except Exception as e:
            pipeline_duration = (datetime.now() - pipeline_start_time).total_seconds()
            logger.error(f"[PIPELINE] Error generating player spotlight for {context_info} after {pipeline_duration:.2f} seconds: {str(e)}")
            raise

    # Helper methods for data collection
    async def _collect_game_data(self, game_id: str) -> Dict[str, Any]:
        """Collect game data and validate response."""
        logger.debug(f"[HELPER-COLLECTOR] Collecting game data for game {game_id}")
        
        try:
            game_data = await self.collector.collect_game_data(game_id)
            logger.info(f"[HELPER-COLLECTOR] Raw game data for {game_id}: {repr(game_data)[:1000]}")
        except Exception as e:
            error_msg = f"Failed to collect game data for game {game_id}: {str(e)}"
            logger.error(f"[HELPER-COLLECTOR] {error_msg}")
            raise ValueError(error_msg) from e
        
        if not game_data:
            error_msg = f"No data returned for game {game_id}"
            logger.error(f"[HELPER-COLLECTOR] {error_msg}")
            raise ValueError(error_msg)
        
        if game_data.get("errors"):
            error_msg = f"API errors for game {game_id}: {game_data.get('errors', [])}"
            logger.error(f"[HELPER-COLLECTOR] {error_msg}")
            raise ValueError(error_msg)
        
        logger.debug(f"[HELPER-COLLECTOR] Game data collected successfully for game {game_id}")
        return game_data

    async def _collect_team_data(self, game_data: Dict[str, Any]) -> Dict[str, Any]:
        """Collect team data for both teams in the game."""
        logger.debug(f"[HELPER-COLLECTOR] Extracting team data from game data")
        
        try:
            fixture_response = game_data.get("response", [{}])[0].get("fixture", {}).get("response", [])
            if not fixture_response:
                logger.warning(f"[HELPER-COLLECTOR] No fixture response found in game data")
                return {"home_team": None, "away_team": None}
            
            fixture = fixture_response[0]
            home_team_id = fixture.get("teams", {}).get("home", {}).get("id")
            away_team_id = fixture.get("teams", {}).get("away", {}).get("id")
            
            logger.debug(f"[HELPER-COLLECTOR] Extracted team IDs - Home: {home_team_id}, Away: {away_team_id}")
            
            # Collect home team data
            home_team_data = None
            if home_team_id:
                logger.debug(f"[HELPER-COLLECTOR] Collecting home team data for team {home_team_id}")
                try:
                    home_team_data = await self.collector.collect_team_data(str(home_team_id))
                    logger.debug(f"[HELPER-COLLECTOR] Home team data collected for team {home_team_id}")
                except Exception as e:
                    logger.warning(f"[HELPER-COLLECTOR] Failed to collect home team data for team {home_team_id}: {str(e)}")
                    home_team_data = None
            else:
                logger.warning(f"[HELPER-COLLECTOR] No home team ID found")
            
            # Collect away team data
            away_team_data = None
            if away_team_id:
                logger.debug(f"[HELPER-COLLECTOR] Collecting away team data for team {away_team_id}")
                try:
                    away_team_data = await self.collector.collect_team_data(str(away_team_id))
                    logger.debug(f"[HELPER-COLLECTOR] Away team data collected for team {away_team_id}")
                except Exception as e:
                    logger.warning(f"[HELPER-COLLECTOR] Failed to collect away team data for team {away_team_id}: {str(e)}")
                    away_team_data = None
            else:
                logger.warning(f"[HELPER-COLLECTOR] No away team ID found")
            
            result = {
                "home_team": home_team_data,
                "away_team": away_team_data,
                "home_team_id": home_team_id,
                "away_team_id": away_team_id,
                "fixture": fixture
            }
            
            logger.info(f"[HELPER-COLLECTOR] Raw team data: {repr(result)[:1000]}")
            logger.debug(f"[HELPER-COLLECTOR] Team data collection completed - Home: {home_team_id}, Away: {away_team_id}")
            return result
            
        except Exception as e:
            error_msg = f"Failed to collect team data: {str(e)}"
            logger.error(f"[HELPER-COLLECTOR] {error_msg}")
            raise ValueError(error_msg) from e

    async def _collect_player_data(self, player_id: str) -> Dict[str, Any]:
        """Collect player data and validate response."""
        logger.debug(f"[HELPER-COLLECTOR] Collecting player data for player {player_id}")
        
        try:
            player_data = await self.collector.collect_player_data(player_id)
            logger.info(f"[HELPER-COLLECTOR] Raw player data for {player_id}: {repr(player_data)[:1000]}")
        except Exception as e:
            error_msg = f"Failed to collect player data for player {player_id}: {str(e)}"
            logger.error(f"[HELPER-COLLECTOR] {error_msg}")
            raise ValueError(error_msg) from e
        
        if not player_data:
            error_msg = f"No data returned for player {player_id}"
            logger.error(f"[HELPER-COLLECTOR] {error_msg}")
            raise ValueError(error_msg)
        
        if player_data.get("errors"):
            error_msg = f"API errors for player {player_id}: {player_data.get('errors', [])}"
            logger.error(f"[HELPER-COLLECTOR] {error_msg}")
            raise ValueError(error_msg)
        
        logger.debug(f"[HELPER-COLLECTOR] Player data collected successfully for player {player_id}")
        return player_data

    # Helper methods for research
    async def _research_game_context(self, game_data: Dict[str, Any], team_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Research contextual information for the game."""
        logger.debug(f"[HELPER-RESEARCHER] Starting game context research")
        research_data = {}
        
        try:
            # Extract team IDs and fixture info
            fixture_response = game_data.get("response", [{}])[0].get("fixture", {}).get("response", [])
            if fixture_response:
                fixture = fixture_response[0]
                home_team_id = fixture.get("teams", {}).get("home", {}).get("id")
                away_team_id = fixture.get("teams", {}).get("away", {}).get("id")
                league_id = fixture.get("league", {}).get("id")
                season = fixture.get("league", {}).get("season")
                
                logger.debug(f"[HELPER-RESEARCHER] Extracted context info - Home: {home_team_id}, Away: {away_team_id}, League: {league_id}, Season: {season}")
                
                # Research team history
                if home_team_id and away_team_id:
                    logger.debug(f"[HELPER-RESEARCHER] Researching team history for teams {home_team_id} vs {away_team_id}")
                    try:
                        team_history = await self.researcher.research_team_history(
                            str(home_team_id), str(away_team_id)
                        )
                        research_data["team_history"] = team_history
                        logger.debug(f"[HELPER-RESEARCHER] Team history research completed")
                    except Exception as e:
                        logger.warning(f"[HELPER-RESEARCHER] Failed to research team history: {str(e)}")
                        research_data["team_history"] = None
                else:
                    logger.warning(f"[HELPER-RESEARCHER] Missing team IDs for history research - Home: {home_team_id}, Away: {away_team_id}")
                
                # Research season trends
                if league_id and season:
                    logger.debug(f"[HELPER-RESEARCHER] Researching season trends for league {league_id}, season {season}")
                    try:
                        season_trends = await self.researcher.research_season_trends(
                            str(league_id), str(season)
                        )
                        research_data["season_trends"] = season_trends
                        logger.debug(f"[HELPER-RESEARCHER] Season trends research completed")
                    except Exception as e:
                        logger.warning(f"[HELPER-RESEARCHER] Failed to research season trends: {str(e)}")
                        research_data["season_trends"] = None
                else:
                    logger.warning(f"[HELPER-RESEARCHER] Missing league/season info for trends research - League: {league_id}, Season: {season}")
            else:
                logger.warning(f"[HELPER-RESEARCHER] No fixture response found for context research")
            
            logger.info(f"[HELPER-RESEARCHER] Raw research context data: {repr(research_data)[:1000]}")
            logger.debug(f"[HELPER-RESEARCHER] Game context research completed with {len(research_data)} data sources")
            return research_data
            
        except Exception as e:
            error_msg = f"Failed to research game context: {str(e)}"
            logger.error(f"[HELPER-RESEARCHER] {error_msg}")
            raise ValueError(error_msg) from e

    async def _research_player_performance(self, player_id: str, game_id: Optional[str] = None) -> Dict[str, Any]:
        """Research player performance data."""
        context_info = f"player {player_id}"
        if game_id:
            context_info += f" in game {game_id}"
        
        logger.debug(f"[HELPER-RESEARCHER] Researching player performance for {context_info}")
        context = {"game_id": game_id} if game_id else {}
        
        try:
            performance_data = await self.researcher.research_player_performance(player_id, context)
            logger.info(f"[HELPER-RESEARCHER] Raw player performance data for {player_id}: {repr(performance_data)[:1000]}")
            logger.debug(f"[HELPER-RESEARCHER] Player performance research completed for {context_info}")
            return performance_data
        except Exception as e:
            error_msg = f"Failed to research player performance for {context_info}: {str(e)}"
            logger.error(f"[HELPER-RESEARCHER] {error_msg}")
            raise ValueError(error_msg) from e

    # Helper methods for storyline generation
    async def _generate_storylines(self, data_list: List[Dict[str, Any]]) -> List[str]:
        """Generate storylines from collected data."""
        logger.debug(f"[HELPER-RESEARCHER] Generating storylines from {len(data_list)} data sources")
        
        try:
            storylines = await self.researcher.generate_storylines(data_list)
            
            if not storylines:
                logger.warning(f"[HELPER-RESEARCHER] No storylines generated from {len(data_list)} data sources")
                return []
            
            logger.info(f"[HELPER-RESEARCHER] Generated {len(storylines)} storylines: {storylines}")
            logger.debug(f"[HELPER-RESEARCHER] Generated {len(storylines)} storylines")
            return storylines
            
        except Exception as e:
            error_msg = f"Failed to generate storylines: {str(e)}"
            logger.error(f"[HELPER-RESEARCHER] {error_msg}")
            raise ValueError(error_msg) from e

    # Helper methods for result formatting
    def _format_result(self, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Format the final result with content and metadata."""
        logger.debug(f"[HELPER] Formatting result with content length {len(content)} and metadata keys: {list(metadata.keys())}")
        
        try:
            # Validate content
            if not content:
                error_msg = "Content cannot be empty"
                logger.error(f"[HELPER] {error_msg}")
                raise ValueError(error_msg)
            
            # Validate metadata
            if not metadata:
                error_msg = "Metadata cannot be empty"
                logger.error(f"[HELPER] {error_msg}")
                raise ValueError(error_msg)
            
            result = {
                "content": content,
                "metadata": {
                    **metadata,
                    "pipeline_version": "1.0.0",
                    "formatted_at": datetime.now().isoformat()
                }
            }
            
            logger.debug(f"[HELPER] Result formatting completed")
            return result
            
        except Exception as e:
            error_msg = f"Failed to format result: {str(e)}"
            logger.error(f"[HELPER] {error_msg}")
            raise ValueError(error_msg) from e

    async def get_pipeline_status(self) -> Dict[str, Any]:
        """Get the current status of all agents in the pipeline."""
        logger.debug(f"[PIPELINE] Getting pipeline status")
        
        status = {
            "pipeline_version": "1.0.0",
            "agents": {
                "data_collector": "initialized",
                "researcher": "initialized", 
                "writer": "initialized"
            },
            "last_updated": datetime.now().isoformat()
        }
        
        logger.debug(f"[PIPELINE] Pipeline status retrieved successfully")
        return status 