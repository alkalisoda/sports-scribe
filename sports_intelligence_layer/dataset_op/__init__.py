"""
Dataset Operations Module

This module handles the IMPORT and PROCESSING of historical statistics
from various data sources and populates the historical_records table.

For READING historical data, use SoccerDatabase from src.database module.
"""

from .database_manager import DatabaseManager
from .historical_processor import HistoricalProcessor
from .player_stats_extractor import PlayerStatsExtractor
from .team_stats_extractor import TeamStatsExtractor

__all__ = [
    "DatabaseManager",
    "HistoricalProcessor",
    "PlayerStatsExtractor",
    "TeamStatsExtractor",
]

__version__ = "1.0.0"
__author__ = "SportsScribe Team"