"""
Configuration for Historical Records Processing

Contains settings and constants for data extraction and processing.
"""

from datetime import datetime
from typing import Dict, List

# Record types for historical_records table
RECORD_TYPES = {
    'SEASON_HIGH': 'season_high',           # Best performance in a season
    'CAREER_HIGH': 'career_high',           # Best career performance
    'SEASON_TOTAL': 'season_total',         # Season totals
    'CAREER_TOTAL': 'career_total',         # Career totals
    'MILESTONE': 'milestone',               # Milestones (100 goals, etc.)
    'TEAM_RECORD': 'team_record',           # Team records
    'LEAGUE_RECORD': 'league_record'        # League records
}

# Entity types
ENTITY_TYPES = {
    'PLAYER': 'player',
    'TEAM': 'team',
    'LEAGUE': 'league'
}

# Player statistics to process
PLAYER_STATS = {
    'goals': {
        'name': 'goals',
        'display_name': 'Goals',
        'milestones': [1, 5, 10, 25, 50, 100, 150, 200, 300, 500]  # Goals milestones
    },
    'assists': {
        'name': 'assists',
        'display_name': 'Assists',
        'milestones': [1, 5, 10, 25, 50, 100, 150, 200]  # Assists milestones
    },
    'rating': {
        'name': 'rating',
        'display_name': 'Rating',
        'milestones': [70, 75, 80, 85, 90, 95]  # Rating milestones (if rating is performance-based)
    },
    'appearances': {
        'name': 'appearances',
        'display_name': 'Appearances',
        'milestones': [1, 10, 25, 50, 100, 200, 300, 400, 500]  # Appearances milestones
    }
}

# Team statistics to process
TEAM_STATS = {
    'founded_year': {
        'name': 'founded_year',
        'display_name': 'Founded Year',
        'milestones': []  # No milestones for founded year
    }
}

# Season configurations
CURRENT_SEASON = '2024-25'
SEASONS_TO_PROCESS = ['2023-24', '2024-25']

# Processing settings
BATCH_SIZE = 50  # Number of records to insert per batch
MAX_RETRIES = 3  # Maximum retries for failed operations
ENABLE_MILESTONE_DETECTION = True  # Whether to detect and record milestones
OVERWRITE_EXISTING = False  # Whether to overwrite existing records

# Logging configuration
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Data validation settings
MIN_VALID_GOALS = 0  # Minimum valid goals count
MAX_VALID_GOALS = 1000  # Maximum reasonable goals count
MIN_VALID_ASSISTS = 0  # Minimum valid assists count
MAX_VALID_ASSISTS = 500  # Maximum reasonable assists count
MIN_VALID_RATING = 0  # Minimum valid rating
MAX_VALID_RATING = 100  # Maximum valid rating
MIN_VALID_APPEARANCES = 0  # Minimum valid appearances
MAX_VALID_APPEARANCES = 1000  # Maximum reasonable appearances

def get_milestone_context(stat_name: str, value: float) -> str:
    """Generate context message for milestone achievements."""
    if stat_name == 'goals':
        if value == 1:
            return "First career goal"
        elif value == 100:
            return "Century of goals milestone"
        elif value == 500:
            return "Exceptional 500 goals milestone"
        else:
            return f"{int(value)} goals milestone achieved"

    elif stat_name == 'assists':
        if value == 1:
            return "First career assist"
        elif value == 100:
            return "Century of assists milestone"
        else:
            return f"{int(value)} assists milestone achieved"

    elif stat_name == 'appearances':
        if value == 1:
            return "Professional debut"
        elif value == 100:
            return "Century of appearances milestone"
        elif value == 500:
            return "Exceptional 500 appearances milestone"
        else:
            return f"{int(value)} appearances milestone achieved"

    elif stat_name == 'rating':
        if value >= 90:
            return f"Exceptional rating of {value} achieved"
        elif value >= 85:
            return f"Outstanding rating of {value} achieved"
        else:
            return f"Rating milestone of {value} achieved"

    return f"{stat_name.title()} milestone of {value} achieved"

def get_season_context(season: str, stat_name: str, value: float, record_type: str) -> str:
    """Generate context message for seasonal records."""
    if record_type == RECORD_TYPES['SEASON_HIGH']:
        return f"Best {stat_name} performance in {season} season: {value}"
    elif record_type == RECORD_TYPES['SEASON_TOTAL']:
        return f"Total {stat_name} in {season} season: {value}"
    elif record_type == RECORD_TYPES['CAREER_HIGH']:
        return f"Career best {stat_name}: {value}"
    elif record_type == RECORD_TYPES['CAREER_TOTAL']:
        return f"Career total {stat_name}: {value}"
    else:
        return f"{stat_name.title()}: {value} ({record_type})"

def is_valid_stat_value(stat_name: str, value: float) -> bool:
    """Validate if a statistic value is within reasonable bounds."""
    if stat_name == 'goals':
        return MIN_VALID_GOALS <= value <= MAX_VALID_GOALS
    elif stat_name == 'assists':
        return MIN_VALID_ASSISTS <= value <= MAX_VALID_ASSISTS
    elif stat_name == 'rating':
        return MIN_VALID_RATING <= value <= MAX_VALID_RATING
    elif stat_name == 'appearances':
        return MIN_VALID_APPEARANCES <= value <= MAX_VALID_APPEARANCES
    else:
        return True  # Unknown stats are considered valid

# Database table mappings
TABLE_MAPPINGS = {
    'players': {
        'id_field': 'id',
        'stats_fields': ['goals', 'assists', 'rating', 'appearances'],
        'additional_fields': ['player_firstname', 'player_lastname', 'team_id', 'season_year']
    },
    'teams': {
        'id_field': 'id',
        'stats_fields': ['team_founded'],
        'additional_fields': ['team_name', 'team_code', 'team_country', 'league_id', 'season_year']
    },
    'player_match_stats': {
        'id_field': 'player_id',
        'stats_fields': ['goals', 'assists', 'minutes_played', 'shots', 'passes', 'tackles', 'saves', 'rating'],
        'additional_fields': ['match_id', 'team_id', 'venue']
    }
}