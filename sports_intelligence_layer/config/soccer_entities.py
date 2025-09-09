"""Soccer Entity Definitions and Configuration.

This module defines the core entities, relationships, and configurations for the soccer
intelligence layer. It provides structured data models and validation for soccer-related
data processing.
"""

from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class Position(Enum):
    """Soccer player positions."""
    GOALKEEPER = "GK"
    DEFENDER = "DEF"
    MIDFIELDER = "MID"
    FORWARD = "FWD"
    UNKNOWN = "UNK"


class CompetitionType(Enum):
    """Types of soccer competitions."""
    LEAGUE = "league"
    CUP = "cup"
    INTERNATIONAL = "international"
    FRIENDLY = "friendly"
    API_FOOTBALL = "api-football"


class MatchStatus(Enum):
    """Match status types."""
    SCHEDULED = "scheduled"
    LIVE = "live"
    FINISHED = "Match Finished"
    POSTPONED = "postponed"
    CANCELLED = "cancelled"


class StatisticType(Enum):
    """Types of soccer statistics."""
    GOALS = "goals"
    ASSISTS = "assists"
    MINUTES_PLAYED = "minutes_played"
    PASSES_COMPLETED = "passes_completed"
    PASS_ACCURACY = "pass_accuracy"
    SHOTS_ON_TARGET = "shots_on_target"
    TACKLES = "tackles"
    INTERCEPTIONS = "interceptions"
    CLEAN_SHEETS = "clean_sheets"
    SAVES = "saves"
    YELLOW_CARDS = "yellow_cards"
    RED_CARDS = "red_cards"
    FOULS_COMMITTED = "fouls_committed"
    FOULS_DRAWN = "fouls_drawn"


@dataclass
class PlayerStatistics:
    """Player statistics model with validation."""
    goals: int = 0
    assists: int = 0
    minutes_played: int = 0
    passes_completed: int = 0
    pass_accuracy: float = 0.0
    shots_on_target: int = 0
    tackles: int = 0
    interceptions: int = 0
    clean_sheets: int = 0
    saves: int = 0
    yellow_cards: int = 0
    red_cards: int = 0
    fouls_committed: int = 0
    fouls_drawn: int = 0
    
    def to_dict(self) -> Dict[str, Union[int, float]]:
        """Convert statistics to dictionary."""
        return {
            "goals": self.goals,
            "assists": self.assists,
            "minutes_played": self.minutes_played,
            "passes_completed": self.passes_completed,
            "pass_accuracy": self.pass_accuracy,
            "shots_on_target": self.shots_on_target,
            "tackles": self.tackles,
            "interceptions": self.interceptions,
            "clean_sheets": self.clean_sheets,
            "saves": self.saves,
            "yellow_cards": self.yellow_cards,
            "red_cards": self.red_cards,
            "fouls_committed": self.fouls_committed,
            "fouls_drawn": self.fouls_drawn
        }


@dataclass
class TeamStatistics:
    """Team statistics model with validation."""
    matches_played: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    goals_scored: int = 0
    goals_conceded: int = 0
    clean_sheets: int = 0
    points: int = 0
    possession_avg: float = 0.0
    pass_accuracy_avg: float = 0.0
    shots_per_game: float = 0.0
    
    def to_dict(self) -> Dict[str, Union[int, float]]:
        """Convert statistics to dictionary."""
        return {
            "matches_played": self.matches_played,
            "wins": self.wins,
            "draws": self.draws,
            "losses": self.losses,
            "goals_scored": self.goals_scored,
            "goals_conceded": self.goals_conceded,
            "clean_sheets": self.clean_sheets,
            "points": self.points,
            "possession_avg": self.possession_avg,
            "pass_accuracy_avg": self.pass_accuracy_avg,
            "shots_per_game": self.shots_per_game
        }


@dataclass
class Player:
    """Player entity with comprehensive attributes."""
    id: str
    name: str
    common_name: str
    nationality: str
    birth_date: Optional[datetime] = None
    position: Position = Position.UNKNOWN
    height_cm: Optional[int] = None
    weight_kg: Optional[int] = None
    team_id: Optional[str] = None
    jersey_number: Optional[int] = None
    preferred_foot: Optional[str] = None
    market_value: Optional[float] = None
    statistics: PlayerStatistics = field(default_factory=PlayerStatistics)
    
    def to_dict(self) -> Dict:
        """Convert player to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "common_name": self.common_name,
            "nationality": self.nationality,
            "birth_date": self.birth_date.isoformat() if self.birth_date else None,
            "position": self.position.value,
            "height_cm": self.height_cm,
            "weight_kg": self.weight_kg,
            "team_id": self.team_id,
            "jersey_number": self.jersey_number,
            "preferred_foot": self.preferred_foot,
            "market_value": self.market_value,
            "statistics": self.statistics.to_dict()
        }


@dataclass
class Team:
    """Team entity with comprehensive attributes."""
    id: str
    name: str
    short_name: str
    country: str
    founded_year: Optional[int] = None
    venue_name: Optional[str] = None
    venue_capacity: Optional[int] = None
    coach_name: Optional[str] = None
    logo_url: Optional[str] = None
    primary_color: Optional[str] = None
    secondary_color: Optional[str] = None
    statistics: TeamStatistics = field(default_factory=TeamStatistics)
    
    def to_dict(self) -> Dict:
        """Convert team to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "short_name": self.short_name,
            "country": self.country,
            "founded_year": self.founded_year,
            "venue_name": self.venue_name,
            "venue_capacity": self.venue_capacity,
            "coach_name": self.coach_name,
            "logo_url": self.logo_url,
            "primary_color": self.primary_color,
            "secondary_color": self.secondary_color,
            "statistics": self.statistics.to_dict()
        }


@dataclass
class Competition:
    """Competition/League entity with comprehensive attributes."""
    id: str
    name: str
    short_name: str
    country: str
    type: CompetitionType
    season: str
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    current_matchday: Optional[int] = None
    number_of_matchdays: Optional[int] = None
    number_of_teams: Optional[int] = None
    current_season_id: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert competition to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "short_name": self.short_name,
            "country": self.country,
            "type": self.type.value,
            "season": self.season,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "current_matchday": self.current_matchday,
            "number_of_matchdays": self.number_of_matchdays,
            "number_of_teams": self.number_of_teams,
            "current_season_id": self.current_season_id
        }


@dataclass
class Match:
    """Match/Fixture entity based on current Supabase structure."""
    id: int
    name: str  # Competition name (e.g., "Premier League")
    type: str  # Source type (e.g., "api-football")
    country: str
    season: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    status: Optional[str] = None
    venue_id: Optional[int] = None
    league_id: Optional[int] = None
    home_team_id: Optional[int] = None
    away_team_id: Optional[int] = None
    goals_home: Optional[int] = None
    goals_away: Optional[int] = None
    goals_home_half_time: Optional[int] = None
    goals_away_half_time: Optional[int] = None
    goals_home_extra_time: Optional[int] = None
    goals_away_extra_time: Optional[int] = None
    penalty_home: Optional[int] = None
    penalty_away: Optional[int] = None
    
    def to_dict(self) -> Dict:
        """Convert match to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "country": self.country,
            "season": self.season,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "status": self.status,
            "venue_id": self.venue_id,
            "league_id": self.league_id,
            "home_team_id": self.home_team_id,
            "away_team_id": self.away_team_id,
            "goals_home": self.goals_home,
            "goals_away": self.goals_away,
            "goals_home_half_time": self.goals_home_half_time,
            "goals_away_half_time": self.goals_away_half_time,
            "goals_home_extra_time": self.goals_home_extra_time,
            "goals_away_extra_time": self.goals_away_extra_time,
            "penalty_home": self.penalty_home,
            "penalty_away": self.penalty_away
        }


# Entity Recognition Configuration
ENTITY_RECOGNITION_CONFIG = {
    "player": {
        "min_name_length": 2,
        "max_name_length": 50,
        "confidence_threshold": 0.8,
        "context_boost_words": [
            "scored", "assisted", "saved", "player", "striker", 
            "midfielder", "defender", "goalkeeper", "captain"
        ]
    },
    "team": {
        "min_name_length": 3,
        "max_name_length": 50,
        "confidence_threshold": 0.85,
        "context_boost_words": [
            "club", "team", "side", "squad", "lineup", "XI"
        ]
    },
    "competition": {
        "min_name_length": 3,
        "max_name_length": 100,
        "confidence_threshold": 0.9,
        "context_boost_words": [
            "league", "cup", "tournament", "competition", 
            "championship", "trophy"
        ]
    }
}

# Common soccer terminology and synonyms for natural language processing
SOCCER_TERMINOLOGY = {
    "match_events": {
        "goal": ["goal", "score", "strike", "shot", "header"],
        "assist": ["assist", "pass", "cross", "setup", "created"],
        "save": ["save", "stop", "block", "parry", "denied"],
        "foul": ["foul", "infraction", "violation", "tackle"],
        "card": ["yellow card", "red card", "booking", "sent off"],
        "substitution": ["substitution", "sub", "change", "replacement"],
        "injury": ["injury", "knock", "strain", "hurt", "injured"]
    },
    "positions": {
        "goalkeeper": ["goalkeeper", "keeper", "goalie", "GK"],
        "defender": ["defender", "centre-back", "full-back", "wing-back", "CB", "RB", "LB"],
        "midfielder": ["midfielder", "central midfielder", "CDM", "CAM", "CM"],
        "forward": ["forward", "striker", "winger", "CF", "ST", "LW", "RW"]
    },
    "match_phases": {
        "attack": ["attack", "offensive", "forward play", "pressing"],
        "defense": ["defense", "defensive", "back line", "defending"],
        "transition": ["transition", "counter", "break", "turnover"],
        "possession": ["possession", "control", "keeping the ball"]
    }
}