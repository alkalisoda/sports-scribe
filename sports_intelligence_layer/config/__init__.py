"""Configuration subpackage for Sports Intelligence Layer.

Expose frequently used configuration enums and models.
"""

from .soccer_entities import (  # noqa: F401
    Position,
    CompetitionType,
    StatisticType,
    PlayerStatistics,
    TeamStatistics,
    Player,
    Team,
    Competition,
    ENTITY_RECOGNITION_CONFIG,
    SOCCER_TERMINOLOGY,
)

__all__ = [
    "Position",
    "CompetitionType",
    "StatisticType",
    "PlayerStatistics",
    "TeamStatistics",
    "Player",
    "Team",
    "Competition",
    "ENTITY_RECOGNITION_CONFIG",
    "SOCCER_TERMINOLOGY",
]