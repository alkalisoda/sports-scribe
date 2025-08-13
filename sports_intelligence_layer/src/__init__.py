"""Source package for Sports Intelligence Layer.

Expose commonly used classes at module level so imports are concise:

    from sports_intelligence_layer.src import SoccerQueryParser, SoccerDatabase
"""

from .query_parser import (  # noqa: F401
    SoccerQueryParser,
    ParsedSoccerQuery,
    SoccerEntity,
    EntityType,
    ComparisonType,
    TimeContext,
)
from .database import SoccerDatabase  # noqa: F401

__all__ = [
    "SoccerQueryParser",
    "ParsedSoccerQuery",
    "SoccerEntity",
    "EntityType",
    "ComparisonType",
    "TimeContext",
    "SoccerDatabase",
]