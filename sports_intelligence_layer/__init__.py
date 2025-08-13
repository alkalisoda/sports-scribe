"""Sports Intelligence Layer package.

Expose the primary public APIs at the top-level so downstream code and tests
can simply do::

    from sports_intelligence_layer import SoccerQueryParser

This avoids fragile relative imports from test modules and makes direct
invocation via `python -m` or pytest discovery more robust.
"""

from .src.query_parser import (  # noqa: F401
    SoccerQueryParser,
    ParsedSoccerQuery,
    SoccerEntity,
    EntityType,
    ComparisonType,
    TimeContext,
)

__all__ = [
    "SoccerQueryParser",
    "ParsedSoccerQuery",
    "SoccerEntity",
    "EntityType",
    "ComparisonType",
    "TimeContext",
]

__version__ = "0.1.0"