"""Sports Intelligence Layer package.

Expose the primary public APIs at the top-level so downstream code and tests
can simply do::

    from sports_intelligence_layer import SoccerQueryParser, SoccerDatabase

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

from .src.database import SoccerDatabase  # noqa: F401

# Import data management tools for dataset operations
from .dataset_op.database_manager import DatabaseManager  # noqa: F401
from .dataset_op.historical_processor import HistoricalProcessor  # noqa: F401

__all__ = [
    "SoccerQueryParser",
    "ParsedSoccerQuery",
    "SoccerEntity",
    "EntityType",
    "ComparisonType",
    "TimeContext",
    "SoccerDatabase",
    "DatabaseManager",
    "HistoricalProcessor",
]

__version__ = "0.1.0"
