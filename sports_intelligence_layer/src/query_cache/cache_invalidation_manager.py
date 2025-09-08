"""
Cache Invalidation Manager for Sports Intelligence Layer.

Handles intelligent cache invalidation based on data updates and entity relationships.
"""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class CacheInvalidationManager:
    """
    Manages cache invalidation for sports data.
    
    Provides methods to invalidate cached data when underlying entities
    are updated, ensuring data consistency across the system.
    """

    def __init__(self, query_cache):
        """
        Initialize the cache invalidation manager.
        
        Args:
            query_cache: QueryCache instance to manage
        """
        self.cache = query_cache

    async def invalidate_player_cache(self, player_id: str) -> int:
        """
        Invalidate all cached queries related to a specific player.
        
        Args:
            player_id: ID of the player whose cache should be invalidated
            
        Returns:
            Number of cache entries invalidated
        """
        patterns = [
            f"query:*player_id*{player_id}*",
            f"query:*{player_id}*",
            "query:*player_stat*"
        ]
        
        total_invalidated = 0
        for pattern in patterns:
            invalidated = await self._invalidate_pattern(pattern)
            total_invalidated += invalidated
            
        logger.info(f"Invalidated {total_invalidated} cache entries for player {player_id}")
        return total_invalidated

    async def invalidate_team_cache(self, team_id: str) -> int:
        """
        Invalidate all cached queries related to a specific team.
        
        Args:
            team_id: ID of the team whose cache should be invalidated
            
        Returns:
            Number of cache entries invalidated
        """
        patterns = [
            f"query:*team*{team_id}*",
            f"query:*{team_id}*",
            "query:*team_stat*"
        ]
        
        total_invalidated = 0
        for pattern in patterns:
            invalidated = await self._invalidate_pattern(pattern)
            total_invalidated += invalidated
            
        logger.info(f"Invalidated {total_invalidated} cache entries for team {team_id}")
        return total_invalidated

    async def invalidate_game_cache(self, game_id: str) -> int:
        """
        Invalidate cached queries for a specific game.
        
        Args:
            game_id: ID of the game whose cache should be invalidated
            
        Returns:
            Number of cache entries invalidated
        """
        patterns = [
            f"query:*game_id*{game_id}*",
            f"query:*{game_id}*",
            "query:*game_data*"
        ]
        
        total_invalidated = 0
        for pattern in patterns:
            invalidated = await self._invalidate_pattern(pattern)
            total_invalidated += invalidated
            
        logger.info(f"Invalidated {total_invalidated} cache entries for game {game_id}")
        return total_invalidated

    async def invalidate_season_cache(self, season: str) -> int:
        """
        Invalidate cached queries for a specific season.
        
        Args:
            season: Season identifier (e.g., "2024-25")
            
        Returns:
            Number of cache entries invalidated
        """
        patterns = [
            f"query:*{season}*",
            "query:*season*",
            "query:*current_season*"
        ]
        
        total_invalidated = 0
        for pattern in patterns:
            invalidated = await self._invalidate_pattern(pattern)
            total_invalidated += invalidated
            
        logger.info(f"Invalidated {total_invalidated} cache entries for season {season}")
        return total_invalidated

    async def invalidate_live_data_cache(self) -> int:
        """
        Invalidate all live/real-time data caches.
        
        Returns:
            Number of cache entries invalidated
        """
        patterns = [
            "query:*live*",
            "query:*current_game*",
            "query:*real_time*"
        ]
        
        total_invalidated = 0
        for pattern in patterns:
            invalidated = await self._invalidate_pattern(pattern)
            total_invalidated += invalidated
            
        logger.info(f"Invalidated {total_invalidated} live data cache entries")
        return total_invalidated

    async def bulk_invalidate(self, player_ids: Optional[List[str]] = None, team_ids: Optional[List[str]] = None, game_ids: Optional[List[str]] = None) -> int:
        """
        Perform bulk invalidation for multiple entities.
        
        Args:
            player_ids: List of player IDs to invalidate
            team_ids: List of team IDs to invalidate
            game_ids: List of game IDs to invalidate
            
        Returns:
            Total number of cache entries invalidated
        """
        total_invalidated = 0
        
        if player_ids:
            for player_id in player_ids:
                total_invalidated += await self.invalidate_player_cache(player_id)
                
        if team_ids:
            for team_id in team_ids:
                total_invalidated += await self.invalidate_team_cache(team_id)
                
        if game_ids:
            for game_id in game_ids:
                total_invalidated += await self.invalidate_game_cache(game_id)
        
        logger.info(f"Bulk invalidation completed: {total_invalidated} total entries")
        return total_invalidated

    async def _invalidate_pattern(self, pattern: str) -> int:
        """
        Delete all keys matching pattern.
        
        Args:
            pattern: Redis pattern to match
            
        Returns:
            Number of keys deleted
        """
        try:
            return await self.cache.invalidate_pattern(pattern)
        except Exception as e:
            logger.error(f"Error invalidating pattern {pattern}: {e}")
            return 0