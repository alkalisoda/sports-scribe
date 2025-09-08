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

        Uses optimized batch invalidation for better performance.

        Args:
            player_id: ID of the player whose cache should be invalidated

        Returns:
            Number of cache entries invalidated
        """
        patterns = [
            f"query:*player_id*{player_id}*",
            f"query:*{player_id}*",
            "query:*player_stat*",
        ]

        # Use batch invalidation for better performance
        total_invalidated = await self.cache.invalidate_patterns_batch(patterns)

        logger.info(
            f"Invalidated {total_invalidated} cache entries for player {player_id}"
        )
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
            "query:*team_stat*",
        ]

        # Use batch invalidation for better performance
        total_invalidated = await self.cache.invalidate_patterns_batch(patterns)

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
            "query:*game_data*",
        ]

        # Use batch invalidation for better performance
        total_invalidated = await self.cache.invalidate_patterns_batch(patterns)

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
        patterns = [f"query:*{season}*", "query:*season*", "query:*current_season*"]

        total_invalidated = 0
        for pattern in patterns:
            invalidated = await self._invalidate_pattern(pattern)
            total_invalidated += invalidated

        logger.info(
            f"Invalidated {total_invalidated} cache entries for season {season}"
        )
        return total_invalidated

    async def invalidate_live_data_cache(self) -> int:
        """
        Invalidate all live/real-time data caches.

        Returns:
            Number of cache entries invalidated
        """
        patterns = ["query:*live*", "query:*current_game*", "query:*real_time*"]

        total_invalidated = 0
        for pattern in patterns:
            invalidated = await self._invalidate_pattern(pattern)
            total_invalidated += invalidated

        logger.info(f"Invalidated {total_invalidated} live data cache entries")
        return total_invalidated

    async def bulk_invalidate(
        self,
        player_ids: Optional[List[str]] = None,
        team_ids: Optional[List[str]] = None,
        game_ids: Optional[List[str]] = None,
        batch_size: int = 100,
    ) -> int:
        """
        Perform optimized bulk invalidation for multiple entities using batching.

        This method is significantly faster than individual invalidations because it:
        - Collects all patterns for all entities at once
        - Uses Redis pipelining for better performance
        - Batches key deletions to avoid Redis limits
        - Eliminates duplicate keys automatically

        Args:
            player_ids: List of player IDs to invalidate
            team_ids: List of team IDs to invalidate
            game_ids: List of game IDs to invalidate
            batch_size: Number of keys to delete per batch (default: 100)

        Returns:
            Total number of cache entries invalidated
        """
        all_patterns = []

        # Collect all patterns for batch processing
        if player_ids:
            for player_id in player_ids:
                all_patterns.extend(
                    [
                        f"query:*player_id*{player_id}*",
                        f"query:*{player_id}*",
                        "query:*player_stat*",
                    ]
                )

        if team_ids:
            for team_id in team_ids:
                all_patterns.extend(
                    [
                        f"query:*team_id*{team_id}*",
                        f"query:*{team_id}*",
                        "query:*team_stat*",
                        "query:*team_data*",
                    ]
                )

        if game_ids:
            for game_id in game_ids:
                all_patterns.extend(
                    [
                        f"query:*game_id*{game_id}*",
                        f"query:*{game_id}*",
                        "query:*game_data*",
                    ]
                )

        if not all_patterns:
            logger.debug("No patterns to invalidate")
            return 0

        # Use optimized batch invalidation
        total_invalidated = await self.cache.invalidate_patterns_batch(
            all_patterns, batch_size
        )

        entity_counts = []
        if player_ids:
            entity_counts.append(f"{len(player_ids)} players")
        if team_ids:
            entity_counts.append(f"{len(team_ids)} teams")
        if game_ids:
            entity_counts.append(f"{len(game_ids)} games")

        logger.info(
            f"🚀 Optimized bulk invalidation completed: {total_invalidated} entries for {', '.join(entity_counts)}"
        )
        return total_invalidated

    async def bulk_invalidate_patterns(
        self, patterns: List[str], batch_size: int = 100
    ) -> int:
        """
        Directly invalidate multiple patterns using optimized batching.

        This is useful for custom invalidation scenarios where you have
        specific patterns to invalidate.

        Args:
            patterns: List of Redis patterns to invalidate
            batch_size: Number of keys to delete per batch

        Returns:
            Number of cache entries invalidated
        """
        return await self.cache.invalidate_patterns_batch(patterns, batch_size)

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
