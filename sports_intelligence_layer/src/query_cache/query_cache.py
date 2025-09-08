"""
Query Cache Implementation for Sports Intelligence Layer.

Provides Redis-based caching for database queries with intelligent TTL management
based on query type and data characteristics.
"""

import hashlib
import json
import logging
from typing import Any, Dict, Optional

try:
    import redis.asyncio as redis
    from redis.asyncio import Redis
except ImportError:
    # Fallback for older redis versions or if redis not installed
    try:
        import redis
        from redis import Redis
    except ImportError:
        redis = None
        Redis = None

logger = logging.getLogger(__name__)


class QueryCacheError(Exception):
    """Custom exception for cache operations."""
    pass


class QueryCache:
    """
    Redis-based query cache with intelligent TTL management.
    
    Features:
    - Automatic cache key generation from query + parameters
    - Smart TTL determination based on query content
    - Hit/miss metrics tracking
    - Graceful error handling
    """

    def __init__(self, redis_client: Redis, default_ttl: int = 3600):
        """
        Initialize the query cache.
        
        Args:
            redis_client: Redis async client instance
            default_ttl: Default TTL in seconds (1 hour)
        """
        self.redis = redis_client
        self.default_ttl = default_ttl
        self.cache_hit_counter = "cache_hits"
        self.cache_miss_counter = "cache_misses"

    def _generate_query_hash(self, query: str, params: Dict[str, Any]) -> str:
        """
        Generate consistent hash for query + parameters.
        
        Args:
            query: Query string or identifier
            params: Query parameters dictionary
            
        Returns:
            SHA256 hash string for cache key
        """
        query_string = f"{query}:{json.dumps(params, sort_keys=True)}"
        return hashlib.sha256(query_string.encode()).hexdigest()

    async def get_cached_result(self, query: str, params: Dict[str, Any]) -> Optional[Dict]:
        """
        Retrieve cached query result.
        
        Args:
            query: Query identifier
            params: Query parameters
            
        Returns:
            Cached result dictionary or None if not found
        """
        query_hash = self._generate_query_hash(query, params)
        
        try:
            cached_data = await self.redis.get(f"query:{query_hash}")
            
            if cached_data:
                await self.redis.incr(self.cache_hit_counter)
                return json.loads(cached_data)
            else:
                await self.redis.incr(self.cache_miss_counter)
                return None
                
        except Exception as e:
            logger.warning(f"Cache retrieval error: {e}")
            return None

    async def cache_result(self, query: str, params: Dict[str, Any], result: Dict, ttl: Optional[int] = None) -> None:
        """
        Cache query result with appropriate TTL.
        
        Args:
            query: Query identifier
            params: Query parameters
            result: Result to cache
            ttl: Time-to-live in seconds (auto-determined if None)
        """
        query_hash = self._generate_query_hash(query, params)
        ttl = ttl or self._determine_ttl(query, result)
        
        try:
            await self.redis.setex(
                f"query:{query_hash}",
                ttl,
                json.dumps(result, default=str)
            )
            
            logger.debug(f"Cached query result with TTL {ttl}s: {query_hash[:12]}...")
            
        except Exception as e:
            logger.error(f"Cache storage error: {e}")

    def _determine_ttl(self, query: str, result: Dict) -> int:
        """
        Determine appropriate TTL based on query type and data freshness.
        
        Args:
            query: Query string to analyze
            result: Query result to analyze
            
        Returns:
            TTL in seconds
        """
        query_lower = query.lower()
        
        if "live" in query_lower or "current_game" in query_lower:
            return 60  # 1 minute for live data
            
        elif "season" in query_lower and "2024-25" in query:
            return 1800  # 30 minutes for current season
            
        elif "career" in query_lower or "historical" in query_lower:
            return 86400  # 24 hours for historical data
            
        elif "goals" in query_lower or "assists" in query_lower:
            return 900  # 15 minutes for player stats
            
        else:
            return self.default_ttl

    async def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate cache entries matching a pattern.
        
        Args:
            pattern: Redis pattern to match (e.g., "query:player_*")
            
        Returns:
            Number of keys deleted
        """
        try:
            keys = await self.redis.keys(pattern)
            if keys:
                deleted = await self.redis.delete(*keys)
                logger.info(f"Invalidated {deleted} cache entries matching: {pattern}")
                return deleted
            return 0
            
        except Exception as e:
            logger.error(f"Cache invalidation error: {e}")
            return 0

    async def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache performance statistics.
        
        Returns:
            Dictionary with hit/miss counts and ratios
        """
        try:
            hits = await self.redis.get(self.cache_hit_counter) or 0
            misses = await self.redis.get(self.cache_miss_counter) or 0
            
            hits = int(hits)
            misses = int(misses)
            total = hits + misses
            
            return {
                "hits": hits,
                "misses": misses,
                "total_requests": total,
                "hit_ratio": hits / total if total > 0 else 0,
                "miss_ratio": misses / total if total > 0 else 0,
            }
            
        except Exception as e:
            logger.error(f"Error fetching cache stats: {e}")
            return {
                "hits": 0,
                "misses": 0,
                "total_requests": 0,
                "hit_ratio": 0,
                "miss_ratio": 0,
                "error": str(e)
            }

    async def clear_cache(self) -> bool:
        """
        Clear all cached query results.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            await self.invalidate_pattern("query:*")
            await self.redis.delete(self.cache_hit_counter, self.cache_miss_counter)
            logger.info("Cache cleared successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False

    async def close(self) -> None:
        """Close Redis connection."""
        try:
            await self.redis.close()
        except Exception as e:
            logger.error(f"Error closing Redis connection: {e}")


def create_query_cache(redis_host: str = "localhost", redis_port: int = 6379, redis_db: int = 0, redis_password: Optional[str] = None, default_ttl: int = 3600) -> QueryCache:
    """
    Create a QueryCache instance with Redis connection.
    
    Args:
        redis_host: Redis server host
        redis_port: Redis server port
        redis_db: Redis database number
        redis_password: Redis password (if required)
        default_ttl: Default TTL in seconds
        
    Returns:
        QueryCache instance
    """
    redis_client = redis.Redis(
        host=redis_host,
        port=redis_port,
        db=redis_db,
        password=redis_password,
        decode_responses=True
    )
    
    return QueryCache(redis_client, default_ttl)