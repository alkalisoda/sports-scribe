"""
Query Cache Implementation for Sports Intelligence Layer.

Provides Redis-based caching for database queries with intelligent TTL management
based on query type and data characteristics.
"""

import hashlib
import json
import logging
from typing import Any, Dict, List, Optional
import types

try:
    import redis.asyncio as redis_async
    from redis.asyncio import Redis as AsyncRedis

    REDIS_AVAILABLE = True
    redis_module: types.ModuleType = redis_async
    RedisClient: Any = AsyncRedis
except ImportError:
    # Fallback for older redis versions or if redis not installed
    try:
        import redis as redis_sync
        from redis import Redis as SyncRedis

        REDIS_AVAILABLE = True
        redis_module = redis_sync
        RedisClient = SyncRedis
    except ImportError:
        redis_module = None  # type: ignore
        RedisClient = None  # type: ignore
        REDIS_AVAILABLE = False

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

    def __init__(self, redis_client: Any, default_ttl: int = 3600):
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
        self._connection_pool: Optional[Any] = None

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

    async def get_cached_result(
        self, query: str, params: Dict[str, Any]
    ) -> Optional[Dict]:
        """
        Retrieve cached query result with atomic counter updates.

        Uses Redis pipeline for atomic operations to ensure accurate
        metrics under high concurrency conditions.

        Args:
            query: Query identifier
            params: Query parameters

        Returns:
            Cached result dictionary or None if not found
        """
        query_hash = self._generate_query_hash(query, params)
        cache_key = f"query:{query_hash}"

        try:
            # Use pipeline for atomic operations
            async with self.redis.pipeline() as pipe:
                # Execute get and counter increment atomically
                pipe.get(cache_key)
                results = await pipe.execute()
                cached_data = results[0]

                # Update metrics atomically based on result
                if cached_data:
                    # Cache hit - increment hit counter atomically
                    async with self.redis.pipeline() as metrics_pipe:
                        metrics_pipe.incr(self.cache_hit_counter)
                        await metrics_pipe.execute()

                    try:
                        return json.loads(cached_data)
                    except (json.JSONDecodeError, TypeError) as e:
                        logger.warning(f"Cache data corruption detected: {e}")
                        # Increment miss counter since we can't use the data
                        async with self.redis.pipeline() as miss_pipe:
                            miss_pipe.incr(self.cache_miss_counter)
                            # Also remove corrupted data
                            miss_pipe.delete(cache_key)
                            await miss_pipe.execute()
                        return None
                else:
                    # Cache miss - increment miss counter atomically
                    async with self.redis.pipeline() as metrics_pipe:
                        metrics_pipe.incr(self.cache_miss_counter)
                        await metrics_pipe.execute()
                    return None

        except Exception as e:
            logger.warning(f"Cache retrieval error: {e}")
            # In case of error, still update miss counter to maintain metrics
            try:
                async with self.redis.pipeline() as error_pipe:
                    error_pipe.incr(self.cache_miss_counter)
                    await error_pipe.execute()
            except Exception:
                pass  # Don't fail on metrics update failure
            return None

    async def cache_result(
        self,
        query: str,
        params: Dict[str, Any],
        result: Dict,
        ttl: Optional[int] = None,
    ) -> None:
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
                f"query:{query_hash}", ttl, json.dumps(result, default=str)
            )

            logger.debug(f"Cached query result with TTL {ttl}s: {query_hash[:12]}...")

        except Exception as e:
            logger.error(f"Cache storage error: {e}")

    async def get_and_increment_atomic(
        self, cache_key: str
    ) -> tuple[Optional[str], bool]:
        """
        Atomically get cache value and increment appropriate counter.

        This is a more efficient version that combines the get operation
        with the counter increment in a single atomic transaction.

        Args:
            cache_key: Redis key to retrieve

        Returns:
            Tuple of (cached_data, was_hit)
        """
        try:
            # Use a Lua script for true atomicity
            lua_script = """
            local cache_key = KEYS[1]
            local hit_counter = KEYS[2]
            local miss_counter = KEYS[3]

            local cached_data = redis.call('GET', cache_key)

            if cached_data then
                redis.call('INCR', hit_counter)
                return {cached_data, 1}
            else
                redis.call('INCR', miss_counter)
                return {false, 0}
            end
            """

            result = await self.redis.eval(
                lua_script,
                3,  # Number of keys
                cache_key,
                self.cache_hit_counter,
                self.cache_miss_counter,
            )

            cached_data = result[0] if result[0] != 0 else None
            was_hit = bool(result[1])

            return cached_data, was_hit

        except Exception as e:
            logger.warning(f"Atomic cache operation failed: {e}")
            return None, False

    async def get_cached_result_atomic(
        self, query: str, params: Dict[str, Any]
    ) -> Optional[Dict]:
        """
        Enhanced atomic version using Lua script for maximum efficiency.

        This version uses a single Redis operation with Lua script to ensure
        true atomicity between cache retrieval and metrics update.

        Args:
            query: Query identifier
            params: Query parameters

        Returns:
            Cached result dictionary or None if not found
        """
        query_hash = self._generate_query_hash(query, params)
        cache_key = f"query:{query_hash}"

        try:
            cached_data, was_hit = await self.get_and_increment_atomic(cache_key)

            if was_hit and cached_data:
                try:
                    return json.loads(cached_data)
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"Cache data corruption detected: {e}")
                    # Clean up corrupted data
                    try:
                        await self.redis.delete(cache_key)
                    except Exception:
                        pass
                    return None

            return None

        except Exception as e:
            logger.warning(f"Atomic cache retrieval error: {e}")
            return None

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

    async def invalidate_patterns_batch(
        self, patterns: List[str], batch_size: int = 100
    ) -> int:
        """
        Efficiently invalidate multiple patterns using batching and pipelining.

        This method optimizes bulk invalidation by:
        - Collecting all keys from multiple patterns
        - Batching key deletions to avoid Redis command limits
        - Using Redis pipeline for better performance

        Args:
            patterns: List of Redis patterns to match
            batch_size: Number of keys to delete per batch (default: 100)

        Returns:
            Total number of keys deleted
        """
        if not patterns:
            return 0

        try:
            # Step 1: Collect all keys from all patterns in parallel
            all_keys = set()  # Use set to avoid duplicates

            # Use pipeline for key collection
            pipe = self.redis.pipeline()
            for pattern in patterns:
                pipe.keys(pattern)

            pattern_results = await pipe.execute()

            # Combine all keys
            for keys_list in pattern_results:
                if keys_list:
                    all_keys.update(keys_list)

            if not all_keys:
                logger.debug("No keys found for patterns")
                return 0

            total_deleted = 0
            keys_list = list(all_keys)

            # Step 2: Delete keys in batches using pipeline
            for i in range(0, len(keys_list), batch_size):
                batch_keys = keys_list[i : i + batch_size]

                pipe = self.redis.pipeline()
                pipe.delete(*batch_keys)
                results = await pipe.execute()

                batch_deleted = sum(results) if results else 0
                total_deleted += batch_deleted

                logger.debug(
                    f"Batch {i // batch_size + 1}: deleted {batch_deleted} keys"
                )

            logger.info(
                f"✅ Batch invalidation completed: {total_deleted} keys deleted from {len(patterns)} patterns"
            )
            return total_deleted

        except Exception as e:
            logger.error(f"❌ Batch invalidation error: {e}")
            return 0

    async def invalidate_keys_batch(
        self, keys: List[str], batch_size: int = 100
    ) -> int:
        """
        Efficiently delete a list of specific keys using batching.

        Args:
            keys: List of specific cache keys to delete
            batch_size: Number of keys to delete per batch

        Returns:
            Number of keys deleted
        """
        if not keys:
            return 0

        try:
            total_deleted = 0

            # Delete keys in batches
            for i in range(0, len(keys), batch_size):
                batch_keys = keys[i : i + batch_size]

                pipe = self.redis.pipeline()
                pipe.delete(*batch_keys)
                results = await pipe.execute()

                batch_deleted = sum(results) if results else 0
                total_deleted += batch_deleted

            logger.info(f"✅ Deleted {total_deleted} specific cache keys in batches")
            return total_deleted

        except Exception as e:
            logger.error(f"❌ Key batch deletion error: {e}")
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
                "error": str(e),
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

    async def health_check(self) -> bool:
        """
        Check if Redis connection is healthy.

        Returns:
            True if Redis is accessible and responding
        """
        try:
            response = await self.redis.ping()
            return response is True
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False

    async def get_redis_info(self) -> Dict[str, Any]:
        """
        Get Redis server information.

        Returns:
            Dictionary with Redis server info
        """
        try:
            info = await self.redis.info()
            return {
                "redis_version": info.get("redis_version", "unknown"),
                "used_memory": info.get("used_memory_human", "unknown"),
                "connected_clients": info.get("connected_clients", 0),
                "total_commands_processed": info.get("total_commands_processed", 0),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
            }
        except Exception as e:
            logger.error(f"Failed to get Redis info: {e}")
            return {"error": str(e)}

    async def close(self) -> None:
        """Close Redis connection."""
        try:
            await self.redis.close()
        except Exception as e:
            logger.error(f"Error closing Redis connection: {e}")


def create_query_cache(
    redis_host: str = "localhost",
    redis_port: int = 6379,
    redis_db: int = 0,
    redis_password: Optional[str] = None,
    default_ttl: int = 3600,
    max_connections: int = 10,
    retry_on_timeout: bool = True,
) -> Optional[QueryCache]:
    """
    Create a QueryCache instance with Redis connection.

    Args:
        redis_host: Redis server host
        redis_port: Redis server port
        redis_db: Redis database number
        redis_password: Redis password (if required)
        default_ttl: Default TTL in seconds
        max_connections: Maximum connections in pool
        retry_on_timeout: Whether to retry on timeout

    Returns:
        QueryCache instance or None if Redis is not available
    """
    if not REDIS_AVAILABLE or redis_module is None:
        logger.warning("Redis is not available, cache will not function")
        return None

    try:
        # Create connection pool for better performance
        pool = redis_module.ConnectionPool(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            password=redis_password,
            decode_responses=True,
            max_connections=max_connections,
            retry_on_timeout=retry_on_timeout,
            health_check_interval=30,
        )

        redis_client = redis_module.Redis(connection_pool=pool)
        cache = QueryCache(redis_client, default_ttl)
        cache._connection_pool = pool

        logger.info(
            f"✅ Query cache created with connection pool (max_connections={max_connections})"
        )
        return cache

    except Exception as e:
        logger.error(f"Failed to create Redis connection: {e}")
        return None
