"""
Smart Query Caching System Implementation
Based on Epic 2 Phase 2B (SIL-005)
"""

import hashlib
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Union
import asyncpg
import redis.asyncio as redis
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class CacheConfig(BaseModel):
    """Configuration for the query cache system"""
    redis_url: str = "redis://localhost:6379"
    default_ttl: int = 3600  # 1 hour
    cache_hit_counter: str = "cache_hits"
    cache_miss_counter: str = "cache_misses"
    
class QueryCache:
    """
    Smart Query Caching System with Redis and PostgreSQL persistence
    
    Features:
    - Redis for high-performance caching
    - PostgreSQL for cache persistence
    - Intelligent TTL based on query type
    - Cache hit/miss tracking
    - Automatic cleanup of expired entries
    """
    
    def __init__(self, config: CacheConfig, db_connection: asyncpg.Connection):
        self.config = config
        self.db_connection = db_connection
        self.redis_client: Optional[redis.Redis] = None
        
    async def initialize(self) -> None:
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.from_url(self.config.redis_url)
            await self.redis_client.ping()
            logger.info("✅ Redis connection established")
        except Exception as e:
            logger.warning(f"⚠️ Redis connection failed, using database only: {e}")
            self.redis_client = None
    
    def _generate_query_hash(self, query: str, params: Dict[str, Any]) -> str:
        """Generate consistent hash for query + parameters"""
        query_string = f"{query}:{json.dumps(params, sort_keys=True)}"
        return hashlib.sha256(query_string.encode()).hexdigest()
    
    def _determine_ttl(self, query: str, result: Dict[str, Any]) -> int:
        """Determine appropriate TTL based on query type and data freshness"""
        query_lower = query.lower()
        
        # Live data - short TTL
        if any(keyword in query_lower for keyword in ["live", "current_game", "real_time"]):
            return 60  # 1 minute
        
        # Current season data - medium TTL
        elif any(keyword in query_lower for keyword in ["season", "2024-25", "this season"]):
            return 1800  # 30 minutes
        
        # Historical data - long TTL
        elif any(keyword in query_lower for keyword in ["career", "historical", "all time"]):
            return 86400  # 24 hours
        
        # Player stats - medium TTL
        elif "player" in query_lower:
            return 3600  # 1 hour
        
        # Default TTL
        else:
            return self.config.default_ttl
    
    async def get_cached_result(self, query: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Retrieve cached query result"""
        query_hash = self._generate_query_hash(query, params)
        
        try:
            # Try Redis first (fastest)
            if self.redis_client:
                cached_data = await self.redis_client.get(f"query:{query_hash}")
                if cached_data:
                    await self.redis_client.incr(self.config.cache_hit_counter)
                    result = json.loads(cached_data)
                    logger.debug(f"🎯 Redis cache hit for query hash: {query_hash[:8]}...")
                    return result
            
            # Fallback to database cache
            db_result = await self._get_cached_from_db(query_hash)
            if db_result:
                # Store in Redis for future requests
                if self.redis_client:
                    ttl = self._determine_ttl(query, db_result)
                    await self.redis_client.setex(
                        f"query:{query_hash}",
                        ttl,
                        json.dumps(db_result)
                    )
                
                await self.redis_client.incr(self.config.cache_hit_counter) if self.redis_client else None
                logger.debug(f"🎯 Database cache hit for query hash: {query_hash[:8]}...")
                return db_result
            
            # Cache miss
            if self.redis_client:
                await self.redis_client.incr(self.config.cache_miss_counter)
            logger.debug(f"❌ Cache miss for query hash: {query_hash[:8]}...")
            return None
            
        except Exception as e:
            logger.error(f"❌ Cache retrieval error: {e}")
            return None
    
    async def cache_result(
        self,
        query: str,
        params: Dict[str, Any],
        result: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> None:
        """Cache query result with appropriate TTL"""
        query_hash = self._generate_query_hash(query, params)
        ttl = ttl or self._determine_ttl(query, result)
        
        try:
            # Store in Redis
            if self.redis_client:
                await self.redis_client.setex(
                    f"query:{query_hash}",
                    ttl,
                    json.dumps(result)
                )
                logger.debug(f"💾 Result cached in Redis with TTL {ttl}s")
            
            # Store in database for persistence
            await self._store_in_db_cache(query_hash, query, result, ttl)
            logger.debug(f"💾 Result persisted in database cache")
            
        except Exception as e:
            logger.error(f"❌ Cache storage error: {e}")
    
    async def _get_cached_from_db(self, query_hash: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached result from database"""
        try:
            query = """
            SELECT result_data, expires_at 
            FROM query_cache 
            WHERE query_hash = $1 AND expires_at > NOW()
            """
            
            row = await self.db_connection.fetchrow(query, query_hash)
            if row:
                # Update hit count and last accessed
                await self.db_connection.execute(
                    """
                    UPDATE query_cache 
                    SET hit_count = hit_count + 1, last_accessed_at = NOW() 
                    WHERE query_hash = $1
                    """,
                    query_hash
                )
                return dict(row['result_data'])
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Database cache retrieval error: {e}")
            return None
    
    async def _store_in_db_cache(
        self,
        query_hash: str,
        query: str,
        result: Dict[str, Any],
        ttl: int
    ) -> None:
        """Store result in database cache"""
        try:
            expires_at = datetime.now() + timedelta(seconds=ttl)
            confidence_score = result.get('confidence_score', 0.9)
            
            insert_query = """
            INSERT INTO query_cache (
                query_hash, query_text, result_data, confidence_score, expires_at
            ) VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (query_hash) 
            DO UPDATE SET 
                result_data = EXCLUDED.result_data,
                confidence_score = EXCLUDED.confidence_score,
                expires_at = EXCLUDED.expires_at,
                hit_count = query_cache.hit_count + 1,
                last_accessed_at = NOW()
            """
            
            await self.db_connection.execute(
                insert_query,
                query_hash,
                query,
                json.dumps(result),
                confidence_score,
                expires_at
            )
            
        except Exception as e:
            logger.error(f"❌ Database cache storage error: {e}")
    
    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate all cache entries matching a pattern"""
        try:
            deleted_count = 0
            
            # Invalidate from Redis
            if self.redis_client:
                keys = await self.redis_client.keys(f"query:*{pattern}*")
                if keys:
                    deleted_count += await self.redis_client.delete(*keys)
            
            # Invalidate from database
            db_deleted = await self.db_connection.fetchval(
                """
                DELETE FROM query_cache 
                WHERE query_text ILIKE $1
                RETURNING COUNT(*)
                """,
                f"%{pattern}%"
            )
            
            deleted_count += db_deleted or 0
            logger.info(f"🗑️ Invalidated {deleted_count} cache entries matching pattern: {pattern}")
            return deleted_count
            
        except Exception as e:
            logger.error(f"❌ Cache invalidation error: {e}")
            return 0
    
    async def cleanup_expired(self) -> int:
        """Clean up expired cache entries"""
        try:
            # Database cleanup is handled by the cleanup_expired_cache() function
            # defined in the SQL schema
            deleted_count = await self.db_connection.fetchval("SELECT cleanup_expired_cache()")
            
            if deleted_count:
                logger.info(f"🧹 Cleaned up {deleted_count} expired cache entries")
            
            return deleted_count or 0
            
        except Exception as e:
            logger.error(f"❌ Cache cleanup error: {e}")
            return 0
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        try:
            stats = {
                "timestamp": datetime.now().isoformat(),
                "redis_available": self.redis_client is not None
            }
            
            # Redis stats
            if self.redis_client:
                cache_hits = await self.redis_client.get(self.config.cache_hit_counter)
                cache_misses = await self.redis_client.get(self.config.cache_miss_counter)
                
                stats.update({
                    "cache_hits": int(cache_hits) if cache_hits else 0,
                    "cache_misses": int(cache_misses) if cache_misses else 0,
                    "redis_memory_info": await self.redis_client.memory_usage("query:*") if cache_hits else 0
                })
                
                # Calculate hit rate
                total_requests = stats["cache_hits"] + stats["cache_misses"]
                stats["hit_rate"] = (stats["cache_hits"] / total_requests) if total_requests > 0 else 0
            
            # Database cache stats
            db_stats = await self.db_connection.fetchrow("""
                SELECT 
                    COUNT(*) as total_cached_queries,
                    COUNT(*) FILTER (WHERE expires_at > NOW()) as active_cached_queries,
                    AVG(hit_count) as avg_hit_count,
                    MAX(hit_count) as max_hit_count
                FROM query_cache
            """)
            
            if db_stats:
                stats.update({
                    "total_cached_queries": db_stats["total_cached_queries"],
                    "active_cached_queries": db_stats["active_cached_queries"],
                    "avg_hit_count": float(db_stats["avg_hit_count"]) if db_stats["avg_hit_count"] else 0,
                    "max_hit_count": db_stats["max_hit_count"]
                })
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Cache stats error: {e}")
            return {"error": str(e)}
    
    async def close(self) -> None:
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.close()
            logger.info("🔌 Redis connection closed")


class CacheInvalidationManager:
    """Manages cache invalidation strategies"""
    
    def __init__(self, query_cache: QueryCache):
        self.cache = query_cache
    
    async def invalidate_player_cache(self, player_id: str) -> int:
        """Invalidate all cached queries related to a specific player"""
        pattern = f"player_id*{player_id}"
        return await self.cache.invalidate_pattern(pattern)
    
    async def invalidate_team_cache(self, team_id: str) -> int:
        """Invalidate all cached queries related to a specific team"""
        pattern = f"team*{team_id}"
        return await self.cache.invalidate_pattern(pattern)
    
    async def invalidate_game_cache(self, game_id: str) -> int:
        """Invalidate cached queries for a specific game"""
        pattern = f"game_id*{game_id}"
        return await self.cache.invalidate_pattern(pattern)
    
    async def invalidate_season_cache(self, season: str) -> int:
        """Invalidate cached queries for a specific season"""
        pattern = f"season*{season}"
        return await self.cache.invalidate_pattern(pattern)