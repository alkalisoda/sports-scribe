"""
Cached Database Query Builder
Integrates caching layer with Sports Intelligence Layer database queries
Based on Epic 2 Phase 2B Implementation Plan
"""

import time
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncpg
from ..utils.query_cache import QueryCache, CacheConfig, CacheInvalidationManager
from .database import SoccerDatabase
from .query_parser import ParsedSportsQuery

logger = logging.getLogger(__name__)

class QueryResult:
    """Structured query result with metadata"""
    
    def __init__(
        self,
        data: List[Dict[str, Any]],
        execution_time: float,
        row_count: int,
        cached: bool = False,
        confidence_score: float = 0.9
    ):
        self.data = data
        self.execution_time = execution_time
        self.row_count = row_count
        self.cached = cached
        self.confidence_score = confidence_score
        self.timestamp = datetime.now()
    
    def dict(self) -> Dict[str, Any]:
        """Convert to dictionary for caching"""
        return {
            "data": self.data,
            "execution_time": self.execution_time,
            "row_count": self.row_count,
            "cached": self.cached,
            "confidence_score": self.confidence_score,
            "timestamp": self.timestamp.isoformat()
        }

class CachedDatabaseQueryBuilder:
    """
    Enhanced database query builder with intelligent caching
    
    Features:
    - Automatic query result caching
    - Cache-first query execution
    - Performance monitoring
    - Cache invalidation on data updates
    """
    
    def __init__(self, supabase_url: str, supabase_key: str, redis_url: str = "redis://localhost:6379"):
        self.soccer_db = SoccerDatabase(supabase_url, supabase_key)
        
        # Initialize cache configuration
        self.cache_config = CacheConfig(redis_url=redis_url)
        self.query_cache: Optional[QueryCache] = None
        self.cache_invalidator: Optional[CacheInvalidationManager] = None
        
    async def initialize(self) -> None:
        """Initialize database and cache connections"""
        # Initialize database connection
        await self.soccer_db.initialize()
        
        # Initialize cache system
        self.query_cache = QueryCache(self.cache_config, self.soccer_db.connection)
        await self.query_cache.initialize()
        
        # Initialize cache invalidation manager
        self.cache_invalidator = CacheInvalidationManager(self.query_cache)
        
        logger.info("✅ Cached database query builder initialized")
    
    async def execute_cached_query(self, parsed_query: ParsedSportsQuery) -> QueryResult:
        """Execute query with caching layer"""
        if not self.query_cache:
            raise RuntimeError("Cache not initialized. Call initialize() first.")
        
        # Generate database query
        sql_query, query_params = self._build_sql_query(parsed_query)
        
        # Check cache first
        cached_result = await self.query_cache.get_cached_result(sql_query, query_params)
        if cached_result:
            return QueryResult(
                data=cached_result["data"],
                execution_time=cached_result["execution_time"],
                row_count=cached_result["row_count"],
                cached=True,
                confidence_score=cached_result.get("confidence_score", 0.9)
            )
        
        # Execute database query
        start_time = time.time()
        try:
            result_data = await self._execute_database_query(sql_query, query_params)
            execution_time = time.time() - start_time
            
            # Create structured result
            query_result = QueryResult(
                data=result_data,
                execution_time=execution_time,
                row_count=len(result_data),
                cached=False,
                confidence_score=self._calculate_confidence_score(parsed_query, result_data)
            )
            
            # Cache the result
            await self.query_cache.cache_result(
                sql_query,
                query_params,
                query_result.dict()
            )
            
            logger.debug(f"🔄 Query executed and cached in {execution_time:.3f}s")
            return query_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"❌ Query execution failed after {execution_time:.3f}s: {e}")
            raise
    
    def _build_sql_query(self, parsed_query: ParsedSportsQuery) -> tuple[str, Dict[str, Any]]:
        """Build SQL query and parameters from parsed query"""
        # This delegates to the existing SoccerDatabase logic
        # but returns both query and parameters for caching
        
        # Extract parameters for cache key generation
        params = {
            "entities": [e.dict() for e in parsed_query.entities],
            "time_context": parsed_query.time_context,
            "statistic_requested": parsed_query.statistic_requested,
            "filters": parsed_query.filters,
            "intent": parsed_query.intent
        }
        
        # Build SQL using existing database logic
        if parsed_query.intent == "stat_lookup":
            sql_query = self._build_stat_lookup_query(parsed_query)
        elif parsed_query.intent == "comparison":
            sql_query = self._build_comparison_query(parsed_query)
        elif parsed_query.intent == "ranking":
            sql_query = self._build_ranking_query(parsed_query)
        else:
            sql_query = self._build_general_query(parsed_query)
        
        return sql_query, params
    
    def _build_stat_lookup_query(self, parsed_query: ParsedSportsQuery) -> str:
        """Build SQL for statistical lookup queries"""
        # Example implementation - adapt based on your schema
        entity = parsed_query.entities[0] if parsed_query.entities else None
        stat = parsed_query.statistic_requested
        
        if entity and entity.type == "player":
            base_query = f"""
            SELECT 
                p.name as player_name,
                SUM(pms.{stat}) as total_{stat},
                COUNT(pms.match_id) as matches_played
            FROM players p
            JOIN player_match_stats pms ON p.id = pms.player_id
            WHERE LOWER(p.name) LIKE LOWER('%{entity.name}%')
            """
            
            # Add time context filters
            if parsed_query.time_context == "this_season":
                base_query += " AND pms.match_date >= '2024-08-01' AND pms.match_date <= '2025-06-30'"
            elif parsed_query.time_context == "last_season":
                base_query += " AND pms.match_date >= '2023-08-01' AND pms.match_date <= '2024-06-30'"
            
            # Add venue filters
            if "venue" in parsed_query.filters:
                base_query += f" AND pms.venue = '{parsed_query.filters['venue']}'"
            
            base_query += " GROUP BY p.id, p.name"
            return base_query
        
        return "SELECT 1 as placeholder"  # Fallback
    
    def _build_comparison_query(self, parsed_query: ParsedSportsQuery) -> str:
        """Build SQL for comparison queries"""
        # Implementation for comparison queries
        return "SELECT 1 as placeholder"  # Placeholder
    
    def _build_ranking_query(self, parsed_query: ParsedSportsQuery) -> str:
        """Build SQL for ranking queries"""
        # Implementation for ranking queries
        return "SELECT 1 as placeholder"  # Placeholder
    
    def _build_general_query(self, parsed_query: ParsedSportsQuery) -> str:
        """Build SQL for general queries"""
        # Implementation for general queries
        return "SELECT 1 as placeholder"  # Placeholder
    
    async def _execute_database_query(self, sql_query: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute SQL query against database"""
        try:
            rows = await self.soccer_db.connection.fetch(sql_query)
            return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"❌ Database query execution error: {e}")
            raise
    
    def _calculate_confidence_score(self, parsed_query: ParsedSportsQuery, result_data: List[Dict[str, Any]]) -> float:
        """Calculate confidence score based on query and result quality"""
        base_confidence = parsed_query.confidence
        
        # Adjust based on result size
        if not result_data:
            return max(0.1, base_confidence * 0.3)  # Low confidence for no results
        elif len(result_data) == 1:
            return base_confidence  # Good confidence for single result
        else:
            return min(0.9, base_confidence * 0.8)  # Slightly lower for multiple results
    
    async def invalidate_player_data(self, player_id: str) -> None:
        """Invalidate cached data for a player"""
        if self.cache_invalidator:
            count = await self.cache_invalidator.invalidate_player_cache(player_id)
            logger.info(f"🗑️ Invalidated {count} cached queries for player {player_id}")
    
    async def invalidate_team_data(self, team_id: str) -> None:
        """Invalidate cached data for a team"""
        if self.cache_invalidator:
            count = await self.cache_invalidator.invalidate_team_cache(team_id)
            logger.info(f"🗑️ Invalidated {count} cached queries for team {team_id}")
    
    async def invalidate_game_data(self, game_id: str) -> None:
        """Invalidate cached data for a game"""
        if self.cache_invalidator:
            count = await self.cache_invalidator.invalidate_game_cache(game_id)
            logger.info(f"🗑️ Invalidated {count} cached queries for game {game_id}")
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get cache and query performance statistics"""
        if not self.query_cache:
            return {"error": "Cache not initialized"}
        
        return await self.query_cache.get_cache_stats()
    
    async def cleanup_cache(self) -> int:
        """Clean up expired cache entries"""
        if self.query_cache:
            return await self.query_cache.cleanup_expired()
        return 0
    
    async def close(self) -> None:
        """Close database and cache connections"""
        if self.query_cache:
            await self.query_cache.close()
        
        if self.soccer_db:
            await self.soccer_db.close()
        
        logger.info("🔌 Cached database query builder closed")

# Integration with existing Sports Intelligence Layer
class EnhancedSoccerIntelligenceLayer:
    """Enhanced Sports Intelligence Layer with caching"""
    
    def __init__(self, supabase_url: str, supabase_key: str, redis_url: str = "redis://localhost:6379"):
        self.cached_db = CachedDatabaseQueryBuilder(supabase_url, supabase_key, redis_url)
        # Initialize other components (parser, etc.) as needed
    
    async def initialize(self) -> None:
        """Initialize the enhanced system"""
        await self.cached_db.initialize()
        logger.info("✅ Enhanced Soccer Intelligence Layer initialized")
    
    async def process_query_with_cache(self, query_text: str) -> Dict[str, Any]:
        """Process natural language query with caching"""
        try:
            # Parse the query (use existing parser)
            from .query_parser import SoccerQueryParser
            parser = SoccerQueryParser()
            parsed_query = parser.parse_query(query_text)
            
            # Execute with caching
            result = await self.cached_db.execute_cached_query(parsed_query)
            
            # Format response
            return {
                "status": "success",
                "query": {
                    "original": query_text,
                    "parsed": parsed_query.dict()
                },
                "result": {
                    "data": result.data,
                    "cached": result.cached,
                    "execution_time_ms": result.execution_time * 1000,
                    "row_count": result.row_count,
                    "confidence_score": result.confidence_score
                },
                "metadata": {
                    "timestamp": result.timestamp.isoformat(),
                    "processing_time_ms": result.execution_time * 1000,
                    "data_source": "supabase_cached"
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Query processing error: {e}")
            return {
                "status": "error",
                "error": str(e),
                "query": query_text
            }
    
    async def close(self) -> None:
        """Close the enhanced system"""
        await self.cached_db.close()