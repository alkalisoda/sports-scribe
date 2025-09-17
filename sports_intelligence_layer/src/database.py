"""Soccer Database Interface (async optimized version).

- Uses both synchronous and asynchronous Supabase clients for optimal performance
- Implements concurrent database operations for multiple queries
- Adds minimal player stat aggregation from player_match_stats
- Provides simple season range helper and parsed-query runner
- Safe ISO datetime parsing (handles trailing 'Z')
- Performance improvements through async patterns and caching
- Updated for new Supabase schema: supports new 'player_firstname'/'player_lastname' fields
- Updated team search to use 'team_name' field, team_code as short_name
- Uses both players table (for basic stats: goals, assists, rating, appearances) and player_match_stats table for detailed statistical queries
- Backward compatible with existing schema while supporting new field names
"""

import logging
import asyncio
import time
import hashlib
import json
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from supabase import create_client, Client

from config.soccer_entities import (
    Player, Team, Competition, Match, PlayerStatistics, TeamStatistics,
    Position, CompetitionType, MatchStatus
)

logger = logging.getLogger(__name__)


class DatabaseError(Exception):
    """Base exception for database operations."""
    pass


def _safe_parse_iso(dt: Optional[str]) -> Optional[datetime]:
    if not dt:
        return None
    try:
        # supabase often returns "...Z"
        return datetime.fromisoformat(dt.replace("Z", "+00:00"))
    except Exception:
        try:
            return datetime.fromisoformat(dt)
        except Exception:
            return None


class SoccerDatabase:
    """High-level interface for soccer database operations (async optimized)."""

    def __init__(self, supabase_url: str, supabase_key: str, max_workers: int = 10):
        """Initialize database connection and cache with async support."""
        self.supabase: Client = create_client(supabase_url, supabase_key)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._performance_stats = {
            "total_queries": 0,
            "total_time": 0.0,
            "concurrent_queries": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        # Cache configuration
        self.cache_ttl_hours = 24  # Cache TTL in hours
        self.max_cache_size = 1000  # Maximum number of cache entries (LRU eviction when exceeded)
        logger.info(f"Initialized SoccerDatabase with {max_workers} worker threads for async operations")

    # ---------- Query Cache Methods ----------
    
    def _generate_cache_key(self, parsed_query: Any) -> str:
        """Generate a SHA256 hash for cache lookup based on query components."""
        try:
            # Create a dictionary with all relevant query components
            query_dict = {
                "entities": [(e.name, e.entity_type.value, e.confidence) for e in parsed_query.entities],
                "time_context": parsed_query.time_context.value,
                "comparison_type": parsed_query.comparison_type.value if parsed_query.comparison_type else None,
                "filters": parsed_query.filters,
                "statistic_requested": parsed_query.statistic_requested,
                "statistics_requested": parsed_query.statistics_requested,
                "query_intent": parsed_query.query_intent
            }
            
            # Convert to JSON string and create SHA256 hash (matching table schema)
            query_json = json.dumps(query_dict, sort_keys=True)
            cache_hash = hashlib.sha256(query_json.encode()).hexdigest()
            
            logger.debug(f"Generated cache hash: {cache_hash} for query: {parsed_query.original_query}")
            return cache_hash
            
        except Exception as e:
            logger.warning(f"Failed to generate cache hash: {e}")
            # Fallback to simple hash of original query
            return hashlib.sha256(parsed_query.original_query.encode()).hexdigest()
    
    def _get_cached_result(self, cache_hash: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached result from cache table using the new schema."""
        try:
            # Query cache table using query_hash field
            response = self.supabase.table('query_cache').select('*').eq('query_hash', cache_hash).execute()
            
            if response.data:
                cache_entry = response.data[0]
                
                # Check if cache entry is still valid (expires_at field)
                expires_at = datetime.fromisoformat(cache_entry['expires_at'].replace('Z', '+00:00'))
                
                if datetime.now(expires_at.tzinfo) < expires_at:
                    self._performance_stats["cache_hits"] += 1
                    logger.info(f"Cache hit for hash: {cache_hash}")
                    
                    # Update last_accessed_at and increment hit_count for LRU tracking
                    try:
                        current_hit_count = cache_entry.get('hit_count', 0)
                        self.supabase.table('query_cache').update({
                            'last_accessed_at': datetime.utcnow().isoformat(),
                            'hit_count': current_hit_count + 1
                        }).eq('query_hash', cache_hash).execute()
                    except Exception as e:
                        logger.warning(f"Failed to update cache access stats for hash {cache_hash}: {e}")
                    
                    # Parse and return the cached result (JSONB format)
                    try:
                        cached_data = cache_entry['result_data']  # Already parsed as dict from JSONB
                        # Ensure cached flag is set correctly
                        result_data = {
                            "status": "success",
                            "cached": True,
                            "cache_hash": cache_hash,
                            "confidence_score": float(cache_entry.get('confidence_score', 0.9)),
                            "hit_count": cache_entry.get('hit_count', 0) + 1,
                            **cached_data
                        }
                        # Override any cached=False that might be in cached_data
                        result_data["cached"] = True
                        return result_data
                    except Exception as e:
                        logger.error(f"Failed to process cached data: {e}")
                        # Delete invalid cache entry
                        self.supabase.table('query_cache').delete().eq('query_hash', cache_hash).execute()
                        return None
                else:
                    logger.info(f"Cache entry expired for hash: {cache_hash}")
                    # Delete expired cache entry
                    self.supabase.table('query_cache').delete().eq('query_hash', cache_hash).execute()
                    return None
            else:
                self._performance_stats["cache_misses"] += 1
                logger.debug(f"Cache miss for hash: {cache_hash}")
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving cached result: {e}")
            self._performance_stats["cache_misses"] += 1
            return None
    
    def _store_cached_result(self, cache_hash: str, result: Dict[str, Any], original_query: str) -> None:
        """Store query result in cache table with new schema and LRU management."""
        try:
            # Check cache size and perform LRU eviction if necessary
            self._enforce_cache_size_limit()
            
            # Calculate expiration time based on TTL
            expires_at = (datetime.utcnow() + timedelta(hours=self.cache_ttl_hours)).isoformat()
            current_time = datetime.utcnow().isoformat()
            
            # Calculate confidence score based on result quality
            confidence_score = self._calculate_confidence_score(result)
            
            # Prepare cache entry with new schema
            cache_data = {
                "query_hash": cache_hash,
                "query_text": original_query,
                "result_data": result,  # JSONB format - Supabase handles conversion
                "confidence_score": confidence_score,
                "expires_at": expires_at,
                "hit_count": 0,  # Initialize hit count
                "created_at": current_time,
                "last_accessed_at": current_time
            }
            
            # Insert cache entry (upsert on conflict with query_hash)
            response = self.supabase.table('query_cache').upsert(cache_data, on_conflict="query_hash").execute()
            
            if response.data:
                logger.info(f"Cached result for hash: {cache_hash}")
            else:
                logger.warning(f"Failed to cache result for hash: {cache_hash}")
                
        except Exception as e:
            logger.error(f"Error storing cached result: {e}")
    
    def _calculate_confidence_score(self, result: Dict[str, Any]) -> float:
        """Calculate confidence score for cache entry based on result quality."""
        try:
            base_score = 0.8
            
            # Adjust based on result status
            if result.get("status") == "success":
                base_score += 0.1
            elif result.get("status") == "error":
                base_score = 0.3
            
            # Adjust based on data availability
            if result.get("value") is not None and result.get("value") > 0:
                base_score += 0.05
            
            # Adjust based on match count (more matches = higher confidence)
            matches = result.get("matches", 0)
            if matches > 10:
                base_score += 0.05
            elif matches == 0:
                base_score -= 0.1
            
            return min(0.99, max(0.01, base_score))
            
        except Exception:
            return 0.8  # Default confidence score
    
    def _enforce_cache_size_limit(self) -> int:
        """Enforce cache size limit using LRU eviction strategy with new schema."""
        try:
            # Get current cache size
            count_response = self.supabase.table('query_cache').select('id', count='exact').execute()
            current_size = count_response.count if hasattr(count_response, 'count') else len(count_response.data or [])
            
            if current_size >= self.max_cache_size:
                # Calculate how many entries to evict (remove 10% of max size to avoid frequent evictions)
                entries_to_evict = max(1, int(self.max_cache_size * 0.1))
                
                logger.info(f"Cache size ({current_size}) exceeds limit ({self.max_cache_size}). Evicting {entries_to_evict} LRU entries.")
                
                # Get least recently used entries (prioritize by last_accessed_at, then by hit_count)
                lru_response = self.supabase.table('query_cache').select('id, query_hash, last_accessed_at, hit_count').order('last_accessed_at', desc=False).order('hit_count', desc=False).limit(entries_to_evict).execute()
                
                if lru_response.data:
                    # Extract IDs to delete
                    ids_to_delete = [entry['id'] for entry in lru_response.data]
                    
                    # Delete LRU entries using ID
                    delete_response = self.supabase.table('query_cache').delete().in_('id', ids_to_delete).execute()
                    
                    deleted_count = len(delete_response.data) if delete_response.data else 0
                    logger.info(f"Evicted {deleted_count} LRU cache entries")
                    
                    return deleted_count
                else:
                    logger.warning("Could not retrieve LRU entries for eviction")
                    return 0
            
            return 0
            
        except Exception as e:
            logger.error(f"Error enforcing cache size limit: {e}")
            return 0
    
    def _cleanup_expired_cache(self) -> int:
        """Clean up expired cache entries using expires_at field."""
        try:
            # Delete entries where expires_at is in the past
            current_time = datetime.utcnow().isoformat()
            response = self.supabase.table('query_cache').delete().lt('expires_at', current_time).execute()
            
            deleted_count = len(response.data) if response.data else 0
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} expired cache entries")
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error cleaning up expired cache: {e}")
            return 0

    # ---------- Basic entity getters (cached) ----------

    @lru_cache(maxsize=1000)
    def get_player(self, player_id: str) -> Optional[Player]:
        """Get player by ID with caching (sync)."""
        try:
            resp = self.supabase.table('players').select('*').eq('id', player_id).single().execute()
            data = resp.data
            if not data:
                return None
            return self._convert_to_player(data)
        except Exception as e:
            logger.exception("Error fetching player %s", player_id)
            raise DatabaseError(f"Failed to fetch player: {e}")
    
    async def get_player_async(self, player_id: str) -> Optional[Player]:
        """Get player by ID with caching (async)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.get_player, player_id)

    @lru_cache(maxsize=1000)
    def get_team(self, team_id: str) -> Optional[Team]:
        """Get team by ID with caching (sync)."""
        try:
            resp = self.supabase.table('teams').select('*').eq('id', team_id).single().execute()
            data = resp.data
            if not data:
                return None
            return self._convert_to_team(data)
        except Exception as e:
            logger.exception("Error fetching team %s", team_id)
            raise DatabaseError(f"Failed to fetch team: {e}")
    
    async def get_team_async(self, team_id: str) -> Optional[Team]:
        """Get team by ID with caching (async)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.get_team, team_id)

    @lru_cache(maxsize=100)
    def get_match(self, match_id: str) -> Optional[Match]:
        """Get match by ID with caching (sync). The 'competitions' table actually stores match data."""
        try:
            resp = self.supabase.table('competitions').select('*').eq('id', match_id).single().execute()
            data = resp.data
            if not data:
                return None
            return self._convert_to_match(data)
        except Exception as e:
            logger.exception("Error fetching match %s", match_id)
            raise DatabaseError(f"Failed to fetch match: {e}")
    
    @lru_cache(maxsize=100)
    def get_competition(self, competition_id: str) -> Optional[Competition]:
        """Get competition by ID with caching (sync). This is a legacy method that may need rework."""
        try:
            # Since competitions table stores match data, we'll create a Competition from match data
            resp = self.supabase.table('competitions').select('*').eq('id', competition_id).single().execute()
            data = resp.data
            if not data:
                return None
            return self._convert_match_to_competition(data)
        except Exception as e:
            logger.exception("Error fetching competition %s", competition_id)
            raise DatabaseError(f"Failed to fetch competition: {e}")

    # ---------- Fuzzy search ----------

    def search_players(self, query: str, limit: int = 10) -> List[Player]:
        """Search players by name (sync)."""
        try:
            # Search by player_firstname and player_lastname (current schema)
            resp = self.supabase.table('players').select('*').or_(
                f"player_firstname.ilike.%{query}%,player_lastname.ilike.%{query}%"
            ).limit(limit).execute()
            
            rows = resp.data or []
            return [self._convert_to_player(r) for r in rows]
        except Exception as e:
            logger.exception("Error searching players: %s", query)
            logger.warning(f"Returning empty list for player search: {query}")
            return []
    
    async def search_players_async(self, query: str, limit: int = 10) -> List[Player]:
        """Search players by name (async)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.search_players, query, limit)

    def search_teams(self, query: str, limit: int = 10) -> List[Team]:
        """Search teams by name (sync)."""
        try:
            resp = self.supabase.table('teams').select('*').ilike('team_name', f"%{query}%").limit(limit).execute()
            rows = resp.data or []
            return [self._convert_to_team(r) for r in rows]
        except Exception as e:
            logger.exception("Error searching teams: %s", query)
            logger.warning(f"Returning empty list for team search: {query}")
            return []
    
    async def search_teams_async(self, query: str, limit: int = 10) -> List[Team]:
        """Search teams by name (async)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.search_teams, query, limit)

    # ---------- Aggregated stats (player_match_stats) ----------

    def season_range(self, season_label: str) -> Tuple[str, str]:
        """
        Return (start_date, end_date) YYYY-MM-DD for a season label like '2024-25' or '2023-24'.
        This is a minimal helper; adjust to your league/calendar as needed.
        """
        # Minimal hardcode to get you moving
        if season_label in {"2024-25", "2024/25", "this_season"}:
            return "2024-08-01", "2025-06-30"
        if season_label in {"2023-24", "2023/24", "last_season"}:
            return "2023-08-01", "2024-06-30"
        # Fallback: current cycle assumption
        return "2024-08-01", "2025-06-30"

    def get_player_stat_sum(
        self,
        player_id: str,
        stat: str,                         # 'goals' | 'assists' | 'minutes_played' ...
        start_date: Optional[str] = None,  # 'YYYY-MM-DD'
        end_date: Optional[str] = None,
        venue: Optional[str] = None,       # 'home' | 'away' | 'neutral'
        last_n: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Minimal aggregation over player_match_stats.
        - If last_n is provided: select latest N rows by match_date then sum in Python.
        - Otherwise: fetch all rows (already filtered) then sum.
        """
        try:
            allowed_stats = {
                "goals", "assists", "minutes_played", "shots_on_target",
                "tackles", "interceptions", "passes_completed", "passes", "clean_sheets", "saves",
                "yellow_cards", "red_cards", "fouls_committed", "fouls_drawn",
                "shots", "pass_accuracy", "rating", "appearances"
            }
            if stat not in allowed_stats:
                return {"status": "not_supported", "reason": f"stat_not_supported:{stat}"}

            qb = (
                self.supabase
                .table("player_match_stats")
                .select(f"{stat}")
                .eq("player_id", player_id)
            )

            # player_match_stats table structure - may not have date fields in current schema
            if start_date and end_date:
                logger.info(f"Date filtering requested - attempting to filter by date range")
                # Try to filter by date if date fields exist
                try:
                    qb = qb.gte("match_date", start_date).lte("match_date", end_date)
                except:
                    logger.info(f"Date filtering not supported in current schema - getting all player data")

            if venue:
                qb = qb.eq("venue", venue)
            if last_n:
                qb = qb.limit(last_n)

            resp = qb.execute()
            rows = resp.data or []
            
            # Check if any data was found
            if not rows:
                return {
                    "status": "no_data", 
                    "reason": "no_matches_found",
                    "matches": 0,
                    "filters": {
                        "start_date": start_date,
                        "end_date": end_date,
                        "venue": venue,
                        "last_n": last_n,
                    },
                }
            
            value = 0
            for r in rows:
                stat_value = r.get(stat)
                if stat_value is not None:
                    if isinstance(stat_value, (int, float)):
                        value += stat_value
                    elif isinstance(stat_value, str):
                        try:
                            value += float(stat_value)
                        except (ValueError, TypeError):
                            continue

            return {
                "value": int(value) if isinstance(value, (int, float)) else value,
                "matches": len(rows),
                "filters": {
                    "start_date": start_date,
                    "end_date": end_date,
                    "venue": venue,
                    "last_n": last_n,
                },
            }
        except Exception as e:
            logger.exception("Error aggregating player stat sum")
            raise DatabaseError(f"Failed to run player stat query: {e}")
    
    async def get_player_stat_sum_async(
        self,
        player_id: str,
        stat: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        venue: Optional[str] = None,
        last_n: Optional[int] = None
    ) -> Dict[str, Any]:
        """Async version of get_player_stat_sum with performance tracking."""
        start_time = time.time()
        self._performance_stats["total_queries"] += 1
        
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor, 
                self.get_player_stat_sum,
                player_id, stat, start_date, end_date, venue, last_n
            )
            
            execution_time = time.time() - start_time
            self._performance_stats["total_time"] += execution_time
            logger.info(f"Async player stat query completed in {execution_time:.3f}s")
            
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            self._performance_stats["total_time"] += execution_time
            logger.error(f"Async player stat query failed after {execution_time:.3f}s: {e}")
            raise
    
    async def get_multiple_player_stats_concurrent(
        self,
        requests: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Execute multiple player stat requests concurrently for improved performance."""
        start_time = time.time()
        self._performance_stats["concurrent_queries"] += len(requests)
        
        logger.info(f"Executing {len(requests)} concurrent player stat queries")
        
        # Create tasks for concurrent execution
        tasks = []
        for req in requests:
            task = self.get_player_stat_sum_async(
                player_id=req.get("player_id"),
                stat=req.get("stat", "goals"),
                start_date=req.get("start_date"),
                end_date=req.get("end_date"),
                venue=req.get("venue"),
                last_n=req.get("last_n")
            )
            tasks.append(task)
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Request {i+1} failed: {result}")
                processed_results.append({
                    "status": "error",
                    "reason": str(result),
                    "request_index": i
                })
            else:
                processed_results.append(result)
        
        execution_time = time.time() - start_time
        logger.info(f"Concurrent execution of {len(requests)} queries completed in {execution_time:.3f}s")
        logger.info(f"Average time per query: {execution_time/len(requests):.3f}s")
        
        return processed_results

    def get_team_players(self, team_name: str) -> List[Dict[str, Any]]:
        """
        Get all players for a given team from Supabase.
        """
        try:
            team_players = []
            
            # First, we need to get the team_id from the teams table
            try:
                # Try exact match first
                team_response = self.supabase.table("teams").select("id, team_name").eq("team_name", team_name).execute()
                if not team_response.data:
                    # Try fuzzy match with ilike (case-insensitive partial match)
                    team_response = self.supabase.table("teams").select("id, team_name").ilike("team_name", f"%{team_name}%").execute()
                    if not team_response.data:
                        logger.warning(f"Team '{team_name}' not found in teams table (tried exact and fuzzy match)")
                        
                        # Debug: Show available teams for troubleshooting
                        try:
                            all_teams = self.supabase.table("teams").select("id, team_name").limit(20).execute()
                            available_teams = [team['team_name'] for team in (all_teams.data or [])]
                            logger.info(f"Available teams in database: {available_teams}")
                        except Exception as debug_e:
                            logger.error(f"Could not fetch available teams for debugging: {debug_e}")
                        
                        return []
                
                team_id = team_response.data[0]['id']
                
                # Now get players for this team using team_id (current schema)
                response = self.supabase.table("players").select("id, player_firstname, player_lastname, position, team_id").eq("team_id", team_id).execute()
                
                if response.data:
                    for player in response.data:
                        # Current schema format (player_firstname + player_lastname)
                        player_name = f"{player.get('player_firstname', '')} {player.get('player_lastname', '')}".strip()
                        if not player_name:
                            player_name = player.get('player_firstname') or player.get('player_lastname') or f"Player {player.get('id', 'Unknown')}"
                            
                        team_players.append({
                            'id': str(player['id']),
                            'name': player_name,
                            'position': player.get('position'),
                            'team_id': str(player['team_id'])
                        })
                
            except Exception as e:
                logger.warning(f"Error getting team players for {team_name}: {e}")
                # Fallback: try to get all players and filter by name pattern
                try:
                    response = self.supabase.table("players").select("id, player_firstname, player_lastname, position, team_id").execute()
                    # This is a simple fallback - in real implementation you'd have proper team mapping
                    for player in response.data:
                        # Current schema format (player_firstname + player_lastname)
                        player_name = f"{player.get('player_firstname', '')} {player.get('player_lastname', '')}".strip()
                        if not player_name:
                            player_name = player.get('player_firstname') or player.get('player_lastname') or f"Player {player.get('id', 'Unknown')}"
                            
                        team_players.append({
                            'id': str(player['id']),
                            'name': player_name,
                            'position': player.get('position'),
                            'team_id': str(player.get('team_id', ''))
                        })
                except Exception as fallback_error:
                    logger.error(f"Fallback also failed for {team_name}: {fallback_error}")
            
            return team_players
            
        except Exception as e:
            logger.exception(f"Error getting team players for {team_name}")
            return []

    # ---------- Convenience: run from ParsedSoccerQuery ----------

    def run_from_parsed(
        self,
        parsed: Any,                        # ParsedSoccerQuery
        player_name_to_id: Optional[Dict[str, str]] = None,
        default_season_label: str = "2024-25"
    ) -> Dict[str, Any]:
        """
        Execute a query from a ParsedSoccerQuery with cache-first approach.
        First checks cache table, then executes query and stores result if not cached.
        """
        try:
            # Generate cache hash for this query
            cache_hash = self._generate_cache_key(parsed)
            
            # Try to get cached result first
            cached_result = self._get_cached_result(cache_hash)
            if cached_result:
                logger.info(f"Returning cached result for query: {parsed.original_query}")
                return cached_result
            
            # Cache miss - execute the actual query
            logger.info(f"Cache miss - executing query: {parsed.original_query}")
            
            # Check if this is a match query (contains "vs", "versus", "match")
            if self._is_match_query(parsed):
                result = self._handle_match_query(parsed, default_season_label)
            else:
                # Pick a player or team entity
                player_name = None
                team_name = None
                for e in parsed.entities:
                    if getattr(e, "entity_type", None):
                        if str(e.entity_type.value) == "player":
                            player_name = e.name
                        elif str(e.entity_type.value) == "team":
                            team_name = e.name
                
                # Handle player queries
                if player_name:
                    result = self._handle_player_query(parsed, player_name, player_name_to_id, default_season_label)
                # Handle team queries
                elif team_name:
                    result = self._handle_team_query(parsed, team_name, default_season_label)
                else:
                    result = {"status": "not_supported", "reason": "no_player_or_team_found"}
            
            # Store successful results in cache (avoid caching errors)
            if result.get("status") == "success":
                # Add cache metadata to result
                result["cached"] = False
                result["cache_hash"] = cache_hash
                
                # Store in cache for future queries
                self._store_cached_result(cache_hash, result, parsed.original_query)
                logger.info(f"Stored result in cache for query: {parsed.original_query}")
            
            return result

        except Exception as e:
            logger.exception("Error in run_from_parsed")
            return {"status": "error", "reason": str(e)}

    def _is_match_query(self, parsed: Any) -> bool:
        """Check if this is a match query (contains vs, versus, match keywords)."""
        query_lower = parsed.original_query.lower()
        match_keywords = ['vs', 'versus', 'match', 'game', 'fixture']
        return any(keyword in query_lower for keyword in match_keywords)

    def _handle_match_query(self, parsed: Any, default_season_label: str = "2024-25") -> Dict[str, Any]:
        """Handle match queries to return match results and statistics."""
        try:
            # Extract team names from entities
            team_entities = [e for e in parsed.entities if e.entity_type.value == "team"]
            
            if len(team_entities) < 2:
                return {"status": "error", "reason": "Need at least 2 teams for match query"}
            
            team1_name = team_entities[0].name
            team2_name = team_entities[1].name
            
            logger.info(f"Processing match query: {team1_name} vs {team2_name}")
            
            # Get team IDs
            team1_id = self._get_team_id_by_name(team1_name)
            team2_id = self._get_team_id_by_name(team2_name)
            
            if not team1_id or not team2_id:
                return {"status": "error", "reason": f"Could not find team IDs for {team1_name} and/or {team2_name}"}
            
            # Find matches between these teams
            match_results = self._get_match_results(team1_id, team2_id)
            
            if not match_results:
                return {"status": "no_data", "reason": "No matches found between these teams"}
            
            # Return the most recent match result
            latest_match = match_results[0]  # Assuming sorted by date
            
            return {
                "status": "success",
                "query_type": "match_result",
                "match": {
                    "team1": {
                        "name": team1_name,
                        "id": team1_id,
                        "goals": latest_match["team1_goals"]
                    },
                    "team2": {
                        "name": team2_name,
                        "id": team2_id,
                        "goals": latest_match["team2_goals"]
                    },
                    "winner": latest_match["winner"],
                    "score": f"{latest_match['team1_goals']}-{latest_match['team2_goals']}",
                    "match_id": latest_match["match_id"],
                    "statistics": latest_match["statistics"]
                }
            }
            
        except Exception as e:
            logger.exception(f"Error handling match query: {e}")
            return {"status": "error", "reason": str(e)}

    def _get_team_id_by_name(self, team_name: str) -> Optional[str]:
        """Get team ID by team name."""
        try:
            # Search for team by name
            teams = self.search_teams(team_name, limit=1)
            if teams:
                return teams[0].id
            return None
        except Exception as e:
            logger.warning(f"Error getting team ID for {team_name}: {e}")
            return None

    def _get_match_results(self, team1_id: str, team2_id: str) -> List[Dict[str, Any]]:
        """Get match results between two teams by analyzing player_match_stats."""
        try:
            # Get all player stats for matches involving both teams
            response = self.supabase.table("player_match_stats").select("*").execute()
            all_stats = response.data or []
            
            # Group by match_id and calculate team goals
            match_data = {}
            
            for stat in all_stats:
                match_id = stat.get("match_id")
                team_id = stat.get("team_id")
                goals = stat.get("goals", 0)
                
                if not match_id or not team_id:
                    continue
                
                if match_id not in match_data:
                    match_data[match_id] = {
                        "team1_goals": 0,
                        "team2_goals": 0,
                        "team1_stats": [],
                        "team2_stats": []
                    }
                
                # Check if this match involves both teams
                teams_in_match = set()
                for existing_stat in all_stats:
                    if existing_stat.get("match_id") == match_id:
                        teams_in_match.add(existing_stat.get("team_id"))
                
                if team1_id in teams_in_match and team2_id in teams_in_match:
                    # This is a match between our two teams
                    if team_id == team1_id:
                        match_data[match_id]["team1_goals"] += goals if goals else 0
                        match_data[match_id]["team1_stats"].append(stat)
                    elif team_id == team2_id:
                        match_data[match_id]["team2_goals"] += goals if goals else 0
                        match_data[match_id]["team2_stats"].append(stat)
            
            # Convert to results format
            results = []
            for match_id, data in match_data.items():
                # Include all matches, even if no goals (0-0 draws)
                # Determine winner
                if data["team1_goals"] > data["team2_goals"]:
                    winner = "team1"
                elif data["team2_goals"] > data["team1_goals"]:
                    winner = "team2"
                else:
                    winner = "draw"
                
                # Calculate additional statistics
                statistics = self._calculate_match_statistics(data["team1_stats"], data["team2_stats"])
                
                results.append({
                    "match_id": match_id,
                    "team1_goals": data["team1_goals"],
                    "team2_goals": data["team2_goals"],
                    "winner": winner,
                    "statistics": statistics
                })
            
            # Sort by match_id (assuming higher numbers are more recent)
            results.sort(key=lambda x: x["match_id"], reverse=True)
            
            return results
            
        except Exception as e:
            logger.exception(f"Error getting match results: {e}")
            return []

    def _calculate_match_statistics(self, team1_stats: List[Dict], team2_stats: List[Dict]) -> Dict[str, Any]:
        """Calculate match statistics from player stats."""
        try:
            # Calculate team totals
            team1_totals = {
                "shots": sum(stat.get("shots", 0) for stat in team1_stats if stat.get("shots")),
                "shots_on_target": sum(stat.get("shots_on_target", 0) for stat in team1_stats if stat.get("shots_on_target")),
                "passes": sum(stat.get("passes", 0) for stat in team1_stats if stat.get("passes")),
                "pass_accuracy": 0,
                "yellow_cards": sum(stat.get("yellow_cards", 0) for stat in team1_stats if stat.get("yellow_cards")),
                "red_cards": sum(stat.get("red_cards", 0) for stat in team1_stats if stat.get("red_cards")),
                "minutes_played": sum(stat.get("minutes_played", 0) or stat.get("minutes", 0) for stat in team1_stats if stat.get("minutes_played") or stat.get("minutes"))
            }
            
            team2_totals = {
                "shots": sum(stat.get("shots", 0) for stat in team2_stats if stat.get("shots")),
                "shots_on_target": sum(stat.get("shots_on_target", 0) for stat in team2_stats if stat.get("shots_on_target")),
                "passes": sum(stat.get("passes", 0) for stat in team2_stats if stat.get("passes")),
                "pass_accuracy": 0,
                "yellow_cards": sum(stat.get("yellow_cards", 0) for stat in team2_stats if stat.get("yellow_cards")),
                "red_cards": sum(stat.get("red_cards", 0) for stat in team2_stats if stat.get("red_cards")),
                "minutes_played": sum(stat.get("minutes_played", 0) or stat.get("minutes", 0) for stat in team2_stats if stat.get("minutes_played") or stat.get("minutes"))
            }
            
            # Calculate pass accuracy
            team1_pass_attempts = sum(stat.get("passes", 0) for stat in team1_stats if stat.get("passes"))
            team1_pass_completed = sum(stat.get("pass_accuracy", 0) for stat in team1_stats if stat.get("pass_accuracy"))
            if team1_pass_attempts > 0:
                team1_totals["pass_accuracy"] = round((team1_pass_completed / team1_pass_attempts) * 100, 1)
            
            team2_pass_attempts = sum(stat.get("passes", 0) for stat in team2_stats if stat.get("passes"))
            team2_pass_completed = sum(stat.get("pass_accuracy", 0) for stat in team2_stats if stat.get("pass_accuracy"))
            if team2_pass_attempts > 0:
                team2_totals["pass_accuracy"] = round((team2_pass_completed / team2_pass_attempts) * 100, 1)
            
            return {
                "team1": team1_totals,
                "team2": team2_totals,
                "total_shots": team1_totals["shots"] + team2_totals["shots"],
                "total_goals": team1_totals.get("goals", 0) + team2_totals.get("goals", 0),
                "total_cards": (team1_totals["yellow_cards"] + team1_totals["red_cards"] + 
                               team2_totals["yellow_cards"] + team2_totals["red_cards"])
            }
            
        except Exception as e:
            logger.exception(f"Error calculating match statistics: {e}")
            return {}

    def _handle_player_query(
        self,
        parsed: Any,
        player_name: str,
        player_name_to_id: Optional[Dict[str, str]] = None,
        default_season_label: str = "2024-25"
    ) -> Dict[str, Any]:
        """Handle player queries"""
        # Resolve player_id
        pid = None
        if player_name_to_id and player_name.lower() in player_name_to_id:
            pid = player_name_to_id[player_name.lower()]
        else:
            # fallback: try fuzzy search in DB
            players = self.search_players(player_name, limit=1)
            pid = players[0].id if players else None

        if not pid:
            # Debug: Show available players for troubleshooting
            logger.warning(f"Player '{player_name}' not found in database")
            try:
                all_players = self.supabase.table("players").select("id, player_firstname, player_lastname").limit(20).execute()
                available_players = [f"{player.get('player_firstname', '')} {player.get('player_lastname', '')}".strip()
                                   for player in (all_players.data or [])]
                logger.info(f"Available players in database: {available_players}")
            except Exception as debug_e:
                logger.error(f"Could not fetch available players for debugging: {debug_e}")
            
            return {"status": "no_data", "reason": "player_not_found"}

        # Map statistics - extend statistical type mapping
        stat_map = {
            "goals": "goals",
            "assists": "assists",
            "ast": "assists",         # Alias for assists
            "minutes": "minutes_played",
            "minutes_played": "minutes_played",
            "shots": "shots",  
            "shots_on_target": "shots_on_target",
            "passes": "passes",
            "pass_completion": "pass_accuracy", 
            "pass_accuracy": "pass_accuracy",
            "tackles": "tackles",
            "interceptions": "interceptions",
            "clean_sheets": "clean_sheets",
            "saves": "saves",
            "yellow_cards": "yellow_cards",
            "red_cards": "red_cards",
            "fouls_committed": "fouls_committed",
            "fouls_drawn": "fouls_drawn",
            "rating": "rating",
            "appearances": "appearances",
            "performance": "performance"
        }
        
        # Check for multiple statistics request (new feature)
        if hasattr(parsed, 'statistics_requested') and parsed.statistics_requested and len(parsed.statistics_requested) > 1:
            return self._handle_multiple_player_statistics(pid, player_name, parsed, stat_map, default_season_label)
        
        # Check if this is a performance query
        if not parsed.statistic_requested or parsed.statistic_requested == "performance":
            return self._get_player_performance(pid, player_name, default_season_label)
        
        stat = stat_map.get(parsed.statistic_requested, "goals")

        # Time/season context
        last_n = None
        start_date, end_date = None, None
        if str(parsed.time_context.value) == "last_n_games":
            n = parsed.filters.get("last_n") if isinstance(parsed.filters, dict) else None
            if isinstance(n, int) and n > 0:
                last_n = n
        elif str(parsed.time_context.value) == "last_season":
            start_date, end_date = self.season_range("last_season")
        else:
            start_date, end_date = self.season_range(default_season_label)

        # Venue filter
        venue = None
        if isinstance(parsed.filters, dict):
            v = parsed.filters.get("venue")
            if v in {"home", "away", "neutral"}:
                venue = v

        result = self.get_player_stat_sum(
            player_id=pid,
            stat=stat,
            start_date=start_date,
            end_date=end_date,
            venue=venue,
            last_n=last_n,
        )

        return {
            "status": "success",
            "value": result.get("value", 0),
            "stat": stat,
            "player_id": pid,
            "player_name": player_name,
            "matches": result.get("matches", 0),
            "filters": result.get("filters", {})
        }

    def _get_player_performance(self, player_id: str, player_name: str, default_season_label: str = "2024-25") -> Dict[str, Any]:
        """Get comprehensive performance stats for a player"""
        try:
            # Get multiple statistics for the player
            stats_to_get = ["goals", "assists", "minutes_played", "shots", "passes", "tackles", "saves", "rating", "appearances"]
            performance_stats = {}
            
            for stat in stats_to_get:
                try:
                    result = self.get_player_stat_sum(
                        player_id=player_id,
                        stat=stat,
                        start_date=None,  # Get all data for performance overview
                        end_date=None,
                        venue=None,
                        last_n=None,
                    )
                    performance_stats[stat] = result.get("value", 0)
                except Exception as e:
                    logger.warning(f"Error getting {stat} for player {player_name}: {e}")
                    performance_stats[stat] = 0
            
            return {
                "status": "success",
                "player_id": player_id,
                "player_name": player_name,
                "performance": performance_stats,
                "query_type": "performance_overview"
            }
            
        except Exception as e:
            logger.exception(f"Error getting performance for player {player_name}")
            return {"status": "error", "reason": str(e)}

    def _handle_multiple_player_statistics(
        self,
        player_id: str,
        player_name: str,
        parsed: Any,
        stat_map: Dict[str, str],
        default_season_label: str = "2024-25"
    ) -> Dict[str, Any]:
        """Handle queries requesting multiple statistics for a player."""
        try:
            # Time/season context (same logic as single stat query)
            last_n = None
            start_date, end_date = None, None
            if str(parsed.time_context.value) == "last_n_games":
                n = parsed.filters.get("last_n") if isinstance(parsed.filters, dict) else None
                if isinstance(n, int) and n > 0:
                    last_n = n
            elif str(parsed.time_context.value) == "last_season":
                start_date, end_date = self.season_range("last_season")
            else:
                start_date, end_date = self.season_range(default_season_label)

            # Venue filter
            venue = None
            if parsed.filters:
                v = parsed.filters.get("venue")
                if v in ("home", "away"):
                    venue = v

            # Collect all requested statistics
            multiple_stats = {}
            total_matches = 0
            
            for stat_requested in parsed.statistics_requested:
                mapped_stat = stat_map.get(stat_requested, stat_requested)
                
                try:
                    result = self.get_player_stat_sum(
                        player_id=player_id,
                        stat=mapped_stat,
                        start_date=start_date,
                        end_date=end_date,
                        venue=venue,
                        last_n=last_n,
                    )
                    
                    multiple_stats[stat_requested] = {
                        "value": result.get("value", 0),
                        "stat": mapped_stat,
                        "matches": result.get("matches", 0)
                    }
                    
                    # Track maximum matches played (some stats may have fewer matches)
                    total_matches = max(total_matches, result.get("matches", 0))
                    
                except Exception as e:
                    logger.warning(f"Error getting {stat_requested} for player {player_name}: {e}")
                    multiple_stats[stat_requested] = {
                        "value": 0,
                        "stat": mapped_stat,
                        "matches": 0
                    }

            return {
                "status": "success",
                "player_id": player_id,
                "player_name": player_name,
                "statistics": multiple_stats,
                "total_matches": total_matches,
                "query_type": "multiple_statistics",
                "filters": {
                    "venue": venue,
                    "last_n": last_n,
                    "start_date": start_date.isoformat() if start_date and hasattr(start_date, 'isoformat') else str(start_date) if start_date else None,
                    "end_date": end_date.isoformat() if end_date and hasattr(end_date, 'isoformat') else str(end_date) if end_date else None
                }
            }
            
        except Exception as e:
            logger.exception(f"Error getting multiple statistics for player {player_name}")
            return {"status": "error", "reason": str(e)}

    def _handle_team_query(
        self,
        parsed: Any,
        team_name: str,
        default_season_label: str = "2024-25"
    ) -> Dict[str, Any]:
        """Handle team queries"""
        # For team queries, we return statistics for all players in the team
        stat_map = {
            "goals": "goals",
            "assists": "assists",
            "ast": "assists",         # Alias for assists
            "minutes": "minutes_played",
            "minutes_played": "minutes_played",
            "shots": "shots",  
            "shots_on_target": "shots_on_target",
            "passes": "passes",
            "pass_completion": "pass_accuracy", 
            "pass_accuracy": "pass_accuracy",
            "tackles": "tackles",
            "interceptions": "interceptions",
            "clean_sheets": "clean_sheets",
            "saves": "saves",
            "yellow_cards": "yellow_cards",
            "red_cards": "red_cards",
            "fouls_committed": "fouls_committed",
            "fouls_drawn": "fouls_drawn",
            "rating": "rating",
            "appearances": "appearances"
        }
        stat = stat_map.get((parsed.statistic_requested or "goals"), "goals")

        # Get team players list
        team_players = self.get_team_players(team_name)
        if not team_players:
            return {"status": "no_data", "reason": "team_players_not_found"}

        # Calculate team total statistics
        total_value = 0
        total_matches = 0
        
        for player in team_players:
            try:
                result = self.get_player_stat_sum(
                    player_id=player['id'],
                    stat=stat,
                    start_date=None,  # Don't use date filtering, get all data directly
                    end_date=None,
                    venue=None,
                    last_n=None,
                )
                if result.get("value"):
                    total_value += result.get("value", 0)
                total_matches += result.get("matches", 0)
            except Exception as e:
                logger.warning(f"Error getting stats for player {player['name']}: {e}")
                continue

        return {
            "status": "success",
            "value": total_value,
            "stat": stat,
            "team_name": team_name,
            "matches": total_matches,
            "player_count": len(team_players)
        }

    # ---------- Converters & aggregators ----------

    def _convert_to_player(self, data: Dict[str, Any]) -> Player:
        """Convert database record to Player object."""
        # Handle current schema format with player_firstname/player_lastname
        player_name = f"{data.get('player_firstname', '')} {data.get('player_lastname', '')}".strip()
        if not player_name:
            player_name = data.get('player_firstname') or data.get('player_lastname') or f"Player {data.get('id', 'Unknown')}"

        return Player(
            id=str(data['id']),  # Convert integer ID to string for compatibility
            name=player_name,
            common_name=player_name,  # Use full name as common name since common_name field doesn't exist
            nationality=data.get('player_nationality') or "",  # Use player_nationality field
            birth_date=None,  # birth_date field doesn't exist in current schema
            position=self._safe_position(data.get('position')),
            height_cm=None,  # height_cm field doesn't exist in current schema
            weight_kg=None,  # weight_kg field doesn't exist in current schema
            team_id=str(data['team_id']) if data.get('team_id') else None,  # Convert integer to string
            jersey_number=None,  # jersey_number field doesn't exist in current schema
            preferred_foot=None,  # preferred_foot field doesn't exist in current schema
            market_value=None  # market_value field doesn't exist in current schema
        )

    def _convert_to_team(self, data: Dict[str, Any]) -> Team:
        """Convert database record to Team object."""
        return Team(
            id=str(data['id']),  # Convert integer ID to string for compatibility
            name=data.get('team_name') or f"Team {data.get('id', 'Unknown')}",
            short_name=data.get('team_code') or data.get('team_name', ''),  # Use team_code as short_name
            country=data.get('team_country') or "",
            founded_year=data.get('team_founded'),
            venue_name=data.get('venue_name'),
            venue_capacity=data.get('venue_capacity'),
            coach_name=None,  # coach_name field doesn't exist in current schema
            logo_url=data.get('team_logo'),
            primary_color=None,  # primary_color field doesn't exist in current schema
            secondary_color=None  # secondary_color field doesn't exist in current schema
        )

    def _convert_to_match(self, data: Dict[str, Any]) -> Match:
        """Convert database record to Match object."""
        return Match(
            id=int(data['id']),
            name=data['name'],
            type=data.get('type', 'api-football'),
            country=data.get('country') or "",
            season=data.get('season') or "",
            start_date=data.get('start_date'),
            end_date=data.get('end_date'),
            status=data.get('status'),
            venue_id=data.get('venueId'),
            league_id=data.get('leagueId'),
            home_team_id=data.get('homeTeamId'),
            away_team_id=data.get('awayTeamId'),
            goals_home=data.get('goalsHome'),
            goals_away=data.get('goalsAway'),
            goals_home_half_time=data.get('goalsHomeHalfTime'),
            goals_away_half_time=data.get('goalsAwayHalfTime'),
            goals_home_extra_time=data.get('goalsHomeExtraTime'),
            goals_away_extra_time=data.get('goalsAwayExtraTime'),
            penalty_home=data.get('penaltyHome'),
            penalty_away=data.get('penaltyAway')
        )
    
    def _convert_match_to_competition(self, data: Dict[str, Any]) -> Competition:
        """Convert match data to Competition object for legacy compatibility."""
        return Competition(
            id=str(data['id']),
            name=data['name'],
            short_name=data.get('name', data['name']),
            country=data.get('country') or "",
            type=self._safe_competition_type(data.get('type')),
            season=data.get('season') or "",
            start_date=_safe_parse_iso(data.get('start_date')),
            end_date=_safe_parse_iso(data.get('end_date')),
            current_matchday=None,
            number_of_matchdays=None,
            number_of_teams=None,
            current_season_id=None
        )

    def _safe_position(self, raw: Optional[str]) -> Position:
        try:
            return Position(raw) if raw else Position.UNKNOWN
        except Exception:
            return Position.UNKNOWN

    def _safe_competition_type(self, raw: Optional[str]) -> CompetitionType:
        try:
            return CompetitionType(raw) if raw else CompetitionType.LEAGUE
        except Exception:
            return CompetitionType.LEAGUE

    # (Optional) legacy aggregators retained for compatibility
    def _aggregate_player_statistics(self, stats_data: List[Dict[str, Any]]) -> PlayerStatistics:
        """Aggregate multiple player statistics records (if you have a player_statistics table)."""
        aggregated = PlayerStatistics()
        for stat in stats_data or []:
            aggregated.goals += stat.get('goals', 0)
            aggregated.assists += stat.get('assists', 0)
            aggregated.minutes_played += stat.get('minutes_played', 0)
            aggregated.passes_completed += stat.get('passes_completed', 0)
            aggregated.shots_on_target += stat.get('shots_on_target', 0)
            aggregated.tackles += stat.get('tackles', 0)
            aggregated.interceptions += stat.get('interceptions', 0)
            aggregated.clean_sheets += stat.get('clean_sheets', 0)
            aggregated.saves += stat.get('saves', 0)
            aggregated.yellow_cards += stat.get('yellow_cards', 0)
            aggregated.red_cards += stat.get('red_cards', 0)
            aggregated.fouls_committed += stat.get('fouls_committed', 0)
            aggregated.fouls_drawn += stat.get('fouls_drawn', 0)
        if stats_data:
            total = len(stats_data)
            aggregated.pass_accuracy = sum(s.get('pass_accuracy', 0) for s in stats_data) / total
        return aggregated

    def _aggregate_team_statistics(self, stats_data: List[Dict[str, Any]]) -> TeamStatistics:
        """Aggregate multiple team statistics records (if you have a team_statistics table)."""
        aggregated = TeamStatistics()
        for stat in stats_data or []:
            aggregated.matches_played += stat.get('matches_played', 0)
            aggregated.wins += stat.get('wins', 0)
            aggregated.draws += stat.get('draws', 0)
            aggregated.losses += stat.get('losses', 0)
            aggregated.goals_scored += stat.get('goals_scored', 0)
            aggregated.goals_conceded += stat.get('goals_conceded', 0)
            aggregated.clean_sheets += stat.get('clean_sheets', 0)
            aggregated.points += stat.get('points', 0)
        if stats_data:
            total = len(stats_data)
            aggregated.possession_avg = sum(s.get('possession_avg', 0) for s in stats_data) / total
            aggregated.pass_accuracy_avg = sum(s.get('pass_accuracy_avg', 0) for s in stats_data) / total
            aggregated.shots_per_game = sum(s.get('shots_per_game', 0) for s in stats_data) / total
        return aggregated
    
    # ---------- Performance monitoring and async main methods ----------
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics for the database operations."""
        stats = self._performance_stats.copy()
        if stats["total_queries"] > 0:
            stats["average_query_time"] = stats["total_time"] / stats["total_queries"]
        else:
            stats["average_query_time"] = 0
        
        # Add cache hit rate
        total_cache_requests = stats["cache_hits"] + stats["cache_misses"]
        if total_cache_requests > 0:
            stats["cache_hit_rate"] = stats["cache_hits"] / total_cache_requests
        else:
            stats["cache_hit_rate"] = 0
            
        return stats
    
    def reset_performance_stats(self):
        """Reset performance statistics."""
        self._performance_stats = {
            "total_queries": 0,
            "total_time": 0.0,
            "concurrent_queries": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        logger.info("Performance statistics reset")
    
    def cleanup_cache(self) -> int:
        """Clean up expired cache entries. Returns number of entries cleaned."""
        return self._cleanup_expired_cache()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache-specific statistics using new schema."""
        try:
            # Get total cache entries
            response = self.supabase.table('query_cache').select('id', count='exact').execute()
            total_entries = response.count if hasattr(response, 'count') else len(response.data or [])
            
            # Get expired entries count
            current_time = datetime.utcnow().isoformat()
            expired_response = self.supabase.table('query_cache').select('id', count='exact').lt('expires_at', current_time).execute()
            expired_entries = expired_response.count if hasattr(expired_response, 'count') else len(expired_response.data or [])
            
            # Get hit count statistics
            hit_stats_response = self.supabase.table('query_cache').select('hit_count').execute()
            hit_counts = [entry.get('hit_count', 0) for entry in (hit_stats_response.data or [])]
            
            total_hits_in_cache = sum(hit_counts)
            avg_hits_per_entry = total_hits_in_cache / max(1, total_entries)
            
            # Calculate cache utilization
            cache_utilization = (total_entries / self.max_cache_size) * 100 if self.max_cache_size > 0 else 0
            
            return {
                "total_cache_entries": total_entries,
                "max_cache_size": self.max_cache_size,
                "cache_utilization_percent": round(cache_utilization, 2),
                "expired_entries": expired_entries,
                "cache_ttl_hours": self.cache_ttl_hours,
                "cache_hits": self._performance_stats["cache_hits"],
                "cache_misses": self._performance_stats["cache_misses"],
                "cache_hit_rate": self._performance_stats["cache_hits"] / max(1, self._performance_stats["cache_hits"] + self._performance_stats["cache_misses"]),
                "total_hits_in_cache": total_hits_in_cache,
                "avg_hits_per_entry": round(avg_hits_per_entry, 2),
                "lru_eviction_enabled": True,
                "entries_until_eviction": max(0, self.max_cache_size - total_entries)
            }
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {
                "error": str(e),
                "cache_hits": self._performance_stats["cache_hits"],
                "cache_misses": self._performance_stats["cache_misses"]
            }
    
    def clear_cache(self) -> int:
        """Clear all cache entries. Returns number of entries cleared."""
        try:
            response = self.supabase.table('query_cache').delete().neq('cache_key', '').execute()
            cleared_count = len(response.data) if response.data else 0
            logger.info(f"Cleared {cleared_count} cache entries")
            return cleared_count
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return 0
    
    def set_cache_config(self, max_size: Optional[int] = None, ttl_hours: Optional[int] = None) -> Dict[str, Any]:
        """Configure cache settings."""
        old_config = {
            "max_cache_size": self.max_cache_size,
            "cache_ttl_hours": self.cache_ttl_hours
        }
        
        if max_size is not None:
            if max_size <= 0:
                raise ValueError("max_size must be greater than 0")
            self.max_cache_size = max_size
            logger.info(f"Updated max cache size to {max_size}")
        
        if ttl_hours is not None:
            if ttl_hours <= 0:
                raise ValueError("ttl_hours must be greater than 0")
            self.cache_ttl_hours = ttl_hours
            logger.info(f"Updated cache TTL to {ttl_hours} hours")
        
        new_config = {
            "max_cache_size": self.max_cache_size,
            "cache_ttl_hours": self.cache_ttl_hours
        }
        
        return {
            "old_config": old_config,
            "new_config": new_config,
            "changes_applied": max_size is not None or ttl_hours is not None
        }
    
    def force_lru_eviction(self, target_size: Optional[int] = None) -> int:
        """Manually trigger LRU eviction to reduce cache to target size."""
        try:
            if target_size is None:
                target_size = int(self.max_cache_size * 0.8)  # Reduce to 80% of max size
            
            # Get current cache size
            count_response = self.supabase.table('query_cache').select('id', count='exact').execute()
            current_size = count_response.count if hasattr(count_response, 'count') else len(count_response.data or [])
            
            if current_size <= target_size:
                logger.info(f"Cache size ({current_size}) is already at or below target ({target_size})")
                return 0
            
            entries_to_evict = current_size - target_size
            logger.info(f"Force evicting {entries_to_evict} LRU entries to reach target size {target_size}")
            
            # Get least recently used entries (prioritize by last_accessed_at, then by hit_count)
            lru_response = self.supabase.table('query_cache').select('id, query_hash, last_accessed_at, hit_count').order('last_accessed_at', desc=False).order('hit_count', desc=False).limit(entries_to_evict).execute()
            
            if lru_response.data:
                # Extract IDs to delete
                ids_to_delete = [entry['id'] for entry in lru_response.data]
                
                # Delete LRU entries
                delete_response = self.supabase.table('query_cache').delete().in_('id', ids_to_delete).execute()
                
                deleted_count = len(delete_response.data) if delete_response.data else 0
                logger.info(f"Force evicted {deleted_count} LRU cache entries")
                
                return deleted_count
            else:
                logger.warning("Could not retrieve LRU entries for force eviction")
                return 0
                
        except Exception as e:
            logger.error(f"Error in force LRU eviction: {e}")
            return 0
    
    async def run_from_parsed_async(
        self,
        parsed: Any,
        player_name_to_id: Optional[Dict[str, str]] = None,
        default_season_label: str = "2024-25"
    ) -> Dict[str, Any]:
        """Async version of run_from_parsed with cache-first approach and enhanced performance."""
        start_time = time.time()
        
        try:
            # Generate cache hash for this query
            cache_hash = self._generate_cache_key(parsed)
            
            # Try to get cached result first (run in executor to avoid blocking)
            loop = asyncio.get_event_loop()
            cached_result = await loop.run_in_executor(
                self.executor,
                self._get_cached_result,
                cache_hash
            )
            
            if cached_result:
                execution_time = time.time() - start_time
                logger.info(f"Returning cached result for query in {execution_time:.3f}s: {parsed.original_query}")
                return cached_result
            
            # Cache miss - execute the actual query
            logger.info(f"Cache miss - executing async query: {parsed.original_query}")
            
            # Check if this is a match query (contains "vs", "versus", "match")
            if self._is_match_query(parsed):
                result = await loop.run_in_executor(
                    self.executor, 
                    self._handle_match_query,
                    parsed, default_season_label
                )
            else:
                # Pick a player or team entity
                player_name = None
                team_name = None
                for e in parsed.entities:
                    if getattr(e, "entity_type", None):
                        if str(e.entity_type.value) == "player":
                            player_name = e.name
                        elif str(e.entity_type.value) == "team":
                            team_name = e.name
                
                # Handle player queries with async
                if player_name:
                    result = await self._handle_player_query_async(
                        parsed, player_name, player_name_to_id, default_season_label
                    )
                # Handle team queries with async
                elif team_name:
                    result = await self._handle_team_query_async(
                        parsed, team_name, default_season_label
                    )
                else:
                    result = {"status": "not_supported", "reason": "no_player_or_team_found"}
            
            # Store successful results in cache (avoid caching errors)
            if result.get("status") == "success":
                # Add cache metadata to result
                result["cached"] = False
                result["cache_hash"] = cache_hash
                
                # Store in cache for future queries (run in executor to avoid blocking)
                await loop.run_in_executor(
                    self.executor,
                    self._store_cached_result,
                    cache_hash, result, parsed.original_query
                )
                
                execution_time = time.time() - start_time
                logger.info(f"Stored result in cache after {execution_time:.3f}s for query: {parsed.original_query}")
            
            return result

        except Exception as e:
            execution_time = time.time() - start_time
            logger.exception(f"Error in async run_from_parsed after {execution_time:.3f}s")
            return {"status": "error", "reason": str(e)}
    
    async def _handle_player_query_async(
        self,
        parsed: Any,
        player_name: str,
        player_name_to_id: Optional[Dict[str, str]] = None,
        default_season_label: str = "2024-25"
    ) -> Dict[str, Any]:
        """Async version of player query handling."""
        # Resolve player_id
        pid = None
        if player_name_to_id and player_name.lower() in player_name_to_id:
            pid = player_name_to_id[player_name.lower()]
        else:
            # Use async search for better performance
            players = await self.search_players_async(player_name, limit=1)
            pid = players[0].id if players else None

        if not pid:
            return {"status": "no_data", "reason": "player_not_found"}

        # Check for multiple statistics request (new feature)
        if hasattr(parsed, 'statistics_requested') and parsed.statistics_requested and len(parsed.statistics_requested) > 1:
            return await self._handle_multiple_player_statistics_async(pid, player_name, parsed, default_season_label)
        
        # Check if this is a performance query
        if not parsed.statistic_requested or parsed.statistic_requested == "performance":
            return await self._get_player_performance_async(pid, player_name, default_season_label)
        
        # Single statistic handling with async
        stat_map = {
            "goals": "goals",
            "assists": "assists",  # Correct field name from players table
            "ast": "assists",      # Alias for assists
            "minutes": "minutes_played",
            "minutes_played": "minutes_played",
            "shots": "shots",
            "shots_on_target": "shots_on_target",
            "passes": "passes",
            "pass_completion": "pass_accuracy",
            "pass_accuracy": "pass_accuracy",
            "tackles": "tackles",
            "interceptions": "interceptions",
            "clean_sheets": "clean_sheets",
            "saves": "saves",
            "yellow_cards": "yellow_cards",
            "red_cards": "red_cards",
            "fouls_committed": "fouls_committed",
            "fouls_drawn": "fouls_drawn",
            "rating": "rating",           # Available in players table
            "appearances": "appearances", # Available in players table
            "performance": "performance"
        }
        
        stat = stat_map.get(parsed.statistic_requested, "goals")

        # Time/season context
        last_n = None
        start_date, end_date = None, None
        if str(parsed.time_context.value) == "last_n_games":
            n = parsed.filters.get("last_n") if isinstance(parsed.filters, dict) else None
            if isinstance(n, int) and n > 0:
                last_n = n
        elif str(parsed.time_context.value) == "last_season":
            start_date, end_date = self.season_range("last_season")
        else:
            start_date, end_date = self.season_range(default_season_label)

        # Venue filter
        venue = None
        if isinstance(parsed.filters, dict):
            v = parsed.filters.get("venue")
            if v in {"home", "away", "neutral"}:
                venue = v

        result = await self.get_player_stat_sum_async(
            player_id=pid,
            stat=stat,
            start_date=start_date,
            end_date=end_date,
            venue=venue,
            last_n=last_n,
        )

        return {
            "status": "success",
            "value": result.get("value", 0),
            "stat": stat,
            "player_id": pid,
            "player_name": player_name,
            "matches": result.get("matches", 0),
            "filters": result.get("filters", {})
        }
    
    async def _handle_multiple_player_statistics_async(
        self,
        player_id: str,
        player_name: str,
        parsed: Any,
        default_season_label: str = "2024-25"
    ) -> Dict[str, Any]:
        """Async version of multiple player statistics handling."""
        stat_map = {
            "goals": "goals",
            "assists": "assists",  # Correct field name from players table
            "ast": "assists",      # Alias for assists
            "minutes": "minutes_played",
            "minutes_played": "minutes_played",
            "shots": "shots",  
            "shots_on_target": "shots_on_target",
            "passes": "passes",
            "pass_completion": "pass_accuracy", 
            "pass_accuracy": "pass_accuracy",
            "tackles": "tackles",
            "interceptions": "interceptions",
            "clean_sheets": "clean_sheets",
            "saves": "saves",
            "yellow_cards": "yellow_cards",
            "red_cards": "red_cards",
            "fouls_committed": "fouls_committed",
            "fouls_drawn": "fouls_drawn",
            "rating": "rating",           # New field
            "appearances": "appearances"  # New field
        }
        
        # Time/season context 
        last_n = None
        start_date, end_date = None, None
        if str(parsed.time_context.value) == "last_n_games":
            n = parsed.filters.get("last_n") if isinstance(parsed.filters, dict) else None
            if isinstance(n, int) and n > 0:
                last_n = n
        elif str(parsed.time_context.value) == "last_season":
            start_date, end_date = self.season_range("last_season")
        else:
            start_date, end_date = self.season_range(default_season_label)

        # Venue filter
        venue = None
        if parsed.filters:
            v = parsed.filters.get("venue")
            if v in ("home", "away"):
                venue = v

        # Create concurrent requests for all statistics
        requests = []
        for stat_requested in parsed.statistics_requested:
            mapped_stat = stat_map.get(stat_requested, stat_requested)
            requests.append({
                "player_id": player_id,
                "stat": mapped_stat,
                "start_date": start_date,
                "end_date": end_date,
                "venue": venue,
                "last_n": last_n
            })
        
        # Execute all requests concurrently
        concurrent_results = await self.get_multiple_player_stats_concurrent(requests)
        
        # Format the results
        multiple_stats = {}
        total_matches = 0
        
        for i, stat_requested in enumerate(parsed.statistics_requested):
            result = concurrent_results[i]
            if not isinstance(result, dict) or "status" in result and result["status"] == "error":
                multiple_stats[stat_requested] = {
                    "value": 0,
                    "stat": stat_map.get(stat_requested, stat_requested),
                    "matches": 0
                }
            else:
                multiple_stats[stat_requested] = {
                    "value": result.get("value", 0),
                    "stat": stat_map.get(stat_requested, stat_requested),
                    "matches": result.get("matches", 0)
                }
                total_matches = max(total_matches, result.get("matches", 0))

        return {
            "status": "success",
            "player_id": player_id,
            "player_name": player_name,
            "statistics": multiple_stats,
            "total_matches": total_matches,
            "query_type": "multiple_statistics",
            "filters": {
                "venue": venue,
                "last_n": last_n,
                "start_date": start_date.isoformat() if start_date and hasattr(start_date, 'isoformat') else str(start_date) if start_date else None,
                "end_date": end_date.isoformat() if end_date and hasattr(end_date, 'isoformat') else str(end_date) if end_date else None
            }
        }
    
    async def _get_player_performance_async(
        self, 
        player_id: str, 
        player_name: str, 
        default_season_label: str = "2024-25"
    ) -> Dict[str, Any]:
        """Async version of player performance retrieval."""
        stats_to_get = ["goals", "assists", "minutes_played", "shots", "passes", "tackles", "saves", "rating", "appearances"]
        
        # Create concurrent requests for all performance stats
        requests = []
        for stat in stats_to_get:
            requests.append({
                "player_id": player_id,
                "stat": stat,
                "start_date": None,
                "end_date": None,
                "venue": None,
                "last_n": None
            })
        
        # Execute all requests concurrently
        concurrent_results = await self.get_multiple_player_stats_concurrent(requests)
        
        # Format performance stats
        performance_stats = {}
        for i, stat in enumerate(stats_to_get):
            result = concurrent_results[i]
            if not isinstance(result, dict) or "status" in result and result["status"] == "error":
                performance_stats[stat] = 0
            else:
                performance_stats[stat] = result.get("value", 0)
        
        return {
            "status": "success",
            "player_id": player_id,
            "player_name": player_name,
            "performance": performance_stats,
            "query_type": "performance_overview"
        }
    
    async def _handle_team_query_async(
        self,
        parsed: Any,
        team_name: str,
        default_season_label: str = "2024-25"
    ) -> Dict[str, Any]:
        """Async version of team query handling."""
        # Get team players list asynchronously
        loop = asyncio.get_event_loop()
        team_players = await loop.run_in_executor(self.executor, self.get_team_players, team_name)
        
        if not team_players:
            return {"status": "no_data", "reason": "team_players_not_found"}

        stat_map = {
            "goals": "goals",
            "assists": "assists",  # Correct field name from players table
            "ast": "assists",      # Alias for assists
            "minutes": "minutes_played",
            "minutes_played": "minutes_played",
            "shots": "shots",  
            "shots_on_target": "shots_on_target",
            "passes": "passes",
            "pass_completion": "pass_accuracy", 
            "pass_accuracy": "pass_accuracy",
            "tackles": "tackles",
            "interceptions": "interceptions",
            "clean_sheets": "clean_sheets",
            "saves": "saves",
            "yellow_cards": "yellow_cards",
            "red_cards": "red_cards",
            "fouls_committed": "fouls_committed",
            "fouls_drawn": "fouls_drawn",
            "rating": "rating",           # New field
            "appearances": "appearances"  # New field
        }
        stat = stat_map.get((parsed.statistic_requested or "goals"), "goals")

        # Create concurrent requests for all team players
        requests = []
        for player in team_players:
            requests.append({
                "player_id": player['id'],
                "stat": stat,
                "start_date": None,
                "end_date": None,
                "venue": None,
                "last_n": None
            })
        
        # Execute all requests concurrently
        concurrent_results = await self.get_multiple_player_stats_concurrent(requests)
        
        # Check if this is a ranking query
        filters = getattr(parsed, 'filters', {})
        ranking_info = filters.get('ranking') if isinstance(filters, dict) else None
        
        if ranking_info and ranking_info.get('type') == 'ranking':
            # Return individual player rankings instead of team total
            player_stats = []
            for i, result in enumerate(concurrent_results):
                if isinstance(result, dict) and not ("status" in result and result["status"] == "error"):
                    player_name = team_players[i].get('name', f"Player {team_players[i].get('id')}")
                    player_stats.append({
                        "player_name": player_name,
                        "player_id": team_players[i].get('id'),
                        "value": result.get("value", 0),
                        "matches": result.get("matches", 0)
                    })
            
            # Sort by value (descending for "most", ascending for "least")
            direction = ranking_info.get('direction', 'highest')
            reverse_sort = (direction == 'highest')
            player_stats.sort(key=lambda x: x['value'], reverse=reverse_sort)
            
            # Get top player(s)
            if player_stats:
                top_player = player_stats[0]
                return {
                    "status": "success",
                    "query_type": "team_player_ranking",
                    "stat": stat,
                    "team_name": team_name,
                    "ranking_type": ranking_info.get('keyword', 'most'),
                    "top_player": top_player,
                    "all_players": player_stats[:10],  # Top 10
                    "player_count": len(team_players)
                }
            else:
                return {"status": "no_data", "reason": "no_player_stats_found"}
        
        # Default: Calculate team totals (for non-ranking queries)
        total_value = 0
        total_matches = 0
        
        for result in concurrent_results:
            if isinstance(result, dict) and not ("status" in result and result["status"] == "error"):
                total_value += result.get("value", 0)
                total_matches += result.get("matches", 0)

        return {
            "status": "success",
            "value": total_value,
            "stat": stat,
            "team_name": team_name,
            "matches": total_matches,
            "player_count": len(team_players)
        }

    # ===== HISTORICAL STATISTICS READING METHODS =====

    def get_historical_stats(self, entity_type: str, entity_id: str, stat_types: List[str] = None,
                           record_types: List[str] = None, limit: int = None) -> List[Dict[str, Any]]:
        """Retrieve historical statistics for a specific entity."""
        try:
            query = self.supabase.table('historical_records').select('*').eq(
                'entity_type', entity_type
            ).eq('entity_id', entity_id)

            if stat_types:
                query = query.in_('stat_name', stat_types)

            if record_types:
                query = query.in_('record_type', record_types)

            if limit:
                query = query.limit(limit)

            # Order by date_achieved descending to get most recent first
            query = query.order('date_achieved', desc=True)

            response = query.execute()
            stats = response.data or []

            logger.info(f"Retrieved {len(stats)} historical stats for {entity_type} {entity_id}")
            return stats

        except Exception as e:
            logger.error(f"Error retrieving historical stats for {entity_type} {entity_id}: {e}")
            return []

    async def get_historical_stats_async(self, entity_type: str, entity_id: str, stat_types: List[str] = None,
                                       record_types: List[str] = None, limit: int = None) -> List[Dict[str, Any]]:
        """Async version of get_historical_stats."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.get_historical_stats,
            entity_type, entity_id, stat_types, record_types, limit
        )

    def get_historical_stats_by_timerange(self, start_date: str, end_date: str,
                                        entity_type: str = None, stat_types: List[str] = None) -> List[Dict[str, Any]]:
        """Retrieve historical statistics within a specific time range."""
        try:
            query = self.supabase.table('historical_records').select('*').gte(
                'date_achieved', start_date
            ).lte('date_achieved', end_date)

            if entity_type:
                query = query.eq('entity_type', entity_type)

            if stat_types:
                query = query.in_('stat_name', stat_types)

            query = query.order('date_achieved', desc=True)

            response = query.execute()
            stats = response.data or []

            logger.info(f"Retrieved {len(stats)} historical stats between {start_date} and {end_date}")
            return stats

        except Exception as e:
            logger.error(f"Error retrieving historical stats by timerange: {e}")
            return []

    async def get_historical_stats_by_timerange_async(self, start_date: str, end_date: str,
                                                    entity_type: str = None, stat_types: List[str] = None) -> List[Dict[str, Any]]:
        """Async version of get_historical_stats_by_timerange."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.get_historical_stats_by_timerange,
            start_date, end_date, entity_type, stat_types
        )

    def get_comparative_historical_stats(self, entity_ids: List[str], stat_type: str,
                                       entity_type: str = 'player', record_type: str = None) -> List[Dict[str, Any]]:
        """Get comparative historical statistics for multiple entities."""
        try:
            query = self.supabase.table('historical_records').select('*').eq(
                'entity_type', entity_type
            ).eq('stat_name', stat_type).in_('entity_id', entity_ids)

            if record_type:
                query = query.eq('record_type', record_type)

            query = query.order('stat_value', desc=True)

            response = query.execute()
            stats = response.data or []

            logger.info(f"Retrieved {len(stats)} comparative historical stats for {stat_type}")
            return stats

        except Exception as e:
            logger.error(f"Error retrieving comparative historical stats: {e}")
            return []

    async def get_comparative_historical_stats_async(self, entity_ids: List[str], stat_type: str,
                                                   entity_type: str = 'player', record_type: str = None) -> List[Dict[str, Any]]:
        """Async version of get_comparative_historical_stats."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.get_comparative_historical_stats,
            entity_ids, stat_type, entity_type, record_type
        )

    def get_entity_best_historical_stats(self, entity_type: str, entity_id: str,
                                       top_n: int = 10) -> List[Dict[str, Any]]:
        """Get the best/highest historical statistics for an entity."""
        try:
            query = self.supabase.table('historical_records').select('*').eq(
                'entity_type', entity_type
            ).eq('entity_id', entity_id).eq('record_type', 'best').order('stat_value', desc=True)

            if top_n:
                query = query.limit(top_n)

            response = query.execute()
            stats = response.data or []

            logger.info(f"Retrieved {len(stats)} best historical stats for {entity_type} {entity_id}")
            return stats

        except Exception as e:
            logger.error(f"Error retrieving best historical stats for {entity_type} {entity_id}: {e}")
            return []

    async def get_entity_best_historical_stats_async(self, entity_type: str, entity_id: str,
                                                   top_n: int = 10) -> List[Dict[str, Any]]:
        """Async version of get_entity_best_historical_stats."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.get_entity_best_historical_stats,
            entity_type, entity_id, top_n
        )

    def get_entity_career_historical_stats(self, entity_type: str, entity_id: str) -> List[Dict[str, Any]]:
        """Get career/total historical statistics for an entity."""
        try:
            query = self.supabase.table('historical_records').select('*').eq(
                'entity_type', entity_type
            ).eq('entity_id', entity_id).eq('record_type', 'career_total').order('stat_name')

            response = query.execute()
            stats = response.data or []

            logger.info(f"Retrieved {len(stats)} career historical stats for {entity_type} {entity_id}")
            return stats

        except Exception as e:
            logger.error(f"Error retrieving career historical stats for {entity_type} {entity_id}: {e}")
            return []

    async def get_entity_career_historical_stats_async(self, entity_type: str, entity_id: str) -> List[Dict[str, Any]]:
        """Async version of get_entity_career_historical_stats."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.get_entity_career_historical_stats,
            entity_type, entity_id
        )

    def get_recent_historical_milestones(self, entity_type: str = None, entity_id: str = None,
                                       days: int = 30, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent historical milestones and achievements."""
        try:
            cutoff_date = (datetime.utcnow() - timedelta(days=days)).isoformat()

            query = self.supabase.table('historical_records').select('*').eq(
                'record_type', 'milestone'
            ).gte('date_achieved', cutoff_date).order('date_achieved', desc=True)

            if entity_type:
                query = query.eq('entity_type', entity_type)

            if entity_id:
                query = query.eq('entity_id', entity_id)

            if limit:
                query = query.limit(limit)

            response = query.execute()
            milestones = response.data or []

            logger.info(f"Retrieved {len(milestones)} recent historical milestones")
            return milestones

        except Exception as e:
            logger.error(f"Error retrieving recent historical milestones: {e}")
            return []

    async def get_recent_historical_milestones_async(self, entity_type: str = None, entity_id: str = None,
                                                   days: int = 30, limit: int = 20) -> List[Dict[str, Any]]:
        """Async version of get_recent_historical_milestones."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.get_recent_historical_milestones,
            entity_type, entity_id, days, limit
        )

    def get_trending_historical_stats(self, stat_type: str, entity_type: str = 'player',
                                    limit: int = 10, record_type: str = 'best') -> List[Dict[str, Any]]:
        """Get trending/top performers for a specific statistic from historical records."""
        try:
            query = self.supabase.table('historical_records').select('*').eq(
                'entity_type', entity_type
            ).eq('stat_name', stat_type).eq('record_type', record_type).order('stat_value', desc=True)

            if limit:
                query = query.limit(limit)

            response = query.execute()
            stats = response.data or []

            logger.info(f"Retrieved {len(stats)} trending historical stats for {stat_type}")
            return stats

        except Exception as e:
            logger.error(f"Error retrieving trending historical stats for {stat_type}: {e}")
            return []

    async def get_trending_historical_stats_async(self, stat_type: str, entity_type: str = 'player',
                                                limit: int = 10, record_type: str = 'best') -> List[Dict[str, Any]]:
        """Async version of get_trending_historical_stats."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.get_trending_historical_stats,
            stat_type, entity_type, limit, record_type
        )

    def query_historical_records(self, filters: Dict[str, Any], limit: int = 100) -> List[Dict[str, Any]]:
        """Advanced query interface for historical records with flexible filtering."""
        try:
            query = self.supabase.table('historical_records').select('*')

            # Apply filters
            for field, value in filters.items():
                if field in ['order_by', 'desc']:
                    continue  # Skip special control fields

                if isinstance(value, list):
                    query = query.in_(field, value)
                elif isinstance(value, dict):
                    # Support for range queries
                    if 'gte' in value:
                        query = query.gte(field, value['gte'])
                    if 'lte' in value:
                        query = query.lte(field, value['lte'])
                    if 'gt' in value:
                        query = query.gt(field, value['gt'])
                    if 'lt' in value:
                        query = query.lt(field, value['lt'])
                    if 'eq' in value:
                        query = query.eq(field, value['eq'])
                else:
                    query = query.eq(field, value)

            # Default ordering
            if 'order_by' in filters:
                order_field = filters['order_by']
                desc = filters.get('desc', False)
                query = query.order(order_field, desc=desc)
            else:
                query = query.order('date_achieved', desc=True)

            if limit:
                query = query.limit(limit)

            response = query.execute()
            records = response.data or []

            logger.info(f"Query returned {len(records)} historical records")
            return records

        except Exception as e:
            logger.error(f"Error in advanced historical records query: {e}")
            return []

    async def query_historical_records_async(self, filters: Dict[str, Any], limit: int = 100) -> List[Dict[str, Any]]:
        """Async version of query_historical_records."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.query_historical_records,
            filters, limit
        )

    def get_entity_historical_summary(self, entity_type: str, entity_id: str) -> Dict[str, Any]:
        """Get a comprehensive summary of an entity's historical statistics."""
        try:
            # Get all historical records for the entity
            response = self.supabase.table('historical_records').select('*').eq(
                'entity_type', entity_type
            ).eq('entity_id', entity_id).execute()

            records = response.data or []

            if not records:
                return {
                    'entity_type': entity_type,
                    'entity_id': entity_id,
                    'total_records': 0,
                    'record_types': {},
                    'statistics': {},
                    'milestones': [],
                    'best_performances': [],
                    'career_totals': []
                }

            # Categorize records
            summary = {
                'entity_type': entity_type,
                'entity_id': entity_id,
                'total_records': len(records),
                'record_types': {},
                'statistics': {},
                'milestones': [],
                'best_performances': [],
                'career_totals': []
            }

            for record in records:
                record_type = record.get('record_type', 'unknown')
                stat_name = record.get('stat_name', 'unknown')

                # Count by record type
                summary['record_types'][record_type] = summary['record_types'].get(record_type, 0) + 1

                # Count by statistic type
                summary['statistics'][stat_name] = summary['statistics'].get(stat_name, 0) + 1

                # Categorize specific records
                if record_type == 'milestone':
                    summary['milestones'].append(record)
                elif record_type == 'best':
                    summary['best_performances'].append(record)
                elif record_type == 'career_total':
                    summary['career_totals'].append(record)

            logger.info(f"Generated historical summary for {entity_type} {entity_id}: {len(records)} total records")
            return summary

        except Exception as e:
            logger.error(f"Error generating entity historical summary: {e}")
            return {}

    async def get_entity_historical_summary_async(self, entity_type: str, entity_id: str) -> Dict[str, Any]:
        """Async version of get_entity_historical_summary."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.get_entity_historical_summary,
            entity_type, entity_id
        )

    def get_player_historical_context(self, player_id: str, stat_type: str = None) -> Dict[str, Any]:
        """Get historical context for a player including career progression and milestones."""
        try:
            # Get all historical records for the player
            historical_stats = self.get_historical_stats('player', player_id)

            if not historical_stats:
                return {
                    'player_id': player_id,
                    'has_historical_data': False,
                    'career_highlights': [],
                    'recent_milestones': [],
                    'best_performances': [],
                    'career_totals': {}
                }

            # Filter by stat_type if provided
            if stat_type:
                historical_stats = [stat for stat in historical_stats if stat.get('stat_name') == stat_type]

            # Categorize the data
            career_highlights = []
            recent_milestones = []
            best_performances = []
            career_totals = {}

            # Recent cutoff (last 365 days)
            recent_cutoff = (datetime.utcnow() - timedelta(days=365)).isoformat()

            for record in historical_stats:
                record_type = record.get('record_type', '')
                date_achieved = record.get('date_achieved', '')

                if record_type == 'milestone':
                    milestone_data = {
                        'stat_name': record.get('stat_name'),
                        'stat_value': record.get('stat_value'),
                        'date_achieved': date_achieved,
                        'description': record.get('description', ''),
                        'verified': record.get('verified', False)
                    }

                    if date_achieved and date_achieved > recent_cutoff:
                        recent_milestones.append(milestone_data)
                    else:
                        career_highlights.append(milestone_data)

                elif record_type == 'best':
                    best_performances.append({
                        'stat_name': record.get('stat_name'),
                        'stat_value': record.get('stat_value'),
                        'date_achieved': date_achieved,
                        'description': record.get('description', ''),
                        'verified': record.get('verified', False)
                    })

                elif record_type == 'career_total':
                    career_totals[record.get('stat_name', 'unknown')] = {
                        'value': record.get('stat_value'),
                        'last_updated': date_achieved,
                        'verified': record.get('verified', False)
                    }

            # Sort by date (most recent first)
            career_highlights.sort(key=lambda x: x.get('date_achieved', ''), reverse=True)
            recent_milestones.sort(key=lambda x: x.get('date_achieved', ''), reverse=True)
            best_performances.sort(key=lambda x: x.get('stat_value', 0), reverse=True)

            return {
                'player_id': player_id,
                'has_historical_data': True,
                'career_highlights': career_highlights[:10],  # Top 10
                'recent_milestones': recent_milestones[:5],   # Last 5
                'best_performances': best_performances[:10], # Top 10
                'career_totals': career_totals,
                'total_historical_records': len(historical_stats)
            }

        except Exception as e:
            logger.error(f"Error getting player historical context for {player_id}: {e}")
            return {
                'player_id': player_id,
                'has_historical_data': False,
                'error': str(e)
            }

    async def get_player_historical_context_async(self, player_id: str, stat_type: str = None) -> Dict[str, Any]:
        """Async version of get_player_historical_context."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.get_player_historical_context,
            player_id, stat_type
        )

    def get_team_historical_context(self, team_id: str, stat_type: str = None) -> Dict[str, Any]:
        """Get historical context for a team including achievements and records."""
        try:
            # Get all historical records for the team
            historical_stats = self.get_historical_stats('team', team_id)

            if not historical_stats:
                return {
                    'team_id': team_id,
                    'has_historical_data': False,
                    'achievements': [],
                    'team_records': [],
                    'season_bests': [],
                    'recent_milestones': []
                }

            # Filter by stat_type if provided
            if stat_type:
                historical_stats = [stat for stat in historical_stats if stat.get('stat_name') == stat_type]

            # Categorize team historical data
            achievements = []
            team_records = []
            season_bests = []
            recent_milestones = []

            # Recent cutoff (last 2 years for teams)
            recent_cutoff = (datetime.utcnow() - timedelta(days=730)).isoformat()

            for record in historical_stats:
                record_type = record.get('record_type', '')
                date_achieved = record.get('date_achieved', '')

                record_data = {
                    'stat_name': record.get('stat_name'),
                    'stat_value': record.get('stat_value'),
                    'date_achieved': date_achieved,
                    'description': record.get('description', ''),
                    'verified': record.get('verified', False)
                }

                if record_type == 'milestone':
                    if date_achieved and date_achieved > recent_cutoff:
                        recent_milestones.append(record_data)
                    else:
                        achievements.append(record_data)

                elif record_type == 'best':
                    team_records.append(record_data)

                elif record_type == 'season_best':
                    season_bests.append(record_data)

            # Sort collections
            achievements.sort(key=lambda x: x.get('date_achieved', ''), reverse=True)
            recent_milestones.sort(key=lambda x: x.get('date_achieved', ''), reverse=True)
            team_records.sort(key=lambda x: x.get('stat_value', 0), reverse=True)
            season_bests.sort(key=lambda x: x.get('date_achieved', ''), reverse=True)

            return {
                'team_id': team_id,
                'has_historical_data': True,
                'achievements': achievements[:15],        # Top 15 achievements
                'team_records': team_records[:10],       # Top 10 records
                'season_bests': season_bests[:10],       # Recent season bests
                'recent_milestones': recent_milestones[:5], # Last 5 milestones
                'total_historical_records': len(historical_stats)
            }

        except Exception as e:
            logger.error(f"Error getting team historical context for {team_id}: {e}")
            return {
                'team_id': team_id,
                'has_historical_data': False,
                'error': str(e)
            }

    async def get_team_historical_context_async(self, team_id: str, stat_type: str = None) -> Dict[str, Any]:
        """Async version of get_team_historical_context."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.get_team_historical_context,
            team_id, stat_type
        )
