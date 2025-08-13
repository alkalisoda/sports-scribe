"""Soccer Database Interface (sync version).

- Uses synchronous Supabase client (create_client)
- Adds minimal player stat aggregation from player_match_stats
- Provides simple season range helper and parsed-query runner
- Safe ISO datetime parsing (handles trailing 'Z')
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from functools import lru_cache
from supabase import create_client, Client

from ..config.soccer_entities import (
    Player, Team, Competition, PlayerStatistics, TeamStatistics,
    Position, CompetitionType
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
    """High-level interface for soccer database operations (synchronous)."""

    def __init__(self, supabase_url: str, supabase_key: str):
        """Initialize database connection and cache."""
        self.supabase: Client = create_client(supabase_url, supabase_key)

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

    @lru_cache(maxsize=100)
    def get_competition(self, competition_id: str) -> Optional[Competition]:
        """Get competition by ID with caching (sync)."""
        try:
            resp = self.supabase.table('competitions').select('*').eq('id', competition_id).single().execute()
            data = resp.data
            if not data:
                return None
            return self._convert_to_competition(data)
        except Exception as e:
            logger.exception("Error fetching competition %s", competition_id)
            raise DatabaseError(f"Failed to fetch competition: {e}")

    # ---------- Fuzzy search ----------

    def search_players(self, query: str, limit: int = 10) -> List[Player]:
        """Search players by name (sync)."""
        try:
            resp = self.supabase.table('players').select('*').ilike('name', f"%{query}%").limit(limit).execute()
            rows = resp.data or []
            return [self._convert_to_player(r) for r in rows]
        except Exception as e:
            logger.exception("Error searching players: %s", query)
            raise DatabaseError(f"Failed to search players: {e}")

    def search_teams(self, query: str, limit: int = 10) -> List[Team]:
        """Search teams by name (sync)."""
        try:
            resp = self.supabase.table('teams').select('*').ilike('name', f"%{query}%").limit(limit).execute()
            rows = resp.data or []
            return [self._convert_to_team(r) for r in rows]
        except Exception as e:
            logger.exception("Error searching teams: %s", query)
            raise DatabaseError(f"Failed to search teams: {e}")

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
                "tackles", "interceptions", "passes_completed", "clean_sheets", "saves",
                "yellow_cards", "red_cards", "fouls_committed", "fouls_drawn"
            }
            if stat not in allowed_stats:
                return {"status": "not_supported", "reason": f"stat_not_supported:{stat}"}

            qb = (
                self.supabase
                .table("player_match_stats")
                .select(f"{stat}, match_date")
                .eq("player_id", player_id)
                .order("match_date", desc=True)
            )

            if start_date and end_date:
                qb = qb.gte("match_date", start_date).lte("match_date", end_date)
            if venue:
                qb = qb.eq("venue", venue)
            if last_n:
                qb = qb.limit(last_n)

            resp = qb.execute()
            rows = resp.data or []
            value = sum((r.get(stat) or 0) for r in rows)

            return {
                "value": int(value),
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

    # ---------- Convenience: run from ParsedSoccerQuery ----------

    def run_from_parsed(
        self,
        parsed: Any,                        # ParsedSoccerQuery
        player_name_to_id: Optional[Dict[str, str]] = None,
        default_season_label: str = "2024-25"
    ) -> Dict[str, Any]:
        """
        Execute a minimal, happy-path query directly from a ParsedSoccerQuery.
        Scope: single player stat lookup (goals/assists/minutes_played), with season & venue & last N support.
        """
        try:
            # 1) pick a player entity
            player_name = None
            for e in parsed.entities:
                if getattr(e, "entity_type", None) and str(e.entity_type.value) == "player":
                    player_name = e.name
                    break
            if not player_name:
                return {"status": "not_supported", "reason": "no_player_found"}

            # 2) resolve player_id
            pid = None
            if player_name_to_id and player_name.lower() in player_name_to_id:
                pid = player_name_to_id[player_name.lower()]
            else:
                # fallback: try fuzzy search in DB
                players = self.search_players(player_name, limit=1)
                pid = players[0].id if players else None

            if not pid:
                return {"status": "no_data", "reason": "player_not_found"}

            # 3) stat
            stat_map = {
                "goals": "goals",
                "assists": "assists",
                "minutes": "minutes_played",
            }
            stat = stat_map.get((parsed.statistic_requested or "goals"), "goals")

            # 4) time/season
            last_n = None
            start_date, end_date = None, None
            if str(parsed.time_context.value) == "last_n_games":
                # parser里通常会在 filters 里塞数字（若你没加，可自己解析再传进来）
                n = parsed.filters.get("last_n") if isinstance(parsed.filters, dict) else None
                if isinstance(n, int) and n > 0:
                    last_n = n
            elif str(parsed.time_context.value) == "last_season":
                start_date, end_date = self.season_range("last_season")
            else:
                # 默认本赛季
                start_date, end_date = self.season_range(default_season_label)

            # 5) venue
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
                "entity": {"type": "player", "id": pid, "name": player_name},
                "stat": stat,
                "result": result,
                "meta": {
                    "query_intent": parsed.query_intent,
                    "confidence": parsed.confidence,
                },
            }
        except Exception as e:
            logger.exception("run_from_parsed failed")
            return {"status": "db_error", "message": str(e)}

    # ---------- Converters & aggregators ----------

    def _convert_to_player(self, data: Dict[str, Any]) -> Player:
        """Convert database record to Player object."""
        return Player(
            id=str(data['id']),
            name=data['name'],
            common_name=data.get('common_name', data['name']),
            nationality=data.get('nationality') or "",
            birth_date=_safe_parse_iso(data.get('birth_date')),
            position=self._safe_position(data.get('position')),
            height_cm=data.get('height_cm'),
            weight_kg=data.get('weight_kg'),
            team_id=str(data['team_id']) if data.get('team_id') else None,
            jersey_number=data.get('jersey_number'),
            preferred_foot=data.get('preferred_foot'),
            market_value=data.get('market_value')
        )

    def _convert_to_team(self, data: Dict[str, Any]) -> Team:
        """Convert database record to Team object."""
        return Team(
            id=str(data['id']),
            name=data['name'],
            short_name=data.get('short_name', data['name']),
            country=data.get('country') or "",
            founded_year=data.get('founded_year'),
            venue_name=data.get('venue_name'),
            venue_capacity=data.get('venue_capacity'),
            coach_name=data.get('coach_name'),
            logo_url=data.get('logo_url'),
            primary_color=data.get('primary_color'),
            secondary_color=data.get('secondary_color')
        )

    def _convert_to_competition(self, data: Dict[str, Any]) -> Competition:
        """Convert database record to Competition object."""
        return Competition(
            id=str(data['id']),
            name=data['name'],
            short_name=data.get('short_name', data['name']),
            country=data.get('country') or "",
            type=self._safe_competition_type(data.get('type')),
            season=data.get('season') or "",
            start_date=_safe_parse_iso(data.get('start_date')) or datetime.utcnow(),
            end_date=_safe_parse_iso(data.get('end_date')) or datetime.utcnow(),
            current_matchday=data.get('current_matchday'),
            number_of_matchdays=data.get('number_of_matchdays'),
            number_of_teams=data.get('number_of_teams'),
            current_season_id=str(data['current_season_id']) if data.get('current_season_id') else None
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
