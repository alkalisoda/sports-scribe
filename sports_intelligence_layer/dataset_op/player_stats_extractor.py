"""
Player Statistics Extractor

Extracts player statistical data and converts it to historical records format.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, date
from collections import defaultdict

try:
    from .config import (
        RECORD_TYPES, ENTITY_TYPES, PLAYER_STATS, CURRENT_SEASON,
        ENABLE_MILESTONE_DETECTION, is_valid_stat_value,
        get_milestone_context, get_season_context
    )
except ImportError:
    from config import (
        RECORD_TYPES, ENTITY_TYPES, PLAYER_STATS, CURRENT_SEASON,
        ENABLE_MILESTONE_DETECTION, is_valid_stat_value,
        get_milestone_context, get_season_context
    )

logger = logging.getLogger(__name__)


class PlayerStatsExtractor:
    """Extracts and processes player statistics for historical records."""

    def __init__(self):
        """Initialize the player stats extractor."""
        self.logger = logger
        self.processed_players = 0
        self.records_generated = 0
        self.errors_encountered = 0

    def extract_all_player_records(self, players_data: List[Dict[str, Any]],
                                 player_match_stats: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """Extract all types of historical records for players."""
        all_records = []

        self.logger.info(f"Starting extraction for {len(players_data)} players")

        # Extract basic player records from players table
        basic_records = self._extract_player_basic_records(players_data)
        all_records.extend(basic_records)

        # Extract match-based records if available
        if player_match_stats:
            match_records = self._extract_player_match_records(player_match_stats)
            all_records.extend(match_records)

        # Extract milestone records
        if ENABLE_MILESTONE_DETECTION:
            milestone_records = self._extract_milestone_records(players_data)
            all_records.extend(milestone_records)

        self.logger.info(f"Extraction completed: {len(all_records)} records generated for {self.processed_players} players")
        return all_records

    def _extract_player_basic_records(self, players_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract basic player records from players table."""
        records = []

        for player in players_data:
            try:
                player_id = str(player['id'])
                player_name = self._get_player_name(player)
                season = player.get('season_year', CURRENT_SEASON)

                self.logger.debug(f"Processing player: {player_name} (ID: {player_id})")

                # Process each statistic
                for stat_name, stat_config in PLAYER_STATS.items():
                    if stat_name in player and player[stat_name] is not None:
                        stat_value = self._convert_stat_value(player[stat_name])

                        if stat_value is not None and is_valid_stat_value(stat_name, stat_value):
                            # Season total record
                            season_record = self._create_player_record(
                                player_id=player_id,
                                player_name=player_name,
                                stat_name=stat_name,
                                stat_value=stat_value,
                                record_type=RECORD_TYPES['SEASON_TOTAL'],
                                season=str(season),
                                context=get_season_context(str(season), stat_name, stat_value, RECORD_TYPES['SEASON_TOTAL'])
                            )
                            records.append(season_record)

                            # Career total would need aggregation across seasons
                            # For now, we'll treat single season as career if it's the only data we have
                            career_record = self._create_player_record(
                                player_id=player_id,
                                player_name=player_name,
                                stat_name=stat_name,
                                stat_value=stat_value,
                                record_type=RECORD_TYPES['CAREER_TOTAL'],
                                context=get_season_context(str(season), stat_name, stat_value, RECORD_TYPES['CAREER_TOTAL'])
                            )
                            records.append(career_record)

                self.processed_players += 1

            except Exception as e:
                self.logger.error(f"Error processing player {player.get('id', 'unknown')}: {e}")
                self.errors_encountered += 1

        self.records_generated += len(records)
        self.logger.info(f"Generated {len(records)} basic player records")
        return records

    def _extract_player_match_records(self, player_match_stats: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract records from player match statistics."""
        records = []
        player_aggregations = defaultdict(lambda: defaultdict(list))

        # Group match stats by player and statistic
        for match_stat in player_match_stats:
            try:
                player_id = str(match_stat.get('player_id', ''))
                if not player_id:
                    continue

                # Process available match statistics
                match_stats = ['goals', 'assists', 'minutes_played', 'rating', 'shots', 'passes', 'tackles', 'saves']
                for stat_name in match_stats:
                    if stat_name in match_stat and match_stat[stat_name] is not None:
                        stat_value = self._convert_stat_value(match_stat[stat_name])
                        if stat_value is not None and is_valid_stat_value(stat_name, stat_value):
                            player_aggregations[player_id][stat_name].append({
                                'value': stat_value,
                                'match_id': match_stat.get('match_id'),
                                'date': match_stat.get('match_date'),
                                'venue': match_stat.get('venue')
                            })

            except Exception as e:
                self.logger.error(f"Error processing match stat: {e}")
                self.errors_encountered += 1

        # Generate records from aggregated data
        for player_id, stats in player_aggregations.items():
            for stat_name, values in stats.items():
                if values:
                    # Career high (best single match performance)
                    max_performance = max(values, key=lambda x: x['value'])
                    career_high_record = self._create_player_record(
                        player_id=player_id,
                        stat_name=stat_name,
                        stat_value=max_performance['value'],
                        record_type=RECORD_TYPES['CAREER_HIGH'],
                        context=f"Best single match {stat_name}: {max_performance['value']} (Match: {max_performance['match_id']})",
                        date_achieved=max_performance.get('date')
                    )
                    records.append(career_high_record)

                    # Career total from match data
                    total_value = sum(v['value'] for v in values)
                    career_total_record = self._create_player_record(
                        player_id=player_id,
                        stat_name=stat_name,
                        stat_value=total_value,
                        record_type=RECORD_TYPES['CAREER_TOTAL'],
                        context=f"Total {stat_name} from {len(values)} matches: {total_value}"
                    )
                    records.append(career_total_record)

        self.records_generated += len(records)
        self.logger.info(f"Generated {len(records)} match-based player records")
        return records

    def _extract_milestone_records(self, players_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract milestone achievement records."""
        records = []

        for player in players_data:
            try:
                player_id = str(player['id'])
                player_name = self._get_player_name(player)

                for stat_name, stat_config in PLAYER_STATS.items():
                    if stat_name in player and player[stat_name] is not None:
                        stat_value = self._convert_stat_value(player[stat_name])

                        if stat_value is not None and is_valid_stat_value(stat_name, stat_value):
                            # Check for milestone achievements
                            milestones = stat_config.get('milestones', [])
                            for milestone in milestones:
                                if stat_value >= milestone:
                                    milestone_record = self._create_player_record(
                                        player_id=player_id,
                                        player_name=player_name,
                                        stat_name=stat_name,
                                        stat_value=milestone,
                                        record_type=RECORD_TYPES['MILESTONE'],
                                        context=get_milestone_context(stat_name, milestone)
                                    )
                                    records.append(milestone_record)

            except Exception as e:
                self.logger.error(f"Error processing milestones for player {player.get('id', 'unknown')}: {e}")
                self.errors_encountered += 1

        self.records_generated += len(records)
        self.logger.info(f"Generated {len(records)} milestone records")
        return records

    def _create_player_record(self, player_id: str, stat_name: str, stat_value: float,
                            record_type: str, player_name: str = None, season: str = None,
                            context: str = None, date_achieved: Any = None) -> Dict[str, Any]:
        """Create a standardized player historical record."""
        record = {
            'record_type': record_type,
            'entity_type': ENTITY_TYPES['PLAYER'],
            'entity_id': player_id,
            'stat_name': stat_name,
            'stat_value': float(stat_value),
            'verified': True
        }

        # Add optional fields
        if context:
            record['context'] = context
        if season:
            record['season'] = season
        if date_achieved:
            if isinstance(date_achieved, str):
                try:
                    record['date_achieved'] = datetime.fromisoformat(date_achieved.replace('Z', '+00:00')).date()
                except:
                    pass
            elif isinstance(date_achieved, (date, datetime)):
                record['date_achieved'] = date_achieved if isinstance(date_achieved, date) else date_achieved.date()

        return record

    def _get_player_name(self, player: Dict[str, Any]) -> str:
        """Extract player name from player data."""
        first_name = player.get('player_firstname', '')
        last_name = player.get('player_lastname', '')
        full_name = f"{first_name} {last_name}".strip()
        return full_name if full_name else f"Player {player.get('id', 'Unknown')}"

    def _convert_stat_value(self, value: Any) -> Optional[float]:
        """Convert various value types to float."""
        if value is None:
            return None

        try:
            if isinstance(value, (int, float)):
                return float(value)
            elif isinstance(value, str):
                # Handle text-based values (like appearances)
                return float(value)
            else:
                self.logger.warning(f"Unknown value type for stat conversion: {type(value)} - {value}")
                return None
        except (ValueError, TypeError):
            self.logger.warning(f"Could not convert value to float: {value}")
            return None

    def get_processing_summary(self) -> Dict[str, Any]:
        """Get summary of processing statistics."""
        return {
            'players_processed': self.processed_players,
            'records_generated': self.records_generated,
            'errors_encountered': self.errors_encountered,
            'success_rate': (self.processed_players / max(1, self.processed_players + self.errors_encountered)) * 100
        }