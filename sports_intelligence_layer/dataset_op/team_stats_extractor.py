"""
Team Statistics Extractor

Extracts team statistical data and converts it to historical records format.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, date
from collections import defaultdict

try:
    from .config import (
        RECORD_TYPES, ENTITY_TYPES, TEAM_STATS, CURRENT_SEASON,
        get_season_context
    )
except ImportError:
    from config import (
        RECORD_TYPES, ENTITY_TYPES, TEAM_STATS, CURRENT_SEASON,
        get_season_context
    )

logger = logging.getLogger(__name__)


class TeamStatsExtractor:
    """Extracts and processes team statistics for historical records."""

    def __init__(self):
        """Initialize the team stats extractor."""
        self.logger = logger
        self.processed_teams = 0
        self.records_generated = 0
        self.errors_encountered = 0

    def extract_all_team_records(self, teams_data: List[Dict[str, Any]],
                                players_data: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """Extract all types of historical records for teams."""
        all_records = []

        self.logger.info(f"Starting extraction for {len(teams_data)} teams")

        # Extract basic team records
        basic_records = self._extract_team_basic_records(teams_data)
        all_records.extend(basic_records)

        # Extract team aggregated player statistics
        if players_data:
            team_player_records = self._extract_team_player_aggregations(teams_data, players_data)
            all_records.extend(team_player_records)

        # Extract team milestones and achievements
        milestone_records = self._extract_team_milestones(teams_data)
        all_records.extend(milestone_records)

        self.logger.info(f"Extraction completed: {len(all_records)} records generated for {self.processed_teams} teams")
        return all_records

    def _extract_team_basic_records(self, teams_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract basic team records from teams table."""
        records = []

        for team in teams_data:
            try:
                team_id = str(team['id'])
                team_name = team.get('team_name', f"Team {team_id}")
                season = team.get('season_year', CURRENT_SEASON)

                self.logger.debug(f"Processing team: {team_name} (ID: {team_id})")

                # Process team founding year as a historical record
                if 'team_founded' in team and team['team_founded'] is not None:
                    founded_year = team['team_founded']
                    founding_record = self._create_team_record(
                        team_id=team_id,
                        team_name=team_name,
                        stat_name='founded_year',
                        stat_value=float(founded_year),
                        record_type=RECORD_TYPES['MILESTONE'],
                        context=f"Club founded in {founded_year}",
                        date_achieved=date(founded_year, 1, 1) if founded_year > 1800 else None
                    )
                    records.append(founding_record)

                # Add team establishment as a milestone
                if 'team_founded' in team and team['team_founded'] is not None:
                    establishment_record = self._create_team_record(
                        team_id=team_id,
                        team_name=team_name,
                        stat_name='establishment',
                        stat_value=1.0,  # Binary: established
                        record_type=RECORD_TYPES['MILESTONE'],
                        context=f"{team_name} officially established",
                        date_achieved=date(team['team_founded'], 1, 1) if team['team_founded'] > 1800 else None
                    )
                    records.append(establishment_record)

                self.processed_teams += 1

            except Exception as e:
                self.logger.error(f"Error processing team {team.get('id', 'unknown')}: {e}")
                self.errors_encountered += 1

        self.records_generated += len(records)
        self.logger.info(f"Generated {len(records)} basic team records")
        return records

    def _extract_team_player_aggregations(self, teams_data: List[Dict[str, Any]],
                                        players_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract team records based on aggregated player statistics."""
        records = []

        # Group players by team
        team_players = defaultdict(list)
        for player in players_data:
            team_id = str(player.get('team_id', ''))
            if team_id:
                team_players[team_id].append(player)

        # Create team lookup for names
        team_lookup = {str(team['id']): team.get('team_name', f"Team {team['id']}") for team in teams_data}

        # Process each team's player statistics
        for team_id, players in team_players.items():
            if team_id not in team_lookup:
                continue

            team_name = team_lookup[team_id]
            season = players[0].get('season_year', CURRENT_SEASON) if players else CURRENT_SEASON

            try:
                self.logger.debug(f"Processing team aggregations: {team_name} (ID: {team_id})")

                # Aggregate team statistics from players
                team_totals = self._calculate_team_totals(players)

                for stat_name, total_value in team_totals.items():
                    if total_value > 0:
                        # Team season total
                        season_total_record = self._create_team_record(
                            team_id=team_id,
                            team_name=team_name,
                            stat_name=f"team_{stat_name}",
                            stat_value=total_value,
                            record_type=RECORD_TYPES['SEASON_TOTAL'],
                            season=str(season),
                            context=f"Team total {stat_name} in {season}: {total_value} (from {len(players)} players)"
                        )
                        records.append(season_total_record)

                        # Team high (if it's the best recorded)
                        team_high_record = self._create_team_record(
                            team_id=team_id,
                            team_name=team_name,
                            stat_name=f"team_{stat_name}",
                            stat_value=total_value,
                            record_type=RECORD_TYPES['TEAM_RECORD'],
                            season=str(season),
                            context=f"Team record for {stat_name}: {total_value} in {season} season"
                        )
                        records.append(team_high_record)

                # Squad size record
                squad_size_record = self._create_team_record(
                    team_id=team_id,
                    team_name=team_name,
                    stat_name='squad_size',
                    stat_value=float(len(players)),
                    record_type=RECORD_TYPES['SEASON_TOTAL'],
                    season=str(season),
                    context=f"Squad size in {season}: {len(players)} players"
                )
                records.append(squad_size_record)

            except Exception as e:
                self.logger.error(f"Error processing team aggregations for {team_id}: {e}")
                self.errors_encountered += 1

        self.records_generated += len(records)
        self.logger.info(f"Generated {len(records)} team aggregation records")
        return records

    def _extract_team_milestones(self, teams_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract team milestone records."""
        records = []
        current_year = datetime.now().year

        for team in teams_data:
            try:
                team_id = str(team['id'])
                team_name = team.get('team_name', f"Team {team_id}")

                # Anniversary milestones
                if 'team_founded' in team and team['team_founded'] is not None:
                    founded_year = team['team_founded']
                    age = current_year - founded_year

                    # Common anniversary milestones
                    milestones = [10, 25, 50, 75, 100, 125, 150]
                    for milestone in milestones:
                        if age >= milestone:
                            anniversary_record = self._create_team_record(
                                team_id=team_id,
                                team_name=team_name,
                                stat_name='anniversary',
                                stat_value=float(milestone),
                                record_type=RECORD_TYPES['MILESTONE'],
                                context=f"{team_name} {milestone}th anniversary milestone",
                                date_achieved=date(founded_year + milestone, 1, 1)
                            )
                            records.append(anniversary_record)

                # Century mark (if founded before 1924 and still active)
                if 'team_founded' in team and team['team_founded'] is not None:
                    if team['team_founded'] <= 1924:  # 100+ years old
                        century_record = self._create_team_record(
                            team_id=team_id,
                            team_name=team_name,
                            stat_name='century_club',
                            stat_value=1.0,
                            record_type=RECORD_TYPES['MILESTONE'],
                            context=f"{team_name} is a century-old football club (founded {team['team_founded']})"
                        )
                        records.append(century_record)

            except Exception as e:
                self.logger.error(f"Error processing team milestones for {team.get('id', 'unknown')}: {e}")
                self.errors_encountered += 1

        self.records_generated += len(records)
        self.logger.info(f"Generated {len(records)} team milestone records")
        return records

    def _calculate_team_totals(self, players: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate team totals from player statistics."""
        totals = defaultdict(float)

        for player in players:
            # Sum up player statistics
            stats_to_sum = ['goals', 'assists', 'appearances']
            for stat in stats_to_sum:
                if stat in player and player[stat] is not None:
                    try:
                        value = float(player[stat]) if isinstance(player[stat], str) else player[stat]
                        if value is not None:
                            totals[stat] += value
                    except (ValueError, TypeError):
                        continue

            # Count players with ratings (for average calculation)
            if 'rating' in player and player['rating'] is not None:
                try:
                    rating = float(player['rating']) if isinstance(player['rating'], str) else player['rating']
                    if rating is not None and rating > 0:
                        totals['total_rating'] += rating
                        totals['rated_players'] += 1
                except (ValueError, TypeError):
                    continue

        # Calculate average rating
        if totals['rated_players'] > 0:
            totals['average_rating'] = totals['total_rating'] / totals['rated_players']

        # Remove helper fields
        if 'total_rating' in totals:
            del totals['total_rating']
        if 'rated_players' in totals:
            del totals['rated_players']

        return dict(totals)

    def _create_team_record(self, team_id: str, team_name: str, stat_name: str, stat_value: float,
                          record_type: str, season: str = None, context: str = None,
                          date_achieved: Any = None) -> Dict[str, Any]:
        """Create a standardized team historical record."""
        record = {
            'record_type': record_type,
            'entity_type': ENTITY_TYPES['TEAM'],
            'entity_id': team_id,
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

    def get_processing_summary(self) -> Dict[str, Any]:
        """Get summary of processing statistics."""
        return {
            'teams_processed': self.processed_teams,
            'records_generated': self.records_generated,
            'errors_encountered': self.errors_encountered,
            'success_rate': (self.processed_teams / max(1, self.processed_teams + self.errors_encountered)) * 100
        }