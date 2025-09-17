"""
Historical Data Processor

Main coordinator for extracting and processing historical statistics data.
"""

import logging
import time
from typing import Dict, List, Any, Optional
from datetime import datetime

try:
    from .database_manager import DatabaseManager
    from .player_stats_extractor import PlayerStatsExtractor
    from .team_stats_extractor import TeamStatsExtractor
    from .config import BATCH_SIZE, OVERWRITE_EXISTING
except ImportError:
    from database_manager import DatabaseManager
    from player_stats_extractor import PlayerStatsExtractor
    from team_stats_extractor import TeamStatsExtractor
    from config import BATCH_SIZE, OVERWRITE_EXISTING

logger = logging.getLogger(__name__)


class HistoricalProcessor:
    """Main processor for historical statistics data."""

    def __init__(self, supabase_url: str, supabase_key: str):
        """Initialize the historical processor."""
        self.db_manager = DatabaseManager(supabase_url, supabase_key)
        self.player_extractor = PlayerStatsExtractor()
        self.team_extractor = TeamStatsExtractor()
        self.logger = logger

        # Processing statistics
        self.start_time = None
        self.end_time = None
        self.total_records_processed = 0
        self.total_records_inserted = 0
        self.total_errors = 0

    def process_all_historical_data(self, include_players: bool = True, include_teams: bool = True,
                                  include_player_matches: bool = True, clear_existing: bool = False) -> Dict[str, Any]:
        """Process all historical data from database tables."""
        self.start_time = time.time()
        self.logger.info("Starting historical data processing...")

        try:
            # Test database connection
            if not self.db_manager.test_connection():
                raise Exception("Database connection failed")

            # Clear existing data if requested
            if clear_existing:
                self._clear_existing_data()

            # Get existing records count for comparison
            existing_counts = self.db_manager.get_existing_historical_records_count()
            self.logger.info(f"Existing historical records: {existing_counts}")

            all_records = []

            # Process players
            if include_players:
                self.logger.info("Processing player statistics...")
                player_records = self._process_players(include_player_matches)
                all_records.extend(player_records)

            # Process teams
            if include_teams:
                self.logger.info("Processing team statistics...")
                team_records = self._process_teams()
                all_records.extend(team_records)

            # Filter out duplicates if not overwriting
            if not OVERWRITE_EXISTING:
                all_records = self._filter_existing_records(all_records)

            # Insert records in batches
            if all_records:
                self.logger.info(f"Inserting {len(all_records)} records into database...")
                self.total_records_inserted = self.db_manager.insert_historical_records_batch(
                    all_records, BATCH_SIZE
                )
            else:
                self.logger.info("No new records to insert")

            self.end_time = time.time()
            return self._generate_processing_summary()

        except Exception as e:
            self.logger.error(f"Error in historical data processing: {e}")
            self.end_time = time.time()
            self.total_errors += 1
            return self._generate_processing_summary()

    def _process_players(self, include_match_stats: bool = True) -> List[Dict[str, Any]]:
        """Process all player-related historical data."""
        records = []

        try:
            # Get player data
            players_data = self.db_manager.get_all_players()
            self.logger.info(f"Retrieved {len(players_data)} players from database")

            # Get player match stats if requested
            player_match_stats = None
            if include_match_stats:
                player_match_stats = self.db_manager.get_player_match_stats()
                self.logger.info(f"Retrieved {len(player_match_stats)} player match stats")

            # Extract player records
            if players_data:
                player_records = self.player_extractor.extract_all_player_records(
                    players_data, player_match_stats
                )
                records.extend(player_records)

                # Log processing summary
                player_summary = self.player_extractor.get_processing_summary()
                self.logger.info(f"Player processing summary: {player_summary}")

        except Exception as e:
            self.logger.error(f"Error processing players: {e}")
            self.total_errors += 1

        return records

    def _process_teams(self) -> List[Dict[str, Any]]:
        """Process all team-related historical data."""
        records = []

        try:
            # Get team data
            teams_data = self.db_manager.get_all_teams()
            self.logger.info(f"Retrieved {len(teams_data)} teams from database")

            # Get player data for team aggregations
            players_data = self.db_manager.get_all_players()

            # Extract team records
            if teams_data:
                team_records = self.team_extractor.extract_all_team_records(
                    teams_data, players_data
                )
                records.extend(team_records)

                # Log processing summary
                team_summary = self.team_extractor.get_processing_summary()
                self.logger.info(f"Team processing summary: {team_summary}")

        except Exception as e:
            self.logger.error(f"Error processing teams: {e}")
            self.total_errors += 1

        return records

    def _filter_existing_records(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter out records that already exist in the database."""
        if not records:
            return records

        filtered_records = []
        skipped_count = 0

        for record in records:
            try:
                exists = self.db_manager.check_existing_record(
                    record['entity_type'],
                    record['entity_id'],
                    record['stat_name'],
                    record['record_type']
                )

                if not exists:
                    filtered_records.append(record)
                else:
                    skipped_count += 1

            except Exception as e:
                self.logger.error(f"Error checking existing record: {e}")
                # Include the record if we can't check (safer approach)
                filtered_records.append(record)

        self.logger.info(f"Filtered {skipped_count} existing records, {len(filtered_records)} new records to insert")
        return filtered_records

    def _clear_existing_data(self):
        """Clear existing historical records (with safety checks)."""
        self.logger.warning("Clearing existing historical records...")

        try:
            # Clear by entity type for safety
            player_deleted = self.db_manager.clear_historical_records(entity_type='player')
            team_deleted = self.db_manager.clear_historical_records(entity_type='team')

            self.logger.info(f"Cleared {player_deleted} player records and {team_deleted} team records")

        except Exception as e:
            self.logger.error(f"Error clearing existing data: {e}")
            raise

    def _generate_processing_summary(self) -> Dict[str, Any]:
        """Generate a comprehensive processing summary."""
        processing_time = (self.end_time - self.start_time) if self.start_time and self.end_time else 0

        # Get individual processor summaries
        player_summary = self.player_extractor.get_processing_summary()
        team_summary = self.team_extractor.get_processing_summary()
        db_summary = self.db_manager.get_statistics_summary()

        summary = {
            'processing_time_seconds': round(processing_time, 2),
            'processing_time_formatted': f"{int(processing_time // 60)}m {int(processing_time % 60)}s",
            'total_records_processed': self.total_records_processed,
            'total_records_inserted': self.total_records_inserted,
            'total_errors': self.total_errors,
            'player_processing': player_summary,
            'team_processing': team_summary,
            'database_stats': db_summary,
            'timestamp': datetime.now().isoformat(),
            'success': self.total_errors == 0
        }

        # Log final summary
        self.logger.info("=" * 60)
        self.logger.info("HISTORICAL DATA PROCESSING COMPLETED")
        self.logger.info("=" * 60)
        self.logger.info(f"Processing Time: {summary['processing_time_formatted']}")
        self.logger.info(f"Records Inserted: {self.total_records_inserted}")
        self.logger.info(f"Errors Encountered: {self.total_errors}")
        self.logger.info(f"Players Processed: {player_summary['players_processed']}")
        self.logger.info(f"Teams Processed: {team_summary['teams_processed']}")
        self.logger.info(f"Success Rate: {db_summary.get('success_rate', 0):.1f}%")
        self.logger.info("=" * 60)

        return summary

    def test_processing(self, limit_players: int = 5, limit_teams: int = 3) -> Dict[str, Any]:
        """Run a limited test of the processing pipeline."""
        self.logger.info(f"Starting test processing (max {limit_players} players, {limit_teams} teams)...")

        try:
            # Test database connection
            if not self.db_manager.test_connection():
                raise Exception("Database connection failed")

            # Get limited data for testing
            all_players = self.db_manager.get_all_players()
            all_teams = self.db_manager.get_all_teams()

            test_players = all_players[:limit_players] if all_players else []
            test_teams = all_teams[:limit_teams] if all_teams else []

            self.logger.info(f"Test data: {len(test_players)} players, {len(test_teams)} teams")

            # Process test data
            test_records = []

            if test_players:
                player_records = self.player_extractor.extract_all_player_records(test_players)
                test_records.extend(player_records)

            if test_teams:
                team_records = self.team_extractor.extract_all_team_records(test_teams, test_players)
                test_records.extend(team_records)

            self.logger.info(f"Generated {len(test_records)} test records")

            # Show sample records
            if test_records:
                self.logger.info("Sample records:")
                for i, record in enumerate(test_records[:3]):
                    self.logger.info(f"  {i+1}. {record['entity_type']} {record['entity_id']} - {record['stat_name']}: {record['stat_value']}")

            return {
                'success': True,
                'test_records_generated': len(test_records),
                'sample_records': test_records[:5]  # Return first 5 for inspection
            }

        except Exception as e:
            self.logger.error(f"Test processing failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }