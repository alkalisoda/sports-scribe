"""
Database Manager for Historical Records Processing

Handles database connections and operations for historical statistics import.
For reading historical data, use SoccerDatabase from src.database module.
"""

import os
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, date
from supabase import create_client, Client
import json

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database connections and operations for historical records."""

    def __init__(self, supabase_url: str, supabase_key: str):
        """Initialize database connection."""
        self.supabase: Client = create_client(supabase_url, supabase_key)
        self.logger = logger
        self.stats_processed = 0
        self.errors_encountered = 0

    def test_connection(self) -> bool:
        """Test database connection."""
        try:
            # Test with a simple query
            response = self.supabase.table('historical_records').select('id').limit(1).execute()
            self.logger.info("Database connection successful")
            return True
        except Exception as e:
            self.logger.error(f"Database connection failed: {e}")
            return False

    def get_all_players(self) -> List[Dict[str, Any]]:
        """Retrieve all players from the players table."""
        try:
            response = self.supabase.table('players').select(
                'id, player_firstname, player_lastname, goals, assists, rating, appearances, team_id, season_year'
            ).execute()

            players = response.data or []
            self.logger.info(f"Retrieved {len(players)} players from database")
            return players
        except Exception as e:
            self.logger.error(f"Error retrieving players: {e}")
            return []

    def get_all_teams(self) -> List[Dict[str, Any]]:
        """Retrieve all teams from the teams table."""
        try:
            response = self.supabase.table('teams').select(
                'id, team_name, team_code, team_country, team_founded, league_id, season_year'
            ).execute()

            teams = response.data or []
            self.logger.info(f"Retrieved {len(teams)} teams from database")
            return teams
        except Exception as e:
            self.logger.error(f"Error retrieving teams: {e}")
            return []

    def get_player_match_stats(self) -> List[Dict[str, Any]]:
        """Retrieve all player match statistics."""
        try:
            response = self.supabase.table('player_match_stats').select('*').execute()

            stats = response.data or []
            self.logger.info(f"Retrieved {len(stats)} player match stats from database")
            return stats
        except Exception as e:
            self.logger.error(f"Error retrieving player match stats: {e}")
            return []

    def insert_historical_record(self, record: Dict[str, Any]) -> bool:
        """Insert a single historical record."""
        try:
            # Ensure all required fields are present
            required_fields = ['record_type', 'entity_type', 'entity_id', 'stat_name', 'stat_value']
            for field in required_fields:
                if field not in record:
                    self.logger.error(f"Missing required field: {field}")
                    return False

            # Convert date to string if it's a date object
            if 'date_achieved' in record and isinstance(record['date_achieved'], date):
                record['date_achieved'] = record['date_achieved'].isoformat()

            response = self.supabase.table('historical_records').insert(record).execute()

            if response.data:
                self.stats_processed += 1
                self.logger.debug(f"Inserted historical record: {record['entity_type']} {record['entity_id']} - {record['stat_name']}")
                return True
            else:
                self.logger.error(f"Failed to insert record: {record}")
                self.errors_encountered += 1
                return False

        except Exception as e:
            self.logger.error(f"Error inserting historical record: {e}")
            self.logger.error(f"Record data: {record}")
            self.errors_encountered += 1
            return False

    def insert_historical_records_batch(self, records: List[Dict[str, Any]], batch_size: int = 50) -> int:
        """Insert multiple historical records in batches."""
        total_inserted = 0

        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]

            try:
                # Process dates in batch
                for record in batch:
                    if 'date_achieved' in record and isinstance(record['date_achieved'], date):
                        record['date_achieved'] = record['date_achieved'].isoformat()

                response = self.supabase.table('historical_records').insert(batch).execute()

                if response.data:
                    batch_inserted = len(response.data)
                    total_inserted += batch_inserted
                    self.stats_processed += batch_inserted
                    self.logger.info(f"Inserted batch {i//batch_size + 1}: {batch_inserted} records")
                else:
                    self.logger.error(f"Failed to insert batch {i//batch_size + 1}")
                    self.errors_encountered += len(batch)

            except Exception as e:
                self.logger.error(f"Error inserting batch {i//batch_size + 1}: {e}")
                self.errors_encountered += len(batch)

        self.logger.info(f"Total records inserted: {total_inserted}")
        return total_inserted

    def check_existing_record(self, entity_type: str, entity_id: str, stat_name: str, record_type: str) -> bool:
        """Check if a historical record already exists."""
        try:
            response = self.supabase.table('historical_records').select('id').eq(
                'entity_type', entity_type
            ).eq('entity_id', entity_id).eq('stat_name', stat_name).eq('record_type', record_type).execute()

            return len(response.data or []) > 0
        except Exception as e:
            self.logger.error(f"Error checking existing record: {e}")
            return False

    def clear_historical_records(self, entity_type: Optional[str] = None, record_type: Optional[str] = None) -> int:
        """Clear historical records (use with caution)."""
        try:
            query = self.supabase.table('historical_records').delete()

            if entity_type:
                query = query.eq('entity_type', entity_type)
            if record_type:
                query = query.eq('record_type', record_type)

            # Add a safety check - only delete if we have specific filters
            if not entity_type and not record_type:
                self.logger.warning("Refusing to delete all historical records without filters")
                return 0

            response = query.execute()
            deleted_count = len(response.data or [])
            self.logger.info(f"Deleted {deleted_count} historical records")
            return deleted_count

        except Exception as e:
            self.logger.error(f"Error clearing historical records: {e}")
            return 0

    def get_statistics_summary(self) -> Dict[str, Any]:
        """Get summary of processing statistics."""
        return {
            'stats_processed': self.stats_processed,
            'errors_encountered': self.errors_encountered,
            'success_rate': (self.stats_processed / max(1, self.stats_processed + self.errors_encountered)) * 100
        }

    def get_existing_historical_records_count(self) -> Dict[str, int]:
        """Get count of existing historical records by type."""
        try:
            response = self.supabase.table('historical_records').select('entity_type, record_type').execute()

            records = response.data or []
            counts = {}

            for record in records:
                key = f"{record['entity_type']}_{record['record_type']}"
                counts[key] = counts.get(key, 0) + 1

            return counts
        except Exception as e:
            self.logger.error(f"Error getting historical records count: {e}")
            return {}