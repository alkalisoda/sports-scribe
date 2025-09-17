#!/usr/bin/env python3
"""
Run limited historical data import for testing with current data
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

def main():
    """Run limited historical data import."""
    print("SportsScribe Historical Records - Limited Import")
    print("=" * 60)

    # Load environment variables
    env_file = Path(__file__).parent.parent / '.env'
    print(f"Loading environment from: {env_file}")
    load_dotenv(env_file)

    try:
        from historical_processor import HistoricalProcessor

        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')

        # Create processor
        processor = HistoricalProcessor(supabase_url, supabase_key)

        print("\nRunning limited import (50 players, 20 teams)...")

        # Get limited data first
        from database_manager import DatabaseManager
        db_manager = DatabaseManager(supabase_url, supabase_key)

        # Get limited datasets
        all_players = db_manager.get_all_players()
        all_teams = db_manager.get_all_teams()

        print(f"Total available: {len(all_players)} players, {len(all_teams)} teams")

        # Process limited data
        limited_players = all_players[:50]  # First 50 players
        limited_teams = all_teams[:20]      # First 20 teams

        print(f"Processing: {len(limited_players)} players, {len(limited_teams)} teams")

        # Process players
        if limited_players:
            print("\nProcessing player statistics...")
            from player_stats_extractor import PlayerStatsExtractor
            player_extractor = PlayerStatsExtractor()

            player_records = player_extractor.extract_all_player_records(limited_players)
            print(f"Generated {len(player_records)} player records")

            # Insert player records
            if player_records:
                inserted = db_manager.insert_historical_records_batch(player_records, 25)
                print(f"Inserted {inserted} player records")

        # Process teams
        if limited_teams:
            print("\nProcessing team statistics...")
            from team_stats_extractor import TeamStatsExtractor
            team_extractor = TeamStatsExtractor()

            team_records = team_extractor.extract_all_team_records(limited_teams, limited_players)
            print(f"Generated {len(team_records)} team records")

            # Insert team records
            if team_records:
                inserted = db_manager.insert_historical_records_batch(team_records, 25)
                print(f"Inserted {inserted} team records")

        # Check final results
        print("\nChecking final results...")
        existing_counts = db_manager.get_existing_historical_records_count()

        if existing_counts:
            print("Historical records created:")
            for record_type, count in existing_counts.items():
                print(f"  {record_type}: {count}")
            total_records = sum(existing_counts.values())
            print(f"  TOTAL: {total_records}")
        else:
            print("No historical records were created")

        print("\n[SUCCESS] Limited import completed!")
        return 0

    except Exception as e:
        print(f"[ERROR] Exception during import: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nImport interrupted by user")
        sys.exit(1)