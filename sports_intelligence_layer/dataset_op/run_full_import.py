#!/usr/bin/env python3
"""
Run full historical data import with environment loading
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv
import time

def main():
    """Run complete historical data import."""
    print("SportsScribe Historical Records - Full Import")
    print("=" * 60)

    # Load environment variables from .env file
    env_file = Path(__file__).parent.parent / '.env'
    print(f"Loading environment from: {env_file}")
    load_dotenv(env_file)

    # Verify environment variables
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')

    if not supabase_url or not supabase_key:
        print("[ERROR] Missing environment variables:")
        print("  SUPABASE_URL:", "Found" if supabase_url else "Missing")
        print("  SUPABASE_SERVICE_ROLE_KEY:", "Found" if supabase_key else "Missing")
        return 1

    try:
        # Import modules
        from historical_processor import HistoricalProcessor

        print(f"Supabase URL: {supabase_url}")
        print("Supabase Key: [REDACTED]")

        # Create processor
        processor = HistoricalProcessor(supabase_url, supabase_key)

        # Start full processing
        print("\nStarting full historical data import...")
        print("This may take several minutes depending on data size...")

        start_time = time.time()

        result = processor.process_all_historical_data(
            include_players=True,
            include_teams=True,
            include_player_matches=True,
            clear_existing=False  # Don't clear existing data
        )

        end_time = time.time()

        print("\n" + "=" * 60)
        print("IMPORT COMPLETED")
        print("=" * 60)

        if result['success']:
            print(f"[SUCCESS] Import completed successfully!")
            print(f"Processing time: {result['processing_time_formatted']}")
            print(f"Records inserted: {result['total_records_inserted']}")
            print(f"Total errors: {result['total_errors']}")

            # Player statistics
            player_stats = result['player_processing']
            print(f"\nPlayer Processing:")
            print(f"  Players processed: {player_stats['players_processed']}")
            print(f"  Records generated: {player_stats['records_generated']}")
            print(f"  Errors: {player_stats['errors_encountered']}")

            # Team statistics
            team_stats = result['team_processing']
            print(f"\nTeam Processing:")
            print(f"  Teams processed: {team_stats['teams_processed']}")
            print(f"  Records generated: {team_stats['records_generated']}")
            print(f"  Errors: {team_stats['errors_encountered']}")

            # Database statistics
            db_stats = result['database_stats']
            print(f"\nDatabase Statistics:")
            print(f"  Success rate: {db_stats.get('success_rate', 0):.1f}%")

            return 0
        else:
            print(f"[ERROR] Import completed with errors")
            print(f"Total errors: {result['total_errors']}")
            return 1

    except Exception as e:
        print(f"[ERROR] Exception during import: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    try:
        exit_code = main()
        print(f"\nImport finished with exit code: {exit_code}")
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nImport interrupted by user")
        sys.exit(1)