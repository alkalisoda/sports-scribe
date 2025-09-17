#!/usr/bin/env python3
"""
Check the results of historical data import
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

def main():
    """Check the results of the import."""
    print("Checking Historical Records Import Results")
    print("=" * 50)

    # Load environment variables
    env_file = Path(__file__).parent.parent / '.env'
    load_dotenv(env_file)

    try:
        from database_manager import DatabaseManager

        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')

        db_manager = DatabaseManager(supabase_url, supabase_key)

        if db_manager.test_connection():
            print("[OK] Database connection established")

            # Check existing historical records
            existing_counts = db_manager.get_existing_historical_records_count()
            print(f"\nHistorical records in database:")

            if existing_counts:
                for record_type, count in existing_counts.items():
                    print(f"  {record_type}: {count}")
                total_records = sum(existing_counts.values())
                print(f"  TOTAL: {total_records}")
            else:
                print("  No historical records found")

            # Check source data
            players = db_manager.get_all_players()
            teams = db_manager.get_all_teams()
            print(f"\nSource data:")
            print(f"  Players: {len(players)}")
            print(f"  Teams: {len(teams)}")

            # Sample data
            if len(players) > 0:
                sample_player = players[0]
                print(f"\nSample player data:")
                print(f"  Name: {sample_player.get('player_firstname', '')} {sample_player.get('player_lastname', '')}")
                print(f"  Goals: {sample_player.get('goals', 'N/A')}")
                print(f"  Assists: {sample_player.get('assists', 'N/A')}")
                print(f"  Team ID: {sample_player.get('team_id', 'N/A')}")

            if len(teams) > 0:
                sample_team = teams[0]
                print(f"\nSample team data:")
                print(f"  Name: {sample_team.get('team_name', 'N/A')}")
                print(f"  Founded: {sample_team.get('team_founded', 'N/A')}")
                print(f"  Country: {sample_team.get('team_country', 'N/A')}")

            return 0
        else:
            print("[ERROR] Database connection failed")
            return 1

    except Exception as e:
        print(f"[ERROR] Exception: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)