#!/usr/bin/env python3
"""
Validate historical records import results and data quality
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

def main():
    """Validate the imported historical records."""
    print("Historical Records Import - Data Quality Validation")
    print("=" * 60)

    # Load environment variables
    env_file = Path(__file__).parent.parent / '.env'
    load_dotenv(env_file)

    try:
        from database_manager import DatabaseManager

        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')

        db_manager = DatabaseManager(supabase_url, supabase_key)

        if not db_manager.test_connection():
            print("[ERROR] Database connection failed")
            return 1

        print("[OK] Database connection established")

        # Get summary statistics
        existing_counts = db_manager.get_existing_historical_records_count()

        if not existing_counts:
            print("[ERROR] No historical records found in database")
            return 1

        print(f"\nImported historical records summary:")
        total_records = 0
        for record_type, count in existing_counts.items():
            print(f"  {record_type}: {count}")
            total_records += count
        print(f"  TOTAL: {total_records}")

        # Query some sample records for validation
        print(f"\nValidating data quality...")

        # Test specific queries to validate record structure
        response = db_manager.supabase.table('historical_records').select('*').limit(5).execute()
        sample_records = response.data or []

        if not sample_records:
            print("[ERROR] No sample records found")
            return 1

        print(f"\nSample records ({len(sample_records)}):")
        for i, record in enumerate(sample_records, 1):
            print(f"  {i}. ID: {record.get('id', 'N/A')}")
            print(f"     Type: {record.get('record_type', 'N/A')}")
            print(f"     Entity: {record.get('entity_type', 'N/A')} {record.get('entity_id', 'N/A')}")
            print(f"     Stat: {record.get('stat_name', 'N/A')} = {record.get('stat_value', 'N/A')}")
            if record.get('context'):
                print(f"     Context: {record.get('context')}")
            if record.get('season'):
                print(f"     Season: {record.get('season')}")
            print()

        # Validate record types
        expected_record_types = ['season_total', 'career_total', 'milestone', 'team_record']
        print("Validating record types:")
        for record_type in expected_record_types:
            count = sum(1 for k in existing_counts.keys() if record_type in k)
            if count > 0:
                print(f"  [OK] {record_type}: Found")
            else:
                print(f"  [WARN] {record_type}: Not found")

        # Validate entity types
        print("\nValidating entity types:")
        player_records = sum(1 for k in existing_counts.keys() if 'player' in k)
        team_records = sum(1 for k in existing_counts.keys() if 'team' in k)

        print(f"  Player records: {player_records}")
        print(f"  Team records: {team_records}")

        if player_records > 0:
            print("  [OK] Player records found")
        else:
            print("  [WARN] No player records found")

        if team_records > 0:
            print("  [OK] Team records found")
        else:
            print("  [WARN] No team records found")

        # Check for data consistency
        print("\nData consistency checks:")

        # Check for records with valid stat values
        response = db_manager.supabase.table('historical_records').select('stat_value').execute()
        all_records = response.data or []

        if all_records:
            stat_values = [r.get('stat_value') for r in all_records if r.get('stat_value') is not None]
            if stat_values:
                min_val = min(stat_values)
                max_val = max(stat_values)
                avg_val = sum(stat_values) / len(stat_values)
                print(f"  Stat values range: {min_val} to {max_val} (avg: {avg_val:.2f})")
                print(f"  [OK] {len(stat_values)} records with valid stat values")

                # Check for reasonable ranges
                if min_val >= 0:
                    print("  [OK] All stat values are non-negative")
                else:
                    print(f"  [WARN] Found negative stat values (min: {min_val})")

                if max_val <= 10000:  # Reasonable upper bound
                    print("  [OK] All stat values are within reasonable range")
                else:
                    print(f"  [WARN] Found very high stat values (max: {max_val})")
            else:
                print("  [WARN] No valid stat values found")

        # Check for verified records
        response = db_manager.supabase.table('historical_records').select('verified').eq('verified', True).execute()
        verified_records = response.data or []
        verified_count = len(verified_records)

        print(f"  Verified records: {verified_count}/{total_records} ({verified_count/total_records*100:.1f}%)")

        if verified_count == total_records:
            print("  [OK] All records are marked as verified")
        else:
            print(f"  [WARN] {total_records - verified_count} records are not verified")

        # Success summary
        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)
        print(f"[OK] Total records imported: {total_records}")
        print(f"[OK] Record types: {len(set(k.split('_')[-1] for k in existing_counts.keys()))}")
        print(f"[OK] Entity types: {2 if player_records > 0 and team_records > 0 else 1}")
        print(f"[OK] Data quality: {'PASS' if verified_count == total_records else 'PARTIAL'}")

        print("\n[SUCCESS] Historical records import validation completed!")
        print("\nThe historical_records table now contains meaningful statistical data")
        print("that can be used for sports intelligence queries and analysis.")

        return 0

    except Exception as e:
        print(f"[ERROR] Exception during validation: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)