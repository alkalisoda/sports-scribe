#!/usr/bin/env python3
"""
Debug script to check Brighton team data in the database.
"""
import os
from dotenv import load_dotenv
from src.database import SoccerDatabase

# Load environment variables
load_dotenv()

def main():
    # Initialize database
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
    
    if not supabase_url or not supabase_key:
        print("❌ Missing Supabase credentials")
        return
    
    db = SoccerDatabase(supabase_url, supabase_key)
    
    print("🔍 Debugging Brighton team data...\n")
    
    # Check all teams in database
    try:
        teams_response = db.supabase.table("teams").select("id, name").execute()
        print(f"📊 Found {len(teams_response.data)} teams in database:")
        for team in teams_response.data[:10]:  # Show first 10
            print(f"   • {team['name']} (ID: {team['id']})")
        print()
        
        # Look for Brighton variations
        brighton_teams = [team for team in teams_response.data if 'brighton' in team['name'].lower()]
        print(f"🔍 Brighton variations found: {len(brighton_teams)}")
        for team in brighton_teams:
            print(f"   • {team['name']} (ID: {team['id']})")
        print()
        
    except Exception as e:
        print(f"❌ Error querying teams: {e}")
        return
    
    # Test get_team_players with different Brighton names
    test_names = ["Brighton", "Brighton & Hove Albion", "Brighton and Hove Albion"]
    
    for name in test_names:
        print(f"🔍 Testing team name: '{name}'")
        players = db.get_team_players(name)
        print(f"   Found {len(players)} players")
        if players:
            print(f"   Sample players:")
            for player in players[:3]:  # Show first 3
                print(f"      • {player['name']} (ID: {player['id']})")
        print()

if __name__ == "__main__":
    main()
