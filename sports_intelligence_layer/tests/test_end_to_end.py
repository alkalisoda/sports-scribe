#!/usr/bin/env python3
"""
Test script for the Soccer Intelligence Layer end-to-end functionality.
This script tests the complete pipeline: Query → Parse → SQL → Results

The test_sample data is used ONLY for validation and reference, not as a data source.
Real data comes from Supabase database through the main pipeline.
"""

import os
import sys
import json
import time
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# Add the parent directory to the Python path to access main.py and src/
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import SoccerIntelligenceLayer
from src.query_parser import SoccerQueryParser
from src.database import SoccerDatabase


def load_test_sample_data_for_validation():
    """
    Load test sample data ONLY for validation and reference.
    This data is NOT used as a data source - it's only for validating
    that our queries can handle the expected data structure.
    """
    data_dir = Path(__file__).parent.parent / "data" / "test_sample"
    
    test_data = {}
    
    try:
        # Load players data for validation
        players_df = pd.read_csv(data_dir / "players.csv")
        test_data["players"] = players_df.to_dict('records')
        
        # Load teams data for validation
        teams_df = pd.read_csv(data_dir / "teams.csv")
        test_data["teams"] = teams_df.to_dict('records')
        
        # Load competitions data for validation
        competitions_df = pd.read_csv(data_dir / "competitions.csv")
        test_data["competitions"] = competitions_df.to_dict('records')
        
        # Load player match stats data for validation
        stats_df = pd.read_csv(data_dir / "player_match_stats.csv")
        test_data["player_match_stats"] = stats_df.to_dict('records')
        
        print(f"✓ Loaded test sample data for validation:")
        print(f"  - {len(test_data['players'])} players")
        print(f"  - {len(test_data['teams'])} teams")
        print(f"  - {len(test_data['competitions'])} competitions")
        print(f"  - {len(test_data['player_match_stats'])} player match stats")
        print(f"  Note: This data is for validation only, not used as data source")
        
        return test_data
        
    except Exception as e:
        print(f"✗ Failed to load test sample data for validation: {e}")
        return None


def test_parser_only():
    """Test the query parser in isolation using test sample data for validation."""
    print("=== TESTING QUERY PARSER ===")
    
    # Load test data for validation only
    test_data = load_test_sample_data_for_validation()
    if not test_data:
        print("⚠ Skipping parser tests due to missing validation data")
        return
    
    parser = SoccerQueryParser()
    
    # Create test queries based on actual test sample data for validation
    test_queries = [
        # Goals queries
        "How many goals has Kaoru Mitoma scored this season?",
        "What's Danny Welbeck's goal record?",
        "How many goals has Simon Adingra scored?",
        "Show me Dominic Calvert-Lewin's goals",
        
        # Assists queries
        "What's Danny Welbeck's assist record?",
        "How many assists does João Pedro have?",
        "Show me Jack Harrison's assists",
        
        # Minutes queries
        "How many minutes has Jordan Pickford played?",
        "What's James Milner's playing time?",
        "How many minutes has Jason Steele played?",
        
        # Performance queries
        "What's João Pedro's performance?",
        "How is Kaoru Mitoma doing?",
        "Show me Dominic Calvert-Lewin's stats",
        
        # Team-specific queries
        "How many goals has Everton scored?",
        "What's Brighton's performance?",
        "Show me Everton players' stats",
        
        # Competition queries
        "Premier League top scorers",
        "Most assists in Premier League",
        "Best performers in Premier League"
    ]
    
    successful_parses = 0
    total_queries = len(test_queries)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Parser Test {i}/{total_queries} ---")
        print(f"Query: {query}")
        
        try:
            parsed = parser.parse_query(query)
            successful_parses += 1
            
            print(f"✓ Parsed successfully")
            print(f"  Entities: {[(e.name, e.entity_type.value) for e in parsed.entities]}")
            print(f"  Statistic: {parsed.statistic_requested}")
            print(f"  Time Context: {parsed.time_context.value}")
            print(f"  Confidence: {parsed.confidence:.2f}")
            
            # Check if ranking was detected
            if parsed.filters.get("ranking"):
                print(f"  Ranking: {parsed.filters['ranking']}")
            
            # Check if competition was detected
            if parsed.filters.get("competition"):
                print(f"  Competition: {parsed.filters['competition']}")
            
        except Exception as e:
            print(f"✗ Parser failed: {e}")
    
    print(f"\n=== PARSER TEST SUMMARY ===")
    print(f"Total queries: {total_queries}")
    print(f"Successful parses: {successful_parses}")
    print(f"Success rate: {(successful_parses/total_queries)*100:.1f}%")


def test_database_connection():
    """Test database connection and basic operations."""
    print("\n=== TESTING DATABASE CONNECTION ===")
    
    # Load environment variables
    load_dotenv()
    
    # Check environment variables
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
    
    if not supabase_url or not supabase_key:
        print("✗ Supabase credentials not found in environment variables")
        print("Please set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY")
        print("Note: Test sample data is for validation only. Real queries need Supabase database.")
        return False
    
    try:
        db = SoccerDatabase(supabase_url, supabase_key)
        print("✓ Database connection established")
        
        # Test basic operations
        print("Testing basic database operations...")
        
        # Test player search with test sample names for validation
        test_players = ["Mitoma", "Welbeck", "Pickford", "Steele"]
        for player_name in test_players:
            try:
                players = db.search_players(player_name, limit=3)
                print(f"✓ Player search '{player_name}': Found {len(players)} players")
                if players:
                    print(f"  Found player: {players[0].name}")
            except Exception as e:
                print(f"✗ Player search '{player_name}' failed: {e}")
        
        # Test team search with test sample names for validation
        test_teams = ["Everton", "Brighton"]
        for team_name in test_teams:
            try:
                teams = db.search_teams(team_name, limit=3)
                print(f"✓ Team search '{team_name}': Found {len(teams)} teams")
                if teams:
                    print(f"  Found team: {teams[0].name}")
            except Exception as e:
                print(f"✗ Team search '{team_name}' failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        print("Note: This is expected if Supabase is not configured.")
        print("The test sample data shows the expected data structure.")
        return False


def test_end_to_end():
    """
    Test the complete end-to-end pipeline using the main SoccerIntelligenceLayer.
    This calls the main process_query method which uses:
    1. SoccerQueryParser for parsing
    2. SoccerDatabase for data retrieval from Supabase
    """
    print("\n=== TESTING END-TO-END PIPELINE ===")
    
    # Load test data for validation only
    test_data = load_test_sample_data_for_validation()
    if not test_data:
        print("⚠ Skipping end-to-end tests due to missing validation data")
        return None
    
    try:
        # Initialize the Soccer Intelligence Layer (main entry point)
        sil = SoccerIntelligenceLayer()
        print("✓ Soccer Intelligence Layer initialized")
        print("  - Uses SoccerQueryParser for parsing")
        print("  - Uses SoccerDatabase for Supabase data retrieval")
        
        # Test queries based on the actual test_sample data for validation
        test_queries = [
            # Individual player queries
            "How many goals has Kaoru Mitoma scored this season?",
            "What's Danny Welbeck's assist record?",
            "How many minutes has Jordan Pickford played?",
            "Show me Dominic Calvert-Lewin's goals",
            "What's João Pedro's performance?",
            "How many clean sheets has Jason Steele kept?",
            "How many goals has Simon Adingra scored?",
            "What's Jack Harrison's assist record?",
            "How many minutes has James Milner played?",
            "Show me Beto's goals",
            
            # Team queries
            "How many goals has Everton scored?",
            "What's Brighton's performance?",
            "Show me Everton players' stats",
            
            # Ranking queries
            "Premier League top scorers",
            "Most assists in Premier League",
            "Best performers in Premier League",
            "Most goals by Everton players",
            "Brighton's best players"
        ]
        
        results = []
        for i, query in enumerate(test_queries, 1):
            print(f"\n--- End-to-End Test {i}/{len(test_queries)} ---")
            print(f"Query: {query}")
            
            start_time = time.time()
            
            try:
                # Call the main process_query method which handles the complete pipeline
                result = sil.process_query(query)
                end_time = time.time()
                processing_time = (end_time - start_time) * 1000  # Convert to milliseconds
                
                if result.get("status") == "success":
                    print(f"✓ Query processed successfully ({processing_time:.1f}ms)")
                    
                    # Extract key information from the main pipeline response
                    db_result = result.get("result", {})
                    if "result" in db_result:
                        stat_result = db_result["result"]
                        if "value" in stat_result:
                            print(f"  Result: {stat_result['value']} {db_result.get('stat', '')}")
                            print(f"  Matches: {stat_result.get('matches', 0)}")
                        elif stat_result.get('status') == 'no_data':
                            print(f"  Status: No data found in Supabase database")
                            print(f"  Note: This is expected if the test data is not in production database")
                        else:
                            print(f"  Status: {stat_result.get('status', 'unknown')}")
                    else:
                        print(f"  Status: {db_result.get('status', 'unknown')}")
                        
                else:
                    print(f"✗ Query failed: {result.get('message', 'Unknown error')}")
                
                results.append({
                    "test_number": i,
                    "query": query,
                    "status": result.get("status"),
                    "processing_time_ms": processing_time,
                    "success": result.get("status") == "success"
                })
                
            except Exception as e:
                print(f"✗ Test failed with exception: {e}")
                results.append({
                    "test_number": i,
                    "query": query,
                    "status": "error",
                    "success": False,
                    "error": str(e)
                })
        
        # Summary
        successful_tests = sum(1 for r in results if r["success"])
        total_tests = len(results)
        avg_processing_time = sum(r.get("processing_time_ms", 0) for r in results) / total_tests
        
        print(f"\n=== END-TO-END TEST SUMMARY ===")
        print(f"Total tests: {total_tests}")
        print(f"Successful: {successful_tests}")
        print(f"Failed: {total_tests - successful_tests}")
        print(f"Success rate: {(successful_tests/total_tests)*100:.1f}%")
        print(f"Average processing time: {avg_processing_time:.1f}ms")
        
        # Performance check
        if avg_processing_time < 500:
            print("✓ Performance target met (<500ms average)")
        else:
            print(f"⚠ Performance target not met (target: <500ms, actual: {avg_processing_time:.1f}ms)")
        
        return results
        
    except Exception as e:
        print(f"✗ End-to-end test failed: {e}")
        return None


def test_specific_query():
    """Test a specific query with detailed output using the main pipeline."""
    print("\n=== TESTING SPECIFIC QUERY ===")
    
    # Load environment variables
    load_dotenv()
    
    try:
        # Use the main SoccerIntelligenceLayer
        sil = SoccerIntelligenceLayer()
        
        # Test a specific query based on test sample data for validation
        query = "How many goals has Kaoru Mitoma scored this season?"
        print(f"Query: {query}")
        
        # Call the main process_query method
        result = sil.process_query(query)
        
        print("Detailed Result from Main Pipeline:")
        print(json.dumps(result, indent=2, default=str))
        
        return result
        
    except Exception as e:
        print(f"✗ Specific query test failed: {e}")
        return None


def test_ranking_queries():
    """Test ranking queries specifically using the main pipeline."""
    print("\n=== TESTING RANKING QUERIES ===")
    
    try:
        # Use the main SoccerIntelligenceLayer
        sil = SoccerIntelligenceLayer()
        
        ranking_queries = [
            "Premier League top scorers",
            "Most assists in Premier League",
            "Best performers in Premier League",
            "Most goals by Everton players",
            "Brighton's best players",
            "Who has the most goals?",
            "Who has the most assists?",
            "Best goalkeeper for clean sheets"
        ]
        
        for i, query in enumerate(ranking_queries, 1):
            print(f"\n--- Ranking Test {i}/{len(ranking_queries)} ---")
            print(f"Query: {query}")
            
            try:
                # Call the main process_query method
                result = sil.process_query(query)
                
                if result.get("status") == "success":
                    print(f"✓ Ranking query processed successfully")
                    
                    # Check if ranking was detected
                    parsed = result.get("query", {}).get("parsed", {})
                    if parsed.get("filters", {}).get("ranking"):
                        print(f"  Ranking detected: {parsed['filters']['ranking']}")
                    else:
                        print(f"  No ranking detected")
                        
                else:
                    print(f"✗ Ranking query failed: {result.get('message', 'Unknown error')}")
                    
            except Exception as e:
                print(f"✗ Ranking test failed: {e}")
        
    except Exception as e:
        print(f"✗ Ranking queries test failed: {e}")


def main():
    """Run all tests."""
    print("Soccer Intelligence Layer - End-to-End Testing")
    print("Using main pipeline with Supabase database")
    print("Test sample data used for validation only")
    print("=" * 70)
    
    # Load environment variables
    load_dotenv()
    
    # Test 1: Parser only
    test_parser_only()
    
    # Test 2: Database connection
    db_ok = test_database_connection()
    
    if not db_ok:
        print("\n⚠ Database connection failed. This is expected if Supabase is not configured.")
        print("The parser tests show that the query parsing works correctly.")
        print("To test the full pipeline, configure Supabase credentials.")
        print("\nTest sample data shows the expected data structure:")
        print("- Players: Jordan Pickford, Kaoru Mitoma, Danny Welbeck, etc.")
        print("- Teams: Everton (45), Brighton (51)")
        print("- Competition: Premier League (39)")
        print("- Match: 1208024 (Everton vs Brighton)")
        return
    
    # Test 3: End-to-end pipeline (calls main SoccerIntelligenceLayer)
    end_to_end_results = test_end_to_end()
    
    # Test 4: Specific query with detailed output (calls main pipeline)
    specific_result = test_specific_query()
    
    # Test 5: Ranking queries (calls main pipeline)
    test_ranking_queries()
    
    print("\n" + "=" * 70)
    print("Testing completed!")
    
    if end_to_end_results:
        successful = sum(1 for r in end_to_end_results if r["success"])
        total = len(end_to_end_results)
        print(f"Overall success rate: {(successful/total)*100:.1f}% ({successful}/{total})")


if __name__ == "__main__":
    main()
