#!/usr/bin/env python3
"""
Test script for the Soccer Intelligence Layer end-to-end functionality.
This script tests the complete pipeline: Query → Parse → SQL → Results
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

# Add the parent directory to the Python path to access main.py and src/
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import SoccerIntelligenceLayer
from src.query_parser import SoccerQueryParser
from src.database import SoccerDatabase


def test_parser_only() -> None:
    """Test the query parser in isolation."""
    print("=== TESTING QUERY PARSER ===")

    parser = SoccerQueryParser()

    test_queries = [
        "How many goals has Kaoru Mitoma scored this season?",
        "What's Danny Welbeck's assist record?",
        "How many minutes has Jordan Pickford played?",
        "Show me Dominic Calvert-Lewin's goals in the last 5 games",
        "What's João Pedro's performance at home?",
        "How many clean sheets has Jason Steele kept?",
        "How many goals has Simon Adingra scored?",
        "What's Jack Harrison's assist record?",
        "How many minutes has James Milner played?",
        "Show me Beto's goals in the last 5 games",
        "How many goals does James have?",
        "Show me Salah's goals, assists, and yellow cards this season",
        "What are the top 3 scorers' goals, minutes played, and shots on target?",
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Parser Test {i}/{len(test_queries)} ---")
        print(f"Query: {query}")

        try:
            parsed = parser.parse_query(query)
            print("✓ Parsed successfully")
            print(
                f"  Entities: {[(e.name, e.entity_type.value) for e in parsed.entities]}"
            )
            print(f"  Statistic: {parsed.statistic_requested}")
            print(f"  Time Context: {parsed.time_context.value}")
            print(f"  Confidence: {parsed.confidence:.2f}")

        except Exception as e:
            print(f"✗ Parser failed: {e}")


def test_database_connection() -> bool:
    """Test database connection and basic operations."""
    print("\n=== TESTING DATABASE CONNECTION ===")

    # Load environment variables
    load_dotenv()

    # Check environment variables
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

    if not supabase_url or not supabase_key:
        print("✗ Supabase credentials not found in environment variables")
        print("Please set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY")
        return False

    try:
        db = SoccerDatabase(supabase_url, supabase_key)
        print("✓ Database connection established")

        # Test basic operations
        print("Testing basic database operations...")

        # Test player search
        players = db.search_players("Mitoma", limit=3)
        print(f"✓ Player search: Found {len(players)} players")
        if players:
            print(f"  Found player: {players[0].name}")

        # Test team search
        teams = db.search_teams("Brighton", limit=3)
        print(f"✓ Team search: Found {len(teams)} teams")
        if teams:
            print(f"  Found team: {teams[0].name}")

        return True

    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        return False


def test_end_to_end() -> list[dict[str, Any]] | None:
    """Test the complete end-to-end pipeline."""
    print("\n=== TESTING END-TO-END PIPELINE ===")

    try:
        # Initialize the Soccer Intelligence Layer
        sil = SoccerIntelligenceLayer()
        print("✓ Soccer Intelligence Layer initialized")

        # Test queries based on the actual test_sample data
        test_queries = [
            "How many goals has Kaoru Mitoma scored this season?",
            "What's Danny Welbeck's assist record?",
            "How many minutes has Jordan Pickford played?",
            "Show me Dominic Calvert-Lewin's goals in the last 5 games",
            "What's João Pedro's performance at home?",
            "How many clean sheets has Jason Steele kept?",
            "How many goals has Simon Adingra scored?",
            "What's Jack Harrison's assist record?",
            "How many minutes has James Milner played?",
            "Show me Beto's goals in the last 5 games",
        ]

        results = []
        for i, query in enumerate(test_queries, 1):
            print(f"\n--- End-to-End Test {i}/{len(test_queries)} ---")
            print(f"Query: {query}")

            start_time = time.time()

            try:
                result = sil.process_query(query)
                end_time = time.time()
                processing_time = (
                    end_time - start_time
                ) * 1000  # Convert to milliseconds

                if result.get("status") == "success":
                    print(f"✓ Query processed successfully ({processing_time:.1f}ms)")

                    # Extract key information
                    db_result = result.get("result", {})
                    if "result" in db_result:
                        stat_result = db_result["result"]
                        if "value" in stat_result:
                            print(
                                f"  Result: {stat_result['value']} {db_result.get('stat', '')}"
                            )
                            print(f"  Matches: {stat_result.get('matches', 0)}")
                        elif stat_result.get("status") == "no_data":
                            print("  Status: No data found in database")
                        else:
                            print(f"  Status: {stat_result.get('status', 'unknown')}")
                    else:
                        print(f"  Status: {db_result.get('status', 'unknown')}")

                else:
                    print(f"✗ Query failed: {result.get('message', 'Unknown error')}")

                results.append(
                    {
                        "test_number": i,
                        "query": query,
                        "status": result.get("status"),
                        "processing_time_ms": processing_time,
                        "success": result.get("status") == "success",
                    }
                )

            except Exception as e:
                print(f"✗ Test failed with exception: {e}")
                results.append(
                    {
                        "test_number": i,
                        "query": query,
                        "status": "error",
                        "success": False,
                        "error": str(e),
                    }
                )

        # Summary
        successful_tests = sum(1 for r in results if r["success"])
        total_tests = len(results)
        avg_processing_time = (
            sum(r.get("processing_time_ms", 0) for r in results) / total_tests
        )

        print("\n=== END-TO-END TEST SUMMARY ===")
        print(f"Total tests: {total_tests}")
        print(f"Successful: {successful_tests}")
        print(f"Failed: {total_tests - successful_tests}")
        print(f"Success rate: {(successful_tests/total_tests)*100:.1f}%")
        print(f"Average processing time: {avg_processing_time:.1f}ms")

        # Performance check
        if avg_processing_time < 500:
            print("✓ Performance target met (<500ms average)")
        else:
            print(
                f"⚠ Performance target not met (target: <500ms, actual: {avg_processing_time:.1f}ms)"
            )

        return results

    except Exception as e:
        print(f"✗ End-to-end test failed: {e}")
        return None


def test_specific_query() -> dict[str, Any] | None:
    """Test a specific query with detailed output."""
    print("\n=== TESTING SPECIFIC QUERY ===")

    # Load environment variables
    load_dotenv()

    try:
        sil = SoccerIntelligenceLayer()

        # Test a specific query
        query = "How many goals has Kaoru Mitoma scored this season?"
        print(f"Query: {query}")

        result = sil.process_query(query)

        print("Detailed Result:")
        print(json.dumps(result, indent=2, default=str))

        return result

    except Exception as e:
        print(f"✗ Specific query test failed: {e}")
        return None


def main() -> None:
    """Run all tests."""
    print("Soccer Intelligence Layer - End-to-End Testing")
    print("=" * 50)

    # Load environment variables
    load_dotenv()

    # Test 1: Parser only
    test_parser_only()

    # Test 2: Database connection
    db_ok = test_database_connection()

    if not db_ok:
        print("\n⚠ Database connection failed. Skipping end-to-end tests.")
        print("Please ensure your Supabase credentials are correct.")
        return

    # Test 3: End-to-end pipeline
    end_to_end_results = test_end_to_end()

    # Test 4: Specific query with detailed output
    test_specific_query()

    print("\n" + "=" * 50)
    print("Testing completed!")

    if end_to_end_results:
        successful = sum(1 for r in end_to_end_results if r["success"])
        total = len(end_to_end_results)
        print(
            f"Overall success rate: {(successful/total)*100:.1f}% ({successful}/{total})"
        )


if __name__ == "__main__":
    main()
