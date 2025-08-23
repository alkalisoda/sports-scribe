#!/usr/bin/env python3
"""Test script for the direct API data collector."""

import asyncio
import logging

from scriber_agents.data_collector import DataCollectorAgent

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_data_collector():
    """Test the direct API data collector."""
    # Initialize the data collector with empty config
    config = {}
    dc = DataCollectorAgent(config)

    print("=" * 60)
    print("Testing Direct API Data Collector")
    print("=" * 60)

    try:
        # Test 1: Game Data Collection
        print("\n1. Testing Game Data Collection...")
        print("-" * 40)
        game_data = await dc.collect_game_data("239625")
        print("✓ Game data collected successfully")
        print(f"  - Results: {game_data.get('results', 'N/A')}")
        print(f"  - Response items: {len(game_data.get('response', []))}")

    except Exception as e:
        print(f"✗ Game data collection failed: {e}")

    try:
        # Test 2: Team Data Collection
        print("\n2. Testing Team Data Collection...")
        print("-" * 40)
        team_data = await dc.collect_team_data("33")
        print("✓ Team data collected successfully")
        print(f"  - Results: {team_data.get('results', 'N/A')}")
        print(f"  - Response items: {len(team_data.get('response', []))}")

    except Exception as e:
        print(f"✗ Team data collection failed: {e}")

    try:
        # Test 3: Player Data Collection
        print("\n3. Testing Player Data Collection...")
        print("-" * 40)
        player_data = await dc.collect_player_data("276", "2023")
        print("✓ Player data collected successfully")
        print(f"  - Results: {player_data.get('results', 'N/A')}")
        print(f"  - Response items: {len(player_data.get('response', []))}")

    except Exception as e:
        print(f"✗ Player data collection failed: {e}")

    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_data_collector())
