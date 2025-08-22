"""Test script for NarrativePlanner agent."""

import asyncio
import json
import logging
import os
import sys
from typing import Any

# Add the parent directory to the path so we can import the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_sample_compact_data() -> dict[str, Any]:
    """Create sample compact game data for testing."""
    return {
        "match_info": {
            "home_team": "Manchester United",
            "away_team": "Liverpool",
            "score": "2-1",
            "venue": "Old Trafford",
            "date": "2024-01-15",
            "competition": "Premier League",
        },
        "events": [
            {
                "type": "Goal",
                "player": "Marcus Rashford",
                "time": "23",
                "team": "Manchester United",
                "detail": "Assisted by Bruno Fernandes",
            },
            {
                "type": "Goal",
                "player": "Mohamed Salah",
                "time": "67",
                "team": "Liverpool",
                "detail": "Penalty kick",
            },
            {
                "type": "Goal",
                "player": "Rasmus Højlund",
                "time": "89",
                "team": "Manchester United",
                "detail": "Last-minute winner",
            },
        ],
        "players": [
            {
                "name": "Marcus Rashford",
                "team": "Manchester United",
                "position": "Forward",
                "rating": 8.5,
                "goals": 1,
                "assists": 0,
            },
            {
                "name": "Rasmus Højlund",
                "team": "Manchester United",
                "position": "Forward",
                "rating": 8.0,
                "goals": 1,
                "assists": 0,
            },
            {
                "name": "Mohamed Salah",
                "team": "Liverpool",
                "position": "Forward",
                "rating": 7.5,
                "goals": 1,
                "assists": 0,
            },
        ],
        "statistics": [
            {
                "team": "Manchester United",
                "possession": "45%",
                "shots": 12,
                "shots_on_target": 5,
                "corners": 6,
            },
            {
                "team": "Liverpool",
                "possession": "55%",
                "shots": 15,
                "shots_on_target": 7,
                "corners": 8,
            },
        ],
        "lineups": [
            {
                "team": "Manchester United",
                "formation": "4-3-3",
                "startXI": [
                    "Onana",
                    "Dalot",
                    "Varane",
                    "Evans",
                    "Shaw",
                    "Casemiro",
                    "Mainoo",
                    "Fernandes",
                    "Rashford",
                    "Højlund",
                    "Garnacho",
                ],
            },
            {
                "team": "Liverpool",
                "formation": "4-3-3",
                "startXI": [
                    "Alisson",
                    "Alexander-Arnold",
                    "Van Dijk",
                    "Konaté",
                    "Robertson",
                    "Szoboszlai",
                    "Mac Allister",
                    "Jones",
                    "Salah",
                    "Núñez",
                    "Díaz",
                ],
            },
        ],
    }


def create_sample_research_data() -> dict[str, Any]:
    """Create sample research data for testing."""
    return {
        "game_analysis": [
            "Manchester United secured a dramatic 2-1 victory over Liverpool with a last-minute winner from Rasmus Højlund",
            "The game was evenly contested with Liverpool dominating possession but United being more clinical in front of goal",
            "Marcus Rashford opened the scoring in the 23rd minute with a well-taken finish",
            "Mohamed Salah equalized from the penalty spot in the 67th minute",
            "Rasmus Højlund scored the winning goal in the 89th minute, securing three crucial points for United",
        ],
        "player_performance": [
            "Marcus Rashford was United's standout performer with a goal and excellent work rate",
            "Rasmus Højlund showed great composure to score the winning goal under pressure",
            "Mohamed Salah was Liverpool's most dangerous player and converted his penalty with confidence",
            "Bruno Fernandes provided the assist for Rashford's opening goal",
        ],
        "historical_context": [
            "This was the 200th meeting between Manchester United and Liverpool in all competitions",
            "United had lost their previous three matches against Liverpool",
            "The victory moves United closer to the top four in the Premier League table",
            "Liverpool remain in the title race despite this setback",
        ],
    }


async def test_narrative_planner():
    """Test the NarrativePlanner functionality."""
    logger.info("Starting NarrativePlanner test")

    try:
        # Import the NarrativePlanner
        from scriber_agents.narrative_planner import NarrativePlanner

        # Initialize the narrative planner with configuration
        config = {"model": "gpt-4o", "temperature": 0.7}

        logger.info("Initializing NarrativePlanner...")
        narrative_planner = NarrativePlanner(config)
        logger.info("NarrativePlanner initialized successfully")

        # Create sample data
        logger.info("Creating sample data...")
        compact_data = create_sample_compact_data()
        research_data = create_sample_research_data()
        logger.info("Sample data created successfully")

        # Test narrative selection
        logger.info("Testing narrative selection...")
        narrative_selection = await narrative_planner.select_narrative(
            compact_data, research_data
        )

        logger.info("Narrative selection completed successfully")
        logger.info(
            f"Primary narrative: {narrative_selection.get('primary_narrative', 'Unknown')}"
        )
        logger.info(
            f"Storytelling focus: {narrative_selection.get('storytelling_focus', 'Unknown')}"
        )

        # Print the full narrative selection
        print("\n" + "=" * 60)
        print("NARRATIVE SELECTION RESULTS")
        print("=" * 60)
        print(json.dumps(narrative_selection, indent=2, ensure_ascii=False))
        print("=" * 60)

        # Test narrative strength analysis
        logger.info("Testing narrative strength analysis...")
        strength_analysis = await narrative_planner.analyze_narrative_strength(
            narrative_selection
        )

        logger.info("Narrative strength analysis completed successfully")

        # Print the strength analysis
        print("\n" + "=" * 60)
        print("NARRATIVE STRENGTH ANALYSIS")
        print("=" * 60)
        print(json.dumps(strength_analysis, indent=2, ensure_ascii=False))
        print("=" * 60)

        logger.info("All tests completed successfully!")
        return True

    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error(
            "Make sure you're running this from the correct directory and the modules are available"
        )
        return False
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


async def test_basic_functionality():
    """Test basic functionality without API calls."""
    logger.info("Testing basic functionality...")

    try:
        from scriber_agents.narrative_planner import NarrativePlanner

        # Test initialization
        config = {"model": "gpt-4o", "temperature": 0.7}
        planner = NarrativePlanner(config)

        # Test fallback narrative creation
        fallback = planner._create_fallback_narrative("Test error")

        # Test validation
        planner._validate_narrative_selection(fallback)

        logger.info("Basic functionality test passed!")
        return True

    except Exception as e:
        logger.error(f"Basic functionality test failed: {e}")
        return False


async def main():
    """Main test function."""
    print("=" * 60)
    print("NARRATIVE PLANNER TEST SUITE")
    print("=" * 60)

    # Test basic functionality first
    basic_success = await test_basic_functionality()

    if basic_success:
        # Test full functionality
        full_success = await test_narrative_planner()

        if full_success:
            print("\n" + "=" * 60)
            print("✅ ALL TESTS PASSED!")
            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print("❌ FULL FUNCTIONALITY TEST FAILED")
            print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ BASIC FUNCTIONALITY TEST FAILED")
        print("=" * 60)


if __name__ == "__main__":
    # Run the tests
    asyncio.run(main())
