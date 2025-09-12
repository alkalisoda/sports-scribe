"""Test script for narrative planner integration in Research Agent."""

import asyncio
import json
import logging
import os
import sys
from typing import Any

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()


def create_sample_game_data() -> dict[str, Any]:
    """Create sample game data for testing."""
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
                "player": "Rasmus Hojlund",
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
                "name": "Rasmus Hojlund",
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
                    "Onana", "Dalot", "Varane", "Evans", "Shaw",
                    "Casemiro", "Mainoo", "Fernandes", 
                    "Rashford", "Hojlund", "Garnacho",
                ],
            },
            {
                "team": "Liverpool",
                "formation": "4-3-3", 
                "startXI": [
                    "Alisson", "Alexander-Arnold", "Van Dijk", "Konaté", "Robertson",
                    "Szoboszlai", "Mac Allister", "Jones",
                    "Salah", "Núñez", "Díaz",
                ],
            },
        ],
    }


async def test_narrative_planner_integration():
    """Test the enhanced Research Agent with narrative planner."""
    logger.info("Testing narrative planner integration")
    
    try:
        from scriber_agents.researcher import ResearchAgent
        
        # Initialize Research Agent
        config = {
            "model": "gpt-4o",
            "temperature": 0.7,
            "narrative_model": "gpt-4o", 
            "narrative_temperature": 0.6
        }
        
        logger.info("Initializing Research Agent with narrative planner...")
        research_agent = ResearchAgent(config)
        logger.info("Research Agent initialized successfully")
        
        # Create sample game data
        game_data = create_sample_game_data()
        logger.info("Sample game data created")
        
        # Test traditional storyline generation
        logger.info("Testing traditional storyline generation...")
        storylines = await research_agent.get_storyline_from_game_data(game_data)
        logger.info(f"Generated {len(storylines)} storylines")
        
        # Test enhanced research with narrative planning
        logger.info("Testing enhanced research with narrative planning...")
        enhanced_result = await research_agent.get_enhanced_research_with_narrative(game_data)
        
        logger.info("Enhanced research completed successfully!")
        
        # Display results
        print("\n" + "=" * 80)
        print("NARRATIVE PLANNER INTEGRATION TEST RESULTS")
        print("=" * 80)
        
        print(f"\nPROCESSING METADATA:")
        for key, value in enhanced_result.processing_metadata.items():
            print(f"  {key}: {value}")
        
        print(f"\nTRADITIONAL STORYLINES ({len(enhanced_result.analysis.storylines)}):")
        for i, storyline in enumerate(enhanced_result.analysis.storylines, 1):
            print(f"  {i}. {storyline}")
        
        print(f"\nNARRATIVE PLAN:")
        print(f"  Primary Narrative: {enhanced_result.narrative_plan.primary_narrative}")
        print(f"  Storytelling Focus: {enhanced_result.narrative_plan.storytelling_focus}")
        print(f"  Narrative Style: {enhanced_result.narrative_plan.narrative_style}")
        print(f"  Target Audience: {enhanced_result.narrative_plan.target_audience}")
        print(f"  Confidence: {enhanced_result.narrative_plan.confidence}")
        
        print(f"\nPRIORITIZED STORYLINES ({len(enhanced_result.narrative_plan.prioritized_storylines)}):")
        for sl in enhanced_result.narrative_plan.prioritized_storylines:
            print(f"  Priority {sl.priority}: {sl.content}")
            print(f"    └─ Angle: {sl.narrative_angle} | Appeal: {sl.audience_appeal} | Type: {sl.story_type}")
        
        print("\n" + "=" * 80)
        logger.info("All tests completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


async def test_fallback_mechanisms():
    """Test fallback mechanisms when AI fails."""
    logger.info("Testing fallback mechanisms")
    
    try:
        from scriber_agents.researcher import ResearchAgent
        
        # Test with mock storylines
        research_agent = ResearchAgent()
        
        # Test fallback narrative plan creation
        test_storylines = [
            "Manchester United secured a dramatic 2-1 victory over Liverpool",
            "Rasmus Hojlund scored the winning goal in the 89th minute", 
            "Marcus Rashford opened the scoring in the first half"
        ]
        
        fallback_plan = research_agent._create_fallback_narrative_plan(test_storylines)
        
        logger.info("Fallback mechanism test completed")
        print(f"\nFALLBACK NARRATIVE PLAN:")
        print(f"  Primary Narrative: {fallback_plan.primary_narrative}")
        print(f"  Confidence: {fallback_plan.confidence}")
        print(f"  Prioritized storylines: {len(fallback_plan.prioritized_storylines)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Fallback test failed: {e}")
        return False


async def main():
    """Main test function."""
    print("NARRATIVE PLANNER INTEGRATION TEST SUITE")
    print("=" * 80)
    
    # Check for required environment variables
    if not os.getenv("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY not found - some tests may fail")
    
    # Test fallback mechanisms first (no API required)
    fallback_success = await test_fallback_mechanisms()
    
    if os.getenv("OPENAI_API_KEY"):
        # Test full integration with API
        integration_success = await test_narrative_planner_integration()
        
        if integration_success and fallback_success:
            print("\nALL TESTS PASSED!")
        else:
            print("\nSOME TESTS FAILED")
    else:
        if fallback_success:
            print("\nFallback tests passed (API tests skipped - no OPENAI_API_KEY)")
        else:
            print("\nFallback tests failed")


if __name__ == "__main__":
    asyncio.run(main())