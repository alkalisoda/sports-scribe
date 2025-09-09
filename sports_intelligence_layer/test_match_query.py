#!/usr/bin/env python3
"""
Simple test script to test match query functionality
"""

import os
import logging
from dotenv import load_dotenv
from src.query_parser import SoccerQueryParser
from src.database import SoccerDatabase

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_match_query():
    """Test the match query functionality"""
    
    # Load environment variables
    load_dotenv()
    
    # Get Supabase credentials
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
    
    if not supabase_url or not supabase_key:
        logger.error("Supabase credentials not found")
        return
    
    # Initialize components
    parser = SoccerQueryParser()
    database = SoccerDatabase(supabase_url, supabase_key)
    
    # Test query
    query = "Brighton vs Everton match stats"
    logger.info(f"Testing query: {query}")
    
    try:
        # Parse the query
        parsed_query = parser.parse_query(query)
        logger.info(f"Parsed query - Entities: {[(e.name, e.entity_type.value) for e in parsed_query.entities]}")
        
        # Execute the query
        result = database.run_from_parsed(parsed_query)
        logger.info(f"Database result: {result}")
        
        # Check if it's a match result
        if result.get('status') == 'success' and result.get('query_type') == 'match_result':
            match_data = result['match']
            team1 = match_data['team1']
            team2 = match_data['team2']
            winner = match_data['winner']
            score = match_data['score']
            
            if winner == 'team1':
                winner_name = team1['name']
            elif winner == 'team2':
                winner_name = team2['name']
            else:
                winner_name = "Draw"
            
            logger.info(f"✅ SUCCESS: {team1['name']} {score} {team2['name']}")
            logger.info(f"   Winner: {winner_name}")
            logger.info(f"   Match ID: {match_data['match_id']}")
            
            # Log match statistics if available
            if 'statistics' in match_data:
                stats = match_data['statistics']
                logger.info(f"   Match Statistics:")
                logger.info(f"      - Total shots: {stats.get('total_shots', 0)}")
                logger.info(f"      - Total goals: {stats.get('total_goals', 0)}")
                logger.info(f"      - Total cards: {stats.get('total_cards', 0)}")
        else:
            logger.error(f"❌ FAILED: {result}")
            
    except Exception as e:
        logger.error(f"❌ ERROR: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    test_match_query()

