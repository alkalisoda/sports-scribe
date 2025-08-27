"""
Main entry point for the Soccer Intelligence Layer (Async Optimized).
Demonstrates the complete end-to-end flow: Query → Parse → SQL → Results
With enhanced performance through async patterns and concurrent execution.
"""

import os
import logging
import time
import asyncio
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv

from src.query_parser import SoccerQueryParser, ParsedSoccerQuery
from src.database import SoccerDatabase, DatabaseError

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('soccer_intelligence.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)


class SoccerIntelligenceLayer:
    """
    Main class that orchestrates the complete end-to-end flow:
    Query → Parse → SQL → Results
    """
    
    def __init__(self, supabase_url: Optional[str] = None, supabase_key: Optional[str] = None):
        """
        Initialize the Soccer Intelligence Layer.
        
        Args:
            supabase_url: Supabase project URL
            supabase_key: Supabase service role key
        """
        logger.info("INITIALIZING SOCCER INTELLIGENCE LAYER")
        logger.info("   Loading environment variables...")
        
        # Load environment variables
        load_dotenv()
        logger.info("   Environment variables loaded successfully")
        
        # Get Supabase credentials
        logger.info("   Getting Supabase credentials...")
        self.supabase_url = supabase_url or os.getenv('SUPABASE_URL')
        self.supabase_key = supabase_key or os.getenv('SUPABASE_SERVICE_ROLE_KEY')
        
        if not self.supabase_url or not self.supabase_key:
            logger.error("   Supabase credentials not found")
            raise ValueError(
                "Supabase credentials not found. Please set SUPABASE_URL and "
                "SUPABASE_SERVICE_ROLE_KEY environment variables or pass them directly."
            )
        
        logger.info("   Supabase credentials obtained successfully")
        logger.info(f"   Supabase URL: {self.supabase_url[:30]}...")
        
        # Initialize components
        logger.info("   Initializing SoccerQueryParser...")
        self.parser = SoccerQueryParser()
        logger.info("   SoccerQueryParser initialized successfully")
        
        logger.info("   Initializing SoccerDatabase...")
        self.database = SoccerDatabase(self.supabase_url, self.supabase_key)
        logger.info("   SoccerDatabase initialized successfully")
        
        logger.info("SOCCER INTELLIGENCE LAYER INITIALIZED SUCCESSFULLY")
        logger.info("   Components ready:")
        logger.info("      - SoccerQueryParser: Ready")
        logger.info("      - SoccerDatabase: Ready")
        logger.info("   Ready to process queries!")
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Sync wrapper for the async process_query_async method.
        
        Args:
            query: Natural language query
            
        Returns:
            Dictionary containing the complete result with metadata
        """
        return asyncio.run(self.process_query_async(query))
    
    async def process_query_async(self, query: str) -> Dict[str, Any]:
        """
        Process a natural language soccer query through the complete async pipeline.
        
        Args:
            query: Natural language query (e.g., "How many goals has Haaland scored this season?")
            
        Returns:
            Dictionary containing the complete result with metadata
        """
        logger.info("=" * 80)
        logger.info(f"STARTING MAIN PIPELINE PROCESS")
        logger.info(f"INPUT QUERY: '{query}'")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        try:
            # Step 1: Parse the query
            logger.info("STEP 1: QUERY PARSING")
            logger.info("   - Initializing SoccerQueryParser...")
            logger.info("   - Calling parser.parse_query()...")
            
            parsed_query = self.parser.parse_query(query)
            
            logger.info("   Query parsing completed successfully")
            logger.info(f"   Parsing Results:")
            logger.info(f"      - Confidence: {parsed_query.confidence:.2f}")
            logger.info(f"      - Entities found: {len(parsed_query.entities)}")
            logger.info(f"      - Statistic requested: {parsed_query.statistic_requested}")
            logger.info(f"      - Time context: {parsed_query.time_context.value}")
            logger.info(f"      - Query intent: {parsed_query.query_intent}")
            
            if parsed_query.entities:
                for i, entity in enumerate(parsed_query.entities, 1):
                    logger.info(f"      - Entity {i}: {entity.name} ({entity.entity_type.value}, conf: {entity.confidence:.2f})")
            
            if parsed_query.filters:
                logger.info(f"      - Filters: {parsed_query.filters}")
            
            # Step 2: Execute the query against the database (async)
            logger.info("STEP 2: DATABASE QUERY EXECUTION (ASYNC)")
            logger.info("   - Using async SoccerDatabase connection...")
            logger.info("   - Calling database.run_from_parsed_async()...")
            
            result = await self.database.run_from_parsed_async(parsed_query)
            
            logger.info("   Database query execution completed")
            logger.info(f"   Database Results:")
            logger.info(f"      - Result status: {result.get('status', 'unknown')}")
            if 'result' in result:
                db_result = result['result']
                logger.info(f"      - Database result type: {type(db_result).__name__}")
                if isinstance(db_result, dict):
                    logger.info(f"      - Result keys: {list(db_result.keys())}")
            
            # Step 3: Format the response
            logger.info("STEP 3: RESPONSE FORMATTING")
            logger.info("   - Calling _format_response()...")
            
            response = self._format_response(query, parsed_query, result)
            
            end_time = time.time()
            processing_time = (end_time - start_time) * 1000
            
            logger.info("   Response formatting completed")
            logger.info(f"   Final Response:")
            logger.info(f"      - Status: {response.get('status')}")
            logger.info(f"      - Processing time: {processing_time:.1f}ms")
            logger.info(f"      - Data source: {response.get('metadata', {}).get('data_source')}")
            
            logger.info("=" * 80)
            logger.info(f"MAIN PIPELINE COMPLETED SUCCESSFULLY")
            logger.info(f"Total processing time: {processing_time:.1f}ms")
            logger.info("=" * 80)
            
            return response
            
        except Exception as e:
            end_time = time.time()
            processing_time = (end_time - start_time) * 1000
            
            logger.error("=" * 80)
            logger.error(f"MAIN PIPELINE FAILED")
            logger.error(f"Error: {e}")
            logger.error(f"Processing time before failure: {processing_time:.1f}ms")
            logger.error("=" * 80)
            
            return {
                "status": "error",
                "message": str(e),
                "query": query,
                "timestamp": self._get_timestamp(),
                "processing_time_ms": processing_time
            }
    
    def _format_response(self, original_query: str, parsed_query: ParsedSoccerQuery, 
                        db_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format the final response with all relevant information.
        """
        logger.info("   Formatting response structure...")
        
        # Format entities
        formatted_entities = []
        for entity in parsed_query.entities:
            formatted_entities.append({
                "name": entity.name,
                "type": entity.entity_type.value,
                "confidence": entity.confidence
            })
        
        logger.info(f"   Formatted {len(formatted_entities)} entities")
        
        # Create parsed query structure
        parsed_structure = {
            "entities": formatted_entities,
            "time_context": parsed_query.time_context.value,
            "statistic_requested": parsed_query.statistic_requested,  # Backward compatibility
            "statistics_requested": getattr(parsed_query, 'statistics_requested', []),  # New multiple stats support
            "comparison_type": parsed_query.comparison_type.value if parsed_query.comparison_type else None,
            "filters": parsed_query.filters,
            "intent": parsed_query.query_intent,
            "confidence": parsed_query.confidence
        }
        
        logger.info(f"   Parsed structure created with {len(parsed_structure)} fields")
        
        # Create metadata
        metadata = {
            "timestamp": self._get_timestamp(),
            "processing_time_ms": 0,  # Will be updated by caller
            "data_source": "supabase"
        }
        
        logger.info("   Metadata created")
        
        # Assemble final response
        response = {
            "status": "success",
            "query": {
                "original": original_query,
                "parsed": parsed_structure
            },
            "result": db_result,
            "metadata": metadata
        }
        
        logger.info(f"   Final response assembled with {len(response)} main sections")
        
        return response
    
    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.utcnow().isoformat()
    
    async def process_multiple_queries_async(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Process multiple queries concurrently for improved performance."""
        logger.info("=" * 80)
        logger.info(f"STARTING CONCURRENT PIPELINE PROCESS")
        logger.info(f"INPUT QUERIES: {len(queries)} queries")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        # Create tasks for concurrent execution
        tasks = [self.process_query_async(query) for query in queries]
        
        # Execute all queries concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Query {i+1} failed: {result}")
                processed_results.append({
                    "status": "error",
                    "message": str(result),
                    "query": queries[i],
                    "timestamp": self._get_timestamp(),
                    "processing_time_ms": 0
                })
            else:
                processed_results.append(result)
        
        execution_time = time.time() - start_time
        
        logger.info("=" * 80)
        logger.info(f"CONCURRENT PIPELINE COMPLETED")
        logger.info(f"Total execution time: {execution_time*1000:.1f}ms")
        logger.info(f"Average time per query: {execution_time*1000/len(queries):.1f}ms")
        logger.info("=" * 80)
        
        return processed_results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics from the database layer."""
        return self.database.get_performance_stats()
    
    def reset_performance_stats(self):
        """Reset performance statistics."""
        self.database.reset_performance_stats()



async def test_async_performance():
    """Test async performance improvements."""
    logger.info("=" * 80)
    logger.info("ASYNC PERFORMANCE TEST")
    logger.info("=" * 80)
    
    try:
        # Initialize the Soccer Intelligence Layer
        sil = SoccerIntelligenceLayer()
        
        # Reset performance stats
        sil.reset_performance_stats()
        
        # Test queries for concurrent execution
        test_queries = [
            "How many goals has Kaoru Mitoma scored?",
            "What's Danny Welbeck's assist record?",
            "How many goals did Danny Welbeck score?",
            "What are Kaoru Mitoma's stats?",
            "Show me Salah's goals, assists, and yellow cards this season",
            "Who scored the most goals for Brighton?",
            "Most assists by Brighton players",
            "Everton players goals",
            "Brighton vs Everton match stats",
            "Abdoulaye Doucouré shots on target"
        ]
        
        logger.info(f"Testing concurrent execution of {len(test_queries)} queries...")
        
        # Test concurrent execution
        start_time = time.time()
        results = await sil.process_multiple_queries_async(test_queries)
        concurrent_time = time.time() - start_time
        
        logger.info("CONCURRENT EXECUTION RESULTS:")
        logger.info(f"   Total time: {concurrent_time*1000:.1f}ms")
        logger.info(f"   Average per query: {concurrent_time*1000/len(test_queries):.1f}ms")
        
        # Show success/failure stats
        successful_queries = sum(1 for r in results if r.get('status') == 'success')
        logger.info(f"   Successful queries: {successful_queries}/{len(test_queries)}")
        
        # Get performance stats
        perf_stats = sil.get_performance_stats()
        logger.info("DATABASE PERFORMANCE STATS:")
        logger.info(f"   Total queries: {perf_stats.get('total_queries', 0)}")
        logger.info(f"   Concurrent queries: {perf_stats.get('concurrent_queries', 0)}")
        logger.info(f"   Average query time: {perf_stats.get('average_query_time', 0)*1000:.1f}ms")
        
        logger.info("=" * 80)
        logger.info("ASYNC PERFORMANCE TEST COMPLETED")
        logger.info("=" * 80)
        
        return results
        
    except Exception as e:
        logger.error(f"Async performance test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return []

def main():
    """
    Main function to demonstrate the end-to-end functionality with detailed logging.
    """
    logger.info("=" * 80)
    logger.info("STARTING MAIN SOCCER INTELLIGENCE LAYER DEMO")
    logger.info("This will show detailed logs for every step of the pipeline")
    logger.info("=" * 80)
    
    try:
        # Initialize the Soccer Intelligence Layer
        logger.info("Initializing Soccer Intelligence Layer...")
        sil = SoccerIntelligenceLayer()
        
        # Test queries based on test_sample data - using actual data from CSV
        test_queries = [
            "How many goals has Kaoru Mitoma scored?",  # Should find 1 goal
            "What's Danny Welbeck's assist record?",    # Should find 1 assist
            "How many goals did Danny Welbeck score?",  # Should find 1 goal
            "What are Kaoru Mitoma's stats?",           # Should find goals, shots, etc.
            "Show me Salah's goals, assists, and yellow cards this season",  # Test multiple statistics
            "Who scored the most goals for Brighton?",  # Should find Kaoru Mitoma (1 goal)
            "Most assists by Brighton players",         # Should find multiple players with 1 assist each
            "Everton players goals",                    # Should find Everton players
            "Brighton vs Everton match stats",          # Should find match 1208024 data
            "Abdoulaye Doucouré shots on target",             # Should find 3 shots on target
            "Jordan Pickford performance"                 # Should find 1 goal, 1 assist
        ]
        
        logger.info(f"Running {len(test_queries)} test queries...")
        
        for i, query in enumerate(test_queries, 1):
            logger.info("=" * 80)
            logger.info(f"TEST {i}/{len(test_queries)}")
            logger.info(f"Query: {query}")
            logger.info("=" * 80)
            
            try:
                # Process the query
                result = sil.process_query(query)
                
                # Display results summary
                logger.info("RESULTS SUMMARY:")
                logger.info(f"   Status: {result.get('status')}")
                logger.info(f"   Processing time: {result.get('metadata', {}).get('processing_time_ms', 0):.1f}ms")
                
                if result.get('status') == 'success':
                    parsed = result.get('query', {}).get('parsed', {})
                    logger.info(f"   Confidence: {parsed.get('confidence', 0):.2f}")
                    logger.info(f"   Entities found: {len(parsed.get('entities', []))}")
                    logger.info(f"   Statistic: {parsed.get('statistic_requested')}")
                    
                    db_result = result.get('result', {})
                    
                    # Check if it's a match query result
                    if 'query_type' in db_result and db_result['query_type'] == 'match_result':
                        match_data = db_result['match']
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
                        
                        logger.info(f"   Match Result: {team1['name']} {score} {team2['name']}")
                        logger.info(f"   Winner: {winner_name}")
                        logger.info(f"   Match ID: {match_data['match_id']}")
                        
                        # Log match statistics if available
                        if 'statistics' in match_data:
                            stats = match_data['statistics']
                            logger.info(f"   Match Statistics:")
                            logger.info(f"      - Total shots: {stats.get('total_shots', 0)}")
                            logger.info(f"      - Total goals: {stats.get('total_goals', 0)}")
                            logger.info(f"      - Total cards: {stats.get('total_cards', 0)}")
                        
                        logger.info(f"Test {i} completed successfully")
                    # Check if it's a multiple statistics query
                    elif 'query_type' in db_result and db_result['query_type'] == 'multiple_statistics':
                        player_name = db_result.get('player_name', 'Unknown')
                        statistics = db_result.get('statistics', {})
                        total_matches = db_result.get('total_matches', 0)
                        
                        logger.info(f"   Multiple Statistics for {player_name}:")
                        for stat_name, stat_data in statistics.items():
                            value = stat_data.get('value', 0)
                            logger.info(f"      - {stat_name.replace('_', ' ').title()}: {value}")
                        logger.info(f"   Total matches: {total_matches}")
                        logger.info(f"Test {i} completed successfully")
                    # Check if it's a performance query (contains 'performance' key)
                    elif 'performance' in db_result:
                        performance = db_result['performance']
                        logger.info(f"   Performance stats: {performance}")
                        logger.info(f"Test {i} completed successfully")
                    # Check if it's a regular query with 'value' key
                    elif 'value' in db_result:
                        value = db_result['value']
                        stat = db_result.get('stat', '')
                        logger.info(f"   Database result: {value} {stat}")
                        logger.info(f"Test {i} completed successfully")
                    # Check if it has a nested 'result' structure (old format)
                    elif 'result' in db_result:
                        stat_result = db_result['result']
                        if 'value' in stat_result:
                            logger.info(f"   Database result: {stat_result['value']} {db_result.get('stat', '')}")
                        elif 'performance' in stat_result:
                            performance = stat_result['performance']
                            logger.info(f"   Performance stats: {performance}")
                        else:
                            logger.info(f"   Database status: {stat_result.get('status', 'unknown')}")
                        logger.info(f"Test {i} completed successfully")
                    else:
                        logger.info(f"   Database status: {db_result.get('status', 'unknown')}")
                        logger.info(f"Test {i} completed FAILED - No data output")

                
            except Exception as e:
                logger.error(f"Test {i} failed: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        logger.info("=" * 80)
        logger.info("SYNC TESTS COMPLETED - NOW RUNNING ASYNC PERFORMANCE TEST")
        logger.info("=" * 80)
        
        # Run async performance test
        asyncio.run(test_async_performance())
        
        logger.info("=" * 80)
        logger.info("ALL TESTS COMPLETED (SYNC + ASYNC)")
        logger.info("Check 'soccer_intelligence.log' for detailed logs")
        logger.info("Performance improvements should be visible in concurrent execution")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error("=" * 80)
        logger.error(f"MAIN DEMO FAILED: {e}")
        logger.error("=" * 80)
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()
