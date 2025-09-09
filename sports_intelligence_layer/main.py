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

# Configure minimal logging - only show important results
logging.basicConfig(
    level=logging.WARNING,  # Reduced log level
    format='%(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('soccer_intelligence.log', mode='w')  # Only log to file
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
        # Load environment variables
        load_dotenv()
        
        # Get Supabase credentials
        self.supabase_url = supabase_url or os.getenv('SUPABASE_URL')
        self.supabase_key = supabase_key or os.getenv('SUPABASE_SERVICE_ROLE_KEY')
        
        if not self.supabase_url or not self.supabase_key:
            raise ValueError(
                "Supabase credentials not found. Please set SUPABASE_URL and "
                "SUPABASE_SERVICE_ROLE_KEY environment variables or pass them directly."
            )
        
        # Initialize components
        self.parser = SoccerQueryParser()
        self.database = SoccerDatabase(self.supabase_url, self.supabase_key)
    
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
        start_time = time.time()
        
        try:
            # Step 1: Parse the query
            parsed_query = self.parser.parse_query(query)
            

            # Step 2: Execute the query against the database (async)
            result = await self.database.run_from_parsed_async(parsed_query)
            
            # Step 3: Format the response
            response = self._format_response(query, parsed_query, result)
            
            end_time = time.time()
            processing_time = (end_time - start_time) * 1000
            response['metadata']['processing_time_ms'] = processing_time
            
            return response
            
        except Exception as e:
            end_time = time.time()
            processing_time = (end_time - start_time) * 1000
            
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
        # Format entities
        formatted_entities = []
        for entity in parsed_query.entities:
            formatted_entities.append({
                "name": entity.name,
                "type": entity.entity_type.value,
                "confidence": entity.confidence
            })
        
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
        
        # Create metadata
        metadata = {
            "timestamp": self._get_timestamp(),
            "processing_time_ms": 0,  # Will be updated by caller
            "data_source": "supabase"
        }
        
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
        
        return response
    
    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.utcnow().isoformat()
    
    async def process_multiple_queries_async(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Process multiple queries concurrently for improved performance."""
        start_time = time.time()
        
        # Create tasks for concurrent execution
        tasks = [self.process_query_async(query) for query in queries]
        
        # Execute all queries concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "status": "error",
                    "message": str(result),
                    "query": queries[i],
                    "timestamp": self._get_timestamp(),
                    "processing_time_ms": 0
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics from the database layer."""
        return self.database.get_performance_stats()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache-specific statistics."""
        return self.database.get_cache_stats()
    
    def reset_performance_stats(self):
        """Reset performance statistics."""
        self.database.reset_performance_stats()
    
    def cleanup_cache(self) -> int:
        """Clean up expired cache entries."""
        return self.database.cleanup_cache()
    
    def clear_cache(self) -> int:
        """Clear all cache entries."""
        return self.database.clear_cache()



# Removed async test function - keeping main simple

def print_query_result(query: str, result: Dict[str, Any], query_num: int = None):
    """Print query result in a clean format."""
    header = f"Query {query_num}: " if query_num else "Query: "
    print(f"\n{header}{query}")
    print("-" * 80)
    
    if result.get('status') != 'success':
        print(f"❌ Error: {result.get('message', 'Unknown error')}")
        return
    
    # Get confidence and processing time
    parsed = result.get('query', {}).get('parsed', {})
    confidence = parsed.get('confidence', 0)
    processing_time = result.get('metadata', {}).get('processing_time_ms', 0)
    
    # Check if result was cached
    db_result = result.get('result', {})
    cached = db_result.get('cached', False)
    
    # Also check for cache_hash which indicates it went through cache system
    has_cache_hash = 'cache_hash' in db_result
    cache_info = "🔥 Cached" if cached else ("🔄 Cache-enabled" if has_cache_hash else "🆕 Fresh")
    
    print(f"✅ Success ({cache_info}, Confidence: {confidence:.2f}, Time: {processing_time:.1f}ms)")
    
    # Extract and display the actual data
    db_result = result.get('result', {})
    
    if 'query_type' in db_result and db_result['query_type'] == 'match_result':
        # Match result
        match_data = db_result['match']
        team1 = match_data['team1']
        team2 = match_data['team2']
        score = match_data['score']
        winner = match_data['winner']
        
        if winner == 'team1':
            winner_name = team1['name']
        elif winner == 'team2':
            winner_name = team2['name']
        else:
            winner_name = "Draw"
        
        print(f"🏆 Match: {team1['name']} {score} {team2['name']}")
        print(f"🥇 Winner: {winner_name}")
        print(f"🆔 Match ID: {match_data['match_id']}")
        
        if 'statistics' in match_data:
            stats = match_data['statistics']
            print(f"📊 Match Stats: Shots({stats.get('total_shots', 0)}), Goals({stats.get('total_goals', 0)}), Cards({stats.get('total_cards', 0)})")
    
    elif 'query_type' in db_result and db_result['query_type'] == 'multiple_statistics':
        # Multiple statistics
        player_name = db_result.get('player_name', 'Unknown')
        statistics = db_result.get('statistics', {})
        total_matches = db_result.get('total_matches', 0)
        
        print(f"👤 Player: {player_name}")
        print(f"🎮 Matches: {total_matches}")
        print("📈 Statistics:")
        for stat_name, stat_data in statistics.items():
            value = stat_data.get('value', 0)
            print(f"   • {stat_name.replace('_', ' ').title()}: {value}")
    
    elif 'query_type' in db_result and db_result['query_type'] == 'team_player_ranking':
        # Team player ranking
        team_name = db_result.get('team_name', 'Team')
        ranking_type = db_result.get('ranking_type', 'most')
        stat = db_result.get('stat', 'goals')
        top_player = db_result.get('top_player', {})
        all_players = db_result.get('all_players', [])
        
        print(f"🏆 {ranking_type.title()} {stat} for {team_name}:")
        print(f"🥇 Top Player: {top_player.get('player_name', 'Unknown')} ({top_player.get('value', 0)} {stat})")
        
        if len(all_players) > 1:
            print("📊 Top Rankings:")
            for i, player in enumerate(all_players[:5], 1):  # Show top 5
                print(f"   {i}. {player.get('player_name', 'Unknown')}: {player.get('value', 0)} {stat}")
    
    elif 'performance' in db_result:
        # Performance query
        performance = db_result['performance']
        print(f"⚽ Performance: {performance}")
    
    elif 'value' in db_result:
        # Regular statistic
        value = db_result['value']
        stat = db_result.get('stat', '')
        entity_name = parsed.get('entities', [{}])[0].get('name', 'Player/Team')
        print(f"📊 {entity_name} {stat}: {value}")
    
    elif 'result' in db_result:
        # Nested result structure
        stat_result = db_result['result']
        if 'value' in stat_result:
            value = stat_result['value']
            stat = db_result.get('stat', '')
            entity_name = parsed.get('entities', [{}])[0].get('name', 'Player/Team')
            print(f"📊 {entity_name} {stat}: {value}")
        elif 'performance' in stat_result:
            performance = stat_result['performance']
            print(f"⚽ Performance: {performance}")
        else:
            print(f"❓ Status: {stat_result.get('status', 'unknown')}")
    
    else:
        # Handle error cases with better messaging
        if 'status' in db_result and db_result.get('status') == 'no_data':
            reason = db_result.get('reason', 'unknown')
            if reason == 'player_not_found':
                print(f"❌ Player not found in database")
                print(f"💡 Hint: Check if the player name is spelled correctly")
            elif reason == 'team_players_not_found':
                print(f"❌ Team not found in database")
                print(f"💡 Hint: Try the full team name (e.g., 'Brighton & Hove Albion' instead of 'Brighton')")
            elif reason == 'no_player_stats_found':
                print(f"❌ No statistics found for the requested players")
                print(f"💡 Hint: Check if the players have data for the current season")
            else:
                print(f"❌ No data found: {reason}")
        elif 'status' in db_result and db_result.get('status') == 'error':
            reason = db_result.get('reason', 'unknown error')
            print(f"❌ Database error: {reason}")
        else:
            print(f"❓ No data found or unrecognized result format")
            print(f"🔍 Raw result keys: {list(db_result.keys())}")
            # Debug: Show the actual result content for troubleshooting
            if 'status' in db_result:
                print(f"🐛 Status: {db_result.get('status')}")
            if 'reason' in db_result:
                print(f"🐛 Reason: {db_result.get('reason')}")
            print(f"🐛 Full result: {db_result}")


def test_cache_functionality(sil: SoccerIntelligenceLayer):
    """Test cache functionality with repeated queries."""
    print("\n🧪 Testing Cache Functionality")
    print("-" * 50)
    
    # Test query that should be cached
    test_query = "How many goals has Kaoru Mitoma scored?"
    
    print(f"Query: {test_query}")
    print("🔄 First execution (should be cache miss)...")
    
    # First execution - should be cache miss
    start_time = time.time()
    result1 = sil.process_query(test_query)
    first_time = (time.time() - start_time) * 1000
    
    cached1 = result1.get('result', {}).get('cached', False)
    print(f"   Result: {'🔥 Cached' if cached1 else '🆕 Fresh'}, Time: {first_time:.1f}ms")
    
    print("🔄 Second execution (should be cache hit)...")
    
    # Second execution - should be cache hit
    start_time = time.time()
    result2 = sil.process_query(test_query)
    second_time = (time.time() - start_time) * 1000
    
    cached2 = result2.get('result', {}).get('cached', False)
    print(f"   Result: {'🔥 Cached' if cached2 else '🆕 Fresh'}, Time: {second_time:.1f}ms")
    
    # Performance comparison
    if cached2 and not cached1:
        improvement = ((first_time - second_time) / first_time) * 100
        print(f"🚀 Cache performance improvement: {improvement:.1f}%")
    
    # Show cache statistics
    try:
        cache_stats = sil.get_cache_stats()
        print(f"\n📊 Cache Statistics:")
        print(f"   Total entries: {cache_stats.get('total_cache_entries', 0)}")
        print(f"   Cache hits: {cache_stats.get('cache_hits', 0)}")
        print(f"   Cache misses: {cache_stats.get('cache_misses', 0)}")
        print(f"   Hit rate: {cache_stats.get('cache_hit_rate', 0):.2%}")
        print(f"   Utilization: {cache_stats.get('cache_utilization_percent', 0):.1f}%")
    except Exception as e:
        print(f"❌ Error getting cache stats: {e}")


def main():
    """
    Main function - clean output showing actual query results.
    """
    print("🚀 SportsScribe Soccer Intelligence Layer")
    print("=" * 80)
    
    try:
        # Initialize the Soccer Intelligence Layer
        print("⚙️  Initializing...")
        sil = SoccerIntelligenceLayer()
        print("✅ Ready!")
        
        # Test queries
        test_queries = [
            "How many goals has Kaoru Mitoma scored?",
            "What's Danny Welbeck's assist record?", 
            "How many goals did Danny Welbeck score?",
            "What are Kaoru Mitoma's stats?",
            "Show me Billy Gilmour's goals, assists, and yellow cards this season",
            "Who scored the most goals for Brighton?",
            "Most assists by Brighton players",
            "Everton players goals",
            "Brighton vs Everton match stats",
            "Abdoulaye Doucouré shots on target",
            "Jordan Pickford performance"
        ]
        
        print(f"\n🔍 Testing {len(test_queries)} queries:\n")
        
        for i, query in enumerate(test_queries, 1):
            try:
                result = sil.process_query(query)
                print_query_result(query, result, i)
            except Exception as e:
                print(f"\nQuery {i}: {query}")
                print("-" * 80)
                print(f"❌ Error: {e}")
        
        print("\n" + "=" * 80)
        print("🎯 All queries completed!")
        
        # Test cache functionality
        test_cache_functionality(sil)
        
        # Show final performance stats
        print("\n📈 Final Performance Statistics:")
        print("-" * 40)
        try:
            perf_stats = sil.get_performance_stats()
            print(f"Total queries: {perf_stats.get('total_queries', 0)}")
            print(f"Average query time: {perf_stats.get('average_query_time', 0):.3f}s")
            print(f"Cache hit rate: {perf_stats.get('cache_hit_rate', 0):.2%}")
            
            cache_stats = sil.get_cache_stats()
            print(f"Cache entries: {cache_stats.get('total_cache_entries', 0)}")
            print(f"Cache utilization: {cache_stats.get('cache_utilization_percent', 0):.1f}%")
        except Exception as e:
            print(f"❌ Error getting performance stats: {e}")
        
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
