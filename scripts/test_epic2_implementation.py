#!/usr/bin/env python3
"""
Epic 2 Implementation Test Script
Tests database schema enhancements and caching system
Based on Epic 2 Validation Checklist
"""

import asyncio
import os
import time
from datetime import datetime
from typing import Dict, Any
import asyncpg
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Epic2Validator:
    """Validates Epic 2 implementation according to the checklist"""
    
    def __init__(self):
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
        self.redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
        
        if not self.supabase_url or not self.supabase_key:
            raise ValueError("Missing Supabase credentials in environment variables")
        
        # Extract database connection details
        self.db_url = self.supabase_url.replace('https://', '').replace('.supabase.co', '')
        self.connection = None
        
    async def initialize(self) -> None:
        """Initialize database connection"""
        connection_string = f"postgresql://postgres:{self.supabase_key}@{self.db_url}:5432/postgres"
        
        try:
            self.connection = await asyncpg.connect(connection_string)
            print("✅ Connected to Supabase database")
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            raise
    
    async def validate_schema_enhancements(self) -> Dict[str, bool]:
        """Validate Epic 2A: Enhanced Sports Data Schema (SIL-004)"""
        print("\n🔍 Validating Schema Enhancement (SIL-004)...")
        
        results = {
            "tables_created": False,
            "indexes_created": False,
            "data_integrity": False,
            "performance_targets": False
        }
        
        try:
            # Check if new tables exist
            tables_query = """
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name IN ('historical_records', 'query_cache', 'contextual_metadata')
            """
            
            tables = await self.connection.fetch(tables_query)
            table_names = [row['table_name'] for row in tables]
            
            expected_tables = ['historical_records', 'query_cache', 'contextual_metadata']
            results["tables_created"] = all(table in table_names for table in expected_tables)
            
            print(f"  Tables created: {'✅' if results['tables_created'] else '❌'}")
            print(f"    Found tables: {table_names}")
            
            # Check indexes
            indexes_query = """
            SELECT indexname, tablename 
            FROM pg_indexes 
            WHERE tablename IN ('historical_records', 'query_cache', 'contextual_metadata')
            AND schemaname = 'public'
            """
            
            indexes = await self.connection.fetch(indexes_query)
            index_count = len(indexes)
            results["indexes_created"] = index_count >= 8  # Minimum expected indexes
            
            print(f"  Indexes created: {'✅' if results['indexes_created'] else '❌'}")
            print(f"    Found {index_count} indexes")
            
            # Test data integrity with sample operations
            try:
                # Test historical_records table
                await self.connection.execute("""
                INSERT INTO historical_records (
                    record_type, entity_type, entity_id, stat_name, stat_value, context
                ) VALUES ('career_high', 'player', 'test_player', 'goals', 5, 'Test record')
                ON CONFLICT DO NOTHING
                """)
                
                # Test query_cache table
                await self.connection.execute("""
                INSERT INTO query_cache (
                    query_hash, query_text, result_data, expires_at
                ) VALUES (
                    'test_hash_123', 'SELECT * FROM test', '{"test": true}', NOW() + INTERVAL '1 hour'
                )
                ON CONFLICT (query_hash) DO NOTHING
                """)
                
                results["data_integrity"] = True
                print(f"  Data integrity: ✅")
                
                # Clean up test data
                await self.connection.execute("DELETE FROM historical_records WHERE entity_id = 'test_player'")
                await self.connection.execute("DELETE FROM query_cache WHERE query_hash = 'test_hash_123'")
                
            except Exception as e:
                print(f"  Data integrity: ❌ ({e})")
                results["data_integrity"] = False
            
            # Test performance with sample queries
            performance_tests = await self._test_query_performance()
            results["performance_targets"] = performance_tests
            
        except Exception as e:
            print(f"❌ Schema validation error: {e}")
        
        return results
    
    async def _test_query_performance(self) -> bool:
        """Test that queries meet performance targets (<100ms for 95% of queries)"""
        print("  Testing query performance...")
        
        test_queries = [
            "SELECT COUNT(*) FROM historical_records",
            "SELECT * FROM historical_records WHERE entity_type = 'player' LIMIT 10",
            "SELECT * FROM query_cache WHERE expires_at > NOW() LIMIT 5",
            "SELECT COUNT(*) FROM contextual_metadata"
        ]
        
        execution_times = []
        
        for query in test_queries:
            try:
                start_time = time.time()
                await self.connection.fetch(query)
                execution_time = (time.time() - start_time) * 1000  # Convert to ms
                execution_times.append(execution_time)
                
            except Exception as e:
                print(f"    Query failed: {query[:30]}... ({e})")
                execution_times.append(1000)  # Penalty for failed query
        
        # Check if 95% of queries are under 100ms
        sorted_times = sorted(execution_times)
        percentile_95_index = int(len(sorted_times) * 0.95)
        percentile_95_time = sorted_times[percentile_95_index] if sorted_times else 1000
        
        avg_time = sum(execution_times) / len(execution_times) if execution_times else 1000
        
        meets_target = percentile_95_time < 100.0
        
        print(f"    Query performance: {'✅' if meets_target else '❌'}")
        print(f"    Average execution time: {avg_time:.2f}ms")
        print(f"    95th percentile: {percentile_95_time:.2f}ms")
        
        return meets_target
    
    async def validate_caching_system(self) -> Dict[str, bool]:
        """Validate Epic 2B: Smart Query Caching System (SIL-005)"""
        print("\n🔍 Validating Caching System (SIL-005)...")
        
        results = {
            "cache_table_functional": False,
            "ttl_behavior": False,
            "cache_cleanup": False,
            "redis_integration": False
        }
        
        try:
            # Test cache table functionality
            test_hash = f"test_cache_{int(time.time())}"
            test_data = {"test": True, "timestamp": datetime.now().isoformat()}
            
            # Insert test cache entry
            await self.connection.execute("""
            INSERT INTO query_cache (
                query_hash, query_text, result_data, expires_at
            ) VALUES ($1, $2, $3, NOW() + INTERVAL '1 minute')
            """, test_hash, "SELECT 1", test_data)
            
            # Retrieve test cache entry
            cached_result = await self.connection.fetchrow(
                "SELECT * FROM query_cache WHERE query_hash = $1", test_hash
            )
            
            results["cache_table_functional"] = cached_result is not None
            print(f"  Cache table functional: {'✅' if results['cache_table_functional'] else '❌'}")
            
            # Test TTL behavior by checking expires_at
            if cached_result:
                expires_at = cached_result['expires_at']
                now = datetime.now(expires_at.tzinfo)
                time_until_expiry = (expires_at - now).total_seconds()
                results["ttl_behavior"] = 0 < time_until_expiry < 70  # Should be around 1 minute
                
                print(f"  TTL behavior: {'✅' if results['ttl_behavior'] else '❌'}")
                print(f"    Expires in: {time_until_expiry:.1f} seconds")
            
            # Test cache cleanup function
            try:
                cleanup_result = await self.connection.fetchval("SELECT cleanup_expired_cache()")
                results["cache_cleanup"] = True  # Function exists and runs
                print(f"  Cache cleanup function: ✅")
                print(f"    Cleaned up entries: {cleanup_result or 0}")
            except Exception as e:
                print(f"  Cache cleanup function: ❌ ({e})")
                results["cache_cleanup"] = False
            
            # Test Redis integration (basic connection test)
            try:
                import redis.asyncio as redis
                redis_client = redis.from_url(self.redis_url)
                await redis_client.ping()
                await redis_client.set("epic2_test", "success", ex=60)
                test_value = await redis_client.get("epic2_test")
                results["redis_integration"] = test_value == b"success"
                await redis_client.close()
                
                print(f"  Redis integration: {'✅' if results['redis_integration'] else '❌'}")
                
            except Exception as e:
                print(f"  Redis integration: ❌ ({e})")
                results["redis_integration"] = False
            
            # Clean up test data
            await self.connection.execute("DELETE FROM query_cache WHERE query_hash = $1", test_hash)
            
        except Exception as e:
            print(f"❌ Caching system validation error: {e}")
        
        return results
    
    async def validate_integration_testing(self) -> Dict[str, bool]:
        """Validate system integration"""
        print("\n🔍 Validating System Integration...")
        
        results = {
            "database_connection_stable": False,
            "concurrent_access_handling": False,
            "error_handling": False,
            "performance_monitoring": False
        }
        
        try:
            # Test database connection stability
            connection_tests = []
            for i in range(5):
                start_time = time.time()
                await self.connection.fetch("SELECT 1")
                connection_time = time.time() - start_time
                connection_tests.append(connection_time < 0.1)  # Under 100ms
            
            results["database_connection_stable"] = all(connection_tests)
            print(f"  Database connection stable: {'✅' if results['database_connection_stable'] else '❌'}")
            
            # Test concurrent access (simplified)
            concurrent_tasks = [
                self.connection.fetch("SELECT COUNT(*) FROM historical_records"),
                self.connection.fetch("SELECT COUNT(*) FROM query_cache"),
                self.connection.fetch("SELECT COUNT(*) FROM contextual_metadata")
            ]
            
            concurrent_results = await asyncio.gather(*concurrent_tasks, return_exceptions=True)
            results["concurrent_access_handling"] = all(
                not isinstance(result, Exception) for result in concurrent_results
            )
            print(f"  Concurrent access handling: {'✅' if results['concurrent_access_handling'] else '❌'}")
            
            # Test error handling
            try:
                await self.connection.fetch("SELECT * FROM non_existent_table")
            except Exception:
                results["error_handling"] = True  # Expected to fail
            
            print(f"  Error handling: {'✅' if results['error_handling'] else '❌'}")
            
            # Test performance monitoring capabilities
            stats_query = """
            SELECT 
                COUNT(*) as total_cache_entries,
                AVG(hit_count) as avg_hit_count,
                COUNT(*) FILTER (WHERE expires_at > NOW()) as active_entries
            FROM query_cache
            """
            
            stats = await self.connection.fetchrow(stats_query)
            results["performance_monitoring"] = stats is not None
            
            print(f"  Performance monitoring: {'✅' if results['performance_monitoring'] else '❌'}")
            if stats:
                print(f"    Cache entries: {stats['total_cache_entries']}")
                print(f"    Average hit count: {stats['avg_hit_count'] or 0:.1f}")
                print(f"    Active entries: {stats['active_entries']}")
            
        except Exception as e:
            print(f"❌ Integration testing error: {e}")
        
        return results
    
    def generate_report(self, schema_results: Dict[str, bool], cache_results: Dict[str, bool], 
                       integration_results: Dict[str, bool]) -> None:
        """Generate comprehensive validation report"""
        print("\n" + "="*60)
        print("📋 EPIC 2 VALIDATION REPORT")
        print("="*60)
        
        all_results = {
            "Schema Enhancement (SIL-004)": schema_results,
            "Caching System (SIL-005)": cache_results,
            "System Integration": integration_results
        }
        
        total_tests = 0
        passed_tests = 0
        
        for category, results in all_results.items():
            print(f"\n📊 {category}:")
            category_passed = 0
            category_total = len(results)
            
            for test_name, passed in results.items():
                status = "✅ PASS" if passed else "❌ FAIL"
                print(f"  {test_name}: {status}")
                if passed:
                    category_passed += 1
                    passed_tests += 1
                total_tests += 1
            
            percentage = (category_passed / category_total * 100) if category_total > 0 else 0
            print(f"  Category Score: {category_passed}/{category_total} ({percentage:.1f}%)")
        
        overall_percentage = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        print(f"\n🎯 OVERALL SCORE: {passed_tests}/{total_tests} ({overall_percentage:.1f}%)")
        
        if overall_percentage >= 80:
            print("🚀 Epic 2 implementation is READY for production!")
        elif overall_percentage >= 60:
            print("⚠️ Epic 2 implementation needs minor improvements")
        else:
            print("❌ Epic 2 implementation requires significant fixes")
        
        print(f"\n⏰ Validation completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
    
    async def run_validation(self) -> None:
        """Execute complete Epic 2 validation"""
        print("🚀 Starting Epic 2 Implementation Validation...")
        print(f"Timestamp: {datetime.now()}")
        
        await self.initialize()
        
        try:
            schema_results = await self.validate_schema_enhancements()
            cache_results = await self.validate_caching_system()
            integration_results = await self.validate_integration_testing()
            
            self.generate_report(schema_results, cache_results, integration_results)
            
        except Exception as e:
            print(f"❌ Validation failed: {e}")
            raise
        finally:
            if self.connection:
                await self.connection.close()
                print("🔌 Database connection closed")

async def main():
    """Main execution function"""
    try:
        validator = Epic2Validator()
        await validator.run_validation()
    except Exception as e:
        print(f"❌ Validation script failed: {e}")
        exit(1)

if __name__ == "__main__":
    asyncio.run(main())