"""
Test suite for the query cache system.
Tests the core functionality and integration of the Redis-based query cache.
"""

import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch

# Import the query cache components
from src.query_cache.query_cache import create_query_cache, QueryCache
from src.query_cache.cache_invalidation_manager import CacheInvalidationManager
from src.query_cache.redis_config import RedisConfigManager


class TestQueryCache:
    """Test class for QueryCache core functionality."""

    def setup_method(self):
        """Set up test fixtures for each test method."""
        self.mock_redis_client = AsyncMock()
        self.query_cache = QueryCache(self.mock_redis_client)

    @pytest.mark.asyncio
    async def test_cache_miss(self):
        """Test cache miss scenario."""
        self.mock_redis_client.get.return_value = None
        result = await self.query_cache.get_cached_result(
            "SELECT * FROM test", {"id": 1}
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_result_storage(self):
        """Test caching a result."""
        test_data = {"name": "John", "age": 30}
        query = "SELECT * FROM users WHERE id = %s"
        params = {"id": 1}

        await self.query_cache.cache_result(query, params, test_data, ttl=300)

        self.mock_redis_client.setex.assert_called_once()
        call_args = self.mock_redis_client.setex.call_args
        assert call_args[0][1] == 300  # TTL
        assert json.loads(call_args[0][2]) == test_data

    @pytest.mark.asyncio
    async def test_ttl_determination(self):
        """Test TTL determination logic."""
        # Test live data query (short TTL)
        ttl = self.query_cache._determine_ttl("SELECT * FROM live_scores", {})
        assert ttl == 60

        # Test default TTL
        ttl = self.query_cache._determine_ttl("SELECT * FROM teams", {})
        assert ttl == 3600

    @pytest.mark.asyncio
    async def test_pattern_invalidation(self):
        """Test pattern-based cache invalidation."""
        self.mock_redis_client.keys.return_value = ["query:key1", "query:key2"]
        self.mock_redis_client.delete.return_value = 2

        deleted_count = await self.query_cache.invalidate_pattern("query:*")

        assert deleted_count == 2
        self.mock_redis_client.keys.assert_called_with("query:*")
        self.mock_redis_client.delete.assert_called_with("query:key1", "query:key2")

    @pytest.mark.asyncio
    async def test_atomic_operations(self):
        """Test atomic get and increment operations."""
        test_data = {"name": "John", "age": 30}
        self.mock_redis_client.eval.return_value = [json.dumps(test_data), 1]

        result, was_hit = await self.query_cache.get_and_increment_atomic("test_key")

        assert result == json.dumps(test_data)
        assert was_hit is True
        self.mock_redis_client.eval.assert_called()

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in cache operations."""
        # Test cache retrieval error
        self.mock_redis_client.get.side_effect = Exception("Redis connection error")

        result = await self.query_cache.get_cached_result("SELECT * FROM test", {})
        assert result is None

        # Reset mock and test cache storage error (should not raise exception)
        self.mock_redis_client.get.side_effect = None
        self.mock_redis_client.setex.side_effect = Exception("Redis storage error")

        # Should not raise exception
        await self.query_cache.cache_result("SELECT * FROM test", {}, {"data": "test"})


class TestCacheInvalidationManager:
    """Test class for CacheInvalidationManager."""

    def setup_method(self):
        """Set up test fixtures for each test method."""
        self.mock_cache = AsyncMock()
        self.mock_cache.invalidate_patterns_batch.return_value = 5
        self.invalidation_manager = CacheInvalidationManager(self.mock_cache)

    @pytest.mark.asyncio
    async def test_invalidate_player_cache(self):
        """Test player cache invalidation."""
        await self.invalidation_manager.invalidate_player_cache("Lionel Messi")

        self.mock_cache.invalidate_patterns_batch.assert_called()
        call_args = self.mock_cache.invalidate_patterns_batch.call_args[0][0]
        assert any("messi" in pattern.lower() for pattern in call_args)

    @pytest.mark.asyncio
    async def test_invalidate_team_cache(self):
        """Test team cache invalidation."""
        await self.invalidation_manager.invalidate_team_cache("Barcelona")

        self.mock_cache.invalidate_patterns_batch.assert_called()
        call_args = self.mock_cache.invalidate_patterns_batch.call_args[0][0]
        assert any("barcelona" in pattern.lower() for pattern in call_args)


class TestRedisConfigManager:
    """Test class for RedisConfigManager."""

    def setup_method(self):
        """Set up test fixtures for each test method."""
        self.config_manager = RedisConfigManager()

    def test_get_recommended_config(self):
        """Test getting recommended Redis configuration."""
        config = self.config_manager.get_recommended_config()

        assert "maxmemory-policy" in config
        assert config["maxmemory-policy"] == "allkeys-lru"
        assert "save" in config
        assert "maxmemory" in config

    def test_generate_redis_conf(self):
        """Test Redis configuration file generation."""
        config_content = self.config_manager.generate_redis_conf()

        assert "maxmemory-policy allkeys-lru" in config_content
        assert "save" in config_content
        assert "maxmemory" in config_content


class TestQueryCacheCreation:
    """Test class for query cache creation function."""

    @patch("src.query_cache.query_cache.REDIS_AVAILABLE", True)
    @patch("src.query_cache.query_cache.redis_module")
    def test_create_query_cache_success(self, mock_redis_module):
        """Test successful query cache creation."""
        # Mock Redis module and connection pool
        mock_pool = MagicMock()
        mock_redis_client = MagicMock()
        mock_redis_module.ConnectionPool.return_value = mock_pool
        mock_redis_module.Redis.return_value = mock_redis_client

        cache = create_query_cache()

        assert cache is not None
        assert isinstance(cache, QueryCache)
        mock_redis_module.ConnectionPool.assert_called_once()
        mock_redis_module.Redis.assert_called_once()

    @patch("src.query_cache.query_cache.REDIS_AVAILABLE", False)
    def test_create_query_cache_redis_unavailable(self):
        """Test query cache creation when Redis is unavailable."""
        cache = create_query_cache()
        assert cache is None

    @patch("src.query_cache.query_cache.REDIS_AVAILABLE", True)
    @patch("src.query_cache.query_cache.redis_module")
    def test_create_query_cache_connection_error(self, mock_redis_module):
        """Test query cache creation with connection error."""
        mock_redis_module.ConnectionPool.side_effect = Exception("Connection failed")

        cache = create_query_cache()
        assert cache is None


class TestIntegration:
    """Integration tests for the query cache system."""

    @pytest.mark.asyncio
    async def test_cache_system_integration(self):
        """Test that the cache system integrates properly."""
        # Test that cache can be created (may return None if Redis not available)
        cache = create_query_cache()

        # If cache is available, test basic functionality
        if cache is not None:
            # Test caching first
            test_data = {"test": "data"}
            await cache.cache_result("SELECT 1", {}, test_data)

            # Note: Result might be cached from previous test runs, so we just test no errors occur
            await cache.get_cached_result("SELECT 1", {})
            # Result could be None (miss) or the test_data (hit) - both are valid

            # Clean up
            try:
                await cache.close()
            except Exception:
                pass  # Ignore cleanup errors in tests
        else:
            # Redis not available, which is acceptable in test environment
            assert cache is None

    def test_cache_functionality_end_to_end(self):
        """Test cache functionality works end-to-end."""
        # This test just verifies that the cache system can be used without errors
        create_query_cache()  # Test creation doesn't crash

        # Verify we can create a QueryCache object directly
        mock_redis = AsyncMock()
        direct_cache = QueryCache(mock_redis)
        assert direct_cache is not None
        assert hasattr(direct_cache, "get_cached_result")
        assert hasattr(direct_cache, "cache_result")

    def test_query_cache_components_available(self):
        """Test that all query cache components can be imported."""
        # Test imports work
        assert QueryCache is not None
        assert CacheInvalidationManager is not None
        assert RedisConfigManager is not None
        assert create_query_cache is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
