"""
Redis Configuration Management for Query Cache

Provides utilities for configuring Redis for optimal cache performance and persistence.
"""

import logging
import subprocess
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class RedisConfigManager:
    """Manages Redis configuration for optimal cache performance."""

    @staticmethod
    def get_recommended_config() -> Dict[str, Any]:
        """
        Get recommended Redis configuration for cache reliability.

        Returns:
            Dictionary of Redis config parameters
        """
        return {
            # Persistence settings
            "save": [
                "60 1000",
                "300 100",
                "900 1",
            ],  # Save if 1000+ keys changed in 60s, etc.
            "stop-writes-on-bgsave-error": "yes",
            "rdbcompression": "yes",
            "rdbchecksum": "yes",
            # Memory management
            "maxmemory-policy": "allkeys-lru",  # Evict least recently used keys when memory full
            "maxmemory": "256mb",  # Adjust based on your needs
            # Performance
            "tcp-keepalive": "300",
            "timeout": "0",
            # Logging
            "loglevel": "notice",
            # Network
            "bind": "127.0.0.1",
            "protected-mode": "yes",
            "port": "6379",
        }

    @staticmethod
    def check_redis_config() -> Dict[str, str]:
        """
        Check current Redis configuration.

        Returns:
            Dictionary of current Redis config values
        """
        config = {}

        try:
            # Get Redis configuration
            result = subprocess.run(
                ["redis-cli", "CONFIG", "GET", "*"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                for i in range(0, len(lines), 2):
                    if i + 1 < len(lines):
                        config[lines[i]] = lines[i + 1]

        except Exception as e:
            logger.error(f"Failed to get Redis config: {e}")

        return config

    @staticmethod
    def apply_cache_optimizations() -> bool:
        """
        Apply recommended Redis configurations for cache optimization.

        Returns:
            True if configurations were applied successfully
        """
        recommended = RedisConfigManager.get_recommended_config()
        success = True

        try:
            for key, value in recommended.items():
                if isinstance(value, list):
                    # Handle multi-value configs like save
                    for v in value:
                        cmd = ["redis-cli", "CONFIG", "SET", key, v]
                        result = subprocess.run(cmd, capture_output=True, text=True)
                        if result.returncode != 0:
                            logger.warning(f"Failed to set {key} {v}: {result.stderr}")
                            success = False
                else:
                    cmd = ["redis-cli", "CONFIG", "SET", key, str(value)]
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode != 0:
                        logger.warning(f"Failed to set {key}: {result.stderr}")
                        success = False

            if success:
                logger.info("✅ Redis cache optimizations applied successfully")
            else:
                logger.warning("⚠️ Some Redis optimizations failed to apply")

        except Exception as e:
            logger.error(f"Failed to apply Redis optimizations: {e}")
            success = False

        return success

    @staticmethod
    def generate_redis_conf() -> str:
        """
        Generate a redis.conf file with recommended settings.

        Returns:
            String content of redis.conf file
        """
        config = RedisConfigManager.get_recommended_config()

        conf_content = """# Redis Configuration for Sports Intelligence Layer Cache
# Generated automatically - modify with care

# Persistence Configuration
"""

        for key, value in config.items():
            if isinstance(value, list):
                for v in value:
                    conf_content += f"{key} {v}\n"
            else:
                conf_content += f"{key} {value}\n"

        conf_content += """
# Additional cache-optimized settings
lazyfree-lazy-eviction yes
lazyfree-lazy-expire yes
lazyfree-lazy-server-del yes

# Append only file (AOF) for durability
appendonly yes
appendfsync everysec
auto-aof-rewrite-percentage 100
auto-aof-rewrite-min-size 64mb

# Client output buffer limits
client-output-buffer-limit normal 0 0 0
client-output-buffer-limit replica 256mb 64mb 60
client-output-buffer-limit pubsub 32mb 8mb 60
"""

        return conf_content

    @staticmethod
    def save_redis_conf(file_path: Optional[str] = None) -> str:
        """
        Save recommended Redis configuration to file.

        Args:
            file_path: Path to save config file (default: ./redis.conf)

        Returns:
            Path to saved configuration file
        """
        if file_path is None:
            file_path = "redis.conf"

        conf_content = RedisConfigManager.generate_redis_conf()

        try:
            Path(file_path).write_text(conf_content)
            logger.info(f"✅ Redis configuration saved to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save Redis config: {e}")
            raise

        return file_path


def setup_redis_for_cache() -> bool:
    """
    Setup Redis with optimal configuration for caching.

    Returns:
        True if setup was successful
    """
    logger.info("🔧 Setting up Redis for optimal cache performance...")

    try:
        # Check if Redis is running
        result = subprocess.run(["redis-cli", "ping"], capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("❌ Redis is not running. Please start Redis first.")
            return False

        # Apply optimizations
        config_manager = RedisConfigManager()
        success = config_manager.apply_cache_optimizations()

        if success:
            # Save configuration to file for persistence
            config_manager.save_redis_conf("cache_redis.conf")
            logger.info("✅ Redis optimized for cache performance")

        return success

    except Exception as e:
        logger.error(f"Failed to setup Redis: {e}")
        return False


if __name__ == "__main__":
    # Run Redis setup when executed directly
    setup_redis_for_cache()
