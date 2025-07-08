#!/usr/bin/env python3
"""
Example usage of the SportsScribe Pipeline.

This script demonstrates how to use the streamlined pipeline
to generate different types of sports articles.
"""

import asyncio
import os
from typing import Dict, Any

from scriber_agents.pipeline import ArticlePipeline
from utils.logging_config import setup_logging


async def main():
    """Demonstrate pipeline usage with different article types."""
    
    # Setup logging
    setup_logging(
        level="INFO",
        log_file="logs/pipeline_example.log",
        include_debug=True
    )
    
    # Initialize the pipeline (uses environment variables automatically)
    print("🚀 Initializing SportsScribe Pipeline...")
    try:
        pipeline = ArticlePipeline()
    except ValueError as e:
        print(f"❌ Error: {str(e)}")
        return
    
    # Example game ID (you can replace with actual game IDs)
    example_game_id = "686314"
    example_player_id = "276"
    
    # Create result directory if it doesn't exist
    os.makedirs("result", exist_ok=True)
    
    try:
        # Example 1: Generate Game Recap
        print(f"\n📝 Generating Game Recap for game {example_game_id}...")
        recap_result = await pipeline.generate_game_recap(example_game_id)
        
        print("✅ Game Recap Generated Successfully!")
        print(f"📊 Metadata: {recap_result['metadata']}")
        print(f"📄 Content Preview: {recap_result['content'][:200]}...")
        with open("result/game_recap.txt", "w", encoding="utf-8") as f:
            f.write(recap_result["content"])
        
        # Example 2: Generate Preview Article
        print(f"\n🔮 Generating Preview Article for game {example_game_id}...")
        preview_result = await pipeline.generate_preview_article(example_game_id)
        
        print("✅ Preview Article Generated Successfully!")
        print(f"📊 Metadata: {preview_result['metadata']}")
        print(f"📄 Content Preview: {preview_result['content'][:200]}...")
        with open("result/preview_article.txt", "w", encoding="utf-8") as f:
            f.write(preview_result["content"])
        
        # Example 3: Generate Player Spotlight
        print(f"\n⭐ Generating Player Spotlight for player {example_player_id}...")
        spotlight_result = await pipeline.generate_player_spotlight(
            example_player_id, 
            game_id=example_game_id
        )
        
        print("✅ Player Spotlight Generated Successfully!")
        print(f"📊 Metadata: {spotlight_result['metadata']}")
        print(f"📄 Content Preview: {spotlight_result['content'][:200]}...")
        with open("result/player_spotlight.txt", "w", encoding="utf-8") as f:
            f.write(spotlight_result["content"])
        
        # Get pipeline status
        print(f"\n📈 Pipeline Status:")
        status = await pipeline.get_pipeline_status()
        print(f"   Version: {status['pipeline_version']}")
        print(f"   Agents: {status['agents']}")
        print(f"   Last Updated: {status['last_updated']}")
        
    except Exception as e:
        print(f"❌ Error during pipeline execution: {str(e)}")
        print("💡 Make sure you have valid API keys and network connectivity")


def print_pipeline_info():
    """Print information about the pipeline structure."""
    print("🏈 SportsScribe Pipeline Structure")
    print("=" * 50)
    print("Pipeline Flow: Data Collector → Researcher → Writer")
    print()
    print("📋 Available Article Types:")
    print("   • Game Recap - Post-match analysis and highlights")
    print("   • Preview Article - Pre-match predictions and analysis")
    print("   • Player Spotlight - Individual player performance focus")
    print()
    print("🔧 Key Features:")
    print("   • Shared OpenAI client for efficiency")
    print("   • Helper methods for clean separation of concerns")
    print("   • Standardized API response structure")
    print("   • Storyline integration for better content focus")
    print("   • Centralized error handling")
    print()
    print("📊 Data Flow:")
    print("   1. Data Collector → Raw sports data from API-Football")
    print("   2. Researcher → Context analysis + Storylines generation")
    print("   3. Writer → AI-generated article content")
    print()


if __name__ == "__main__":
    print_pipeline_info()
    
    # Run the async main function
    asyncio.run(main()) 