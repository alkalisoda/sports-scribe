#!/usr/bin/env python3
"""
Test script for the player pipeline fixes.
"""

import asyncio
import json
import os
import sys
from dotenv import load_dotenv

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scriber_agents.pipeline import ArticlePipeline

load_dotenv()

async def test_player_pipeline():
    """Test the player pipeline with the fixes."""
    
    print("🧪 Testing Player Pipeline Fixes")
    print("=" * 50)
    
    try:
        # Initialize pipeline
        pipeline = ArticlePipeline()
        print("✅ Pipeline initialized successfully")
        
        # Test with a player ID that has no data (like in the error log)
        player_id = "9876543"
        game_id = "1234567"
        
        print(f"\n🎯 Testing player spotlight generation for player {player_id}")
        print("-" * 50)
        
        # Test player spotlight generation
        result = await pipeline.generate_player_spotlight(player_id, game_id)
        
        print("✅ Player spotlight generation completed successfully!")
        print(f"   Content length: {len(result.get('content', ''))}")
        print(f"   Metadata keys: {list(result.get('metadata', {}).keys())}")
        
        # Show a preview of the content
        content = result.get('content', '')
        if content:
            print(f"\n📝 Content Preview:")
            print(f"   {content[:200]}...")
        
        print("\n" + "=" * 50)
        print("🎉 Player Pipeline Test Completed Successfully!")
        
    except Exception as e:
        print(f"❌ Player pipeline test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_player_pipeline()) 