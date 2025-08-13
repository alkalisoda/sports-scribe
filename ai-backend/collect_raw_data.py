#!/usr/bin/env python3
"""
Simple Raw Data Collector

This script uses the existing pipeline to collect raw game data
and saves it as JSON files to a data folder.
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Add the scriber_agents directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'scriber_agents')))

from scriber_agents.pipeline import AgentPipeline
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def collect_raw_game_data(game_ids: list[str]):
    """Collect raw game data using the existing pipeline and save as JSON."""
    
    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Create games subdirectory
    games_dir = data_dir / "games"
    games_dir.mkdir(exist_ok=True)
    
    pipeline = AgentPipeline()
    
    for game_id in game_ids:
        try:
            logger.info(f"Collecting raw data for game ID: {game_id}")
            
            # Get raw game data using the pipeline's internal method
            raw_game_data = await pipeline._collect_game_data(game_id)
            
            if raw_game_data:
                # Create filename with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{timestamp}_game_{game_id}.json"
                file_path = games_dir / filename
                
                # Save raw data as JSON
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(raw_game_data, f, indent=2, ensure_ascii=False, default=str)
                
                logger.info(f"✅ Raw data saved for game {game_id} to: {file_path}")
                
                # Also save a summary of what was collected
                summary = {
                    "game_id": game_id,
                    "collection_timestamp": timestamp,
                    "data_keys": list(raw_game_data.keys()) if isinstance(raw_game_data, dict) else "Not a dict",
                    "response_count": len(raw_game_data.get("response", [])) if isinstance(raw_game_data, dict) else 0,
                    "errors": raw_game_data.get("errors", []) if isinstance(raw_game_data, dict) else [],
                    "results": raw_game_data.get("results", 0) if isinstance(raw_game_data, dict) else 0
                }
                
                summary_filename = f"{timestamp}_game_{game_id}_summary.json"
                summary_path = games_dir / summary_filename
                
                with open(summary_path, 'w', encoding='utf-8') as f:
                    json.dump(summary, f, indent=2, ensure_ascii=False)
                
                logger.info(f"📊 Summary saved for game {game_id} to: {summary_path}")
                
            else:
                logger.warning(f"⚠️  No raw data returned for game {game_id}")
                
        except Exception as e:
            logger.error(f"❌ Error collecting data for game {game_id}: {e}")
    
    logger.info(f"Data collection completed. Check the 'data/games' folder for results.")

async def main():
    """Main function to run the data collection."""
    # Game IDs to collect data for
    game_ids = ["1208021", "1208022", "1208023", "1208024", "1208025"]
    
    logger.info(f"Starting raw data collection for {len(game_ids)} games...")
    await collect_raw_game_data(game_ids)
    logger.info("Raw data collection completed!")

if __name__ == "__main__":
    asyncio.run(main())
