import asyncio
import logging
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scriber_agents.pipeline import AgentPipeline
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

async def test_game_recap(game_id: str) -> str:
    pipeline = AgentPipeline()

    raw_game_data = await pipeline._collect_game_data(game_id)
    logger.info(f"📝 Raw game data: {raw_game_data}")

    result = await pipeline.generate_game_recap(game_id)

    content = result.get("content", "")
    logger.info(f"📝 Article length: {len(content)} characters")

    result_dir = os.path.join(os.path.dirname(__file__), "..", "result")
    os.makedirs(result_dir, exist_ok=True)
    output_path = os.path.join(result_dir, f"game_recap_{game_id}.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"📝 Raw game data: {raw_game_data}\n")
        f.write('\n' + "=" * 50 + "\n")
        f.write(f"Generated article:\n")
        f.write("=" * 50 + "\n")
        f.write(content)

    return result

if __name__ == "__main__":
    for game_id in ["1208021", "1208023"]:
        result = asyncio.run(test_game_recap(game_id))
        print(result)
    # game_id = "1208023"
    # result = asyncio.run(test_game_recap(game_id))
    # print(result)
