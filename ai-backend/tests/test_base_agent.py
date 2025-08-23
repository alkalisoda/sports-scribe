# agents/data_collector_agent.py
import asyncio
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dotenv import load_dotenv

from scriber_agents.base import DataCollectorAgent

load_dotenv()

if __name__ == "__main__":
    agent = DataCollectorAgent()
    agent.initialize({})
    task = {
        "user_prompt": "Please query all Premier League (league ID: 39) matches for 2024-08-16",
        "prompt": (
            "You are a football data agent. "
            "When the user asks for match information, always output the full details of all matches you find, "
            "including teams, scores, date, and venue. "
            "Do not summarize or ask the user if they want details—just output the full data directly."
        ),
    }
    result = asyncio.run(agent.execute(task))
    print(result)
