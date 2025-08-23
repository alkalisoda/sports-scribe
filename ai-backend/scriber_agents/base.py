# agents/base_agent.py
import http.client
import json
import os
import urllib.parse

from agents import Agent, Runner, function_tool
from base_agent import BaseAgent
from dotenv import load_dotenv

load_dotenv()


@function_tool
def get_fixtures(league: str, date: str) -> dict:
    conn = http.client.HTTPSConnection("api-football-v1.p.rapidapi.com")
    headers = {
        "x-rapidapi-host": "api-football-v1.p.rapidapi.com",
        "x-rapidapi-key": os.getenv("RAPIDAPI_KEY"),
    }
    year = date.split("-")[0]
    params = {"league": league, "date": date, "season": year}
    query_string = "?" + urllib.parse.urlencode(params)
    conn.request("GET", f"/v3/fixtures{query_string}", headers=headers)
    response = conn.getresponse()
    if response.status != 200:
        return {
            "error": f"API request failed with status {response.status}: {response.reason}"
        }
    data = response.read()
    print(data)
    try:
        result = json.loads(data.decode("utf-8"))
        return {"raw_api_result": result, "summary": "API result fetched successfully"}
    except json.JSONDecodeError:
        return {
            "error": "Failed to parse JSON response",
            "raw_response": data.decode("utf-8"),
        }


class DataCollectorAgent(BaseAgent):
    def __init__(self):
        self.api_key = os.getenv("RAPIDAPI_KEY")
        if not self.api_key:
            raise ValueError("RAPIDAPI_KEY environment variable is not set")

    def initialize(self, config):
        pass

    async def execute(self, task):
        prompt = task.get("prompt") or "You are a football data agent."
        model = os.getenv("OPENAI_MODEL", "gpt-4o")
        user_prompt = (
            task.get("user_prompt")
            or "Please query all Premier League (league ID: 39) matches for 2010-08-14"
        )

        agent = Agent(
            name="DataCollectorAgent",
            instructions=prompt,
            tools=[get_fixtures],
            model=model,
        )
        result = await Runner.run(agent, user_prompt)
        return result

    def finalize(self):
        pass

    def get_fixtures(self, league: str, date: str) -> dict:
        conn = http.client.HTTPSConnection("api-football-v1.p.rapidapi.com")
        headers = {
            "x-rapidapi-host": "api-football-v1.p.rapidapi.com",
            "x-rapidapi-key": self.api_key,
        }
        year = date.split("-")[0]
        params = {"league": league, "date": date, "season": year}
        query_string = "?" + urllib.parse.urlencode(params)
        conn.request("GET", f"/v3/fixtures{query_string}", headers=headers)
        response = conn.getresponse()
        if response.status != 200:
            return {
                "error": f"API request failed with status {response.status}: {response.reason}"
            }
        data = response.read()
        try:
            return json.loads(data.decode("utf-8"))
        except json.JSONDecodeError:
            return {
                "error": "Failed to parse JSON response",
                "raw_response": data.decode("utf-8"),
            }

    @staticmethod
    def function_schema():
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_fixtures",
                    "description": "Get football match information for specified league and date",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "league": {
                                "type": "string",
                                "description": "League ID (e.g., 39 for Premier League, 140 for La Liga)",
                            },
                            "date": {
                                "type": "string",
                                "description": "Match date in YYYY-MM-DD format",
                            },
                        },
                        "required": ["league", "date"],
                    },
                },
            }
        ]
