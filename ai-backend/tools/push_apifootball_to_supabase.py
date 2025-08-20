import os
import asyncio
import logging
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv()

# Local import from same folder
from sports_apis import APIFootballClient

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Supabase Setup ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- Data Transformation ---
def transform_fixture_to_competition(fixture: dict) -> dict:
    return {
        "id": fixture["fixture"]["id"],
        "name": fixture["league"]["name"],
        "type": "api-football", # hard-coded api-football
        "country": fixture["league"].get("country"),
        "season": fixture["league"].get("season"),
        "start_date": fixture["fixture"].get("date"),
        "end_date": None,
        "status": fixture["fixture"]["status"].get("long"),
        "venueId": fixture["fixture"]["venue"].get("id"),
        "leagueId": fixture["league"].get("id"),
        "homeTeamId": fixture["teams"]["home"].get("id"),
        "awayTeamId": fixture["teams"]["away"].get("id"),
        "goalsHome": fixture["goals"].get("home"),
        "goalsAway": fixture["goals"].get("away"),
        "goalsHomeHalfTime": fixture["score"]["halftime"].get("home"),
        "goalsAwayHalfTime": fixture["score"]["halftime"].get("away"),
        "goalsHomeExtraTime": fixture["score"]["extratime"].get("home"),
        "goalsAwayExtraTime": fixture["score"]["extratime"].get("away"),
        "penaltyHome": fixture["score"]["penalty"].get("home"),
        "penaltyAway": fixture["score"]["penalty"].get("away"),
    }

# --- Push to Supabase ---
async def push_fixtures_to_supabase(fixtures: list[dict]):
    batch = [transform_fixture_to_competition(f) for f in fixtures]
    if batch:
        response = supabase.table("competitions").insert(batch).execute()
        logger.info("Inserted %d fixtures into Supabase", len(batch))
    else:
        logger.warning("No fixtures to insert.")

# --- Main Execution ---
async def main():
    api_key = os.getenv("API_FOOTBALL_KEY")
    api_key_header = os.getenv("API_FOOTBALL_KEY_HEADER")
    base_url = os.getenv("API_FOOTBALL_BASE_URL")

    client = APIFootballClient(api_key, api_key_header, base_url)

    fixtures = await client.get_fixtures(league_id=39, season=2023)
    await push_fixtures_to_supabase(fixtures)

if __name__ == "__main__":
    asyncio.run(main())
