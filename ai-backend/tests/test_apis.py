import http.client
import os

from dotenv import load_dotenv

load_dotenv()
# Get API key from environment variable
api_key = os.getenv("RAPIDAPI_KEY")
if not api_key:
    raise ValueError("RAPIDAPI_KEY environment variable is not set")

conn = http.client.HTTPSConnection("api-football-v1.p.rapidapi.com")

headers = {
    "x-rapidapi-host": "api-football-v1.p.rapidapi.com",
    "x-rapidapi-key": api_key,
}

conn.request("GET", "/v3/teams?id=33", headers=headers)

res = conn.getresponse()
data = res.read()

print(data.decode("utf-8"))
