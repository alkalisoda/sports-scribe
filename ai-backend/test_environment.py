"""Test script to verify all dependencies are properly installed."""

import sys

print(f"Python version: {sys.version}")

# Test core dependencies
try:
    import openai
    print(f"✅ OpenAI package imported successfully - Version: {openai.__version__}")
except ImportError as e:
    print(f"❌ OpenAI import failed: {e}")

try:
    from agents import Agent
    print(f"✅ OpenAI Agents package imported successfully - Agent class: {Agent}")
except ImportError as e:
    print(f"❌ OpenAI Agents import failed: {e}")

try:
    import fastapi
    print(f"✅ FastAPI package imported successfully - Version: {fastapi.__version__}")
except ImportError as e:
    print(f"❌ FastAPI import failed: {e}")

try:
    from pydantic import BaseModel
    print(f"✅ Pydantic package imported successfully - BaseModel: {BaseModel}")
except ImportError as e:
    print(f"❌ Pydantic import failed: {e}")

try:
    from supabase import create_client
    print(f"✅ Supabase package imported successfully - create_client: {create_client}")
except ImportError as e:
    print(f"❌ Supabase import failed: {e}")

try:
    import aiohttp
    print(f"✅ Aiohttp package imported successfully - Version: {aiohttp.__version__}")
except ImportError as e:
    print(f"❌ Aiohttp import failed: {e}")

try:
    from dotenv import load_dotenv
    print(f"✅ Python-dotenv package imported successfully - load_dotenv: {load_dotenv}")
except ImportError as e:
    print(f"❌ Python-dotenv import failed: {e}")

try:
    import structlog
    print(f"✅ Structlog package imported successfully - Version: {structlog.__version__}")
except ImportError as e:
    print(f"❌ Structlog import failed: {e}")

print("\n🎉 Environment test completed!")
