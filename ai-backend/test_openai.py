"""Test OpenAI API connection."""

import os

import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up OpenAI client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def test_openai_connection():
    """Test basic OpenAI API connection."""
    if (
        not os.getenv("OPENAI_API_KEY")
        or os.getenv("OPENAI_API_KEY") == "your_openai_api_key_here"
    ):
        print("⚠️  OpenAI API key not set. Skipping connection test.")
        return

    try:
        # Test with a simple completion
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[{"role": "user", "content": "Say 'Hello from Sport Scribe AI!'"}],
            max_tokens=50,
        )

        print("✅ OpenAI API connection successful!")
        print(f"Response: {response.choices[0].message.content}")

    except Exception as e:
        print(f"❌ OpenAI API connection failed: {e}")


if __name__ == "__main__":
    test_openai_connection()
