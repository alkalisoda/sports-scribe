import os
import openai
from typing import Dict, Any

class WriterAgent:
    """
    AI agent that generates complete football articles using collected data and research insights.
    """
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided or set as OPENAI_API_KEY environment variable.")
        self.client = openai.OpenAI(api_key=self.api_key)

    def generate_article(self, game_info: Dict[str, Any], team_info: Dict[str, Any], player_info: Dict[str, Any], research: Dict[str, Any]) -> str:
        prompt = self._build_prompt(game_info, team_info, player_info, research)
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": "You are a professional sports journalist."},
                      {"role": "user", "content": prompt}],
            max_tokens=900,
            temperature=0.8
        )
        article = response.choices[0].message.content.strip()
        self._validate_article(article)
        return article

    def _build_prompt(self, game_info, team_info, player_info, research) -> str:
        prompt = f"""
Write a professional football game recap article (400-600 words) with the following structure:
- Headline
- Introduction (context, teams, stakes)
- Body (game storyline, key moments, player performances, relevant statistics, quotes)
- Conclusion (summary, implications)

Game Info: {game_info}
Team Info: {team_info}
Player Info: {player_info}
Research Insights: {research}

Do not invent statistics or quotes. Use only the provided data. Maintain a consistent, professional tone.
"""
        return prompt

    def _validate_article(self, article: str):
        word_count = len(article.split())
        if word_count < 400 or word_count > 600:
            raise ValueError(f"Article length out of bounds: {word_count} words.")
        if not ("Headline" in article or article.split('\n')[0].strip()):
            raise ValueError("Article missing headline.")
        if not any(section in article for section in ["Introduction", "Body", "Conclusion"]):
            raise ValueError("Article missing required sections.")
        # Add more checks as needed