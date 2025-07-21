import logging
from typing import Dict, Any
from dotenv import load_dotenv

from agents import Agent, Runner

load_dotenv()
logger = logging.getLogger(__name__)

class WriterAgent:
    """
    AI agent that generates complete football articles using collected data and research insights.
    """
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the Writer Agent with configuration."""
        self.config = config or {}
        
        # Initialize the writer agent
        self.agent = Agent(
            instructions="""You are a professional sports journalist specializing in writing engaging football game recaps.
            Your task is to create compelling, well-structured articles that capture the excitement and significance of football matches.
            
            Guidelines:
            - Write in a professional, engaging tone
            - Use only the provided data - do not invent statistics or quotes
            - Follow the exact structure provided in the template
            - Maintain consistency in style and tone
            - Focus on the most important storylines and moments
            - Create articles that are 400-600 words in length
            
            Always return complete, well-formatted articles ready for publication.""",
            name="WriterAgent",
            output_type=str,
            model=self.config.get("model", "gpt-4o"),
        )
        
        logger.info("Writer Agent initialized successfully")

    async def generate_game_recap(self, game_info: Dict[str, Any], research: Dict[str, Any]) -> str:
        """Generate a complete football game recap article."""
        logger.info("Generating game recap article")
        
        try:
            prompt = self._build_prompt(game_info, research)
            result = await Runner.run(self.agent, prompt)
            article = result.final_output_as(str).strip()
            self._validate_article(article)
            return article
            
        except Exception as e:
            logger.error(f"Error generating game recap: {e}")
            raise

    def _build_prompt(self, game_info, research) -> str:
        logger.info(f"Building prompt for game recap")
        logger.info(f"Game Info: {game_info}")
        logger.info(f"Research Insights: {research}")

        # Extract different types of research data
        storylines = research.get("game_analysis", [])  # Current match events only
        historical_context = research.get("historical_context", [])  # Background information only
        player_performance = research.get("player_performance", [])  # Current match player events only

        prompt = f"""
            Write a professional football game recap article (400-600 words) with the following structure:
            - Headline
            - Introduction (context, teams, stakes)
            - Body (game storyline, key moments, player performances, relevant statistics, quotes)
            - Conclusion (summary, implications)
            Include [Headline, Introduction, Body, Conclusion] in the article to make it easier for the junior writer to understand the structure.

            Template for game recap:
            {self.get_game_recap_template()}

            CRITICAL: You must clearly distinguish between CURRENT MATCH DATA and HISTORICAL/BACKGROUND DATA.

            CURRENT MATCH DATA (Primary Focus - This is what actually happened in this specific game):
            - Game Info: {game_info}
            - Storylines (Current Match Events): {storylines}
            - Player Performance (Current Match Events): {player_performance}
            - This contains the actual events, scores, players, and moments from THIS SPECIFIC MATCH
            - Use this as your main source for describing what happened in the game
            - Focus on: goals, cards, substitutions, key moments, final score, venue, date

            SUBSTITUTION DATA STRUCTURE:
            - Substitution events have: "player" (who went OFF), "assist" (who came ON), "time", "detail"
            - If "assist" is null/missing, the substitution data is incomplete
            - Lineup data shows: "startXI" (starters), "substitutes" (bench players)
            - Only mention substitutions when both "player" and "assist" fields are present

            HISTORICAL/BACKGROUND DATA (Context Only - Use sparingly for introduction/context):
            - Historical Context: {historical_context}
            - This contains background information, historical context, and analysis
            - Use this ONLY for:
              * Brief introduction context (team history, league position, etc.)
              * Background information that helps explain the significance
              * Historical rivalry or previous meetings (if relevant)
            - DO NOT confuse this with current match events
            - DO NOT use historical statistics as if they happened in this game

            CRITICAL MATCHING RULES:
            - When mentioning players, teams, or events, use EXACTLY the names and details from the provided data
            - Do not mix up player names, team names, or event times
            - If a player name is unclear or incomplete in the data, do not guess or complete it
            - Verify that each player mentioned actually participated in the specific event described
            - Only mention players who have clear, verifiable actions in the match events
            - Double-check all player names, team names, and event details against the provided data

            CRITICAL SUBSTITUTION RULES:
            - ONLY mention substitutions when you have COMPLETE information about who went OFF and who came ON
            - In substitution events: "player" field = who went OFF, "assist" field = who came ON
            - If "assist" field is null or missing, DO NOT mention the substitution at all
            - DO NOT guess or assume who came on as a substitute
            - DO NOT mention partial substitution information (e.g., "Player X was substituted off" without knowing who replaced them)
            - Cross-reference with lineup data: "startXI" = starters, "substitutes" = bench players
            - Only describe substitutions that are strategically important and have complete information
            - When in doubt about substitution details, exclude rather than include

            Instructions:
            - Write a complete article following the template structure exactly
            - PRIORITIZE CURRENT MATCH DATA - focus on what actually happened in this specific game
            - Use historical/background data ONLY for context and introduction, not as main story
            - When describing events, clearly indicate they happened in THIS match
            - Do not mix up historical statistics with current match statistics
            - Use only the provided data - do not invent statistics or quotes
            - Use data efficiently and do not miss critical information from the current match data like goals, score, etc.
            - Maintain a consistent, professional tone, and do not make professional mistakes like using wrong team names, wrong scores, etc.
            - Ensure the article is between 400-600 words
            - Include all required sections: Headline, Introduction, Body, Conclusion
            - The main story should be about THIS GAME, not historical background
            - Be extremely careful with player names, team names, and event details - use only what is explicitly stated in the data
            - CRITICAL: For substitutions, only mention them when you have complete information (both who went off AND who came on)
            - CRITICAL: If substitution data is incomplete (missing "assist" field), do not mention the substitution at all
            """
        return prompt
    
    def get_game_recap_template(self):
        return """
        Template: Match Report Structure (400-600 words)
        
        Headline: [Team A] [Score] [Team B]: [Key moment/player] [action verb] [competition context]
        - Concise, engaging headline that captures the main story
        - Include teams, background, score, and key narrative element
        
        Introduction: Context, teams, and stakes
        - Establish result significance with score and competition context
        - Example: "[Winning team] secured a [score] victory over [losing team] in [competition], with [key factor] proving decisive."
        - Introduce background of the game and teams
        - Set up the stakes and importance of the match
        
        Body: Game storyline, key moments, player performances, relevant statistics, quotes
        - Describe key moments in temporal sequence, emphasizing turning points and goals
        - Focus on game-changing incidents rather than comprehensive play-by-play
        - Include individual standout performances and tactical decisions
        - Integrate relevant statistics (possession, shots, etc.) and player quotes
        - Maintain narrative flow while covering all essential game elements
        
        Conclusion: Summary and implications
        - Summarize the key outcome and its significance
        - Address competitive implications (league standings, qualification scenarios, season trajectory)
        - Provide forward-looking perspective on what this result means for both teams
        """

    def _validate_article(self, article: str):
        word_count = len(article.split())
        if word_count < 400 or word_count > 600:
            raise ValueError(f"Article length out of bounds: {word_count} words.")
        if not ("Headline" in article or article.split('\n')[0].strip()):
            raise ValueError("Article missing headline.")
        if not any(section in article for section in ["Introduction", "Body", "Conclusion"]):
            raise ValueError("Article missing required sections.")
        