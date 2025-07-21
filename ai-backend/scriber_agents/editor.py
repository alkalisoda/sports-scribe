import logging
from typing import Any, List, Dict
from dotenv import load_dotenv
import json
from agents import Agent, Runner

load_dotenv()
logger = logging.getLogger(__name__)

class Editor:
    def __init__(self, config: dict):
        self.config = config or {}
        
        # Initialize single agent for all editing tasks
        self.agent = Agent(
            instructions=self.get_base_prompt(),
            name="Editor",
            output_type=str,
            model=self.config.get("model", "gpt-4o-mini"),
        )

        logger.info("Editor initialized successfully")
    
    def get_base_prompt(self) -> str:
        return """
        You are a professional sports editor specializing in football/soccer articles.
        You can perform different types of editing tasks based on the specific instructions provided.
        
        Your core capabilities:
        1. Fact-checking: Verify factual accuracy against provided game data
        2. Terminology checking: Correct sports terminology usage
        
        Always maintain the original writing style, tone, and structure.
        Only correct errors - do not change correct information.
        If no errors are found, return the original text unchanged.
        """
    
    def get_fact_checking_prompt(self) -> str:
        return """
        TASK: FACT-CHECKING
        
        You are a professional sports fact-checker specializing in football/soccer.
        Your task is to verify the factual accuracy of sports articles against provided game data.
        
        ABSOLUTE RULE:
        - You MUST ONLY use the provided game data for this specific match. DO NOT use any historical data, external knowledge, or make any assumptions not explicitly supported by the game data. If information is missing, do not invent or speculate.
        
        CRITICAL INSTRUCTIONS:
        1. Compare the article content with the provided game data
        2. Identify any factual errors or inconsistencies
        3. Correct ONLY the factual errors - do not change correct information
        4. Maintain the original writing style and tone
        5. Preserve the article structure and flow
        6. If no errors are found, return the original text unchanged
        
        FACT CHECKING CRITERIA:
        - Player names and spellings
        - Team names and spellings  
        - Match scores and results
        - Goal scorers and assist providers
        - Match events (goals, cards, substitutions)
        - Match timing and chronology
        - Venue and competition details
        - Statistics and numbers
        
        CRITICAL SUBSTITUTION RULES:
        - Check "startXI" vs "substitutes" arrays to determine who started vs who came on
        - "startXI" = players who started the match
        - "substitutes" = players who were on the bench
        - In events, "type": "subst" means a substitution occurred
        - Check the "player" field to see WHO was substituted OFF
        - Check the "assist" field to see WHO came ON as replacement
        - CRITICAL: ONLY mention substitutions when BOTH "player" AND "assist" fields are present
        - If "assist" field is null or missing, DO NOT mention the substitution at all
        - Example: If player A is in "startXI" and player B is in "substitutes", and there's a "subst" event with player A and assist B, then B replaced A
        - Focus on significant substitutions that impact the game
        - Only add missing substitutions if they are strategically important AND have complete data
        - DO NOT guess or assume who came on as a substitute
        - DO NOT mention partial substitution information (e.g., "Player X was substituted off" without knowing who replaced them)
        
        SEASON INFORMATION:
        - Check the "league.season" field for the correct season
        - Use format like "2021/22 season" not just "2021 season"
        
        PLAYER STATUS VERIFICATION:
        - Cross-reference events with lineup data
        - Verify if a player "started", "came on as substitute", or "was substituted off"
        - Be precise about substitution direction (on vs off)
        
        TEAM VERIFICATION:
        - Ensure players are correctly associated with their teams
        - Check team names in events vs lineup data
        
        OUTPUT FORMAT:
        - If errors found: Return the corrected article with factual errors fixed
        - If no errors: Return the original article unchanged
        - Do not add explanations, comments, or notes in the output
        - Do not add asterisks (*) or explanatory text
        - Return only the corrected article text without any editorial notes
        - The article should read naturally without any meta-commentary
        
        Remember: Only correct factual errors, preserve everything else exactly as written.
        """
    
    def get_terminology_checking_prompt(self) -> str:
        return """
        TASK: TERMINOLOGY CHECKING
        
        You are a professional sports terminology expert specializing in football/soccer.
        Your task is to verify and correct sports terminology usage in articles.
        
        ABSOLUTE RULE:
        - You MUST ONLY use the provided game data for this specific match. DO NOT use any historical data, external knowledge, or make any assumptions not explicitly supported by the game data. If information is missing, do not invent or speculate.
        
        CRITICAL INSTRUCTIONS:
        1. Review the article for sports terminology accuracy
        2. Identify any incorrect or inappropriate sports terms
        3. Correct ONLY the terminology errors - do not change correct terms
        4. Maintain the original writing style and tone
        5. Preserve the article structure and flow
        6. If no errors are found, return the original text unchanged
        
        TERMINOLOGY CHECKING CRITERIA:
        - Football/soccer specific terms (e.g., "goal kick" vs "kick-off")
        - Position names (e.g., "striker", "midfielder", "defender")
        - Action verbs (e.g., "scored", "assisted", "booked", "substituted")
        - Competition terms (e.g., "league", "cup", "championship")
        - Tactical terms (e.g., "formation", "tactics", "strategy")
        - Time-related terms (e.g., "first half", "second half", "extra time")
        - Statistical terms (e.g., "possession", "shots on target", "clean sheet")
        
        COMMON TERMINOLOGY CORRECTIONS:
        - "Soccer" → "football" (in international context)
        - "Field" → "pitch" (in football context)
        - "Game" → "match" (in football context)
        - "Player" → specific position when context allows
        - "Team" → specific team name when available
        
        OUTPUT FORMAT:
        - If errors found: Return the corrected article with terminology errors fixed
        - If no errors: Return the original article unchanged
        - Do not add explanations or comments in the output
        - Return only the corrected article text
        
        Remember: Only correct terminology errors, preserve everything else exactly as written.
        """

    async def edit_with_facts(self, text: str, game_info: Dict[str, Any]) -> str:
        """
        Edit article to correct factual errors based on game data.
        
        Args:
            text: The article text to fact-check
            game_info: Game data to verify facts against
            
        Returns:
            Corrected article text with factual errors fixed
        """
        try:
            logger.info("Starting fact-checking process")
            
            # Extract key data for easier verification
            response_data = game_info.get("response", [])
            if response_data and len(response_data) > 0:
                fixture_data = response_data[0]
                
                # Extract key information for fact-checking
                teams = fixture_data.get("teams", {})
                goals = fixture_data.get("goals", {})
                score = fixture_data.get("score", {})
                events = fixture_data.get("events", [])
                lineups = fixture_data.get("lineups", [])
                league = fixture_data.get("league", {})
                
                # Create a simplified data structure for fact-checking
                fact_check_data = {
                    "teams": teams,
                    "goals": goals,
                    "score": score,
                    "events": events,
                    "lineups": lineups,
                    "league": league,
                    "season": league.get("season"),
                    "venue": fixture_data.get("fixture", {}).get("venue", {}),
                    "referee": fixture_data.get("fixture", {}).get("referee"),
                    "date": fixture_data.get("fixture", {}).get("date")
                }
            else:
                fact_check_data = game_info
            
            # Prepare the prompt with game data
            prompt = f"""
            {self.get_fact_checking_prompt()}
            
            ARTICLE TO FACT-CHECK:
            {text}
            
            GAME DATA FOR VERIFICATION:
            {json.dumps(fact_check_data, indent=2, ensure_ascii=False)}
            
            Please fact-check the article against the provided game data and return the corrected version.
            Pay special attention to:
            1. Substitution events - who came on vs who went off
            2. Player status - who started vs who was a substitute
            3. Season information - use correct season format
            4. Team associations - ensure players are correctly linked to teams
            5. Focus on accuracy over completeness - only correct factual errors
            6. Maintain natural flow and readability of the article
            
            Only correct factual errors, preserve everything else unchanged.
            Do not add any notes, asterisks, or explanatory text to the article.
            """
            
            # Run fact-checking
            result = await Runner.run(self.agent, prompt)
            corrected_text = result.final_output_as(str).strip()
            
            logger.info("Fact-checking completed successfully")
            return corrected_text
            
        except Exception as e:
            logger.error(f"Error during fact-checking: {e}")
            # Return original text if fact-checking fails
            return text
    
    async def edit_with_terms(self, text: str) -> str:
        """
        Edit article to correct sports terminology usage.
        
        Args:
            text: The article text to check for terminology errors
            
        Returns:
            Corrected article text with terminology errors fixed
        """
        try:
            logger.info("Starting terminology checking process")
            
            # Prepare the prompt
            prompt = f"""
            {self.get_terminology_checking_prompt()}
            
            ARTICLE TO CHECK FOR TERMINOLOGY ERRORS:
            {text}
            
            Please check the article for sports terminology accuracy and return the corrected version.
            Only correct terminology errors, preserve everything else unchanged.
            """
            
            # Run terminology checking
            result = await Runner.run(self.agent, prompt)
            corrected_text = result.final_output_as(str).strip()
            
            logger.info("Terminology checking completed successfully")
            return corrected_text
            
        except Exception as e:
            logger.error(f"Error during terminology checking: {e}")
            # Return original text if terminology checking fails
            return text
    
    def validate_editing_result(self, original_text: str, edited_text: str) -> Dict[str, Any]:
        """
        Validate the editing result to ensure quality.
        
        Args:
            original_text: Original article text
            edited_text: Edited article text
            
        Returns:
            Validation results dictionary
        """
        try:
            validation_result = {
                "original_length": len(original_text.split()),
                "edited_length": len(edited_text.split()),
                "length_change": len(edited_text.split()) - len(original_text.split()),
                "has_changes": original_text != edited_text,
                "preserves_structure": self._check_structure_preservation(original_text, edited_text),
                "validation_passed": True
            }
            
            # Check if length change is reasonable (within 10% of original)
            length_ratio = abs(validation_result["length_change"]) / validation_result["original_length"]
            if length_ratio > 0.1:
                validation_result["warning"] = f"Significant length change detected: {validation_result['length_change']} words"
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error during validation: {e}")
            return {
                "validation_passed": False,
                "error": str(e)
            }
    
    def _check_structure_preservation(self, original_text: str, edited_text: str) -> bool:
        """
        Check if the article structure is preserved after editing.
        
        Args:
            original_text: Original article text
            edited_text: Edited article text
            
        Returns:
            True if structure is preserved, False otherwise
        """
        try:
            # Check for key structural elements
            structure_elements = ["Headline", "Introduction", "Body", "Conclusion"]
            
            original_has_structure = all(element in original_text for element in structure_elements)
            edited_has_structure = all(element in edited_text for element in structure_elements)
            
            return original_has_structure == edited_has_structure
            
        except Exception as e:
            logger.error(f"Error checking structure preservation: {e}")
            return False