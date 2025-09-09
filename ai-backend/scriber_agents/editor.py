import logging
from typing import Any, List, Dict, Tuple
from dotenv import load_dotenv
import json
import asyncio
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()
logger = logging.getLogger(__name__)

class Editor:
    async def _safe_chain_call(self, chain, input_data: dict, operation_name: str, timeout: float = 45.0):
        """Make a safe LangChain call with timeout."""
        try:
            result = await asyncio.wait_for(
                chain.ainvoke(input_data),
                timeout=timeout
            )
            return result
        except asyncio.TimeoutError:
            logger.error(f"{operation_name} timed out after {timeout} seconds")
            raise asyncio.TimeoutError(f"{operation_name} operation timed out")
        except Exception as e:
            logger.error(f"Error in {operation_name}: {e}")
            raise e

    def __init__(self, config: dict):
        self.config = config or {}
        
        # Initialize LangChain LLM
        self.llm = ChatOpenAI(
            model=self.config.get("model", "gpt-4o-mini"),
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.1,
            max_retries=3,
            request_timeout=30.0
        )
        
        # Initialize parsers
        self.json_parser = JsonOutputParser()
        self.string_parser = StrOutputParser()
        
        # Initialize specialized chains for different error types
        self.score_process_chain = self._create_json_chain("score_process")
        self.player_performance_chain = self._create_json_chain("player_performance")
        self.substitution_chain = self._create_json_chain("substitution")
        self.statistics_chain = self._create_json_chain("statistics")
        self.disciplinary_chain = self._create_json_chain("disciplinary")
        self.background_info_chain = self._create_json_chain("background_info")
        self.terminology_chain = self._create_json_chain("terminology")
        self.final_editor_chain = self._create_string_chain("final_editor")

        logger.info("Editor initialized successfully with LangChain modular validators")
    
    def _create_json_chain(self, prompt_type: str):
        """Create a LangChain chain for JSON output."""
        prompt_method = getattr(self, f"get_{prompt_type}_prompt")
        system_prompt = prompt_method()
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input_text}")
        ])
        
        return prompt | self.llm | self.json_parser
    
    def _create_string_chain(self, prompt_type: str):
        """Create a LangChain chain for string output."""
        prompt_method = getattr(self, f"get_{prompt_type}_prompt")
        system_prompt = prompt_method()
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input_text}")
        ])
        
        return prompt | self.llm | self.string_parser
    
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
        - If you see "second goal" or "brace" in the article, make sure it is real in the data. If the player only assisted, do not use "second goal" or "brace".
        - Note that "a goal and an assist" is not two goals, do not use "second goal" or "brace" unless it is real in the data
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
        - The goal can not be assigned to the assist player:
            - EXAMPLE: If Player A scores one goal assisted by Player B, and Player B scores one goal assisted by Player A, DO NOT write that either player "scored a double" or "netted twice".
                - For example, in the match where Arsenal beat Wolves 2-0, Saka scored once (assisted by Havertz) and Havertz scored once (assisted by Saka). Neither scored twice — this must NOT be described as a "brace" or "double".
        - When counting goals per player, treat only explicit scoring events in the CURRENT MATCH DATA as valid.
        - A player who scored one goal and provided one assist MUST NOT be described as scoring twice.
        - For clarity: DO NOT use phrases like "brace", "double", "netted twice", "second tally", or similar variations unless the player is explicitly recorded as scoring two distinct goals.
        - Goal count per player must match the number of goal events where the player is listed as "scorer".
        - Assist does NOT count as a goal. It could mean a goal assist or a substitution. Make sure to check the "type" field to determine if it is a substitution or a goal assist. A substitution is not a goal.

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
    
    def get_terminology_prompt(self) -> str:
        return """
        TASK: TERMINOLOGY VALIDATION
        
        You are a professional sports terminology expert specializing in football/soccer.
        Your task is to identify errors related to sports terminology usage in articles.
        
        ABSOLUTE RULE:
        - You MUST ONLY use the provided game data for this specific match. DO NOT use any historical data, external knowledge, or make any assumptions not explicitly supported by the game data.
        
        VALIDATION CRITERIA:
        1. Football/soccer specific terms (e.g., "goal kick" vs "kick-off")
        2. Position names (e.g., "striker", "midfielder", "defender")
        3. Action verbs (e.g., "scored", "assisted", "booked", "substituted")
        4. Competition terms (e.g., "league", "cup", "championship")
        5. Tactical terms (e.g., "formation", "tactics", "strategy")
        6. Time-related terms (e.g., "first half", "second half", "extra time")
        7. Statistical terms (e.g., "possession", "shots on target", "clean sheet")
        
        COMMON TERMINOLOGY ISSUES:
        - "Soccer" vs "football" (in international context)
        - "Field" vs "pitch" (in football context)
        - "Game" vs "match" (in football context)
        - Generic "player" vs specific position when context allows
        - Generic "team" vs specific team name when available
        
        OUTPUT FORMAT:
        Return a JSON object with the following structure:
        {{
            "errors_found": boolean,
            "error_type": "terminology",
            "errors": [
                {{
                    "error_description": "description of the terminology error",
                    "original_text": "exact text that contains the error",
                    "correction_suggestion": "suggested correction",
                    "severity": "high/medium/low"
                }}
            ],
            "corrected_sections": [
                {{
                    "original": "original text section",
                    "corrected": "corrected text section"
                }}
            ]
        }}
        
        If no errors found, return:
        {{
            "errors_found": false,
            "error_type": "terminology",
            "errors": [],
            "corrected_sections": []
        }}
        """

    def get_score_process_prompt(self) -> str:
        return """
        TASK: SCORE AND MATCH PROCESS VALIDATION
        
        You are a professional sports fact-checker specializing in football/soccer match scores and process.
        Your task is to identify errors related to match scores, goals, and match progression.
        
        ABSOLUTE RULES:
        - You MUST ONLY use the provided game data for this specific match. DO NOT use any historical data, external knowledge, or make any assumptions not explicitly supported by the game data.
        
        VALIDATION CRITERIA:
        1. Match final score accuracy
        2. Goal timing and sequence
        3. Goal scorers and assist providers
        4. Match progression (first half, second half, extra time)
        5. Match result (win, draw, loss)
        6. Goal descriptions and celebrations

        CRITICAL RULES:
        - A player who scored one goal and provided one assist MUST NOT be described as scoring twice
        - If a player scores 1 goal and assists another, they MUST NOT be described as scoring a second goal or "netting twice".
        - Any phrase implying a second goal ("scored again", "second goal", "sealed it with his brace", etc.) MUST only be used if the player scored *two separate goals* as "scorer" in the events list.
        - Check whether the player's name appears exactly twice as a "scorer". Otherwise, flag any statement implying multiple goals as factual error.
        - "Hat-trick" only for exactly 3 goals
        - Assist does NOT count as a goal, Example: If player A scores one goal assisted by Player B, and Player B scores one goal assisted by Player A, They both scored 1 goal each, DO NOT write that either player "scored a double" or "netted twice".
        
        ERROR IDENTIFICATION RULES:
        - Only report errors where the article text directly contradicts the game data
        - Be precise about the exact text that contains the error
        - Provide specific correction suggestions that directly address the factual error
        - Do not suggest rewording or style improvements
        
        OUTPUT FORMAT:
        Return a JSON object with the following structure:
        {{
            "errors_found": boolean,
            "error_type": "score_process",
            "errors": [
                {{
                    "error_description": "description of the factual error",
                    "original_text": "exact text that contains the error",
                    "correction_suggestion": "exact replacement text to fix the error",
                    "severity": "high/medium/low"
                }}
            ],
            "corrected_sections": [
                {{
                    "original": "original text section",
                    "corrected": "corrected text section"
                }}
            ]
        }}
        
        If no errors found, return:
        {{
            "errors_found": false,
            "error_type": "score_process",
            "errors": [],
            "corrected_sections": []
        }}
        """
    
    def get_player_performance_prompt(self) -> str:
        return """
        TASK: PLAYER PERFORMANCE VALIDATION
        
        You are a professional sports fact-checker specializing in football/soccer player performance.
        Your task is to identify errors related to individual player performances and achievements.
        
        ABSOLUTE RULES:
        - You MUST ONLY use the provided game data for this specific match. DO NOT use any historical data, external knowledge, or make any assumptions not explicitly supported by the game data.
        - ONLY identify factual errors - do not suggest improvements or enhancements
        - ONLY report errors that are clearly incorrect based on the provided data
        - DO NOT make subjective judgments about writing quality or style
        
        VALIDATION CRITERIA:
        1. Player goal scoring (number of goals, timing)
        2. Player assists (number of assists, timing)
        3. Player achievements (hat-tricks, braces, etc.)
        4. Player performance descriptions
        5. Player role and position accuracy
        6. Player impact on the match
        
        CRITICAL RULES:
        - A player who scored one goal and provided one assist MUST NOT be described as scoring twice
        - DO NOT use phrases like "brace", "double", "netted twice" unless the player scored exactly 2 goals
        - "Hat-trick" only for exactly 3 goals
        - Assist does NOT count as a goal
        
        ERROR IDENTIFICATION RULES:
        - Only report errors where the article text directly contradicts the game data
        - Be precise about the exact text that contains the error
        - Provide specific correction suggestions that directly address the factual error
        - Do not suggest rewording or style improvements
        
        OUTPUT FORMAT:
        Return a JSON object with the following structure:
        {{
            "errors_found": boolean,
            "error_type": "player_performance",
            "errors": [
                {{
                    "error_description": "description of the factual error",
                    "original_text": "exact text that contains the error",
                    "correction_suggestion": "exact replacement text to fix the error",
                    "severity": "high/medium/low"
                }}
            ],
            "corrected_sections": [
                {{
                    "original": "original text section",
                    "corrected": "corrected text section"
                }}
            ]
        }}
        
        If no errors found, return:
        {{
            "errors_found": false,
            "error_type": "player_performance",
            "errors": [],
            "corrected_sections": []
        }}
        """
    
    def get_substitution_prompt(self) -> str:
        return """
        TASK: SUBSTITUTION AND PLAYER STATUS VALIDATION
        
        You are a professional sports fact-checker specializing in football/soccer substitutions and player status.
        Your task is to identify errors related to player substitutions and starting/bench status.
        
        ABSOLUTE RULE:
        - You MUST ONLY use the provided game data for this specific match. DO NOT use any historical data, external knowledge, or make any assumptions not explicitly supported by the game data.
        
        VALIDATION CRITERIA:
        1. Starting XI vs substitutes
        2. Substitution events (who came on, who went off)
        3. Substitution timing
        4. Player status descriptions (started, came on, was substituted)
        5. Substitution impact on the game
        
        CRITICAL RULES:
        - Check "startXI" vs "substitutes" arrays to determine who started vs who was on bench
        - "type": "subst" events show substitutions
        - "player" field = who was substituted OFF
        - "assist" field = who came ON as replacement
        - ONLY mention substitutions when BOTH "player" AND "assist" fields are present
        - DO NOT guess or assume who came on as substitute
        - DO NOT mention partial substitution information
        
        OUTPUT FORMAT:
        Return a JSON object with the following structure:
        {{
            "errors_found": boolean,
            "error_type": "substitution",
            "errors": [
                {{
                    "error_description": "description of the error",
                    "original_text": "exact text that contains the error",
                    "correction_suggestion": "suggested correction",
                    "severity": "high/medium/low"
                }}
            ],
            "corrected_sections": [
                {{
                    "original": "original text section",
                    "corrected": "corrected text section"
                }}
            ]
        }}
        
        If no errors found, return:
        {{
            "errors_found": false,
            "error_type": "substitution",
            "errors": [],
            "corrected_sections": []
        }}
        """
    
    def get_statistics_prompt(self) -> str:
        return """
        TASK: MATCH STATISTICS VALIDATION
        
        You are a professional sports fact-checker specializing in football/soccer match statistics.
        Your task is to identify errors related to match statistics and data.
        
        ABSOLUTE RULE:
        - You MUST ONLY use the provided game data for this specific match. DO NOT use any historical data, external knowledge, or make any assumptions not explicitly supported by the game data.
        
        VALIDATION CRITERIA:
        1. Possession statistics
        2. Shots and shots on target
        3. Corner kicks
        4. Fouls and free kicks
        5. Offsides
        6. Other match statistics (passes, tackles, etc.)
        7. Team performance metrics
        
        OUTPUT FORMAT:
        Return a JSON object with the following structure:
        {{
            "errors_found": boolean,
            "error_type": "statistics",
            "errors": [
                {{
                    "error_description": "description of the error",
                    "original_text": "exact text that contains the error",
                    "correction_suggestion": "suggested correction",
                    "severity": "high/medium/low"
                }}
            ],
            "corrected_sections": [
                {{
                    "original": "original text section",
                    "corrected": "corrected text section"
                }}
            ]
        }}
        
        If no errors found, return:
        {{
            "errors_found": false,
            "error_type": "statistics",
            "errors": [],
            "corrected_sections": []
        }}
        """
    
    def get_disciplinary_prompt(self) -> str:
        return """
        TASK: DISCIPLINARY EVENTS VALIDATION
        
        You are a professional sports fact-checker specializing in football/soccer disciplinary events.
        Your task is to identify errors related to yellow cards, red cards, and disciplinary actions.
        
        ABSOLUTE RULE:
        - You MUST ONLY use the provided game data for this specific match. DO NOT use any historical data, external knowledge, or make any assumptions not explicitly supported by the game data.
        
        VALIDATION CRITERIA:
        1. Yellow card events (timing, players, reasons)
        2. Red card events (timing, players, reasons)
        3. Disciplinary action descriptions
        4. Card accumulation and consequences
        5. Referee decisions and timing
        
        OUTPUT FORMAT:
        Return a JSON object with the following structure:
        {{
            "errors_found": boolean,
            "error_type": "disciplinary",
            "errors": [
                {{
                    "error_description": "description of the error",
                    "original_text": "exact text that contains the error",
                    "correction_suggestion": "suggested correction",
                    "severity": "high/medium/low"
                }}
            ],
            "corrected_sections": [
                {{
                    "original": "original text section",
                    "corrected": "corrected text section"
                }}
            ]
        }}
        
        If no errors found, return:
        {{
            "errors_found": false,
            "error_type": "disciplinary",
            "errors": [],
            "corrected_sections": []
        }}
        """
    
    def get_background_info_prompt(self) -> str:
        return """
        TASK: BACKGROUND INFORMATION VALIDATION
        
        You are a professional sports fact-checker specializing in football/soccer background information.
        Your task is to identify errors related to background information and ensure it's properly placed in the introduction.
        
        ABSOLUTE RULE:
        - You MUST ONLY use the provided game data for this specific match. DO NOT use any historical data, external knowledge, or make any assumptions not explicitly supported by the game data.
        
        VALIDATION CRITERIA:
        1. Season information accuracy
        2. League and competition details
        3. Team background and context
        4. Player background information
        5. Historical context relevance
        6. Background information placement (should be in introduction)
        
        CRITICAL RULES:
        - Background information should be accurate and relevant to this specific match
        - Background information should primarily appear in the introduction
        - Avoid mixing background info with match events
        - Ensure season format is correct (e.g., "2021/22 season")
        
        OUTPUT FORMAT:
        Return a JSON object with the following structure:
        {{
            "errors_found": boolean,
            "error_type": "background_info",
            "errors": [
                {{
                    "error_description": "description of the error",
                    "original_text": "exact text that contains the error",
                    "correction_suggestion": "suggested correction",
                    "severity": "high/medium/low"
                }}
            ],
            "corrected_sections": [
                {{
                    "original": "original text section",
                    "corrected": "corrected text section"
                }}
            ]
        }}
        
        If no errors found, return:
        {{
            "errors_found": false,
            "error_type": "background_info",
            "errors": [],
            "corrected_sections": []
        }}
        """
    
    def get_final_editor_prompt(self) -> str:
        return """
        TASK: FINAL ARTICLE EDITOR
        
        You are a professional sports editor specializing in football/soccer articles.
        Your task is to apply ONLY the corrections identified by the validation agents and produce the final corrected article.
        
        ABSOLUTE RESTRICTIONS:
        - ONLY correct errors that are explicitly identified in the validation results
        - DO NOT make any changes that are not specifically requested in the validation results
        - DO NOT add, remove, or modify any content unless it is a direct correction of an identified error
        - DO NOT improve, enhance, or rewrite any parts of the article
        - DO NOT change the writing style, tone, or structure beyond what is necessary for error correction
        - DO NOT add any new information, even if it seems relevant or helpful
        - DO NOT make assumptions about what might be "better" or "more accurate"
        
        INSTRUCTIONS:
        1. Review the validation results carefully
        2. Apply ONLY the specific corrections listed in the validation results
        3. Make minimal changes - only what is absolutely necessary to fix identified errors
        4. Preserve all original content that is not explicitly marked as needing correction
        5. Maintain the exact same structure and flow as the original article
        
        VALIDATION TYPES TO HANDLE:
        - score_process: Match scores, goals, and match progression errors
        - player_performance: Player achievements, goals, assists, and performance descriptions
        - substitution: Player substitutions, starting XI, and player status
        - statistics: Match statistics and data accuracy
        - disciplinary: Yellow cards, red cards, and disciplinary actions
        - background_info: Season information, league details, and background context
        - terminology: Sports terminology usage and accuracy
        
        CRITICAL RULES:
        - Apply corrections exactly as suggested in the validation results
        - Do not add any new information not supported by the game data
        - Do not add explanatory notes, asterisks, or any meta-commentary
        - Return only the corrected article text
        - If no errors are found in validation results, return the original article unchanged
        - If validation results are empty or indicate no errors, return the original article unchanged
        
        ERROR CORRECTION PROCESS:
        1. For each error in the validation results:
           - Locate the exact text mentioned in "original_text"
           - Replace it with the exact text from "correction_suggestion"
           - Make no other changes to that section
        2. If no errors are found, return the original article unchanged
        3. Do not make any other modifications
        
        OUTPUT FORMAT:
        Return the final corrected article text only, without any additional notes or explanations.
        If no corrections are needed, return the original article exactly as provided.
        """
    
    async def validate_article(self, text: str, game_info: Dict[str, Any], research_insights: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Run all validation checks on the article and return comprehensive error report.
        
        Args:
            text: The article text to validate
            game_info: Game data to verify facts against
            research_insights: Research insights and context data
            
        Returns:
            Comprehensive validation results with all error types
        """
        try:
            logger.info("Starting comprehensive article validation")
            
            # Extract and structure data for different validation types
            validation_data = self._prepare_validation_data(game_info, research_insights)
            
            # Run all validation checks in parallel with appropriate data
            validation_tasks = [
                self._validate_score_process(text, validation_data["score_process"]),
                self._validate_player_performance(text, validation_data["player_performance"]),
                self._validate_substitutions(text, validation_data["substitution"]),
                self._validate_statistics(text, validation_data["statistics"]),
                self._validate_disciplinary(text, validation_data["disciplinary"]),
                self._validate_background_info(text, validation_data["background_info"]),
                self._validate_terminology(text, validation_data["terminology"])
            ]
            
            # Wait for all validations to complete
            validation_results = await asyncio.gather(*validation_tasks, return_exceptions=True)
            
            # Compile comprehensive results
            comprehensive_results = {
                "total_errors": 0,
                "error_types": {},
                "all_errors": [],
                "validation_summary": {}
            }
            
            error_types = [
                "score_process", "player_performance", "substitution", 
                "statistics", "disciplinary", "background_info", "terminology"
            ]
            
            for i, result in enumerate(validation_results):
                if isinstance(result, Exception):
                    logger.error(f"Validation error in {error_types[i]}: {result}")
                    comprehensive_results["error_types"][error_types[i]] = {
                        "errors_found": False,
                        "error": str(result)
                    }
                else:
                    comprehensive_results["error_types"][error_types[i]] = result
                    if result.get("errors_found", False):
                        comprehensive_results["total_errors"] += len(result.get("errors", []))
                        comprehensive_results["all_errors"].extend(result.get("errors", []))
            
            logger.info(f"Validation completed. Total errors found: {comprehensive_results['total_errors']}")
            logger.info(f"Validation results: {comprehensive_results}")
            logger.info(f"Original article: {text}")
            return comprehensive_results
            
        except Exception as e:
            logger.error(f"Error during article validation: {e}")
            return {
                "total_errors": 0,
                "error_types": {},
                "all_errors": [],
                "validation_summary": {"error": str(e)}
            }
    
    async def edit_with_facts(self, text: str, game_info: Dict[str, Any], research_insights: Dict[str, Any] = None) -> str:
        """
        Edit article to correct factual errors based on comprehensive validation.
        
        Args:
            text: The article text to fact-check
            game_info: Game data to verify facts against
            research_insights: Research insights and context data
            
        Returns:
            Corrected article text with factual errors fixed
        """
        try:
            logger.info("Starting comprehensive fact-checking process")
            
            # First, run all validations
            validation_results = await self.validate_article(text, game_info, research_insights)

            
            # Prepare the final editor prompt with all validation results
            prompt = f"""
            {self.get_final_editor_prompt()}
            
            ORIGINAL ARTICLE:
            {text}
            
            GAME DATA:
            {json.dumps(game_info, indent=2, ensure_ascii=False)}
            
            RESEARCH INSIGHTS:
            {json.dumps(research_insights, indent=2, ensure_ascii=False) if research_insights else "{}"}
            
            VALIDATION RESULTS:
            {json.dumps(validation_results, indent=2, ensure_ascii=False)}
            
            Please apply all the corrections identified in the validation results and return the final corrected article.
            """
            
            # Run final editing with safe timeout
            try:
                result = await self._safe_chain_call(
                    self.final_editor_chain, 
                    {"input_text": prompt}, 
                    "final editing", 
                    timeout=60.0
                )
                corrected_text = result.strip()
                
                logger.info("Comprehensive fact-checking completed successfully")
                return corrected_text
                
            except asyncio.TimeoutError:
                logger.error("Final editing timed out after 60 seconds")
                # Return original text with a note about timeout
                return f"{text}\n\n[Note: Automated fact-checking timed out - article returned as-is]"
            
        except Exception as e:
            logger.error(f"Error during fact-checking: {e}")
            # Return original text if fact-checking fails
            return text
    
    def _prepare_validation_data(self, game_info: Dict[str, Any], research_insights: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Prepare validation data for different validation types.
        
        Args:
            game_info: Game data from pipeline
            research_insights: Research insights from pipeline
            
        Returns:
            Dictionary with data prepared for each validation type
        """
        try:
            # Extract base game data
            base_game_data = self._extract_game_data(game_info)
            
            # Prepare data for each validation type
            validation_data = {
                "score_process": self._prepare_score_process_data(base_game_data),
                "player_performance": self._prepare_player_performance_data(base_game_data, research_insights),
                "substitution": self._prepare_substitution_data(base_game_data),
                "statistics": self._prepare_statistics_data(base_game_data),
                "disciplinary": self._prepare_disciplinary_data(base_game_data),
                "background_info": self._prepare_background_info_data(base_game_data, research_insights),
                "terminology": self._prepare_terminology_data(base_game_data, research_insights)
            }
            
            return validation_data
            
        except Exception as e:
            logger.error(f"Error preparing validation data: {e}")
            # Return empty data structure if preparation fails
            return {
                "score_process": {},
                "player_performance": {},
                "substitution": {},
                "statistics": {},
                "disciplinary": {},
                "background_info": {},
                "terminology": {}
            }
    
    def _extract_game_data(self, game_info: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and structure game data for validation."""
        try:
            # Handle both raw API response format and compact format
            if "response" in game_info:
                # Raw API response format
                response_data = game_info.get("response", [])
                if response_data and len(response_data) > 0:
                    fixture_data = response_data[0]
                    
                    return {
                        "teams": fixture_data.get("teams", {}),
                        "goals": fixture_data.get("goals", {}),
                        "score": fixture_data.get("score", {}),
                        "events": fixture_data.get("events", []),
                        "lineups": fixture_data.get("lineups", []),
                        "league": fixture_data.get("league", {}),
                        "season": fixture_data.get("league", {}).get("season"),
                        "venue": fixture_data.get("fixture", {}).get("venue", {}),
                        "referee": fixture_data.get("fixture", {}).get("referee"),
                        "date": fixture_data.get("fixture", {}).get("date")
                    }
            else:
                # Compact format from pipeline
                return game_info
                
        except Exception as e:
            logger.error(f"Error extracting game data: {e}")
            return game_info
    
    def _prepare_score_process_data(self, base_game_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for score and match process validation."""
        return {
            "teams": base_game_data.get("teams", {}),
            "goals": base_game_data.get("goals", {}),
            "score": base_game_data.get("score", {}),
            "events": base_game_data.get("events", []),
            "league": base_game_data.get("league", {}),
            "fixture": {
                "date": base_game_data.get("date"),
                "venue": base_game_data.get("venue", {})
            }
        }
    
    def _prepare_player_performance_data(self, base_game_data: Dict[str, Any], research_insights: Dict[str, Any] = None) -> Dict[str, Any]:
        """Prepare data for player performance validation."""
        data = {
            "events": base_game_data.get("events", []),
            "lineups": base_game_data.get("lineups", []),
            "teams": base_game_data.get("teams", {})
        }
        
        # Add research insights if available
        if research_insights:
            data["research_insights"] = research_insights.get("player_performance", [])
        
        return data
    
    def _prepare_substitution_data(self, base_game_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for substitution validation."""
        return {
            "events": base_game_data.get("events", []),
            "lineups": base_game_data.get("lineups", []),
            "teams": base_game_data.get("teams", {})
        }
    
    def _prepare_statistics_data(self, base_game_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for statistics validation."""
        return {
            "statistics": base_game_data.get("statistics", []),
            "teams": base_game_data.get("teams", {})
        }
    
    def _prepare_disciplinary_data(self, base_game_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for disciplinary validation."""
        return {
            "events": base_game_data.get("events", []),
            "teams": base_game_data.get("teams", {}),
            "fixture": {
                "referee": base_game_data.get("referee")
            }
        }
    
    def _prepare_background_info_data(self, base_game_data: Dict[str, Any], research_insights: Dict[str, Any] = None) -> Dict[str, Any]:
        """Prepare data for background information validation."""
        data = {
            "league": base_game_data.get("league", {}),
            "teams": base_game_data.get("teams", {}),
            "fixture": {
                "date": base_game_data.get("date"),
                "venue": base_game_data.get("venue", {})
            }
        }
        
        # Add research insights if available
        if research_insights:
            data["research_insights"] = {
                "historical_context": research_insights.get("historical_context", []),
                "game_analysis": research_insights.get("game_analysis", [])
            }
        
        return data
    
    def _prepare_terminology_data(self, base_game_data: Dict[str, Any], research_insights: Dict[str, Any] = None) -> Dict[str, Any]:
        """Prepare data for terminology validation."""
        data = {
            "teams": base_game_data.get("teams", {}),
            "league": base_game_data.get("league", {}),
            "events": base_game_data.get("events", [])
        }
        
        # Add research insights if available
        if research_insights:
            data["research_insights"] = research_insights
        
        return data
    
    async def _validate_score_process(self, text: str, game_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate score and match process."""
        try:
            input_text = f"""
            ARTICLE TO VALIDATE:
            {text}
            
            GAME DATA:
            {json.dumps(game_data, indent=2, ensure_ascii=False)}
            
            Please validate the article for score and match process errors.
            """
            
            result = await self._safe_chain_call(
                self.score_process_chain, 
                {"input_text": input_text}, 
                "score process validation"
            )
            return result
        except Exception as e:
            logger.error(f"Error in score process validation: {e}")
            return {"errors_found": False, "error": str(e)}
    
    async def _validate_player_performance(self, text: str, game_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate player performance."""
        try:
            input_text = f"""
            ARTICLE TO VALIDATE:
            {text}
            
            GAME DATA:
            {json.dumps(game_data, indent=2, ensure_ascii=False)}
            
            Please validate the article for player performance errors.
            """
            
            result = await self._safe_chain_call(
                self.player_performance_chain, 
                {"input_text": input_text}, 
                "player performance validation"
            )
            return result
        except Exception as e:
            logger.error(f"Error in player performance validation: {e}")
            return {"errors_found": False, "error": str(e)}
    
    async def _validate_substitutions(self, text: str, game_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate substitutions and player status."""
        try:
            input_text = f"""
            ARTICLE TO VALIDATE:
            {text}
            
            GAME DATA:
            {json.dumps(game_data, indent=2, ensure_ascii=False)}
            
            Please validate the article for substitution and player status errors.
            """
            
            result = await self._safe_chain_call(
                self.substitution_chain, 
                {"input_text": input_text}, 
                "substitution validation"
            )
            return result
        except Exception as e:
            logger.error(f"Error in substitution validation: {e}")
            return {"errors_found": False, "error": str(e)}
    
    async def _validate_statistics(self, text: str, game_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate match statistics."""
        try:
            input_text = f"""
            ARTICLE TO VALIDATE:
            {text}
            
            GAME DATA:
            {json.dumps(game_data, indent=2, ensure_ascii=False)}
            
            Please validate the article for statistics errors.
            """
            
            result = await self._safe_chain_call(
                self.statistics_chain, 
                {"input_text": input_text}, 
                "statistics validation"
            )
            return result
        except Exception as e:
            logger.error(f"Error in statistics validation: {e}")
            return {"errors_found": False, "error": str(e)}
    
    async def _validate_disciplinary(self, text: str, game_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate disciplinary events."""
        try:
            input_text = f"""
            ARTICLE TO VALIDATE:
            {text}
            
            GAME DATA:
            {json.dumps(game_data, indent=2, ensure_ascii=False)}
            
            Please validate the article for disciplinary event errors.
            """
            
            result = await self._safe_chain_call(
                self.disciplinary_chain, 
                {"input_text": input_text}, 
                "disciplinary validation"
            )
            return result
        except Exception as e:
            logger.error(f"Error in disciplinary validation: {e}")
            return {"errors_found": False, "error": str(e)}
    
    async def _validate_background_info(self, text: str, game_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate background information."""
        try:
            input_text = f"""
            ARTICLE TO VALIDATE:
            {text}
            
            GAME DATA:
            {json.dumps(game_data, indent=2, ensure_ascii=False)}
            
            Please validate the article for background information errors.
            """
            
            result = await self._safe_chain_call(
                self.background_info_chain, 
                {"input_text": input_text}, 
                "background info validation"
            )
            return result
        except Exception as e:
            logger.error(f"Error in background info validation: {e}")
            return {"errors_found": False, "error": str(e)}
    
    async def _validate_terminology(self, text: str, game_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate terminology usage."""
        try:
            input_text = f"""
            ARTICLE TO VALIDATE:
            {text}
            
            GAME DATA:
            {json.dumps(game_data, indent=2, ensure_ascii=False)}
            
            Please validate the article for terminology errors.
            """
            
            result = await self._safe_chain_call(
                self.terminology_chain, 
                {"input_text": input_text}, 
                "terminology validation"
            )
            return result
        except Exception as e:
            logger.error(f"Error in terminology validation: {e}")
            return {"errors_found": False, "error": str(e)}
    
    async def edit_with_terms(self, text: str, game_info: Dict[str, Any] = None) -> str:
        """
        Edit article to correct sports terminology usage.
        
        Args:
            text: The article text to check for terminology errors
            game_info: Optional game data for context
            
        Returns:
            Corrected article text with terminology errors fixed
        """
        try:
            logger.info("Starting terminology checking process")
            
            # Extract game data if provided
            game_data = self._extract_game_data(game_info) if game_info else {}
            
            # Run terminology validation
            terminology_result = await self._validate_terminology(text, game_data)
            
            if terminology_result.get('errors_found', False):
                # Apply corrections using final editor
                prompt = f"""
                {self.get_final_editor_prompt()}
                
                ORIGINAL ARTICLE:
                {text}
                
                GAME DATA:
                {json.dumps(game_data, indent=2, ensure_ascii=False)}
                
                VALIDATION RESULTS:
                {json.dumps({"error_types": {"terminology": terminology_result}}, indent=2, ensure_ascii=False)}
                
                Please apply all the terminology corrections identified in the validation results and return the final corrected article.
                """
                
                result = await self._safe_chain_call(
                    self.final_editor_chain, 
                    {"input_text": prompt}, 
                    "terminology editing"
                )
                corrected_text = result.strip()
            else:
                corrected_text = text
            
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