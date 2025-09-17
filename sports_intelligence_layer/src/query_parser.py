from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import re
import json
import logging
from pathlib import Path
import unicodedata
from datetime import datetime, timedelta

class EntityType(Enum):
    PLAYER = "player"
    TEAM = "team"
    COMPETITION = "competition"
    STATISTIC = "statistic"
    TIME_PERIOD = "time_period"
    OPPONENT = "opponent"
    VENUE = "venue"

class ComparisonType(Enum):
    VS_AVERAGE = "vs_average"
    VS_CAREER = "vs_career"
    VS_OPPONENT = "vs_opponent"
    VS_SEASON = "vs_season"
    HEAD_TO_HEAD = "head_to_head"
    LEAGUE_RANKING = "league_ranking"

class TimeContext(Enum):
    THIS_SEASON = "this_season"
    LAST_SEASON = "last_season"
    CAREER = "career"
    LAST_N_GAMES = "last_n_games"
    CURRENT_MONTH = "current_month"
    CHAMPIONS_LEAGUE = "champions_league"
    LEAGUE_ONLY = "league_only"

@dataclass
class SoccerEntity:
    name: str
    entity_type: EntityType
    aliases: List[str] = field(default_factory=list)
    confidence: float = 1.0

@dataclass
class ParsedSoccerQuery:
    original_query: str
    entities: List[SoccerEntity]
    time_context: TimeContext
    comparison_type: Optional[ComparisonType] = None
    filters: Dict[str, Any] = field(default_factory=dict)
    statistic_requested: Optional[str] = None  # Deprecated: kept for backward compatibility
    statistics_requested: List[str] = field(default_factory=list)  # New: supports multiple stats
    confidence: float = 1.0
    query_intent: str = "stat_lookup"  # stat_lookup, comparison, historical, context

class SoccerQueryParser:
    def __init__(self):
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Create console handler if not exists
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # Base directory for external dictionaries (optional)
        data_dir = (Path(__file__).resolve().parent.parent / "data")

        # Load alias dictionaries if present; otherwise fall back to built-ins
        self.player_alias_dict: Dict[str, List[str]] = self._load_dict_if_exists(
            data_dir / "players.json",
            default={
                "erling haaland": ["haaland", "erling haaland", "erling"],
                "lionel messi": ["messi", "lionel messi"],
                "karim benzema": ["benzema", "karim benzema"],
                "mohamed salah": ["salah", "mo salah", "mohamed salah"],
                "kevin de bruyne": ["de bruyne", "kdb", "kevin de bruyne"],
                "harry kane": ["kane", "harry kane"]
            },
        )

        self.team_alias_dict: Dict[str, List[str]] = self._load_dict_if_exists(
            data_dir / "teams.json",
            default={
                "arsenal": ["arsenal", "gunners", "arsenal fc"],
                "liverpool": ["liverpool", "reds", "liverpool fc"],
                "real madrid": ["real madrid", "madrid"],
                "barcelona": ["barcelona", "barca"],
                "manchester city": ["manchester city", "man city", "city"],
                "manchester united": ["manchester united", "man utd", "united"],
                "chelsea": ["chelsea"],
                "bayern munich": ["bayern munich", "bayern"],
                "juventus": ["juventus", "juve"],
                "psg": ["psg", "paris saint-germain", "paris"]
            },
        )

        # Known sets for quick checks (lowercased canonical keys and aliases)
        self.known_players = {alias for aliases in self.player_alias_dict.values() for alias in aliases}
        self.known_teams = {alias for aliases in self.team_alias_dict.values() for alias in aliases}

        self.logger.info(f"Loaded {len(self.player_alias_dict)} player entities with {len(self.known_players)} total aliases")
        self.logger.info(f"Loaded {len(self.team_alias_dict)} team entities with {len(self.known_teams)} total aliases")
        
        # Compiled regex for fast alias detection
        self.player_alias_regex = self._compile_alias_regex(self.known_players)
        self.team_alias_regex = self._compile_alias_regex(self.known_teams)
        
        # Load derby/rivalry knowledge
        self.derby_knowledge = self._load_derby_knowledge(data_dir)
        
        # Load tactical context patterns
        self.tactical_patterns = self._load_tactical_patterns(data_dir)
        
        # Load special cases configuration
        self.special_cases = self._load_special_cases(data_dir)
        
        # Load ranking keywords configuration
        self.ranking_keywords = self._load_ranking_keywords(data_dir)
        
        # Define pattern strings
        player_pattern_strings = [
            r'(?:has|have|did)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:scored|assisted|played)',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s*\'s',
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:performance|stats?|statistics)',
            r'\b(?:player|striker|midfielder|defender|goalkeeper)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)'
        ]
        
        team_pattern_strings = [
            r'\b(Arsenal|Barcelona|Real Madrid|Manchester United|Liverpool|Chelsea|Bayern Munich|PSG|Inter Milan|AC Milan|Juventus|Manchester City|Tottenham|Atletico Madrid|Borussia Dortmund|City|United)\b',
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:record|performance|results?)\b'
        ]
        
        # Pre-compile player and team patterns for better performance
        self.compiled_player_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in player_pattern_strings]
        self.compiled_team_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in team_pattern_strings]
        
        # Statistics patterns; allow external override via data/statistics.json
        default_stat_patterns = {
            'goals': r'\b(?:goals?|scored|scoring|goalscorer|shots?|shooting)\b',
            'assists': r'\b(?:assists?|assisted|assisting)\b',
            'yellow_cards': r'\b(?:yellow cards?|yellows?|bookings?|booked)\b',
            'red_cards': r'\b(?:red cards?|reds?|sent off|dismissals?)\b',
            'clean_sheets': r'\b(?:clean sheets?|shutouts?)\b',
            'pass_completion': r'\b(?:pass completion|passing accuracy|pass rate)\b',
            'possession': r'\b(?:possession|ball possession)\b',
            'tackles': r'\b(?:tackles?|tackling)\b',
            'saves': r'\b(?:saves?|saving)\b',
            'minutes': r'\b(?:minutes?|mins?|playing time)\b',
            'performance': r'\b(?:performance|stats?|statistics|overall|complete)\b'
        }
        self.stat_patterns = self._load_stat_patterns(data_dir / "statistics.json", default_stat_patterns)
        
        # Pre-compile statistics patterns for performance
        self.compiled_stat_patterns = {}
        for stat_name, pattern in self.stat_patterns.items():
            self.compiled_stat_patterns[stat_name] = re.compile(pattern, re.IGNORECASE)
        
        # Create a fast lookup dictionary for common statistics
        self._stat_keyword_lookup = {}
        for stat_name, pattern in self.stat_patterns.items():
            # Extract keywords from pattern for fast preliminary check
            keywords = self._extract_keywords_from_pattern(pattern)
            for keyword in keywords:
                if keyword not in self._stat_keyword_lookup:
                    self._stat_keyword_lookup[keyword] = []
                self._stat_keyword_lookup[keyword].append(stat_name)
        
        # Time patterns - pre-compile for performance
        time_pattern_strings = {
            TimeContext.THIS_SEASON: r'\b(?:this season|current season|2024-25|2024/25)\b',
            TimeContext.LAST_SEASON: r'\b(?:last season|previous season|2023-24|2023/24)\b',
            TimeContext.CAREER: r'\b(?:career|all time|total|overall)\b',
            TimeContext.LAST_N_GAMES: r'\b(?:last|past)\s+(\d+)\s+(?:games?|matches?)\b',
            TimeContext.CHAMPIONS_LEAGUE: r'\b(?:Champions League|UCL|CL)\b',
            TimeContext.LEAGUE_ONLY: r'\b(?:Premier League|La Liga|Serie A|Bundesliga|Ligue 1|league)\b'
        }
        
        self.compiled_time_patterns = {}
        for time_context, pattern in time_pattern_strings.items():
            self.compiled_time_patterns[time_context] = re.compile(pattern, re.IGNORECASE)
        
        # Comparison patterns - pre-compile for performance
        comparison_pattern_strings = {
            ComparisonType.VS_AVERAGE: r'\b(?:compared to|vs|versus)\s+(?:average|normal|typical)\b',
            ComparisonType.VS_CAREER: r'\b(?:compared to|vs|versus)?\s+(?:career|overall)\s+average\b',
            ComparisonType.VS_OPPONENT: r'\b(?:compared to|vs|versus)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b',
            ComparisonType.HEAD_TO_HEAD: r'\b(?:head to head|h2h)\s+(?:record|against)\b'
        }
        
        self.compiled_comparison_patterns = {}
        for comp_type, pattern in comparison_pattern_strings.items():
            self.compiled_comparison_patterns[comp_type] = re.compile(pattern, re.IGNORECASE)
        
        # Additional commonly used patterns
        self._compiled_common_patterns = {
            'derby': re.compile(r'\b(?:derby|derbies)\b', re.IGNORECASE),
            'big_six': re.compile(r'\b(?:big six|top 6|top six)\b', re.IGNORECASE),
            'vs_keywords': re.compile(r'\b(?:vs|versus|against)\b', re.IGNORECASE),
            'home_venue': re.compile(r'\b(?:at home|home games?|home matches?|home form|home record|home performance)\b', re.IGNORECASE),
            'away_venue': re.compile(r'\b(?:away from home|on the road|away games?|away matches?|away form|away record|away performance|away)\b', re.IGNORECASE),
            'context_keywords': re.compile(r'\b(?:context|significance|important|why|how significant|storylines?|fans|game|verify|verification)\b', re.IGNORECASE),
            'historical_keywords': re.compile(r'\b(?:when|history|last time|historical|first.*since|since.*first|career|all.*time|milestone|record|achievement|trend|progression|legacy|breakthrough|debut|anniversary)\b', re.IGNORECASE),
            'comparison_keywords': re.compile(r'\b(?:compare|better|worse|than)\b', re.IGNORECASE),
            'milestone_keywords': re.compile(r'\b(?:milestone|achievement|record|breakthrough|first.*time|debut|anniversary|legacy|historic|unprecedented)\b', re.IGNORECASE),
            'trend_keywords': re.compile(r'\b(?:trend|trending|progression|improvement|decline|development|evolution|trajectory|pattern)\b', re.IGNORECASE),
            'career_keywords': re.compile(r'\b(?:career|all.*time|lifetime|total|overall|entire|whole.*career)\b', re.IGNORECASE)
        }
        
        # Dictionary for query normalization
        self._normalization_lookup = {}
        
    def _extract_keywords_from_pattern(self, pattern: str) -> List[str]:
        """Extract keywords from regex pattern for fast lookup."""
        # Simple keyword extraction for common patterns
        keywords = []
        # Remove regex symbols and split by | for alternatives
        clean_pattern = pattern.replace('\\b', '').replace('(?:', '').replace(')', '').replace('?', '')
        parts = clean_pattern.split('|')
        for part in parts:
            if part.strip() and len(part.strip()) > 2:
                keywords.append(part.strip().lower())
        return keywords

    def parse_query(self, query: str) -> ParsedSoccerQuery:
        """Parse a natural language soccer query into structured components."""
        self.logger.info(f"=== PARSING QUERY: '{query}' ===")
        
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
            
        entities = self._extract_entities(query)
        self.logger.info(f"Extracted {len(entities)} entities: {[(e.name, e.entity_type.value, e.confidence) for e in entities]}")
        
        time_context = self._extract_time_context(query)
        self.logger.info(f"Time context: {time_context.value}")
        
        comparison_type = self._extract_comparison_type(query)
        if comparison_type:
            self.logger.info(f"Comparison type: {comparison_type.value}")
        
        # Extract both single and multiple statistics
        statistics = self._extract_statistics(query)
        statistic = statistics[0] if statistics else None  # For backward compatibility
        if statistics:
            self.logger.info(f"Statistics requested: {statistics}")
        
        filters = self._extract_filters(query)
        if filters:
            self.logger.info(f"Filters extracted: {filters}")
        
        intent = self._determine_intent(query, entities, comparison_type)
        self.logger.info(f"Query intent: {intent}")
        
        confidence = self._calculate_confidence(entities, time_context, statistic)
        self.logger.info(f"Overall confidence: {confidence:.2f}")
        
        return ParsedSoccerQuery(
            original_query=query,
            entities=entities,
            time_context=time_context,
            comparison_type=comparison_type,
            filters=filters,
            statistic_requested=statistic,  # Backward compatibility
            statistics_requested=statistics,  # New multiple statistics support
            confidence=confidence,
            query_intent=intent
        )
    
    def _extract_entities(self, query: str) -> List[SoccerEntity]:
        """Extract player, team, and other entities from the query."""
        entities: List[SoccerEntity] = []
        added_keys: set = set()
        
        self.logger.info("--- Entity Extraction Phase ---")
        
        # First: alias-based extraction using compiled regex (players and teams)
        self.logger.info("1. Alias-based extraction (regex)")
        for match in re.finditer(self.player_alias_regex, query):
            alias_surface = match.group(0)
            key = self._normalize_text(alias_surface)
            self.logger.info(f"   Found player alias: '{alias_surface}' -> normalized: '{key}'")
            if key not in added_keys:
                entities.append(SoccerEntity(
                    name=self._title_or_preserve(alias_surface),
                    entity_type=EntityType.PLAYER,
                    confidence=0.97,
                ))
                added_keys.add(key)
                self.logger.info(f"    Added player entity: {self._title_or_preserve(alias_surface)} (confidence: 0.97)")
        
        for match in re.finditer(self.team_alias_regex, query):
            alias_surface = match.group(0)
            key = self._normalize_text(alias_surface)
            self.logger.info(f"   Found team alias: '{alias_surface}' -> normalized: '{key}'")
            if key not in added_keys:
                entities.append(SoccerEntity(
                    name=self._title_or_preserve(alias_surface),
                    entity_type=EntityType.TEAM,
                    confidence=0.95,
                ))
                added_keys.add(key)
                self.logger.info(f"    Added team entity: {self._title_or_preserve(alias_surface)} (confidence: 0.95)")
        
        # Then try pattern matching for unknown entities
        self.logger.info("2. Pattern-based extraction")
        # Extract players using pre-compiled patterns
        for compiled_pattern in self.compiled_player_patterns:
            matches = compiled_pattern.finditer(query)
            for match in matches:
                player_name = match.group(1)
                self.logger.info(f"   Pattern match for player: '{player_name}'")
                if self._is_likely_player(player_name):
                    # Check if we already have this player
                    if not any(e.name.lower() == player_name.lower() for e in entities):
                        entities.append(SoccerEntity(
                            name=player_name,
                            entity_type=EntityType.PLAYER,
                            confidence=0.85
                        ))
                        self.logger.info(f"    Added pattern-based player: {player_name} (confidence: 0.85)")
                    else:
                        self.logger.info(f"    Skipped duplicate player: {player_name}")
        
        # Extract teams using pre-compiled patterns
        for compiled_pattern in self.compiled_team_patterns:
            matches = compiled_pattern.finditer(query)
            for match in matches:
                team_name = match.group(1)
                self.logger.info(f"   Pattern match for team: '{team_name}'")
                # Check if we already have this team
                if not any(e.name.lower() == team_name.lower() for e in entities):
                    entities.append(SoccerEntity(
                        name=team_name,
                        entity_type=EntityType.TEAM,
                        confidence=0.9
                    ))
                    self.logger.info(f"    Added pattern-based team: {team_name} (confidence: 0.9)")
                else:
                    self.logger.info(f"    Skipped duplicate team: {team_name}")
        
        # Filter out common false positives and derby names
        self.logger.info("3. False positive filtering")
        original_count = len(entities)
        entities = [e for e in entities if not self._is_false_positive(e.name)]
        filtered_count = len(entities)
        if original_count != filtered_count:
            self.logger.info(f"   Filtered out {original_count - filtered_count} false positives")
        
        # Additional deduplication: remove overlapping team names
        self.logger.info("4. Overlapping entity deduplication")
        deduplicated_entities = []
        for entity in entities:
            is_duplicate = False
            for existing in deduplicated_entities:
                if (entity.entity_type == existing.entity_type and 
                    self._is_overlapping_entity(entity.name, existing.name)):
                    self.logger.info(f"    Removed overlapping entity: '{entity.name}' (overlaps with '{existing.name}')")
                    is_duplicate = True
                    break
            if not is_duplicate:
                deduplicated_entities.append(entity)
        
        # Add derby teams if derby is mentioned but teams not explicitly found
        self.logger.info("5. Derby team addition")
        derby_teams_added = self._add_derby_teams(query, deduplicated_entities, added_keys)
        if derby_teams_added:
            self.logger.info(f"   Added {derby_teams_added} derby teams")
        
        return deduplicated_entities
    
    def _extract_time_context(self, query: str) -> TimeContext:
        """Determine the time context of the query using pre-compiled patterns."""
        for time_context, compiled_pattern in self.compiled_time_patterns.items():
            if compiled_pattern.search(query):
                return time_context
        
        # Default to current season if no time context found
        return TimeContext.THIS_SEASON
    
    def _extract_comparison_type(self, query: str) -> Optional[ComparisonType]:
        """Extract comparison type if present using pre-compiled patterns."""
        # Special case for career average (keep this as-is since it's rarely used)
        if re.search(r'\b(?:career|overall)\s+average\b', query, re.IGNORECASE):
            return ComparisonType.VS_CAREER
            
        for comp_type, compiled_pattern in self.compiled_comparison_patterns.items():
            if compiled_pattern.search(query):
                return comp_type
                
        # Check for implicit comparisons using pre-compiled pattern
        if self._compiled_common_patterns['comparison_keywords'].search(query):
            return ComparisonType.VS_OPPONENT
            
        return None
    
    def _extract_statistics(self, query: str) -> List[str]:
        """Extract all statistics being requested from the query."""
        statistics = []
        
        # First, try to detect multiple statistics mentioned explicitly
        # Look for patterns like "goals, assists, and yellow cards" or "goals and assists"
        multiple_stats_pattern = r'\b(\w+(?:\s+\w+)*?)(?:,|\s+and\s+|\s+&\s+)(\w+(?:\s+\w+)*?)(?:(?:,|\s+and\s+|\s+&\s+)(\w+(?:\s+\w+)*?))*\b'
        
        # Also look for specific conjunctive patterns (support multi-word stats like "yellow cards")
        # Be more precise with word boundaries to avoid matching too much
        conjunctive_patterns = [
            r'\b(goals?|assists?|yellow\s+cards?|red\s+cards?|clean\s+sheets?|tackles?|saves?|minutes?),\s*(goals?|assists?|yellow\s+cards?|red\s+cards?|clean\s+sheets?|tackles?|saves?|minutes?),?\s*and\s*(goals?|assists?|yellow\s+cards?|red\s+cards?|clean\s+sheets?|tackles?|saves?|minutes?)\b',
            r'\b(goals?|assists?|yellow\s+cards?|red\s+cards?|clean\s+sheets?|tackles?|saves?|minutes?)\s+and\s+(goals?|assists?|yellow\s+cards?|red\s+cards?|clean\s+sheets?|tackles?|saves?|minutes?)\b',
            r'\b(goals?|assists?|yellow\s+cards?|red\s+cards?|clean\s+sheets?|tackles?|saves?|minutes?),\s*(goals?|assists?|yellow\s+cards?|red\s+cards?|clean\s+sheets?|tackles?|saves?|minutes?)\b',
        ]
        
        query_lower = query.lower()
        found_multi_pattern = False
        
        # Check for conjunctive patterns first
        best_match_stats = []
        best_match_count = 0
        
        for i, pattern in enumerate(conjunctive_patterns, 1):
            self.logger.info(f"Testing conjunctive pattern {i}: {pattern}")
            matches = re.finditer(pattern, query_lower)
            for match in matches:
                potential_stats = [g for g in match.groups() if g]
                self.logger.info(f"   Found match groups: {potential_stats}")
                matched_stats = []
                for potential_stat in potential_stats:
                    # Check if this potential stat matches any known stat pattern
                    for stat_name, stat_pattern in self.stat_patterns.items():
                        if re.search(stat_pattern, potential_stat, re.IGNORECASE):
                            if stat_name not in matched_stats:
                                matched_stats.append(stat_name)
                                self.logger.info(f"   Matched '{potential_stat}' to stat '{stat_name}'")
                                break
                
                if len(matched_stats) >= 2:
                    # Keep the best match (longest list of statistics)
                    if len(matched_stats) > best_match_count:
                        best_match_stats = matched_stats.copy()
                        best_match_count = len(matched_stats)
                        self.logger.info(f"New best match: {matched_stats} (count: {len(matched_stats)})")
                    elif len(matched_stats) == best_match_count and matched_stats != best_match_stats:
                        self.logger.info(f"Equal match found: {matched_stats}, keeping first one")
        
        # Use the best match found
        if best_match_stats:
            found_multi_pattern = True
            statistics.extend(best_match_stats)
            self.logger.info(f"Final multiple statistics from conjunctive pattern: {best_match_stats}")
            self.logger.info(f"Current statistics list: {statistics}")
        
        # If we didn't find a multi-pattern, fall back to single statistic detection
        if not found_multi_pattern:
            single_stat = self._extract_single_statistic(query)
            if single_stat:
                statistics.append(single_stat)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_statistics = []
        for stat in statistics:
            if stat not in seen:
                seen.add(stat)
                unique_statistics.append(stat)
        
        return unique_statistics
    
    def _extract_single_statistic(self, query: str) -> Optional[str]:
        """Extract a single statistic being requested (backward compatibility)."""
        # First check for ranking keywords that might indicate what stat we're looking for
        ranking_metrics = self.ranking_keywords.get("ranking_metrics", {})
        ranking_directions = self.ranking_keywords.get("ranking_direction", {})
        
        # Check for ranking patterns first
        for stat_name, keywords in ranking_metrics.items():
            for keyword in keywords:
                # Check if this metric keyword appears with any ranking direction
                for direction, direction_keywords in ranking_directions.items():
                    for direction_keyword in direction_keywords:
                        pattern = rf'\b{re.escape(direction_keyword)}\s+{re.escape(keyword)}\b'
                        if re.search(pattern, query, re.IGNORECASE):
                            return stat_name
                        
                        # Also check for "keyword" + "direction" pattern
                        pattern = rf'\b{re.escape(keyword)}\s+{re.escape(direction_keyword)}\b'
                        if re.search(pattern, query, re.IGNORECASE):
                            return stat_name
        
        # Check for specific ranking question patterns
        ranking_patterns = self.ranking_keywords.get("ranking_patterns", {})
        ranking_questions = ranking_patterns.get("ranking_question", [])
        
        for question_pattern in ranking_questions:
            if re.search(rf'\b{re.escape(question_pattern)}\b', query, re.IGNORECASE):
                # Try to match the question with specific metrics
                for stat_name, keywords in ranking_metrics.items():
                    for keyword in keywords:
                        if keyword.lower() in query.lower():
                            return stat_name
        
        # Fast keyword-based pre-filtering before regex matching
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        # Check if any keywords from our lookup appear in the query
        potential_stats = set()
        for word in query_words:
            if word in self._stat_keyword_lookup:
                potential_stats.update(self._stat_keyword_lookup[word])
        
        # If we have potential matches, only check those patterns
        if potential_stats:
            for stat_name in potential_stats:
                if self.compiled_stat_patterns[stat_name].search(query):
                    return stat_name
        else:
            # Fallback: check all patterns (shouldn't happen often)
            for stat_name, compiled_pattern in self.compiled_stat_patterns.items():
                if compiled_pattern.search(query):
                    return stat_name
        
        return None
    
    def _extract_statistic(self, query: str) -> Optional[str]:
        """Extract the main statistic being requested (backward compatibility)."""
        return self._extract_single_statistic(query)
    
    def _extract_filters(self, query: str) -> Dict[str, Any]:
        """Extract additional filters like home/away, competition type."""
        filters = {}
        
        self.logger.info("--- Filter Extraction Phase ---")
        
        # Home/Away detection
        venue = self._detect_venue(query)
        if venue:
            filters['venue'] = venue
            self.logger.info(f"    Detected: {venue.upper()} venue")
            
        # Big Six detection using pre-compiled pattern
        if self._compiled_common_patterns['big_six'].search(query):
            filters['opponent_tier'] = 'top_6'
            self.logger.info("    Detected: Big Six opponent tier")
            
        # Derby detection using pre-compiled pattern
        if self._compiled_common_patterns['derby'].search(query):
            filters['match_type'] = 'derby'
            self.logger.info("    Detected: Derby match type")
        
        # Enhanced derby detection using knowledge base
        derby_info = self._detect_derby_from_entities(query)
        if derby_info:
            filters['derby_info'] = derby_info
            self.logger.info(f"    Detected derby: {derby_info['name']} ({derby_info['teams']})")
        
        # Ranking query detection
        ranking_info = self._detect_ranking_query(query)
        if ranking_info:
            filters['ranking'] = ranking_info
            self.logger.info(f"    Detected ranking query: {ranking_info}")
        
        # Competition detection
        competition = self._detect_competition(query)
        if competition:
            filters['competition'] = competition
            self.logger.info(f"    Detected competition: {competition}")
        
        # Position detection
        position = self._detect_position(query)
        if position:
            filters['position'] = position
            self.logger.info(f"    Detected position: {position}")
        
        # Tactical context detection
        tactical_context = self._extract_tactical_context(query)
        if tactical_context:
            filters['tactical_context'] = tactical_context
            self.logger.info(f"    Detected tactical context: {tactical_context}")

        # Historical context detection
        historical_context = self._extract_historical_context(query)
        if historical_context:
            filters['historical_context'] = historical_context
            self.logger.info(f"    Detected historical context: {historical_context}")

        return filters
    
    def _determine_intent(self, query: str, entities: List[SoccerEntity],
                         comparison_type: Optional[ComparisonType]) -> str:
        """Determine the overall intent of the query using pre-compiled patterns."""
        # First check for context queries using pre-compiled pattern
        if self._compiled_common_patterns['context_keywords'].search(query):
            return "context"

        # Enhanced historical query detection
        if self._compiled_common_patterns['historical_keywords'].search(query):
            return "historical"

        # Check for milestone queries
        if self._compiled_common_patterns['milestone_keywords'].search(query):
            return "historical"

        # Check for trend analysis queries
        if self._compiled_common_patterns['trend_keywords'].search(query):
            return "historical"

        # Check for career queries
        if self._compiled_common_patterns['career_keywords'].search(query):
            return "historical"

        # Then check for comparison queries
        if comparison_type or self._compiled_common_patterns['comparison_keywords'].search(query):
            # But don't count "against" alone as comparison
            if not (re.search(r'\bagainst\b', query, re.IGNORECASE) and
                   not self._compiled_common_patterns['vs_keywords'].search(query)):
                return "comparison"

        # Default to stat lookup
        return "stat_lookup"
    
    def _is_likely_player(self, name: str) -> bool:
        """Determine if a name is likely a player."""
        if not name:
            return False
        name = name.strip()
        
        # Check if it's a known player
        if self._normalize_text(name) in {self._normalize_text(x) for x in self.known_players}:
            return True
            
        # Check if it's a known team (to avoid misclassification)
        if self._normalize_text(name) in {self._normalize_text(x) for x in self.known_teams}:
            return False
            
        # Basic name validation
        return (len(name.split()) <= 3 and 
                all(part[0].isupper() for part in name.split()) and
                not self._is_false_positive(name))
    
    def _is_false_positive(self, name: str) -> bool:
        """Check if a name is likely a false positive."""
        false_positives = self.special_cases.get("false_positives", {})
        
        # Check common words
        common_words = false_positives.get("common_words", [
            "what", "how", "when", "where", "who", "why",
            "show", "tell", "give", "find", "get", "let"
        ])
        if name.lower() in common_words:
            return True
            
        # Check derby names
        derby_names = false_positives.get("derby_names", [])
        if self._normalize_text(name) in [self._normalize_text(d) for d in derby_names]:
            return True
            
        return False
    
    def _calculate_confidence(self, entities: List[SoccerEntity], 
                            time_context: TimeContext, statistic: Optional[str]) -> float:
        """Calculate overall confidence in the query parsing."""
        base_confidence = 0.5
        
        self.logger.info("--- Confidence Calculation ---")
        self.logger.info(f"   Base confidence: {base_confidence}")
        
        if entities:
            base_confidence += 0.3
            self.logger.info(f"   +0.3 for entities found (total: {base_confidence})")
        if time_context != TimeContext.THIS_SEASON:  # Explicit time context found
            base_confidence += 0.1
            self.logger.info(f"   +0.1 for explicit time context (total: {base_confidence})")
        if statistic:
            base_confidence += 0.1
            self.logger.info(f"   +0.1 for statistic found (total: {base_confidence})")
            
        return min(base_confidence, 1.0)

    # ----------------------------
    # Helper methods (loading/regex)
    # ----------------------------

    def _load_dict_if_exists(self, path: Path, default: Dict[str, List[str]]) -> Dict[str, List[str]]:
        try:
            if path.exists():
                self.logger.info(f"Loading external dictionary: {path}")
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Ensure values are lists of strings
                    normalized: Dict[str, List[str]] = {}
                    for canonical, aliases in data.items():
                        alias_list = [a for a in aliases if isinstance(a, str)]
                        # Include canonical itself as an alias to guarantee recognition
                        if isinstance(canonical, str):
                            alias_list = list({canonical, *alias_list})
                            normalized[canonical] = alias_list
                    return normalized or default
            else:
                self.logger.info(f"External dictionary not found: {path}, using defaults")
        except Exception:
            # Fall back silently to defaults if malformed
            self.logger.warning(f"Failed to load external dictionary: {path}, using defaults")
            pass
        return default

    def _compile_alias_regex(self, aliases: List[str]) -> re.Pattern:
        # Normalize and sort by length to prefer longer phrases first
        unique_aliases = sorted({self._escape_alias(a) for a in aliases if a}, key=len, reverse=True)
        if not unique_aliases:
            # Fallback to a regex that never matches
            return re.compile(r"a^")
        pattern = r"\b(?:" + "|".join(unique_aliases) + r")\b"
        self.logger.debug(f"Compiled regex pattern: {pattern}")
        return re.compile(pattern, re.IGNORECASE)

    def _escape_alias(self, alias: str) -> str:
        # Escape regex special chars but keep spaces; allow dots/apostrophes literally
        return re.escape(alias).replace("\\ ", " ")

    def _normalize_text(self, text: str) -> str:
        no_accents = unicodedata.normalize("NFKD", text)
        no_accents = "".join([c for c in no_accents if not unicodedata.combining(c)])
        return no_accents.lower().strip()

    def _title_or_preserve(self, surface: str) -> str:
        """Keep one-word exact case (e.g., City) else Title-case multi-words."""
        # Check if this term should preserve its case from special cases
        case_preservation = self.special_cases.get("normalization_rules", {}).get("case_preservation", [])
        if surface.upper() in case_preservation:
            return surface
        
        if len(surface.split()) == 1:
            # Capitalize first letter but preserve all-caps like PSG
            if surface.isupper():
                return surface
            return surface[0].upper() + surface[1:]
        return surface.title()

    def _load_stat_patterns(self, path: Path, default: Dict[str, str]) -> Dict[str, str]:
        try:
            if path.exists():
                self.logger.info(f"Loading statistics patterns: {path}")
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                compiled: Dict[str, str] = {}
                for key, synonyms in data.items():
                    if not isinstance(synonyms, list) or not synonyms:
                        continue
                    escaped = [self._escape_alias(s) for s in synonyms if isinstance(s, str)]
                    if not escaped:
                        continue
                    compiled[key] = r"\b(?:" + "|".join(escaped) + r")\b"
                return compiled or default
            else:
                self.logger.info(f"Statistics patterns not found: {path}, using defaults")
        except Exception:
            self.logger.warning(f"Failed to load statistics patterns: {path}, using defaults")
            pass
        return default

    def _load_derby_knowledge(self, data_dir: Path) -> Dict[str, Dict[str, Any]]:
        """Load derby and rivalry knowledge from data file."""
        default_derbies = {
            "north_london_derby": {
                "teams": ["arsenal", "tottenham"],
                "names": ["North London Derby"],
                "league": "Premier League",
                "locality": "London"
            },
            "el_clasico": {
                "teams": ["real madrid", "barcelona"],
                "names": ["El Clásico", "El Clasico", "The Classic"],
                "league": "La Liga",
                "locality": "Spain"
            },
            "manchester_derby": {
                "teams": ["manchester united", "manchester city"],
                "names": ["Manchester Derby"],
                "league": "Premier League",
                "locality": "Manchester"
            },
            "merseyside_derby": {
                "teams": ["liverpool", "everton"],
                "names": ["Merseyside Derby"],
                "league": "Premier League",
                "locality": "Liverpool"
            }
        }
        
        try:
            derby_path = data_dir / "derbies.json"
            if derby_path.exists():
                self.logger.info(f"Loading derby knowledge: {derby_path}")
                with open(derby_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data
            else:
                self.logger.info(f"Derby knowledge not found: {derby_path}, using defaults")
        except Exception:
            self.logger.warning(f"Failed to load derby knowledge: {derby_path}, using defaults")
        
        return default_derbies

    def _load_tactical_patterns(self, data_dir: Path) -> Dict[str, List[str]]:
        """Load tactical context patterns from data file."""
        default_patterns = {
            "formations": ["4-3-3", "4-4-2", "3-5-2", "4-2-3-1", "3-4-3"],
            "styles": ["pressing", "counterattack", "possession", "defensive", "attacking"],
            "situations": ["early goal", "late goal", "red card", "yellow card", "penalty", "var"],
            "timing": ["first half", "second half", "extra time", "injury time"]
        }
        
        try:
            tactical_path = data_dir / "tactical.json"
            if tactical_path.exists():
                self.logger.info(f"Loading tactical patterns: {tactical_path}")
                with open(tactical_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data
            else:
                self.logger.info(f"Tactical patterns not found: {tactical_path}, using defaults")
        except Exception:
            self.logger.warning(f"Failed to load tactical patterns: {tactical_path}, using defaults")
        
        return default_patterns

    def _load_special_cases(self, data_dir: Path) -> Dict[str, Dict[str, Any]]:
        """Load special cases configuration from data file."""
        default_special_cases = {
            "el_clasico_override": {
                "name": "El Clásico",
                "teams": ["real madrid", "barcelona"],
                "league": "La Liga",
                "locality": "Spain"
            },
            "north_london_derby_override": {
                "name": "North London Derby",
                "teams": ["arsenal", "tottenham"],
                "league": "Premier League",
                "locality": "London"
            },
            "manchester_derby_override": {
                "name": "Manchester Derby",
                "teams": ["manchester united", "manchester city"],
                "league": "Premier League",
                "locality": "Manchester"
            },
            "merseyside_derby_override": {
                "name": "Merseyside Derby",
                "teams": ["liverpool", "everton"],
                "league": "Premier League",
                "locality": "Liverpool"
            },
            "false_positives": {
                "common_words": [
                    "what", "how", "when", "where", "who", "why",
                    "show", "tell", "give", "find", "get", "let"
                ]
            },
            "entity_overlaps": {
                "team_overlaps": [
                    ["arsenal", "tottenham"],
                    ["liverpool", "everton"],
                    ["manchester city", "city"],
                    ["manchester united", "united"]
                ]
            },
            "derby_mappings": {
                "el_clasico": {
                    "name": "El Clásico",
                    "teams": ["real madrid", "barcelona"],
                    "league": "La Liga",
                    "locality": "Spain",
                    "trigger_terms": ["el clasico", "clasico"]
                },
                "north_london_derby": {
                    "name": "North London Derby",
                    "teams": ["arsenal", "tottenham"],
                    "league": "Premier League",
                    "locality": "London",
                    "trigger_terms": ["north london derby", "north_london_derby"]
                },
                "manchester_derby": {
                    "name": "Manchester Derby",
                    "teams": ["manchester united", "manchester city"],
                    "league": "Premier League",
                    "locality": "Manchester",
                    "trigger_terms": ["manchester derby", "manchester_derby"]
                },
                "merseyside_derby": {
                    "name": "Merseyside Derby",
                    "teams": ["liverpool", "everton"],
                    "league": "Premier League",
                    "locality": "Liverpool",
                    "trigger_terms": ["merseyside derby", "merseyside_derby"]
                }
            }
        }
        
        try:
            special_cases_path = data_dir / "special_cases.json"
            if special_cases_path.exists():
                self.logger.info(f"Loading special cases: {special_cases_path}")
                with open(special_cases_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data
            else:
                self.logger.info(f"Special cases not found: {special_cases_path}, using defaults")
        except Exception:
            self.logger.warning(f"Failed to load special cases: {special_cases_path}, using defaults")
        
        return default_special_cases

    def _load_ranking_keywords(self, data_dir: Path) -> Dict[str, Dict[str, Any]]:
        """Load ranking keywords configuration from data file."""
        default_ranking_keywords = {
            "ranking_direction": {
                "highest": ["most", "highest", "best", "top", "greatest"],
                "lowest": ["least", "lowest", "worst", "bottom", "minimum"]
            },
            "ranking_metrics": {
                "goals": ["goals", "scored", "scoring", "goalscorer"],
                "assists": ["assists", "assisted", "assisting", "assister"],
                "goal_contributions": ["g/a", "goals and assists", "goal contributions"],
                "clean_sheets": ["clean sheets", "clean sheet", "shutouts"],
                "hat_tricks": ["hat tricks", "hat trick", "hat-tricks"],
                "chances_created": ["chances created", "chance creation", "key passes"],
                "take_ons": ["take ons", "take on", "dribbles", "dribbling"],
                "xg_overperformance": ["xg overperformance", "xG overperformance"],
                "through_balls": ["through balls", "through ball", "through-balls"],
                "goals_per_game": ["goals per game", "gpg", "goals/game"],
                "assists_per_90": ["assists per 90", "assists per 90 minutes", "assists/90"]
            }
        }
        
        try:
            ranking_keywords_path = data_dir / "ranking_keywords.json"
            if ranking_keywords_path.exists():
                self.logger.info(f"Loading ranking keywords: {ranking_keywords_path}")
                with open(ranking_keywords_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data
            else:
                self.logger.info(f"Ranking keywords not found: {ranking_keywords_path}, using defaults")
        except Exception:
            self.logger.warning(f"Failed to load ranking keywords: {ranking_keywords_path}, using defaults")
        
        return default_ranking_keywords

    def _detect_derby_from_entities(self, query: str) -> Optional[Dict[str, Any]]:
        """Detect derby matches from team entities in the query."""
        # Extract team names from query
        team_names = []
        for match in re.finditer(self.team_alias_regex, query):
            team_names.append(self._normalize_text(match.group(0)))
        
        # Check for special case derby mappings from data file
        derby_mappings = self.special_cases.get("derby_mappings", {})
        for derby_key, derby_info in derby_mappings.items():
            # Check if any trigger terms are in the query
            trigger_terms = derby_info.get("trigger_terms", [])
            for term in trigger_terms:
                if term.lower() in query.lower():
                    return {
                        "key": derby_key,
                        "name": derby_info["name"],
                        "teams": derby_info["teams"],
                        "league": derby_info.get("league"),
                        "locality": derby_info.get("locality")
                    }
        
        if len(team_names) < 2:
            return None
        
        # Check if any team pair matches a known derby
        for derby_key, derby_info in self.derby_knowledge.items():
            derby_teams = set(derby_info["teams"])
            query_teams = set(team_names)
            
            if derby_teams.issubset(query_teams):
                return {
                    "key": derby_key,
                    "name": derby_info["names"][0] if derby_info["names"] else derby_key,
                    "teams": derby_info["teams"],
                    "league": derby_info.get("league"),
                    "locality": derby_info.get("locality")
                }
        
        return None

    def _extract_tactical_context(self, query: str) -> Dict[str, Any]:
        """Extract tactical context from the query."""
        context = {}
        
        # Check for formations
        for formation in self.tactical_patterns.get("formations", []):
            if re.search(rf"\b{re.escape(formation)}\b", query, re.IGNORECASE):
                context["formation"] = formation
                break
        
        # Check for playing styles
        detected_styles = []
        for style in self.tactical_patterns.get("styles", []):
            if re.search(rf"\b{re.escape(style)}\b", query, re.IGNORECASE):
                detected_styles.append(style)
        if detected_styles:
            context["style"] = detected_styles
        
        # Check for match situations
        detected_situations = []
        for situation in self.tactical_patterns.get("situations", []):
            if re.search(rf"\b{re.escape(situation)}\b", query, re.IGNORECASE):
                detected_situations.append(situation)
        if detected_situations:
            context["situations"] = detected_situations
        
        # Check for timing context
        for timing in self.tactical_patterns.get("timing", []):
            if re.search(rf"\b{re.escape(timing)}\b", query, re.IGNORECASE):
                context["timing"] = timing
                break
        
        return context

    def _extract_historical_context(self, query: str) -> Dict[str, Any]:
        """Extract historical context information from the query."""
        historical_context = {}

        # Check for milestone-related queries
        if self._compiled_common_patterns['milestone_keywords'].search(query):
            historical_context['type'] = 'milestone'
            historical_context['keywords'] = self._extract_milestone_keywords(query)

        # Check for trend-related queries
        elif self._compiled_common_patterns['trend_keywords'].search(query):
            historical_context['type'] = 'trend'
            historical_context['direction'] = self._extract_trend_direction(query)

        # Check for career-related queries
        elif self._compiled_common_patterns['career_keywords'].search(query):
            historical_context['type'] = 'career'
            historical_context['scope'] = 'entire_career'

        # Check for specific historical periods
        historical_periods = self._extract_historical_periods(query)
        if historical_periods:
            historical_context['periods'] = historical_periods

        # Check for record-related queries
        if re.search(r'\b(?:record|best|worst|highest|lowest)\b', query, re.IGNORECASE):
            historical_context['record_type'] = self._extract_record_type(query)

        # Check for comparative historical context
        if re.search(r'\b(?:compare.*history|historical.*comparison|career.*vs)\b', query, re.IGNORECASE):
            historical_context['comparison'] = True

        return historical_context

    def _extract_milestone_keywords(self, query: str) -> List[str]:
        """Extract milestone-related keywords from query."""
        milestone_keywords = []

        milestone_terms = ['milestone', 'achievement', 'record', 'breakthrough', 'first time',
                          'debut', 'anniversary', 'legacy', 'historic', 'unprecedented']

        for term in milestone_terms:
            if re.search(rf'\b{re.escape(term)}\b', query, re.IGNORECASE):
                milestone_keywords.append(term)

        return milestone_keywords

    def _extract_trend_direction(self, query: str) -> Optional[str]:
        """Extract trend direction from query."""
        if re.search(r'\b(?:improvement|improving|better|increase|rising)\b', query, re.IGNORECASE):
            return 'improving'
        elif re.search(r'\b(?:decline|declining|worse|decrease|falling)\b', query, re.IGNORECASE):
            return 'declining'
        elif re.search(r'\b(?:progression|development|evolution)\b', query, re.IGNORECASE):
            return 'developing'
        else:
            return 'general'

    def _extract_historical_periods(self, query: str) -> List[str]:
        """Extract specific historical periods mentioned in query."""
        periods = []

        # Season patterns
        season_matches = re.findall(r'\b(?:20\d{2}[-/]?\d{2})\b', query)
        periods.extend(season_matches)

        # Era patterns
        if re.search(r'\b(?:early career|prime|peak|late career)\b', query, re.IGNORECASE):
            era_match = re.search(r'\b(early career|prime|peak|late career)\b', query, re.IGNORECASE)
            if era_match:
                periods.append(era_match.group(1))

        # Decade patterns
        decade_matches = re.findall(r'\b(?:19|20)\d0s\b', query)
        periods.extend(decade_matches)

        return periods

    def _extract_record_type(self, query: str) -> str:
        """Extract the type of record being queried."""
        if re.search(r'\b(?:best|highest|most)\b', query, re.IGNORECASE):
            return 'best'
        elif re.search(r'\b(?:worst|lowest|least)\b', query, re.IGNORECASE):
            return 'worst'
        elif re.search(r'\b(?:record)\b', query, re.IGNORECASE):
            return 'record'
        else:
            return 'general'

    def _detect_venue(self, query: str) -> Optional[str]:
        """Intelligently detect venue (home/away) from query using pre-compiled patterns."""
        # Use pre-compiled patterns for faster detection
        if self._compiled_common_patterns['away_venue'].search(query):
            return 'away'
        
        if self._compiled_common_patterns['home_venue'].search(query):
            return 'home'
        
        return None

    def _add_derby_teams(self, query: str, entities: List[SoccerEntity], added_keys: set) -> int:
        """Add derby teams as entities if derby is mentioned but teams not explicitly found."""
        derby_teams_added = 0
        
        # Check for derby mappings from special cases
        derby_mappings = self.special_cases.get("derby_mappings", {})
        for derby_key, derby_info in derby_mappings.items():
            # Check if any trigger terms are in the query
            trigger_terms = derby_info.get("trigger_terms", [])
            for term in trigger_terms:
                if term.lower() in query.lower():
                    # Check if derby teams are already present as entities
                    derby_teams = derby_info.get("teams", [])
                    existing_team_names = {e.name.lower() for e in entities if e.entity_type == EntityType.TEAM}
                    
                    # Only add derby teams if no teams are already present
                    if not existing_team_names:
                        # For queries like "Early goal in El Clasico", we should only add one team
                        # to represent the derby context, not both teams
                        if len(derby_teams) > 0:
                            # Add only the first team as a representative
                            team_name = derby_teams[0]
                            entities.append(SoccerEntity(
                                name=team_name.title(),
                                entity_type=EntityType.TEAM,
                                confidence=0.8  # Lower confidence since it's inferred
                            ))
                            derby_teams_added += 1
                            self.logger.info(f"    Added derby team: {team_name.title()} (from {derby_info['name']})")
                    else:
                        # Check if any existing teams are part of this derby
                        for team_name in derby_teams:
                            team_already_present = False
                            for existing_team in existing_team_names:
                                if team_name.lower() in existing_team or existing_team in team_name.lower():
                                    team_already_present = True
                                    break
                            
                            if not team_already_present:
                                # Add the team as an entity
                                entities.append(SoccerEntity(
                                    name=team_name.title(),
                                    entity_type=EntityType.TEAM,
                                    confidence=0.8  # Lower confidence since it's inferred
                                ))
                                derby_teams_added += 1
                                self.logger.info(f"    Added derby team: {team_name.title()} (from {derby_info['name']})")
        
        return derby_teams_added

    def _is_overlapping_entity(self, name1: str, name2: str) -> bool:
        """Check if two entity names overlap in a way that suggests they are the same entity."""
        name1_lower = name1.lower()
        name2_lower = name2.lower()
        
        # Case 1: Exact match
        if name1_lower == name2_lower:
            return True
        
        # Case 2: Check against configured overlaps from special cases
        overlaps = self.special_cases.get("entity_overlaps", {}).get("team_overlaps", [])
        for overlap_pair in overlaps:
            if name1_lower in overlap_pair and name2_lower in overlap_pair:
                return True
        
        # Case 3: Check if they're from the same canonical team (most important)
        for canonical, aliases in self.team_alias_dict.items():
            if name1_lower in aliases and name2_lower in aliases:
                return True
        
        # Case 4: One is substring of the other (e.g., "City" in "Man City")
        if name1_lower in name2_lower or name2_lower in name1_lower:
            # But be careful: "United" should not match "Manchester United" if they're different teams
            # Only allow this if they're from the same canonical team
            for canonical, aliases in self.team_alias_dict.items():
                if name1_lower in aliases and name2_lower in aliases:
                    return True
        
        # Case 5: Special handling for "Man City" vs "Manchester City" and similar cases
        # Check if both names are aliases of the same canonical team
        canonical1 = None
        canonical2 = None
        
        for canonical, aliases in self.team_alias_dict.items():
            if name1_lower in aliases:
                canonical1 = canonical
            if name2_lower in aliases:
                canonical2 = canonical
        
        if canonical1 and canonical2 and canonical1 == canonical2:
            return True
        
        return False

    def _detect_ranking_query(self, query: str) -> Optional[Dict[str, Any]]:
        """Detect if this is a ranking query and extract ranking information."""
        ranking_directions = self.ranking_keywords.get("ranking_direction", {})
        
        # Check for highest ranking keywords
        highest_keywords = ranking_directions.get("highest", [])
        for keyword in highest_keywords:
            if re.search(rf'\b{re.escape(keyword)}\b', query, re.IGNORECASE):
                return {
                    'type': 'ranking',
                    'direction': 'highest',
                    'keyword': keyword
                }
        
        # Check for lowest ranking keywords
        lowest_keywords = ranking_directions.get("lowest", [])
        for keyword in lowest_keywords:
            if re.search(rf'\b{re.escape(keyword)}\b', query, re.IGNORECASE):
                return {
                    'type': 'ranking',
                    'direction': 'lowest',
                    'keyword': keyword
                }
        
        # Check for ranking question patterns
        ranking_patterns = self.ranking_keywords.get("ranking_patterns", {})
        ranking_questions = ranking_patterns.get("ranking_question", [])
        
        for question_pattern in ranking_questions:
            if re.search(rf'\b{re.escape(question_pattern)}\b', query, re.IGNORECASE):
                return {
                    'type': 'ranking',
                    'direction': 'highest',  # Most ranking questions are about highest
                    'keyword': question_pattern
                }
        
        return None

    def _detect_competition(self, query: str) -> Optional[str]:
        """Detect competition from the query."""
        ranking_competitions = self.ranking_keywords.get("ranking_competitions", {})
        
        for comp_name, keywords in ranking_competitions.items():
            for keyword in keywords:
                if re.search(rf'\b{re.escape(keyword)}\b', query, re.IGNORECASE):
                    return comp_name
        return None

    def _detect_position(self, query: str) -> Optional[str]:
        """Detect player position from the query."""
        ranking_positions = self.ranking_keywords.get("ranking_positions", {})
        
        for pos_name, keywords in ranking_positions.items():
            for keyword in keywords:
                if re.search(rf'\b{re.escape(keyword)}\b', query, re.IGNORECASE):
                    return pos_name
        return None

# Example usage and testing
if __name__ == "__main__":
    parser = SoccerQueryParser()
    
    test_queries = [
        "How many goals has Haaland scored this season?",
        "What's Arsenal's home record in the Premier League?",
        "How does Messi's pass completion compare to his career average?",
        "When did Barcelona last beat Real Madrid in El Clasico?",
        "What's Liverpool's clean sheet record against the big six?",
        "How significant is Salah's performance against City?"
    ]
    
    for query in test_queries:
        parsed = parser.parse_query(query)
        print(f"\nQuery: {query}")
        print(f"Intent: {parsed.query_intent}")
        print(f"Entities: {[(e.name, e.entity_type.value) for e in parsed.entities]}")
        print(f"Statistic: {parsed.statistic_requested}")
        print(f"Time Context: {parsed.time_context.value}")
        print(f"Comparison: {parsed.comparison_type.value if parsed.comparison_type else None}")
        print(f"Filters: {parsed.filters}")
        print(f"Confidence: {parsed.confidence:.2f}")