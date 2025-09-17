"""
Query Template Validator

A utility script to validate and test query patterns against the template.
Helps ensure queries follow the recommended patterns and structure.
"""

import json
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class QueryValidationResult:
    """Result of query validation."""
    is_valid: bool
    category: Optional[str]
    pattern: Optional[str]
    confidence: float
    suggestions: List[str]
    detected_entities: List[str]
    detected_statistics: List[str]
    issues: List[str]


class QueryTemplateValidator:
    """Validates queries against the template patterns."""

    def __init__(self, template_path: str = "data/QUERY_PATTERNS_TEMPLATE.json"):
        """Initialize with template file."""
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                self.template = json.load(f)
        except FileNotFoundError:
            print(f"Template file {template_path} not found. Using minimal template.")
            self.template = self._create_minimal_template()

        self.categories = self.template.get("query_categories", {})
        self.best_practices = self.template.get("best_practices", {})

        # Compile regex patterns for entity detection
        self._compile_patterns()

    def _create_minimal_template(self) -> Dict[str, Any]:
        """Create a minimal template if file is not found."""
        return {
            "query_categories": {
                "1_direct_data_access": {
                    "patterns": {
                        "player_basic_stats": {
                            "statistics": ["goals", "assists", "rating"],
                            "examples": ["Messi goals", "Ronaldo assists"]
                        }
                    }
                }
            },
            "best_practices": {
                "statistic_specification": {
                    "supported_stats": ["goals", "assists", "rating", "appearances"]
                }
            }
        }

    def _compile_patterns(self):
        """Compile regex patterns for entity and statistic detection."""
        # Common player names pattern
        self.player_pattern = re.compile(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b')

        # Common team names pattern
        self.team_pattern = re.compile(r'\b(?:Manchester|Real|Barcelona|Arsenal|Liverpool|Chelsea|Bayern|PSG|City)\b', re.IGNORECASE)

        # Statistics pattern
        self.stats_pattern = re.compile(r'\b(?:goals?|assists?|rating|appearances?|minutes?|shots?|passes?|tackles?|cards?)\b', re.IGNORECASE)

        # Time context pattern
        self.time_pattern = re.compile(r'\b(?:this season|last season|career|last \d+ games?|at home|away)\b', re.IGNORECASE)

    def validate_query(self, query: str) -> QueryValidationResult:
        """Validate a query against the template patterns."""
        query = query.strip()

        # Detect entities and statistics
        detected_entities = self._detect_entities(query)
        detected_statistics = self._detect_statistics(query)

        # Determine query category
        category, pattern, confidence = self._classify_query(query, detected_entities, detected_statistics)

        # Check for issues
        issues = self._check_issues(query, detected_entities, detected_statistics)

        # Generate suggestions
        suggestions = self._generate_suggestions(query, category, issues)

        # Determine if valid
        is_valid = len(issues) == 0 and confidence > 0.6

        return QueryValidationResult(
            is_valid=is_valid,
            category=category,
            pattern=pattern,
            confidence=confidence,
            suggestions=suggestions,
            detected_entities=detected_entities,
            detected_statistics=detected_statistics,
            issues=issues
        )

    def _detect_entities(self, query: str) -> List[str]:
        """Detect player and team entities in the query."""
        entities = []

        # Detect potential player names
        player_matches = self.player_pattern.findall(query)
        for match in player_matches:
            if len(match.split()) <= 3:  # Reasonable name length
                entities.append(f"player:{match}")

        # Detect team names
        team_matches = self.team_pattern.findall(query)
        for match in team_matches:
            entities.append(f"team:{match}")

        return entities

    def _detect_statistics(self, query: str) -> List[str]:
        """Detect statistics mentioned in the query."""
        stats = []

        stats_matches = self.stats_pattern.findall(query)
        for match in stats_matches:
            stats.append(match.lower())

        return stats

    def _classify_query(self, query: str, entities: List[str], statistics: List[str]) -> tuple:
        """Classify the query into a category."""
        query_lower = query.lower()

        # Direct data access patterns
        if any(stat in query_lower for stat in ["goals", "assists", "rating"]) and entities:
            if len(entities) == 1:
                return "1_direct_data_access", "player_basic_stats", 0.9

        # Ranking patterns
        if any(word in query_lower for word in ["most", "best", "top", "highest"]):
            return "3_ranking_and_sorting", "top_performers", 0.85

        # Comparison patterns
        if "vs" in query_lower or "versus" in query_lower or len(entities) >= 2:
            return "4_comparison_queries", "player_vs_player", 0.8

        # Historical patterns
        if any(word in query_lower for word in ["career", "history", "when", "milestone"]):
            return "5_historical_queries", "career_milestones", 0.8

        # Context patterns
        if any(word in query_lower for word in ["why", "significance", "important", "context"]):
            return "6_contextual_queries", "significance_questions", 0.75

        # Performance analysis
        if any(word in query_lower for word in ["performance", "form", "analysis"]):
            return "2_statistical_analysis", "performance_overview", 0.7

        return "unknown", "unclassified", 0.3

    def _check_issues(self, query: str, entities: List[str], statistics: List[str]) -> List[str]:
        """Check for common issues in the query."""
        issues = []

        # Check for entities
        if not entities:
            issues.append("No player or team entities detected")

        # Check for statistics
        if not statistics and not any(word in query.lower() for word in ["why", "when", "context"]):
            issues.append("No specific statistics mentioned")

        # Check for ambiguous entities
        if any(":" in entity for entity in entities):
            player_count = sum(1 for e in entities if e.startswith("player:"))
            if player_count > 1:
                player_names = [e.split(":")[1] for e in entities if e.startswith("player:")]
                if any(name.lower() in ["messi", "ronaldo"] for name in player_names):
                    issues.append("Consider using full names to avoid ambiguity")

        # Check query length
        if len(query.split()) < 3:
            issues.append("Query might be too short for accurate processing")

        # Check for supported statistics
        supported_stats = self.best_practices.get("statistic_specification", {}).get("supported_stats", [])
        for stat in statistics:
            if stat not in supported_stats:
                issues.append(f"Statistic '{stat}' might not be fully supported")

        return issues

    def _generate_suggestions(self, query: str, category: str, issues: List[str]) -> List[str]:
        """Generate suggestions to improve the query."""
        suggestions = []

        # Suggestions based on issues
        for issue in issues:
            if "No player or team" in issue:
                suggestions.append("Add specific player or team names (e.g., 'Lionel Messi', 'Manchester City')")
            elif "No specific statistics" in issue:
                suggestions.append("Specify what statistic you want (goals, assists, rating, etc.)")
            elif "too short" in issue:
                suggestions.append("Add more context like time period ('this season', 'career')")
            elif "ambiguity" in issue:
                suggestions.append("Use full names instead of common names")

        # Suggestions based on category
        if category == "1_direct_data_access":
            suggestions.append("Consider adding time context: 'this season', 'last 10 games'")
        elif category == "3_ranking_and_sorting":
            suggestions.append("Specify the scope: 'in Manchester City', 'in Premier League'")
        elif category == "4_comparison_queries":
            suggestions.append("Use clear comparison format: 'Player A vs Player B statistics'")
        elif category == "5_historical_queries":
            suggestions.append("Be specific about time period or milestone type")

        return suggestions

    def get_example_queries(self, category: str) -> List[str]:
        """Get example queries for a specific category."""
        examples = []

        category_data = self.categories.get(category, {})
        patterns = category_data.get("patterns", {})

        for pattern_name, pattern_data in patterns.items():
            pattern_examples = pattern_data.get("examples", [])
            examples.extend(pattern_examples[:3])  # Limit to 3 examples per pattern

        return examples

    def validate_batch(self, queries: List[str]) -> Dict[str, QueryValidationResult]:
        """Validate multiple queries."""
        results = {}
        for i, query in enumerate(queries):
            results[f"query_{i+1}"] = self.validate_query(query)
        return results

    def print_validation_report(self, result: QueryValidationResult, query: str):
        """Print a formatted validation report."""
        print(f"\n{'='*60}")
        print(f"QUERY VALIDATION REPORT")
        print(f"{'='*60}")
        print(f"Query: '{query}'")
        print(f"Valid: {'Yes' if result.is_valid else 'No'}")
        print(f"Category: {result.category}")
        print(f"Pattern: {result.pattern}")
        print(f"Confidence: {result.confidence:.2f}")

        if result.detected_entities:
            print(f"\nDetected Entities:")
            for entity in result.detected_entities:
                print(f"  - {entity}")

        if result.detected_statistics:
            print(f"\nDetected Statistics:")
            for stat in result.detected_statistics:
                print(f"  - {stat}")

        if result.issues:
            print(f"\nIssues:")
            for issue in result.issues:
                print(f"  - {issue}")

        if result.suggestions:
            print(f"\nSuggestions:")
            for suggestion in result.suggestions:
                print(f"  - {suggestion}")

        print(f"{'='*60}")


def main():
    """Main function to demonstrate the validator."""
    validator = QueryTemplateValidator()

    # Test queries
    test_queries = [
        "How many goals does Messi have?",
        "Messi goals",
        "Who has the most goals in Arsenal?",
        "Messi vs Ronaldo",
        "Why is El Clasico important?",
        "Haaland performance this season",
        "goals",  # Bad query
        "Lionel Messi career milestones in Barcelona"
    ]

    print("Query Template Validator Demo")
    print("=" * 50)

    for query in test_queries:
        result = validator.validate_query(query)
        validator.print_validation_report(result, query)

        if not result.is_valid:
            print(f"\nExample queries for category '{result.category}':")
            examples = validator.get_example_queries(result.category)
            for example in examples[:3]:
                print(f"  - {example}")


if __name__ == "__main__":
    main()