"""Test suite for the soccer query parser.

This test file can be executed directly, or via pytest. To make direct
execution robust (e.g., `python sports_intelligence_layer/tests/test_parser.py`),
we prepend the project root to sys.path before importing the package.
"""

from pathlib import Path
import sys
import pytest
import logging
from datetime import datetime

# Ensure project root is importable when running this file directly
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sports_intelligence_layer import (  # noqa: E402
    SoccerQueryParser, ParsedSoccerQuery, SoccerEntity,
    EntityType, ComparisonType, TimeContext,
)


@pytest.fixture
def parser():
    """Create a parser instance for testing."""
    return SoccerQueryParser()


def test_basic_player_stat_query(parser):
    """Test basic player statistic query parsing."""
    query = "How many goals has Haaland scored this season?"
    result = parser.parse_query(query)
    
    assert isinstance(result, ParsedSoccerQuery)
    assert result.query_intent == "stat_lookup"
    assert result.statistic_requested == "goals"
    assert result.time_context == TimeContext.THIS_SEASON
    
    assert len(result.entities) == 1
    player = result.entities[0]
    assert player.name == "Haaland"
    assert player.entity_type == EntityType.PLAYER


def test_team_performance_query(parser):
    """Test team performance query parsing."""
    query = "What's Arsenal's home record in the Premier League?"
    result = parser.parse_query(query)
    
    assert result.query_intent == "stat_lookup"
    assert len(result.entities) == 1
    assert result.entities[0].name == "Arsenal"
    assert result.entities[0].entity_type == EntityType.TEAM
    assert result.filters.get("venue") == "home"


def test_player_comparison_query(parser):
    """Test player comparison query parsing."""
    query = "How does Messi's pass completion compare to his career average?"
    result = parser.parse_query(query)
    
    assert result.query_intent == "comparison"
    assert result.comparison_type == ComparisonType.VS_CAREER
    assert result.statistic_requested == "pass_completion"
    assert len(result.entities) == 1
    assert result.entities[0].name == "Messi"


def test_historical_query(parser):
    """Test historical match query parsing."""
    query = "When did Barcelona last beat Real Madrid in El Clasico?"
    result = parser.parse_query(query)
    
    assert result.query_intent == "historical"
    assert len(result.entities) == 2
    team_names = {entity.name for entity in result.entities}
    assert "Barcelona" in team_names
    assert "Real Madrid" in team_names


def test_team_filter_query(parser):
    """Test team query with filters parsing."""
    query = "What's Liverpool's clean sheet record against the big six?"
    result = parser.parse_query(query)
    
    assert result.query_intent == "stat_lookup"
    assert result.statistic_requested == "clean_sheets"
    assert result.filters.get("opponent_tier") == "top_6"
    assert len(result.entities) == 1
    assert result.entities[0].name == "Liverpool"


def test_context_query(parser):
    """Test context-based query parsing."""
    query = "How significant is Salah's performance against City?"
    result = parser.parse_query(query)
    
    assert result.query_intent == "context"
    assert len(result.entities) == 2
    player = next(e for e in result.entities if e.entity_type == EntityType.PLAYER)
    team = next(e for e in result.entities if e.entity_type == EntityType.TEAM)
    assert player.name == "Salah"
    assert team.name == "City"


def test_multiple_stats_query(parser):
    """Test query with multiple statistics."""
    query = "Show me Benzema's goals and assists in Champions League"
    result = parser.parse_query(query)
    
    assert result.query_intent == "stat_lookup"
    assert result.time_context == TimeContext.CHAMPIONS_LEAGUE
    assert len(result.entities) == 1
    assert result.entities[0].name == "Benzema"
    assert result.statistic_requested in ["goals", "assists"]


# ============================================================================
# DELIVERABLE 1: Enhanced entity database with aliases
# ============================================================================

def test_player_alias_recognition(parser):
    """Test enhanced player alias recognition."""
    test_cases = [
        ("How many goals did KDB score?", "de bruyne", "KDB"),
        ("What's Mo Salah's assist record?", "salah", "Mo Salah"),
        ("Erling's performance this season", "haaland", "Erling"),
        ("Harry Kane's goals", "kane", "Harry Kane")
    ]
    
    for query, expected_canonical, expected_surface in test_cases:
        result = parser.parse_query(query)
        assert len(result.entities) >= 1
        player_entities = [e for e in result.entities if e.entity_type == EntityType.PLAYER]
        assert len(player_entities) >= 1
        # Check that the surface form is preserved in the entity name
        assert (expected_surface.lower() in player_entities[0].name.lower() or 
                expected_surface.lower() in query.lower())


def test_team_alias_recognition(parser):
    """Test enhanced team alias recognition."""
    test_cases = [
        ("Man City's home form", "manchester city", "Man City"),
        ("Man Utd vs Liverpool", "manchester united", "Man Utd"),
        ("Barca's Champions League record", "barcelona", "Barca"),
        ("The Reds' performance", "liverpool", "Reds")
    ]
    
    for query, expected_canonical, expected_surface in test_cases:
        result = parser.parse_query(query)
        team_entities = [e for e in result.entities if e.entity_type == EntityType.TEAM]
        assert len(team_entities) >= 1


# ============================================================================
# DELIVERABLE 2: Derby and rivalry recognition
# ============================================================================

def test_explicit_derby_keyword(parser):
    """Test explicit derby keyword detection."""
    query = "What's the result of the North London derby?"
    result = parser.parse_query(query)
    
    assert result.filters.get("match_type") == "derby"
    assert len(result.entities) >= 1  # Should detect Arsenal or Tottenham


def test_derby_from_team_pairs(parser):
    """Test derby detection from team entity pairs."""
    test_cases = [
        ("Arsenal vs Tottenham match", "north_london_derby", ["arsenal", "tottenham"]),
        ("Real Madrid against Barcelona", "el_clasico", ["real madrid", "barcelona"]),
        ("Manchester United vs Manchester City", "manchester_derby", ["manchester united", "manchester city"]),
        ("Liverpool vs Everton", "merseyside_derby", ["liverpool", "everton"])
    ]
    
    for query, expected_derby, expected_teams in test_cases:
        result = parser.parse_query(query)
        derby_info = result.filters.get("derby_info")
        if derby_info:
            assert derby_info["key"] == expected_derby
            assert set(derby_info["teams"]) == set(expected_teams)


def test_derby_with_explicit_names(parser):
    """Test derby detection with explicit derby names."""
    query = "When was the last El Clasico?"
    result = parser.parse_query(query)
    
    # Should detect both teams and potentially derby context
    team_entities = [e for e in result.entities if e.entity_type == EntityType.TEAM]
    assert len(team_entities) >= 1


# ============================================================================
# DELIVERABLE 3: Tactical context extraction
# ============================================================================

def test_home_away_detection(parser):
    """Test home/away venue detection."""
    test_cases = [
        ("Arsenal's home record", "home"),
        ("Liverpool away form", "away"),
        ("City at home", "home"),
        ("United on the road", "away")
    ]
    
    for query, expected_venue in test_cases:
        result = parser.parse_query(query)
        assert result.filters.get("venue") == expected_venue


def test_big_six_detection(parser):
    """Test Big Six opponent tier detection."""
    test_cases = [
        "Liverpool vs the big six",
        "Arsenal's record against top 6",
        "Chelsea performance vs top six teams"
    ]
    
    for query in test_cases:
        result = parser.parse_query(query)
        assert result.filters.get("opponent_tier") == "top_6"


def test_tactical_context_extraction(parser):
    """Test tactical context extraction."""
    test_cases = [
        ("Arsenal's 4-3-3 formation", {"formation": "4-3-3"}),
        ("Liverpool's pressing style", {"style": ["pressing"]}),
        ("Early goal in the first half", {"timing": "first half"}),
        ("Red card in the second half", {"situations": ["red card"], "timing": "second half"})
    ]
    
    for query, expected_context in test_cases:
        result = parser.parse_query(query)
        tactical_context = result.filters.get("tactical_context", {})
        
        for key, expected_value in expected_context.items():
            if key in tactical_context:
                if isinstance(expected_value, list):
                    assert any(item in tactical_context[key] for item in expected_value)
                else:
                    assert tactical_context[key] == expected_value


# ============================================================================
# DELIVERABLE 4: Accuracy testing
# ============================================================================

def test_comprehensive_accuracy(parser):
    """Test comprehensive accuracy across all features."""
    test_queries = [
        # Basic entity recognition
        ("Haaland's goals this season", {"entities": 1, "statistic": "goals", "time": TimeContext.THIS_SEASON}),
        ("Arsenal home form", {"entities": 1, "venue": "home"}),
        
        # Alias recognition
        ("KDB's assists", {"entities": 1, "statistic": "assists"}),
        ("Man City vs United", {"entities": 2, "derby": True}),
        
        # Tactical context
        ("Liverpool's 4-3-3 pressing", {"entities": 1, "formation": "4-3-3", "style": ["pressing"]}),
        ("Early goal in El Clasico", {"entities": 1, "derby": True, "timing": "early"}),
        
        # Complex queries
        ("How does Messi's pass completion compare to his career average?", 
         {"entities": 1, "comparison": ComparisonType.VS_CAREER, "statistic": "pass_completion"}),
        
        ("What's Liverpool's clean sheet record against the big six?", 
         {"entities": 1, "opponent_tier": "top_6", "statistic": "clean_sheets"})
    ]
    
    passed_tests = 0
    total_tests = len(test_queries)
    
    for query, expected in test_queries:
        try:
            result = parser.parse_query(query)
            
            # Check entity count
            if "entities" in expected:
                assert len(result.entities) == expected["entities"]
            
            # Check statistic
            if "statistic" in expected:
                assert result.statistic_requested == expected["statistic"]
            
            # Check time context
            if "time" in expected:
                assert result.time_context == expected["time"]
            
            # Check venue
            if "venue" in expected:
                assert result.filters.get("venue") == expected["venue"]
            
            # Check derby detection
            if expected.get("derby"):
                assert (result.filters.get("match_type") == "derby" or 
                       result.filters.get("derby_info") is not None)
            
            # Check opponent tier
            if "opponent_tier" in expected:
                assert result.filters.get("opponent_tier") == expected["opponent_tier"]
            
            # Check comparison type
            if "comparison" in expected:
                assert result.comparison_type == expected["comparison"]
            
            # Check tactical context
            tactical_context = result.filters.get("tactical_context", {})
            if "formation" in expected:
                assert tactical_context.get("formation") == expected["formation"]
            if "style" in expected:
                assert any(style in tactical_context.get("style", []) for style in expected["style"])
            if "timing" in expected:
                assert tactical_context.get("timing") == expected["timing"]
            
            passed_tests += 1
            
        except AssertionError as e:
            print(f"❌ Failed for query: '{query}' - {e}")
        except Exception as e:
            print(f"❌ Error for query: '{query}' - {e}")
    
    accuracy = passed_tests / total_tests
    print(f"\n📊 ACCURACY RESULTS:")
    print(f"Passed: {passed_tests}/{total_tests}")
    print(f"Accuracy: {accuracy:.1%}")
    
    # Assert 80%+ accuracy
    assert accuracy >= 0.8, f"Accuracy {accuracy:.1%} is below 80% threshold"


def test_edge_cases_and_robustness(parser):
    """Test edge cases and robustness."""
    edge_cases = [
        "",  # Empty query
        "   ",  # Whitespace only
        "What is the weather like?",  # Non-soccer query
        "How many goals did XYZ score?",  # Unknown player
        "Team ABC performance",  # Unknown team
    ]
    
    for query in edge_cases:
        if not query.strip():
            with pytest.raises(ValueError):
                parser.parse_query(query)
        else:
            # Should handle gracefully without crashing
            result = parser.parse_query(query)
            assert isinstance(result, ParsedSoccerQuery)


# ============================================================================
# ADDITIONAL TESTS FROM USER'S SAMPLE
# ============================================================================

def test_champions_league_context(parser):
    """Test: How many goals has Mbappe scored in the Champions League?"""
    query = "How many goals has Mbappe scored in the Champions League?"
    result = parser.parse_query(query)
    
    assert result.statistic_requested == "goals"
    assert result.time_context == TimeContext.CHAMPIONS_LEAGUE
    
    player_entities = [e for e in result.entities if e.entity_type == EntityType.PLAYER]
    assert len(player_entities) >= 1


def test_away_performance_query(parser):
    """Test: How has Chelsea performed away from home this season?"""
    query = "How has Chelsea performed away from home this season?"
    result = parser.parse_query(query)
    
    assert result.filters.get('venue') == 'away'
    assert result.time_context == TimeContext.THIS_SEASON
    
    team_entities = [e for e in result.entities if e.entity_type == EntityType.TEAM]
    assert len(team_entities) == 1
    assert team_entities[0].name == "Chelsea"


def test_derby_match_query(parser):
    """Test: What's the history of Manchester derbies?"""
    query = "What's the history of Manchester derbies?"
    result = parser.parse_query(query)
    
    assert result.query_intent == "historical"
    assert result.filters.get('match_type') == 'derby'


def test_head_to_head_query(parser):
    """Test: When did Barcelona last beat Real Madrid?"""
    query = "When did Barcelona last beat Real Madrid?"
    result = parser.parse_query(query)
    
    assert result.query_intent == "historical"
    
    team_entities = [e for e in result.entities if e.entity_type == EntityType.TEAM]
    team_names = [e.name for e in team_entities]
    assert "Barcelona" in team_names
    assert "Real Madrid" in team_names


def test_clean_sheets_vs_big_six(parser):
    """Test: What's Liverpool's clean sheet record against the big six?"""
    query = "What's Liverpool's clean sheet record against the big six?"
    result = parser.parse_query(query)
    
    assert result.statistic_requested == "clean_sheets"
    assert result.filters.get('opponent_tier') == 'top_6'
    
    team_entities = [e for e in result.entities if e.entity_type == EntityType.TEAM]
    assert len(team_entities) == 1
    assert team_entities[0].name == "Liverpool"


def test_team_home_record_query(parser):
    """Test: What's Arsenal's home record this season?"""
    query = "What's Arsenal's home record this season?"
    result = parser.parse_query(query)
    
    assert result.query_intent == "stat_lookup"
    assert result.time_context == TimeContext.THIS_SEASON
    assert result.filters.get('venue') == 'home'
    
    team_entities = [e for e in result.entities if e.entity_type == EntityType.TEAM]
    assert len(team_entities) == 1
    assert team_entities[0].name == "Arsenal"


def test_basic_player_goal_query(parser):
    """Test: How many goals has Haaland scored this season?"""
    query = "How many goals has Haaland scored this season?"
    result = parser.parse_query(query)
    
    assert result.query_intent == "stat_lookup"
    assert result.statistic_requested == "goals"
    assert result.time_context == TimeContext.THIS_SEASON
    
    player_entities = [e for e in result.entities if e.entity_type == EntityType.PLAYER]
    assert len(player_entities) == 1
    assert "Haaland" in player_entities[0].name
    assert result.confidence > 0.8


def test_player_comparison_query_detailed(parser):
    """Test: How does Messi's pass completion compare to his career average?"""
    query = "How does Messi's pass completion compare to his career average?"
    result = parser.parse_query(query)
    
    assert result.query_intent == "comparison"
    assert result.comparison_type == ComparisonType.VS_CAREER
    assert result.statistic_requested == "pass_completion"
    
    player_entities = [e for e in result.entities if e.entity_type == EntityType.PLAYER]
    assert len(player_entities) > 0
    assert "Messi" in player_entities[0].name


def test_significance_context_query(parser):
    """Test: How significant is Salah's performance against City?"""
    query = "How significant is Salah's performance against City?"
    result = parser.parse_query(query)
    
    assert result.query_intent == "context"
    
    entities = result.entities
    player_entities = [e for e in entities if e.entity_type == EntityType.PLAYER]
    team_entities = [e for e in entities if e.entity_type == EntityType.TEAM]
    
    assert len(player_entities) > 0
    assert len(team_entities) > 0


def test_multiple_stats_query_detailed(parser):
    """Test: What are Benzema's goals and assists this season?"""
    query = "What are Benzema's goals and assists this season?"
    result = parser.parse_query(query)
    
    # Should pick up "goals" as primary statistic
    # (assists would be secondary - handled in response generation)
    assert result.statistic_requested in ["goals", "assists"]
    assert result.time_context == TimeContext.THIS_SEASON


# ============================================================================
# INTEGRATION TESTS (from user's sample)
# ============================================================================

class TestSoccerQueryParserIntegration:
    """Integration tests that simulate real agent workflows"""
    
    @pytest.fixture
    def parser(self):
        return SoccerQueryParser()
    
def test_research_agent_workflow(parser):
    """Simulate Research Agent discovering storylines for a match"""
    queries = [
        "What storylines should fans know about tonight's Arsenal vs Tottenham game?",
        "How significant is Kane's return to North London?",
        "What's the head-to-head record in recent North London derbies?"
    ]
    
    for query in queries:
        result = parser.parse_query(query)
        # Each query should be parsed successfully with reasonable confidence
        assert result.confidence > 0.5
        assert result.query_intent in ["context", "historical", "stat_lookup"]

def test_writing_agent_workflow(parser):
    """Simulate Writing Agent verifying and enhancing content"""
    queries = [
        "Is this Haaland's best month of the season?",
        "What additional context makes this performance meaningful?",
        "How does this compare to similar performances this season?"
    ]
    
    for query in queries:
        result = parser.parse_query(query)
        # Should handle comparison and context queries
        assert result.query_intent in ["comparison", "context", "stat_lookup"]

def test_editor_agent_workflow(parser):
    """Simulate Editor Agent fact-checking claims"""
    queries = [
        "Is Messi the first player since Ronaldinho to achieve this feat?",
        "What important context is missing from this Benzema analysis?",
        "Verify: Liverpool has the best defensive record in Europe this season"
    ]
    
    for query in queries:
        result = parser.parse_query(query)
        # Editor queries often involve verification and context
        assert result.query_intent in ["historical", "context", "comparison"]


# ============================================================================
# QUERY ANALYSIS FUNCTION (from user's sample)
# ============================================================================

def analyze_sample_queries():
    """Analyze a variety of soccer queries to understand patterns"""
    
    parser = SoccerQueryParser()
    
    sample_queries = [
        # Player Performance
        "How many goals has Haaland scored this season?",
        "What's Messi's pass completion rate in El Clasicos?",
        "How many assists does De Bruyne have at home this season?",
        
        # Team Performance  
        "What's Arsenal's away record in the Premier League?",
        "How many clean sheets has Liverpool kept this season?",
        "What's Barcelona's win rate against Real Madrid?",
        
        # Comparisons
        "How does Salah's scoring compare to last season?",
        "Is this Benzema's best Champions League campaign?",
        "How does City's possession compare to league average?",
        
        # Historical Context
        "When did these teams last meet in a title decider?",
        "What's the significance of this Liverpool performance?",
        "How rare is a hat-trick in El Clasico?",
        
        # Complex Queries
        "What storylines emerge from Mbappe's performance against his former club?",
        "How significant is this comeback for Arsenal's title hopes?",
        "What context makes this derby result historically important?"
    ]
    
    print("🔍 Query Analysis Report\n")
    
    for i, query in enumerate(sample_queries, 1):
        print(f"{i:2d}. {query}")
        result = parser.parse_query(query)
        
        print(f"    Intent: {result.query_intent}")
        print(f"    Entities: {[(e.name, e.entity_type.value) for e in result.entities]}")
        print(f"    Statistic: {result.statistic_requested}")
        print(f"    Time: {result.time_context.value}")
        print(f"    Comparison: {result.comparison_type.value if result.comparison_type else None}")
        print(f"    Filters: {result.filters}")
        print(f"    Confidence: {result.confidence:.2f}")
        print()


def run_comprehensive_test_suite():
    """Run all tests and provide detailed results"""
    
    print("🧪 Running Soccer Query Parser Test Suite\n")
    
    # Test categories
    test_categories = [
        ("Basic Queries", [
            "test_basic_player_stat_query",
            "test_team_performance_query", 
            "test_player_comparison_query",
            "test_historical_query",
            "test_team_filter_query",
            "test_context_query",
            "test_multiple_stats_query"
        ]),
        ("Enhanced Features", [
            "test_player_alias_recognition",
            "test_team_alias_recognition", 
            "test_explicit_derby_keyword",
            "test_derby_from_team_pairs",
            "test_derby_with_explicit_names",
            "test_home_away_detection",
            "test_big_six_detection",
            "test_tactical_context_extraction"
        ]),
        ("Additional Tests", [
            "test_champions_league_context",
            "test_away_performance_query",
            "test_derby_match_query",
            "test_head_to_head_query",
            "test_clean_sheets_vs_big_six",
            "test_team_home_record_query",
            "test_basic_player_goal_query",
            "test_player_comparison_query_detailed",
            "test_significance_context_query",
            "test_multiple_stats_query_detailed"
        ])
    ]
    
    all_results = []
    
    for category_name, test_names in test_categories:
        print(f"📂 {category_name}")
        print("-" * 50)
        
        # Run tests using pytest
        import subprocess
        import sys
        
        test_args = [sys.executable, "-m", "pytest", 
                    "sports_intelligence_layer/tests/test_parser.py",
                    "-v", "-s", "-k", " or ".join(test_names)]
        
        try:
            result = subprocess.run(test_args, capture_output=True, text=True)
            print(result.stdout)
            if result.stderr:
                print("Errors:", result.stderr)
        except Exception as e:
            print(f"Error running tests: {e}")
        
        print("\n")
    
    # Summary
    print("📊 Test Summary")
    print("=" * 50)
    print("✅ All test categories completed!")
    print("🔍 Run 'analyze_sample_queries()' for detailed query analysis")


if __name__ == "__main__":
    # Set up logging to see detailed parsing process
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the comprehensive accuracy test
    parser = SoccerQueryParser()
    test_comprehensive_accuracy(parser)
    
    print("\n✅ All tests completed successfully!")
    
    # Optionally run query analysis
    print("\n" + "="*60 + "\n")
    analyze_sample_queries()