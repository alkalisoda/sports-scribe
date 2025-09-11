# Updated Pipeline with NarrativePlanner and StylizedWriter

## Overview

The SportsScribe pipeline has been updated to include a new narrative planning
step and stylized writing capability, following the flowchart:

```text
DataCollector → ResearchAgent → NarrativePlanner
                    ↓               ↓
              WriterAgent → StylizedWriter → Editor → Final Article
```

## New Pipeline Flow

### 1. Data Collection

- **DataCollector**: Gathers raw game data from sports APIs
- Extracts compact game data format (match_info, events, players, statistics, lineups)

### 2. Research

- **ResearchAgent**: Analyzes game data and provides contextual insights
- Generates game analysis, player performance, and historical context

### 3. Narrative Planning

- **NarrativePlanner**: Analyzes data and research to select compelling
  narrative angles
- Outputs narrative selection with primary narrative, supporting narratives,
  character arcs, storytelling focus, and social hooks

### 4. Article Generation (Two Paths)

- **WriterAgent**: Generates factual article based on research insights
- **StylizedWriter**: Transforms factual article using narrative plan to
  create emotionally engaging content

### 5. Editing

- **Editor**: Reviews and refines the stylized article for quality and accuracy

## Key Components

### NarrativePlanner

- **Purpose**: Selects compelling narrative angles for sports articles
- **Input**: CompactGameData + ResearchInsights
- **Output**: NarrativeSelection (primary_narrative, supporting_narratives,
  character_arcs, storytelling_focus, social_hooks)

### StylizedWriter

- **Purpose**: Transforms factual articles into emotionally engaging narratives
- **Input**: Factual article + NarrativeSelection
- **Output**: Stylized article with narrative elements

## Updated Pipeline Output

The pipeline now returns enhanced metadata:

```json
{
    "success": true,
    "game_id": "1208021",
    "article_type": "game_recap",
    "content": "Final edited article content",
    "narrative_metadata": {
        "primary_narrative": "Dramatic comeback victory",
        "storytelling_focus": "drama",
        "supporting_narratives": ["Key player performance", "Tactical masterclass"],
        "character_arcs": [
            {
                "character": "Player Name",
                "arc": "Rising from bench to hero",
                "significance": "Game-changing impact"
            }
        ],
        "social_hooks": ["Incredible comeback!", "Heroic performance"]
    },
    "article_versions": {
        "factual_article": "Original factual content",
        "stylized_article": "Narrative-enhanced content",
        "final_article": "Edited final content"
    },
    "editing_metadata": {
        "original_length": 450,
        "edited_length": 480,
        "length_change": 30,
        "has_changes": true,
        "validation_passed": true
    }
}
```

## Usage

### Running the Updated Pipeline

```python
from scriber_agents.pipeline import ArticlePipeline

# Initialize pipeline
pipeline = ArticlePipeline()

# Generate article with narrative planning
result = await pipeline.generate_game_recap("1208021")

# Access different versions
factual_article = result["article_versions"]["factual_article"]
stylized_article = result["article_versions"]["stylized_article"]
final_article = result["content"]

# Access narrative metadata
narrative = result["narrative_metadata"]["primary_narrative"]
storytelling_focus = result["narrative_metadata"]["storytelling_focus"]
```

### Testing

Run the updated pipeline test:

```bash
cd sports-scribe/ai-backend
python test_updated_pipeline.py
```

## Benefits

1. **Enhanced Storytelling**: Articles now have compelling narrative structures
2. **Emotional Engagement**: Stylized writing creates deeper reader connections
3. **Social Media Optimization**: Built-in social hooks for better sharing
4. **Character Development**: Player and team storylines add human interest
5. **Flexible Output**: Access to both factual and stylized versions

## Configuration

The pipeline uses the same configuration for all agents:

```python
config = {
    "model": "gpt-4o",
    "temperature": 0.7,
    "max_tokens": 2000
}
```

## Error Handling

- If NarrativePlanner fails, the pipeline falls back to factual article only
- If StylizedWriter fails, the pipeline returns the factual article
- Comprehensive error logging and metadata tracking
- Graceful degradation at each step

## Future Enhancements

1. **A/B Testing**: Compare factual vs. stylized article performance
2. **Audience Targeting**: Tailor narratives for specific audience segments
3. **Multi-language Support**: Generate narratives in different languages
4. **Performance Metrics**: Track narrative effectiveness over time
