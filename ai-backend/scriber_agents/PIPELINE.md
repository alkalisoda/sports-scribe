# SportsScribe Agent Pipeline Documentation

## Overview

The SportsScribe system uses a streamlined multi-agent pipeline to generate
high-quality sports articles:

```text
Data Collector → Researcher → Writer
```

Each agent has specific responsibilities and passes structured data to the next
agent in the pipeline. The pipeline uses a shared OpenAI client for all AI
operations and helper methods for clean separation of concerns.

## Standardized API Response Structure

All API calls return a standardized structure:

```json
{
  "get": "endpoint_name",
  "parameters": {"param1": "value1"},
  "errors": [],
  "results": 1,
  "paging": {"current": 1, "total": 1},
  "response": [...]
}
```

## Agent Pipeline Flow

### 1. Data Collector Agent (`data_collector.py`)

**Purpose**: Gathers raw sports data from API-Football via RapidAPI

**Key Functions**:

- `collect_game_data(game_id: str) → Dict[str, Any]`
- `collect_team_data(team_id: str) → Dict[str, Any]`
- `collect_player_data(player_id: str) → Dict[str, Any]`
- `collect_league_data(league_id: str, season: str) → Dict[str, Any]`

**Output Data Structure**:

```python
{
    "get": "game_data",
    "parameters": {"game_id": "123"},
    "errors": [],
    "results": 1,
    "paging": {},
    "response": [
        {
            ...
        }
    ]
}
```

### 2. Research Agent (`researcher.py`)

**Purpose**: Analyzes data and generates storylines for articles

**Key Functions**:

- `research_team_history(team_id: str, opponent_id: str) → Dict[str, Any]`
- `research_player_performance(player_id: str, context: Dict[str, Any])
  → Dict[str, Any]`
- `research_season_trends(league: str, season: str) → Dict[str, Any]`
- `analyze_game_data(game_data: Dict[str, Any]) → Dict[str, Any]`
- `generate_storylines(data_list: List[Dict[str, Any]]) → List[str]`

**Input**: Standardized API responses from Data Collector
**Output**: Storylines list and contextual analysis

**Storylines Example**:

```python
[
    "Manchester United secures victory over Liverpool",
    "High-scoring thriller with 5+ goals",
    "Dramatic finish with late goal",
    "Outstanding individual performances"
]
```

### 3. Writer Agent (`writer.py`)

**Purpose**: Generates engaging articles using AI and storylines

**Key Functions**:

- `generate_article(game_info: Dict[str, Any], team_info: Dict[str, Any],
  player_info: Dict[str, Any], research: Dict[str, Any]) → str`

**Input**: Game info + Team info + Player info + Research data
**Output**: Article content (string)

## Pipeline Architecture

### Main Pipeline Class Structure

```python
class ArticlePipeline:
    def __init__(self, config):
        # Initialize shared OpenAI client
        self.openai_client = AsyncOpenAI(api_key=config["openai_api_key"])

        # Initialize all agents with shared client
        self.collector = DataCollectorAgent(config, openai_client=self.openai_client)
        self.researcher = ResearchAgent(config, openai_client=self.openai_client)
        self.writer = WriterAgent(config)

    # Main generation methods
    async def generate_game_recap(self, game_id: str) -> Dict[str, Any]
    async def generate_preview_article(self, game_id: str) -> Dict[str, Any]
    async def generate_player_spotlight(self, player_id: str,
                                        game_id: Optional[str] = None) ->
                                        Dict[str, Any]

    # Helper methods for data collection
    async def _collect_game_data(self, game_id: str) -> Dict[str, Any]
    async def _collect_team_data(self, game_data: Dict[str, Any]) -> Dict[str, Any]
    async def _collect_player_data(self, player_id: str) -> Dict[str, Any]

    # Helper methods for research
    async def _research_game_context(self, game_data: Dict[str, Any],
                                     team_data: Dict[str, Any] = None) ->
                                     Dict[str, Any]
    async def _research_player_performance(self, player_id: str,
                                           game_id: Optional[str] = None) ->
                                           Dict[str, Any]

    # Helper methods for storyline generation
    async def _generate_storylines(self, data_list: List[Dict[str, Any]]) ->
                                   List[str]

    # Helper methods for result formatting
    def _format_result(self, content: str, metadata: Dict[str, Any]) ->
                        Dict[str, Any]
```

## Updated Pipeline Integration

### Main Pipeline Function

```python
async def generate_game_recap(game_id: str) -> Dict[str, Any]:
    """
    Main pipeline function that orchestrates all agents.

    Pipeline: Data Collection → Research → Storyline Generation → Content Writing
    """
    # Step 1: Data Collection
    game_data = await self._collect_game_data(game_id)
    team_data = await self._collect_team_data(game_data)

    # Step 2: Research & Context
    research_data = await self._research_game_context(game_data, team_data)

    # Step 3: Storyline Generation
    storylines = await self._generate_storylines([
        game_data, team_data["home_team"], team_data["away_team"]
    ])

    # Step 4: Content Generation
    article_content = await self.writer.generate_article(
        game_data, team_data, player_data, research_data
    )

    # Step 5: Return Results
    return self._format_result(content=article_content, metadata={...})
```

## Data Flow Summary

1. **Data Collector** → Standardized API responses (fixtures, teams, players)
2. **Researcher** → Storylines list + Contextual analysis
3. **Writer** → AI-generated article content using storylines

## Function Call Dependencies

```text
generate_game_recap()
├── _collect_game_data()
├── _collect_team_data()
├── _research_game_context()
├── _generate_storylines()
├── writer.generate_article()
└── _format_result()
```

## Helper Methods Breakdown

### Data Collection Helpers

- `_collect_game_data()`: Collects and validates game data
- `_collect_team_data()`: Extracts team IDs and collects team data
- `_collect_player_data()`: Collects and validates player data

### Research Helpers

- `_research_game_context()`: Researches team history and season trends
- `_research_player_performance()`: Researches player performance data

### Storyline Helpers

- `_generate_storylines()`: Generates prioritized storylines from collected data

### Result Formatting

- `_format_result()`: Combines content and metadata with pipeline version

## Storyline Generation Process

1. **Data Analysis**: Researcher analyzes raw API data
2. **Context Extraction**: Identifies key events, statistics, and trends
3. **Storyline Creation**: Generates compelling narrative hooks
4. **Prioritization**: Selects top 10 most relevant storylines
5. **Integration**: Passes storylines directly to Writer for article generation

## API Integration Details

### API-Football Endpoints Used

- `/fixtures` - Game details and scores
- `/fixtures/events` - Match events (goals, cards, etc.)
- `/fixtures/lineups` - Team formations and players
- `/fixtures/statistics` - Match statistics
- `/teams` - Team information
- `/teams/statistics` - Team performance data
- `/players` - Player information and stats
- `/standings` - League standings
- `/players/topscorers` - Top scorers

### Error Handling

- API failures return standardized error structure
- Missing data scenarios handled gracefully
- Fallback content generation when AI services unavailable

## Configuration Requirements

Each agent requires configuration for:

- RapidAPI key for API-Football access
- OpenAI API key for content generation
- Model parameters (temperature, max_tokens)
- Style guidelines and quality thresholds

## Key Improvements in New Structure

1. **Shared OpenAI Client**: All agents use the same client instance for
   efficiency
2. **Helper Methods**: Cleaner separation of concerns and better maintainability
3. **Standardized Data Flow**: Consistent input/output formats across all agents
4. **Storyline Integration**: Direct storylines input to writer for better
   content focus
5. **Error Handling**: Centralized validation and error management
6. **Modular Design**: Easy to extend and maintain
