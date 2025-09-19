# Soccer Intelligence Layer

A complete end-to-end system for processing natural language soccer queries
and retrieving data from Supabase.

## Overview

This system implements the complete pipeline: **Query → Parse → SQL → Results**

- **Query**: Natural language soccer questions
  (e.g., "How many goals has Haaland scored this season?")
- **Parse**: Extract entities, statistics, time context, and filters
- **SQL**: Generate and execute database queries against Supabase
- **Results**: Return structured data with metadata

## Features

- ✅ Natural language query parsing
- ✅ Entity recognition (players, teams, competitions)
- ✅ Statistical analysis (goals, assists, minutes, etc.)
- ✅ Time context handling (this season, last season, career, etc.)
- ✅ Filter support (home/away, venue, etc.)
- ✅ Supabase integration
- ✅ Performance optimized (<500ms response time)
- ✅ Comprehensive error handling
- ✅ Detailed logging and debugging

## Quick Start

### 1. Install Dependencies

```bash
cd sports_intelligence_layer
pip install -r requirements.txt
```

### 2. Set Environment Variables

Create a `.env` file in the project root:

```bash
# Supabase Configuration
SUPABASE_URL=your_supabase_project_url
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
```

### 3. Run the End-to-End Test

```bash
python tests/test_end_to_end.py
```

### 4. Use in Your Code

```python
from main import SoccerIntelligenceLayer

# Initialize the system
sil = SoccerIntelligenceLayer()

# Process a query
query = "How many goals has Kaoru Mitoma scored this season?"
result = sil.process_query(query)

print(result)
```

## Database Schema

The system expects the following tables in your Supabase database:

### Players Table

```sql
CREATE TABLE players (
    id UUID PRIMARY KEY,
    name TEXT NOT NULL,
    position TEXT,
    team_id UUID REFERENCES teams(id),
    -- other fields as needed
);
```

### Teams Table

```sql
CREATE TABLE teams (
    id UUID PRIMARY KEY,
    name TEXT NOT NULL,
    -- other fields as needed
);
```

### Player Match Stats Table

```sql
CREATE TABLE player_match_stats (
    match_id UUID,
    player_id UUID REFERENCES players(id),
    team_id UUID REFERENCES teams(id),
    minutes INTEGER,
    goals INTEGER,
    assists INTEGER,
    shots INTEGER,
    shots_on_target INTEGER,
    passes INTEGER,
    pass_accuracy INTEGER,
    yellow_cards INTEGER,
    red_cards INTEGER,
    match_date DATE,
    venue TEXT, -- 'home', 'away', 'neutral'
    PRIMARY KEY (match_id, player_id)
);
```

## Example Queries

The system can handle various types of queries:

### Basic Statistics

- "How many goals has Kaoru Mitoma scored this season?"
- "What's Danny Welbeck's assist record?"
- "How many minutes has Jordan Pickford played?"

### Time-based Queries

- "Show me Dominic Calvert-Lewin's goals in the last 5 games"
- "What's João Pedro's performance this season?"
- "How many clean sheets has Jason Steele kept last season?"

### Venue-based Queries

- "What's João Pedro's performance at home?"
- "How many goals has Mitoma scored away from home?"

## API Response Format

```json
{
  "status": "success",
  "query": {
    "original": "How many goals has Kaoru Mitoma scored this season?",
    "parsed": {
      "entities": [
        {
          "name": "Kaoru Mitoma",
          "type": "player",
          "confidence": 0.97
        }
      ],
      "time_context": "this_season",
      "statistic_requested": "goals",
      "comparison_type": null,
      "filters": {},
      "intent": "stat_lookup",
      "confidence": 0.9
    }
  },
  "result": {
    "entity": {
      "type": "player",
      "id": "106835",
      "name": "Kaoru Mitoma"
    },
    "stat": "goals",
    "result": {
      "value": 1,
      "matches": 1,
      "filters": {
        "start_date": "2024-08-01",
        "end_date": "2025-06-30",
        "venue": null,
        "last_n": null
      }
    },
    "meta": {
      "query_intent": "stat_lookup",
      "confidence": 0.9
    }
  },
  "metadata": {
    "timestamp": "2024-01-15T10:30:00.000Z",
    "processing_time_ms": 150,
    "data_source": "supabase"
  }
}
```

## Performance

- **Target**: <500ms average response time
- **Optimizations**:
  - LRU caching for entity lookups
  - Compiled regex patterns
  - Efficient database queries
  - Minimal data transfer

## Testing

Run comprehensive tests:

```bash
# Test parser only
python -c "
from src.query_parser import SoccerQueryParser
parser = SoccerQueryParser()
print(parser.parse_query('How many goals has Haaland scored?'))
"

# Test database connection
python -c "
from src.database import SoccerDatabase
import os
db = SoccerDatabase(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_SERVICE_ROLE_KEY'))
print('Connection successful')
"

# Run full end-to-end test
python tests/test_end_to_end.py
```

## Error Handling

The system handles various error scenarios:

- **Invalid queries**: Returns structured error with suggestions
- **Database connection issues**: Graceful fallback with error messages
- **Missing data**: Clear indication when no data is found
- **Parsing failures**: Confidence scoring and fallback strategies

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `SUPABASE_URL` | Your Supabase project URL | Yes |
| `SUPABASE_SERVICE_ROLE_KEY` | Your Supabase service role key | Yes |

### Customization

You can customize the system by:

1. **Adding new entities**: Modify `data/players.json` and `data/teams.json`
2. **Extending statistics**: Add new patterns in `data/statistics.json`
3. **Custom filters**: Implement new filter types in the parser
4. **Database schema**: Extend tables and update the database interface

## Troubleshooting

### Common Issues

1. **"Supabase credentials not found"**
   - Ensure `.env` file exists with correct credentials
   - Check that environment variables are loaded

2. **"Player not found"**
   - Verify player exists in database
   - Check spelling and aliases in `data/players.json`

3. **"Database connection failed"**
   - Verify Supabase URL and key are correct
   - Check network connectivity
   - Ensure database tables exist

4. **"Performance target not met"**
   - Check database indexes
   - Monitor query execution time
   - Consider caching strategies

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is part of the SportsScribe system.
