# Query Examples Guide

This guide provides practical examples of how to formulate different types of sports queries based on the JSON template.

## 🎯 Quick Reference

| Query Type | Template | Database Method | Example |
|------------|----------|-----------------|---------|
| **Basic Stats** | `{player} {stat}` | `get_player_stat_sum()` | "Messi goals" |
| **Ranking** | `Most {stat} in {team}` | `ranking filters` | "Most goals in Arsenal" |
| **Comparison** | `{player1} vs {player2}` | `comparative_stats()` | "Messi vs Ronaldo" |
| **Historical** | `{player} career {stat}` | `get_historical_stats()` | "Messi career goals" |
| **Context** | `Why is {event} significant?` | `context analysis` | "Why is El Clasico important?" |

## 📝 Query Categories with Examples

### 1. Direct Data Access ✅

**Purpose**: Get simple, direct statistical information

```python
# Basic player stats
"How many goals does Messi have?"
"Ronaldo's assists this season"
"Kevin De Bruyne's rating"

# Team stats
"Manchester City's total goals"
"Arsenal wins this season"
"Barcelona's clean sheets"

# With time filters
"Haaland's goals in last 10 games"
"Salah's assists at home"
"Liverpool's away form"
```

**Expected Output**: Single numerical value with context
```
"Messi has 15 goals this season in 20 appearances."
```

### 2. Statistical Analysis 📊

**Purpose**: Get comprehensive performance data

```python
# Performance overview
"Messi's performance this season"
"How is Haaland performing?"
"Show me Salah's stats"

# Multiple statistics
"Messi's goals and assists"
"Ronaldo's shots and rating"
"De Bruyne's passes and key passes"

# Calculated metrics
"Average goals per game for Haaland"
"Messi's shot conversion rate"
"Arsenal's points per game"
```

**Expected Output**: Multi-stat summary
```
"Messi this season: 15 goals, 12 assists, 8.7 rating in 20 appearances"
```

### 3. Ranking & Sorting 🏆

**Purpose**: Find top/bottom performers

```python
# Team rankings
"Who has the most goals in Manchester City?"
"Best rated player in Arsenal?"
"Top scorer in Barcelona?"

# League rankings
"Top 10 goal scorers in Premier League"
"Best assist providers this season"
"Highest rated players in La Liga"

# Bottom performers
"Who has the fewest goals in Chelsea?"
"Lowest rated player in Liverpool?"
"Least appearances in Real Madrid?"
```

**Expected Output**: Ranked list
```
"Top scorers in Arsenal: 1. Saka (12 goals), 2. Jesus (8 goals), 3. Martinelli (6 goals)"
```

### 4. Comparison Queries ⚖️

**Purpose**: Compare multiple entities

```python
# Player vs Player
"Messi vs Ronaldo goals"
"Haaland vs Mbappe this season"
"Compare Salah and Mane performance"

# Team vs Team
"Manchester City vs Arsenal points"
"Barcelona vs Real Madrid head to head"
"Liverpool vs Chelsea goals scored"

# Multiple comparisons
"Compare Messi, Ronaldo, and Neymar"
"Top 3 Premier League teams this season"
```

**Expected Output**: Side-by-side comparison
```
"Messi: 15 goals (0.75/game) vs Ronaldo: 12 goals (0.67/game) this season"
```

### 5. Historical Queries 📚

**Purpose**: Get historical data and trends

```python
# Career milestones
"Messi's career milestones"
"Ronaldo's major achievements"
"When did Pele score 1000 goals?"

# Records
"Messi's best goal scoring season"
"Ronaldo's Champions League records"
"Arsenal's longest unbeaten run"

# Trends
"Haaland's progression this season"
"How has Messi's performance changed?"
"Salah's goal scoring trend"

# When questions
"When did Barcelona last win Champions League?"
"Last time Arsenal won the league?"
"First time Messi scored in World Cup?"
```

**Expected Output**: Historical context with dates
```
"Messi's major milestones: 2009 - First Ballon d'Or, 2012 - 91 goals record, 2021 - Copa America"
```

### 6. Contextual Queries 🤔

**Purpose**: Get explanations and background

```python
# Significance
"Why is El Clasico important?"
"What makes Messi vs Ronaldo special?"
"Significance of Arsenal's unbeaten run"

# Derby context
"Manchester derby history"
"North London derby significance"
"Milan derby importance"

# Verification
"Is Messi really the GOAT?"
"Verify Ronaldo's goal record"
"Confirm Pep's coaching record"
```

**Expected Output**: Explanatory context
```
"El Clasico is significant because it's between Spain's two biggest clubs, Real Madrid and Barcelona, with over 100 years of rivalry..."
```

### 7. Advanced Analytics 🔬

**Purpose**: Complex multi-dimensional analysis

```python
# Form analysis
"Haaland's form in last 5 games"
"Arsenal's recent performance"
"Liverpool's current form"

# Venue analysis
"Messi's home vs away goals"
"Manchester City's home record"
"Barcelona's away form"

# Seasonal comparison
"Haaland this season vs last season"
"Arsenal's improvement from last year"
"Chelsea's decline analysis"
```

**Expected Output**: Multi-faceted analysis
```
"Haaland's recent form: 8 goals in last 5 games (1.6/game), 90% shot accuracy, 4 different competitions"
```

## 🔧 Best Practices

### ✅ Do This
```python
# Clear entity specification
"Lionel Messi goals this season"
"Manchester City vs Arsenal"

# Specific statistics
"Haaland's goals and assists"
"Salah's shots on target"

# Time context included
"Ronaldo's performance this season"
"Barcelona's last 10 games"
```

### ❌ Avoid This
```python
# Ambiguous entities
"Messi goals" (which Messi?)
"City wins" (Manchester City? other City?)

# Vague statistics
"Player performance" (what metrics?)
"Team stats" (which stats?)

# No time context
"Goals scored" (when? which season?)
```

## 🚀 Quick Start Examples

### For Beginners
```python
# Start with simple queries
"Messi goals"
"Arsenal wins"
"Ronaldo rating"

# Add time context
"Messi goals this season"
"Arsenal wins at home"
"Ronaldo rating in Champions League"
```

### For Advanced Users
```python
# Complex analytical queries
"Compare Messi's goal scoring rate at home vs away this season"
"Analyze Haaland's performance trend in last 15 games across all competitions"
"Historical comparison of Ronaldo's Champions League goals by season"
```

## 🎪 Interactive Examples

Try these queries to test different patterns:

```bash
# Basic stats
python query_test.py "How many goals does Haaland have?"

# Rankings
python query_test.py "Who has the most assists in Manchester City?"

# Comparisons
python query_test.py "Messi vs Ronaldo career goals"

# Historical
python query_test.py "When did Arsenal last win the Premier League?"

# Context
python query_test.py "Why is the Manchester derby significant?"
```

## 🔍 Debugging Tips

If your query doesn't work:

1. **Check entity names**: Use full names or common aliases
2. **Verify statistics**: Use supported stat names from the template
3. **Add time context**: Specify "this season", "career", etc.
4. **Simplify first**: Start with basic query, then add complexity

## 📚 Related Files

- `QUERY_PATTERNS_TEMPLATE.json` - Complete template with all patterns
- `DATABASE_USAGE_GUIDE.md` - Database class usage guide
- `src/query_parser.py` - Query parsing implementation
- `src/database.py` - Database reading methods