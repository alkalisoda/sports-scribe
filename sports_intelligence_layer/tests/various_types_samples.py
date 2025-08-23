"""
SIL: Test Examples

1. Ambiguous Entity References

# Multiple players with same last name
"How many goals has Smith scored this season?"  # Could be multiple Smiths
"What's Williams' assist record?"  # Common surname

# Partial name matches that could be multiple people
"How is Alex performing this season?"  # Alex Oxlade-Chamberlain vs other Alex players
"Show me Taylor's stats"  # Multiple Taylors in football

# Similar team names
"How is United doing?"  # Manchester United vs Newcastle United vs other United teams
"What's City's record?"  # Manchester City vs other City teams


2. Complex Temporal Queries
# Relative time periods that need calculation
"How has Messi performed in the last 3 months?"
"Show me Kane's goals since January 15th"
"What's Liverpool's form over the past 6 weeks?"
"How many assists did De Bruyne get between March and May?"

# Cross-season comparisons
"Compare Haaland's first 10 games this season vs last season"
"How does Arsenal's December record compare across the last 3 years?"

# Holiday/special periods
"How many goals were scored during Christmas fixtures?"
"What's the team's performance during international breaks?"

3. Compound Statistical Queries

# Multiple statistics in one query
"Show me Salah's goals, assists, and yellow cards this season"
"What are the top 3 scorers' goals, minutes played, and shots on target?"

# Conditional statistics
"How many goals has Benzema scored when Real Madrid was losing?"
"What's Liverpool's win rate when Salah doesn't score?"
"Show me City's clean sheets in games where they scored 3+ goals"

# Rate-based statistics
"What's Mbappe's goals per 90 minutes ratio?"
"Show me the team's points per game at home vs away"
"What's the goalkeeper's saves per shot ratio?"



4. Tactical and Formation Queries
# Formation-specific questions
"How effective is Arsenal when playing 4-2-3-1 vs 4-3-3?"
"What's Liverpool's win rate with a false 9?"
"Show me City's possession stats when using inverted wingers"

# Position-specific queries
"How many goals have Arsenal's center-backs scored?"
"What's the combined assists from Liverpool's fullbacks?"
"Show me defensive midfielders with the most tackles"

# Substitution patterns
"How often does Guardiola make tactical substitutions before 60 minutes?"
"What's the team's scoring rate after making their first substitution?"


5. Weather and External Factors
# Weather conditions
"How does Liverpool perform in rainy conditions?"
"What's City's record in games below 5 degrees Celsius?"
"Show me goals scored in snow conditions"

# Time of day / kick-off times
"How does Arsenal perform in early kick-offs vs evening games?"
"What's the team's record in 12:30 PM starts?"
"Show me late goal statistics in evening matches"

# Stadium-specific
"How many goals has Salah scored at Old Trafford specifically?"
"What's Liverpool's record at newly built stadiums?"
"Show me penalty conversion rates at Wembley"


6. Financial and Transfer Context
# Transfer-related questions
"How has the team performed since the January transfer window?"
"What's the goal return on the summer signings?"
"Show me performance before and after the manager's new contract"

# Value-based queries
"How many goals per million spent on strikers?"
"What's the points return on defensive investments?"
"Show me academy players vs purchased players statistics"


7. Injury and Suspension Context
# Availability-based queries
"How does the team perform without their captain?"
"What's Liverpool's record when 3+ key players are injured?"
"Show me goal-scoring when the main striker is suspended"

# Recovery patterns
"How do players perform in their first game back from injury?"
"What's the team's form immediately after international duty?"
"Show me rotation policy effectiveness during fixture congestion"




8. Referee and Official Bias
# Referee-specific patterns
"How many penalties does this referee typically award?"
"What's Liverpool's record with referee Mike Dean?"
"Show me yellow card patterns with different officials"

# VAR-related queries
"How many VAR decisions have gone against Arsenal this season?"
"What's the goal difference in pre-VAR vs post-VAR matches?"
"Show me overturned decisions impact on final results"

# Multi-competition queries
"How does Mbappe's Champions League form compare to Ligue 1?"
"What's the goal difference between domestic and European games?"
"Show me players who perform better internationally than domestically"

9. Cross-League and International Context


# League comparison
"How would Haaland's goals translate to Serie A scoring rates?"
"Compare Premier League vs Bundesliga defensive statistics"
"What's the pace difference between La Liga and Premier League?"

10. Nonsensical but Plausible Queries
# Grammatically correct but logically flawed
"How many goals has the stadium scored this season?"
"What's the grass's assist record?"
"Show me the referee's clean sheet statistics"

# Impossible combinations
"How many hat-tricks has the goalkeeper scored in defense?"
"What's the team's batting average in football?"
"Show me the offside trap's goal-scoring record"

# Time paradoxes
"How will Messi perform next season based on last season?"
"What's tomorrow's match result prediction based on yesterday's training?"
"Show me future goals that have already been scored"


11. Extremely Vague Queries
# Ultra-generic requests
"Show me everything about football"
"What's happening in sports?"
"Tell me about the thing with the ball"
"How good is good?"

# Pronoun confusion
"How is he doing this season?"  # No antecedent
"What's their record against them?"  # Ambiguous pronouns
"Show me his stats compared to theirs"  # Multiple unclear references

12. Technical Edge Cases
# SQL injection attempts (benign)
"How many goals has Robert'); DROP TABLE players; -- scored?"
"What's the team's record WHERE 1=1; DELETE FROM stats?"

# Unicode and special characters
"How many goals has Müller scored this season?"
"What's São Paulo's record?"
"Show me Žan Celar's statistics"

# Very long queries
"How many goals has this extremely long named player whose full name is..." (300+ characters)

# Empty components
"How many  has  scored this season?"  # Missing entity and stat
"What's  record against ?"  # Missing both entities

13. Emotional/Subjective Queries
# Sentiment-based questions
"How frustrated are Arsenal fans with their attack?"
"What's the team's confidence level after the loss?"
"Show me the most heartbreaking defeats this season"

# Opinion-based queries
"Who is the most overrated player in the league?"
"What's the worst refereeing decision this season?"
"Which team has the most boring playing style?"


14. Meta-Queries About the System
# Self-referential questions
"How accurate are your statistics?"
"What data are you missing about this player?"
"How confident are you in this analysis?"

# System capability questions
"Can you predict next week's results?"
"Do you know about amateur leagues?"
"What's your favorite team?"


"""
