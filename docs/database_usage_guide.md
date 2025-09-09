# SportsScribe 数据库调用详解

## 概述

`SoccerDatabase` 类是 SportsScribe 项目的核心数据库接口，提供了对足球数据的全面访问和操作功能。该类采用了现代的异步编程模式，支持高性能的并发查询，并实现了智能缓存机制。

## 目录

1. [初始化和配置](#初始化和配置)
2. [基础实体查询](#基础实体查询)
3. [异步操作](#异步操作)
4. [统计数据聚合](#统计数据聚合)
5. [查询解析系统](#查询解析系统)
6. [性能优化](#性能优化)
7. [实际使用示例](#实际使用示例)
8. [最佳实践](#最佳实践)

---

## 初始化和配置

### 基础初始化

```python
from src.database import SoccerDatabase

# 创建数据库实例
db = SoccerDatabase(
    supabase_url="your_supabase_url",
    supabase_key="your_supabase_key",
    max_workers=10  # 异步操作的线程池大小
)
```

### 配置参数说明

- `supabase_url`: Supabase 项目的 URL
- `supabase_key`: Supabase 项目的 API 密钥
- `max_workers`: 用于异步操作的线程池大小，默认为 10

---

## 基础实体查询

### 球员查询

#### 通过 ID 获取球员信息

```python
# 同步方式
player = db.get_player("player_123")
if player:
    print(f"球员姓名: {player.name}")
    print(f"位置: {player.position}")
    print(f"国籍: {player.nationality}")

# 异步方式
player = await db.get_player_async("player_123")
```

#### 搜索球员

```python
# 按名字模糊搜索球员
players = db.search_players("Messi", limit=5)
for player in players:
    print(f"{player.name} - {player.position}")

# 异步搜索
players = await db.search_players_async("Ronaldo", limit=5)
```

### 球队查询

#### 通过 ID 获取球队信息

```python
# 同步方式
team = db.get_team("team_456")
if team:
    print(f"球队名称: {team.name}")
    print(f"国家: {team.country}")
    print(f"主场: {team.venue_name}")

# 异步方式
team = await db.get_team_async("team_456")
```

#### 搜索球队

```python
# 按名字搜索球队
teams = db.search_teams("Barcelona", limit=3)
for team in teams:
    print(f"{team.name} - {team.country}")

# 异步搜索
teams = await db.search_teams_async("Manchester", limit=3)
```

#### 获取球队球员列表

```python
# 获取指定球队的所有球员
team_players = db.get_team_players("Barcelona")
for player in team_players:
    print(f"{player['name']} - {player['position']}")
```

### 比赛查询

```python
# 通过 ID 获取比赛信息
match = db.get_match("match_789")
if match:
    print(f"比赛: {match.name}")
    print(f"主队进球: {match.goals_home}")
    print(f"客队进球: {match.goals_away}")
```

---

## 异步操作

### 单个异步查询

```python
import asyncio

async def get_player_info():
    # 异步获取球员信息
    player = await db.get_player_async("player_123")
    return player

# 运行异步函数
player = asyncio.run(get_player_info())
```

### 并发查询

```python
async def get_multiple_players():
    # 并发获取多个球员信息
    tasks = [
        db.get_player_async("player_1"),
        db.get_player_async("player_2"),
        db.get_player_async("player_3")
    ]
    
    players = await asyncio.gather(*tasks)
    return players

# 执行并发查询
players = asyncio.run(get_multiple_players())
```

---

## 统计数据聚合

### 球员统计查询

#### 基础统计查询

```python
# 获取球员的进球数
result = db.get_player_stat_sum(
    player_id="player_123",
    stat="goals"
)

print(f"总进球数: {result['value']}")
print(f"参与比赛数: {result['matches']}")
```

#### 带过滤条件的统计查询

```python
# 获取球员在主场的助攻数
result = db.get_player_stat_sum(
    player_id="player_123",
    stat="assists",
    venue="home",  # 主场比赛
    last_n=10      # 最近10场比赛
)

print(f"主场助攻数: {result['value']}")
```

#### 支持的统计类型

```python
# 所有支持的统计类型
supported_stats = [
    "goals",           # 进球
    "assists",         # 助攻
    "minutes_played",  # 上场时间
    "shots_on_target", # 射正
    "tackles",         # 铲断
    "interceptions",   # 拦截
    "passes_completed",# 传球成功
    "clean_sheets",    # 零封
    "saves",           # 扑救
    "yellow_cards",    # 黄牌
    "red_cards",       # 红牌
    "fouls_committed", # 犯规
    "fouls_drawn",     # 被犯规
    "shots",           # 射门
    "passes",          # 传球
    "pass_accuracy"    # 传球准确率
]
```

### 异步统计查询

```python
# 异步获取球员统计
result = await db.get_player_stat_sum_async(
    player_id="player_123",
    stat="goals",
    start_date="2024-08-01",
    end_date="2024-12-31"
)
```

### 批量并发统计查询

```python
# 批量获取多个球员的不同统计数据
requests = [
    {
        "player_id": "player_1",
        "stat": "goals",
        "venue": "home"
    },
    {
        "player_id": "player_2", 
        "stat": "assists",
        "last_n": 5
    },
    {
        "player_id": "player_3",
        "stat": "minutes_played",
        "start_date": "2024-01-01",
        "end_date": "2024-12-31"
    }
]

# 并发执行所有查询
results = await db.get_multiple_player_stats_concurrent(requests)

for i, result in enumerate(results):
    if result.get("status") != "error":
        print(f"请求 {i+1}: {result['value']}")
    else:
        print(f"请求 {i+1} 失败: {result['reason']}")
```

---

## 查询解析系统

### 自然语言查询处理

数据库支持通过 `run_from_parsed` 方法处理解析后的自然语言查询：

```python
# 假设 parsed 是从查询解析器得到的结果
result = db.run_from_parsed(
    parsed=parsed_query,
    player_name_to_id={"messi": "player_123"},
    default_season_label="2024-25"
)
```

### 支持的查询类型

#### 1. 球员统计查询

```python
# 示例查询: "Messi scored how many goals?"
# 解析后会调用球员统计查询
{
    "status": "success",
    "value": 25,
    "stat": "goals",
    "player_name": "Messi",
    "matches": 30
}
```

#### 2. 球队统计查询

```python
# 示例查询: "How many goals did Barcelona score?"
# 解析后会调用球队统计查询
{
    "status": "success",
    "value": 85,
    "stat": "goals", 
    "team_name": "Barcelona",
    "player_count": 25
}
```

#### 3. 比赛结果查询

```python
# 示例查询: "Barcelona vs Real Madrid result"
# 解析后会调用比赛查询
{
    "status": "success",
    "query_type": "match_result",
    "match": {
        "team1": {"name": "Barcelona", "goals": 2},
        "team2": {"name": "Real Madrid", "goals": 1},
        "winner": "team1",
        "score": "2-1"
    }
}
```

#### 4. 球员综合表现查询

```python
# 示例查询: "Messi performance"
# 返回球员的综合统计
{
    "status": "success",
    "query_type": "performance_overview",
    "performance": {
        "goals": 25,
        "assists": 15,
        "minutes_played": 2700,
        "shots": 120,
        "passes": 1800,
        "tackles": 45
    }
}
```

### 异步查询解析

```python
# 异步处理查询解析
result = await db.run_from_parsed_async(
    parsed=parsed_query,
    player_name_to_id=player_mapping,
    default_season_label="2024-25"
)
```

---

## 性能优化

### 缓存机制

数据库类使用了 `@lru_cache` 装饰器对频繁查询的数据进行缓存：

```python
# 缓存配置
@lru_cache(maxsize=1000)  # 球员缓存
@lru_cache(maxsize=1000)  # 球队缓存  
@lru_cache(maxsize=100)   # 比赛缓存
```

### 性能监控

```python
# 获取性能统计
stats = db.get_performance_stats()
print(f"总查询数: {stats['total_queries']}")
print(f"总耗时: {stats['total_time']:.2f}秒")
print(f"平均查询时间: {stats['average_query_time']:.3f}秒")
print(f"并发查询数: {stats['concurrent_queries']}")

# 重置性能统计
db.reset_performance_stats()
```

### 并发优化

```python
# 使用并发查询提高性能
async def optimized_team_analysis(team_name):
    # 并发获取球队的多项统计
    requests = []
    stats = ["goals", "assists", "yellow_cards", "red_cards"]
    
    team_players = db.get_team_players(team_name)
    
    for player in team_players:
        for stat in stats:
            requests.append({
                "player_id": player['id'],
                "stat": stat
            })
    
    # 一次性并发执行所有查询
    results = await db.get_multiple_player_stats_concurrent(requests)
    
    # 处理结果...
    return process_team_stats(results, team_players, stats)
```

---

## 实际使用示例

### 示例 1: 获取球员赛季统计

```python
async def get_player_season_stats(player_name, season="2024-25"):
    """获取球员赛季统计数据"""
    
    # 搜索球员
    players = await db.search_players_async(player_name, limit=1)
    if not players:
        return {"error": "Player not found"}
    
    player = players[0]
    
    # 获取赛季日期范围
    start_date, end_date = db.season_range(season)
    
    # 并发获取多项统计
    requests = [
        {"player_id": player.id, "stat": "goals", "start_date": start_date, "end_date": end_date},
        {"player_id": player.id, "stat": "assists", "start_date": start_date, "end_date": end_date},
        {"player_id": player.id, "stat": "minutes_played", "start_date": start_date, "end_date": end_date},
        {"player_id": player.id, "stat": "yellow_cards", "start_date": start_date, "end_date": end_date}
    ]
    
    results = await db.get_multiple_player_stats_concurrent(requests)
    
    return {
        "player": player.name,
        "season": season,
        "stats": {
            "goals": results[0].get("value", 0),
            "assists": results[1].get("value", 0), 
            "minutes": results[2].get("value", 0),
            "yellow_cards": results[3].get("value", 0)
        },
        "matches_played": max(r.get("matches", 0) for r in results)
    }

# 使用示例
stats = await get_player_season_stats("Messi", "2024-25")
print(stats)
```

### 示例 2: 比较两支球队

```python
async def compare_teams(team1_name, team2_name, stat="goals"):
    """比较两支球队的指定统计数据"""
    
    # 获取两支球队的球员
    team1_players = db.get_team_players(team1_name)
    team2_players = db.get_team_players(team2_name)
    
    if not team1_players or not team2_players:
        return {"error": "One or both teams not found"}
    
    # 创建并发请求
    requests = []
    
    # 团队1的请求
    for player in team1_players:
        requests.append({
            "player_id": player['id'],
            "stat": stat,
            "team": "team1"
        })
    
    # 团队2的请求
    for player in team2_players:
        requests.append({
            "player_id": player['id'], 
            "stat": stat,
            "team": "team2"
        })
    
    # 执行并发查询
    results = await db.get_multiple_player_stats_concurrent(requests)
    
    # 计算团队总计
    team1_total = sum(r.get("value", 0) for r in results[:len(team1_players)])
    team2_total = sum(r.get("value", 0) for r in results[len(team1_players):])
    
    return {
        "comparison": {
            team1_name: {"total": team1_total, "players": len(team1_players)},
            team2_name: {"total": team2_total, "players": len(team2_players)}
        },
        "stat": stat,
        "winner": team1_name if team1_total > team2_total else team2_name
    }

# 使用示例
comparison = await compare_teams("Barcelona", "Real Madrid", "goals")
print(comparison)
```

### 示例 3: 球队表现分析

```python
async def analyze_team_performance(team_name, last_n_games=None):
    """分析球队表现"""
    
    team_players = db.get_team_players(team_name)
    if not team_players:
        return {"error": "Team not found"}
    
    # 定义要分析的统计类型
    stats_to_analyze = [
        "goals", "assists", "shots", "passes", 
        "tackles", "yellow_cards", "red_cards"
    ]
    
    # 创建并发请求
    requests = []
    for player in team_players:
        for stat in stats_to_analyze:
            requests.append({
                "player_id": player['id'],
                "stat": stat,
                "last_n": last_n_games
            })
    
    # 执行并发查询
    results = await db.get_multiple_player_stats_concurrent(requests)
    
    # 处理结果
    team_stats = {}
    results_per_stat = len(team_players)
    
    for i, stat in enumerate(stats_to_analyze):
        stat_results = results[i * results_per_stat:(i + 1) * results_per_stat]
        team_stats[stat] = {
            "total": sum(r.get("value", 0) for r in stat_results),
            "average_per_player": sum(r.get("value", 0) for r in stat_results) / len(team_players)
        }
    
    return {
        "team": team_name,
        "analysis_scope": f"Last {last_n_games} games" if last_n_games else "All games",
        "player_count": len(team_players),
        "statistics": team_stats
    }

# 使用示例
analysis = await analyze_team_performance("Barcelona", last_n_games=10)
print(analysis)
```

---

## 最佳实践

### 1. 优先使用异步方法

```python
# ✅ 推荐：使用异步方法
player = await db.get_player_async("player_123")

# ❌ 不推荐：在异步环境中使用同步方法
player = db.get_player("player_123")  # 会阻塞事件循环
```

### 2. 利用并发查询

```python
# ✅ 推荐：使用并发查询
requests = [
    {"player_id": "p1", "stat": "goals"},
    {"player_id": "p2", "stat": "goals"},
    {"player_id": "p3", "stat": "goals"}
]
results = await db.get_multiple_player_stats_concurrent(requests)

# ❌ 不推荐：串行查询
results = []
for player_id in ["p1", "p2", "p3"]:
    result = await db.get_player_stat_sum_async(player_id, "goals")
    results.append(result)
```

### 3. 合理使用缓存

```python
# ✅ 缓存会自动处理频繁访问的数据
player = db.get_player("player_123")  # 第一次查询数据库
player = db.get_player("player_123")  # 第二次从缓存获取
```

### 4. 错误处理

```python
# ✅ 推荐：完整的错误处理
try:
    result = await db.get_player_stat_sum_async("player_123", "goals")
    if result.get("status") == "error":
        print(f"查询失败: {result.get('reason')}")
    elif result.get("status") == "no_data":
        print("未找到数据")
    else:
        print(f"进球数: {result.get('value', 0)}")
except DatabaseError as e:
    print(f"数据库错误: {e}")
except Exception as e:
    print(f"未知错误: {e}")
```

### 5. 性能监控

```python
# ✅ 推荐：定期监控性能
async def monitored_query():
    # 执行查询
    result = await db.get_player_stat_sum_async("player_123", "goals")
    
    # 检查性能统计
    stats = db.get_performance_stats()
    if stats["average_query_time"] > 1.0:  # 如果平均查询时间超过1秒
        print("⚠️  查询性能下降，考虑优化")
    
    return result
```

### 6. 批量操作优化

```python
# ✅ 推荐：批量获取球队所有球员统计
async def get_team_all_stats(team_name, stats_list):
    team_players = db.get_team_players(team_name)
    
    # 为所有球员和所有统计类型创建请求
    requests = []
    for player in team_players:
        for stat in stats_list:
            requests.append({
                "player_id": player['id'],
                "stat": stat
            })
    
    # 一次性并发执行
    results = await db.get_multiple_player_stats_concurrent(requests)
    
    # 组织结果
    organized_results = {}
    for i, player in enumerate(team_players):
        player_stats = {}
        for j, stat in enumerate(stats_list):
            result_index = i * len(stats_list) + j
            player_stats[stat] = results[result_index].get("value", 0)
        organized_results[player['name']] = player_stats
    
    return organized_results

# 使用示例
team_stats = await get_team_all_stats("Barcelona", ["goals", "assists", "minutes_played"])
```

---

## 总结

`SoccerDatabase` 类提供了完整的足球数据访问解决方案，具有以下特点：

1. **高性能**: 支持异步操作和并发查询
2. **智能缓存**: 自动缓存频繁访问的数据
3. **灵活查询**: 支持多种过滤条件和统计类型
4. **自然语言支持**: 可以处理解析后的自然语言查询
5. **性能监控**: 内置性能统计和监控功能
6. **错误处理**: 完善的异常处理机制

通过合理使用这些功能，可以构建高效、可靠的足球数据应用程序。
