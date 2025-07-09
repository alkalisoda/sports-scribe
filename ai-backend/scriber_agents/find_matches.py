import json
from base_agent import BaseAgent
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
load_dotenv()

def find_matches_in_season():
    """Find matches in the 2010 Premier League season"""
    agent = BaseAgent()
    
    # 2010赛季的开始和结束日期
    start_date = datetime(2010, 8, 14)
    end_date = datetime(2011, 5, 17)
    
    current_date = start_date
    match_dates = []
    
    print("Searching for matches in 2010 Premier League season...")
    print(f"Season: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print("-" * 50)
    
    while current_date <= end_date:
        date_str = current_date.strftime("%Y-%m-%d")
        try:
            result = agent.get_fixtures("39", date_str)
            data = json.loads(result)
            
            if data.get("response") and len(data["response"]) > 0:
                print(f"✅ Found matches on {date_str}: {len(data['response'])} matches")
                match_dates.append({
                    "date": date_str,
                    "matches": data["response"]
                })
                
                # 显示前几场比赛的详细信息
                for i, match in enumerate(data["response"][:3]):
                    home_team = match.get("teams", {}).get("home", {}).get("name", "Unknown")
                    away_team = match.get("teams", {}).get("away", {}).get("name", "Unknown")
                    print(f"   {i+1}. {home_team} vs {away_team}")
                
                if len(data["response"]) > 3:
                    print(f"   ... and {len(data['response']) - 3} more matches")
                print()
                
                # 找到几个比赛日就停止，避免API调用过多
                if len(match_dates) >= 5:
                    break
            else:
                print(f"❌ No matches on {date_str}")
                
        except Exception as e:
            print(f"❌ Error on {date_str}: {str(e)}")
        
        current_date += timedelta(days=1)
    
    print(f"\nFound {len(match_dates)} match dates:")
    for match_date in match_dates:
        print(f"- {match_date['date']}: {len(match_date['matches'])} matches")
    
    return match_dates

if __name__ == "__main__":
    find_matches_in_season() 