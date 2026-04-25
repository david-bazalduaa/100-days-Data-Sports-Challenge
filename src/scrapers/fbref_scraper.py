"""
Day 6: FBref Scraper

Objective:
Build a web scraper using requests & BeautifulSoup for FBref player stats.

Instructions:
- Fill in the required functions/classes below.
"""
import time
import requests
import pandas as pd
from io import StringIO

# Verified Wayback Machine URL mapping for the top leagues (Early 2024 snapshots)
LEAGUE_URLS = {
    "Premier League": "https://web.archive.org/web/20240224151703/https://fbref.com/en/comps/9/stats/Premier-League-Stats",
    "La Liga": "https://web.archive.org/web/20240223180000/https://fbref.com/en/comps/12/stats/La-Liga-Stats",
    "Serie A": "https://web.archive.org/web/20240224170000/https://fbref.com/en/comps/11/stats/Serie-A-Stats",
    "Bundesliga": "https://web.archive.org/web/20240225120000/https://fbref.com/en/comps/20/stats/Bundesliga-Stats",
    "Ligue 1": "https://web.archive.org/web/20240224190000/https://fbref.com/en/comps/13/stats/Ligue-1-Stats"
}

def scrape_league(league_name):
    # 1. Validation check
    if league_name not in LEAGUE_URLS:
        print(f"Error: '{league_name}' is not supported. Please choose from: {list(LEAGUE_URLS.keys())}")
        return
        
    url = LEAGUE_URLS[league_name]
    print(f"Starting data extraction for: {league_name}")
    
    # 2. Ethical pause to respect server limits
    time.sleep(3) 
    
    try:
        response = requests.get(url)
        response.raise_for_status() 
        
        # 3. Remove HTML comments to reveal hidden tables
        uncommented_html = response.text.replace('', '')
        html_string = StringIO(uncommented_html)
        
        # 4. Extract the tables
        tables = pd.read_html(html_string)
        df = tables[0]
        
        # 5. Flatten the MultiIndex columns
        flattened_columns = [col[1] if 'Unnamed' in col[0] else f"{col[0]}_{col[1]}" for col in df.columns]
        df.columns = flattened_columns
        
        # 6. Save the clean data dynamically
        safe_filename = league_name.lower().replace(" ", "_")
        output_path = f"/Users/davidbazalduamendez/Documents/GitHub/100-days-Data-Sports-Challenge/data/raw/fbref/fbref_{safe_filename}_2023.csv"
        
        df.to_csv(output_path, index=False)
        print(f"Data saved successfully to {output_path}")
        
    except Exception as e:
        print(f"An error occurred during scraping: {e}")

if __name__ == "__main__":
    scrape_league("La Liga")