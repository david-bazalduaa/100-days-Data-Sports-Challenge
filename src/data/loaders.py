"""
Day 7: Unified Ingestion Pipeline

Objective:
Create functions: load_statsbomb, load_nfl, load_fbref. Add caching and error handling.

Instructions:
- Fill in the required functions/classes below.
"""

# Your code here

import os
import time
import requests
import logging
import pandas as pd
from io import StringIO

# 1. Professional Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

RAW_DATA_DIR = "C:\\Users\\david\\Desktop\\David\\Documentos\\David\\GitHub\\100-days-Data-Sports-Challenge\\data\\raw"
os.makedirs(RAW_DATA_DIR, exist_ok=True) # Ensure directory exists

# FBref League IDs mapping
FBREF_LEAGUE_IDS = {
    "Premier League": "9",
    "Championship": "10",        # England Tier 2
    "Serie A": "11",
    "La Liga": "12",
    "Ligue 1": "13",
    "Bundesliga": "20",
    "Major League Soccer": "22", # USA
    "Eredivisie": "23",          # Netherlands
    "Serie A Brazil": "24",      # Brazil
    "Liga MX": "31",             # Mexico
    "Primeira Liga": "32"        # Portugal
}

def _check_local_cache(filepath: str) -> pd.DataFrame:
    """Checks if a file exists locally. Returns DataFrame or None."""
    if os.path.exists(filepath):
        logger.info(f"Local cache found. Loading data from: {filepath}")
        return pd.read_csv(filepath)
    logger.info(f"No local cache found for: {filepath}. Proceeding to download.")
    return None

def load_fbref(league: str, season: str = "2023-2024") -> pd.DataFrame:
    """
    Loads FBref stats dynamically based on user input.
    """
    if league not in FBREF_LEAGUE_IDS:
        logger.error(f"League '{league}' not supported. Choose from: {list(FBREF_LEAGUE_IDS.keys())}")
        return None

    league_id = FBREF_LEAGUE_IDS[league]
    safe_filename = league.lower().replace(" ", "_")
    filepath = f"{RAW_DATA_DIR}/fbref/fbref_{safe_filename}_{season}.csv"
    
    # Check cache
    cached_data = _check_local_cache(filepath)
    if cached_data is not None:
        return cached_data
        
    logger.info(f"Starting data extraction for: {league} ({season})")
    time.sleep(3) # Ethical pause
    
    # Professional Headers to simulate a real browser
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9"
    }
    
    # Dynamic URL Construction
    url = f"https://fbref.com/en/comps/{league_id}/{season}/stats/{season}-{league.replace(' ', '-')}-Stats"
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status() 
        
        uncommented_html = response.text.replace('', '')
        html_string = StringIO(uncommented_html)
        
        tables = pd.read_html(html_string)
        df = tables[0]
        
        # Flatten MultiIndex
        flattened_columns = [col[1] if 'Unnamed' in col[0] else f"{col[0]}_{col[1]}" for col in df.columns]
        df.columns = flattened_columns
        
        df.to_csv(filepath, index=False)
        logger.info(f"Data successfully saved to {filepath}")
        return df
        
    except Exception as e:
        logger.error(f"Failed to load FBref data: {e}")
        return None

from statsbombpy import sb
import nfl_data_py as nfl

def load_statsbomb(competition_id: int, season_id: int) -> pd.DataFrame:
    """
    Loads StatsBomb match data for a specific competition and season.
    """
    filepath = f"{RAW_DATA_DIR}/statsbomb/sb_matches_{competition_id}_{season_id}.csv"

    # 1. Check Cache
    cached_data = _check_local_cache(filepath)
    if cached_data is not None:
        return cached_data
        
    # 2. Download via API
    logger.info(f"Downloading StatsBomb data for comp: {competition_id}, season: {season_id}")
    try:
        df = sb.matches(competition_id=competition_id, season_id=season_id)
        
        df.to_csv(filepath, index=False)
        logger.info(f"Data successfully saved to {filepath}")
        return df
        
    except Exception as e:
        logger.error(f"Failed to load StatsBomb data: {e}")
        return None

def load_nfl(years: list, data_type: str = "pbp") -> pd.DataFrame:
    """
    Loads NFL data for given years. 
    Uses Parquet format for optimized storage and fast reading.
    """
    # Create a safe string from the list of years
    years_str = "_".join(map(str, years))
    
    # NEW: Change extension to .parquet
    filepath = f"{RAW_DATA_DIR}/nfl/nfl_{data_type}_{years_str}.parquet"
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # 1. Check Cache (NEW: read_parquet)
    if os.path.exists(filepath):
        logger.info(f"Local cache found. Loading Parquet data from: {filepath}")
        return pd.read_parquet(filepath, engine="pyarrow")
        
    # 2. Download via API
    logger.info(f"Downloading NFL {data_type} data for years: {years}")
    try:
        if data_type == "pbp":
            df = nfl.import_pbp_data(years)
        elif data_type == "roster":
            df = nfl.import_rosters(years)
        else:
            logger.error(f"Unsupported NFL data type: {data_type}")
            return None
            
        # NEW: Save as Parquet instead of CSV
        df.to_parquet(filepath, index=False, engine="pyarrow")
        logger.info(f"Data successfully saved to {filepath}")
        return df
        
    except Exception as e:
        logger.error(f"Failed to load NFL data: {e}")
        return None