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
import ssl

# Bypass SSL certificate errors on macOS for data downloads
ssl._create_default_https_context = ssl._create_unverified_context


# 1. Professional Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Rutas dinámicas para que funcione tanto en Windows como en Mac
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
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
        if filepath.endswith('.parquet'):
            return pd.read_parquet(filepath, engine='pyarrow')
        else:
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

def load_statsbomb_matches(competition_id: int, season_id: int) -> pd.DataFrame:
    """
    Loads StatsBomb match data for a specific competition and season.
    """
    # 1. Cambiamos la extensión a .parquet
    filepath = f"{RAW_DATA_DIR}/statsbomb/sb_matches_{competition_id}_{season_id}.parquet"

    # 2. Check Cache
    cached_data = _check_local_cache(filepath)
    if cached_data is not None:
        return cached_data
        
    # 3. Download via API
    logger.info(f"Downloading StatsBomb data for comp: {competition_id}, season: {season_id}")
    try:
        import statsbombpy.sb as sb
        df = sb.matches(competition_id=competition_id, season_id=season_id)
        
        # 4. Usamos to_parquet en lugar de to_csv
        df.to_parquet(filepath, index=False, engine='pyarrow')
        logger.info(f"Data successfully saved to {filepath}")
        return df
        
    except Exception as e:
        logger.error(f"Failed to load StatsBomb data: {e}")
        return None

def load_statsbomb_events(competition_id: int, season_id: int) -> pd.DataFrame:
    """
    Loads StatsBomb EVENT data for a specific competition and season,
    saving it directly as a Parquet file for better performance.
    """
    # Nota el cambio de extensión a .parquet
    filepath = f"{RAW_DATA_DIR}/statsbomb/sb_events_{competition_id}_{season_id}.parquet"

    # 1. Check Cache (Asegúrate de que tu función _check_local_cache lea parquet si aplica)
    cached_data = _check_local_cache(filepath)
    if cached_data is not None:
        return cached_data
        
    # 2. Download via API
    logger.info(f"Downloading StatsBomb EVENTS for comp: {competition_id}, season: {season_id}")
    try:
        # A. Obtenemos los partidos para extraer la lista de match_ids
        import statsbombpy.sb as sb
        matches_df = sb.matches(competition_id=competition_id, season_id=season_id)
        match_ids = matches_df['match_id'].tolist()
        
        # B. Iteramos sobre cada partido para descargar sus eventos
        all_events = []
        for match_id in match_ids:
            # sb.events() extrae pases, tiros, faltas, etc., de un solo partido
            match_events = sb.events(match_id=match_id)
            all_events.append(match_events)
            
        # C. Unimos todos los DataFrames en uno solo
        df_events = pd.concat(all_events, ignore_index=True)
        
        # D. Guardamos en formato Parquet
        df_events.to_parquet(filepath, index=False, engine='pyarrow')
        logger.info(f"Events data successfully saved to {filepath}")
        
        return df_events
        
    except Exception as e:
        logger.error(f"Failed to load StatsBomb event data: {e}")
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
            df = nfl.import_seasonal_rosters(years)
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

def load_metrica_tracking(url: str, team_name: str) -> pd.DataFrame:
    """
    Loads and cleans Metrica Sports tracking data from a URL.
    Standardizes column names to {Team}_{PlayerID}_{Coord} format.
    """
    logger.info(f"Downloading Metrica tracking data for {team_name}...")
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        # Read the first 3 rows to extract metadata for headers
        header_df = pd.read_csv(StringIO(response.text), nrows=3, header=None)
        player_ids = header_df.iloc[1].values
        
        # Load the actual data
        df = pd.read_csv(StringIO(response.text), skiprows=3, header=None)
        
        # Professional Column Renaming
        columns = ['Period', 'Frame', 'Time [s]']
        
        # Metrica tracking data columns after the first 3 (Period, Frame, Time):
        # Pairs of (X, Y) for each player, then finally (X, Y) for the ball.
        # We use the player IDs from Row 1.
        data_cols = df.columns[3:]
        for i in range(0, len(data_cols)-2, 2):
            p_id = player_ids[i+3]
            columns.append(f"{team_name}_{p_id}_X")
            columns.append(f"{team_name}_{p_id}_Y")
            
        # Add Ball columns (last two)
        columns.append("Ball_X")
        columns.append("Ball_Y")
        
        df.columns = columns
        logger.info(f"Successfully loaded tracking data for {team_name}. Shape: {df.shape}")
        return df
        
    except Exception as e:
        logger.error(f"Failed to load Metrica tracking data: {e}")
        return None