import sys
import os
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# 1. Path configuration and imports
sys.path.append(os.path.abspath('src/data')) 
from loaders import load_nfl

PROCESSED_DIR = "data/processed/nfl/"
os.makedirs(PROCESSED_DIR, exist_ok=True)

print("--- 🏈 Starting NFL ETL Pipeline (Star Schema) ---")

print("\n1. Extracting data...")
SEASON = [2023]
pbp_df = load_nfl(years=SEASON, data_type="pbp")
roster_df = load_nfl(years=SEASON, data_type="roster")

if pbp_df is None or roster_df is None:
    print("Data extraction failed.")
    sys.exit(1)

print("\n2. Transforming dimensions and facts...")

dim_games = pbp_df[['game_id', 'season', 'week', 'home_team', 'away_team']].dropna(subset=['game_id']).drop_duplicates(subset=['game_id'])

dim_teams = pd.DataFrame({'team_id': pbp_df['home_team'].dropna().unique()})
dim_teams['team_abbr'] = dim_teams['team_id'] 
dim_teams['team_name'] = dim_teams['team_abbr']

dim_players = roster_df[['player_id', 'player_name', 'position', 'college']].copy()
dim_players.dropna(subset=['player_id'], inplace=True)
dim_players.drop_duplicates(subset=['player_id'], inplace=True)

pbp_df['drive_id'] = pbp_df['game_id'] + "_" + pbp_df['drive'].astype(str)
dim_drives = pbp_df[['drive_id', 'game_id', 'posteam', 'fixed_drive_result', 'drive_play_count']].dropna(subset=['drive_id']).drop_duplicates(subset=['drive_id'])

pbp_df['global_play_id'] = pbp_df['game_id'] + "_" + pbp_df['play_id'].astype(str)

fact_columns = [
    'global_play_id', 'game_id', 'drive_id', 
    'passer_player_id', 'rusher_player_id', 'receiver_player_id',
    'posteam', 'defteam', 'down', 'ydstogo', 'yardline_100', 
    'epa', 'yards_gained', 'play_type'
]
existing_cols = [col for col in fact_columns if col in pbp_df.columns]
fact_plays = pbp_df[existing_cols].copy()
fact_plays.rename(columns={'global_play_id': 'play_id'}, inplace=True)

print("\n3. Loading to Parquet format...")
dim_games.to_parquet(f"{PROCESSED_DIR}dim_games.parquet", index=False)
dim_teams.to_parquet(f"{PROCESSED_DIR}dim_teams.parquet", index=False)
dim_players.to_parquet(f"{PROCESSED_DIR}dim_players.parquet", index=False)
dim_drives.to_parquet(f"{PROCESSED_DIR}dim_drives.parquet", index=False)
fact_plays.to_parquet(f"{PROCESSED_DIR}fact_plays.parquet", index=False)

print(f"✅ Pipeline complete! NFL Star Schema ready at: {PROCESSED_DIR}")
