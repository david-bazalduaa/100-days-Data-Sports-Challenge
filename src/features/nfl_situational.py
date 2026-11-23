"""NFL Situational Analysis Feature Engineering Module.

Day 25: NFL Situational Analysis.
Phase 2: EDA & Feature Engineering.

This module provides functional pipelines to construct high-leverage situational
features across NFL play-by-play data:
- Red Zone efficiency indicators (yardline_100 <= 20).
- Two-minute drill pressure flags (half_seconds_remaining <= 120).
- Third down conversion tracking by distance buckets (Short, Medium, Long).
- Formation impact (Under Center deception / play-action proxy vs. Shotgun).
- Aggregation pipelines to build team-level situational feature stores.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd


def build_situational_play_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer play-level situational boolean flags and contextual buckets.

    Parameters
    ----------
    df : pd.DataFrame
        Clean offensive scrimmage plays (pass and run with valid downs and EPA).

    Returns
    -------
    pd.DataFrame
        Enriched plays dataframe containing situational feature columns.
    """
    res = df.copy()

    # 1. High-leverage contextual binary indicators
    res["is_redzone"] = (res["yardline_100"] <= 20).astype(int)
    res["is_two_minute"] = (res["half_seconds_remaining"] <= 120).astype(int)
    res["is_third_down"] = (res["down"] == 3).astype(int)
    res["is_shotgun"] = res["shotgun"].fillna(0).astype(int)
    res["is_under_center_pass"] = (
        (res["play_type"] == "pass") & (res["shotgun"] == 0)
    ).astype(int)
    res["is_goal_to_go"] = (res["yardline_100"] <= res["ydstogo"]).astype(int)
    res["success"] = (res["epa"] > 0).astype(int)

    # 2. Distance bucketing for situational conversions
    distance_bins = [0, 3, 7, 100]
    distance_labels = ["Short (1-3 yds)", "Medium (4-7 yds)", "Long (8+ yds)"]
    res["distance_bucket"] = pd.cut(
        res["ydstogo"],
        bins=distance_bins,
        labels=distance_labels,
        right=True,
    )

    return res


def compute_team_situational_features(situational_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate high-leverage situational execution metrics per NFL franchise.

    Parameters
    ----------
    situational_df : pd.DataFrame
        Plays dataframe enriched with situational flags.

    Returns
    -------
    pd.DataFrame
        Team-level situational feature table sorted by overall Red Zone efficiency.
    """
    teams = sorted(situational_df["posteam"].dropna().unique())
    records = []

    for tm in teams:
        tm_plays = situational_df[situational_df["posteam"] == tm]

        # Red Zone
        rz = tm_plays[tm_plays["is_redzone"] == 1]
        rz_plays = len(rz)
        rz_pass_rate = (rz["play_type"] == "pass").mean() if rz_plays > 0 else np.nan
        rz_epa = rz["epa"].mean() if rz_plays > 0 else np.nan
        rz_td_rate = rz["touchdown"].mean() if rz_plays > 0 else np.nan
        rz_success = rz["success"].mean() if rz_plays > 0 else np.nan

        # Third Downs
        third = tm_plays[tm_plays["is_third_down"] == 1]
        third_plays = len(third)
        third_conv = third["third_down_converted"].mean() if third_plays > 0 else np.nan
        third_epa = third["epa"].mean() if third_plays > 0 else np.nan

        # Third Down Distance Splits
        third_short = third[third["distance_bucket"] == "Short (1-3 yds)"]
        third_short_conv = third_short["third_down_converted"].mean() if len(third_short) > 0 else np.nan

        third_long = third[third["distance_bucket"] == "Long (8+ yds)"]
        third_long_conv = third_long["third_down_converted"].mean() if len(third_long) > 0 else np.nan

        # Two-Minute Drill
        two_min = tm_plays[tm_plays["is_two_minute"] == 1]
        two_min_plays = len(two_min)
        two_min_epa = two_min["epa"].mean() if two_min_plays > 0 else np.nan
        two_min_pass_rate = (two_min["play_type"] == "pass").mean() if two_min_plays > 0 else np.nan

        # Passing Formations (Under Center Deception vs Shotgun)
        passes = tm_plays[tm_plays["play_type"] == "pass"]
        uc_passes = passes[passes["shotgun"] == 0]
        sg_passes = passes[passes["shotgun"] == 1]

        uc_rate = len(uc_passes) / len(passes) if len(passes) > 0 else np.nan
        uc_epa = uc_passes["epa"].mean() if len(uc_passes) > 0 else np.nan
        sg_epa = sg_passes["epa"].mean() if len(sg_passes) > 0 else np.nan

        records.append({
            "posteam": tm,
            "total_plays": len(tm_plays),
            "rz_plays": rz_plays,
            "rz_pass_rate": rz_pass_rate,
            "rz_mean_epa": rz_epa,
            "rz_touchdown_rate": rz_td_rate,
            "rz_success_rate": rz_success,
            "third_down_plays": third_plays,
            "third_down_conv_rate": third_conv,
            "third_down_mean_epa": third_epa,
            "third_short_conv_rate": third_short_conv,
            "third_long_conv_rate": third_long_conv,
            "two_min_plays": two_min_plays,
            "two_min_pass_rate": two_min_pass_rate,
            "two_min_mean_epa": two_min_epa,
            "under_center_pass_rate": uc_rate,
            "under_center_mean_epa": uc_epa,
            "shotgun_mean_epa": sg_epa,
            "play_action_epa_edge": uc_epa - sg_epa if pd.notna(uc_epa) and pd.notna(sg_epa) else np.nan,
        })

    res = pd.DataFrame(records)
    return res.sort_values("rz_mean_epa", ascending=False).reset_index(drop=True)


def build_nfl_situational_feature_store(
    pbp_path: Union[str, Path],
    output_dir: Union[str, Path] = "data/processed/nfl",
) -> Dict[str, pd.DataFrame]:
    """Execute end-to-end situational feature pipeline and persist feature tables.

    Parameters
    ----------
    pbp_path : str or Path
        Path to raw play-by-play parquet file.
    output_dir : str or Path, default="data/processed/nfl"
        Target directory to save parquet tables.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary holding 'fact_plays_situational' and 'features_team_situational'.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    situational_cols = [
        "play_id", "game_id", "season", "week", "posteam", "defteam",
        "down", "ydstogo", "yardline_100", "play_type", "epa", "yards_gained",
        "shotgun", "no_huddle", "half_seconds_remaining", "game_seconds_remaining", "qtr",
        "third_down_converted", "third_down_failed", "touchdown", "pass_touchdown", "rush_touchdown",
        "passer_player_id", "passer_player_name", "rusher_player_id", "rusher_player_name"
    ]

    print(f"Loading PBP data from {pbp_path}...")
    raw = pd.read_parquet(pbp_path, columns=situational_cols)

    clean = raw[
        (raw["play_type"].isin(["pass", "run"]))
        & (raw["down"].notna())
        & (raw["epa"].notna())
    ].copy()
    clean["down"] = clean["down"].astype(int)

    print("Building situational play-level features...")
    plays_situational = build_situational_play_features(clean)
    plays_out_path = out_dir / "fact_plays_situational_2023.parquet"
    plays_situational.to_parquet(plays_out_path, index=False)
    print(f"-> Saved play-level situational table: {plays_out_path} ({len(plays_situational)} plays)")

    print("Computing team-level situational features...")
    team_situational = compute_team_situational_features(plays_situational)
    team_out_path = out_dir / "features_team_situational_2023.parquet"
    team_situational.to_parquet(team_out_path, index=False)
    print(f"-> Saved team situational feature store: {team_out_path} ({len(team_situational)} teams)")

    return {
        "fact_plays_situational": plays_situational,
        "features_team_situational": team_situational,
    }


def main():
    """CLI execution entrypoint for situational feature generation."""
    data_dir = Path("data")
    pbp_path = data_dir / "raw" / "nfl" / "nfl_pbp_2023.parquet"
    output_dir = data_dir / "processed" / "nfl"

    if not pbp_path.exists():
        raise FileNotFoundError(f"Raw PBP file not found at {pbp_path}")

    build_nfl_situational_feature_store(
        pbp_path=pbp_path,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
