"""NFL Player Efficiency Metrics Feature Engineering Module.

Day 21: NFL Feature Engineering: Player Efficiency Metrics.
Phase 2: EDA & Feature Engineering.

This module provides functional pipelines to compute advanced player-level
efficiency metrics for Quarterbacks (QB), Running Backs (RB), and Pass
Catchers (WR/TE) from nflfastR play-by-play tracking data.

Metrics computed:
- QB: EPA/dropback, CPOE, aDOT, sack rate, scramble rate, success rate.
- RB: EPA/rush, success rate, breakaway run rate (10+ and 15+ yds), negative run rate.
- WR/TE: Target share, air yards share, WOPR, yards per target, YAC per reception, EPA/target.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd


def load_clean_pbp(
    pbp_path: Union[str, Path],
    players_path: Optional[Union[str, Path]] = None,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Load and prepare selected tracking columns from raw play-by-play data.

    Parameters
    ----------
    pbp_path : str or Path
        Path to the raw nflfastR play-by-play parquet file.
    players_path : str or Path, optional
        Path to dim_players.parquet containing official player roster positions.

    Returns
    -------
    Tuple[pd.DataFrame, Optional[pd.DataFrame]]
        Loaded play-by-play dataframe with essential tracking fields and
        optional players dimension dataframe.
    """
    feature_cols = [
        "play_id", "game_id", "season", "week", "posteam", "defteam",
        "down", "ydstogo", "yardline_100", "play_type", "epa", "yards_gained",
        # QB Tracking
        "passer_player_id", "passer_player_name", "qb_dropback", "pass_attempt",
        "complete_pass", "incomplete_pass", "interception", "sack", "cpoe",
        "air_yards", "qb_scramble",
        # Rusher Tracking
        "rusher_player_id", "rusher_player_name", "rush_attempt", "rush_touchdown",
        # Receiver Tracking
        "receiver_player_id", "receiver_player_name", "yards_after_catch",
        "pass_touchdown", "route",
    ]

    pbp_df = pd.read_parquet(pbp_path, columns=feature_cols)

    players_df = None
    if players_path is not None and Path(players_path).exists():
        players_df = pd.read_parquet(players_path)

    return pbp_df, players_df


def compute_qb_efficiency_features(
    pbp_df: pd.DataFrame,
    players_df: Optional[pd.DataFrame] = None,
    min_dropbacks: int = 1,
) -> pd.DataFrame:
    """Compute quarterback efficiency features across all initiated dropbacks.

    Consolidates dropbacks across pass attempts, sacks, and QB scrambles.

    Parameters
    ----------
    pbp_df : pd.DataFrame
        Enriched play-by-play dataframe.
    players_df : pd.DataFrame, optional
        Player dimension table containing official roster position.
    min_dropbacks : int, default=1
        Minimum dropbacks threshold to filter output.

    Returns
    -------
    pd.DataFrame
        Quarterback efficiency feature table sorted by EPA per dropback.
    """
    dropbacks = pbp_df[pbp_df["qb_dropback"] == 1].copy()

    # Consolidate QB identity: sacks/passes use passer_player_id, scrambles use rusher_player_id
    dropbacks["qb_id"] = dropbacks["passer_player_id"].fillna(dropbacks["rusher_player_id"])
    dropbacks["qb_name"] = dropbacks["passer_player_name"].fillna(dropbacks["rusher_player_name"])

    qb_agg = dropbacks.groupby(["qb_id", "qb_name", "posteam"]).agg(
        dropbacks=("play_id", "count"),
        pass_attempts=("pass_attempt", "sum"),
        completions=("complete_pass", "sum"),
        sacks=("sack", "sum"),
        scrambles=("qb_scramble", "sum"),
        interceptions=("interception", "sum"),
        pass_touchdowns=("pass_touchdown", "sum"),
        total_epa=("epa", "sum"),
        epa_per_dropback=("epa", "mean"),
        success_rate=("epa", lambda s: (s > 0).mean()),
        cpoe=("cpoe", "mean"),
        adot=("air_yards", "mean"),
    ).reset_index()

    # Derived rates
    qb_agg["sack_rate"] = qb_agg["sacks"] / qb_agg["dropbacks"]
    qb_agg["scramble_rate"] = qb_agg["scrambles"] / qb_agg["dropbacks"]
    qb_agg["completion_rate"] = qb_agg["completions"] / qb_agg["pass_attempts"].replace(0, np.nan)

    if players_df is not None:
        qb_agg = qb_agg.merge(
            players_df[["player_id", "player_name", "position", "college"]],
            left_on="qb_id",
            right_on="player_id",
            how="left",
        )

    filtered = qb_agg[qb_agg["dropbacks"] >= min_dropbacks].copy()
    return filtered.sort_values("epa_per_dropback", ascending=False).reset_index(drop=True)


def compute_rb_efficiency_features(
    pbp_df: pd.DataFrame,
    players_df: Optional[pd.DataFrame] = None,
    min_rushes: int = 1,
) -> pd.DataFrame:
    """Compute running back efficiency and explosiveness features.

    Parameters
    ----------
    pbp_df : pd.DataFrame
        Enriched play-by-play dataframe.
    players_df : pd.DataFrame, optional
        Player dimension table containing official roster position.
    min_rushes : int, default=1
        Minimum carries threshold to filter output.

    Returns
    -------
    pd.DataFrame
        Running back efficiency feature table sorted by EPA per rush.
    """
    rushes = pbp_df[(pbp_df["rush_attempt"] == 1) & (pbp_df["rusher_player_id"].notna())].copy()
    rushes["success"] = (rushes["epa"] > 0).astype(int)
    rushes["is_explosive_10"] = (rushes["yards_gained"] >= 10).astype(int)
    rushes["is_explosive_15"] = (rushes["yards_gained"] >= 15).astype(int)
    rushes["is_negative"] = (rushes["yards_gained"] < 0).astype(int)

    rb_agg = rushes.groupby(["rusher_player_id", "rusher_player_name", "posteam"]).agg(
        rushes=("play_id", "count"),
        total_rushing_yards=("yards_gained", "sum"),
        ypc=("yards_gained", "mean"),
        median_yards=("yards_gained", "median"),
        total_epa=("epa", "sum"),
        epa_per_rush=("epa", "mean"),
        success_rate=("success", "mean"),
        breakaway_run_rate_10=("is_explosive_10", "mean"),
        breakaway_run_rate_15=("is_explosive_15", "mean"),
        negative_run_rate=("is_negative", "mean"),
        rush_touchdowns=("rush_touchdown", "sum"),
    ).reset_index()

    if players_df is not None:
        rb_agg = rb_agg.merge(
            players_df[["player_id", "player_name", "position", "college"]],
            left_on="rusher_player_id",
            right_on="player_id",
            how="left",
        )

    filtered = rb_agg[rb_agg["rushes"] >= min_rushes].copy()
    return filtered.sort_values("epa_per_rush", ascending=False).reset_index(drop=True)


def compute_receiver_efficiency_features(
    pbp_df: pd.DataFrame,
    players_df: Optional[pd.DataFrame] = None,
    min_targets: int = 1,
) -> pd.DataFrame:
    """Compute wide receiver and tight end volume opportunity and efficiency features.

    Parameters
    ----------
    pbp_df : pd.DataFrame
        Enriched play-by-play dataframe.
    players_df : pd.DataFrame, optional
        Player dimension table containing official roster position.
    min_targets : int, default=1
        Minimum targets threshold to filter output.

    Returns
    -------
    pd.DataFrame
        Receiver efficiency feature table sorted by Weighted Opportunity Rating (WOPR).
    """
    targets = pbp_df[(pbp_df["pass_attempt"] == 1) & (pbp_df["receiver_player_id"].notna())].copy()

    team_totals = targets.groupby("posteam").agg(
        team_targets=("play_id", "count"),
        team_air_yards=("air_yards", "sum"),
    ).reset_index()

    rec_agg = targets.groupby(["receiver_player_id", "receiver_player_name", "posteam"]).agg(
        targets=("play_id", "count"),
        receptions=("complete_pass", "sum"),
        receiving_yards=("yards_gained", lambda s: s[targets.loc[s.index, "complete_pass"] == 1].sum()),
        receiving_tds=("pass_touchdown", "sum"),
        yac=("yards_after_catch", "sum"),
        air_yards=("air_yards", "sum"),
        total_epa=("epa", "sum"),
        epa_per_target=("epa", "mean"),
        success_rate=("epa", lambda s: (s > 0).mean()),
    ).reset_index()

    rec_agg = rec_agg.merge(team_totals, on="posteam", how="left")
    rec_agg["target_share"] = rec_agg["targets"] / rec_agg["team_targets"]
    rec_agg["air_yards_share"] = rec_agg["air_yards"] / rec_agg["team_air_yards"]
    rec_agg["wopr"] = 1.5 * rec_agg["target_share"] + 0.7 * rec_agg["air_yards_share"]
    rec_agg["catch_rate"] = rec_agg["receptions"] / rec_agg["targets"]
    rec_agg["yards_per_target"] = rec_agg["receiving_yards"] / rec_agg["targets"]
    rec_agg["yac_per_reception"] = rec_agg["yac"] / rec_agg["receptions"].replace(0, np.nan)

    if players_df is not None:
        rec_agg = rec_agg.merge(
            players_df[["player_id", "player_name", "position", "college"]],
            left_on="receiver_player_id",
            right_on="player_id",
            how="left",
        )

    filtered = rec_agg[rec_agg["targets"] >= min_targets].copy()
    return filtered.sort_values("wopr", ascending=False).reset_index(drop=True)


def build_nfl_player_feature_store(
    pbp_path: Union[str, Path],
    players_path: Optional[Union[str, Path]] = None,
    output_dir: Union[str, Path] = "data/processed/nfl",
) -> Dict[str, pd.DataFrame]:
    """Execute end-to-end player feature pipelines and persist feature store parquets.

    Parameters
    ----------
    pbp_path : str or Path
        Path to raw play-by-play data.
    players_path : str or Path, optional
        Path to dim_players.parquet.
    output_dir : str or Path, default="data/processed/nfl"
        Target directory to save feature tables.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary holding the generated dataframes: 'qb', 'rb', and 'wr'.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading raw PBP from {pbp_path}...")
    pbp_df, players_df = load_clean_pbp(pbp_path, players_path)

    print("Computing QB efficiency features...")
    qb_features = compute_qb_efficiency_features(pbp_df, players_df)
    qb_out_path = out_dir / "features_qb_efficiency_2023.parquet"
    qb_features.to_parquet(qb_out_path, index=False)
    print(f"-> Saved QB feature store: {qb_out_path} ({len(qb_features)} records)")

    print("Computing RB efficiency features...")
    rb_features = compute_rb_efficiency_features(pbp_df, players_df)
    rb_out_path = out_dir / "features_rb_efficiency_2023.parquet"
    rb_features.to_parquet(rb_out_path, index=False)
    print(f"-> Saved RB feature store: {rb_out_path} ({len(rb_features)} records)")

    print("Computing WR/TE efficiency features...")
    wr_features = compute_receiver_efficiency_features(pbp_df, players_df)
    wr_out_path = out_dir / "features_wr_efficiency_2023.parquet"
    wr_features.to_parquet(wr_out_path, index=False)
    print(f"-> Saved WR/TE feature store: {wr_out_path} ({len(wr_features)} records)")

    return {
        "qb": qb_features,
        "rb": rb_features,
        "wr": wr_features,
    }


def main():
    """CLI execution entrypoint for player feature store generation."""
    data_dir = Path("data")
    pbp_path = data_dir / "raw" / "nfl" / "nfl_pbp_2023.parquet"
    players_path = data_dir / "processed" / "nfl" / "dim_players.parquet"
    output_dir = data_dir / "processed" / "nfl"

    if not pbp_path.exists():
        raise FileNotFoundError(f"Raw PBP file not found at {pbp_path}")

    build_nfl_player_feature_store(
        pbp_path=pbp_path,
        players_path=players_path,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
