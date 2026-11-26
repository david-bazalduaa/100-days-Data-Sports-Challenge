"""Day 31: Draft Success Features.

Modular feature engineering pipeline transforming cached NFL Combine and
Draft Picks datasets into a leakage-free Draft Feature Store.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

POS_MAPPING: dict[str, str] = {
    "WR": "WR", "RB": "RB", "FB": "RB", "QB": "QB", "TE": "TE",
    "OT": "OL", "OG": "OL", "C": "OL", "G": "OL", "T": "OL", "OL": "OL",
    "DE": "EDGE", "EDGE": "EDGE", "OLB": "EDGE",
    "DT": "DL", "NT": "DL", "DL": "DL",
    "LB": "LB", "ILB": "LB", "MLB": "LB",
    "CB": "DB", "S": "DB", "FS": "DB", "SS": "DB", "SAF": "DB", "DB": "DB",
    "K": "ST", "P": "ST", "LS": "ST",
}

CONFERENCE_MAP: dict[str, str] = {
    "Alabama": "SEC", "LSU": "SEC", "Georgia": "SEC", "Florida": "SEC",
    "Texas A&M": "SEC", "Auburn": "SEC", "South Carolina": "SEC", "Arkansas": "SEC",
    "Mississippi St.": "SEC", "Mississippi": "SEC", "Tennessee": "SEC", "Missouri": "SEC",
    "Kentucky": "SEC", "Vanderbilt": "SEC",
    "Ohio St.": "Big Ten", "Michigan": "Big Ten", "Penn St.": "Big Ten", "Iowa": "Big Ten",
    "Wisconsin": "Big Ten", "Michigan St.": "Big Ten", "Nebraska": "Big Ten", "Illinois": "Big Ten",
    "Minnesota": "Big Ten", "Northwestern": "Big Ten", "Purdue": "Big Ten", "Indiana": "Big Ten",
    "Maryland": "Big Ten", "Rutgers": "Big Ten",
    "Clemson": "ACC", "Miami (FL)": "ACC", "Florida St.": "ACC", "North Carolina": "ACC",
    "Virginia Tech": "ACC", "Pittsburgh": "ACC", "North Carolina St.": "ACC", "Louisville": "ACC",
    "Virginia": "ACC", "Boston Col.": "ACC", "Georgia Tech": "ACC", "Wake Forest": "ACC",
    "Syracuse": "ACC", "Duke": "ACC",
    "Oklahoma": "Big 12", "Texas": "Big 12", "TCU": "Big 12", "Baylor": "Big 12",
    "West Virginia": "Big 12", "Oklahoma St.": "Big 12", "Kansas St.": "Big 12",
    "Texas Tech": "Big 12", "Iowa St.": "Big 12", "Kansas": "Big 12",
    "USC": "Pac-12", "Stanford": "Pac-12", "UCLA": "Pac-12", "Oregon": "Pac-12",
    "Washington": "Pac-12", "Utah": "Pac-12", "California": "Pac-12", "Arizona St.": "Pac-12",
    "Arizona": "Pac-12", "Colorado": "Pac-12", "Oregon St.": "Pac-12", "Washington St.": "Pac-12",
    "Notre Dame": "Notre Dame",
}

REALIGNMENT_OVERRIDES: list[tuple[str, int, str]] = [
    ("Nebraska", 2011, "Big 12"),
    ("Colorado", 2011, "Big 12"),
    ("Utah", 2011, "G5 / FCS / Other"),
    ("Texas A&M", 2012, "Big 12"),
    ("Missouri", 2012, "Big 12"),
    ("TCU", 2012, "G5 / FCS / Other"),
    ("West Virginia", 2012, "G5 / FCS / Other"),
    ("Pittsburgh", 2013, "G5 / FCS / Other"),
    ("Syracuse", 2013, "G5 / FCS / Other"),
    ("Louisville", 2014, "G5 / FCS / Other"),
    ("Maryland", 2014, "ACC"),
    ("Rutgers", 2014, "G5 / FCS / Other"),
]


def parse_height_to_inches(ht_val: object) -> float:
    """Convert 'ft-in' string (e.g. '6-3') to total inches."""
    if pd.isna(ht_val):
        return np.nan
    s = str(ht_val).strip()
    if "-" in s:
        parts = s.split("-")
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            return float(int(parts[0]) * 12 + int(parts[1]))
    return np.nan


def build_draft_features(draft_df: pd.DataFrame, combine_df: pd.DataFrame) -> pd.DataFrame:
    """Build leakage-free draft pick feature store from draft_picks and combine."""
    combine_cols = [
        "pfr_id", "cfb_id", "ht", "wt", "forty", "bench",
        "vertical", "broad_jump", "cone", "shuttle",
    ]
    combine_pfr = (
        combine_df[combine_cols]
        .dropna(subset=["pfr_id"])
        .drop_duplicates(subset=["pfr_id"], keep="first")
    )
    combine_cfb = (
        combine_df[combine_cols]
        .dropna(subset=["cfb_id"])
        .drop_duplicates(subset=["cfb_id"], keep="first")
    )

    df = draft_df.merge(combine_pfr, left_on="pfr_player_id", right_on="pfr_id", how="left")
    unmatched_mask = df["pfr_id"].isna() & df["cfb_player_id"].notna()
    if unmatched_mask.any():
        fallback = df.loc[unmatched_mask, ["cfb_player_id"]].merge(
            combine_cfb, left_on="cfb_player_id", right_on="cfb_id", how="left"
        )
        for col in ["ht", "wt", "forty", "bench", "vertical", "broad_jump", "cone", "shuttle"]:
            df.loc[unmatched_mask, col] = fallback[col].values

    drill_raw_cols = ["ht", "wt", "forty", "vertical", "broad_jump", "bench", "cone", "shuttle"]
    df["has_combine_data"] = df[drill_raw_cols].notna().any(axis=1).astype(int)
    df["pos_group"] = df["position"].map(POS_MAPPING).fillna("OTHER")

    # Cohort-adjusted target (capped at 9-year w_av saturation)
    max_observed_season = int(df["to"].max())
    df["eligible_seasons"] = (max_observed_season - df["season"] + 1).clip(lower=1, upper=9)
    df["w_av_clean"] = df["w_av"].fillna(0.0)
    df["target_wav_per_yr"] = df["w_av_clean"] / df["eligible_seasons"]

    # Positional normalized Combine metrics
    df["ht_inches"] = df["ht"].apply(parse_height_to_inches)
    df["speed_score"] = np.where(
        df["wt"].notna() & df["forty"].notna() & (df["forty"] > 0),
        (df["wt"] * 200.0) / (df["forty"] ** 4),
        np.nan,
    )
    higher_is_better = ["ht_inches", "wt", "vertical", "broad_jump", "bench", "speed_score"]
    lower_is_better = ["forty", "cone", "shuttle"]
    z_cols = []
    for col in higher_is_better + lower_is_better:
        z_col = f"{col}_z"
        grp = df.groupby("pos_group")[col]
        z_vals = (df[col] - grp.transform("mean")) / grp.transform("std").replace(0, np.nan)
        if col in lower_is_better:
            z_vals = -z_vals
        df[z_col] = z_vals
        z_cols.append(z_col)

    core_drill_z = ["forty_z", "vertical_z", "broad_jump_z", "bench_z", "cone_z", "shuttle_z"]
    df["combine_drills_completed"] = df[core_drill_z].notna().sum(axis=1)
    df["explosiveness_z"] = df[["vertical_z", "broad_jump_z"]].mean(axis=1)
    df["agility_z"] = df[["cone_z", "shuttle_z"]].mean(axis=1)
    df["athletic_composite_z"] = df[core_drill_z].mean(axis=1)
    all_z = z_cols + ["explosiveness_z", "agility_z", "athletic_composite_z"]
    df[all_z] = df[all_z].fillna(0.0)

    # Time-aware conference pedigree & Empirical Bayes prior draft rate
    df["conference_group"] = df["college"].map(CONFERENCE_MAP).fillna("G5 / FCS / Other")
    for school, join_yr, old_conf in REALIGNMENT_OVERRIDES:
        df.loc[(df["college"] == school) & (df["season"] < join_yr), "conference_group"] = old_conf

    df["is_power5"] = (df["conference_group"] != "G5 / FCS / Other").astype(int)
    strength_map = {
        "SEC": 3, "Big Ten": 3,
        "ACC": 2, "Big 12": 2, "Pac-12": 2, "Notre Dame": 2,
        "G5 / FCS / Other": 1,
    }
    df["conference_strength"] = df["conference_group"].map(strength_map)

    school_yr = df.groupby(["college", "season"]).size().rename("yr_picks").reset_index()
    grid = pd.MultiIndex.from_product(
        [df["college"].unique(), sorted(df["season"].unique())],
        names=["college", "season"],
    ).to_frame(index=False)
    grid = grid.merge(school_yr, on=["college", "season"], how="left").fillna({"yr_picks": 0})
    grid = grid.sort_values(["college", "season"])
    grid["cum_prior_picks"] = grid.groupby("college")["yr_picks"].cumsum() - grid["yr_picks"]
    grid["prior_years"] = grid["season"] - int(df["season"].min())
    df = df.merge(grid[["college", "season", "cum_prior_picks", "prior_years"]], on=["college", "season"], how="left")

    prior_weight = 2.0
    prior_rate = np.where(df["is_power5"] == 1, 3.0, 0.75)
    df["school_prior_picks_rate"] = (
        (df["cum_prior_picks"] + prior_weight * prior_rate) / (df["prior_years"] + prior_weight)
    )
    df["school_tier"] = pd.cut(
        df["school_prior_picks_rate"],
        bins=[-np.inf, 1.5, 4.0, np.inf],
        labels=["Developmental / Small", "Mid-Tier Pipeline", "NFL Factory"],
    )

    # Non-linear draft capital & age
    df["log_pick"] = np.log(df["pick"])
    df["draft_value_curve"] = df["pick"] ** (-0.40)
    df["draft_day"] = np.select(
        [df["round"] == 1, df["round"].isin([2, 3])],
        ["Day 1 (R1)", "Day 2 (R2-R3)"],
        default="Day 3 (R4-R7)",
    )
    df["age_at_draft"] = df["age"].fillna(df.groupby("pos_group")["age"].transform("median"))
    df["is_young_prospect"] = (df["age_at_draft"] <= 21).astype(int)

    id_meta = ["season", "pfr_player_id", "cfb_player_id", "pfr_player_name", "team", "position", "pos_group", "college"]
    features = [
        "pick", "round", "log_pick", "draft_value_curve", "age_at_draft", "is_young_prospect",
        "ht_inches_z", "wt_z", "forty_z", "vertical_z", "broad_jump_z",
        "bench_z", "cone_z", "shuttle_z", "speed_score_z",
        "explosiveness_z", "agility_z", "athletic_composite_z",
        "has_combine_data", "combine_drills_completed",
        "is_power5", "conference_strength", "school_prior_picks_rate",
        "conference_group", "school_tier", "draft_day",
    ]
    targets = ["w_av_clean", "eligible_seasons", "target_wav_per_yr"]
    return df[id_meta + features + targets].copy()


def main() -> None:
    """Load raw NFL draft/combine datasets and persist the draft feature store."""
    repo_root = Path(__file__).resolve().parents[2]
    raw_dir = repo_root / "data" / "raw" / "nfl"
    out_path = repo_root / "data" / "features" / "nfl" / "draft_features.parquet"

    combine_df = pd.read_parquet(raw_dir / "combine_2010_2023.parquet")
    draft_df = pd.read_parquet(raw_dir / "draft_picks_2010_2023.parquet")

    feature_store = build_draft_features(draft_df, combine_df)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    feature_store.to_parquet(out_path, index=False)
    print(f"Saved {len(feature_store):,} rows x {feature_store.shape[1]} cols -> {out_path}")


if __name__ == "__main__":
    main()
