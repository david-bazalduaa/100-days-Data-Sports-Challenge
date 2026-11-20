"""Temporal and Rolling Features in Sports Feature Engineering Module.

Day 22: Rolling and Temporal Features in Sports.
Phase 2: EDA & Feature Engineering.

This module provides functional pipelines to compute anti-leakage temporal features
across sports entities (teams or players) over chronological match sequences:
- Multi-window rolling averages (e.g. 3, 5, 10 games).
- Form and momentum indicators (short-term vs. long-term differential).
- Exponentially weighted moving averages (EWMA with decay).
- Expanding cumulative means across prior games.
- Built-in mathematical anti-leakage perturbation testing.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd


def compute_rolling_features(
    df: pd.DataFrame,
    entity_col: str,
    time_cols: List[str],
    metric_cols: List[str],
    windows: List[int] = [3, 5, 10],
    min_periods: int = 1,
) -> pd.DataFrame:
    """Compute anti-leakage multi-window rolling averages for grouped sports entities.

    Enforces strict chronological sorting and mandatory .shift(1) via grouped transform
    so that game t only consumes information from games t-1, t-2, ...

    Parameters
    ----------
    df : pd.DataFrame
        Input entity-match observations.
    entity_col : str
        Column identifying the grouping entity (e.g., 'posteam' or 'player_id').
    time_cols : List[str]
        Chronological ordering columns (e.g., ['season', 'week'] or ['match_date']).
    metric_cols : List[str]
        List of numeric metrics to calculate rolling averages for.
    windows : List[int], default=[3, 5, 10]
        List of rolling window sizes.
    min_periods : int, default=1
        Minimum number of observations in window required to produce a value.

    Returns
    -------
    pd.DataFrame
        Dataframe enriched with rolling feature columns named `{metric}_roll_{window}`.
    """
    sort_cols = [entity_col] + time_cols
    res = df.sort_values(sort_cols).copy()

    for metric in metric_cols:
        for w in windows:
            col_name = f"{metric}_roll_{w}"
            res[col_name] = res.groupby(entity_col)[metric].transform(
                lambda s, win=w: s.shift(1).rolling(window=win, min_periods=min_periods).mean()
            )

    return res


def compute_ewma_features(
    df: pd.DataFrame,
    entity_col: str,
    time_cols: List[str],
    metric_cols: List[str],
    spans: List[int] = [3, 5],
    min_periods: int = 1,
) -> pd.DataFrame:
    """Compute exponentially weighted moving averages (EWMA) with recency decay.

    Parameters
    ----------
    df : pd.DataFrame
        Input observations.
    entity_col : str
        Entity identifier column.
    time_cols : List[str]
        Chronological ordering columns.
    metric_cols : List[str]
        Numeric metrics to decay-weight.
    spans : List[int], default=[3, 5]
        Decay span parameters.
    min_periods : int, default=1
        Minimum observations required.

    Returns
    -------
    pd.DataFrame
        Dataframe enriched with EWMA columns named `{metric}_ewma_span{span}`.
    """
    sort_cols = [entity_col] + time_cols
    res = df.sort_values(sort_cols).copy()

    for metric in metric_cols:
        for span in spans:
            col_name = f"{metric}_ewma_span{span}"
            res[col_name] = res.groupby(entity_col)[metric].transform(
                lambda s, sp=span: s.shift(1).ewm(span=sp, min_periods=min_periods).mean()
            )

    return res


def compute_expanding_features(
    df: pd.DataFrame,
    entity_col: str,
    time_cols: List[str],
    metric_cols: List[str],
    min_periods: int = 1,
) -> pd.DataFrame:
    """Compute cumulative expanding means across all prior season matches.

    Parameters
    ----------
    df : pd.DataFrame
        Input observations.
    entity_col : str
        Entity identifier column.
    time_cols : List[str]
        Chronological ordering columns.
    metric_cols : List[str]
        Numeric metrics to accumulate.
    min_periods : int, default=1
        Minimum observations required.

    Returns
    -------
    pd.DataFrame
        Dataframe enriched with expanding columns named `{metric}_expanding`.
    """
    sort_cols = [entity_col] + time_cols
    res = df.sort_values(sort_cols).copy()

    for metric in metric_cols:
        col_name = f"{metric}_expanding"
        res[col_name] = res.groupby(entity_col)[metric].transform(
            lambda s: s.shift(1).expanding(min_periods=min_periods).mean()
        )

    return res


def compute_momentum_features(
    df: pd.DataFrame,
    metric_cols: List[str],
    short_window: int = 3,
    long_window: int = 10,
) -> pd.DataFrame:
    """Calculate form and momentum differential (short-term minus long-term baseline).

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe already containing rolling features.
    metric_cols : List[str]
        Metric base names.
    short_window : int, default=3
        Fast reaction window size.
    long_window : int, default=10
        Baseline window size.

    Returns
    -------
    pd.DataFrame
        Dataframe enriched with momentum columns named `{metric}_momentum_{short}_{long}`.
    """
    res = df.copy()
    for metric in metric_cols:
        short_col = f"{metric}_roll_{short_window}"
        long_col = f"{metric}_roll_{long_window}"
        if short_col in res.columns and long_col in res.columns:
            momentum_col = f"{metric}_momentum_{short_window}_{long_window}"
            res[momentum_col] = res[short_col] - res[long_col]

    return res


def build_temporal_feature_pipeline(
    df: pd.DataFrame,
    entity_col: str,
    time_cols: List[str],
    metric_cols: List[str],
    windows: List[int] = [3, 5, 10],
    spans: List[int] = [3, 5],
    min_periods: int = 1,
) -> pd.DataFrame:
    """End-to-end execution of all temporal features with anti-leakage guarantees.

    Parameters
    ----------
    df : pd.DataFrame
        Input observations.
    entity_col : str
        Entity identifier column.
    time_cols : List[str]
        Chronological ordering columns.
    metric_cols : List[str]
        Numeric metrics.
    windows : List[int], default=[3, 5, 10]
        Rolling window sizes.
    spans : List[int], default=[3, 5]
        EWMA span parameters.
    min_periods : int, default=1
        Minimum observations required.

    Returns
    -------
    pd.DataFrame
        Fully enriched temporal feature dataframe.
    """
    res = compute_rolling_features(
        df=df,
        entity_col=entity_col,
        time_cols=time_cols,
        metric_cols=metric_cols,
        windows=windows,
        min_periods=min_periods,
    )

    res = compute_ewma_features(
        df=res,
        entity_col=entity_col,
        time_cols=time_cols,
        metric_cols=metric_cols,
        spans=spans,
        min_periods=min_periods,
    )

    res = compute_expanding_features(
        df=res,
        entity_col=entity_col,
        time_cols=time_cols,
        metric_cols=metric_cols,
        min_periods=min_periods,
    )

    res = compute_momentum_features(
        df=res,
        metric_cols=metric_cols,
        short_window=windows[0],
        long_window=windows[-1],
    )

    return res


def validate_anti_leakage(
    df: pd.DataFrame,
    entity_col: str,
    time_cols: List[str],
    metric_col: str,
    test_entity: str,
    test_time_val: Union[int, str],
    perturbed_value: float = 999.0,
    window: int = 3,
) -> bool:
    """Empirical perturbation test verifying zero future data leakage.

    Alters a future observation and mathematically validates that prior
    and current features remain 100% invariant.

    Parameters
    ----------
    df : pd.DataFrame
        Input baseline dataframe.
    entity_col : str
        Entity identifier column.
    time_cols : List[str]
        Chronological ordering columns (primary time index assumed as time_cols[-1]).
    metric_col : str
        Target metric to perturb.
    test_entity : str
        Specific entity to run the test on.
    test_time_val : int or str
        Time value at which perturbation is injected.
    perturbed_value : float, default=999.0
        Arbitrary extreme value to inject.
    window : int, default=3
        Window size to test.

    Returns
    -------
    bool
        True if all prior features are strictly identical (zero leakage).
    """
    time_key = time_cols[-1]

    # Baseline calculation
    base_df = compute_rolling_features(
        df=df,
        entity_col=entity_col,
        time_cols=time_cols,
        metric_cols=[metric_col],
        windows=[window],
        min_periods=1,
    )

    # Injected perturbed calculation
    perturbed_raw = df.copy()
    mask = (perturbed_raw[entity_col] == test_entity) & (perturbed_raw[time_key] == test_time_val)
    if not mask.any():
        raise ValueError(f"No matching row found for {entity_col}={test_entity} and {time_key}={test_time_val}")

    perturbed_raw.loc[mask, metric_col] = perturbed_value

    perturbed_features = compute_rolling_features(
        df=perturbed_raw,
        entity_col=entity_col,
        time_cols=time_cols,
        metric_cols=[metric_col],
        windows=[window],
        min_periods=1,
    )

    roll_col = f"{metric_col}_roll_{window}"

    # Extract prior records
    base_prior = base_df[
        (base_df[entity_col] == test_entity) & (base_df[time_key] <= test_time_val)
    ][roll_col].reset_index(drop=True)

    perturbed_prior = perturbed_features[
        (perturbed_features[entity_col] == test_entity) & (perturbed_features[time_key] <= test_time_val)
    ][roll_col].reset_index(drop=True)

    # Validate that prior features did not change
    pd.testing.assert_series_equal(base_prior, perturbed_prior, check_exact=True)

    # Validate that subsequent record DID change
    subsequent_mask = (base_df[entity_col] == test_entity) & (base_df[time_key] > test_time_val)
    if subsequent_mask.any():
        base_post = base_df.loc[subsequent_mask, roll_col].iloc[0]
        pert_post = perturbed_features.loc[subsequent_mask, roll_col].iloc[0]
        assert base_post != pert_post, "Perturbed future value failed to propagate forward!"

    return True


def main():
    """CLI execution entrypoint for team temporal feature store generation."""
    data_dir = Path("data")
    raw_pbp_path = data_dir / "raw" / "nfl" / "nfl_pbp_2023.parquet"
    output_dir = data_dir / "processed" / "nfl"

    if not raw_pbp_path.exists():
        raise FileNotFoundError(f"Raw PBP file not found at {raw_pbp_path}")

    print("Building team-game base observations...")
    raw_pbp = pd.read_parquet(
        raw_pbp_path,
        columns=["play_id", "game_id", "season", "week", "posteam", "play_type", "epa", "posteam_score_post"],
    )
    clean_plays = raw_pbp[raw_pbp["posteam"].notna() & raw_pbp["epa"].notna()].copy()

    team_weekly = clean_plays.groupby(["posteam", "season", "week", "game_id"]).agg(
        total_plays=("play_id", "count"),
        mean_epa=("epa", "mean"),
        pass_epa=("epa", lambda s: s[clean_plays.loc[s.index, "play_type"] == "pass"].mean()),
        rush_epa=("epa", lambda s: s[clean_plays.loc[s.index, "play_type"] == "run"].mean()),
        success_rate=("epa", lambda s: (s > 0).mean()),
        points_scored=("posteam_score_post", "max"),
    ).reset_index()

    print("Executing temporal feature pipeline...")
    temporal_df = build_temporal_feature_pipeline(
        df=team_weekly,
        entity_col="posteam",
        time_cols=["season", "week"],
        metric_cols=["mean_epa", "pass_epa", "rush_epa", "points_scored", "success_rate"],
        windows=[3, 5, 10],
        spans=[3, 5],
        min_periods=1,
    )

    print("Running anti-leakage validation test...")
    is_valid = validate_anti_leakage(
        df=team_weekly,
        entity_col="posteam",
        time_cols=["season", "week"],
        metric_col="mean_epa",
        test_entity="SF",
        test_time_val=10,
        perturbed_value=999.0,
        window=3,
    )
    print(f"Anti-leakage validation passed: {is_valid}")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "features_team_temporal_2023.parquet"
    temporal_df.to_parquet(out_path, index=False)
    print(f"-> Saved temporal feature store to {out_path} ({len(temporal_df)} records x {temporal_df.shape[1]} cols)")


if __name__ == "__main__":
    main()
