"""
Player Similarity Engine Module.

Day 28 of the 100 Days of Data & Sports Challenge (Phase 2: EDA & Feature Engineering).
Provides vector-space modeling and cosine-similarity searching for finding statistical
twins and replacement archetypes across football per-90 metrics.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances


DEFAULT_STYLE_FEATURES = [
    "passes_completed_p90",
    "progressive_passes_p90",
    "key_passes_p90",
    "shots_p90",
    "xg_p90",
    "dribbles_completed_p90",
    "ball_recoveries_p90",
    "interceptions_p90",
    "pressures_p90"
]


class PlayerSimilarityEngine:
    """
    Engine for finding player archetypes and tactical replacements using vector similarity.

    Attributes:
        feature_cols (List[str]): Quantitative per-90 metrics used for style profiling.
        scaler_type (str): 'standard' for StandardScaler or 'robust' for RobustScaler.
        scaler: Fitted scikit-learn scaler instance.
        df (pd.DataFrame): Candidate player dataset with per-90 metrics.
        X_scaled (np.ndarray): Normalized multidimensional feature matrix.
    """

    def __init__(
        self,
        feature_cols: Optional[List[str]] = None,
        scaler_type: str = "standard"
    ) -> None:
        """
        Initializes the similarity engine with specified feature space and scaler.

        Args:
            feature_cols: List of column names to construct the player style vector.
            scaler_type: 'standard' (zero mean, unit variance) or 'robust' (median/IQR).
        """
        self.feature_cols = feature_cols or DEFAULT_STYLE_FEATURES
        self.scaler_type = scaler_type.lower()
        if self.scaler_type == "robust":
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()

        self.df: Optional[pd.DataFrame] = None
        self.X_scaled: Optional[np.ndarray] = None
        self._is_fitted: bool = False

    def fit(self, df: pd.DataFrame) -> "PlayerSimilarityEngine":
        """
        Fits the scaler on the candidate player feature space.

        Args:
            df: DataFrame containing player records and per-90 metric columns.

        Returns:
            Fitted PlayerSimilarityEngine instance.
        """
        missing_cols = [c for c in self.feature_cols if c not in df.columns]
        if missing_cols:
            raise KeyError(f"Missing required style feature columns: {missing_cols}")

        self.df = df.copy().reset_index(drop=True)
        raw_matrix = self.df[self.feature_cols].values
        self.X_scaled = self.scaler.fit_transform(raw_matrix)
        self._is_fitted = True
        return self

    def find_similar(
        self,
        player_name: str,
        top_n: int = 5,
        same_position: bool = True,
        min_minutes: float = 200.0
    ) -> pd.DataFrame:
        """
        Identifies the top N most similar players based on Cosine Similarity.

        Args:
            player_name: Full or partial string name of the target player.
            top_n: Number of similar players to return.
            same_position: If True, restrains candidate pool to target's position_group.
            min_minutes: Minimum total minutes played required for candidate eligibility.

        Returns:
            DataFrame of top N similar players with similarity scores and key metrics.
        """
        if not self._is_fitted or self.df is None or self.X_scaled is None:
            raise RuntimeError("Engine must be fitted before running similarity searches.")

        # Locate target player
        matches = self.df[self.df["player_name"].str.contains(player_name, case=False, na=False)]
        if matches.empty:
            raise ValueError(f"Player matching '{player_name}' not found in candidate pool.")

        target_idx = matches.index[0]
        target_row = self.df.loc[target_idx]
        target_vec = self.X_scaled[target_idx].reshape(1, -1)

        # Build candidate mask
        mask = (self.df["total_minutes_played"] >= min_minutes) & (self.df["player_id"] != target_row["player_id"])
        if same_position and "position_group" in self.df.columns:
            mask = mask & (self.df["position_group"] == target_row["position_group"])

        candidate_indices = self.df[mask].index
        if len(candidate_indices) == 0:
            raise ValueError("No eligible candidates found matching the specified filters.")

        candidate_vecs = self.X_scaled[candidate_indices]
        sim_scores = cosine_similarity(target_vec, candidate_vecs)[0]

        candidates_df = self.df.loc[candidate_indices].copy()
        candidates_df["similarity_score"] = np.round(sim_scores * 100.0, 1)

        results = candidates_df.sort_values("similarity_score", ascending=False).head(top_n)

        display_cols = [
            "player_name", "team_name", "primary_position", "position_group",
            "total_minutes_played", "similarity_score"
        ] + [c for c in self.feature_cols if c in candidates_df.columns]

        return results[display_cols].reset_index(drop=True)

    def compare_players(
        self,
        player_a_name: str,
        player_b_name: str
    ) -> pd.DataFrame:
        """
        Computes the pairwise cosine similarity and head-to-head normalized profile.

        Args:
            player_a_name: Name of player A.
            player_b_name: Name of player B.

        Returns:
            DataFrame comparing raw and standardized feature values side by side.
        """
        if not self._is_fitted or self.df is None or self.X_scaled is None:
            raise RuntimeError("Engine must be fitted before running comparisons.")

        matches_a = self.df[self.df["player_name"].str.contains(player_a_name, case=False, na=False)]
        matches_b = self.df[self.df["player_name"].str.contains(player_b_name, case=False, na=False)]

        if matches_a.empty or matches_b.empty:
            raise ValueError("Both players must exist in the fitted dataset.")

        idx_a = matches_a.index[0]
        idx_b = matches_b.index[0]

        row_a = self.df.loc[idx_a]
        row_b = self.df.loc[idx_b]

        vec_a = self.X_scaled[idx_a].reshape(1, -1)
        vec_b = self.X_scaled[idx_b].reshape(1, -1)

        pair_similarity = cosine_similarity(vec_a, vec_b)[0, 0] * 100.0

        comparison = pd.DataFrame({
            "metric": self.feature_cols,
            f"{row_a['player_name']} (raw)": [row_a[c] for c in self.feature_cols],
            f"{row_b['player_name']} (raw)": [row_b[c] for c in self.feature_cols],
            f"{row_a['player_name']} (scaled)": vec_a[0],
            f"{row_b['player_name']} (scaled)": vec_b[0]
        })
        comparison.attrs["similarity_score"] = np.round(pair_similarity, 1)
        return comparison


def main() -> None:
    """Demonstrates standalone functionality of PlayerSimilarityEngine."""
    data_path = Path("data/processed/football/player_per90_metrics.parquet")
    if not data_path.exists():
        print(f"Data file not found at {data_path}. Run Day 27 pipeline first.")
        return

    df = pd.read_parquet(data_path)
    engine = PlayerSimilarityEngine(scaler_type="standard")
    engine.fit(df)

    print("=== Demo 1: Searching similar midfielders to Luka Modrić ===")
    results_modric = engine.find_similar("Luka Modrić", top_n=5, same_position=True)
    print(results_modric[["player_name", "team_name", "similarity_score", "passes_completed_p90", "xg_p90"]])

    print("\n=== Demo 2: Searching similar attackers to Kylian Mbappé ===")
    results_mbappe = engine.find_similar("Kylian Mbappé", top_n=5, same_position=False)
    print(results_mbappe[["player_name", "team_name", "primary_position", "similarity_score", "xg_p90", "dribbles_completed_p90"]])


if __name__ == "__main__":
    main()
