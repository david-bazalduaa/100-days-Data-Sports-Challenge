"""
Player Similarity Engine Module.

Day 28-29 of the 100 Days of Data & Sports Challenge (Phase 2: EDA & Feature Engineering).
Provides vector-space modeling and cosine-similarity searching for finding statistical
twins and replacement archetypes across football per-90 metrics. Supports programmatic
import as well as command-line interface (CLI) execution with automatic plot generation.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

        # Locate target player (case-insensitive substring match)
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

        res = results[display_cols].reset_index(drop=True)
        res.attrs["target_player"] = target_row["player_name"]
        res.attrs["target_team"] = target_row["team_name"]
        return res

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

        if matches_a.empty:
            raise ValueError(f"Player A matching '{player_a_name}' not found.")
        if matches_b.empty:
            raise ValueError(f"Player B matching '{player_b_name}' not found.")

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
        comparison.attrs["player_a"] = row_a["player_name"]
        comparison.attrs["player_b"] = row_b["player_name"]
        return comparison


def plot_similarity_results(
    results_df: pd.DataFrame,
    target_player: str,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Renders an editorial horizontal bar chart of similarity scores.

    Args:
        results_df: DataFrame returned by find_similar().
        target_player: Name of the player being searched.
        save_path: Optional file path to persist figure.

    Returns:
        Matplotlib Figure instance.
    """
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    y_labels = [f"{r['player_name']} ({r['team_name']})" for _, r in results_df.iterrows()][::-1]
    scores = results_df["similarity_score"].values[::-1]
    colors = sns.color_palette("viridis", len(scores))

    bars = ax.barh(y_labels, scores, color=colors, edgecolor="white", linewidth=0.8, height=0.6)
    ax.set_xlim(65, 100)
    ax.set_xlabel("Cosine Similarity Match (%)", fontsize=11, fontweight="bold", color="white")
    ax.set_title(
        f"{target_player}: Top Tactical Replacement Matches\n(Multidimensional Style Space Alignment)",
        fontsize=13, fontweight="bold", color="white", pad=12
    )
    ax.grid(axis="x", alpha=0.2, linestyle="--")

    for bar, score in zip(bars, scores):
        ax.text(score + 0.6, bar.get_y() + bar.get_height() / 2, f"{score:.1f}%", va="center", fontsize=10.5, fontweight="bold", color="white")

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, facecolor="#0e1117", bbox_inches="tight")
        print(f"Saved similarity plot to: {save_path}")
    return fig


def plot_head_to_head(
    comp_df: pd.DataFrame,
    player_a: str,
    player_b: str,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plots a dual horizontal diverging bar chart of standardized style metrics.

    Args:
        comp_df: DataFrame returned by compare_players().
        player_a: Target player name.
        player_b: Clone match player name.
        save_path: Optional file path to persist figure.

    Returns:
        Matplotlib Figure instance.
    """
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(11, 6.2))
    fig.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    metrics = comp_df["metric"].str.replace("_p90", "").str.replace("_", " ").str.title().values
    vals_a = comp_df[f"{player_a} (scaled)"].values
    vals_b = comp_df[f"{player_b} (scaled)"].values

    y_pos = np.arange(len(metrics))
    bar_width = 0.36

    ax.barh(y_pos - bar_width / 2, vals_a, height=bar_width, color="#feca57", label=f"{player_a} (Target)", edgecolor="black", linewidth=0.8)
    ax.barh(y_pos + bar_width / 2, vals_b, height=bar_width, color="#54a0ff", label=f"{player_b} (Match)", edgecolor="black", linewidth=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(metrics, fontsize=10, color="white")
    ax.axvline(0, color="#666666", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Standardized Metric Value (Z-Score)", fontsize=11, fontweight="bold", color="white")
    sim = comp_df.attrs.get("similarity_score", 0.0)
    ax.set_title(
        f"Head-to-Head Tactical Style Alignment\n{player_a} vs. {player_b} (Match Score: {sim:.1f}%)",
        fontsize=13, fontweight="bold", color="white", pad=12
    )
    ax.legend(loc="lower right", facecolor="#1a1f2c", edgecolor="#444444", fontsize=9.5)
    ax.grid(axis="x", alpha=0.2, linestyle="--")

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, facecolor="#0e1117", bbox_inches="tight")
        print(f"Saved head-to-head plot to: {save_path}")
    return fig


def main() -> None:
    """CLI execution entrypoint for PlayerSimilarityEngine."""
    parser = argparse.ArgumentParser(description="FootballScout: Player Similarity Search Engine")
    parser.add_argument("--player", type=str, default="Luka Modrić", help="Target player name to search")
    parser.add_argument("--top-n", type=int, default=5, help="Number of similar players to return")
    parser.add_argument("--same-position", action="store_true", default=True, help="Filter to same position group")
    parser.add_argument("--any-position", action="store_false", dest="same_position", help="Search across all outfield positions")
    parser.add_argument("--min-minutes", type=float, default=200.0, help="Minimum tournament minutes played")
    parser.add_argument("--scaler", type=str, default="standard", choices=["standard", "robust"], help="Feature scaling method")
    parser.add_argument("--save-plots", action="store_true", help="Generate and save PNG figures to reports/figures/")
    args = parser.parse_args()

    data_path = Path("data/processed/football/player_per90_metrics.parquet")
    if not data_path.exists():
        data_path = Path("../../data/processed/football/player_per90_metrics.parquet")
    if not data_path.exists():
        print(f"Error: Dataset not found at {data_path}. Ensure Day 27 pipeline has run.")
        return

    df = pd.read_parquet(data_path)
    engine = PlayerSimilarityEngine(scaler_type=args.scaler)
    engine.fit(df)

    print(f"\nSearching top {args.top_n} replacements for: '{args.player}' (Position Filter: {args.same_position}, Min Mins: {args.min_minutes})...\n")
    try:
        results = engine.find_similar(
            player_name=args.player,
            top_n=args.top_n,
            same_position=args.same_position,
            min_minutes=args.min_minutes
        )
        target_full_name = results.attrs.get("target_player", args.player)
        print(results[["player_name", "team_name", "primary_position", "similarity_score", "total_minutes_played", "passes_completed_p90", "xg_p90"]].to_string())

        if args.save_plots and len(results) > 0:
            figures_dir = Path("reports/figures")
            figures_dir.mkdir(parents=True, exist_ok=True)
            
            # 1. Similarity ranking plot
            clean_name = target_full_name.split()[-1].lower()
            plot_similarity_results(
                results_df=results,
                target_player=target_full_name,
                save_path=figures_dir / f"player_similarity_{clean_name}.png"
            )
            
            # 2. Head-to-Head with top candidate
            top_candidate = results.iloc[0]["player_name"]
            comp_df = engine.compare_players(target_full_name, top_candidate)
            plot_head_to_head(
                comp_df=comp_df,
                player_a=target_full_name,
                player_b=top_candidate,
                save_path=figures_dir / f"head_to_head_{clean_name}.png"
            )
            print("\nAll figures generated and persisted successfully!")

    except Exception as exc:
        print(f"Search failed: {exc}")


if __name__ == "__main__":
    main()
