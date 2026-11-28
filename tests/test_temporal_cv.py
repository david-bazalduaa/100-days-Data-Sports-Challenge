"""Unit tests for src/validation/temporal_cv.py."""

import unittest
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, cross_val_score

from src.validation.temporal_cv import (
    ExpandingWindowSplit,
    RollingWindowSplit,
    SeasonBasedSplit,
)


class TestTemporalCV(unittest.TestCase):
    """Verify strict temporal causality, embargo gaps, and sklearn compatibility."""

    def setUp(self) -> None:
        rng = np.random.default_rng(42)
        weeks = np.repeat(np.arange(1, 21), 4)
        self.df = pd.DataFrame({
            "week": weeks,
            "f1": rng.normal(size=len(weeks)),
            "f2": rng.normal(size=len(weeks)),
            "y": rng.normal(size=len(weeks)),
        })

    def test_season_based_split_causality(self) -> None:
        seasons = np.repeat(np.arange(2015, 2024), 5)
        X = np.zeros((len(seasons), 2))
        cv = SeasonBasedSplit(min_train_seasons=4, val_seasons=1)
        self.assertEqual(cv.get_n_splits(X, groups=seasons), 5)
        for tr, va in cv.split(X, groups=seasons):
            self.assertEqual(len(set(tr) & set(va)), 0)
            self.assertLess(seasons[tr].max(), seasons[va].min())

    def test_expanding_and_rolling_gap_enforcement(self) -> None:
        weeks = self.df["week"].values
        exp_cv = ExpandingWindowSplit(min_train_size=8, val_size=2, step_size=2, gap=1)
        roll_cv = RollingWindowSplit(window_size=6, val_size=2, step_size=2, gap=1)

        prev_train_len = 0
        for tr, va in exp_cv.split(self.df, groups=weeks):
            self.assertGreater(len(tr), prev_train_len)
            prev_train_len = len(tr)
            self.assertLess(weeks[tr].max() + 1, weeks[va].min())

        for tr, va in roll_cv.split(self.df, groups=weeks):
            self.assertEqual(len(np.unique(weeks[tr])), 6)
            self.assertLess(weeks[tr].max() + 1, weeks[va].min())

    def test_time_col_support_and_nan_rejection(self) -> None:
        cv = RollingWindowSplit(window_size=5, val_size=2, step_size=3, time_col="week")
        splits = list(cv.split(self.df))
        self.assertEqual(len(splits), cv.get_n_splits(self.df))

        bad_df = self.df.copy()
        bad_df.loc[0, "week"] = np.nan
        with self.assertRaises(ValueError):
            list(cv.split(bad_df))

    def test_sklearn_cross_val_and_grid_search(self) -> None:
        X = self.df[["f1", "f2"]]
        y = self.df["y"]
        groups = self.df["week"].values
        cv = ExpandingWindowSplit(min_train_size=8, val_size=2, step_size=4, gap=1)

        scores = cross_val_score(Ridge(), X, y, cv=cv, groups=groups, scoring="r2")
        self.assertEqual(len(scores), cv.get_n_splits(X, y, groups=groups))

        gs = GridSearchCV(Ridge(), {"alpha": [0.1, 1.0]}, cv=cv, scoring="neg_root_mean_squared_error")
        gs.fit(X, y, groups=groups)
        self.assertIn(gs.best_params_["alpha"], [0.1, 1.0])


if __name__ == "__main__":
    unittest.main()
