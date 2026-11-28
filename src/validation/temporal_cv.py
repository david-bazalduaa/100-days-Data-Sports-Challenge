"""Day 36: Temporal Validation Framework.

Scikit-learn compatible temporal cross-validators (`SeasonBasedSplit`,
`ExpandingWindowSplit`, `RollingWindowSplit`) designed to prevent look-ahead
data leakage in sports analytics pipelines.
"""

from __future__ import annotations

from typing import Iterator
import numpy as np
from sklearn.model_selection import BaseCrossValidator


class SeasonBasedSplit(BaseCrossValidator):
    """Chronological split across discrete seasons (past -> future).

    Each fold trains on all seasons prior to the cutoff and validates on the
    next `val_seasons` upcoming season(s), preventing future-season leakage.
    """

    def __init__(self, min_train_seasons: int = 3, val_seasons: int = 1) -> None:
        if min_train_seasons < 1:
            raise ValueError("min_train_seasons must be >= 1.")
        if val_seasons < 1:
            raise ValueError("val_seasons must be >= 1.")
        self.min_train_seasons = min_train_seasons
        self.val_seasons = val_seasons

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Return the number of chronological season folds."""
        if groups is None:
            raise ValueError("`groups` (season array) must be provided to get_n_splits.")
        unique_seasons = np.unique(groups)
        n_splits = len(unique_seasons) - self.min_train_seasons - self.val_seasons + 1
        return max(0, n_splits)

    def split(
        self, X, y=None, groups=None
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield (train_idx, val_idx) ordered strictly from past to future."""
        if groups is None:
            raise ValueError("`groups` (season array) must be provided to split.")
        groups_arr = np.asarray(groups)
        unique_seasons = np.sort(np.unique(groups_arr))
        max_start = len(unique_seasons) - self.val_seasons + 1

        for i in range(self.min_train_seasons, max_start):
            train_seasons = unique_seasons[:i]
            val_seasons = unique_seasons[i : i + self.val_seasons]
            train_idx = np.where(np.isin(groups_arr, train_seasons))[0]
            val_idx = np.where(np.isin(groups_arr, val_seasons))[0]
            yield train_idx, val_idx


class ExpandingWindowSplit(BaseCrossValidator):
    """Expanding-window temporal cross-validator over ordered time units.

    Trains on all unique time units up to the cutoff `[0 : i]` and validates on
    `[i + gap : i + gap + val_size]`, advancing by `step_size`.
    """

    def __init__(
        self,
        min_train_size: int = 8,
        val_size: int = 2,
        step_size: int = 2,
        gap: int = 0,
    ) -> None:
        if min_train_size < 1 or val_size < 1 or step_size < 1 or gap < 0:
            raise ValueError("Invalid window parameters: sizes/step must be >= 1 and gap >= 0.")
        self.min_train_size = min_train_size
        self.val_size = val_size
        self.step_size = step_size
        self.gap = gap

    def _get_cutoffs(self, unique_times: np.ndarray) -> range:
        max_start = len(unique_times) - self.gap - self.val_size + 1
        return range(self.min_train_size, max(self.min_train_size, max_start), self.step_size)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Return the number of expanding-window folds."""
        if groups is None:
            raise ValueError("`groups` (time unit array) must be provided to get_n_splits.")
        unique_times = np.sort(np.unique(groups))
        return len(self._get_cutoffs(unique_times))

    def split(
        self, X, y=None, groups=None
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield (train_idx, val_idx) with an expanding history and optional gap."""
        if groups is None:
            raise ValueError("`groups` (time unit array) must be provided to split.")
        groups_arr = np.asarray(groups)
        unique_times = np.sort(np.unique(groups_arr))
        for i in self._get_cutoffs(unique_times):
            train_times = unique_times[:i]
            val_times = unique_times[i + self.gap : i + self.gap + self.val_size]
            yield (
                np.where(np.isin(groups_arr, train_times))[0],
                np.where(np.isin(groups_arr, val_times))[0],
            )


class RollingWindowSplit(BaseCrossValidator):
    """Fixed-width rolling-window temporal cross-validator over ordered time units.

    Trains on the most recent `window_size` units `[i - window_size : i]` and
    validates on `[i + gap : i + gap + val_size]`, discarding older history.
    """

    def __init__(
        self,
        window_size: int = 6,
        val_size: int = 2,
        step_size: int = 2,
        gap: int = 0,
    ) -> None:
        if window_size < 1 or val_size < 1 or step_size < 1 or gap < 0:
            raise ValueError("Invalid window parameters: sizes/step must be >= 1 and gap >= 0.")
        self.window_size = window_size
        self.val_size = val_size
        self.step_size = step_size
        self.gap = gap

    def _get_cutoffs(self, unique_times: np.ndarray) -> range:
        max_start = len(unique_times) - self.gap - self.val_size + 1
        return range(self.window_size, max(self.window_size, max_start), self.step_size)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Return the number of rolling-window folds."""
        if groups is None:
            raise ValueError("`groups` (time unit array) must be provided to get_n_splits.")
        unique_times = np.sort(np.unique(groups))
        return len(self._get_cutoffs(unique_times))

    def split(
        self, X, y=None, groups=None
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield (train_idx, val_idx) using a fixed-length trailing window."""
        if groups is None:
            raise ValueError("`groups` (time unit array) must be provided to split.")
        groups_arr = np.asarray(groups)
        unique_times = np.sort(np.unique(groups_arr))
        for i in self._get_cutoffs(unique_times):
            train_times = unique_times[i - self.window_size : i]
            val_times = unique_times[i + self.gap : i + self.gap + self.val_size]
            yield (
                np.where(np.isin(groups_arr, train_times))[0],
                np.where(np.isin(groups_arr, val_times))[0],
            )

