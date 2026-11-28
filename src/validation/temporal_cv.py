"""Day 36: Temporal Validation Framework.

Scikit-learn compatible temporal cross-validators (`SeasonBasedSplit`,
`ExpandingWindowSplit`, `RollingWindowSplit`) designed to prevent look-ahead
data leakage in sports analytics pipelines.
"""

from __future__ import annotations

from typing import Iterator
import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator


def _resolve_time_groups(X, groups, time_col: str | None) -> np.ndarray:
    """Resolve and validate the 1D temporal group array from `groups` or `X[time_col]`."""
    if groups is not None:
        raw = groups
    elif time_col is not None and X is not None and hasattr(X, "__getitem__"):
        raw = X[time_col]
    else:
        raise ValueError("Provide `groups` or set `time_col` with a DataFrame `X`.")

    arr = np.asarray(raw)
    if arr.ndim != 1:
        raise ValueError(f"Temporal groups must be 1D, got shape {arr.shape}.")
    if X is not None and hasattr(X, "__len__") and len(arr) != len(X):
        raise ValueError(f"Length mismatch: len(X)={len(X)} vs len(groups)={len(arr)}.")
    if pd.isna(arr).any():
        raise ValueError("Temporal groups contain missing/NaN values.")
    return arr


class _BaseTemporalWindowSplit(BaseCrossValidator):
    """Shared base class for chronological window cross-validators."""

    def __init__(
        self,
        val_size: int = 1,
        step_size: int = 1,
        gap: int = 0,
        time_col: str | None = None,
    ) -> None:
        if val_size < 1 or step_size < 1 or gap < 0:
            raise ValueError("`val_size` and `step_size` must be >= 1, and `gap` >= 0.")
        self.val_size = val_size
        self.step_size = step_size
        self.gap = gap
        self.time_col = time_col

    def _train_window_slice(self, cutoff_idx: int) -> slice:
        raise NotImplementedError

    def _min_cutoff(self) -> int:
        raise NotImplementedError

    def _get_cutoffs(self, unique_times: np.ndarray) -> range:
        start = self._min_cutoff()
        max_start = len(unique_times) - self.gap - self.val_size + 1
        return range(start, max(start, max_start), self.step_size)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Return the number of chronological folds."""
        arr = _resolve_time_groups(X, groups, self.time_col)
        unique_times = np.sort(np.unique(arr))
        return len(self._get_cutoffs(unique_times))

    def split(
        self, X, y=None, groups=None
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield (train_idx, val_idx) ordered strictly from past to future."""
        arr = _resolve_time_groups(X, groups, self.time_col)
        unique_times = np.sort(np.unique(arr))
        for i in self._get_cutoffs(unique_times):
            train_times = unique_times[self._train_window_slice(i)]
            val_times = unique_times[i + self.gap : i + self.gap + self.val_size]
            yield (
                np.where(np.isin(arr, train_times))[0],
                np.where(np.isin(arr, val_times))[0],
            )


class SeasonBasedSplit(_BaseTemporalWindowSplit):
    """Chronological split across discrete seasons (past -> future)."""

    def __init__(
        self,
        min_train_seasons: int = 3,
        val_seasons: int = 1,
        gap: int = 0,
        time_col: str | None = None,
    ) -> None:
        if min_train_seasons < 1:
            raise ValueError("`min_train_seasons` must be >= 1.")
        super().__init__(val_size=val_seasons, step_size=1, gap=gap, time_col=time_col)
        self.min_train_seasons = min_train_seasons
        self.val_seasons = val_seasons

    def _min_cutoff(self) -> int:
        return self.min_train_seasons

    def _train_window_slice(self, cutoff_idx: int) -> slice:
        return slice(0, cutoff_idx)


class ExpandingWindowSplit(_BaseTemporalWindowSplit):
    """Expanding-window temporal cross-validator over ordered time units."""

    def __init__(
        self,
        min_train_size: int = 8,
        val_size: int = 2,
        step_size: int = 2,
        gap: int = 0,
        time_col: str | None = None,
    ) -> None:
        if min_train_size < 1:
            raise ValueError("`min_train_size` must be >= 1.")
        super().__init__(val_size=val_size, step_size=step_size, gap=gap, time_col=time_col)
        self.min_train_size = min_train_size

    def _min_cutoff(self) -> int:
        return self.min_train_size

    def _train_window_slice(self, cutoff_idx: int) -> slice:
        return slice(0, cutoff_idx)


class RollingWindowSplit(_BaseTemporalWindowSplit):
    """Fixed-width rolling-window temporal cross-validator over ordered time units."""

    def __init__(
        self,
        window_size: int = 6,
        val_size: int = 2,
        step_size: int = 2,
        gap: int = 0,
        time_col: str | None = None,
    ) -> None:
        if window_size < 1:
            raise ValueError("`window_size` must be >= 1.")
        super().__init__(val_size=val_size, step_size=step_size, gap=gap, time_col=time_col)
        self.window_size = window_size

    def _min_cutoff(self) -> int:
        return self.window_size

    def _train_window_slice(self, cutoff_idx: int) -> slice:
        return slice(cutoff_idx - self.window_size, cutoff_idx)
