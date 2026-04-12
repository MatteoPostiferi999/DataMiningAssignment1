import pandas as pd

FEATURES_PATH = "data/processed/features.csv"


def build_daily_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot the long-format raw data into a wide daily DataFrame.
    One row per (patient, day), one column per variable.
    Target: average mood per day.
    To be implemented in Task 1C.
    """
    raise NotImplementedError


def build_sliding_window_dataset(
    daily_df: pd.DataFrame,
    window_size: int = 5,
) -> pd.DataFrame:
    """
    Transform the daily DataFrame into an instance-based dataset using a
    sliding window of `window_size` days.
    Each instance aggregates history (mean, std, trend, etc.) over the window
    and the target is the mood of the next day.
    To be implemented in Task 1C.
    """
    raise NotImplementedError
