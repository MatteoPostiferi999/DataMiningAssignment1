import pandas as pd
import numpy as np

RAW_PATH = "data/raw/dataset_mood_smartphone.csv"
CLEANED_PATH = "data/processed/cleaned.csv"


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    """Load the raw dataset."""
    return pd.read_csv(path)


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove extreme/incorrect values per variable using IQR-based bounds.
    To be implemented in Task 1B.
    """
    raise NotImplementedError


def impute_forward_fill(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values using forward fill (carry last observation forward).
    Suitable for time series. To be implemented in Task 1B.
    """
    raise NotImplementedError


def impute_linear_interpolation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values using linear interpolation between known values.
    Suitable for time series. To be implemented in Task 1B.
    """
    raise NotImplementedError
