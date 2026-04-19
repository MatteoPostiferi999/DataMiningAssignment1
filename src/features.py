import numpy as np
import pandas as pd

FEATURES_PATH = "data/processed/features_tabular.csv"
SEQUENTIAL_PATH = "data/processed/features_sequential.npz"

SENSOR_COLS = [
    "screen", "call", "sms",
    "appCat.builtin", "appCat.communication", "appCat.entertainment",
    "appCat.finance", "appCat.game", "appCat.office", "appCat.other",
    "appCat.social", "appCat.travel", "appCat.unknown",
    "appCat.utilities", "appCat.weather",
]

MOOD_COLS = ["mood", "mood_std", "mood_min", "mood_max"]
SELF_REPORT_COLS = ["circumplex.arousal", "circumplex.valence", "activity"]
INTERP_COLS = MOOD_COLS + SELF_REPORT_COLS

CNN_CHANNELS = [
    "mood", "mood_std", "mood_min", "mood_max",
    "circumplex.valence", "circumplex.arousal", "activity",
    "log1p_screen", "is_weekend",
    "appCat_social_comm", "appCat_leisure", "appCat_other",
]

TABULAR_FEATURE_COLS = [
    # Current-day values
    "mood", "mood_std", "mood_min", "mood_max", "mood_count",
    "circumplex.valence", "circumplex.arousal", "activity",
    "screen", "call", "sms",
    # Time-of-day
    "mood_morning", "mood_evening",
    # Calendar
    "is_weekend", "day_of_week",
    # Lags
    "mood_lag1", "mood_lag2", "valence_lag1",
    # Rolling mood
    "mood_rmean_7d", "mood_rstd_7d", "mood_rmin_7d", "mood_rmax_7d",
    "mood_rtrend_7d",
    # Rolling other
    "valence_rmean_7d", "valence_rstd_7d",
    "activity_rmean_7d", "activity_rstd_7d",
    "screen_rmean_7d", "screen_rstd_7d",
    # Per-user
    "user_mood_mean", "user_mood_std",
] + [f"appCat.{x}" for x in [
    "builtin", "communication", "entertainment", "finance", "game",
    "office", "other", "social", "travel", "unknown", "utilities", "weather",
]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    x = np.arange(window, dtype=float)
    x -= x.mean()
    denom = (x * x).sum()

    def _slope(arr):
        return np.dot(x, arr) / denom

    return series.rolling(window, min_periods=window).apply(_slope, raw=True)


# ---------------------------------------------------------------------------
# Step 0 — Reindex to continuous calendar + re-interpolate
# ---------------------------------------------------------------------------

def reindex_to_calendar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    frames = []
    for uid, grp in df.groupby("id"):
        grp = grp.sort_values("date").set_index("date")
        full = pd.date_range(grp.index.min(), grp.index.max(), freq="D")
        r = grp.reindex(full)
        r["id"] = uid

        sensor_present = [c for c in SENSOR_COLS if c in r.columns]
        r[sensor_present] = r[sensor_present].fillna(0)

        for c in INTERP_COLS:
            if c in r.columns:
                r[c] = r[c].interpolate(method="linear", limit=3)

        if "mood_count" in r.columns:
            r["mood_count"] = r["mood_count"].fillna(0)

        r["gap_flag"] = r["mood"].isna().astype(int)

        r.index.name = "date"
        frames.append(r.reset_index())
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Step 1 — Time-of-day features from raw mood readings
# ---------------------------------------------------------------------------

def compute_time_of_day_features(long_mood: pd.DataFrame) -> pd.DataFrame:
    lm = long_mood.copy()
    lm["time"] = pd.to_datetime(lm["time"])
    lm["date"] = lm["time"].dt.normalize()
    lm["hour"] = lm["time"].dt.hour

    morning = (
        lm[lm["hour"] < 12]
        .groupby(["id", "date"])["mood"]
        .mean()
        .rename("mood_morning")
    )
    evening = (
        lm[lm["hour"] >= 18]
        .groupby(["id", "date"])["mood"]
        .mean()
        .rename("mood_evening")
    )
    return pd.concat([morning, evening], axis=1).reset_index()


# ---------------------------------------------------------------------------
# Step 2 — Log transforms + grouped appCats
# ---------------------------------------------------------------------------

def add_transforms(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in SENSOR_COLS:
        if col in df.columns:
            df[f"log1p_{col}"] = np.log1p(df[col].clip(lower=0))

    df["appCat_social_comm"] = np.log1p(
        df[["appCat.social", "appCat.communication"]]
        .fillna(0).sum(axis=1).clip(lower=0)
    )
    df["appCat_leisure"] = np.log1p(
        df[["appCat.entertainment", "appCat.game"]]
        .fillna(0).sum(axis=1).clip(lower=0)
    )
    other_cols = [
        "appCat.builtin", "appCat.finance", "appCat.office", "appCat.other",
        "appCat.travel", "appCat.unknown", "appCat.utilities", "appCat.weather",
    ]
    df["appCat_other"] = np.log1p(
        df[other_cols].fillna(0).sum(axis=1).clip(lower=0)
    )
    return df


# ---------------------------------------------------------------------------
# Step 3 — Calendar features
# ---------------------------------------------------------------------------

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    return df


# ---------------------------------------------------------------------------
# Step 4 — Per-user baseline features
# ---------------------------------------------------------------------------

def add_user_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    non_gap = df[df["gap_flag"] == 0]
    stats = non_gap.groupby("id")["mood"].agg(
        user_mood_mean="mean", user_mood_std="std"
    )
    return df.merge(stats, on="id", how="left")


# ---------------------------------------------------------------------------
# Step 5 — Lag features (gap-aware)
# ---------------------------------------------------------------------------

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["id", "date"])
    for col, lag, name in [
        ("mood", 1, "mood_lag1"),
        ("mood", 2, "mood_lag2"),
        ("circumplex.valence", 1, "valence_lag1"),
    ]:
        shifted_val = df.groupby("id")[col].shift(lag)
        shifted_date = df.groupby("id")["date"].shift(lag)
        gap = (df["date"] - shifted_date).dt.days
        df[name] = shifted_val.where(gap == lag)
    return df


# ---------------------------------------------------------------------------
# Step 6 — Rolling window features
# ---------------------------------------------------------------------------

def add_rolling_features(df: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    df = df.copy().sort_values(["id", "date"])
    w = window

    for uid, idx in df.groupby("id").groups.items():
        grp = df.loc[idx]

        mood = grp["mood"]
        df.loc[idx, "mood_rmean_7d"] = mood.rolling(w, min_periods=w).mean()
        df.loc[idx, "mood_rstd_7d"] = mood.rolling(w, min_periods=w).std()
        df.loc[idx, "mood_rmin_7d"] = mood.rolling(w, min_periods=w).min()
        df.loc[idx, "mood_rmax_7d"] = mood.rolling(w, min_periods=w).max()
        df.loc[idx, "mood_rtrend_7d"] = _rolling_slope(mood, w)

        val = grp["circumplex.valence"]
        df.loc[idx, "valence_rmean_7d"] = val.rolling(w, min_periods=w).mean()
        df.loc[idx, "valence_rstd_7d"] = val.rolling(w, min_periods=w).std()

        act = grp["activity"]
        df.loc[idx, "activity_rmean_7d"] = act.rolling(w, min_periods=w).mean()
        df.loc[idx, "activity_rstd_7d"] = act.rolling(w, min_periods=w).std()

        scr = grp["log1p_screen"] if "log1p_screen" in grp.columns else grp["screen"]
        df.loc[idx, "screen_rmean_7d"] = scr.rolling(w, min_periods=w).mean()
        df.loc[idx, "screen_rstd_7d"] = scr.rolling(w, min_periods=w).std()

    return df


# ---------------------------------------------------------------------------
# Step 7 — Target: next calendar day's mood
# ---------------------------------------------------------------------------

def add_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["id", "date"])
    shifted_mood = df.groupby("id")["mood"].shift(-1)
    shifted_date = df.groupby("id")["date"].shift(-1)
    gap = (shifted_date - df["date"]).dt.days
    df["target"] = shifted_mood.where(gap == 1)
    return df


# ---------------------------------------------------------------------------
# Main enrichment pipeline
# ---------------------------------------------------------------------------

def enrich_daily(
    cleaned: pd.DataFrame,
    long_mood: pd.DataFrame,
    window: int = 7,
    drop_february: bool = True,
) -> pd.DataFrame:
    df = cleaned.copy()
    df["date"] = pd.to_datetime(df["date"])

    if drop_february:
        df = df[df["date"].dt.month != 2].copy()

    df = reindex_to_calendar(df)

    tod = compute_time_of_day_features(long_mood)
    tod["date"] = pd.to_datetime(tod["date"])
    df = df.merge(tod, on=["id", "date"], how="left")

    df = add_transforms(df)
    df = add_calendar_features(df)
    df = add_user_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df, window=window)
    df = add_target(df)
    return df


# ---------------------------------------------------------------------------
# Dataset builders
# ---------------------------------------------------------------------------

def build_tabular_dataset(
    enriched: pd.DataFrame,
    window: int = 7,
) -> pd.DataFrame:
    df = enriched.copy().sort_values(["id", "date"])

    df = df.dropna(subset=["target"])

    rolling_cols = [c for c in df.columns if c.endswith("_7d")]
    df = df.dropna(subset=rolling_cols)

    gap_in_window = df.groupby("id")["gap_flag"].transform(
        lambda s: s.rolling(window, min_periods=1).max()
    )
    df = df[gap_in_window == 0]

    keep = ["id", "date"] + [
        c for c in TABULAR_FEATURE_COLS if c in df.columns
    ] + ["target"]
    return df[keep].reset_index(drop=True)


def build_sequential_dataset(
    enriched: pd.DataFrame,
    window: int = 7,
    channels: list | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    if channels is None:
        channels = [c for c in CNN_CHANNELS if c in enriched.columns]

    df = enriched.copy().sort_values(["id", "date"]).reset_index(drop=True)

    X_list, y_list, meta_list = [], [], []

    for uid, grp in df.groupby("id"):
        grp = grp.reset_index(drop=True)
        vals = grp[channels].values
        targets = grp["target"].values
        flags = grp["gap_flag"].values

        for i in range(window, len(grp)):
            if np.isnan(targets[i - 1]):
                continue
            win_flags = flags[i - window : i]
            if win_flags.max() > 0:
                continue
            win_data = vals[i - window : i]
            if np.any(np.isnan(win_data)):
                continue

            X_list.append(win_data)
            y_list.append(targets[i - 1])
            meta_list.append({"id": uid, "date": grp.loc[grp.index[i - 1], "date"]})

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.float32)
    meta = pd.DataFrame(meta_list)
    return X, y, meta
