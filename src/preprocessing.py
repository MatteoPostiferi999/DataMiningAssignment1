import pandas as pd
import numpy as np

RAW_PATH = "data/raw/dataset_mood_smartphone.csv"
CLEANED_PATH = "data/processed/cleaned.csv"

# Variables aggregated as mean (self-reports)
MEAN_VARS = ["mood", "circumplex.arousal", "circumplex.valence", "activity"]
# Variables aggregated as sum (sensor counts / durations)
SUM_VARS = [
    "screen", "call", "sms",
    "appCat.builtin", "appCat.communication", "appCat.entertainment",
    "appCat.finance", "appCat.game", "appCat.office", "appCat.other",
    "appCat.social", "appCat.travel", "appCat.unknown",
    "appCat.utilities", "appCat.weather",
]
# Variables that are zero-imputed (missing = no activity that day)
ZERO_IMPUTE_VARS = [
    "screen", "call", "sms",
    "appCat.builtin", "appCat.communication", "appCat.entertainment",
    "appCat.finance", "appCat.game", "appCat.office", "appCat.other",
    "appCat.social", "appCat.travel", "appCat.unknown",
    "appCat.utilities", "appCat.weather",
]
# Variables compared for LOCF vs interpolation (true unknowns)
INTERP_VARS = ["mood", "circumplex.arousal", "circumplex.valence", "activity"]

# Hard domain bounds per variable (inclusive [lo, hi])
DOMAIN_BOUNDS = {
    "mood":                 (1,    10),
    "circumplex.arousal":   (-2,   2),
    "circumplex.valence":   (-2,   2),
    "activity":             (0,    1),
    # All appCat.* have a lower bound of 0 (time cannot be negative)
    "appCat.builtin":       (0,    None),
    "appCat.communication": (0,    None),
    "appCat.entertainment": (0,    None),
    "appCat.finance":       (0,    None),
    "appCat.game":          (0,    None),
    "appCat.office":        (0,    None),
    "appCat.other":         (0,    None),
    "appCat.social":        (0,    None),
    "appCat.travel":        (0,    None),
    "appCat.unknown":       (0,    None),
    "appCat.utilities":     (0,    None),
    "appCat.weather":       (0,    None),
    "screen":               (0,    None),
}


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    """Load the raw dataset."""
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Step 1 — Domain clipping on long format
# ---------------------------------------------------------------------------

def remove_outliers(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Apply domain-based lower/upper bounds per variable on the raw long-format
    data.  Returns:
      - the cleaned DataFrame
      - a summary report (counts per variable)
      - a detail DataFrame with the original rows that were clipped

    This intentionally does NOT apply IQR-based caps here because IQR on
    raw session-level records is too aggressive (individual sessions are
    short, so the IQR upper bound can be unrealistically low).  IQR caps
    are applied later on daily aggregates in clip_daily_outliers().
    """
    df = df.copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    report_rows = []
    clipped_rows = []
    for var, (lo, hi) in DOMAIN_BOUNDS.items():
        mask = df["variable"] == var
        vals = df.loc[mask, "value"]
        n_total = mask.sum()

        n_lo = (vals < lo).sum() if lo is not None else 0
        n_hi = (vals > hi).sum() if hi is not None else 0

        # Collect the original rows before clipping
        if lo is not None:
            bad_lo = df.loc[mask & (df["value"] < lo)].copy()
            bad_lo["bound_violated"] = "lower"
            bad_lo["bound_value"] = lo
            clipped_rows.append(bad_lo)
        if hi is not None:
            bad_hi = df.loc[mask & (df["value"] > hi)].copy()
            bad_hi["bound_violated"] = "upper"
            bad_hi["bound_value"] = hi
            clipped_rows.append(bad_hi)

        if lo is not None:
            df.loc[mask, "value"] = df.loc[mask, "value"].clip(lower=lo)
        if hi is not None:
            df.loc[mask, "value"] = df.loc[mask, "value"].clip(upper=hi)

        if n_lo + n_hi > 0:
            report_rows.append({
                "variable":    var,
                "n_records":   int(n_total),
                "n_clipped_lo": int(n_lo),
                "n_clipped_hi": int(n_hi),
                "n_clipped":   int(n_lo + n_hi),
                "pct_clipped": round(100 * (n_lo + n_hi) / n_total, 2) if n_total else 0,
            })

    report = pd.DataFrame(report_rows)
    clipped_detail = pd.concat(clipped_rows, ignore_index=True) if clipped_rows else pd.DataFrame()
    return df, report, clipped_detail


# ---------------------------------------------------------------------------
# Step 2 — Pivot to wide format
# ---------------------------------------------------------------------------

def pivot_to_wide(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert long-format data to wide format: one row per (id, date).
    Aggregation strategy:
      - mean: mood, circumplex.arousal, circumplex.valence, activity
      - sum:  screen, call, sms, appCat.*
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["time"]).dt.normalize()  # date at midnight

    sum_present  = [v for v in SUM_VARS  if v in df["variable"].unique()]
    mean_present = [v for v in MEAN_VARS if v in df["variable"].unique()]

    sum_wide = (
        df[df["variable"].isin(sum_present)]
        .pivot_table(index=["id", "date"], columns="variable", values="value", aggfunc="sum")
        .reset_index()
    )
    mean_wide = (
        df[df["variable"].isin(mean_present)]
        .pivot_table(index=["id", "date"], columns="variable", values="value", aggfunc="mean")
        .reset_index()
    )

    wide = sum_wide.merge(mean_wide, on=["id", "date"], how="outer")
    wide = wide.sort_values(["id", "date"]).reset_index(drop=True)
    return wide


# ---------------------------------------------------------------------------
# Step 3 — IQR-based caps on daily aggregates
# ---------------------------------------------------------------------------

def clip_daily_outliers(
    df_wide: pd.DataFrame,
    multiplier: float = 1.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply IQR-based upper caps (Q3 + multiplier*IQR) to daily aggregate
    columns.  Applied after pivot so the unit is 'total usage per day',
    which is the relevant scale for modelling.

    Only upper bounds are applied (lower bounds are handled by domain
    clipping in remove_outliers(); daily sums cannot be negative after
    that step).

    Returns the capped DataFrame and a report.
    """
    df_wide = df_wide.copy()
    cap_cols = [c for c in SUM_VARS if c in df_wide.columns]

    report_rows = []
    for col in cap_cols:
        vals = df_wide[col].dropna()
        q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
        iqr = q3 - q1
        upper = q3 + multiplier * iqr

        n_above = (df_wide[col] > upper).sum()
        df_wide[col] = df_wide[col].clip(upper=upper)

        report_rows.append({
            "variable":   col,
            "q1":         round(q1, 1),
            "q3":         round(q3, 1),
            "iqr_upper":  round(upper, 1),
            "n_clipped":  int(n_above),
            "pct_clipped": round(100 * n_above / len(vals), 2) if len(vals) else 0,
        })

    report = pd.DataFrame(report_rows)
    return df_wide, report


# ---------------------------------------------------------------------------
# Step 4 — Zero imputation for sensor-activity columns
# ---------------------------------------------------------------------------

def impute_zeros(df_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Fill NaN with 0 for columns where missing means 'no activity recorded
    that day' rather than 'unknown value'.  This covers all appCat.*,
    screen, call, and sms columns.
    """
    df_wide = df_wide.copy()
    zero_cols = [c for c in ZERO_IMPUTE_VARS if c in df_wide.columns]
    df_wide[zero_cols] = df_wide[zero_cols].fillna(0)
    return df_wide


# ---------------------------------------------------------------------------
# Step 5a — Prolonged gap flagging
# ---------------------------------------------------------------------------

def flag_prolonged_gaps(
    df_wide: pd.DataFrame,
    ref_col: str = "mood",
    threshold: int = 3,
) -> pd.DataFrame:
    """
    Per user, identify calendar dates that fall inside a run of more than
    `threshold` consecutive missing days (based on `ref_col`).  Adds a
    binary `gap_flag` column: 1 = prolonged gap, 0 = normal.

    These rows will not be imputed; the flag is kept as a feature for
    downstream models.
    """
    df_wide = df_wide.copy()
    df_wide["gap_flag"] = 0

    for uid, grp in df_wide.groupby("id"):
        # Build a full calendar index spanning the user's observation window
        if grp["date"].isna().all():
            continue
        full_idx = pd.date_range(grp["date"].min(), grp["date"].max(), freq="D")
        # Series of whether mood is available (True) or missing (False)
        presence = grp.set_index("date")[ref_col].reindex(full_idx).notna()

        # Find runs of consecutive missing days
        in_gap = False
        run_start = None
        gap_dates = set()

        dates = presence.index.tolist()
        for i, d in enumerate(dates):
            if not presence[d]:
                if not in_gap:
                    in_gap = True
                    run_start = i
            else:
                if in_gap:
                    run_length = i - run_start
                    if run_length > threshold:
                        gap_dates.update(dates[run_start:i])
                    in_gap = False
        # Handle gap that reaches the end of the series
        if in_gap:
            run_length = len(dates) - run_start
            if run_length > threshold:
                gap_dates.update(dates[run_start:])

        if gap_dates:
            mask = (df_wide["id"] == uid) & (df_wide["date"].isin(gap_dates))
            df_wide.loc[mask, "gap_flag"] = 1

    return df_wide


# ---------------------------------------------------------------------------
# Step 5b — Imputation: LOCF (forward fill)
# ---------------------------------------------------------------------------

def impute_forward_fill(
    df_wide: pd.DataFrame,
    limit: int = 3,
) -> pd.DataFrame:
    """
    Impute INTERP_VARS using Last Observation Carried Forward (LOCF),
    applied per user.

    `limit` caps the maximum number of consecutive NaNs that will be
    filled — naturally avoiding imputation across prolonged gaps (> limit
    days).

    Gap-flagged rows are treated as hard segment boundaries: non-gap rows
    are extracted and ffilled independently, so a value from before a
    prolonged gap cannot leak across it.

    Leading NaNs (before the first observed value) are backward-filled
    with the first observation, subject to the same `limit`.

    Rows flagged with gap_flag=1 are always left as NaN.
    """
    df_wide = df_wide.copy()
    interp_cols = [c for c in INTERP_VARS if c in df_wide.columns]

    for uid, grp_idx in df_wide.groupby("id").groups.items():
        grp = df_wide.loc[grp_idx].sort_values("date")
        non_gap_mask = grp["gap_flag"] == 0

        for col in interp_cols:
            series = grp[col].copy()

            # Extract only non-gap positions
            seg = series[non_gap_mask].copy()

            # Forward fill (respects limit)
            filled = seg.ffill(limit=limit)

            # Backward fill only for leading NaNs at the start of the
            # series (positions before the first observed value)
            first_valid = seg.first_valid_index()
            if first_valid is not None:
                leading_mask = seg.isna() & (seg.index < first_valid)
                leading_run = leading_mask.sum()
                if 0 < leading_run <= limit:
                    leading_bfill = filled.bfill(limit=limit)
                    filled = filled.where(~leading_mask, leading_bfill)

            # Write back to non-gap positions only
            df_wide.loc[filled.index, col] = filled.values

    return df_wide


# ---------------------------------------------------------------------------
# Step 5c — Imputation: linear interpolation
# ---------------------------------------------------------------------------

def impute_linear_interpolation(
    df_wide: pd.DataFrame,
    limit: int = 3,
) -> pd.DataFrame:
    """
    Impute INTERP_VARS using linear interpolation between known values,
    applied per user.

    `limit` caps the maximum gap that interpolation will bridge, avoiding
    fabrication of data across prolonged missing stretches.

    Gap-flagged rows are treated as hard segment boundaries: the non-gap
    rows are extracted, interpolated independently, and written back.
    This prevents a short gap next to a prolonged gap from being counted
    as part of the long run.

    Rows flagged with gap_flag=1 are always left as NaN.
    """
    df_wide = df_wide.copy()
    interp_cols = [c for c in INTERP_VARS if c in df_wide.columns]

    for uid, grp_idx in df_wide.groupby("id").groups.items():
        grp = df_wide.loc[grp_idx].sort_values("date")
        non_gap_mask = grp["gap_flag"] == 0

        for col in interp_cols:
            series = grp[col].copy()

            # Extract only non-gap positions so NaN run lengths are
            # computed without gap_flag=1 rows inflating them
            seg = series[non_gap_mask].copy()

            # Count the length of each consecutive NaN run
            is_nan = seg.isna()
            nan_groups = is_nan.ne(is_nan.shift()).cumsum()
            nan_run_lengths = is_nan.groupby(nan_groups).transform("sum")

            # Interpolate all interior gaps
            interpolated = seg.interpolate(method="linear")

            # Mask out runs longer than limit
            too_long = is_nan & (nan_run_lengths > limit)
            interpolated = interpolated.where(~too_long, np.nan)

            # Handle leading NaNs (before first observed value) via bfill
            first_valid = seg.first_valid_index()
            if first_valid is not None:
                leading_mask = is_nan & (seg.index < first_valid)
                leading_run = leading_mask.sum()
                if 0 < leading_run <= limit:
                    leading_bfill = interpolated.bfill(limit=limit)
                    interpolated = interpolated.where(~leading_mask, leading_bfill)

            # Write back to non-gap positions only; gap_flag=1 rows stay
            # at their original value (NaN or observed)
            df_wide.loc[interpolated.index, col] = interpolated.values

    return df_wide
