import io
import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

import allocation_check as ac

# -----------------------------
# LOGGING
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("morning_comp")

# -----------------------------
# CONFIG
# -----------------------------
EXCLUDE_ALL_USERS = ["not running", "dlr acc", "z server"]
EXCLUDE_RUNNING = ["z server"]

# Servers always excluded from Fixed tab (regardless of mode)
FIXED_EXCLUDE_SERVERS = {"not running", "dlr acc", "z server"}

COLS = ["userid", "alias", "allocation", "max_loss", "server", "algo"]

# Mode keys
MODE_0DTE = "0DTE"
MODE_1DTE = "1DTE"
MODE_4DTE = "4DTE"

# Allocation threshold below which accounts are flagged (per mode)
ALLOC_THRESHOLD = {
    MODE_4DTE: 100_000,
    MODE_1DTE: 60_000,
}

# Mode-specific filters applied ONLY on the All Users (Main) sheet.
MODE_FILTERS = {
    MODE_0DTE: {
        "apply_base_exclusions": True,
        "runningtype": None,
        "runningdays": None,
    },
    MODE_1DTE: {
        "apply_base_exclusions": True,
        "runningtype": {"pos", "int"},
        "runningdays": {"daily", "1dte/0dte"},
    },
    MODE_4DTE: {
        "apply_base_exclusions": False,
        "runningtype": {"pos"},
        "runningdays": {"daily"},
    },
}

# Supported main-sheet names (case-insensitive)
MAIN_SHEET_NAMES = {"main", "mfibain"}

# App sections
SECTION_LOGIN = "Login Check"
SECTION_ALLOCATION = "Allocation Check"

# -----------------------------
# FIXED ALLOCATION
# -----------------------------
# Internal name for the "FIX (CR)" column on the All Users (Main) sheet.
FIX_CR_COL = "fix_cr"

# Expected allocation = FIX (CR) value x this multiplier.
# The sheet stores allocation in a scaled unit where 1 CR == 100,000,
# so FIX (CR) = 1.6 -> 160,000 and FIX (CR) = 0.8 -> 80,000.
FIX_CR_MULTIPLIER = 100_000

# Cell values that mean "no fixed allocation" rather than "bad data".
FIX_CR_BLANK_TOKENS = {"", "nan", "none", "null", "-", "na", "n/a"}

FIXED_TAB_COLUMNS = [
    "userid", "alias", "server", "algo", FIX_CR_COL,
    "expected_allocation", "actual_allocation", "status", "_match",
]

FIX_CR_INVALID_COLUMNS = ["userid", "alias", "server", "algo", "fix_cr_raw", "reason"]

# -----------------------------
# OPERATOR
# -----------------------------
# Internal name for the "Operator Name" column on the All Users (Main) sheet.
OPERATOR_COL = "operator_name"

# Operator is resolved from the All Users sheet by (algo, server) and appended
# as the LAST column of every table. The Running Users file has no operator
# column, so running-sourced rows are resolved through the same lookup.
OPERATOR_UNKNOWN = ""

# -----------------------------
# 0 SL
# -----------------------------
# Internal name for the "SL" column on the All Users (Main) sheet.
SL_COL = "sl"

# Only a numeric zero counts as "0 SL". A blank cell means "no SL recorded"
# and is deliberately NOT counted.
ZERO_SL_COUNT_COL = "count of 0 SL accounts"
ZERO_SL_COLUMNS = ["algo", "server", ZERO_SL_COUNT_COL, OPERATOR_COL]

# -----------------------------
# COLUMN NORMALIZATION
# -----------------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns.str.lower()
        .str.strip()
        .str.replace(" ", "", regex=False)
    )
    column_map = {
        "userid": "userid",
        "useralias": "alias",
        "alias": "alias",
        "allocation": "allocation",
        "maxloss": "max_loss",
        "server": "server",
        "algo": "algo",
        "runningtype": "runningtype",
        "runningdays": "runningdays",
        "remarks/algo8previousdayrealisedmtm": "remark",
        "remarks": "remark",
        "remark": "remark",
        # "FIX (CR)" -> lower/strip/space-stripped -> "fix(cr)"
        "fix(cr)": FIX_CR_COL,
        "fixcr": FIX_CR_COL,
        "fix(crore)": FIX_CR_COL,
        "fix(cr.)": FIX_CR_COL,
        # "SL" -> "sl". Distinct from "Check SL" ("checksl") and
        # "Sl Check" ("slcheck"), which are left untouched.
        "sl": SL_COL,
        # "Operator Name" -> "operatorname"
        "operatorname": OPERATOR_COL,
        "operator": OPERATOR_COL,
        # Allocation Check inputs
        "subcategory": "subcategory",
        "sub_category": "subcategory",
        "category": "category",
        "capital": "capital",
    }
    return df.rename(columns=column_map)


# -----------------------------
# VALUE NORMALIZATION
# -----------------------------
def normalize_values(df: pd.DataFrame, is_running: bool = False) -> pd.DataFrame:
    df = df.copy()
    df["userid"] = (
        df["userid"]
        .astype(str)
        .str.strip()
        .str.replace(" ", "", regex=False)
        .str.upper()
    )
    df["alias"] = df["alias"].astype(str).str.strip()
    df["server"] = df["server"].astype(str).str.strip().str.lower()
    df["allocation"] = pd.to_numeric(df["allocation"], errors="coerce")
    df["max_loss"] = pd.to_numeric(df["max_loss"], errors="coerce")
    if is_running:
        df["allocation"] = df["allocation"] / 100
        df["max_loss"] = df["max_loss"].abs()
    df["allocation"] = np.floor(df["allocation"])
    df["max_loss"] = np.floor(df["max_loss"])
    for c in ("runningtype", "runningdays"):
        if c in df.columns:
            df[c] = (
                df[c]
                .astype(str)
                .str.strip()
                .str.replace(" ", "", regex=False)
                .str.lower()
            )
    if OPERATOR_COL in df.columns:
        df[OPERATOR_COL] = (
            df[OPERATOR_COL]
            .astype(str)
            .str.strip()
            .replace({"nan": OPERATOR_UNKNOWN, "None": OPERATOR_UNKNOWN,
                      "NaT": OPERATOR_UNKNOWN, "<NA>": OPERATOR_UNKNOWN})
        )
    return df


# -----------------------------
# MODE-SPECIFIC FILTER
# -----------------------------
def apply_mode_filter(df_all: pd.DataFrame, mode: str) -> pd.DataFrame:
    cfg = MODE_FILTERS.get(mode, MODE_FILTERS[MODE_0DTE])
    rt_allowed = cfg.get("runningtype")
    rd_allowed = cfg.get("runningdays")
    if rt_allowed is None and rd_allowed is None:
        return df_all
    df = df_all.copy()
    before = len(df)
    missing = [
        c for c in ("runningtype", "runningdays")
        if (rt_allowed is not None and c == "runningtype" and c not in df.columns)
        or (rd_allowed is not None and c == "runningdays" and c not in df.columns)
    ]
    if missing:
        st.error(
            f"Mode '{mode}' requires column(s) {missing} in the All Users (Main) sheet. "
            "Please add them or switch to 0DTE."
        )
        logger.error("Missing required columns for mode %s: %s", mode, missing)
        st.stop()
    if rt_allowed is not None:
        df = df[df["runningtype"].isin(rt_allowed)]
    if rd_allowed is not None:
        df = df[df["runningdays"].isin(rd_allowed)]
    after = len(df)
    logger.info("Mode %s filter: %d -> %d rows (dropped %d)", mode, before, after, before - after)
    return df


# -----------------------------
# LOADERS
# -----------------------------
def load_all_users(file) -> Optional[pd.DataFrame]:
    try:
        excel = pd.ExcelFile(file)
    except Exception as e:
        st.error(f"Failed to open All Users Excel file: {e}")
        logger.exception("Failed to open All Users Excel")
        return None
    sheet_name = None
    for s in excel.sheet_names:
        if s.lower() in MAIN_SHEET_NAMES:
            sheet_name = s
            break
    if not sheet_name:
        st.error(
            f"Main sheet not found in All Users file. "
            f"Expected a sheet named one of: {sorted(MAIN_SHEET_NAMES)}. "
            f"Found: {excel.sheet_names}"
        )
        return None
    logger.info("Loading main sheet: '%s'", sheet_name)
    return pd.read_excel(file, sheet_name=sheet_name)


def load_running_users(file) -> pd.DataFrame:
    if file.name.lower().endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)


# -----------------------------
# DUPLICATES
# -----------------------------
def get_duplicates(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """
    Duplicate user IDs. algo/server are carried through so the caller can
    resolve the operator; they are dropped before display.
    """
    dup = df[df.duplicated(subset=["userid"], keep=False)].copy()
    dup["Found in"] = source
    return dup[["userid", "algo", "server", "Found in"]]


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates(subset=["userid"], keep=False)


# -----------------------------
# FILTERING (server-based base exclusions)
# -----------------------------
def split_extra(
    df: pd.DataFrame,
    exclude_list: list,
    source_name: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    extra = df[df["server"].isin(exclude_list)].copy()
    extra["not found in"] = source_name
    clean = df[~df["server"].isin(exclude_list)].copy()
    return clean, extra


# -----------------------------
# DIFFERENCE TAB
# -----------------------------
def get_difference(all_df: pd.DataFrame, run_df: pd.DataFrame) -> pd.DataFrame:
    merged = all_df.merge(run_df, on="userid", how="inner", suffixes=("_all", "_run"))
    diff_mask = (
        (merged["allocation_all"] != merged["allocation_run"]) |
        (merged["max_loss_all"] != merged["max_loss_run"]) |
        (merged["server_all"] != merged["server_run"]) |
        (merged["algo_all"] != merged["algo_run"])
    )
    diff = merged[diff_mask].copy()
    return diff[[
        "userid",
        "alias_all", "alias_run",
        "allocation_all", "allocation_run",
        "max_loss_all", "max_loss_run",
        "server_all", "server_run",
        "algo_all", "algo_run",
    ]]


# -----------------------------
# NOT FOUND TAB
# -----------------------------
def get_not_found(all_df: pd.DataFrame, run_df: pd.DataFrame) -> pd.DataFrame:
    all_ids = set(all_df["userid"])
    run_ids = set(run_df["userid"])
    missing_in_run = (
        all_df[all_df["userid"].isin(all_ids - run_ids)][["userid", "server", "algo"]]
        .copy()
    )
    missing_in_run["Not found in"] = "Running"
    missing_in_all = (
        run_df[run_df["userid"].isin(run_ids - all_ids)][["userid", "server", "algo"]]
        .copy()
    )
    missing_in_all["Not found in"] = "All User"
    return pd.concat([missing_in_run, missing_in_all], ignore_index=True)[
        ["userid", "server", "algo", "Not found in"]
    ]


# -----------------------------
# PIVOT / SUMMARY TABLES
# -----------------------------
def build_not_found_summary(not_found_tab: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot of the Not Found tab: count of distinct userid per algo/server,
    broken out by which sheet they were missing from.
    Columns: algo, server, count_of_user, not_found_in
    """
    if not_found_tab.empty:
        return pd.DataFrame(columns=["algo", "server", "count_of_user", "Not found in"])
    summary = (
        not_found_tab
        .groupby(["algo", "server", "Not found in"], dropna=False)["userid"]
        .nunique()
        .reset_index()
        .rename(columns={"userid": "count_of_user"})
    )
    summary = summary[["algo", "server", "count_of_user", "Not found in"]]
    return summary.sort_values(["algo", "server", "Not found in"]).reset_index(drop=True)


ALL_USER_COUNT_COL = "count of user_id All user"
RUNNING_COUNT_COL = "count of user_id running"


def _distinct_userid_counts(df: pd.DataFrame, count_col: str) -> pd.DataFrame:
    if df.empty or "algo" not in df.columns or "server" not in df.columns:
        return pd.DataFrame(columns=["algo", "server", count_col])
    return (
        df.groupby(["algo", "server"], dropna=False)["userid"]
        .nunique()
        .reset_index()
        .rename(columns={"userid": count_col})
    )


def build_running_pivot(df_all_mode_filtered: pd.DataFrame, df_run_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot combining, per algo/server:
      - count of user_id All user: distinct userid from the All Users sheet,
        filtered ONLY by the active DTE mode's RunningType/RunningDays rule
        (0DTE -> no filter/all counted, 1DTE -> RunningType in {POS, INT} &
        RunningDays in {DAILY, 1DTE/0DTE}, 4DTE -> RunningType = POS &
        RunningDays = DAILY). No dedup / server-exclusion is applied here,
        only the mode filter.
      - count of user_id running: distinct userid from the Running Users
        sheet exactly as uploaded (before dedup / server exclusions).
    Columns: algo, server, count of user_id All user, count of user_id running
    """
    all_counts = _distinct_userid_counts(df_all_mode_filtered, ALL_USER_COUNT_COL)
    run_counts = _distinct_userid_counts(df_run_raw, RUNNING_COUNT_COL)

    summary = pd.merge(all_counts, run_counts, on=["algo", "server"], how="outer")
    for col in (ALL_USER_COUNT_COL, RUNNING_COUNT_COL):
        if col not in summary.columns:
            summary[col] = 0
        summary[col] = summary[col].fillna(0).astype(int)

    summary = summary[["algo", "server", ALL_USER_COUNT_COL, RUNNING_COUNT_COL]]
    return summary.sort_values(["algo", "server"]).reset_index(drop=True)


# -----------------------------
# OPERATOR LOOKUP
# -----------------------------
def build_operator_lookup(df_all: pd.DataFrame) -> dict:
    """
    Build an (algo, server) -> operator_name lookup from the All Users sheet.

    Operator is an attribute of the algo/server pair rather than of the
    individual account, which is what lets running-sourced rows (the Running
    Users file carries no operator column) be resolved through the same map.

    If a pair somehow carries more than one operator, the most frequent one
    wins and a warning is logged -- the alternative, silently picking an
    arbitrary row, would misattribute accounts.
    """
    if OPERATOR_COL not in df_all.columns:
        logger.warning(
            "No 'Operator Name' column found in the main sheet -- "
            "operator will be blank in every table."
        )
        return {}

    df = df_all[["algo", "server", OPERATOR_COL]].copy()
    df = df[df[OPERATOR_COL].notna() & (df[OPERATOR_COL] != OPERATOR_UNKNOWN)]
    if df.empty:
        logger.warning("'Operator Name' column is present but entirely empty.")
        return {}

    lookup: dict = {}
    ambiguous: list = []
    for (algo, server), group in df.groupby(["algo", "server"], dropna=False):
        counts = group[OPERATOR_COL].value_counts()
        lookup[(algo, server)] = counts.index[0]
        if len(counts) > 1:
            ambiguous.append(f"algo {algo} / {server}: {list(counts.index)}")

    if ambiguous:
        logger.warning(
            "%d algo/server pair(s) map to more than one operator; using the "
            "most frequent. %s", len(ambiguous), "; ".join(ambiguous),
        )
    logger.info("Operator lookup built for %d algo/server pair(s).", len(lookup))
    return lookup


def attach_operator(
    df: pd.DataFrame,
    lookup: dict,
    algo_col: str = "algo",
    server_col: str = "server",
) -> pd.DataFrame:
    """
    Append operator_name as the LAST column of df.

    Uses a dict lookup rather than a merge on purpose: a merge can silently
    multiply rows if the right-hand side is not unique, and can reorder the
    frame. This maps positionally, so row count and order are guaranteed
    unchanged. Unresolved pairs get a blank operator, never a wrong one.
    """
    out = df.copy()
    if algo_col not in out.columns or server_col not in out.columns:
        logger.warning(
            "Cannot attach operator: missing '%s' / '%s'.", algo_col, server_col
        )
        out[OPERATOR_COL] = OPERATOR_UNKNOWN
        return out

    out[OPERATOR_COL] = [
        lookup.get((algo, server), OPERATOR_UNKNOWN)
        for algo, server in zip(out[algo_col], out[server_col])
    ]
    return out


# -----------------------------
# 0 SL PIVOT
# -----------------------------
def coerce_sl(series: pd.Series) -> pd.Series:
    """
    Coerce the raw SL column to numeric.

    Accepts 0, 0.0, "0", " 0 " and "0%". Blank/empty cells and non-numeric
    text become NaN, so they are never counted as zero.
    """
    cleaned = (
        series.astype(str)
        .str.strip()
        .str.rstrip("%")
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")


def build_zero_sl_pivot(df_all: pd.DataFrame, operator_lookup: dict) -> pd.DataFrame:
    """
    Pivot of accounts running with zero stop-loss.

    Columns: algo | server | count of 0 SL accounts | operator_name

    Counts distinct userid per algo/server where the SL column is numerically
    zero. A blank SL cell means "no SL recorded" and is NOT counted.

    The caller passes the All Users frame already filtered by the active DTE
    mode; no server exclusions are applied, so NOT RUNNING / DLR ACC / Z SERVER
    accounts still surface here.
    """
    empty = pd.DataFrame(columns=ZERO_SL_COLUMNS)

    if SL_COL not in df_all.columns:
        logger.warning("No 'SL' column found in the main sheet -- 0 SL tab will be empty.")
        return empty
    if df_all.empty:
        return empty

    sl_numeric = coerce_sl(df_all[SL_COL])
    zero_sl = df_all[sl_numeric == 0]

    logger.info(
        "0 SL: %d of %d row(s) have a numeric zero SL (%d blank/non-numeric).",
        len(zero_sl), len(df_all), int(sl_numeric.isna().sum()),
    )
    if zero_sl.empty:
        return empty

    summary = (
        zero_sl.groupby(["algo", "server"], dropna=False)["userid"]
        .nunique()
        .reset_index()
        .rename(columns={"userid": ZERO_SL_COUNT_COL})
    )
    summary = attach_operator(summary, operator_lookup)
    return summary[ZERO_SL_COLUMNS].sort_values(["algo", "server"]).reset_index(drop=True)


# -----------------------------
# FIX (CR) COLUMN PARSING
# -----------------------------
def coerce_fix_cr(df: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Coerce the raw FIX (CR) column into a clean numeric Series.

    A cell is treated as:
      - blank   -> account simply has no fixed allocation; skipped silently.
      - invalid -> cell is populated but unusable (non-numeric text, zero,
                   or negative). Excluded from the Fixed tab AND reported
                   back so the caller can surface it as a data-entry issue.
      - valid   -> positive number, kept.

    Returns:
        values:  float Series aligned to df.index, NaN for blank/invalid rows.
        invalid: DataFrame of the offending rows (may be empty).
    """
    raw = df[FIX_CR_COL]
    raw_str = raw.astype(str).str.strip()
    is_blank = raw.isna() | raw_str.str.lower().isin(FIX_CR_BLANK_TOKENS)

    numeric = pd.to_numeric(raw, errors="coerce")
    is_invalid = ~is_blank & (numeric.isna() | (numeric <= 0))

    invalid = df.loc[is_invalid, ["userid", "alias", "server", "algo"]].copy()
    invalid["fix_cr_raw"] = raw_str[is_invalid]
    invalid["reason"] = np.where(
        numeric[is_invalid].isna(), "Non-numeric value", "Zero or negative value"
    )

    if not invalid.empty:
        logger.warning(
            "Ignoring %d row(s) with an unusable FIX (CR) value: %s",
            len(invalid), invalid["userid"].tolist(),
        )

    return numeric.where(~is_invalid), invalid.reset_index(drop=True)


# -----------------------------
# FIXED TAB
# -----------------------------
def build_fixed_tab(df_all: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build the Fixed Allocation tab from the FIX (CR) column.

    expected_allocation = FIX (CR) x FIX_CR_MULTIPLIER
    (1 -> 100,000 | 1.6 -> 160,000 | 0.8 -> 80,000 | 3 -> 300,000)

    Only rows with a positive FIX (CR) value participate. A blank cell means
    the account is not on a fixed allocation and is skipped.
    Always excludes DLR ACC / NOT RUNNING / Z SERVER regardless of mode.

    Returns:
        fixed:   the Fixed tab rows.
        invalid: rows whose FIX (CR) cell was populated but unusable.
    """
    empty_fixed = pd.DataFrame(columns=FIXED_TAB_COLUMNS)
    empty_invalid = pd.DataFrame(columns=FIX_CR_INVALID_COLUMNS)

    if FIX_CR_COL not in df_all.columns:
        logger.warning(
            "No 'FIX (CR)' column found in the main sheet -- Fixed tab will be empty."
        )
        return empty_fixed, empty_invalid

    # Always exclude irrelevant server groups
    df = df_all[~df_all["server"].isin(FIXED_EXCLUDE_SERVERS)].copy()
    if df.empty:
        return empty_fixed, empty_invalid

    df[FIX_CR_COL], invalid = coerce_fix_cr(df)
    df = df[df[FIX_CR_COL].notna()].copy()
    if df.empty:
        logger.info("Fixed tab: no accounts with a FIX (CR) value.")
        return empty_fixed, invalid

    # round(), not floor(): floor on a binary-float product can land one
    # rupee low (e.g. x.xx * 100000 -> ...999.9999). The expected amount is
    # always a whole rupee figure, so rounding is the correct semantic.
    df["expected_allocation"] = (df[FIX_CR_COL] * FIX_CR_MULTIPLIER).round(0)
    df["_match"] = df["expected_allocation"] == df["allocation"]
    df["status"] = df["_match"].map({True: "Match", False: "Mismatch"})

    result = df[[
        "userid", "alias", "server", "algo", FIX_CR_COL,
        "expected_allocation", "allocation", "status", "_match",
    ]].rename(columns={"allocation": "actual_allocation"})

    logger.info(
        "Fixed tab: %d FIX account(s), %d mismatch(es), %d invalid FIX (CR) value(s).",
        len(result), int((~result["_match"]).sum()), len(invalid),
    )
    return result.reset_index(drop=True), invalid


def _format_fix_cr(value: float) -> str:
    """Render FIX (CR) without noisy trailing zeros: 1.0 -> '1', 1.6 -> '1.6'."""
    return "" if pd.isna(value) else f"{value:g}"


def style_fixed_tab(df: pd.DataFrame):
    """Bold red + white on mismatches; dark green + light on matches."""
    def row_style(row):
        if not df.at[row.name, "_match"]:
            return ["background-color: #b71c1c; color: #ffffff"] * len(row)
        return ["background-color: #1b5e20; color: #e8f5e9"] * len(row)

    display = df.drop(columns=["_match"])
    return display.style.apply(row_style, axis=1).format({
        FIX_CR_COL: _format_fix_cr,
        "expected_allocation": "{:,.0f}",
        "actual_allocation": "{:,.0f}",
    })


# -----------------------------
# ALLOCATION TAB (DTE-based)
# -----------------------------
def build_allocation_tab(df_all: pd.DataFrame, mode: str) -> pd.DataFrame:
    """
    Flag accounts whose allocation falls below the mode threshold.
    4DTE -> threshold = 100,000
    1DTE -> threshold =  60,000
    0DTE -> not applicable
    """
    threshold = ALLOC_THRESHOLD.get(mode)
    if threshold is None:
        return pd.DataFrame(columns=["server", "algo", "userid", "alias", "allocation"])
    flagged = df_all[
        (df_all["allocation"] > 0) &
        (df_all["allocation"] < threshold)
    ].copy()
    return (
        flagged[["server", "algo", "userid", "alias", "allocation"]]
        .sort_values(["server", "allocation"])
        .reset_index(drop=True)
    )


# -----------------------------
# EXPORT
# -----------------------------
def to_excel(
    diff: pd.DataFrame,
    not_found: pd.DataFrame,
    extra: pd.DataFrame,
    duplicate: pd.DataFrame,
    fixed: pd.DataFrame,
    allocation: pd.DataFrame,
    not_found_summary: Optional[pd.DataFrame] = None,
    running_summary: Optional[pd.DataFrame] = None,
    zero_sl: Optional[pd.DataFrame] = None,
) -> io.BytesIO:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        if running_summary is not None:
            running_summary.to_excel(writer, sheet_name="Summary", index=False)
        if zero_sl is not None:
            zero_sl.to_excel(writer, sheet_name="0 SL", index=False)
        diff.to_excel(writer, sheet_name="Difference", index=False)
        not_found.to_excel(writer, sheet_name="Not Found", index=False)
        if not_found_summary is not None:
            not_found_summary.to_excel(writer, sheet_name="Not Found Summary", index=False)
        extra.to_excel(writer, sheet_name="Extra", index=False)
        duplicate.to_excel(writer, sheet_name="Duplicate", index=False)

        # Fixed tab
        fixed_export = fixed.drop(columns=["_match"], errors="ignore").copy()
        fixed_export.to_excel(writer, sheet_name="Fixed", index=False)
        wb = writer.book
        ws = writer.sheets["Fixed"]
        red_fmt = wb.add_format({"bg_color": "#b71c1c", "font_color": "#ffffff", "bold": True})
        green_fmt = wb.add_format({"bg_color": "#1b5e20", "font_color": "#e8f5e9"})
        if not fixed.empty and "_match" in fixed.columns:
            for row_idx, match_val in enumerate(fixed["_match"], start=1):
                ws.set_row(row_idx, None, red_fmt if not match_val else green_fmt)

        allocation.to_excel(writer, sheet_name="Allocation", index=False)

    output.seek(0)
    return output


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Morning Sheet Comparison", layout="wide")

# Tighten default Streamlit spacing so tables/sections read as a clean,
# client-ready report instead of the default airy dashboard look.
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.75rem;
            padding-bottom: 1.5rem;
        }
        h1, h2, h3 { margin-bottom: 0.25rem; }
        [data-testid="stCaptionContainer"] { margin-bottom: 0.5rem; }
        hr { margin: 0.6rem 0 1rem 0; }
        div[data-testid="stMetric"] {
            background-color: rgba(127, 127, 127, 0.07);
            border-radius: 6px;
            padding: 8px 10px;
        }
        div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stDataFrame"]) {
            margin-bottom: 0.25rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

def render_table(df, hide_index: bool = True) -> None:
    """
    Render any table (plain DataFrame or a pandas Styler from .style.apply()
    / .format()) sized to its actual column count, instead of always
    stretching to the full page width. A 2-4 column table stretched with
    use_container_width=True leaves a large, unprofessional empty gap on
    wide screens; a wide table (~9+ columns) genuinely needs the space.
    This scales the rendered width to fit the content either way.
    """
    underlying = df.data if hasattr(df, "data") else df
    n_cols = max(len(underlying.columns), 1)
    width_fraction = min(1.0, max(0.3, 0.12 + 0.10 * n_cols))
    if width_fraction >= 0.98:
        st.dataframe(df, use_container_width=True, hide_index=hide_index)
    else:
        col, _ = st.columns([width_fraction, 1 - width_fraction])
        with col:
            st.dataframe(df, use_container_width=True, hide_index=hide_index)


def render_login_check(mode: str) -> None:
    """Existing morning reconciliation: All Users vs Running Users."""
    st.info(f"Running in **{mode}** mode. Change mode from the sidebar.")

    col1, col2 = st.columns(2)
    with col1:
        file_all = st.file_uploader("Upload All Users Excel", type=["xlsx"])
    with col2:
        file_run = st.file_uploader("Upload Running Users", type=["csv", "xlsx"])


    if file_all and file_run:

        st.success("Files uploaded successfully. Processing...")

        df_all = load_all_users(file_all)
        df_run = load_running_users(file_run)

        if df_all is None:
            st.stop()

        df_all = normalize_values(normalize_columns(df_all), is_running=False)
        df_run = normalize_values(normalize_columns(df_run), is_running=True)

        if OPERATOR_COL not in df_all.columns:
            st.warning(
                "No **Operator Name** column found in the All Users sheet -- "
                "the operator column will be blank in every table."
            )
        if SL_COL not in df_all.columns:
            st.warning(
                "No **SL** column found in the All Users sheet -- the 0 SL tab will be empty."
            )

        # Raw running snapshot (as uploaded, before dedup / server exclusions) --
        # used for the Summary tab pivot.
        df_run_raw = df_run.copy()

        df_all = apply_mode_filter(df_all, mode)

        # Snapshot right after the DTE mode filter only (no dedup / server
        # exclusions yet) -- used for the Summary tab's "All user" count.
        df_all_mode_filtered = df_all.copy()

        dup_all = get_duplicates(df_all, "All User")
        dup_run = get_duplicates(df_run, "Running")
        duplicate_tab = pd.concat([dup_all, dup_run], ignore_index=True)

        df_all = remove_duplicates(df_all)
        df_run = remove_duplicates(df_run)

        cfg = MODE_FILTERS[mode]
        if cfg["apply_base_exclusions"]:
            df_all_clean, extra_all = split_extra(df_all, EXCLUDE_ALL_USERS, "AllUser")
            df_run_clean, extra_run = split_extra(df_run, EXCLUDE_RUNNING, "Running")
            extra_tab = pd.concat([extra_all, extra_run], ignore_index=True)[COLS + ["not found in"]]
        else:
            df_all_clean = df_all
            df_run_clean = df_run
            extra_tab = pd.DataFrame(columns=COLS + ["not found in"])

        diff_tab = get_difference(df_all_clean, df_run_clean)
        not_found_tab = get_not_found(df_all_clean, df_run_clean)

        # Fixed tab: mode-filtered df_all, but DLR ACC / NOT RUNNING stripped inside
        fixed_tab, fix_cr_invalid = build_fixed_tab(df_all)

        allocation_tab = build_allocation_tab(df_all_clean, mode)

        not_found_summary_tab = build_not_found_summary(not_found_tab)
        running_pivot_tab = build_running_pivot(df_all_mode_filtered, df_run_raw)

        # --- OPERATOR ---
        # The userid universe for the operator lookup and the 0 SL pivot is the
        # All Users sheet as filtered by the active DTE mode.
        operator_lookup = build_operator_lookup(df_all_mode_filtered)

        zero_sl_tab = build_zero_sl_pivot(df_all_mode_filtered, operator_lookup)

        # Append operator_name as the last column of every table. The Difference
        # tab has no plain algo/server, so resolve it off the All Users side.
        diff_tab = attach_operator(diff_tab, operator_lookup, "algo_all", "server_all")
        not_found_tab = attach_operator(not_found_tab, operator_lookup)
        extra_tab = attach_operator(extra_tab, operator_lookup)
        duplicate_tab = attach_operator(duplicate_tab, operator_lookup)[
            ["userid", "Found in", OPERATOR_COL]
        ]
        allocation_tab = attach_operator(allocation_tab, operator_lookup)
        not_found_summary_tab = attach_operator(not_found_summary_tab, operator_lookup)
        running_pivot_tab = attach_operator(running_pivot_tab, operator_lookup)

        # Fixed tab keeps _match last so the styler can still find it.
        if not fixed_tab.empty:
            fixed_tab = attach_operator(fixed_tab, operator_lookup)
            fixed_tab = fixed_tab[
                [c for c in fixed_tab.columns if c != "_match"] + ["_match"]
            ]
        fix_cr_invalid = attach_operator(fix_cr_invalid, operator_lookup)

        # --- SUMMARY ---
        st.markdown(f"### Summary -- {mode}")
        m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
        m1.metric("Differences", len(diff_tab))
        m2.metric("Not Found", len(not_found_tab))
        m3.metric("Extra Accounts", len(extra_tab))
        m4.metric("Duplicates", len(duplicate_tab))
        m5.metric("FIX Accounts", len(fixed_tab))
        fixed_mismatches = int((~fixed_tab["_match"]).sum()) if not fixed_tab.empty else 0
        m6.metric("FIX Mismatches", fixed_mismatches)
        zero_sl_accounts = int(zero_sl_tab[ZERO_SL_COUNT_COL].sum()) if not zero_sl_tab.empty else 0
        m7.metric("0 SL Accounts", zero_sl_accounts)
        st.markdown("---")

        # --- TABS ---
        tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
            ["Summary", "Difference", "Not Found", "Extra", "Duplicate",
             "Fixed", "Allocation", "0 SL"]
        )

        with tab0:
            st.subheader(f"Running Users Pivot -- {mode}")
            st.caption(
                "Distinct userid count per Algo / Server. **All user** = from the "
                f"All Users sheet filtered only by the **{mode}** RunningType/RunningDays "
                "rule (no dedup / server exclusions applied here). **Running** = from "
                "the Running Users sheet exactly as uploaded (before dedup / server exclusions)."
            )
            if running_pivot_tab.empty:
                st.info("No data available to summarize.")
            else:
                render_table(running_pivot_tab)

        with tab1:
            st.subheader("Differences Between Sheets")
            render_table(diff_tab)

        with tab2:
            st.subheader("Missing Users")
            render_table(not_found_tab)

            st.markdown("---")
            st.subheader("Missing Users -- Summary by Algo / Server")
            if not_found_summary_tab.empty:
                st.info("No missing users to summarize.")
            else:
                render_table(not_found_summary_tab)

        with tab3:
            st.subheader("Excluded / Extra Accounts")
            render_table(extra_tab)

        with tab4:
            st.subheader("Duplicate User IDs")
            render_table(duplicate_tab)

        with tab5:
            st.subheader("Fixed Allocation Accounts")
            st.caption(
                "Accounts with a value in the **FIX (CR)** column of the main sheet. "
                f"Expected allocation = FIX (CR) x {FIX_CR_MULTIPLIER:,} "
                "(1 -> 100,000 | 1.6 -> 160,000 | 0.8 -> 80,000). "
                "DLR ACC / NOT RUNNING / Z SERVER are excluded. "
                f"Showing **{len(fixed_tab)}** FIX accounts "
                f"(**{fixed_mismatches}** mismatches) for **{mode}** mode."
            )

            if not fix_cr_invalid.empty:
                st.warning(
                    f"{len(fix_cr_invalid)} account(s) have an unusable **FIX (CR)** "
                    "value and were excluded from this tab. Please correct the sheet."
                )
                render_table(fix_cr_invalid)

            if fixed_tab.empty:
                st.info("No FIX accounts found for this mode.")
            else:
                c1, c2, c3 = st.columns(3)
                c1.metric("Total FIX Accounts", len(fixed_tab))
                c2.metric("Matching", int(fixed_tab["_match"].sum()))
                c3.metric("Mismatched", fixed_mismatches)
                render_table(style_fixed_tab(fixed_tab))

        with tab6:
            threshold = ALLOC_THRESHOLD.get(mode)
            if threshold is None:
                st.info("Allocation check is not applicable for 0DTE mode.")
            else:
                st.subheader(f"Low Allocation Accounts -- {mode}")
                st.caption(
                    f"Accounts with allocation > 0 and < {threshold:,} in {mode} mode. "
                    f"Found {len(allocation_tab)} account(s)."
                )
                if allocation_tab.empty:
                    st.success(f"All accounts meet the {threshold:,} allocation threshold.")
                else:
                    def _style_alloc_row(row):
                        return ["background-color: #e65100; color: #ffffff; font-weight: bold"] * len(row)

                    render_table(
                        allocation_tab.style.apply(_style_alloc_row, axis=1).format(
                            {"allocation": "{:,.0f}"}
                        )
                    )

        with tab7:
            st.subheader(f"0 SL Accounts -- {mode}")
            st.caption(
                "Accounts running with **zero stop-loss**: distinct userid count per "
                f"Algo / Server where the **SL** column is numerically 0, across the "
                f"**{mode}** userid set. Blank SL cells mean 'no SL recorded' and are "
                "**not** counted. All servers are included -- no exclusions applied here."
            )
            if zero_sl_tab.empty:
                st.success("No accounts are running with 0 SL.")
            else:
                c1, c2 = st.columns(2)
                c1.metric("0 SL Accounts", zero_sl_accounts)
                c2.metric("Algo / Server Groups", len(zero_sl_tab))

                def _style_zero_sl_row(row):
                    return ["background-color: #4a148c; color: #ffffff; font-weight: bold"] * len(row)

                render_table(
                    zero_sl_tab.style.apply(_style_zero_sl_row, axis=1).format(
                        {ZERO_SL_COUNT_COL: "{:,.0f}"}
                    )
                )

        # --- DOWNLOAD ---
        excel_bytes = to_excel(
            diff_tab, not_found_tab, extra_tab, duplicate_tab, fixed_tab, allocation_tab,
            not_found_summary=not_found_summary_tab, running_summary=running_pivot_tab,
            zero_sl=zero_sl_tab,
        )
        st.download_button(
            "Download Full Report",
            data=excel_bytes,
            file_name=f"user_comparison_{mode}.xlsx",
        )


# -----------------------------
# ALLOCATION CHECK SECTION
# -----------------------------
def load_all_users_for_section(file) -> Optional[pd.DataFrame]:
    """Load + normalize the All Users Main sheet. Returns None on failure."""
    raw = load_all_users(file)
    if raw is None:
        return None
    return normalize_values(normalize_columns(raw), is_running=False)


def load_jainam_sheet(file, rules: dict) -> Optional[pd.DataFrame]:
    """
    Read the Jainam sheet from the SAME All Users workbook.

    Returns None if the sheet is absent -- JA accounts are then reported as
    mismatches with an explanatory remark rather than silently skipped.
    """
    sheet_name = ac.jainam_config(rules)["sheet_name"]
    try:
        excel = pd.ExcelFile(file)
        actual = next(
            (s for s in excel.sheet_names if s.strip().lower() == sheet_name.strip().lower()),
            None,
        )
        if actual is None:
            logger.warning("No '%s' sheet in the All Users workbook.", sheet_name)
            return None
        return pd.read_excel(excel, sheet_name=actual)
    except Exception as exc:  # noqa: BLE001 - surfaced to the user
        st.warning(f"Could not read the '{sheet_name}' sheet: {exc}")
        logger.exception("Failed to read the Jainam sheet")
        return None


def load_running_for_allocation(file) -> Optional[pd.DataFrame]:
    """
    Load the Running file for the Allocation Check.

    Only userid and capital are needed. capital must NOT go through
    normalize_values(), which divides allocation by 100 -- that scaling does
    not apply to capital.
    """
    try:
        raw = pd.read_csv(file) if file.name.lower().endswith(".csv") else pd.read_excel(file)
    except Exception as exc:  # noqa: BLE001 - surfaced to the user
        st.error(f"Failed to read the Running Users file: {exc}")
        logger.exception("Failed to read running file for Allocation Check")
        return None

    df = normalize_columns(raw)
    if "userid" not in df.columns:
        st.error("Running Users file has no 'userId' column.")
        return None
    if ac.CAPITAL_COL not in df.columns:
        st.error(
            "Running Users file has no **capital** column, which the Allocation "
            f"Check requires. Columns found: {sorted(df.columns)[:15]}"
        )
        return None

    df["userid"] = (
        df["userid"].astype(str).str.strip().str.replace(" ", "", regex=False).str.upper()
    )
    return df


def render_allocation_check(mode: str) -> None:
    """Expected trading allocation from running capital vs the All Users sheet."""
    try:
        rules = ac.load_rules()
    except ac.AllocationRulesError as exc:
        st.error(f"Allocation rules could not be loaded:\n\n{exc}")
        st.info(f"Expected at: `{ac.resolve_rules_path()}`")
        return

    st.info(
        f"Running in **{mode}** mode. Expected allocation = round-half-up("
        f"capital x SubCategory %, {rules['rounding']['basis']:,}) / "
        f"{rules['rounding']['divisor']}."
    )

    with st.expander("Active rules -- click to view and edit"):
        st.caption(
            "Enter a **whole percentage** of running capital: `100` = 100%, "
            "`60` = 60%, **`0` = Exclude**. Set Method to *Jainam sheet* for "
            "SubCategories checked against the Jainam sheet instead of capital. "
            "Add or remove rows as needed, then Save."
        )

        editor_key = "rules_editor"
        edited = st.data_editor(
            ac.rules_to_editor(rules),
            key=editor_key,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            column_config={
                ac.EDITOR_SUBCATEGORY: st.column_config.TextColumn(
                    "SubCategory", help="Must match the SubCategory column in the sheet.",
                    required=True, width="small",
                ),
                ac.EDITOR_PCT: st.column_config.NumberColumn(
                    "% of capital", help="100 = 100%, 60 = 60%, 0 = Exclude.",
                    min_value=0, max_value=100, step=1, format="%d", width="small",
                ),
                ac.EDITOR_METHOD: st.column_config.SelectboxColumn(
                    "Method", options=ac.METHOD_LABELS,
                    help="Capital % uses running capital. Jainam sheet checks against "
                         "the Jainam tab of the All Users workbook.",
                    width="medium",
                ),
                ac.EDITOR_NOTE: st.column_config.TextColumn("note", width="large"),
            },
        )

        b1, b2, _ = st.columns([1, 1, 3])
        if b1.button("Save rules", type="primary", key="save_rules"):
            try:
                updated = dict(rules)
                updated["subcategories"] = ac.editor_to_subcategories(edited)
                path = ac.save_rules(updated)
                st.success(f"Rules saved to `{path}`. Re-running the check.")
                st.rerun()
            except ac.AllocationRulesError as exc:
                st.error(f"Not saved -- {exc}")
            except Exception as exc:  # noqa: BLE001 - must not lose the user's edits
                st.error(f"Not saved -- unexpected error: {exc}")
                logger.exception("Failed to save allocation rules")

        if b2.button("Reload from file", key="reload_rules"):
            st.rerun()

        st.caption(
            f"Loaded from `{ac.resolve_rules_path()}` (version {rules.get('version')}). "
            "A backup of the previous file is kept as `allocation_rules.json.bak`."
        )

    prev_req = ac.previous_day_requirement(mode, rules)
    show_prev = prev_req in (ac.PREV_REQUIRED, ac.PREV_OPTIONAL)

    cols = st.columns(3 if show_prev else 2)
    with cols[0]:
        file_all = st.file_uploader("Upload Today's All Users Excel", type=["xlsx"],
                                    key="alloc_all")
    with cols[1]:
        file_run = st.file_uploader("Upload Running Users", type=["csv", "xlsx"],
                                    key="alloc_run")
    file_prev = None
    if show_prev:
        with cols[2]:
            label = (
                "Upload Previous Day All Users (required)"
                if prev_req == ac.PREV_REQUIRED
                else "Upload Previous Day All Users (optional)"
            )
            file_prev = st.file_uploader(label, type=["xlsx"], key="alloc_prev")

    if prev_req == ac.PREV_REQUIRED:
        st.caption(
            f"**{mode}** requires all three files. INT accounts are checked against "
            "running capital; POS accounts are checked against the previous day's allocation."
        )
    elif prev_req == ac.PREV_OPTIONAL:
        st.caption(
            f"**{mode}**: without the previous-day sheet every account is checked "
            "against running capital. With it, POS + DAILY accounts are instead "
            "checked against the previous day's allocation."
        )

    if not (file_all and file_run):
        st.caption("Upload the required files to run the check.")
        return

    if prev_req == ac.PREV_REQUIRED and file_prev is None:
        st.error(
            f"**{mode} mode requires the previous day's All Users sheet.** "
            "POS accounts are checked against the previous day's allocation, so "
            "the check cannot run without it. Upload all three files."
        )
        return

    df_all = load_all_users_for_section(file_all)
    if df_all is None:
        return
    df_run = load_running_for_allocation(file_run)
    if df_run is None:
        return

    df_prev = None
    if file_prev is not None:
        df_prev = load_all_users_for_section(file_prev)
        if df_prev is None:
            st.error("Could not read the previous-day All Users sheet.")
            return
        st.success(
            f"Previous-day sheet loaded ({len(df_prev)} rows). "
            "POS accounts will be checked against it."
        )

    df_jainam = load_jainam_sheet(file_all, rules)
    jainam_sheet_name = ac.jainam_config(rules)["sheet_name"]

    try:
        tables = ac.build_allocation_check(
            df_all, df_run, mode, rules, df_prev=df_prev, df_jainam=df_jainam
        )
        in_scope, _ = ac.apply_dte_scope(df_all, mode, rules)
    except ac.AllocationRulesError as exc:
        st.error(str(exc))
        return

    if df_jainam is None and not tables["jexceptions"].empty:
        st.warning(
            f"{len(tables['jexceptions'])} JA account(s) in scope but no "
            f"**{jainam_sheet_name}** sheet was found in the All Users workbook. "
            "They are reported as mismatches."
        )

    result = tables["result"]
    n_match = int((result["status"] == "Match").sum()) if not result.empty else 0
    n_mismatch = len(result) - n_match

    ok, msg = ac.reconcile(len(in_scope), tables)
    if not ok:
        st.error(f"Internal reconciliation failed -- {msg}. Do not trust these numbers.")

    dup_ids = in_scope["userid"][in_scope["userid"].duplicated(keep=False)].unique()
    if len(dup_ids):
        st.warning(
            f"{len(dup_ids)} userid(s) appear more than once in the in-scope All "
            f"Users rows: {', '.join(map(str, dup_ids[:10]))}"
            f"{' ...' if len(dup_ids) > 10 else ''}. Each occurrence is checked "
            "separately, so they appear as repeated rows below."
        )

    prevday = tables["prevday_result"]
    p_match = int((prevday["status"] == "Match").sum()) if not prevday.empty else 0
    p_mismatch = len(prevday) - p_match

    st.markdown(f"### Allocation Check Summary -- {mode}")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("In Scope", len(in_scope))
    m2.metric("Capital Rule", len(result), delta=f"{n_mismatch} mismatch" if n_mismatch else None,
              delta_color="inverse")
    m3.metric("Capital Match", n_match)
    m4.metric("Prev-Day Rule", len(prevday),
              delta=f"{p_mismatch} mismatch" if p_mismatch else None, delta_color="inverse")
    m5.metric("Prev-Day Match", p_match)
    m6.metric("Unmapped SubCat", len(tables["unknown_subcategory"]))
    st.caption(msg)
    st.markdown("---")

    consolidated = ac.build_consolidated(tables, in_scope)

    tc, t0, tp, tj, tf, t1, t2, t3 = st.tabs(
        ["All Accounts", "Capital Rule", "Previous Day", "Jainam (JA)", "Fixed",
         "By SubCategory", "Unmapped / Excluded", "Not Checked"]
    )

    with tc:
        st.subheader(f"All In-Scope Accounts -- {mode}")
        st.caption(
            "One row per in-scope account across both check methods. "
            "**rule** is the percentage for capital-rule accounts, 'Previous Day' "
            "for previous-day accounts. **expected allocation** is whichever value "
            "the account was measured against. **category capital** and **capital** "
            "apply only to capital-rule rows."
        )
        if consolidated.empty:
            st.info("No accounts in scope for this mode.")
        else:
            counts = ac.consolidated_status_counts(consolidated)
            cols = st.columns(max(len(counts), 1))
            for col, (_, r) in zip(cols, counts.iterrows()):
                col.metric(r["status"], int(r["accounts"]))

            status_filter = st.multiselect(
                "Filter by status",
                options=list(counts["status"]),
                default=[s for s in (ac.STATUS_MISMATCH, ac.STATUS_NEW_USER)
                         if s in set(counts["status"])] or list(counts["status"]),
                key="cons_filter",
            )
            cview = (
                consolidated[consolidated["status"].isin(status_filter)]
                if status_filter else consolidated
            )

            _palette = {
                ac.STATUS_MISMATCH: "background-color: #b71c1c; color: #ffffff",
                ac.STATUS_MATCH: "background-color: #1b5e20; color: #e8f5e9",
                ac.STATUS_NEW_USER: "background-color: #0d47a1; color: #e3f2fd",
                ac.STATUS_NOT_CHECKED: "background-color: #424242; color: #eeeeee",
            }

            def _cstyle(row):
                return [_palette.get(cview.at[row.name, "status"], "")] * len(row)

            if cview.empty:
                st.info("No rows for the selected status.")
            else:
                st.caption(f"Showing {len(cview):,} of {len(consolidated):,} accounts.")
                render_table(cview.style.apply(_cstyle, axis=1).format({
                    "maxloss": "{:,.0f}",
                    "allocation": "{:,.0f}",
                    "expected allocation": "{:,.0f}",
                    "category capital": "{:,.0f}",
                    "capital": "{:,.0f}",
                }, na_rep=""))

    with t0:
        st.subheader(f"Expected vs Actual Allocation -- {mode}")
        if result.empty:
            st.info("No accounts to check in this mode.")
        else:
            only_mismatch = st.checkbox("Show mismatches only", value=True)
            view = result[result["status"] == "Mismatch"] if only_mismatch else result

            def _style(row):
                bad = view.at[row.name, "status"] == "Mismatch"
                colour = ("background-color: #b71c1c; color: #ffffff"
                          if bad else "background-color: #1b5e20; color: #e8f5e9")
                return [colour] * len(row)

            if view.empty:
                st.success("No mismatches.")
            else:
                render_table(view.style.apply(_style, axis=1).format({
                    "pct": "{:g}%",
                    ac.CAPITAL_COL: "{:,.0f}",
                    "category_capital": "{:,.0f}",
                    "rounded_capital": "{:,.0f}",
                    "expected_allocation": "{:,.0f}",
                    "actual_allocation": "{:,.0f}",
                    "difference": "{:,.0f}",
                }))

    with tp:
        st.subheader(f"Today vs Previous Day Allocation -- {mode}")
        st.caption(
            "A straight allocation-vs-allocation comparison against the previous "
            "day's All Users sheet. SubCategory percentages and running capital "
            "play no part here -- the two allocations must simply be equal."
        )
        if prevday.empty and tables["prevday_new"].empty:
            if not show_prev:
                st.info(f"The previous-day check does not apply in {mode} mode.")
            elif df_prev is None:
                st.info(
                    "No previous-day sheet uploaded, so every account was checked "
                    "against running capital instead."
                )
            else:
                st.info("No accounts routed to the previous-day check in this mode.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Compared", len(prevday))
            c2.metric("Match", p_match)
            c3.metric("Mismatch", p_mismatch)

            if not prevday.empty:
                only_mm = st.checkbox("Show mismatches only", value=True, key="prev_mm")
                pview = prevday[prevday["status"] == "Mismatch"] if only_mm else prevday

                def _pstyle(row):
                    bad = pview.at[row.name, "status"] == "Mismatch"
                    colour = ("background-color: #b71c1c; color: #ffffff"
                              if bad else "background-color: #1b5e20; color: #e8f5e9")
                    return [colour] * len(row)

                if pview.empty:
                    st.success("No mismatches against the previous day.")
                else:
                    render_table(pview.style.apply(_pstyle, axis=1).format({
                        "previous_allocation": "{:,.0f}",
                        "today_allocation": "{:,.0f}",
                        "difference": "{:,.0f}",
                    }))

            new_accts = tables["prevday_new"]
            st.markdown("---")
            st.markdown(f"**New accounts -- not in the previous-day sheet** ({len(new_accts)})")
            st.caption("Reported separately, not counted as mismatches.")
            if new_accts.empty:
                st.success("None.")
            else:
                render_table(new_accts.drop(columns=["previous_allocation", "difference"]))

    with tj:
        jainam = tables["jainam_result"]
        st.subheader(f"JA Accounts vs the {jainam_sheet_name} Sheet")
        st.caption(
            f"SubCategory **JA** accounts are checked against the **{jainam_sheet_name}** "
            "sheet inside the same All Users workbook, not against running capital. "
            f"Expected allocation = `ALLOCATION x {ac.jainam_config(rules)['multiplier']:,}` "
            "(4 -> 4,00,000). An ALLOCATION of 0 means the expected allocation IS zero. "
            "All rows are used regardless of Date; the sheet's Total row is dropped. "
            "This check overrides the mode routing."
        )
        if jainam.empty:
            st.info("No JA accounts in scope for this mode.")
        else:
            j_match = int((jainam["status"] == ac.STATUS_MATCH).sum())
            j1, j2, j3 = st.columns(3)
            j1.metric("JA Accounts", len(jainam))
            j2.metric("Match", j_match)
            j3.metric("Mismatch", len(jainam) - j_match)

            def _jstyle(row):
                bad = jainam.at[row.name, "status"] != ac.STATUS_MATCH
                colour = ("background-color: #b71c1c; color: #ffffff"
                          if bad else "background-color: #1b5e20; color: #e8f5e9")
                return [colour] * len(row)

            render_table(jainam.style.apply(_jstyle, axis=1).format({
                "jainam_allocation": "{:,.0f}",
                "expected_allocation": "{:,.0f}",
                "actual_allocation": "{:,.0f}",
                "difference": "{:,.0f}",
            }, na_rep=""))

    with tf:
        fixed = tables["fix_result"]
        fix_mult = ac.fix_config(rules)["multiplier"]
        st.subheader(f"Fixed Allocation Accounts -- {mode}")
        st.caption(
            f"Accounts with a value in the **FIX (CR)** column. Expected "
            f"allocation = `FIX (CR) x {fix_mult:,}` (3 -> 3,00,000). This rule has "
            "the **highest precedence** -- it overrides the capital, previous-day "
            "and Jainam rules."
        )
        if fixed.empty:
            st.info("No fixed-allocation accounts in scope for this mode.")
        else:
            f_match = int((fixed["status"] == ac.STATUS_MATCH).sum())
            f1, f2, f3 = st.columns(3)
            f1.metric("Fixed Accounts", len(fixed))
            f2.metric("Match", f_match)
            f3.metric("Mismatch", len(fixed) - f_match)

            def _fstyle(row):
                bad = fixed.at[row.name, "status"] != ac.STATUS_MATCH
                colour = ("background-color: #b71c1c; color: #ffffff"
                          if bad else "background-color: #1b5e20; color: #e8f5e9")
                return [colour] * len(row)

            render_table(fixed.style.apply(_fstyle, axis=1).format({
                "fix_cr": lambda v: "" if pd.isna(v) else f"{v:g}",
                "expected_allocation": "{:,.0f}",
                "actual_allocation": "{:,.0f}",
                "difference": "{:,.0f}",
            }, na_rep=""))

        if not tables["fix_invalid"].empty:
            st.warning(
                f"{len(tables['fix_invalid'])} account(s) have an unusable "
                "**FIX (CR)** value (0, negative or non-numeric). They were not "
                "checked by any rule -- correct the sheet."
            )
            render_table(tables["fix_invalid"])

    with t1:
        st.subheader("Match / Mismatch by SubCategory")
        st.caption("Capital-rule accounts only.")
        summary = ac.build_summary(result)
        if summary.empty:
            st.info("Nothing to summarise.")
        else:
            render_table(summary)

    with t2:
        st.subheader("SubCategories not defined in the rules file")
        unknown = tables["unknown_subcategory"]
        if unknown.empty:
            st.success("Every account maps to a defined SubCategory.")
        else:
            st.warning(
                f"{len(unknown)} account(s) have a SubCategory that is not in the "
                "rules file. They were NOT checked. Add them to "
                "`config/allocation_rules.json` or correct the sheet."
            )
            counts = (
                unknown[ac.SUBCATEGORY_COL].replace("", "<blank>")
                .value_counts().rename_axis("SubCategory").reset_index(name="accounts")
            )
            render_table(counts)
            render_table(unknown)

        st.markdown("---")
        st.subheader("Excluded by rule")
        st.caption("SubCategories configured with action 'exclude' (PVT / PGB / PPS / PRD).")
        if tables["excluded"].empty:
            st.info("No excluded-SubCategory accounts in this mode.")
        else:
            render_table(tables["excluded"])

        st.markdown("---")
        st.subheader("JExceptions Acc (JA)")
        st.caption("Expected allocation for these is derived separately -- rule pending.")
        if tables["jexceptions"].empty:
            st.info("No JA accounts in this mode.")
        else:
            render_table(tables["jexceptions"])

    with t3:
        st.subheader("In scope but not checked")
        nr, nc = tables["not_in_running"], tables["no_capital"]
        st.markdown(f"**Not present in the Running file** -- {len(nr)} account(s)")
        if nr.empty:
            st.success("None.")
        else:
            render_table(nr)
        st.markdown(f"**No usable capital (blank or <= 0)** -- {len(nc)} account(s)")
        if nc.empty:
            st.success("None.")
        else:
            render_table(nc)

        unroutable = tables["unroutable"]
        st.markdown(f"**Cannot route to a check method** -- {len(unroutable)} account(s)")
        st.caption(
            "In scope, but matched no routing rule for this mode -- usually a blank "
            "or unexpected Running Type."
        )
        if unroutable.empty:
            st.success("None.")
        else:
            render_table(unroutable)

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        consolidated.to_excel(writer, sheet_name="All Accounts", index=False)
        ac.build_summary(result).to_excel(writer, sheet_name="Summary", index=False)
        result.to_excel(writer, sheet_name="Capital Rule", index=False)
        if not result.empty:
            result[result["status"] == "Mismatch"].to_excel(
                writer, sheet_name="Capital Mismatches", index=False)
        tables["prevday_result"].to_excel(writer, sheet_name="Previous Day", index=False)
        if not tables["prevday_result"].empty:
            tables["prevday_result"][tables["prevday_result"]["status"] == "Mismatch"].to_excel(
                writer, sheet_name="Prev Day Mismatches", index=False)
        tables["prevday_new"].to_excel(writer, sheet_name="New Accounts", index=False)
        tables["jainam_result"].to_excel(writer, sheet_name="Jainam JA", index=False)
        tables["fix_result"].to_excel(writer, sheet_name="Fixed", index=False)
        tables["fix_invalid"].to_excel(writer, sheet_name="Fixed Invalid", index=False)
        tables["unroutable"].to_excel(writer, sheet_name="Cannot Route", index=False)
        tables["unknown_subcategory"].to_excel(writer, sheet_name="Unmapped SubCat", index=False)
        tables["excluded"].to_excel(writer, sheet_name="Excluded", index=False)
        tables["jexceptions"].to_excel(writer, sheet_name="JExceptions", index=False)
        tables["not_in_running"].to_excel(writer, sheet_name="Not In Running", index=False)
        tables["no_capital"].to_excel(writer, sheet_name="No Capital", index=False)
    buf.seek(0)
    st.download_button(
        "Download Allocation Check Report",
        data=buf,
        file_name=f"allocation_check_{mode}.xlsx",
        key="alloc_dl",
    )


# -----------------------------
# SIDEBAR / DISPATCH
# -----------------------------
with st.sidebar:
    st.markdown("### Section")
    section = st.radio(
        "Select check:",
        options=[SECTION_LOGIN, SECTION_ALLOCATION],
        index=0,
        help=(
            "Login Check: All Users vs Running Users reconciliation.\n"
            "Allocation Check: expected trading allocation derived from running capital."
        ),
    )
    st.markdown("### Comparison Mode")
    mode = st.radio(
        "Select mode:",
        options=[MODE_0DTE, MODE_1DTE, MODE_4DTE],
        index=0,
        help=(
            "0DTE: all userids except excluded servers.\n"
            "1DTE: RunningType POS & RunningDays in {1DTE/0DTE, DAILY}.\n"
            "4DTE: RunningType POS & RunningDays DAILY."
        ),
    )
    st.caption(f"Section: **{section}**  |  Mode: **{mode}**")

st.markdown(
    f"""
    <h1 style='text-align: center; color: #1f77b4;'>Megaserve Technologies</h1>
    <h3 style='text-align: center; color: gray;'>{section}</h3>
    <hr>
    """,
    unsafe_allow_html=True,
)

if section == SECTION_LOGIN:
    render_login_check(mode)
else:
    render_allocation_check(mode)
