import io
import logging  # noqa
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

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

COLS = ["userid", "alias", "allocation", "max_loss", "server", "algo"]

# Mode keys
MODE_0DTE = "0DTE"
MODE_1DTE = "1DTE"
MODE_4DTE = "4DTE"

# Mode-specific filters applied ONLY on the All Users (Main) sheet.
# Values are stored lowercase because we normalize values to lowercase before matching.
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

# -----------------------------
# COLUMN NORMALIZATION
# -----------------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns.str.lower()
        .str.strip()
        .str.replace(" ", "")
    )

    column_map = {
        "userid": "userid",
        "useralias": "alias",
        "alias": "alias",
        "allocation": "allocation",
        "maxloss": "max_loss",
        "server": "server",
        "algo": "algo",
        # New columns for 1DTE / 4DTE filtering (no rename needed, but kept here for clarity)
        "runningtype": "runningtype",
        "runningdays": "runningdays",
    }

    return df.rename(columns=column_map)

# -----------------------------
# VALUE NORMALIZATION
# -----------------------------
def normalize_values(df: pd.DataFrame, is_running: bool = False) -> pd.DataFrame:
    df = df.copy()

    # USERID → CLEAN + UPPERCASE
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

    # FLOOR INSTEAD OF ROUND
    df["allocation"] = np.floor(df["allocation"])
    df["max_loss"] = np.floor(df["max_loss"])

    # Normalize new filter columns (if present) — lowercase, strip, no inner spaces
    for c in ("runningtype", "runningdays"):
        if c in df.columns:
            df[c] = (
                df[c]
                .astype(str)
                .str.strip()
                .str.replace(" ", "", regex=False)
                .str.lower()
            )

    return df

# -----------------------------
# MODE-SPECIFIC FILTER (All Users sheet only)
# -----------------------------
def apply_mode_filter(df_all: pd.DataFrame, mode: str) -> pd.DataFrame:
    """
    Apply mode-specific column filters to the All Users dataframe.
    1DTE: runningtype in {POS, INT} AND runningdays in {DAILY, 1DTE/0DTE}
    4DTE: runningtype in {POS}        AND runningdays in {DAILY}
    0DTE: no extra filter
    Filtered-out rows are silently dropped.
    """
    cfg = MODE_FILTERS.get(mode, MODE_FILTERS[MODE_0DTE])

    rt_allowed = cfg.get("runningtype")
    rd_allowed = cfg.get("runningdays")

    if rt_allowed is None and rd_allowed is None:
        return df_all

    df = df_all.copy()
    before = len(df)

    # Validate required columns exist
    missing = [
        c for c in ("runningtype", "runningdays")
        if (rt_allowed is not None and c == "runningtype" and c not in df.columns)
        or (rd_allowed is not None and c == "runningdays" and c not in df.columns)
    ]
    if missing:
        st.error(
            f"Mode '{mode}' requires column(s) {missing} in the All Users (Main) sheet. "
            f"Please add them or switch to 0DTE."
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
        if s.lower() == "main":
            sheet_name = s
            break

    if not sheet_name:
        st.error("Main sheet not found in All Users file.")
        return None

    return pd.read_excel(file, sheet_name=sheet_name)

def load_running_users(file) -> pd.DataFrame:
    if file.name.lower().endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)

# -----------------------------
# DUPLICATES
# -----------------------------
def get_duplicates(df: pd.DataFrame, source: str) -> pd.DataFrame:
    dup = df[df.duplicated(subset=["userid"], keep=False)].copy()
    dup["Found in"] = source
    return dup[["userid", "Found in"]]

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

    # Users in All Users but not in Running — pull server/algo from All Users
    missing_in_run = (
        all_df[all_df["userid"].isin(all_ids - run_ids)][["userid", "server", "algo"]]
        .copy()
    )
    missing_in_run["Not found in"] = "Running"

    # Users in Running but not in All Users — pull server/algo from Running
    missing_in_all = (
        run_df[run_df["userid"].isin(run_ids - all_ids)][["userid", "server", "algo"]]
        .copy()
    )
    missing_in_all["Not found in"] = "All User"

    return pd.concat([missing_in_run, missing_in_all], ignore_index=True)[
        ["userid", "server", "algo", "Not found in"]
    ]

# -----------------------------
# EXPORT
# -----------------------------
def to_excel(
    diff: pd.DataFrame,
    not_found: pd.DataFrame,
    extra: pd.DataFrame,
    duplicate: pd.DataFrame,
) -> io.BytesIO:
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        diff.to_excel(writer, sheet_name="Difference", index=False)
        not_found.to_excel(writer, sheet_name="Not Found", index=False)
        extra.to_excel(writer, sheet_name="Extra", index=False)
        duplicate.to_excel(writer, sheet_name="Duplicate", index=False)

    output.seek(0)
    return output

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Morning Sheet Comparison", layout="wide")

# SIDEBAR — MODE SELECTOR
with st.sidebar:
    st.markdown("### Comparison Mode")
    mode = st.radio(
        "Select mode:",
        options=[MODE_0DTE, MODE_1DTE, MODE_4DTE],
        index=0,
        help=(
            "0DTE: Base server exclusions only.\n"
            "1DTE: 0DTE filters + RunningType in {POS, INT} & RunningDays in {DAILY, 1DTE/0DTE}.\n"
            "4DTE: RunningType = POS & RunningDays = DAILY (no base server exclusions)."
        ),
    )
    st.caption(f"Active mode: **{mode}**")

# HEADER
st.markdown(
    """
    <h1 style='text-align: center; color: #1f77b4;'>Megaserve Technologies</h1>
    <h3 style='text-align: center; color: gray;'>Morning Sheet Comparison</h3>
    <hr>
    """,
    unsafe_allow_html=True,
)

st.info(f"Running in **{mode}** mode. Change mode from the sidebar.")

# FILE UPLOAD SECTION
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

    # Normalize columns + values
    df_all = normalize_values(normalize_columns(df_all), is_running=False)
    df_run = normalize_values(normalize_columns(df_run), is_running=True)

    # MODE-SPECIFIC FILTER (applied only on All Users sheet)
    df_all = apply_mode_filter(df_all, mode)

    # Duplicates
    dup_all = get_duplicates(df_all, "All User")
    dup_run = get_duplicates(df_run, "Running")
    duplicate_tab = pd.concat([dup_all, dup_run], ignore_index=True)

    # Remove duplicates
    df_all = remove_duplicates(df_all)
    df_run = remove_duplicates(df_run)

    # Base exclusions (only for modes that opt in)
    cfg = MODE_FILTERS[mode]
    if cfg["apply_base_exclusions"]:
        df_all_clean, extra_all = split_extra(df_all, EXCLUDE_ALL_USERS, "AllUser")
        df_run_clean, extra_run = split_extra(df_run, EXCLUDE_RUNNING, "Running")
        extra_tab = pd.concat([extra_all, extra_run], ignore_index=True)[COLS + ["not found in"]]
    else:
        # 4DTE: no base server exclusions — Extra tab is empty
        df_all_clean = df_all
        df_run_clean = df_run
        extra_tab = pd.DataFrame(columns=COLS + ["not found in"])

    # Difference
    diff_tab = get_difference(df_all_clean, df_run_clean)

    # Not Found
    not_found_tab = get_not_found(df_all_clean, df_run_clean)

    # -----------------------------
    # SUMMARY METRICS
    # -----------------------------
    st.markdown(f"### Summary — {mode}")

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Differences", len(diff_tab))
    m2.metric("Not Found", len(not_found_tab))
    m3.metric("Extra Accounts", len(extra_tab))
    m4.metric("Duplicates", len(duplicate_tab))

    st.markdown("---")

    # -----------------------------
    # TABS
    # -----------------------------
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Difference", "Not Found", "Extra", "Duplicate"]
    )

    with tab1:
        st.subheader("Differences Between Sheets")
        st.dataframe(diff_tab, use_container_width=True)

    with tab2:
        st.subheader("Missing Users")
        st.dataframe(not_found_tab, use_container_width=True)

    with tab3:
        st.subheader("Excluded / Extra Accounts")
        st.dataframe(extra_tab, use_container_width=True)

    with tab4:
        st.subheader("Duplicate User IDs")
        st.dataframe(duplicate_tab, use_container_width=True)

    # -----------------------------
    # DOWNLOAD BUTTON
    # -----------------------------
    excel = to_excel(diff_tab, not_found_tab, extra_tab, duplicate_tab)

    st.download_button(
        "Download Full Report",
        data=excel,
        file_name=f"user_comparison_{mode}.xlsx",
    )
