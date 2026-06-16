import io
import logging
import re
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
# FIX REMARK PARSING
# -----------------------------
_FIX_PATTERN = re.compile(
    r"(?:FIX(?:ED)?)\s*(?:(?:AT|ON|FOR)\s*)?(\d+(?:\.\d+)?)\s*(CR|L)\b",
    re.IGNORECASE,
)


def parse_fix_allocation(remark: str) -> Optional[float]:
    """
    Parse fixed allocation amount from a remark string.
    FIX N CR  -> N x 100,000
    FIX N L   -> N x 1,000
    Returns None if no FIX pattern is found.
    """
    if not isinstance(remark, str):
        return None
    m = _FIX_PATTERN.search(remark)
    if not m:
        return None
    amount = float(m.group(1))
    unit = m.group(2).upper()
    return amount * 100_000 if unit == "CR" else amount * 1_000


# -----------------------------
# FIXED TAB
# -----------------------------
def build_fixed_tab(df_all: pd.DataFrame) -> pd.DataFrame:
    """
    Returns rows from df_all that have a FIX remark.
    Always excludes DLR ACC / NOT RUNNING / Z SERVER regardless of mode.
    """
    if "remark" not in df_all.columns:
        logger.warning("No 'remark' column found in main sheet -- Fixed tab will be empty.")
        return pd.DataFrame(
            columns=["userid", "alias", "server", "algo", "remark",
                     "expected_allocation", "actual_allocation", "status", "_match"]
        )
    # Always exclude irrelevant server groups
    df = df_all[~df_all["server"].isin(FIXED_EXCLUDE_SERVERS)].copy()
    df["expected_allocation"] = df["remark"].apply(parse_fix_allocation)
    df = df[df["expected_allocation"].notna()].copy()
    df["expected_allocation"] = np.floor(df["expected_allocation"])
    df["_match"] = df["expected_allocation"] == df["allocation"]
    df["status"] = df["_match"].map({True: "Match", False: "Mismatch"})
    result = df[[
        "userid", "alias", "server", "algo",
        "remark", "expected_allocation", "allocation", "status", "_match",
    ]].rename(columns={"allocation": "actual_allocation"})
    return result.reset_index(drop=True)


def style_fixed_tab(df: pd.DataFrame):
    """Bold red + white on mismatches; dark green + light on matches."""
    def row_style(row):
        if not df.at[row.name, "_match"]:
            return ["background-color: #b71c1c; color: #ffffff"] * len(row)
        return ["background-color: #1b5e20; color: #e8f5e9"] * len(row)

    display = df.drop(columns=["_match"])
    return display.style.apply(row_style, axis=1).format(
        {"expected_allocation": "{:,.0f}", "actual_allocation": "{:,.0f}"}
    )


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
) -> io.BytesIO:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        diff.to_excel(writer, sheet_name="Difference", index=False)
        not_found.to_excel(writer, sheet_name="Not Found", index=False)
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

st.markdown(
    """
    <h1 style='text-align: center; color: #1f77b4;'>Megaserve Technologies</h1>
    <h3 style='text-align: center; color: gray;'>Morning Sheet Comparison</h3>
    <hr>
    """,
    unsafe_allow_html=True,
)

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

    df_all = apply_mode_filter(df_all, mode)

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
    fixed_tab = build_fixed_tab(df_all)

    allocation_tab = build_allocation_tab(df_all_clean, mode)

    # --- SUMMARY ---
    st.markdown(f"### Summary -- {mode}")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Differences", len(diff_tab))
    m2.metric("Not Found", len(not_found_tab))
    m3.metric("Extra Accounts", len(extra_tab))
    m4.metric("Duplicates", len(duplicate_tab))
    m5.metric("FIX Accounts", len(fixed_tab))
    fixed_mismatches = int((~fixed_tab["_match"]).sum()) if not fixed_tab.empty else 0
    m6.metric("FIX Mismatches", fixed_mismatches)
    st.markdown("---")

    # --- TABS ---
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["Difference", "Not Found", "Extra", "Duplicate", "Fixed", "Allocation"]
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

    with tab5:
        st.subheader("Fixed Allocation Accounts")
        st.caption(
            "Accounts with a FIX remark in the main sheet. "
            "DLR ACC / NOT RUNNING servers are excluded. "
            f"Showing **{len(fixed_tab)}** FIX accounts "
            f"(**{fixed_mismatches}** mismatches) for **{mode}** mode."
        )
        if fixed_tab.empty:
            st.info("No FIX accounts found for this mode.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Total FIX Accounts", len(fixed_tab))
            c2.metric("Matching", int(fixed_tab["_match"].sum()))
            c3.metric("Mismatched", fixed_mismatches)
            st.dataframe(
                style_fixed_tab(fixed_tab),
                use_container_width=True,
                hide_index=True,
            )

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

                st.dataframe(
                    allocation_tab.style.apply(_style_alloc_row, axis=1).format(
                        {"allocation": "{:,.0f}"}
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

    # --- DOWNLOAD ---
    excel_bytes = to_excel(
        diff_tab, not_found_tab, extra_tab, duplicate_tab, fixed_tab, allocation_tab
    )
    st.download_button(
        "Download Full Report",
        data=excel_bytes,
        file_name=f"user_comparison_{mode}.xlsx",
    )
