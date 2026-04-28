import streamlit as st
import pandas as pd
import numpy as np
import io

# -----------------------------
# CONFIG
# -----------------------------
EXCLUDE_ALL_USERS = ["not running", "dlr acc", "z server"]
EXCLUDE_RUNNING = ["z server"]

COLS = ["userid", "alias", "allocation", "max_loss", "server", "algo"]

# -----------------------------
# COLUMN NORMALIZATION
# -----------------------------
def normalize_columns(df):
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
    }

    return df.rename(columns=column_map)

# -----------------------------
# VALUE NORMALIZATION
# -----------------------------
def normalize_values(df, is_running=False):
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

    return df

# -----------------------------
# LOADERS
# -----------------------------
def load_all_users(file):
    excel = pd.ExcelFile(file)
    sheet_name = None

    for s in excel.sheet_names:
        if s.lower() == "main":
            sheet_name = s
            break

    if not sheet_name:
        st.error("Main sheet not found")
        return None

    return pd.read_excel(file, sheet_name=sheet_name)

def load_running_users(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)

# -----------------------------
# DUPLICATES
# -----------------------------
def get_duplicates(df, source):
    dup = df[df.duplicated(subset=["userid"], keep=False)].copy()
    dup["Found in"] = source
    return dup[["userid", "Found in"]]

def remove_duplicates(df):
    return df.drop_duplicates(subset=["userid"], keep=False)

# -----------------------------
# FILTERING
# -----------------------------
def split_extra(df, exclude_list, source_name):
    extra = df[df["server"].isin(exclude_list)].copy()
    extra["not found in"] = source_name

    clean = df[~df["server"].isin(exclude_list)].copy()

    return clean, extra

# -----------------------------
# DIFFERENCE TAB
# -----------------------------
def get_difference(all_df, run_df):

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
        "algo_all", "algo_run"
    ]]

# -----------------------------
# NOT FOUND TAB
# -----------------------------
def get_not_found(all_df, run_df):

    all_ids = set(all_df["userid"])
    run_ids = set(run_df["userid"])

    missing_in_run = pd.DataFrame({
        "userid": list(all_ids - run_ids),
        "Not found in": "Running"
    })

    missing_in_all = pd.DataFrame({
        "userid": list(run_ids - all_ids),
        "Not found in": "All User"
    })

    return pd.concat([missing_in_run, missing_in_all], ignore_index=True)

# -----------------------------
# EXPORT
# -----------------------------
def to_excel(diff, not_found, extra, duplicate):
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

# HEADER
st.markdown(
    """
    <h1 style='text-align: center; color: #1f77b4;'>Megaserve Technologies</h1>
    <h3 style='text-align: center; color: gray;'>Morning Sheet Comparison</h3>
    <hr>
    """,
    unsafe_allow_html=True
)

# FILE UPLOAD SECTION
col1, col2 = st.columns(2)

with col1:
    file_all = st.file_uploader("📂 Upload All Users Excel", type=["xlsx"])

with col2:
    file_run = st.file_uploader("📂 Upload Running Users", type=["csv", "xlsx"])


if file_all and file_run:

    st.success("✅ Files uploaded successfully. Processing...")

    df_all = load_all_users(file_all)
    df_run = load_running_users(file_run)

    if df_all is None:
        st.stop()

    # Normalize
    df_all = normalize_values(normalize_columns(df_all), False)
    df_run = normalize_values(normalize_columns(df_run), True)

    # Duplicates
    dup_all = get_duplicates(df_all, "All User")
    dup_run = get_duplicates(df_run, "Running")
    duplicate_tab = pd.concat([dup_all, dup_run], ignore_index=True)

    # Remove duplicates
    df_all = remove_duplicates(df_all)
    df_run = remove_duplicates(df_run)

    # Extra
    df_all_clean, extra_all = split_extra(df_all, EXCLUDE_ALL_USERS, "AllUser")
    df_run_clean, extra_run = split_extra(df_run, EXCLUDE_RUNNING, "Running")

    extra_tab = pd.concat([extra_all, extra_run], ignore_index=True)[COLS + ["not found in"]]

    # Difference
    diff_tab = get_difference(df_all_clean, df_run_clean)

    # Not Found
    not_found_tab = get_not_found(df_all_clean, df_run_clean)

    # -----------------------------
    # SUMMARY METRICS
    # -----------------------------
    st.markdown("### 📊 Summary")

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
        ["🔍 Difference", "❌ Not Found", "📦 Extra", "⚠️ Duplicate"]
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
        "⬇️ Download Full Report",
        data=excel,
        file_name="user_comparison.xlsx"
    )