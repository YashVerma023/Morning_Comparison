"""
Verification harness for the 0 SL pivot and the operator_name column.

Runs the real functions from morning_comp.py against the dummy sheet in Data/,
plus synthetic edge cases, plus an end-to-end pipeline run that mirrors the
Streamlit flow (including the Excel export) using a derived Running file.

Usage:
    python tests/test_zero_sl_operator.py
    ALL_USERS_FILE=/path/to/other.xlsx python tests/test_zero_sl_operator.py
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from morning_comp import (  # noqa: E402
    COLS,
    EXCLUDE_ALL_USERS,
    EXCLUDE_RUNNING,
    MODE_0DTE,
    MODE_1DTE,
    MODE_4DTE,
    MODE_FILTERS,
    OPERATOR_COL,
    SL_COL,
    ZERO_SL_COUNT_COL,
    attach_operator,
    build_allocation_tab,
    build_fixed_tab,
    build_not_found_summary,
    build_operator_lookup,
    build_running_pivot,
    build_zero_sl_pivot,
    coerce_sl,
    apply_mode_filter,
    get_difference,
    get_duplicates,
    get_not_found,
    normalize_columns,
    normalize_values,
    remove_duplicates,
    split_extra,
    to_excel,
)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")

DEFAULT_SHEET = Path(__file__).resolve().parents[1] / "Data" / "All User Details Daily Updated.xlsx"
ALL_USERS_FILE = Path(os.environ.get("ALL_USERS_FILE", DEFAULT_SHEET))

_failures: List[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}" + (f" -- {detail}" if detail else ""))
    if not condition:
        _failures.append(label)


def load_main() -> pd.DataFrame:
    raw = pd.read_excel(ALL_USERS_FILE, sheet_name="Main")
    return normalize_values(normalize_columns(raw.copy()), is_running=False)


# ---------------------------------------------------------------------
# 1. SL coercion
# ---------------------------------------------------------------------
def test_coerce_sl() -> None:
    print("\n1. SL coercion -- what counts as zero")
    s = pd.Series([0, 0.0, "0", " 0 ", "0%", "0.0", np.nan, "", "  ",
                   2, 0.02, "2%", "NO SL", "abc", None])
    got = coerce_sl(s)
    is_zero = got == 0

    for idx, label in [(0, "0"), (1, "0.0"), (2, "'0'"), (3, "' 0 '"),
                       (4, "'0%'"), (5, "'0.0'")]:
        check(f"{label} counts as zero", bool(is_zero[idx]))
    for idx, label in [(6, "NaN"), (7, "empty string"), (8, "whitespace"),
                       (13, "'abc'"), (14, "None")]:
        check(f"{label} does NOT count", not bool(is_zero[idx]))
    for idx, label in [(9, "2"), (10, "0.02"), (11, "'2%'")]:
        check(f"non-zero {label} does NOT count", not bool(is_zero[idx]))
    check("'NO SL' does NOT count (numeric-only rule)", not bool(is_zero[12]))


# ---------------------------------------------------------------------
# 2. Operator lookup + attach
# ---------------------------------------------------------------------
def test_operator_helpers() -> None:
    print("\n2. Operator lookup and attachment")
    src = pd.DataFrame({
        "algo": [1, 1, 7, 7, 7, 19],
        "server": ["vs3", "vs3", "vs4", "vs4", "vs4", "vs22"],
        OPERATOR_COL: ["MOHITC", "MOHITC", "SHIVANSHU", "SHIVANSHU", "DEEPAKT", ""],
    })
    lookup = build_operator_lookup(src)
    check("clean pair resolves", lookup.get((1, "vs3")) == "MOHITC")
    check("ambiguous pair takes the majority", lookup.get((7, "vs4")) == "SHIVANSHU",
          f"got {lookup.get((7, 'vs4'))}")
    check("all-blank pair is not in the lookup", (19, "vs22") not in lookup)

    # int vs float algo must resolve to the same key (Python hashes 1 == 1.0)
    target = pd.DataFrame({
        "userid": ["A", "B", "C", "D"],
        "algo": [1, 1.0, np.float64(7.0), 99],
        "server": ["vs3", "vs3", "vs4", "vs99"],
    })
    out = attach_operator(target, lookup)
    check("row count unchanged", len(out) == len(target), f"{len(target)} -> {len(out)}")
    check("row order unchanged", list(out["userid"]) == ["A", "B", "C", "D"])
    check("operator is the LAST column", out.columns[-1] == OPERATOR_COL,
          f"columns: {list(out.columns)}")
    check("int algo resolves", out.loc[0, OPERATOR_COL] == "MOHITC")
    check("float algo resolves to same key", out.loc[1, OPERATOR_COL] == "MOHITC")
    check("np.float64 algo resolves", out.loc[2, OPERATOR_COL] == "SHIVANSHU")
    check("unknown pair is blank, never guessed", out.loc[3, OPERATOR_COL] == "")

    missing = attach_operator(pd.DataFrame({"userid": ["A"]}), lookup)
    check("missing algo/server does not crash", OPERATOR_COL in missing.columns)

    empty = attach_operator(pd.DataFrame(columns=["userid", "algo", "server"]), lookup)
    check("empty frame does not crash", len(empty) == 0 and OPERATOR_COL in empty.columns)


# ---------------------------------------------------------------------
# 3. 0 SL pivot against the real sheet
# ---------------------------------------------------------------------
def test_zero_sl_pivot() -> None:
    print(f"\n3. 0 SL pivot: {ALL_USERS_FILE.name}")
    if not ALL_USERS_FILE.exists():
        check("sheet exists", False, str(ALL_USERS_FILE))
        return

    df = load_main()
    check("SL header mapped", SL_COL in df.columns)
    check("Operator Name header mapped", OPERATOR_COL in df.columns)

    for mode in (MODE_0DTE, MODE_1DTE, MODE_4DTE):
        scoped = apply_mode_filter(df.copy(), mode)
        lookup = build_operator_lookup(scoped)
        pivot = build_zero_sl_pivot(scoped, lookup)

        expected_rows = int((coerce_sl(scoped[SL_COL]) == 0).sum())
        expected_ids = scoped[coerce_sl(scoped[SL_COL]) == 0]["userid"].nunique()
        total = int(pivot[ZERO_SL_COUNT_COL].sum()) if not pivot.empty else 0

        print(f"       {mode:>5}: {len(scoped):>3} rows in scope | "
              f"{expected_rows:>3} with SL=0 | pivot total {total:>3} | "
              f"{len(pivot)} group(s)")

        check(f"{mode}: pivot total == distinct 0-SL userids",
              total == expected_ids, f"{total} vs {expected_ids}")
        check(f"{mode}: columns exactly as specified",
              list(pivot.columns) == ["algo", "server", ZERO_SL_COUNT_COL, OPERATOR_COL],
              str(list(pivot.columns)))
        check(f"{mode}: counts are all positive",
              pivot.empty or bool((pivot[ZERO_SL_COUNT_COL] > 0).all()))
        check(f"{mode}: no duplicate algo/server group",
              pivot.empty or not pivot.duplicated(["algo", "server"]).any())

    # Blank SL must never be counted.
    scoped = apply_mode_filter(df.copy(), MODE_0DTE)
    blanks = int(coerce_sl(scoped[SL_COL]).isna().sum())
    pivot = build_zero_sl_pivot(scoped, build_operator_lookup(scoped))
    check("blank SL rows excluded from the count",
          int(pivot[ZERO_SL_COUNT_COL].sum()) + blanks == len(scoped),
          f"{int(pivot[ZERO_SL_COUNT_COL].sum())} + {blanks} vs {len(scoped)}")

    print("\n       0DTE pivot:")
    print(pivot.to_string(index=False))

    # Servers must NOT be excluded here.
    excluded_present = pivot["server"].isin(EXCLUDE_ALL_USERS).any()
    check("excluded servers are KEPT (as specified)", bool(excluded_present),
          "no NOT RUNNING/DLR ACC rows had SL=0 in this sheet"
          if not excluded_present else "")


# ---------------------------------------------------------------------
# 4. End-to-end pipeline, mirroring the Streamlit flow
# ---------------------------------------------------------------------
def test_end_to_end() -> None:
    print("\n4. End-to-end pipeline (all three modes)")
    if not ALL_USERS_FILE.exists():
        return

    base = load_main()

    for mode in (MODE_0DTE, MODE_1DTE, MODE_4DTE):
        df_all = base.copy()

        # Derive a plausible Running file from All Users: drop a few accounts,
        # perturb a few allocations, and add one account that exists only in
        # Running. Running files carry no operator/SL columns.
        run_src = df_all[~df_all["server"].isin(EXCLUDE_ALL_USERS)].head(300).copy()
        df_run = run_src[COLS].copy().reset_index(drop=True)
        df_run = df_run.iloc[5:].reset_index(drop=True)
        df_run.loc[0:3, "allocation"] = df_run.loc[0:3, "allocation"] + 1000
        df_run = pd.concat([df_run, pd.DataFrame([{
            "userid": "ONLYINRUN1", "alias": "x", "allocation": 100000.0,
            "max_loss": 1000.0, "server": "vs3", "algo": 1.0,
        }])], ignore_index=True)

        df_run_raw = df_run.copy()
        df_all = apply_mode_filter(df_all, mode)
        df_all_mode_filtered = df_all.copy()

        duplicate_tab = pd.concat(
            [get_duplicates(df_all, "All User"), get_duplicates(df_run, "Running")],
            ignore_index=True,
        )
        df_all = remove_duplicates(df_all)
        df_run = remove_duplicates(df_run)

        if MODE_FILTERS[mode]["apply_base_exclusions"]:
            df_all_clean, extra_all = split_extra(df_all, EXCLUDE_ALL_USERS, "AllUser")
            df_run_clean, extra_run = split_extra(df_run, EXCLUDE_RUNNING, "Running")
            extra_tab = pd.concat([extra_all, extra_run], ignore_index=True)[COLS + ["not found in"]]
        else:
            df_all_clean, df_run_clean = df_all, df_run
            extra_tab = pd.DataFrame(columns=COLS + ["not found in"])

        diff_tab = get_difference(df_all_clean, df_run_clean)
        not_found_tab = get_not_found(df_all_clean, df_run_clean)
        fixed_tab, fix_cr_invalid = build_fixed_tab(df_all)
        allocation_tab = build_allocation_tab(df_all_clean, mode)
        not_found_summary_tab = build_not_found_summary(not_found_tab)
        running_pivot_tab = build_running_pivot(df_all_mode_filtered, df_run_raw)

        lookup = build_operator_lookup(df_all_mode_filtered)
        zero_sl_tab = build_zero_sl_pivot(df_all_mode_filtered, lookup)

        before = {
            "Difference": len(diff_tab), "Not Found": len(not_found_tab),
            "Extra": len(extra_tab), "Duplicate": len(duplicate_tab),
            "Fixed": len(fixed_tab), "Allocation": len(allocation_tab),
            "Not Found Summary": len(not_found_summary_tab),
            "Summary": len(running_pivot_tab),
        }

        diff_tab = attach_operator(diff_tab, lookup, "algo_all", "server_all")
        not_found_tab = attach_operator(not_found_tab, lookup)
        extra_tab = attach_operator(extra_tab, lookup)
        duplicate_tab = attach_operator(duplicate_tab, lookup)[
            ["userid", "Found in", OPERATOR_COL]
        ]
        allocation_tab = attach_operator(allocation_tab, lookup)
        not_found_summary_tab = attach_operator(not_found_summary_tab, lookup)
        running_pivot_tab = attach_operator(running_pivot_tab, lookup)
        if not fixed_tab.empty:
            fixed_tab = attach_operator(fixed_tab, lookup)
            fixed_tab = fixed_tab[[c for c in fixed_tab.columns if c != "_match"] + ["_match"]]
        fix_cr_invalid = attach_operator(fix_cr_invalid, lookup)

        tables = {
            "Difference": diff_tab, "Not Found": not_found_tab,
            "Extra": extra_tab, "Duplicate": duplicate_tab,
            "Fixed": fixed_tab, "Allocation": allocation_tab,
            "Not Found Summary": not_found_summary_tab,
            "Summary": running_pivot_tab,
        }

        print(f"\n       --- {mode} ---")
        for name, tbl in tables.items():
            check(f"{mode} / {name}: has operator column", OPERATOR_COL in tbl.columns)
            check(f"{mode} / {name}: no rows gained or lost",
                  len(tbl) == before[name], f"{before[name]} -> {len(tbl)}")
            last = tbl.columns[-1]
            expected_last = "_match" if name == "Fixed" and "_match" in tbl.columns else OPERATOR_COL
            check(f"{mode} / {name}: operator is last visible column",
                  last == expected_last, f"last is '{last}'")

        # 0 SL sanity for this mode
        check(f"{mode} / 0 SL: operator resolved for every group",
              zero_sl_tab.empty or bool((zero_sl_tab[OPERATOR_COL] != "").all()),
              f"blank operator on {int((zero_sl_tab[OPERATOR_COL] == '').sum())} group(s)"
              if not zero_sl_tab.empty else "")

        # Excel export must not raise and must contain the 0 SL sheet.
        buf = to_excel(
            diff_tab, not_found_tab, extra_tab, duplicate_tab, fixed_tab,
            allocation_tab, not_found_summary=not_found_summary_tab,
            running_summary=running_pivot_tab, zero_sl=zero_sl_tab,
        )
        sheets = pd.ExcelFile(buf).sheet_names
        check(f"{mode} / export: '0 SL' sheet present", "0 SL" in sheets, str(sheets))
        expected_sheets = ["Summary", "0 SL", "Difference", "Not Found",
                           "Not Found Summary", "Extra", "Duplicate",
                           "Fixed", "Allocation"]
        check(f"{mode} / export: all 9 sheets present",
              sheets == expected_sheets, str(sheets))

        exported = pd.read_excel(buf, sheet_name="0 SL")
        check(f"{mode} / export: 0 SL sheet matches the tab",
              len(exported) == len(zero_sl_tab)
              and list(exported.columns) == list(zero_sl_tab.columns))

        fixed_sheet = pd.read_excel(buf, sheet_name="Fixed")
        check(f"{mode} / export: Fixed sheet has operator, not _match",
              OPERATOR_COL in fixed_sheet.columns and "_match" not in fixed_sheet.columns,
              str(list(fixed_sheet.columns)))


if __name__ == "__main__":
    test_coerce_sl()
    test_operator_helpers()
    test_zero_sl_pivot()
    test_end_to_end()

    print("\n" + "=" * 60)
    if _failures:
        print(f"FAILED ({len(_failures)}):")
        for f in _failures:
            print(f"  - {f}")
        sys.exit(1)
    print("ALL CHECKS PASSED")
