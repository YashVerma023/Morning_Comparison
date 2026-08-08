"""
Verification harness for the FIX (CR) based Fixed tab.

Runs the real functions from morning_comp.py against the dummy sheet in
Data/ plus a set of synthetic edge cases. No Streamlit runtime required.

Usage:
    python tests/test_fixed_tab.py
    ALL_USERS_FILE=/path/to/other.xlsx python tests/test_fixed_tab.py
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
    FIX_CR_COL,
    FIX_CR_MULTIPLIER,
    FIXED_EXCLUDE_SERVERS,
    build_fixed_tab,
    coerce_fix_cr,
    normalize_columns,
    normalize_values,
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


# ---------------------------------------------------------------------
# 1. Multiplier / precision
# ---------------------------------------------------------------------
def test_multiplier_precision() -> None:
    print("\n1. Expected-allocation arithmetic")
    cases = {
        1.0: 100_000, 3.0: 300_000, 1.6: 160_000, 0.8: 80_000,
        1.4: 140_000, 1.2: 120_000, 7.0: 700_000, 2.675: 267_500,
        0.07: 7_000, 1.15: 115_000,
    }
    s = pd.Series(list(cases))
    got = (s * FIX_CR_MULTIPLIER).round(0)
    for fix_cr, expected in cases.items():
        actual = float(got[s[s == fix_cr].index[0]])
        check(f"FIX (CR) {fix_cr} -> {expected:,}", actual == expected, f"got {actual:,.0f}")

    # The bug that round() guards against: floor() lands a rupee low.
    floored = np.floor(s * FIX_CR_MULTIPLIER)
    drift = [
        (f, int(e - floored[s[s == f].index[0]]))
        for f, e in cases.items()
        if floored[s[s == f].index[0]] != e
    ]
    print(f"       (floor() would have been wrong on: {drift or 'none of these values'})")


# ---------------------------------------------------------------------
# 2. Edge cases on the FIX (CR) cell
# ---------------------------------------------------------------------
def test_coerce_edge_cases() -> None:
    print("\n2. FIX (CR) cell edge cases")
    df = pd.DataFrame({
        "userid": ["BLANK1", "BLANK2", "BLANK3", "DASH", "ZERO", "NEG", "TEXT", "GOOD", "STRNUM", "SPACED"],
        "alias": ["a"] * 10,
        "server": ["vs1"] * 10,
        "algo": [1] * 10,
        FIX_CR_COL: [np.nan, "", None, "-", 0, -2, "FIX 1 CR", 1.6, "2", " 1.4 "],
    })
    values, invalid = coerce_fix_cr(df)

    blank_ids = {"BLANK1", "BLANK2", "BLANK3", "DASH"}
    check("blank / empty / None / '-' are skipped silently",
          values[df["userid"].isin(blank_ids)].isna().all()
          and not invalid["userid"].isin(blank_ids).any())

    check("zero is reported invalid", "ZERO" in set(invalid["userid"]))
    check("negative is reported invalid", "NEG" in set(invalid["userid"]))
    check("non-numeric text is reported invalid", "TEXT" in set(invalid["userid"]))
    check("invalid rows carry a reason", invalid["reason"].notna().all(),
          str(sorted(set(invalid["reason"]))))
    check("valid float kept", float(values[df["userid"] == "GOOD"].iloc[0]) == 1.6)
    check("numeric string kept", float(values[df["userid"] == "STRNUM"].iloc[0]) == 2.0)
    check("whitespace-padded numeric string kept",
          float(values[df["userid"] == "SPACED"].iloc[0]) == 1.4)
    check("invalid rows excluded from values", values[df["userid"].isin(invalid["userid"])].isna().all())


# ---------------------------------------------------------------------
# 3. Missing column must not crash
# ---------------------------------------------------------------------
def test_missing_column() -> None:
    print("\n3. Missing FIX (CR) column")
    df = pd.DataFrame({
        "userid": ["A"], "alias": ["a"], "server": ["vs1"],
        "algo": [1], "allocation": [100_000.0],
    })
    fixed, invalid = build_fixed_tab(df)
    check("returns empty frames, no exception", fixed.empty and invalid.empty)
    check("empty frame still carries the schema", FIX_CR_COL in fixed.columns)


# ---------------------------------------------------------------------
# 4. Real dummy sheet
# ---------------------------------------------------------------------
def test_against_dummy_sheet() -> None:
    print(f"\n4. Real sheet: {ALL_USERS_FILE.name}")
    if not ALL_USERS_FILE.exists():
        check("sheet exists", False, str(ALL_USERS_FILE))
        return

    raw = pd.read_excel(ALL_USERS_FILE, sheet_name="Main")
    raw_fix_col = raw["FIX (CR)"].copy()  # normalize_columns renames in place
    df = normalize_values(normalize_columns(raw.copy()), is_running=False)

    check("FIX (CR) header mapped to fix_cr", FIX_CR_COL in df.columns,
          f"columns seen: {[c for c in df.columns if 'fix' in str(c).lower()]}")

    fixed, invalid = build_fixed_tab(df)

    raw_populated = int(raw_fix_col.notna().sum())
    excluded = df[df["server"].isin(FIXED_EXCLUDE_SERVERS) & df[FIX_CR_COL].notna()]
    print(f"       raw populated FIX (CR) cells : {raw_populated}")
    print(f"       dropped by server exclusion  : {len(excluded)}")
    print(f"       rows on the Fixed tab        : {len(fixed)}")
    print(f"       invalid FIX (CR) values      : {len(invalid)}")

    check("no row lost unaccounted for",
          len(fixed) + len(excluded) + len(invalid) == raw_populated,
          f"{len(fixed)} + {len(excluded)} + {len(invalid)} vs {raw_populated}")

    check("excluded servers absent from tab",
          not fixed["server"].isin(FIXED_EXCLUDE_SERVERS).any())
    check("every tab row has a positive FIX (CR)", (fixed[FIX_CR_COL] > 0).all())
    check("expected == FIX (CR) x multiplier",
          bool(((fixed[FIX_CR_COL] * FIX_CR_MULTIPLIER).round(0) == fixed["expected_allocation"]).all()))
    check("status agrees with _match",
          bool((fixed["status"] == fixed["_match"].map({True: "Match", False: "Mismatch"})).all()))
    check("no NaN in expected_allocation", not fixed["expected_allocation"].isna().any())
    check("userid unique on tab", fixed["userid"].is_unique,
          f"{int(fixed['userid'].duplicated().sum())} dup(s)")

    n_match = int(fixed["_match"].sum())
    print(f"       match / mismatch             : {n_match} / {len(fixed) - n_match}")

    # Spot-check the values the business rule was defined with.
    print("\n       Spot-check of user-stated rule:")
    for target in (1.0, 3.0, 1.6, 0.8):
        rows = fixed[fixed[FIX_CR_COL] == target]
        if rows.empty:
            print(f"         FIX (CR) {target:g}: no rows in this sheet")
            continue
        exp = sorted(set(rows["expected_allocation"]))
        check(f"FIX (CR) {target:g} -> {target * FIX_CR_MULTIPLIER:,.0f}",
              exp == [target * FIX_CR_MULTIPLIER], f"got {exp}")

    print("\n       Sample of mismatches:")
    cols = ["userid", "alias", "server", FIX_CR_COL, "expected_allocation", "actual_allocation"]
    print(fixed[~fixed["_match"]][cols].head(8).to_string(index=False))


if __name__ == "__main__":
    test_multiplier_precision()
    test_coerce_edge_cases()
    test_missing_column()
    test_against_dummy_sheet()

    print("\n" + "=" * 60)
    if _failures:
        print(f"FAILED ({len(_failures)}):")
        for f in _failures:
            print(f"  - {f}")
        sys.exit(1)
    print("ALL CHECKS PASSED")
