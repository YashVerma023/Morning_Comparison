"""
Verification harness for Indian (lakh / crore) number formatting.

Usage:
    python tests/test_indian_format.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from morning_comp import (  # noqa: E402
    MONEY_COLUMNS,
    format_indian,
    indian_formats,
)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")

_failures: List[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}" + (f" -- {detail}" if detail else ""))
    if not condition:
        _failures.append(label)


def test_grouping() -> None:
    print("\n1. Indian digit grouping")
    cases = {
        0: "0",
        7: "7",
        99: "99",
        999: "999",
        1_000: "1,000",
        40_000: "40,000",
        99_999: "99,999",
        1_00_000: "1,00,000",          # 100000
        4_00_000: "4,00,000",          # 400000
        12_00_000: "12,00,000",        # 1200000
        99_99_999: "99,99,999",        # 9999999
        1_00_00_000: "1,00,00,000",    # 10000000  <- the example given
        2_00_69_877: "2,00,69,877",    # 20069877  real capital value
        1_13_22_199: "1,13,22,199",    # 11322199  the worked example
        1_00_00_00_000: "1,00,00,00,000",  # 1000000000
    }
    for value, expected in cases.items():
        got = format_indian(value)
        check(f"{value} -> {expected}", got == expected, f"got {got}")

    print("\n   negatives and blanks:")
    check("-120000 -> -1,20,000", format_indian(-120_000) == "-1,20,000",
          format_indian(-120_000))
    check("-10000000 -> -1,00,00,000", format_indian(-10_000_000) == "-1,00,00,000",
          format_indian(-10_000_000))
    check("NaN -> blank", format_indian(np.nan) == "")
    check("None -> blank", format_indian(None) == "")
    check("pd.NA -> blank", format_indian(pd.NA) == "")

    print("\n   floats round to whole rupees:")
    check("400000.0 -> 4,00,000", format_indian(400_000.0) == "4,00,000")
    check("6594654.1 -> 65,94,654", format_indian(6_594_654.1) == "65,94,654",
          format_indian(6_594_654.1))

    print("\n   not the western grouping:")
    for value in (10_000_000, 1_00_000, 1_13_22_199):
        western = f"{value:,}"
        check(f"{value} differs from western '{western}'",
              format_indian(value) != western, f"indian {format_indian(value)}")


def test_format_map() -> None:
    print("\n2. Formatter map picks the right columns")
    df = pd.DataFrame({
        "user_id": ["A"], "allocation": [100000.0], "capital": [10000000.0],
        "expected allocation": [120000.0], "category capital": [12000000.0],
        "maxloss": [40000.0], "pct": [60.0], "status": ["Match"],
        "count of 0 SL accounts": [23],
    })
    fmt = indian_formats(df)
    for col in ("allocation", "capital", "expected allocation",
                "category capital", "maxloss"):
        check(f"'{col}' formatted", col in fmt)
    for col in ("user_id", "status", "pct", "count of 0 SL accounts"):
        check(f"'{col}' NOT formatted", col not in fmt,
              "counts and percentages must stay plain")

    check("extra formatters merge in",
          "pct" in indian_formats(df, {"pct": "{:g}%"}))

    styled = df.style.format(indian_formats(df), na_rep="")
    html = styled.to_html()
    check("rendered output contains 1,00,00,000", "1,00,00,000" in html)
    check("rendered output has no 10,000,000", "10,000,000" not in html)


def test_money_columns_cover_the_app() -> None:
    print("\n3. Money column list covers the real tables")
    from allocation_check import (
        CONSOLIDATED_COLUMNS, FIX_COLUMNS, JAINAM_COLUMNS, PREVDAY_COLUMNS,
        RESULT_COLUMNS,
    )
    expected_money = {
        "allocation", "capital", "expected allocation", "category capital", "maxloss",
        "expected_allocation", "actual_allocation", "category_capital",
        "rounded_capital", "difference", "previous_allocation", "today_allocation",
        "jainam_allocation",
    }
    all_cols = set(CONSOLIDATED_COLUMNS) | set(RESULT_COLUMNS) | set(PREVDAY_COLUMNS) \
        | set(JAINAM_COLUMNS) | set(FIX_COLUMNS)
    for col in sorted(expected_money & all_cols):
        check(f"'{col}' is in MONEY_COLUMNS", col.lower() in MONEY_COLUMNS)

    # A percentage or identifier must never be money-formatted.
    for col in ("pct", "userid", "user_id", "status", "server", "algo", "fix_cr"):
        check(f"'{col}' is not money", col not in MONEY_COLUMNS)


if __name__ == "__main__":
    test_grouping()
    test_format_map()
    test_money_columns_cover_the_app()

    print("\n" + "=" * 60)
    if _failures:
        print(f"FAILED ({len(_failures)}):")
        for f in _failures:
            print(f"  - {f}")
        sys.exit(1)
    print("ALL CHECKS PASSED")
