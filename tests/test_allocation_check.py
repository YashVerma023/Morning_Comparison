"""
Verification harness for the Allocation Check.

Asserts the six worked examples supplied by the business, the half-up vs
banker's rounding distinction, per-DTE-mode scoping, rules-file validation,
and that no in-scope account is ever silently dropped.

Usage:
    python tests/test_allocation_check.py
"""

from __future__ import annotations

import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from allocation_check import (  # noqa: E402
    AllocationRulesError,
    CAPITAL_COL,
    SUBCATEGORY_COL,
    apply_dte_scope,
    build_allocation_check,
    build_summary,
    load_rules,
    reconcile,
    round_to_basis,
)
from morning_comp import normalize_columns, normalize_values  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")

ROOT = Path(__file__).resolve().parents[1]
ALL_USERS_FILE = ROOT / "Data" / "All User Details Daily Updated.xlsx"
RUNNING_FILE = ROOT / "Data" / "running-users 06AUG26 0908.csv"

_failures: List[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}" + (f" -- {detail}" if detail else ""))
    if not condition:
        _failures.append(label)


# ---------------------------------------------------------------------
# 1. The six business-supplied examples
# ---------------------------------------------------------------------
def test_worked_examples() -> None:
    print("\n1. Business-supplied rounding examples (basis 20,00,000, half-up)")
    cases = [
        (11_020_000, 12_000_000, 120_000),
        (10_920_000, 10_000_000, 100_000),
        (8_500_000,   8_000_000,  80_000),
        (9_000_000,  10_000_000, 100_000),   # the half-way case
        (2_222_222,   2_000_000,  20_000),
        (3_001_000,   4_000_000,  40_000),
        (11_322_199, 12_000_000, 120_000),   # from the original explanation
    ]
    s = pd.Series([c[0] for c in cases], dtype="float64")
    rounded = round_to_basis(s, 2_000_000, "half_up")
    for i, (cap, exp_round, exp_alloc) in enumerate(cases):
        got_round = float(rounded[i])
        got_alloc = got_round / 100
        check(f"{cap:,} -> {exp_round:,} -> allocation {exp_alloc:,}",
              got_round == exp_round and got_alloc == exp_alloc,
              f"got {got_round:,.0f} / {got_alloc:,.0f}")

    print("\n   half-up vs numpy banker's rounding:")
    banker = np.round(s / 2_000_000) * 2_000_000
    diffs = [(int(s[i]), float(banker[i]), float(rounded[i]))
             for i in range(len(s)) if banker[i] != rounded[i]]
    check("np.round would have been wrong on at least one example", bool(diffs),
          "; ".join(f"{c:,}: np.round={b:,.0f} vs half-up={h:,.0f}" for c, b, h in diffs))


# ---------------------------------------------------------------------
# 2. Rounding modes
# ---------------------------------------------------------------------
def test_rounding_modes() -> None:
    print("\n2. Rounding modes")
    s = pd.Series([9_000_000.0, 11_322_199.0])
    check("floor mode", list(round_to_basis(s, 2_000_000, "floor")) == [8_000_000, 10_000_000])
    check("ceil mode", list(round_to_basis(s, 2_000_000, "ceil")) == [10_000_000, 12_000_000])
    check("half_up mode", list(round_to_basis(s, 2_000_000, "half_up")) == [10_000_000, 12_000_000])
    try:
        round_to_basis(s, 2_000_000, "bogus")
        check("unknown mode raises", False)
    except AllocationRulesError:
        check("unknown mode raises", True)


# ---------------------------------------------------------------------
# 3. Rules file validation
# ---------------------------------------------------------------------
def test_rules_validation() -> None:
    print("\n3. Rules file loading and validation")
    rules = load_rules()
    check("packaged rules load", isinstance(rules, dict))
    check("13 subcategories defined", len(rules["subcategories"]) == 13,
          str(len(rules["subcategories"])))
    check("basis is 20,00,000", rules["rounding"]["basis"] == 2_000_000)
    check("mode is half_up", rules["rounding"]["mode"] == "half_up")
    check("excluded servers as specified",
          set(rules["excluded_servers"]) == {"dlr acc", "not running"},
          str(rules["excluded_servers"]))

    pcts = {k: v.get("pct") for k, v in rules["subcategories"].items() if v["action"] == "check"}
    check("CC/CCG at 100%", pcts["CC"] == 1.0 and pcts["CCG"] == 1.0)
    check("CCV/MSS/MSV at 60%",
          pcts["CCV"] == 0.6 and pcts["MSS"] == 0.6 and pcts["MSV"] == 0.6)
    check("MSR/MSP/MSN at 20%",
          pcts["MSR"] == 0.2 and pcts["MSP"] == 0.2 and pcts["MSN"] == 0.2)
    check("PVT/PGB/PPS/PRD excluded",
          all(rules["subcategories"][k]["action"] == "exclude"
              for k in ("PVT", "PGB", "PPS", "PRD")))
    check("JA is a jexception", rules["subcategories"]["JA"]["action"] == "jexception")

    bad_cases = {
        "pct as 60 not 0.60": {"subcategories": {"CC": {"action": "check", "pct": 60}}},
        "unknown action": {"subcategories": {"CC": {"action": "frobnicate"}}},
        "check without pct": {"subcategories": {"CC": {"action": "check"}}},
        "negative basis": {"rounding": {"basis": -1, "divisor": 100, "mode": "half_up"}},
        "bad rounding mode": {"rounding": {"basis": 2000000, "divisor": 100, "mode": "nearest"}},
    }
    for label, patch in bad_cases.items():
        broken = json.loads(json.dumps(rules))
        broken.update(patch)
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump(broken, fh)
            tmp = fh.name
        try:
            load_rules(tmp)
            check(f"rejects {label}", False, "loaded without error")
        except AllocationRulesError:
            check(f"rejects {label}", True)
        finally:
            Path(tmp).unlink(missing_ok=True)

    try:
        load_rules("/nonexistent/path/rules.json")
        check("missing file raises", False)
    except AllocationRulesError:
        check("missing file raises", True)


# ---------------------------------------------------------------------
# 4. Synthetic end-to-end, with every edge case present
# ---------------------------------------------------------------------
def test_synthetic() -> None:
    print("\n4. Synthetic pipeline with all edge cases")
    rules = load_rules()
    all_users = pd.DataFrame([
        # userid, sub, server, rt, rd, allocation
        ("M1", "CC",   "vs1", "POS", "DAILY",     120_000.0),  # match
        ("M2", "MSR",  "vs1", "POS", "DAILY",       1.0),      # mismatch
        ("M3", "JA",   "vs1", "POS", "DAILY",     100_000.0),  # jexception
        ("M4", "PVT",  "vs1", "POS", "DAILY",     100_000.0),  # excluded
        ("M5", "ZZZ",  "vs1", "POS", "DAILY",     100_000.0),  # unknown subcat
        ("M6", "CC",   "vs1", "POS", "DAILY",     100_000.0),  # not in running
        ("M7", "CC",   "vs1", "POS", "DAILY",     100_000.0),  # null capital
        ("M8", "CC",   "vs1", "POS", "DAILY",     100_000.0),  # zero capital
        ("M9", "CC",   "DLR ACC", "POS", "DAILY", 100_000.0),  # excluded server
        ("MA", "CC",   "vs1", "INT", "0DTE",      100_000.0),  # out of 4DTE scope
        ("MB", "",     "vs1", "POS", "DAILY",     100_000.0),  # blank subcat
    ], columns=["userid", "subcategory", "server", "runningtype", "runningdays", "allocation"])
    all_users["alias"] = "a"
    all_users["algo"] = 1
    all_users["operator_name"] = "OP1"

    running = pd.DataFrame({
        "userid": ["M1", "M2", "M3", "M4", "M5", "M7", "M8", "M9", "MA", "MB"],
        "capital": [12_000_000.0, 10_000_000.0, 1e7, 1e7, 1e7,
                    np.nan, 0.0, 1e7, 1e7, 1e7],
    })

    t = build_allocation_check(all_users, running, "4DTE", rules)

    check("M1 matches", (t["result"].query("userid=='M1'")["status"] == "Match").all())
    check("M2 mismatches", (t["result"].query("userid=='M2'")["status"] == "Mismatch").all())
    check("MSR expected = 20% of 1cr rounded",
          float(t["result"].query("userid=='M2'")["expected_allocation"].iloc[0]) == 20_000.0,
          str(t["result"].query("userid=='M2'")["expected_allocation"].tolist()))
    check("JA -> jexceptions", list(t["jexceptions"]["userid"]) == ["M3"])
    check("PVT -> excluded", list(t["excluded"]["userid"]) == ["M4"])
    check("unknown subcat flagged", set(t["unknown_subcategory"]["userid"]) == {"M5", "MB"},
          str(list(t["unknown_subcategory"]["userid"])))
    check("blank subcat flagged, not silently dropped",
          "MB" in set(t["unknown_subcategory"]["userid"]))
    check("not in running", list(t["not_in_running"]["userid"]) == ["M6"])
    check("null / zero capital", set(t["no_capital"]["userid"]) == {"M7", "M8"},
          str(list(t["no_capital"]["userid"])))
    check("excluded server out of scope", "M9" in set(t["out_of_scope"]["userid"]))
    check("INT/0DTE out of 4DTE scope", "MA" in set(t["out_of_scope"]["userid"]))

    in_scope, _ = apply_dte_scope(all_users, "4DTE", rules)
    ok, msg = reconcile(len(in_scope), t)
    check("every in-scope account accounted for", ok, msg)

    # A duplicated running userid must not multiply result rows.
    dup_running = pd.concat([running, running.head(1)], ignore_index=True)
    t2 = build_allocation_check(all_users, dup_running, "4DTE", rules)
    check("duplicate running userid does not multiply rows",
          len(t2["result"]) == len(t["result"]),
          f"{len(t['result'])} -> {len(t2['result'])}")

    # Missing capital column must raise, not silently produce nothing.
    try:
        build_allocation_check(all_users, running.drop(columns=["capital"]), "4DTE", rules)
        check("missing capital column raises", False)
    except AllocationRulesError:
        check("missing capital column raises", True)


# ---------------------------------------------------------------------
# 5. Real files
# ---------------------------------------------------------------------
def test_real_files() -> None:
    print("\n5. Real files")
    if not (ALL_USERS_FILE.exists() and RUNNING_FILE.exists()):
        check("input files exist", False)
        return

    rules = load_rules()
    df_all = normalize_values(
        normalize_columns(pd.read_excel(ALL_USERS_FILE, sheet_name="Main")), is_running=False
    )
    run_raw = normalize_columns(pd.read_csv(RUNNING_FILE))
    run_raw["userid"] = (
        run_raw["userid"].astype(str).str.strip().str.replace(" ", "", regex=False).str.upper()
    )

    check("subcategory column mapped", SUBCATEGORY_COL in df_all.columns)
    check("capital column mapped", CAPITAL_COL in run_raw.columns)

    # 0DTE now requires the previous-day sheet; covered in section 8.
    for mode in ("1DTE", "4DTE"):
        t = build_allocation_check(df_all, run_raw, mode, rules)
        in_scope, _ = apply_dte_scope(df_all, mode, rules)
        ok, msg = reconcile(len(in_scope), t)

        res = t["result"]
        n_match = int((res["status"] == "Match").sum()) if not res.empty else 0
        print(f"\n       --- {mode} --- in scope {len(in_scope)}")
        print(f"       checked {len(res)} | match {n_match} | mismatch {len(res)-n_match} | "
              f"unknown {len(t['unknown_subcategory'])} | excl {len(t['excluded'])} | "
              f"JA {len(t['jexceptions'])} | not-in-run {len(t['not_in_running'])} | "
              f"no-cap {len(t['no_capital'])}")
        check(f"{mode}: reconciles", ok, msg)
        check(f"{mode}: no NaN expected_allocation",
              res.empty or not res["expected_allocation"].isna().any())
        check(f"{mode}: expected is a multiple of basis/divisor",
              res.empty or bool((res["expected_allocation"] % (2_000_000 / 100) == 0).all()))
        check(f"{mode}: userid unique in result",
              res.empty or res["userid"].is_unique)
        check(f"{mode}: excluded servers absent from result",
              res.empty or not res["server"].astype(str).str.lower()
              .isin(rules["excluded_servers"]).any())

        if not res.empty:
            recomputed = round_to_basis(res[CAPITAL_COL] * res["pct"], 2_000_000, "half_up") / 100
            check(f"{mode}: expected_allocation reproducible from capital x pct",
                  bool((recomputed == res["expected_allocation"]).all()))

    print("\n       Summary by SubCategory (1DTE, no previous-day sheet):")
    t = build_allocation_check(df_all, run_raw, "1DTE", rules)
    print(build_summary(t["result"]).to_string(index=False))
    print("\n       Unknown SubCategories flagged (1DTE):")
    u = t["unknown_subcategory"]
    print(u[SUBCATEGORY_COL].replace("", "<blank>").value_counts().to_string()
          if not u.empty else "       (none)")


# ---------------------------------------------------------------------
# 6. Routing: mode x previous-day sheet
# ---------------------------------------------------------------------
def test_routing() -> None:
    from allocation_check import (
        METHOD_CAPITAL, METHOD_PREVIOUS_DAY, PREV_OPTIONAL, PREV_REQUIRED,
        PREV_UNUSED, previous_day_requirement, route_accounts,
    )
    print("\n6. Routing by mode and previous-day availability")
    rules = load_rules()

    check("0DTE requires prev-day", previous_day_requirement("0DTE", rules) == PREV_REQUIRED)
    check("1DTE optional", previous_day_requirement("1DTE", rules) == PREV_OPTIONAL)
    check("4DTE unused", previous_day_requirement("4DTE", rules) == PREV_UNUSED)

    df = pd.DataFrame({
        "userid": ["P_1D", "P_DLY", "P_0D", "I_0D", "BLANK"],
        "runningtype": ["POS", "POS", "POS", "INT", ""],
        "runningdays": ["1DTE/0DTE", "DAILY", "0DTE", "0DTE", ""],
        "server": ["vs1"] * 5,
        "allocation": [1.0] * 5,
        "subcategory": ["CC"] * 5,
    })

    # 4DTE: everything to capital
    routed, un = route_accounts(df, "4DTE", False, rules)
    check("4DTE -> all capital", len(routed[METHOD_CAPITAL]) == 5 and un.empty)

    # 1DTE without prev: everything to capital
    routed, un = route_accounts(df, "1DTE", False, rules)
    check("1DTE no prev -> all capital", len(routed[METHOD_CAPITAL]) == 5 and un.empty)

    # 1DTE with prev: POS+1DTE/0DTE -> capital, POS+DAILY -> prev-day
    routed, un = route_accounts(df, "1DTE", True, rules)
    check("1DTE+prev: POS/1DTE-0DTE -> capital",
          list(routed[METHOD_CAPITAL]["userid"]) == ["P_1D"],
          str(list(routed[METHOD_CAPITAL]["userid"])))
    check("1DTE+prev: POS/DAILY -> previous day",
          list(routed[METHOD_PREVIOUS_DAY]["userid"]) == ["P_DLY"],
          str(list(routed[METHOD_PREVIOUS_DAY]["userid"])))
    check("1DTE+prev: others unroutable", set(un["userid"]) == {"P_0D", "I_0D", "BLANK"},
          str(list(un["userid"])))

    # 0DTE with prev: INT -> capital, POS -> prev-day
    routed, un = route_accounts(df, "0DTE", True, rules)
    check("0DTE: INT -> capital", list(routed[METHOD_CAPITAL]["userid"]) == ["I_0D"],
          str(list(routed[METHOD_CAPITAL]["userid"])))
    check("0DTE: all POS -> previous day",
          set(routed[METHOD_PREVIOUS_DAY]["userid"]) == {"P_1D", "P_DLY", "P_0D"},
          str(list(routed[METHOD_PREVIOUS_DAY]["userid"])))
    check("0DTE: blank running type is unroutable, not dropped",
          list(un["userid"]) == ["BLANK"])

    # 0DTE without prev must refuse rather than silently fall back
    try:
        route_accounts(df, "0DTE", False, rules)
        check("0DTE without prev-day raises", False, "no error raised")
    except AllocationRulesError:
        check("0DTE without prev-day raises", True)


# ---------------------------------------------------------------------
# 7. Previous-day comparison
# ---------------------------------------------------------------------
def test_previous_day() -> None:
    from allocation_check import build_previous_day_check
    print("\n7. Previous-day allocation comparison")

    today = pd.DataFrame({
        "userid": ["SAME", "CHANGED", "NEWACC", "BOTHBLANK"],
        "alias": ["a"] * 4,
        "server": ["vs1"] * 4,
        "algo": [1] * 4,
        "runningtype": ["POS"] * 4,
        "runningdays": ["DAILY"] * 4,
        "subcategory": ["CC", "PVT", "JA", "CC"],   # subcategory must be ignored
        "allocation": [100_000.0, 120_000.0, 50_000.0, np.nan],
        "operator_name": ["OP"] * 4,
    })
    prev = pd.DataFrame({
        "userid": ["SAME", "CHANGED", "BOTHBLANK"],
        "allocation": [100_000.0, 90_000.0, np.nan],
    })

    result, new = build_previous_day_check(today, prev)
    by = dict(zip(result["userid"], result["status"]))
    check("equal allocation -> Match", by.get("SAME") == "Match")
    check("changed allocation -> Mismatch", by.get("CHANGED") == "Mismatch")
    check("difference is today - previous",
          float(result.query("userid=='CHANGED'")["difference"].iloc[0]) == 30_000.0)
    check("both blank -> Match (not a phantom mismatch)", by.get("BOTHBLANK") == "Match")
    check("new account excluded from result", "NEWACC" not in by)
    check("new account reported separately", list(new["userid"]) == ["NEWACC"])
    check("new account is not a mismatch",
          (new["status"] == "New / no prior").all())
    check("SubCategory ignored: PVT and JA still compared",
          {"CHANGED"} <= set(result["userid"]) and "JA" not in set(result["status"]))
    check("row count preserved", len(result) + len(new) == len(today),
          f"{len(result)}+{len(new)} vs {len(today)}")

    dup_prev = pd.concat([prev, prev.head(1)], ignore_index=True)
    r2, _ = build_previous_day_check(today, dup_prev)
    check("duplicate prev-day userid does not multiply rows", len(r2) == len(result))

    try:
        build_previous_day_check(today, prev.drop(columns=["allocation"]))
        check("prev sheet without allocation raises", False)
    except AllocationRulesError:
        check("prev sheet without allocation raises", True)


# ---------------------------------------------------------------------
# 8. Real files, all modes, with previous day
# ---------------------------------------------------------------------
def test_real_with_previous_day() -> None:
    print("\n8. Real files with a previous-day sheet")
    if not (ALL_USERS_FILE.exists() and RUNNING_FILE.exists()):
        return
    rules = load_rules()
    df_all = normalize_values(
        normalize_columns(pd.read_excel(ALL_USERS_FILE, sheet_name="Main")), is_running=False
    )
    run_raw = normalize_columns(pd.read_csv(RUNNING_FILE))
    run_raw["userid"] = (
        run_raw["userid"].astype(str).str.strip().str.replace(" ", "", regex=False).str.upper()
    )

    # Stand-in previous day. Build it from the accounts that actually get routed
    # to the previous-day check, otherwise the mismatch/new paths never fire.
    scope_0dte, _ = apply_dte_scope(df_all, "0DTE", rules)
    pos_ids = scope_0dte[
        scope_0dte["runningtype"].astype(str).str.strip().str.lower() == "pos"
    ]["userid"].tolist()
    changed_ids = set(pos_ids[:15])   # allocations differ -> Mismatch
    dropped_ids = set(pos_ids[15:20])  # absent from prev sheet -> New / no prior

    prev = df_all[["userid", "allocation"]].copy()
    prev = prev[~prev["userid"].isin(dropped_ids)].reset_index(drop=True)
    prev.loc[prev["userid"].isin(changed_ids), "allocation"] += 5_000

    for mode in ("0DTE", "1DTE", "4DTE"):
        t = build_allocation_check(df_all, run_raw, mode, rules, df_prev=prev)
        in_scope, _ = apply_dte_scope(df_all, mode, rules)
        ok, msg = reconcile(len(in_scope), t)
        pv, pn = t["prevday_result"], t["prevday_new"]
        print(f"       {mode:>5}: scope {len(in_scope):>3} | capital {len(t['result']):>3} | "
              f"prev-day {len(pv):>3} (mismatch "
              f"{int((pv['status']=='Mismatch').sum()) if not pv.empty else 0}) | "
              f"new {len(pn):>2} | unroutable {len(t['unroutable']):>2}")
        check(f"{mode}: reconciles with prev-day", ok, msg)
        check(f"{mode}: no account in both paths",
              pv.empty or t["result"].empty or
              not set(pv["userid"]) & set(t["result"]["userid"]))

        if mode == "0DTE":
            mm = set(pv[pv["status"] == "Mismatch"]["userid"]) if not pv.empty else set()
            check("0DTE: seeded mismatches detected on real data",
                  mm == changed_ids, f"{len(mm)} found, {len(changed_ids)} seeded")
            check("0DTE: seeded new accounts detected on real data",
                  set(pn["userid"]) == dropped_ids,
                  f"{len(pn)} found, {len(dropped_ids)} seeded")
            check("0DTE: new accounts are not counted as mismatches",
                  not (set(pn["userid"]) & mm))
            check("0DTE: every non-seeded account matches",
                  int((pv["status"] == "Match").sum()) == len(pv) - len(changed_ids))

    # 1DTE must behave differently with and without the sheet.
    without = build_allocation_check(df_all, run_raw, "1DTE", rules, df_prev=None)
    with_prev = build_allocation_check(df_all, run_raw, "1DTE", rules, df_prev=prev)
    check("1DTE: prev-day sheet changes the routing",
          len(without["result"]) > len(with_prev["result"])
          and with_prev["prevday_result"].shape[0] > 0,
          f"capital {len(without['result'])} -> {len(with_prev['result'])}, "
          f"prev-day {len(with_prev['prevday_result'])}")

    # 0DTE must refuse without the sheet.
    try:
        build_allocation_check(df_all, run_raw, "0DTE", rules, df_prev=None)
        check("0DTE without prev-day refuses on real data", False)
    except AllocationRulesError:
        check("0DTE without prev-day refuses on real data", True)


# ---------------------------------------------------------------------
# 9. Consolidated table
# ---------------------------------------------------------------------
def test_consolidated() -> None:
    from allocation_check import (
        CONSOLIDATED_COLUMNS, STATUS_MATCH, STATUS_MISMATCH, STATUS_NEW_USER,
        STATUS_NOT_CHECKED, build_consolidated, consolidated_status_counts,
    )
    print("\n9. Consolidated 'All Accounts' table")
    if not (ALL_USERS_FILE.exists() and RUNNING_FILE.exists()):
        return
    rules = load_rules()
    df_all = normalize_values(
        normalize_columns(pd.read_excel(ALL_USERS_FILE, sheet_name="Main")), is_running=False
    )
    run_raw = normalize_columns(pd.read_csv(RUNNING_FILE))
    run_raw["userid"] = (
        run_raw["userid"].astype(str).str.strip().str.replace(" ", "", regex=False).str.upper()
    )
    scope0, _ = apply_dte_scope(df_all, "0DTE", rules)
    pos_ids = scope0[
        scope0["runningtype"].astype(str).str.strip().str.lower() == "pos"
    ]["userid"].tolist()
    changed, dropped = set(pos_ids[:15]), set(pos_ids[15:20])
    prev = df_all[["userid", "allocation"]].copy()
    prev = prev[~prev["userid"].isin(dropped)].reset_index(drop=True)
    prev.loc[prev["userid"].isin(changed), "allocation"] += 5_000

    jainam_sheet = pd.read_excel(ALL_USERS_FILE, sheet_name="Jainam")

    for mode in ("0DTE", "1DTE", "4DTE"):
        p = prev if mode != "4DTE" else None
        t = build_allocation_check(df_all, run_raw, mode, rules,
                                   df_prev=p, df_jainam=jainam_sheet)
        in_scope, _ = apply_dte_scope(df_all, mode, rules)
        cons = build_consolidated(t, in_scope)

        check(f"{mode}: headers exactly as specified",
              list(cons.columns) == CONSOLIDATED_COLUMNS, str(list(cons.columns)))
        check(f"{mode}: one row per in-scope account",
              len(cons) == len(in_scope), f"{len(cons)} vs {len(in_scope)}")
        # Duplicate userids in the source sheet legitimately produce duplicate
        # rows here. What must never happen is the table INVENTING duplicates,
        # so compare against the input rather than asserting uniqueness.
        src_dupes = int(in_scope["userid"].duplicated().sum())
        out_dupes = int(cons["user_id"].duplicated().sum())
        check(f"{mode}: duplicates faithfully mirror the source, none invented",
              out_dupes == src_dupes, f"source {src_dupes}, output {out_dupes}")
        check(f"{mode}: statuses limited to the four defined values",
              set(cons["status"]) <= {STATUS_MATCH, STATUS_MISMATCH,
                                      STATUS_NOT_CHECKED, STATUS_NEW_USER},
              str(sorted(set(cons["status"]))))
        check(f"{mode}: every 'Not under check' row carries a remark",
              bool((cons.loc[cons["status"] == STATUS_NOT_CHECKED, "remark"] != "").all()))
        check(f"{mode}: capital columns blank for previous-day rows",
              bool(cons.loc[cons["rule"] == "Previous Day", "capital"].isna().all()))
        # A checked row normally carries the value it was measured against. The
        # one legitimate exception is a JA account absent from the Jainam sheet:
        # it is a Mismatch by rule, but there is no expected value to show.
        checked = cons[cons["status"].isin([STATUS_MATCH, STATUS_MISMATCH])]
        blank_expected = checked[checked["expected allocation"].isna()]
        check(f"{mode}: checked rows have an expected allocation, "
              "except JA rows absent from the Jainam sheet",
              blank_expected.empty
              or bool(blank_expected["remark"]
                      .str.contains("No row in|No 'Jainam' sheet", regex=True).all()),
              f"{len(blank_expected)} blank, reasons: "
              f"{sorted(set(blank_expected['remark']))}")
        check(f"{mode}: 'Not under check' rows have no expected allocation",
              bool(cons.loc[cons["status"] == STATUS_NOT_CHECKED,
                            "expected allocation"].isna().all()))

        pct_rules = set(cons.loc[cons["rule"].str.endswith("%", na=False), "rule"])
        check(f"{mode}: capital rules shown as percentages",
              pct_rules <= {"100%", "60%", "20%"}, str(sorted(pct_rules)))

        counts = consolidated_status_counts(cons)
        check(f"{mode}: status counts sum to the table",
              int(counts["accounts"].sum()) == len(cons))

        if mode == "0DTE":
            print(f"\n       0DTE consolidated ({len(cons)} rows):")
            print(counts.to_string(index=False))
            check("0DTE: seeded new users show as 'New user'",
                  set(cons.loc[cons["status"] == STATUS_NEW_USER, "user_id"]) == dropped)
            print("\n       sample rows:")
            print(cons.head(6).to_string(index=False))


# ---------------------------------------------------------------------
# 10. Jainam sheet check (SubCategory JA)
# ---------------------------------------------------------------------
def test_jainam() -> None:
    from allocation_check import (
        RULE_JAINAM, STATUS_MATCH, STATUS_MISMATCH, build_consolidated,
        build_jainam_check, jainam_config, prepare_jainam_sheet,
    )
    print("\n10. Jainam sheet check (SubCategory JA)")
    rules = load_rules()
    cfg = jainam_config(rules)
    check("multiplier is 1,00,000", cfg["multiplier"] == 100_000)
    check("sheet name is Jainam", cfg["sheet_name"] == "Jainam")

    # --- real sheet: the Total row must be dropped ---
    if ALL_USERS_FILE.exists():
        raw_j = pd.read_excel(ALL_USERS_FILE, sheet_name="Jainam")
        prepared = prepare_jainam_sheet(raw_j, rules)
        check("Total row dropped from the real sheet",
              len(prepared) == len(raw_j) - 1 and "TOTAL" not in set(prepared["userid"]),
              f"{len(raw_j)} -> {len(prepared)}")
        check("prepared sheet has unique userids", prepared["userid"].is_unique)

        # The sheet's own 'Main Allocation' column independently confirms the rule.
        real = raw_j[raw_j["UserID"].astype(str).str.strip().str.lower() != "total"]
        check("ALLOCATION x 1,00,000 equals the sheet's own 'Main Allocation'",
              bool(((real["ALLOCATION"] * 100_000) == real["Main Allocation"]).all()))

    # --- synthetic: every branch ---
    ja = pd.DataFrame({
        "userid": ["J_OK", "J_BAD", "J_ZERO_OK", "J_ZERO_BAD", "J_ABSENT"],
        "alias": ["a"] * 5,
        "server": ["vs1"] * 5,
        "algo": [1] * 5,
        "subcategory": ["JA"] * 5,
        "allocation": [400_000.0, 300_000.0, 0.0, 200_000.0, 100_000.0],
        "operator_name": ["OP"] * 5,
    })
    sheet = pd.DataFrame({
        "UserID": ["J_OK", "J_BAD", "J_ZERO_OK", "J_ZERO_BAD", "Total"],
        "ALLOCATION": [4, 5, 0, 0, 9],
    })
    res = build_jainam_check(ja, sheet, rules)
    by = dict(zip(res["userid"], res["status"]))

    check("4 -> 4,00,000 matches", by.get("J_OK") == STATUS_MATCH)
    check("expected is ALLOCATION x 1,00,000",
          float(res.query("userid=='J_OK'")["expected_allocation"].iloc[0]) == 400_000.0)
    check("5 vs 3,00,000 mismatches", by.get("J_BAD") == STATUS_MISMATCH)
    check("ALLOCATION 0 with allocation 0 -> Match", by.get("J_ZERO_OK") == STATUS_MATCH)
    check("ALLOCATION 0 with non-zero allocation -> Mismatch",
          by.get("J_ZERO_BAD") == STATUS_MISMATCH)
    check("absent from Jainam sheet -> Mismatch", by.get("J_ABSENT") == STATUS_MISMATCH)
    check("absent row carries the right remark",
          "No row" in res.query("userid=='J_ABSENT'")["remark"].iloc[0])
    check("Total row never becomes an account", "TOTAL" not in by)
    check("row count preserved", len(res) == len(ja), f"{len(res)} vs {len(ja)}")

    missing_sheet = build_jainam_check(ja, None, rules)
    check("no Jainam sheet -> all JA mismatch",
          (missing_sheet["status"] == STATUS_MISMATCH).all() and len(missing_sheet) == len(ja))

    try:
        build_jainam_check(ja, pd.DataFrame({"Foo": [1]}), rules)
        check("malformed Jainam sheet raises", False)
    except AllocationRulesError:
        check("malformed Jainam sheet raises", True)

    # --- JA must override routing and reconcile ---
    all_users = pd.DataFrame({
        "userid": ["J_OK", "J_BAD", "C1"],
        "alias": ["a"] * 3,
        "server": ["vs1"] * 3,
        "algo": [1] * 3,
        "runningtype": ["POS", "POS", "POS"],
        "runningdays": ["DAILY"] * 3,
        "subcategory": ["JA", "JA", "CC"],
        "allocation": [400_000.0, 300_000.0, 100_000.0],
        "max_loss": [1000.0] * 3,
        "operator_name": ["OP"] * 3,
    })
    running = pd.DataFrame({"userid": ["J_OK", "J_BAD", "C1"],
                            "capital": [1e7, 1e7, 1e7]})
    prev = pd.DataFrame({"userid": ["J_OK", "J_BAD", "C1"],
                         "allocation": [400_000.0, 300_000.0, 100_000.0]})

    t = build_allocation_check(all_users, running, "0DTE", rules,
                               df_prev=prev, df_jainam=sheet)
    in_scope, _ = apply_dte_scope(all_users, "0DTE", rules)
    ok, msg = reconcile(len(in_scope), t)
    check("JA overrides routing: not in the previous-day result",
          not set(t["prevday_result"]["userid"]) & {"J_OK", "J_BAD"},
          str(list(t["prevday_result"]["userid"])))
    check("JA overrides routing: not in the capital result",
          not set(t["result"]["userid"]) & {"J_OK", "J_BAD"})
    check("JA accounts land in jainam_result",
          set(t["jainam_result"]["userid"]) == {"J_OK", "J_BAD"})
    check("reconciles with JA (no double counting)", ok, msg)

    cons = build_consolidated(t, in_scope)
    check("consolidated: one row per in-scope account",
          len(cons) == len(in_scope), f"{len(cons)} vs {len(in_scope)}")
    check("consolidated: JA rows labelled 'Jainam Sheet'",
          set(cons.loc[cons["user_id"].isin(["J_OK", "J_BAD"]), "rule"]) == {RULE_JAINAM},
          str(set(cons.loc[cons["user_id"].isin(["J_OK", "J_BAD"]), "rule"])))
    check("consolidated: JA expected allocation carried through",
          float(cons.loc[cons["user_id"] == "J_OK", "expected allocation"].iloc[0]) == 400_000.0)
    check("consolidated: JA has no capital columns",
          bool(cons.loc[cons["user_id"].isin(["J_OK", "J_BAD"]), "capital"].isna().all()))


# ---------------------------------------------------------------------
# 11. FIX (CR) exception
# ---------------------------------------------------------------------
def test_fix_exception() -> None:
    from allocation_check import (
        FIX_CR_COL, RULE_FIX, RULE_JAINAM, STATUS_MATCH, STATUS_MISMATCH,
        STATUS_NOT_CHECKED, build_consolidated, build_fix_check, fix_config,
        split_fix_accounts,
    )
    print("\n11. FIX (CR) exception")
    rules = load_rules()
    cfg = fix_config(rules)
    check("multiplier is 1,00,000", cfg["multiplier"] == 100_000)
    check("rule label is 'Fixed'", RULE_FIX == "Fixed")
    check("JA rule label is 'Jainam'", RULE_JAINAM == "Jainam")

    df = pd.DataFrame({
        "userid": ["F3", "F1_5", "F_BAD", "F_ZERO", "F_NEG", "F_TEXT", "PLAIN"],
        "alias": ["a"] * 7,
        "server": ["vs1"] * 7,
        "algo": [1] * 7,
        "subcategory": ["CC"] * 7,
        "allocation": [300_000.0, 150_000.0, 100_000.0, 1.0, 1.0, 1.0, 100_000.0],
        FIX_CR_COL: [3, 1.5, 4, 0, -2, "abc", np.nan],
        "operator_name": ["OP"] * 7,
    })
    fixed, invalid, rest = split_fix_accounts(df, rules)
    check("positive FIX values routed to the FIX rule",
          set(fixed["userid"]) == {"F3", "F1_5", "F_BAD"}, str(list(fixed["userid"])))
    check("0 / negative / text reported as invalid",
          set(invalid["userid"]) == {"F_ZERO", "F_NEG", "F_TEXT"}, str(list(invalid["userid"])))
    check("blank FIX continues down the normal path",
          list(rest["userid"]) == ["PLAIN"], str(list(rest["userid"])))
    check("split loses nothing",
          len(fixed) + len(invalid) + len(rest) == len(df))

    res = build_fix_check(fixed, rules)
    by = dict(zip(res["userid"], res["status"]))
    exp = dict(zip(res["userid"], res["expected_allocation"]))
    check("3 -> 3,00,000", exp.get("F3") == 300_000.0, str(exp.get("F3")))
    check("1.5 -> 1,50,000", exp.get("F1_5") == 150_000.0, str(exp.get("F1_5")))
    check("matching allocation -> Match", by.get("F3") == STATUS_MATCH)
    check("differing allocation -> Mismatch", by.get("F_BAD") == STATUS_MISMATCH)

    # --- precedence: FIX beats JA, capital and previous-day ---
    all_users = pd.DataFrame({
        "userid": ["FIX_JA", "PURE_JA", "FIX_CC", "PURE_CC"],
        "alias": ["a"] * 4,
        "server": ["vs1"] * 4,
        "algo": [1] * 4,
        "runningtype": ["POS"] * 4,
        "runningdays": ["DAILY"] * 4,
        "subcategory": ["JA", "JA", "CC", "CC"],
        "allocation": [300_000.0, 400_000.0, 200_000.0, 100_000.0],
        "max_loss": [1000.0] * 4,
        FIX_CR_COL: [3, np.nan, 2, np.nan],
        "operator_name": ["OP"] * 4,
    })
    running = pd.DataFrame({"userid": all_users["userid"], "capital": [1e7] * 4})
    prev = pd.DataFrame({"userid": all_users["userid"],
                         "allocation": [300_000.0, 400_000.0, 200_000.0, 100_000.0]})
    jsheet = pd.DataFrame({"UserID": ["FIX_JA", "PURE_JA", "Total"],
                           "ALLOCATION": [9, 4, 13]})

    t = build_allocation_check(all_users, running, "0DTE", rules,
                               df_prev=prev, df_jainam=jsheet)
    in_scope, _ = apply_dte_scope(all_users, "0DTE", rules)
    ok, msg = reconcile(len(in_scope), t)

    check("FIX beats JA", set(t["fix_result"]["userid"]) == {"FIX_JA", "FIX_CC"},
          str(list(t["fix_result"]["userid"])))
    check("FIX_JA absent from the Jainam result",
          "FIX_JA" not in set(t["jainam_result"]["userid"]),
          str(list(t["jainam_result"]["userid"])))
    check("FIX_JA uses 3 x 1,00,000, not the Jainam 9",
          float(t["fix_result"].query("userid=='FIX_JA'")["expected_allocation"].iloc[0])
          == 300_000.0)
    check("FIX beats previous-day",
          not set(t["prevday_result"]["userid"]) & {"FIX_JA", "FIX_CC"})
    check("FIX beats capital rule",
          not set(t["result"]["userid"]) & {"FIX_JA", "FIX_CC"})
    check("non-FIX JA still goes to Jainam",
          set(t["jainam_result"]["userid"]) == {"PURE_JA"})
    check("reconciles with FIX (no double counting)", ok, msg)

    cons = build_consolidated(t, in_scope)
    check("consolidated: one row per in-scope account", len(cons) == len(in_scope))
    rules_by_id = dict(zip(cons["user_id"], cons["rule"]))
    check("consolidated: FIX rows labelled 'Fixed'",
          rules_by_id.get("FIX_JA") == "Fixed" and rules_by_id.get("FIX_CC") == "Fixed",
          str(rules_by_id))
    check("consolidated: JA row labelled 'Jainam'", rules_by_id.get("PURE_JA") == "Jainam")

    # --- real sheet ---
    if ALL_USERS_FILE.exists() and RUNNING_FILE.exists():
        df_all = normalize_values(
            normalize_columns(pd.read_excel(ALL_USERS_FILE, sheet_name="Main")),
            is_running=False)
        run_raw = normalize_columns(pd.read_csv(RUNNING_FILE))
        run_raw["userid"] = (run_raw["userid"].astype(str).str.strip()
                             .str.replace(" ", "", regex=False).str.upper())
        jr = pd.read_excel(ALL_USERS_FILE, sheet_name="Jainam")
        prev_r = df_all[["userid", "allocation"]].copy()
        for mode in ("0DTE", "1DTE", "4DTE"):
            t = build_allocation_check(df_all, run_raw, mode, rules,
                                       df_prev=prev_r, df_jainam=jr)
            sc, _ = apply_dte_scope(df_all, mode, rules)
            ok, msg = reconcile(len(sc), t)
            fr = t["fix_result"]
            print(f"       {mode:>5}: scope {len(sc):>3} | fixed {len(fr):>3} "
                  f"(match {int((fr['status']==STATUS_MATCH).sum()) if not fr.empty else 0}) "
                  f"| jainam {len(t['jainam_result']):>2} | capital {len(t['result']):>3} "
                  f"| prev-day {len(t['prevday_result']):>3}")
            check(f"{mode}: reconciles with FIX + JA", ok, msg)
            check(f"{mode}: expected is always FIX x 1,00,000",
                  fr.empty or bool((fr["fix_cr"] * 100_000 == fr["expected_allocation"]).all()))
            cons = build_consolidated(t, sc)
            check(f"{mode}: consolidated one row per account",
                  len(cons) == len(sc), f"{len(cons)} vs {len(sc)}")


if __name__ == "__main__":
    test_worked_examples()
    test_rounding_modes()
    test_rules_validation()
    test_synthetic()
    test_real_files()
    test_routing()
    test_previous_day()
    test_real_with_previous_day()
    test_consolidated()
    test_jainam()
    test_fix_exception()

    print("\n" + "=" * 60)
    if _failures:
        print(f"FAILED ({len(_failures)}):")
        for f in _failures:
            print(f"  - {f}")
        sys.exit(1)
    print("ALL CHECKS PASSED")
