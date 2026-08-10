"""
Allocation Check -- expected trading allocation vs the All Users sheet.

Independent of the Login Check in morning_comp.py. Needs two inputs:
  * All Users (Main sheet) -- allocation, SubCategory, Running Type/Days, server
  * Running Users          -- capital

Rule (all parameters live in config/allocation_rules.json):

    category_capital    = capital x pct(SubCategory)
    rounded             = round_half_up(category_capital, 20,00,000)
    expected_allocation = rounded / 100
    status              = Match if expected == All Users allocation else Mismatch

Rounding is half-UP by design. numpy/Python round() rounds halves to even, which
turns a category capital of 90,00,000 into 80,00,000 instead of 1,00,00,000.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("allocation_check")

# -----------------------------
# CONFIG LOCATION
# -----------------------------
DEFAULT_RULES_PATH = Path(__file__).resolve().parent / "config" / "allocation_rules.json"
RULES_PATH_ENV = "ALLOCATION_RULES_PATH"

VALID_ACTIONS = {"check", "exclude", "jexception"}
VALID_ROUNDING_MODES = {"half_up", "floor", "ceil"}

# Check methods an account can be routed to.
METHOD_CAPITAL = "capital"
METHOD_PREVIOUS_DAY = "previous_day"
VALID_METHODS = {METHOD_CAPITAL, METHOD_PREVIOUS_DAY}

# Previous-day sheet requirement per mode.
PREV_REQUIRED = "required"
PREV_OPTIONAL = "optional"
PREV_UNUSED = "unused"

# -----------------------------
# COLUMN NAMES
# -----------------------------
SUBCATEGORY_COL = "subcategory"
CAPITAL_COL = "capital"

RESULT_COLUMNS = [
    "userid", "alias", "server", "algo", SUBCATEGORY_COL,
    "pct", CAPITAL_COL, "category_capital", "rounded_capital",
    "expected_allocation", "actual_allocation", "difference", "status",
    "operator_name",
]

PREVDAY_COLUMNS = [
    "userid", "alias", "server", "algo", "runningtype", "runningdays",
    SUBCATEGORY_COL, "previous_allocation", "today_allocation",
    "difference", "status", "operator_name",
]

# -----------------------------
# CONSOLIDATED VIEW
# -----------------------------
STATUS_MATCH = "Match"
STATUS_MISMATCH = "Mismatch"
STATUS_NOT_CHECKED = "Not under check"
STATUS_NEW_USER = "New user"

RULE_PREVIOUS_DAY = "Previous Day"
RULE_JAINAM = "Jainam"
RULE_FIX = "Fixed"
RULE_NOT_CHECKED = "Not under check"

# Internal name for the All Users "FIX (CR)" column. Declared here rather than
# imported from morning_comp, which imports this module.
FIX_CR_COL = "fix_cr"
FIX_BLANK_TOKENS = {"", "nan", "none", "null", "-", "na", "n/a"}

FIX_COLUMNS = [
    "userid", "alias", "server", "algo", SUBCATEGORY_COL,
    "fix_cr", "expected_allocation", "actual_allocation",
    "difference", "status", "operator_name",
]

JAINAM_COLUMNS = [
    "userid", "alias", "server", "algo", SUBCATEGORY_COL,
    "jainam_allocation", "expected_allocation", "actual_allocation",
    "difference", "status", "remark", "operator_name",
]

# Exactly the headers requested by the business, in order. `remark` carries the
# specific reason behind a 'Not under check' verdict -- without it that status
# is unactionable.
CONSOLIDATED_COLUMNS = [
    "user_id", "user alias", "sub category", "rule", "maxloss", "allocation",
    "expected allocation", "category capital", "capital", "status", "remark",
]


class AllocationRulesError(Exception):
    """Raised when the rules JSON is missing, malformed or internally invalid."""


# -----------------------------
# RULES
# -----------------------------
def resolve_rules_path(path: Optional[str] = None) -> Path:
    """Explicit argument wins, then the env var, then the packaged default."""
    if path:
        return Path(path)
    env = os.environ.get(RULES_PATH_ENV)
    return Path(env) if env else DEFAULT_RULES_PATH


def load_rules(path: Optional[str] = None) -> dict:
    """
    Load and validate the rules JSON.

    Validation is strict on purpose: a typo in this file silently changes the
    allocation every account is measured against, so it must fail loudly at
    startup rather than produce plausible wrong numbers.
    """
    rules_path = resolve_rules_path(path)
    try:
        with open(rules_path, "r", encoding="utf-8") as fh:
            rules = json.load(fh)
    except FileNotFoundError as exc:
        raise AllocationRulesError(f"Rules file not found: {rules_path}") from exc
    except json.JSONDecodeError as exc:
        raise AllocationRulesError(
            f"Rules file is not valid JSON ({rules_path}), line {exc.lineno}: {exc.msg}"
        ) from exc
    except OSError as exc:
        raise AllocationRulesError(f"Cannot read rules file {rules_path}: {exc}") from exc

    _validate_rules(rules, rules_path)
    logger.info(
        "Loaded allocation rules v%s from %s (%d subcategories).",
        rules.get("version", "?"), rules_path, len(rules["subcategories"]),
    )
    return rules


def _validate_rules(rules: dict, source: Path) -> None:
    for key in ("rounding", "excluded_servers", "dte_filters", "subcategories"):
        if key not in rules:
            raise AllocationRulesError(f"Rules file {source} is missing '{key}'.")

    rounding = rules["rounding"]
    basis = rounding.get("basis")
    divisor = rounding.get("divisor")
    mode = rounding.get("mode")
    if not isinstance(basis, (int, float)) or basis <= 0:
        raise AllocationRulesError(f"rounding.basis must be a positive number, got {basis!r}.")
    if not isinstance(divisor, (int, float)) or divisor == 0:
        raise AllocationRulesError(f"rounding.divisor must be a non-zero number, got {divisor!r}.")
    if mode not in VALID_ROUNDING_MODES:
        raise AllocationRulesError(
            f"rounding.mode must be one of {sorted(VALID_ROUNDING_MODES)}, got {mode!r}."
        )

    if not isinstance(rules["excluded_servers"], list):
        raise AllocationRulesError("excluded_servers must be a list.")

    if "excluded_algos" in rules and not isinstance(rules["excluded_algos"], list):
        raise AllocationRulesError(
            f"excluded_algos must be a list, got {rules['excluded_algos']!r}."
        )

    if not rules["subcategories"]:
        raise AllocationRulesError("subcategories is empty -- nothing would be checked.")

    for name, cfg in rules["subcategories"].items():
        if not isinstance(cfg, dict):
            raise AllocationRulesError(f"SubCategory '{name}' must map to an object.")
        action = cfg.get("action")
        if action not in VALID_ACTIONS:
            raise AllocationRulesError(
                f"SubCategory '{name}' has action {action!r}; "
                f"must be one of {sorted(VALID_ACTIONS)}."
            )
        if action in ("check", "exclude"):
            pct = cfg.get("pct", 0 if action == "exclude" else None)
            if not isinstance(pct, (int, float)) or isinstance(pct, bool):
                raise AllocationRulesError(
                    f"SubCategory '{name}' has action '{action}' but pct is {pct!r}. "
                    "pct must be a whole percent, e.g. 60 for 60%."
                )
            if pct < 0 or pct > 100:
                raise AllocationRulesError(
                    f"SubCategory '{name}' pct must be between 0 and 100, got {pct!r}."
                )
            # A value between 0 and 1 is almost certainly a fraction written by
            # mistake (0.6 meaning 60%), which would deploy 0.6% of capital.
            if 0 < pct < 1:
                raise AllocationRulesError(
                    f"SubCategory '{name}' pct is {pct!r}, which reads as {pct}% of "
                    f"capital. If you meant {pct * 100:g}%, write {pct * 100:g}. "
                    "Percentages are whole numbers here."
                )
            if action == "check" and pct == 0:
                raise AllocationRulesError(
                    f"SubCategory '{name}' has action 'check' with pct 0. "
                    "Use action 'exclude' for a 0% SubCategory."
                )

    for mode_name, cfg in rules["dte_filters"].items():
        if not isinstance(cfg, dict):
            raise AllocationRulesError(f"dte_filters['{mode_name}'] must be an object.")
        for field in ("runningtype", "runningdays"):
            val = cfg.get(field)
            if val is not None and not isinstance(val, list):
                raise AllocationRulesError(
                    f"dte_filters['{mode_name}'].{field} must be a list or null, got {val!r}."
                )

    prev = rules.get("previous_day", {})
    if not isinstance(prev, dict):
        raise AllocationRulesError("previous_day must be an object.")
    for bucket in ("required", "optional", "unused"):
        if bucket in prev and not isinstance(prev[bucket], list):
            raise AllocationRulesError(f"previous_day.{bucket} must be a list.")

    routing = rules.get("routing", {})
    if not isinstance(routing, dict):
        raise AllocationRulesError("routing must be an object.")
    for mode_name, cfg in routing.items():
        if not isinstance(cfg, dict):
            raise AllocationRulesError(f"routing['{mode_name}'] must be an object.")
        for variant, rule_list in cfg.items():
            if variant not in ("with_previous_day", "without_previous_day"):
                raise AllocationRulesError(
                    f"routing['{mode_name}'] has unknown key '{variant}'; expected "
                    "'with_previous_day' or 'without_previous_day'."
                )
            if not isinstance(rule_list, list):
                raise AllocationRulesError(
                    f"routing['{mode_name}']['{variant}'] must be a list."
                )
            for i, rule in enumerate(rule_list):
                if not isinstance(rule, dict):
                    raise AllocationRulesError(
                        f"routing['{mode_name}']['{variant}'][{i}] must be an object."
                    )
                method = rule.get("method")
                if method not in VALID_METHODS:
                    raise AllocationRulesError(
                        f"routing['{mode_name}']['{variant}'][{i}].method is "
                        f"{method!r}; must be one of {sorted(VALID_METHODS)}."
                    )
                for field in ("runningtype", "runningdays"):
                    val = rule.get(field)
                    if val is not None and not isinstance(val, list):
                        raise AllocationRulesError(
                            f"routing['{mode_name}']['{variant}'][{i}].{field} "
                            f"must be a list, got {val!r}."
                        )


METHOD_LABEL_CAPITAL = "Capital %"
METHOD_LABEL_JAINAM = "Jainam sheet"
METHOD_LABELS = [METHOD_LABEL_CAPITAL, METHOD_LABEL_JAINAM]

EDITOR_SUBCATEGORY = "SubCategory"
EDITOR_PCT = "% of capital"
EDITOR_METHOD = "Method"
EDITOR_NOTE = "note"
EDITOR_COLUMNS = [EDITOR_SUBCATEGORY, EDITOR_PCT, EDITOR_METHOD, EDITOR_NOTE]


def pct_fraction(pct_percent: float) -> float:
    """Whole percent (60) -> multiplier (0.60)."""
    return float(pct_percent) / 100.0


def rules_summary(rules: dict) -> pd.DataFrame:
    """Human-readable view of the active rules, for display in the UI."""
    rows = []
    for name, cfg in rules["subcategories"].items():
        pct = cfg.get("pct")
        action = cfg["action"]
        if action == "jexception":
            shown = "-"
        elif action == "exclude":
            shown = "0% (excluded)"
        else:
            shown = f"{pct:g}%"
        rows.append({
            "SubCategory": name,
            "action": action,
            "% of capital": shown,
            "note": cfg.get("reason", ""),
        })
    return pd.DataFrame(rows)


def rules_to_editor(rules: dict) -> pd.DataFrame:
    """
    Rules -> editable table.

    Percentages are shown as whole numbers: 100 means 100%, 0 means Exclude.
    """
    rows = []
    for name, cfg in rules["subcategories"].items():
        action = cfg["action"]
        rows.append({
            EDITOR_SUBCATEGORY: name,
            EDITOR_PCT: 0 if action != "check" else float(cfg.get("pct", 0)),
            EDITOR_METHOD: (
                METHOD_LABEL_JAINAM if action == "jexception" else METHOD_LABEL_CAPITAL
            ),
            EDITOR_NOTE: cfg.get("reason", ""),
        })
    return pd.DataFrame(rows, columns=EDITOR_COLUMNS)


def editor_to_subcategories(edited: pd.DataFrame) -> dict:
    """
    Editable table -> the subcategories block.

    0 means Exclude. Any other value 1-100 is a percentage of running capital.
    Raises AllocationRulesError on anything that cannot be interpreted, so a
    bad edit is refused rather than silently changing what accounts are measured
    against.
    """
    out: dict = {}
    seen: set = set()

    for i, row in edited.iterrows():
        name = str(row.get(EDITOR_SUBCATEGORY, "")).strip().upper()
        if not name or name in ("NAN", "NONE"):
            continue  # blank row from the editor
        if name in seen:
            raise AllocationRulesError(f"SubCategory '{name}' appears more than once.")
        seen.add(name)

        method = str(row.get(EDITOR_METHOD, METHOD_LABEL_CAPITAL)).strip()
        note = str(row.get(EDITOR_NOTE, "") or "").strip()

        if method == METHOD_LABEL_JAINAM:
            out[name] = {"action": "jexception",
                         "reason": note or "Checked against the Jainam sheet"}
            continue

        raw = row.get(EDITOR_PCT)
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            raise AllocationRulesError(
                f"SubCategory '{name}' has a blank percentage. "
                "Enter 0 to exclude it, or 1-100 for a percentage."
            )
        try:
            pct = float(raw)
        except (TypeError, ValueError) as exc:
            raise AllocationRulesError(
                f"SubCategory '{name}' percentage {raw!r} is not a number."
            ) from exc

        if pct < 0 or pct > 100:
            raise AllocationRulesError(
                f"SubCategory '{name}' percentage must be between 0 and 100, got {pct:g}."
            )
        if 0 < pct < 1:
            raise AllocationRulesError(
                f"SubCategory '{name}' percentage {pct:g} reads as {pct:g}% of capital. "
                f"If you meant {pct * 100:g}%, enter {pct * 100:g}."
            )

        if pct == 0:
            out[name] = {"action": "exclude", "pct": 0,
                         "reason": note or "Excluded from expected-allocation calculation"}
        else:
            entry = {"action": "check", "pct": int(pct) if float(pct).is_integer() else pct}
            if note:
                entry["reason"] = note
            out[name] = entry

    if not out:
        raise AllocationRulesError("At least one SubCategory must be defined.")
    return out


def parse_excluded_algos(text: str) -> list:
    """
    Parse the UI's comma/space separated algo list.

    "8, 19" -> [8, 19]. Numeric values are stored as numbers so the JSON stays
    readable; anything non-numeric is kept verbatim and still matched.
    """
    if text is None:
        return []
    tokens = [t.strip() for t in str(text).replace("\n", ",").replace(" ", ",").split(",")]
    out: list = []
    seen: set = set()
    for token in tokens:
        if not token:
            continue
        key = algo_key(token)
        if not key or key in seen:
            continue
        seen.add(key)
        try:
            number = float(token)
            out.append(int(number) if number.is_integer() else number)
        except ValueError:
            out.append(token)
    return out


def format_excluded_algos(rules: dict) -> str:
    """Excluded algos as an editable comma-separated string."""
    return ", ".join(str(a) for a in (rules.get("excluded_algos") or []))


def save_rules(rules: dict, path: Optional[str] = None) -> Path:
    """
    Validate then write the rules JSON, keeping a backup of the previous file.

    Validation runs BEFORE the write: an invalid edit must never reach disk,
    or the next run would refuse to start.
    """
    rules_path = resolve_rules_path(path)
    _validate_rules(rules, rules_path)

    if rules_path.exists():
        backup = rules_path.with_suffix(".json.bak")
        try:
            backup.write_text(rules_path.read_text(encoding="utf-8"), encoding="utf-8")
        except OSError as exc:
            logger.warning("Could not write rules backup: %s", exc)

    tmp = rules_path.with_suffix(".json.tmp")
    try:
        rules_path.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(rules, fh, indent=2)
            fh.write("\n")
        tmp.replace(rules_path)   # atomic: never leaves a half-written rules file
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        raise AllocationRulesError(f"Could not write {rules_path}: {exc}") from exc

    logger.info("Saved allocation rules to %s", rules_path)
    return rules_path


# -----------------------------
# ROUNDING
# -----------------------------
def round_to_basis(values: pd.Series, basis: float, mode: str = "half_up") -> pd.Series:
    """
    Round each value to a multiple of `basis`.

    half_up is implemented as floor(x / basis + 0.5) rather than np.round,
    because np.round uses banker's rounding: np.round(4.5) == 4, which would
    turn a category capital of 90,00,000 into 80,00,000 instead of 1,00,00,000.
    """
    scaled = values / basis
    if mode == "half_up":
        return np.floor(scaled + 0.5) * basis
    if mode == "floor":
        return np.floor(scaled) * basis
    if mode == "ceil":
        return np.ceil(scaled) * basis
    raise AllocationRulesError(f"Unsupported rounding mode: {mode!r}")


# -----------------------------
# SCOPING
# -----------------------------
def _norm_text(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.replace(" ", "", regex=False)
        .str.lower()
    )


def apply_dte_scope(
    df_all: pd.DataFrame, mode: str, rules: dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Restrict the All Users frame to the accounts this DTE mode checks.

    Returns (in_scope, out_of_scope). Excluded servers are removed in every
    mode; the RunningType / RunningDays filters come from the rules file.
    """
    df = df_all.copy()
    excluded = {str(s).strip().lower() for s in rules["excluded_servers"]}

    server = df["server"].astype(str).str.strip().str.lower()
    server_drop = server.isin(excluded)

    cfg = rules["dte_filters"].get(mode)
    if cfg is None:
        raise AllocationRulesError(
            f"No dte_filters entry for mode '{mode}'. "
            f"Available: {sorted(rules['dte_filters'])}"
        )

    keep = ~server_drop
    for field, key in (("runningtype", "runningtype"), ("runningdays", "runningdays")):
        allowed = cfg.get(key)
        if allowed is None:
            continue
        if field not in df.columns:
            raise AllocationRulesError(
                f"Mode '{mode}' filters on '{field}' but the All Users sheet "
                f"has no such column."
            )
        allowed_norm = {str(a).strip().replace(" ", "").lower() for a in allowed}
        keep &= _norm_text(df[field]).isin(allowed_norm)

    in_scope = df[keep].copy()
    out_of_scope = df[~keep].copy()
    logger.info(
        "Allocation Check %s scope: %d of %d accounts (%d excluded server, "
        "%d out of DTE filter).",
        mode, len(in_scope), len(df), int(server_drop.sum()),
        len(out_of_scope) - int(server_drop.sum()),
    )
    return in_scope, out_of_scope


# -----------------------------
# ROUTING
# -----------------------------
def previous_day_requirement(mode: str, rules: dict) -> str:
    """Whether the previous-day sheet is required / optional / unused for a mode."""
    prev = rules.get("previous_day", {})
    if mode in prev.get("required", []):
        return PREV_REQUIRED
    if mode in prev.get("optional", []):
        return PREV_OPTIONAL
    return PREV_UNUSED


def route_accounts(
    in_scope: pd.DataFrame, mode: str, has_previous_day: bool, rules: dict
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Split in-scope accounts by which check method applies to each.

    Rules are evaluated in file order; the first match wins. An account that
    matches no rule is returned in `unroutable` so it can be shown rather than
    silently disappearing.

    Returns ({method: frame}, unroutable).
    """
    variant = "with_previous_day" if has_previous_day else "without_previous_day"
    mode_routing = rules.get("routing", {}).get(mode)
    if mode_routing is None:
        raise AllocationRulesError(
            f"No routing defined for mode '{mode}'. "
            f"Available: {sorted(rules.get('routing', {}))}"
        )
    rule_list = mode_routing.get(variant)
    if rule_list is None:
        raise AllocationRulesError(
            f"routing['{mode}'] has no '{variant}' entry."
        )
    if not rule_list:
        raise AllocationRulesError(
            f"Mode '{mode}' has no routing rules for '{variant}'. "
            "The previous-day All Users sheet is required for this mode."
        )

    remaining = in_scope.copy()
    routed: Dict[str, pd.DataFrame] = {m: [] for m in VALID_METHODS}

    for rule in rule_list:
        if remaining.empty:
            break
        mask = pd.Series(True, index=remaining.index)
        for field in ("runningtype", "runningdays"):
            allowed = rule.get(field)
            if allowed is None:
                continue
            if field not in remaining.columns:
                raise AllocationRulesError(
                    f"Routing for mode '{mode}' filters on '{field}' but the "
                    "All Users sheet has no such column."
                )
            allowed_norm = {str(a).strip().replace(" ", "").lower() for a in allowed}
            mask &= _norm_text(remaining[field]).isin(allowed_norm)
        routed[rule["method"]].append(remaining[mask])
        remaining = remaining[~mask]

    out = {
        method: (pd.concat(frames, ignore_index=False) if frames else in_scope.iloc[0:0].copy())
        for method, frames in routed.items()
    }
    logger.info(
        "Routing %s (%s): %d capital, %d previous-day, %d unroutable.",
        mode, variant, len(out[METHOD_CAPITAL]),
        len(out[METHOD_PREVIOUS_DAY]), len(remaining),
    )
    if not remaining.empty:
        logger.warning(
            "%d in-scope account(s) matched no routing rule for mode %s.",
            len(remaining), mode,
        )
    return out, remaining


# -----------------------------
# PREVIOUS-DAY COMPARISON
# -----------------------------
def build_previous_day_check(
    today: pd.DataFrame, previous: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compare today's allocation against the previous day's, per userid.

    This is a straight allocation-vs-allocation comparison: SubCategory rules
    and running capital play no part.

    Returns (result, new_accounts) where new_accounts are userids absent from
    the previous-day sheet -- reported separately, never as a mismatch.
    """
    empty_result = pd.DataFrame(columns=PREVDAY_COLUMNS)
    if today.empty:
        return empty_result, pd.DataFrame(columns=PREVDAY_COLUMNS)

    if "allocation" not in previous.columns or "userid" not in previous.columns:
        raise AllocationRulesError(
            "Previous-day All Users sheet must have 'userId' and 'allocation' columns."
        )

    prev = previous[["userid", "allocation"]].rename(
        columns={"allocation": "previous_allocation"}
    ).copy()
    dupes = int(prev["userid"].duplicated().sum())
    if dupes:
        logger.warning(
            "Previous-day sheet has %d duplicate userid(s); keeping the first.", dupes
        )
        prev = prev.drop_duplicates(subset=["userid"], keep="first")

    work = today.copy()
    for col in ("alias", "runningtype", "runningdays", SUBCATEGORY_COL, "operator_name"):
        if col not in work.columns:
            work[col] = ""
    if "algo" not in work.columns:
        work["algo"] = np.nan

    before = len(work)
    merged = work.merge(prev, on="userid", how="left")
    if len(merged) != before:
        raise AllocationRulesError(
            f"Previous-day join changed the row count ({before} -> {len(merged)})."
        )

    merged["today_allocation"] = pd.to_numeric(merged["allocation"], errors="coerce")
    merged["previous_allocation"] = pd.to_numeric(
        merged["previous_allocation"], errors="coerce"
    )

    is_new = merged["previous_allocation"].isna() & ~merged["userid"].isin(prev["userid"])
    new_accounts = merged[is_new].copy()
    new_accounts["status"] = "New / no prior"
    new_accounts["difference"] = np.nan

    compared = merged[~is_new].copy()
    compared["difference"] = compared["today_allocation"] - compared["previous_allocation"]
    same = compared["today_allocation"] == compared["previous_allocation"]
    both_blank = compared["today_allocation"].isna() & compared["previous_allocation"].isna()
    compared["status"] = np.where(same | both_blank, "Match", "Mismatch")

    result = compared[PREVDAY_COLUMNS].sort_values(
        ["status", "server", "userid"], ascending=[False, True, True]
    ).reset_index(drop=True)

    logger.info(
        "Previous-day check: %d compared (%d match, %d mismatch), %d new accounts.",
        len(result), int((result["status"] == "Match").sum()),
        int((result["status"] == "Mismatch").sum()), len(new_accounts),
    )
    return result, new_accounts[PREVDAY_COLUMNS].reset_index(drop=True)


# -----------------------------
# EXCLUDED ALGOS
# -----------------------------
def algo_key(value) -> str:
    """
    Canonical form of an algo for comparison.

    Algo arrives as int64 from Excel, int from CSV and sometimes text, so
    1, 1.0 and "1" must all compare equal.
    """
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if not text:
        return ""
    try:
        number = float(text)
    except ValueError:
        return text.upper()
    return str(int(number)) if number.is_integer() else str(number)


def excluded_algo_keys(rules: dict) -> set:
    """Configured excluded algos, in canonical form."""
    return {
        key for key in (algo_key(a) for a in rules.get("excluded_algos", []) or [])
        if key
    }


def split_excluded_algos(
    df: pd.DataFrame, rules: dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a frame into (excluded_by_algo, rest).

    Excluded algos stay in scope and are reported as 'Not under check' rather
    than being dropped, so a skipped account is always visible.
    """
    keys = excluded_algo_keys(rules)
    empty = df.iloc[0:0].copy()
    if not keys or df.empty or "algo" not in df.columns:
        return empty, df

    mask = df["algo"].map(algo_key).isin(keys)
    if mask.any():
        logger.info(
            "Excluded algos %s: %d account(s) skipped by every rule.",
            sorted(keys), int(mask.sum()),
        )
    return df[mask].copy(), df[~mask].copy()


# -----------------------------
# FIX (CR) EXCEPTION
# -----------------------------
def fix_config(rules: dict) -> dict:
    """FIX settings with safe defaults if the block is absent."""
    cfg = rules.get("fix", {}) or {}
    return {
        "enabled": cfg.get("enabled", True),
        "column": cfg.get("column", FIX_CR_COL),
        "multiplier": cfg.get("multiplier", 100_000),
    }


def split_fix_accounts(
    df: pd.DataFrame, rules: dict
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a frame into (fixed, invalid_fix, rest).

    fixed       positive FIX (CR) value -> checked by the FIX rule
    invalid_fix populated but unusable (0, negative, text) -> reported, not checked
    rest        blank FIX (CR) -> continues down the normal path

    A blank cell is not an error: it simply means the account is not fixed.
    """
    cfg = fix_config(rules)
    empty = df.iloc[0:0].copy()
    if not cfg["enabled"] or cfg["column"] not in df.columns or df.empty:
        return empty, empty, df

    raw = df[cfg["column"]]
    raw_str = raw.astype(str).str.strip()
    is_blank = raw.isna() | raw_str.str.lower().isin(FIX_BLANK_TOKENS)
    numeric = pd.to_numeric(raw, errors="coerce")

    fixed_mask = ~is_blank & numeric.notna() & (numeric > 0)
    invalid_mask = ~is_blank & ~fixed_mask

    if invalid_mask.any():
        logger.warning(
            "%d account(s) have an unusable FIX (CR) value and were not checked: %s",
            int(invalid_mask.sum()), df.loc[invalid_mask, "userid"].tolist(),
        )
    logger.info("FIX exception: %d fixed account(s).", int(fixed_mask.sum()))
    return df[fixed_mask].copy(), df[invalid_mask].copy(), df[~fixed_mask & ~invalid_mask].copy()


def build_fix_check(fix_accounts: pd.DataFrame, rules: dict) -> pd.DataFrame:
    """
    Check fixed-allocation accounts.

        expected allocation = FIX (CR) x multiplier   (3 -> 3,00,000)

    Highest-precedence rule: these accounts bypass the capital, previous-day
    and Jainam checks entirely.
    """
    if fix_accounts.empty:
        return pd.DataFrame(columns=FIX_COLUMNS)

    cfg = fix_config(rules)
    work = fix_accounts.copy()
    for col in ("alias", SUBCATEGORY_COL, "operator_name"):
        if col not in work.columns:
            work[col] = ""
    if "algo" not in work.columns:
        work["algo"] = np.nan

    work["fix_cr"] = pd.to_numeric(work[cfg["column"]], errors="coerce")
    work["expected_allocation"] = work["fix_cr"] * cfg["multiplier"]
    work["actual_allocation"] = pd.to_numeric(work["allocation"], errors="coerce")
    work["difference"] = work["actual_allocation"] - work["expected_allocation"]
    work["status"] = np.where(
        work["expected_allocation"] == work["actual_allocation"],
        STATUS_MATCH, STATUS_MISMATCH,
    )

    result = work[FIX_COLUMNS].sort_values(
        ["status", "userid"], ascending=[False, True]
    ).reset_index(drop=True)
    logger.info(
        "FIX check: %d account(s), %d match, %d mismatch.",
        len(result), int((result["status"] == STATUS_MATCH).sum()),
        int((result["status"] == STATUS_MISMATCH).sum()),
    )
    return result


# -----------------------------
# JAINAM SHEET CHECK (SubCategory JA)
# -----------------------------
def jainam_config(rules: dict) -> dict:
    """Jainam settings with safe defaults if the block is absent."""
    cfg = rules.get("jainam", {}) or {}
    return {
        "sheet_name": cfg.get("sheet_name", "Jainam"),
        "userid_column": cfg.get("userid_column", "UserID"),
        "allocation_column": cfg.get("allocation_column", "ALLOCATION"),
        "multiplier": cfg.get("multiplier", 100_000),
        "exclude_userids": {
            str(u).strip().lower() for u in cfg.get("exclude_userids", ["total", ""])
        },
    }


def prepare_jainam_sheet(df_jainam: pd.DataFrame, rules: dict) -> pd.DataFrame:
    """
    Normalise the Jainam sheet to (userid, jainam_allocation).

    Drops the trailing Total row and any configured non-account rows. All rows
    are used regardless of Date. A userid appearing more than once keeps its
    first occurrence, with a warning.
    """
    cfg = jainam_config(rules)
    uid_col, alloc_col = cfg["userid_column"], cfg["allocation_column"]

    cols = {str(c).strip().lower(): c for c in df_jainam.columns}
    uid_actual = cols.get(uid_col.strip().lower())
    alloc_actual = cols.get(alloc_col.strip().lower())
    if uid_actual is None or alloc_actual is None:
        raise AllocationRulesError(
            f"The '{cfg['sheet_name']}' sheet must contain '{uid_col}' and "
            f"'{alloc_col}' columns. Found: {list(df_jainam.columns)[:12]}"
        )

    out = df_jainam[[uid_actual, alloc_actual]].copy()
    out.columns = ["userid", "jainam_allocation"]
    out["userid"] = (
        out["userid"].astype(str).str.strip().str.replace(" ", "", regex=False).str.upper()
    )

    drop = out["userid"].str.lower().isin(cfg["exclude_userids"]) | out["userid"].isin(
        ["NAN", "NONE", ""]
    )
    if drop.any():
        logger.info(
            "Jainam sheet: dropped %d non-account row(s) (e.g. the Total row).",
            int(drop.sum()),
        )
    out = out[~drop]

    out["jainam_allocation"] = pd.to_numeric(out["jainam_allocation"], errors="coerce")

    dupes = int(out["userid"].duplicated().sum())
    if dupes:
        logger.warning(
            "Jainam sheet has %d duplicate userid(s); keeping the first occurrence.", dupes
        )
        out = out.drop_duplicates(subset=["userid"], keep="first")

    return out.reset_index(drop=True)


def build_jainam_check(
    ja_accounts: pd.DataFrame, df_jainam: Optional[pd.DataFrame], rules: dict
) -> pd.DataFrame:
    """
    Check JA accounts against the Jainam sheet.

        expected allocation = Jainam ALLOCATION x multiplier   (4 -> 4,00,000)

    ALLOCATION 0 means the expected allocation IS zero, so a non-zero Main
    allocation is a mismatch. A JA account with no row in the Jainam sheet is
    also a mismatch, per the business rule.
    """
    if ja_accounts.empty:
        return pd.DataFrame(columns=JAINAM_COLUMNS)

    cfg = jainam_config(rules)
    work = ja_accounts.copy()
    for col in ("alias", SUBCATEGORY_COL, "operator_name"):
        if col not in work.columns:
            work[col] = ""
    if "algo" not in work.columns:
        work["algo"] = np.nan

    if df_jainam is None or df_jainam.empty:
        work["jainam_allocation"] = np.nan
        work["expected_allocation"] = np.nan
        work["actual_allocation"] = pd.to_numeric(work["allocation"], errors="coerce")
        work["difference"] = np.nan
        work["status"] = STATUS_MISMATCH
        work["remark"] = f"No '{cfg['sheet_name']}' sheet found in the All Users workbook"
        logger.warning(
            "Jainam check: %d JA account(s) but no '%s' sheet -- all reported as mismatch.",
            len(work), cfg["sheet_name"],
        )
        return work[JAINAM_COLUMNS].reset_index(drop=True)

    prepared = prepare_jainam_sheet(df_jainam, rules)

    before = len(work)
    merged = work.merge(prepared, on="userid", how="left")
    if len(merged) != before:
        raise AllocationRulesError(
            f"Jainam join changed the row count ({before} -> {len(merged)})."
        )

    present = merged["userid"].isin(prepared["userid"])
    merged["expected_allocation"] = merged["jainam_allocation"] * cfg["multiplier"]
    merged["actual_allocation"] = pd.to_numeric(merged["allocation"], errors="coerce")
    merged["difference"] = merged["actual_allocation"] - merged["expected_allocation"]

    equal = merged["expected_allocation"] == merged["actual_allocation"]
    merged["status"] = np.where(present & equal, STATUS_MATCH, STATUS_MISMATCH)
    merged["remark"] = np.where(
        ~present,
        f"No row in the '{cfg['sheet_name']}' sheet",
        np.where(equal, "", "Allocation differs from the Jainam sheet"),
    )

    result = merged[JAINAM_COLUMNS].sort_values(
        ["status", "userid"], ascending=[False, True]
    ).reset_index(drop=True)

    logger.info(
        "Jainam check: %d JA account(s), %d match, %d mismatch (%d absent from the sheet).",
        len(result), int((result["status"] == STATUS_MATCH).sum()),
        int((result["status"] == STATUS_MISMATCH).sum()), int((~present).sum()),
    )
    return result


# -----------------------------
# MAIN CHECK
# -----------------------------
def build_allocation_check(
    df_all: pd.DataFrame,
    df_run: pd.DataFrame,
    mode: str,
    rules: dict,
    df_prev: Optional[pd.DataFrame] = None,
    df_jainam: Optional[pd.DataFrame] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Run the Allocation Check for one DTE mode.

    Order of operations:
      1. Scope by server / DTE filter.
      2. Classify by SubCategory. JA accounts are pulled out here and always
         go to the Jainam-sheet check -- that check OVERRIDES the mode routing.
      3. Route the remaining checkable accounts to capital or previous-day.

    Every in-scope account lands in exactly one output frame:

        result              capital rule, Match / Mismatch
        prevday_result      today's vs the previous day's allocation
        prevday_new         absent from the previous-day sheet
        jainam_result       JA accounts vs the Jainam sheet
        unknown_subcategory SubCategory not present in the rules file
        excluded            SubCategory with action 'exclude'
        not_in_running      capital-routed but absent from the Running file
        no_capital          in Running but capital is blank or <= 0
        unroutable          matched no routing rule
        out_of_scope        filtered out by server / DTE rule (not in scope)
    """
    required = {"userid", "allocation", "server"}
    missing = required - set(df_all.columns)
    if missing:
        raise AllocationRulesError(
            f"All Users sheet is missing required column(s): {sorted(missing)}"
        )
    if SUBCATEGORY_COL not in df_all.columns:
        raise AllocationRulesError(
            "All Users sheet has no 'SubCategory' column -- the Allocation "
            "Check cannot run without it."
        )
    if CAPITAL_COL not in df_run.columns:
        raise AllocationRulesError(
            "Running Users file has no 'capital' column -- the Allocation "
            "Check cannot run without it."
        )

    in_scope, out_of_scope = apply_dte_scope(df_all, mode, rules)

    for frame in (in_scope, out_of_scope):
        for col, default in (("alias", ""), ("operator_name", "")):
            if col not in frame.columns:
                frame[col] = default
        if "algo" not in frame.columns:
            frame["algo"] = np.nan

    display_cols = ["userid", "alias", "server", "algo", SUBCATEGORY_COL,
                    "allocation", "operator_name"]

    def _slice(frame: pd.DataFrame) -> pd.DataFrame:
        if frame is None or frame.empty:
            return pd.DataFrame(columns=display_cols)
        return frame[display_cols].reset_index(drop=True)

    def _tables(**kw) -> Dict[str, pd.DataFrame]:
        base = {
            "result": pd.DataFrame(columns=RESULT_COLUMNS),
            "unknown_subcategory": pd.DataFrame(columns=display_cols),
            "excluded": pd.DataFrame(columns=display_cols),
            "jexceptions": pd.DataFrame(columns=display_cols),
            "jainam_result": pd.DataFrame(columns=JAINAM_COLUMNS),
            "fix_result": pd.DataFrame(columns=FIX_COLUMNS),
            "fix_invalid": pd.DataFrame(columns=display_cols),
            "excluded_algo": pd.DataFrame(columns=display_cols),
            "not_in_running": pd.DataFrame(columns=display_cols),
            "no_capital": pd.DataFrame(columns=display_cols),
            "prevday_result": pd.DataFrame(columns=PREVDAY_COLUMNS),
            "prevday_new": pd.DataFrame(columns=PREVDAY_COLUMNS),
            "unroutable": pd.DataFrame(columns=display_cols),
            "out_of_scope": out_of_scope,
        }
        base.update(kw)
        return base

    if in_scope.empty:
        logger.warning("Allocation Check %s: no accounts in scope.", mode)
        return _tables()

    # --- 0a. excluded algos: skipped by every rule, but kept visible ---
    excluded_algo, after_algo = split_excluded_algos(in_scope, rules)

    # --- 0b. FIX (CR) exception: overrides Jainam, capital and previous-day ---
    fix_accounts, fix_invalid, remaining = split_fix_accounts(after_algo, rules)
    fix_result = build_fix_check(fix_accounts, rules)

    # --- 1. classify by SubCategory (JA is extracted before any routing) ---
    work = remaining.copy()
    work[SUBCATEGORY_COL] = work[SUBCATEGORY_COL].astype(str).str.strip().str.upper()
    work.loc[work[SUBCATEGORY_COL].isin(["NAN", "NONE", ""]), SUBCATEGORY_COL] = ""

    sub_rules = {k.strip().upper(): v for k, v in rules["subcategories"].items()}
    action = work[SUBCATEGORY_COL].map(lambda s: sub_rules.get(s, {}).get("action"))

    unknown = work[action.isna()].copy()
    excluded = work[action == "exclude"].copy()
    ja_accounts = work[action == "jexception"].copy()
    checkable = work[action == "check"].copy()

    if not unknown.empty:
        logger.warning(
            "Allocation Check %s: %d account(s) have a SubCategory not defined "
            "in the rules file: %s",
            mode, len(unknown),
            sorted(set(unknown[SUBCATEGORY_COL].replace("", "<blank>"))),
        )

    # --- 2. JA accounts -> Jainam sheet ---
    jainam_result = build_jainam_check(ja_accounts, df_jainam, rules)

    # --- 3. route the rest ---
    has_prev = df_prev is not None and not df_prev.empty
    routed, unroutable = route_accounts(checkable, mode, has_prev, rules)

    prevday_result = pd.DataFrame(columns=PREVDAY_COLUMNS)
    prevday_new = pd.DataFrame(columns=PREVDAY_COLUMNS)
    prevday_accounts = routed[METHOD_PREVIOUS_DAY]
    if not prevday_accounts.empty:
        if not has_prev:
            raise AllocationRulesError(
                f"Mode '{mode}' routes {len(prevday_accounts)} account(s) to the "
                "previous-day check, but no previous-day All Users sheet was provided."
            )
        prevday_result, prevday_new = build_previous_day_check(prevday_accounts, df_prev)

    def _finalise(result, not_in_running, no_capital) -> Dict[str, pd.DataFrame]:
        return _tables(
            result=result,
            unknown_subcategory=_slice(unknown),
            excluded=_slice(excluded),
            jexceptions=_slice(ja_accounts),
            jainam_result=jainam_result,
            fix_result=fix_result,
            fix_invalid=_slice(fix_invalid),
            excluded_algo=_slice(excluded_algo),
            not_in_running=_slice(not_in_running),
            no_capital=_slice(no_capital),
            prevday_result=prevday_result,
            prevday_new=prevday_new,
            unroutable=_slice(unroutable),
        )

    capital_accounts = routed[METHOD_CAPITAL]
    if capital_accounts.empty:
        return _finalise(pd.DataFrame(columns=RESULT_COLUMNS), None, None)

    # --- 4. capital rule ---
    run = df_run[["userid", CAPITAL_COL]].copy()
    dup_run = int(run["userid"].duplicated().sum())
    if dup_run:
        logger.warning(
            "Running file has %d duplicate userid(s); keeping the first "
            "occurrence for the capital lookup.", dup_run,
        )
        run = run.drop_duplicates(subset=["userid"], keep="first")

    before = len(capital_accounts)
    merged = capital_accounts.merge(run, on="userid", how="left")
    if len(merged) != before:
        raise AllocationRulesError(
            f"Capital join changed the row count ({before} -> {len(merged)})."
        )

    capital = pd.to_numeric(merged[CAPITAL_COL], errors="coerce")
    merged[CAPITAL_COL] = capital

    not_in_running = merged[capital.isna() & ~merged["userid"].isin(run["userid"])].copy()
    no_capital = merged[
        (capital.isna() & merged["userid"].isin(run["userid"])) | (capital <= 0)
    ].copy()

    valid = merged[capital.notna() & (capital > 0)].copy()
    if valid.empty:
        return _finalise(pd.DataFrame(columns=RESULT_COLUMNS), not_in_running, no_capital)

    basis = rules["rounding"]["basis"]
    divisor = rules["rounding"]["divisor"]
    round_mode = rules["rounding"].get("mode", "half_up")

    # pct is a WHOLE percent in the rules file (60 == 60%).
    valid["pct"] = valid[SUBCATEGORY_COL].map(lambda s: float(sub_rules[s]["pct"]))
    valid["category_capital"] = valid[CAPITAL_COL] * valid["pct"].map(pct_fraction)
    valid["rounded_capital"] = round_to_basis(valid["category_capital"], basis, round_mode)
    valid["expected_allocation"] = valid["rounded_capital"] / divisor
    valid["actual_allocation"] = pd.to_numeric(valid["allocation"], errors="coerce")
    valid["difference"] = valid["actual_allocation"] - valid["expected_allocation"]
    valid["status"] = np.where(
        valid["expected_allocation"] == valid["actual_allocation"],
        STATUS_MATCH, STATUS_MISMATCH,
    )

    result = valid[RESULT_COLUMNS].sort_values(
        ["status", SUBCATEGORY_COL, "server", "userid"],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)

    logger.info(
        "Allocation Check %s: capital %d (%d match), prev-day %d, jainam %d, "
        "unknown %d, excluded %d, not-in-running %d, no-capital %d, unroutable %d.",
        mode, len(result), int((result["status"] == STATUS_MATCH).sum()),
        len(prevday_result), len(jainam_result), len(unknown), len(excluded),
        len(not_in_running), len(no_capital), len(unroutable),
    )
    return _finalise(result, not_in_running, no_capital)


# -----------------------------
# SUMMARY
# -----------------------------
def build_summary(result: pd.DataFrame) -> pd.DataFrame:
    """Match / mismatch counts per SubCategory, for a quick read of the check."""
    if result.empty:
        return pd.DataFrame(columns=[SUBCATEGORY_COL, "pct", "checked", "match", "mismatch"])
    grouped = (
        result.assign(_m=(result["status"] == "Match").astype(int))
        .groupby(SUBCATEGORY_COL, dropna=False)
        .agg(pct=("pct", "first"), checked=("userid", "size"), match=("_m", "sum"))
        .reset_index()
    )
    grouped["mismatch"] = grouped["checked"] - grouped["match"]
    grouped["pct"] = grouped["pct"].map(lambda p: "" if pd.isna(p) else f"{p:g}%")
    return grouped.sort_values(SUBCATEGORY_COL).reset_index(drop=True)


def build_consolidated(
    tables: Dict[str, pd.DataFrame], in_scope: pd.DataFrame
) -> pd.DataFrame:
    """
    One row per in-scope account, across both check methods.

    `rule` is the percentage for capital-rule rows ('60%'), 'Previous Day' for
    previous-day rows, and 'Not under check' otherwise.

    `expected allocation` is the capital-derived expectation for capital rows
    and the previous day's allocation for previous-day rows -- the value the
    account was actually measured against, whichever method applied.

    `category capital` and `capital` are only meaningful for capital rows and
    are left blank elsewhere rather than filled with a misleading zero.
    """
    max_loss_lookup: Dict[str, float] = {}
    if "max_loss" in in_scope.columns:
        ml = in_scope[["userid", "max_loss"]].drop_duplicates("userid", keep="first")
        max_loss_lookup = dict(zip(ml["userid"], ml["max_loss"]))

    def base(frame: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=frame.index)
        out["user_id"] = frame["userid"]
        out["user alias"] = frame.get("alias", "")
        out["sub category"] = frame.get(SUBCATEGORY_COL, "")
        out["maxloss"] = frame["userid"].map(max_loss_lookup)
        return out

    parts: List[pd.DataFrame] = []

    # --- capital rule ---
    cap = tables.get("result", pd.DataFrame())
    if not cap.empty:
        part = base(cap)
        part["rule"] = cap["pct"].map(lambda p: f"{p:g}%" if pd.notna(p) else "")
        part["allocation"] = cap["actual_allocation"]
        part["expected allocation"] = cap["expected_allocation"]
        part["category capital"] = cap["category_capital"]
        part["capital"] = cap[CAPITAL_COL]
        part["status"] = cap["status"]
        part["remark"] = np.where(
            cap["status"] == STATUS_MATCH, "",
            "Allocation differs from capital-derived expectation",
        )
        parts.append(part)

    # --- previous-day rule ---
    prev = tables.get("prevday_result", pd.DataFrame())
    if not prev.empty:
        part = base(prev)
        part["rule"] = RULE_PREVIOUS_DAY
        part["allocation"] = prev["today_allocation"]
        part["expected allocation"] = prev["previous_allocation"]
        part["category capital"] = np.nan
        part["capital"] = np.nan
        part["status"] = prev["status"]
        part["remark"] = np.where(
            prev["status"] == STATUS_MATCH, "",
            "Allocation differs from previous day",
        )
        parts.append(part)

    new_users = tables.get("prevday_new", pd.DataFrame())
    if not new_users.empty:
        part = base(new_users)
        part["rule"] = RULE_PREVIOUS_DAY
        part["allocation"] = new_users["today_allocation"]
        part["expected allocation"] = np.nan
        part["category capital"] = np.nan
        part["capital"] = np.nan
        part["status"] = STATUS_NEW_USER
        part["remark"] = "Not present in the previous day's All Users sheet"
        parts.append(part)

    # --- FIX (CR) exception ---
    fix = tables.get("fix_result", pd.DataFrame())
    if not fix.empty:
        part = base(fix)
        part["rule"] = RULE_FIX
        part["allocation"] = fix["actual_allocation"]
        part["expected allocation"] = fix["expected_allocation"]
        part["category capital"] = np.nan
        part["capital"] = np.nan
        part["status"] = fix["status"]
        part["remark"] = np.where(
            fix["status"] == STATUS_MATCH, "",
            "Allocation differs from the FIX (CR) value",
        )
        parts.append(part)

    # --- Jainam sheet rule (SubCategory JA) ---
    jainam = tables.get("jainam_result", pd.DataFrame())
    if not jainam.empty:
        part = base(jainam)
        part["rule"] = RULE_JAINAM
        part["allocation"] = jainam["actual_allocation"]
        part["expected allocation"] = jainam["expected_allocation"]
        part["category capital"] = np.nan
        part["capital"] = np.nan
        part["status"] = jainam["status"]
        part["remark"] = jainam["remark"]
        parts.append(part)

    # --- everything in scope but not checked ---
    not_checked = [
        ("excluded_algo", "Algo excluded by rule"),
        ("fix_invalid", "FIX (CR) value is unusable (0, negative or non-numeric)"),
        ("unknown_subcategory", "SubCategory not defined in the rules file"),
        ("excluded", "SubCategory excluded by rule"),
        ("not_in_running", "Not present in the Running file"),
        ("no_capital", "Capital is blank or <= 0 in the Running file"),
        ("unroutable", "Matched no routing rule for this mode"),
    ]
    for key, remark in not_checked:
        frame = tables.get(key, pd.DataFrame())
        if frame.empty:
            continue
        part = base(frame)
        part["rule"] = RULE_NOT_CHECKED
        part["allocation"] = pd.to_numeric(frame.get("allocation"), errors="coerce")
        part["expected allocation"] = np.nan
        part["category capital"] = np.nan
        part["capital"] = np.nan
        part["status"] = STATUS_NOT_CHECKED
        part["remark"] = remark
        parts.append(part)

    if not parts:
        return pd.DataFrame(columns=CONSOLIDATED_COLUMNS)

    combined = pd.concat(parts, ignore_index=True)

    # Mismatches first, then new users, then not-checked, then matches.
    order = {STATUS_MISMATCH: 0, STATUS_NEW_USER: 1, STATUS_NOT_CHECKED: 2, STATUS_MATCH: 3}
    combined["_o"] = combined["status"].map(order).fillna(9)
    combined = (
        combined.sort_values(["_o", "sub category", "user_id"])
        .drop(columns=["_o"])
        .reset_index(drop=True)
    )
    return combined[CONSOLIDATED_COLUMNS]


def consolidated_status_counts(consolidated: pd.DataFrame) -> pd.DataFrame:
    """Row counts per status, for the metric strip above the table."""
    if consolidated.empty:
        return pd.DataFrame(columns=["status", "accounts"])
    return (
        consolidated["status"].value_counts()
        .rename_axis("status").reset_index(name="accounts")
    )


def reconcile(scoped_total: int, tables: Dict[str, pd.DataFrame]) -> Tuple[bool, str]:
    """
    Confirm every in-scope account landed in exactly one bucket.

    Cheap invariant, but it is the thing that catches a silent drop before an
    operator does.
    """
    # NOTE: 'jexceptions' is the raw JA account list and 'jainam_result' is the
    # same accounts after checking. Only one may be counted, or JA is doubled.
    counted = (
        len(tables["result"])
        + len(tables["unknown_subcategory"])
        + len(tables["excluded"])
        + len(tables["not_in_running"])
        + len(tables["no_capital"])
        + len(tables.get("prevday_result", []))
        + len(tables.get("prevday_new", []))
        + len(tables.get("jainam_result", []))
        + len(tables.get("fix_result", []))
        + len(tables.get("fix_invalid", []))
        + len(tables.get("excluded_algo", []))
        + len(tables.get("unroutable", []))
    )
    ok = counted == scoped_total
    msg = (
        f"{counted} of {scoped_total} in-scope accounts accounted for"
        if ok else
        f"MISMATCH: {counted} accounted for, {scoped_total} in scope "
        f"({scoped_total - counted} unaccounted)"
    )
    return ok, msg
