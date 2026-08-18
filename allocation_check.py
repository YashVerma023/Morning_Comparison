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

# Fallback used only when the rules file has no rounding block at all.
DEFAULT_ROUNDING_BASIS = 2_000_000

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
RULE_BROKER = "Broker"
RULE_NOT_CHECKED = "Not under check"

# Broker rule methods.
BROKER_METHOD_CAPITAL_PCT = "capital_pct"
BROKER_METHOD_FIX = "fix_allocation"
VALID_BROKER_METHODS = {BROKER_METHOD_CAPITAL_PCT, BROKER_METHOD_FIX}

BROKER_COL = "broker"

# -----------------------------
# 0 SL MAX-LOSS RULE
# -----------------------------
SL_COL = "sl"
RULE_ZERO_SL = "0 SL"

ZERO_SL_COLUMNS = [
    "userid", "alias", "server", "algo", SUBCATEGORY_COL, "sl",
    "allocation", "max_loss", "expected_max_loss", "difference",
    "status", "operator_name",
]

BROKER_COLUMNS = [
    "userid", "alias", "server", "algo", SUBCATEGORY_COL, BROKER_COL,
    "broker_rule", "pct", "fix_cr", CAPITAL_COL, "category_capital",
    "rounded_capital", "expected_allocation", "actual_allocation",
    "difference", "status", "operator_name",
]

# Internal name for the All Users "FIX (CR)" column. Declared here rather than
# imported from morning_comp, which imports this module.
FIX_CR_COL = "fix_cr"
FIX_BLANK_TOKENS = {"", "nan", "none", "null", "-", "na", "n/a"}

FIX_COLUMNS = [
    "userid", "alias", "server", "algo", SUBCATEGORY_COL,
    "fix_cr", "pct", "fixed_capital", "category_capital", "rounded_capital",
    "expected_allocation", "actual_allocation", "difference", "status",
    "operator_name",
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

    brokers = rules.get("broker_rules") or {}
    if not isinstance(brokers, dict):
        raise AllocationRulesError("broker_rules must be an object.")
    for name, cfg in brokers.items():
        if not isinstance(cfg, dict):
            raise AllocationRulesError(f"Broker '{name}' must map to an object.")
        method = cfg.get("method")
        if method not in VALID_BROKER_METHODS:
            raise AllocationRulesError(
                f"Broker '{name}' has method {method!r}; must be one of "
                f"{sorted(VALID_BROKER_METHODS)}."
            )
        if method == BROKER_METHOD_CAPITAL_PCT:
            pct = cfg.get("pct")
            if not isinstance(pct, (int, float)) or isinstance(pct, bool):
                raise AllocationRulesError(
                    f"Broker '{name}' method 'capital_pct' needs a numeric pct, "
                    f"got {pct!r}."
                )
            if pct <= 0 or pct > 100:
                raise AllocationRulesError(
                    f"Broker '{name}' pct must be between 0 and 100, got {pct!r}."
                )
            if 0 < pct < 1:
                raise AllocationRulesError(
                    f"Broker '{name}' pct is {pct!r}, which reads as {pct}% of "
                    f"capital. If you meant {pct * 100:g}%, write {pct * 100:g}."
                )
        else:
            mult = cfg.get("multiplier", 100_000)
            if not isinstance(mult, (int, float)) or isinstance(mult, bool) or mult <= 0:
                raise AllocationRulesError(
                    f"Broker '{name}' multiplier must be a positive number, got {mult!r}."
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


BROKER_EDITOR_NAME = "Broker"
BROKER_EDITOR_METHOD = "Rule"
BROKER_EDITOR_VALUE = "Value"
BROKER_EDITOR_COLUMNS = [BROKER_EDITOR_NAME, BROKER_EDITOR_METHOD, BROKER_EDITOR_VALUE]

BROKER_LABEL_PCT = "% of capital"
BROKER_LABEL_FIX = "From FIX (CR)"
BROKER_METHOD_LABELS = [BROKER_LABEL_PCT, BROKER_LABEL_FIX]


def broker_rules_to_editor(rules: dict) -> pd.DataFrame:
    """Broker rules -> editable table."""
    rows = []
    for name, cfg in (rules.get("broker_rules") or {}).items():
        is_pct = cfg.get("method") == BROKER_METHOD_CAPITAL_PCT
        rows.append({
            BROKER_EDITOR_NAME: name,
            BROKER_EDITOR_METHOD: BROKER_LABEL_PCT if is_pct else BROKER_LABEL_FIX,
            BROKER_EDITOR_VALUE: float(
                cfg.get("pct", 0) if is_pct else cfg.get("multiplier", 100_000)
            ),
        })
    return pd.DataFrame(rows, columns=BROKER_EDITOR_COLUMNS)


def editor_to_broker_rules(edited: pd.DataFrame) -> dict:
    """
    Editable table -> the broker_rules block.

    '% of capital' takes a percentage 1-100. 'From FIX (CR)' takes the
    multiplier applied to the FIX value (1,00,000 means FIX 1 -> 1,00,000).
    """
    out: dict = {}
    seen: set = set()
    for _, row in edited.iterrows():
        name = str(row.get(BROKER_EDITOR_NAME, "")).strip().upper()
        if not name or name in ("NAN", "NONE"):
            continue
        if name in seen:
            raise AllocationRulesError(f"Broker '{name}' appears more than once.")
        seen.add(name)

        label = str(row.get(BROKER_EDITOR_METHOD, BROKER_LABEL_PCT)).strip()
        raw = row.get(BROKER_EDITOR_VALUE)
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            raise AllocationRulesError(f"Broker '{name}' has a blank value.")
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise AllocationRulesError(
                f"Broker '{name}' value {raw!r} is not a number."
            ) from exc

        if label == BROKER_LABEL_FIX:
            if value <= 0:
                raise AllocationRulesError(
                    f"Broker '{name}' multiplier must be positive, got {value:g}."
                )
            out[name] = {"method": BROKER_METHOD_FIX,
                         "multiplier": int(value) if value.is_integer() else value}
        else:
            if value <= 0 or value > 100:
                raise AllocationRulesError(
                    f"Broker '{name}' percentage must be between 1 and 100, got {value:g}."
                )
            if 0 < value < 1:
                raise AllocationRulesError(
                    f"Broker '{name}' percentage {value:g} reads as {value:g}% of "
                    f"capital. If you meant {value * 100:g}%, enter {value * 100:g}."
                )
            out[name] = {"method": BROKER_METHOD_CAPITAL_PCT,
                         "pct": int(value) if value.is_integer() else value}
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


# Stable key order so a save from the UI keeps the file readable instead of
# reshuffling it. Anything not listed is appended in its existing order.
CONFIG_KEY_ORDER = [
    "version", "_doc", "rounding", "excluded_servers", "excluded_algos",
    "dte_filters", "previous_day", "routing", "subcategories", "broker_rules",
    "fix", "jainam", "zero_sl",
]


def _tidy_rules(rules: dict) -> dict:
    """
    Order keys and drop legacy prose keys.

    Documentation lives in config/README.md, so any leftover `_..._comment`
    blobs from older versions are stripped on save rather than being carried
    forward and re-cluttering the file.
    """
    cleaned = {
        k: v for k, v in rules.items()
        if not (k.startswith("_") and k.endswith("_comment"))
    }
    ordered = {k: cleaned[k] for k in CONFIG_KEY_ORDER if k in cleaned}
    for key, value in cleaned.items():
        if key not in ordered:
            ordered[key] = value
    return ordered


def save_rules(rules: dict, path: Optional[str] = None) -> Path:
    """
    Validate then write the rules JSON, keeping a backup of the previous file.

    Validation runs BEFORE the write: an invalid edit must never reach disk,
    or the next run would refuse to start.
    """
    rules_path = resolve_rules_path(path)
    _validate_rules(rules, rules_path)
    rules = _tidy_rules(rules)

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
# 0 SL MAX-LOSS CHECK
# -----------------------------
def zero_sl_config(rules: dict) -> dict:
    """0 SL settings with safe defaults if the block is absent."""
    cfg = rules.get("zero_sl", {}) or {}
    return {
        "enabled": cfg.get("enabled", True),
        "sl_column": cfg.get("sl_column", SL_COL),
        "max_loss_multiplier": cfg.get("max_loss_multiplier", 30),
    }


def coerce_sl(series: pd.Series) -> pd.Series:
    """
    Coerce the SL column to numeric.

    Accepts 0, 0.0, "0", " 0 " and "0%". Blank cells and non-numeric text
    become NaN so they are never treated as zero.
    """
    cleaned = series.astype(str).str.strip().str.rstrip("%").str.strip()
    return pd.to_numeric(cleaned, errors="coerce")


def build_zero_sl_check(in_scope: pd.DataFrame, rules: dict) -> pd.DataFrame:
    """
    Accounts running with zero stop-loss must satisfy:

        max_loss = allocation x max_loss_multiplier   (default 30)

    This checks MAX LOSS, not allocation, so it runs alongside the allocation
    rules rather than replacing them. A blank SL cell means 'no SL recorded'
    and is not checked.
    """
    cfg = zero_sl_config(rules)
    empty = pd.DataFrame(columns=ZERO_SL_COLUMNS)
    if not cfg["enabled"] or in_scope.empty:
        return empty

    sl_col = cfg["sl_column"]
    if sl_col not in in_scope.columns:
        logger.warning(
            "No '%s' column in the All Users sheet -- the 0 SL check did not run.",
            sl_col,
        )
        return empty
    if "max_loss" not in in_scope.columns:
        logger.warning("No 'max_loss' column -- the 0 SL check did not run.")
        return empty

    sl_numeric = coerce_sl(in_scope[sl_col])
    work = in_scope[sl_numeric == 0].copy()
    if work.empty:
        logger.info("0 SL check: no accounts with a zero SL in scope.")
        return empty

    work["sl"] = sl_numeric[work.index]
    for col in ("alias", "operator_name"):
        if col not in work.columns:
            work[col] = ""
    if "algo" not in work.columns:
        work["algo"] = np.nan
    work[SUBCATEGORY_COL] = (
        work[SUBCATEGORY_COL].map(normalize_subcategory)
        if SUBCATEGORY_COL in work.columns else ""
    )

    work["allocation"] = pd.to_numeric(work["allocation"], errors="coerce")
    work["max_loss"] = pd.to_numeric(work["max_loss"], errors="coerce")
    work["expected_max_loss"] = work["allocation"] * cfg["max_loss_multiplier"]
    work["difference"] = work["max_loss"] - work["expected_max_loss"]
    both_blank = work["max_loss"].isna() & work["expected_max_loss"].isna()
    work["status"] = np.where(
        (work["expected_max_loss"] == work["max_loss"]) | both_blank,
        STATUS_MATCH, STATUS_MISMATCH,
    )

    result = work[ZERO_SL_COLUMNS].sort_values(
        ["status", "server", "userid"], ascending=[False, True, True]
    ).reset_index(drop=True)
    logger.info(
        "0 SL check: %d account(s) with zero SL, %d match, %d mismatch "
        "(max_loss = allocation x %s).",
        len(result), int((result["status"] == STATUS_MATCH).sum()),
        int((result["status"] == STATUS_MISMATCH).sum()), cfg["max_loss_multiplier"],
    )
    return result


# -----------------------------
# BROKER RULES
# -----------------------------
def broker_rules(rules: dict) -> dict:
    """Broker rules keyed by upper-cased broker name."""
    return {
        str(k).strip().upper(): v
        for k, v in (rules.get("broker_rules") or {}).items()
        if str(k).strip()
    }


def split_broker_accounts(
    df: pd.DataFrame, rules: dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split into (accounts with a broker rule, everything else)."""
    configured = broker_rules(rules)
    empty = df.iloc[0:0].copy()
    if not configured or df.empty or BROKER_COL not in df.columns:
        if configured and BROKER_COL not in df.columns and not df.empty:
            logger.warning(
                "Broker rules are configured but the All Users sheet has no "
                "'Broker' column -- broker rules were not applied."
            )
        return empty, df

    key = df[BROKER_COL].map(lambda v: str(v).strip().upper() if v is not None else "")
    mask = key.isin(configured)
    if mask.any():
        logger.info(
            "Broker rules matched %d account(s) across %s.",
            int(mask.sum()), sorted(set(key[mask])),
        )
    return df[mask].copy(), df[~mask].copy()


def build_broker_check(
    broker_accounts: pd.DataFrame, df_run: pd.DataFrame, rules: dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply per-broker rules.

        capital_pct     expected = round(running_capital x pct%, basis) / divisor
        fix_allocation  expected = FIX (CR) x multiplier

    Outranks the FIX, Jainam and category rules; outranked only by excluded algos.

    Returns (result, unresolved) where unresolved holds accounts whose rule
    could not be applied -- no capital, or no FIX value -- reported rather than
    given an invented expectation.
    """
    empty_result = pd.DataFrame(columns=BROKER_COLUMNS)
    if broker_accounts.empty:
        return empty_result, broker_accounts.copy()

    configured = broker_rules(rules)
    basis = rules["rounding"]["basis"]
    divisor = rules["rounding"]["divisor"]
    round_mode = rules["rounding"].get("mode", "half_up")

    work = broker_accounts.copy()
    for col in ("alias", "operator_name"):
        if col not in work.columns:
            work[col] = ""
    if "algo" not in work.columns:
        work["algo"] = np.nan
    if SUBCATEGORY_COL in work.columns:
        work[SUBCATEGORY_COL] = work[SUBCATEGORY_COL].map(normalize_subcategory)
    else:
        work[SUBCATEGORY_COL] = ""

    work[BROKER_COL] = work[BROKER_COL].map(
        lambda v: str(v).strip().upper() if v is not None else ""
    )
    work["broker_rule"] = work[BROKER_COL].map(
        lambda b: configured.get(b, {}).get("method", "")
    )

    # Capital comes from the Running file, deduplicated so it cannot multiply rows.
    run = df_run[["userid", CAPITAL_COL]].drop_duplicates(subset=["userid"], keep="first")
    before = len(work)
    work = work.merge(run, on="userid", how="left")
    if len(work) != before:
        raise AllocationRulesError(
            f"Broker capital join changed the row count ({before} -> {len(work)})."
        )
    work[CAPITAL_COL] = pd.to_numeric(work[CAPITAL_COL], errors="coerce")

    fix_col = fix_config(rules)["column"]
    work["fix_cr"] = (
        pd.to_numeric(work[fix_col], errors="coerce") if fix_col in work.columns
        else np.nan
    )
    work["actual_allocation"] = pd.to_numeric(work["allocation"], errors="coerce")

    work["pct"] = np.nan
    work["category_capital"] = np.nan
    work["rounded_capital"] = np.nan
    work["expected_allocation"] = np.nan
    work["_reason"] = ""

    is_pct = work["broker_rule"] == BROKER_METHOD_CAPITAL_PCT
    if is_pct.any():
        pct = work.loc[is_pct, BROKER_COL].map(
            lambda b: float(configured[b].get("pct", 0))
        )
        work.loc[is_pct, "pct"] = pct
        usable = is_pct & work[CAPITAL_COL].notna() & (work[CAPITAL_COL] > 0)
        work.loc[usable, "category_capital"] = (
            work.loc[usable, CAPITAL_COL] * work.loc[usable, "pct"].map(pct_fraction)
        )
        work.loc[usable, "rounded_capital"] = round_to_basis(
            work.loc[usable, "category_capital"], basis, round_mode
        )
        work.loc[usable, "expected_allocation"] = work.loc[usable, "rounded_capital"] / divisor
        work.loc[is_pct & ~usable, "_reason"] = (
            "Broker rule needs running capital, which is missing or <= 0"
        )

    is_fix = work["broker_rule"] == BROKER_METHOD_FIX
    if is_fix.any():
        mult = work.loc[is_fix, BROKER_COL].map(
            lambda b: float(configured[b].get("multiplier", 100_000))
        )
        usable = is_fix & work["fix_cr"].notna() & (work["fix_cr"] > 0)
        work.loc[usable, "expected_allocation"] = work.loc[usable, "fix_cr"] * mult[usable]
        work.loc[is_fix & ~usable, "_reason"] = (
            "Broker rule needs a FIX (CR) value, which is missing"
        )

    unknown_method = ~is_pct & ~is_fix
    if unknown_method.any():
        work.loc[unknown_method, "_reason"] = "Broker rule method not recognised"

    unresolved = work[work["expected_allocation"].isna()].copy()
    if not unresolved.empty:
        logger.warning(
            "Broker rules: %d account(s) could not be resolved: %s",
            len(unresolved),
            sorted({f"{u} ({r})" for u, r in
                    zip(unresolved["userid"], unresolved["_reason"])}),
        )

    resolved = work[work["expected_allocation"].notna()].copy()
    if resolved.empty:
        return empty_result, unresolved

    resolved["difference"] = resolved["actual_allocation"] - resolved["expected_allocation"]
    resolved["status"] = np.where(
        resolved["expected_allocation"] == resolved["actual_allocation"],
        STATUS_MATCH, STATUS_MISMATCH,
    )
    result = resolved[BROKER_COLUMNS].sort_values(
        ["status", BROKER_COL, "userid"], ascending=[False, True, True]
    ).reset_index(drop=True)

    logger.info(
        "Broker check: %d account(s) checked, %d match, %d mismatch, %d unresolved.",
        len(result), int((result["status"] == STATUS_MATCH).sum()),
        int((result["status"] == STATUS_MISMATCH).sum()), len(unresolved),
    )
    return result, unresolved


# -----------------------------
# EXCLUDED ALGOS
# -----------------------------
def normalize_subcategory(value) -> str:
    """
    Canonical SubCategory string, safe for any input dtype.

    Blank / NaN / NA -> "". Numeric SubCategories such as 13199650 stringify
    without a trailing ".0", so they match a rules entry of the same name.
    """
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, float) and float(value).is_integer():
        text = str(int(value))
    else:
        text = str(value)
    text = text.strip().upper()
    return "" if text in ("NAN", "NONE", "NAT", "<NA>", "NULL") else text


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


def jainam_subcategories(rules: dict) -> set:
    """SubCategory names checked against the Jainam sheet (action 'jexception')."""
    return {
        str(name).strip().upper()
        for name, cfg in rules.get("subcategories", {}).items()
        if isinstance(cfg, dict) and cfg.get("action") == "jexception"
    }


def split_excluded_algos(
    df: pd.DataFrame, rules: dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a frame into (excluded_by_algo, rest).

    Excluded algos stay in scope and are reported as 'Not under check' rather
    than being dropped, so a skipped account is always visible.

    Jainam-type accounts (SubCategory action 'jexception', currently MSJ) are
    EXEMPT: they are still checked against the Jainam sheet even when their
    algo is excluded. Keyed off the action rather than the literal name so
    renaming the SubCategory does not silently break the exemption.
    """
    keys = excluded_algo_keys(rules)
    empty = df.iloc[0:0].copy()
    if not keys or df.empty or "algo" not in df.columns:
        return empty, df

    mask = df["algo"].map(algo_key).isin(keys)

    exempt_subs = jainam_subcategories(rules)
    if exempt_subs and SUBCATEGORY_COL in df.columns:
        is_jainam = df[SUBCATEGORY_COL].map(normalize_subcategory).isin(exempt_subs)
        exempted = int((mask & is_jainam).sum())
        if exempted:
            logger.info(
                "Excluded algos: %d Jainam-type account(s) (%s) exempted and still "
                "checked against the Jainam sheet.", exempted, sorted(exempt_subs),
            )
        mask &= ~is_jainam

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
    """
    FIX settings with safe defaults if the block is absent.

    capital_multiplier converts the FIX (CR) figure into a capital amount:
    FIX 1 -> 1,00,00,000. The legacy key `multiplier` is still read so an old
    rules file does not silently fall back to a default.
    """
    cfg = rules.get("fix", {}) or {}
    return {
        "enabled": cfg.get("enabled", True),
        "column": cfg.get("column", FIX_CR_COL),
        "capital_multiplier": cfg.get(
            "capital_multiplier", cfg.get("multiplier", 10_000_000)
        ),
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


def build_fix_check(
    fix_accounts: pd.DataFrame, rules: dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Check fixed-capital accounts.

    FIX (CR) fixes the account's CAPITAL rather than its allocation:

        fixed_capital       = FIX (CR) x capital_multiplier    (1 -> 1,00,00,000)
        category_capital    = fixed_capital x pct(SubCategory)
        rounded             = round_half_up(category_capital, basis)
        expected allocation = rounded / divisor

    So FIX 1 on a 40% category gives 1,00,00,000 -> 40,00,000 -> 40,000.

    Overrides the Running file's capital, the previous-day rule and the Jainam
    rule, but not an excluded algo.

    Returns (result, no_category) where no_category holds FIX accounts whose
    SubCategory carries no percentage -- there is nothing to apply the fixed
    capital to, so they are reported rather than given an invented number.
    """
    empty_result = pd.DataFrame(columns=FIX_COLUMNS)
    if fix_accounts.empty:
        return empty_result, fix_accounts.copy()

    cfg = fix_config(rules)
    work = fix_accounts.copy()
    for col in ("alias", "operator_name"):
        if col not in work.columns:
            work[col] = ""
    if "algo" not in work.columns:
        work["algo"] = np.nan

    work[SUBCATEGORY_COL] = work[SUBCATEGORY_COL].map(normalize_subcategory)
    sub_rules = {str(k).strip().upper(): v for k, v in rules["subcategories"].items()}

    def _pct(sub: str):
        cfg_sub = sub_rules.get(sub, {})
        if cfg_sub.get("action") != "check":
            return np.nan
        pct = cfg_sub.get("pct")
        return float(pct) if isinstance(pct, (int, float)) and not isinstance(pct, bool) \
            else np.nan

    work["pct"] = work[SUBCATEGORY_COL].map(_pct)

    no_category = work[work["pct"].isna()].copy()
    if not no_category.empty:
        logger.warning(
            "FIX check: %d account(s) have no category percentage and were not "
            "checked: %s",
            len(no_category),
            sorted({f"{u} ({s or '<blank>'})" for u, s
                    in zip(no_category["userid"], no_category[SUBCATEGORY_COL])}),
        )

    work = work[work["pct"].notna()].copy()
    if work.empty:
        return empty_result, no_category

    basis = rules["rounding"]["basis"]
    divisor = rules["rounding"]["divisor"]
    round_mode = rules["rounding"].get("mode", "half_up")

    work["fix_cr"] = pd.to_numeric(work[cfg["column"]], errors="coerce")
    work["fixed_capital"] = work["fix_cr"] * cfg["capital_multiplier"]
    work["category_capital"] = work["fixed_capital"] * work["pct"].map(pct_fraction)
    work["rounded_capital"] = round_to_basis(work["category_capital"], basis, round_mode)
    work["expected_allocation"] = work["rounded_capital"] / divisor
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
        "FIX check: %d account(s) checked, %d match, %d mismatch, %d without a "
        "category percentage.",
        len(result), int((result["status"] == STATUS_MATCH).sum()),
        int((result["status"] == STATUS_MISMATCH).sum()), len(no_category),
    )
    return result, no_category


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

    def _slice(frame: pd.DataFrame, extra: Optional[List[str]] = None) -> pd.DataFrame:
        cols = display_cols + [c for c in (extra or [])]
        if frame is None or frame.empty:
            return pd.DataFrame(columns=cols)
        keep = [c for c in cols if c in frame.columns]
        return frame[keep].reset_index(drop=True)

    def _tables(**kw) -> Dict[str, pd.DataFrame]:
        base = {
            "result": pd.DataFrame(columns=RESULT_COLUMNS),
            "unknown_subcategory": pd.DataFrame(columns=display_cols),
            "excluded": pd.DataFrame(columns=display_cols),
            "jexceptions": pd.DataFrame(columns=display_cols),
            "jainam_result": pd.DataFrame(columns=JAINAM_COLUMNS),
            "fix_result": pd.DataFrame(columns=FIX_COLUMNS),
            "fix_invalid": pd.DataFrame(columns=display_cols),
            "fix_no_category": pd.DataFrame(columns=display_cols),
            "broker_result": pd.DataFrame(columns=BROKER_COLUMNS),
            "broker_unresolved": pd.DataFrame(columns=display_cols),
            "zero_sl_result": zero_sl_result,
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

    # --- 0 SL max-loss check: runs alongside the allocation rules, not instead
    # of them, so an account may appear here AND on its allocation rule.
    zero_sl_result = build_zero_sl_check(in_scope, rules)

    # --- 0a. excluded algos: skipped by every rule, but kept visible ---
    excluded_algo, after_algo = split_excluded_algos(in_scope, rules)

    # --- 0b. broker rules: outrank Jainam, previous-day, FIX and category ---
    broker_accounts, after_broker = split_broker_accounts(after_algo, rules)
    broker_result, broker_unresolved = build_broker_check(broker_accounts, df_run, rules)

    # --- 1. classify by SubCategory (Jainam is extracted before any routing) ---
    work = after_broker.copy()
    # Element-wise rather than .astype(str).str.upper(): on a nullable dtype
    # (string[python], Float64) the .str accessor propagates NA instead of
    # stringifying it, leaving floats mixed in with strings downstream.
    work[SUBCATEGORY_COL] = work[SUBCATEGORY_COL].map(normalize_subcategory)

    sub_rules = {str(k).strip().upper(): v for k, v in rules["subcategories"].items()}
    action = work[SUBCATEGORY_COL].map(lambda s: sub_rules.get(s, {}).get("action"))

    unknown = work[action.isna()].copy()
    excluded = work[action == "exclude"].copy()
    ja_accounts = work[action == "jexception"].copy()
    checkable = work[action == "check"].copy()

    if not unknown.empty:
        # str() on every element: this is only a log line and must never be the
        # thing that takes the check down.
        seen = sorted({str(v) if str(v) else "<blank>" for v in unknown[SUBCATEGORY_COL]})
        logger.warning(
            "Allocation Check %s: %d account(s) have a SubCategory not defined "
            "in the rules file: %s", mode, len(unknown), seen,
        )

    # --- 2. Jainam accounts (MSJ) -> Jainam sheet, ahead of previous-day ---
    jainam_result = build_jainam_check(ja_accounts, df_jainam, rules)

    # --- 3. route: previous-day outranks FIX, so routing happens BEFORE the
    # FIX split. A POS account goes to the previous-day check even when it
    # carries a FIX (CR) value.
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

    # --- 4. FIX applies only to what is left for the capital rule ---
    fix_accounts, fix_invalid, capital_accounts = split_fix_accounts(
        routed[METHOD_CAPITAL], rules
    )
    fix_result, fix_no_category = build_fix_check(fix_accounts, rules)

    def _finalise(result, not_in_running, no_capital) -> Dict[str, pd.DataFrame]:
        return _tables(
            result=result,
            unknown_subcategory=_slice(unknown),
            excluded=_slice(excluded),
            jexceptions=_slice(ja_accounts),
            jainam_result=jainam_result,
            fix_result=fix_result,
            fix_invalid=_slice(fix_invalid),
            fix_no_category=_slice(fix_no_category),
            broker_result=broker_result,
            broker_unresolved=_slice(broker_unresolved, extra=["_reason"]),
            zero_sl_result=zero_sl_result,
            excluded_algo=_slice(excluded_algo),
            not_in_running=_slice(not_in_running),
            no_capital=_slice(no_capital),
            prevday_result=prevday_result,
            prevday_new=prevday_new,
            unroutable=_slice(unroutable),
        )

    if capital_accounts.empty:
        return _finalise(pd.DataFrame(columns=RESULT_COLUMNS), None, None)

    # --- 5. capital rule ---
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

    # --- broker rules ---
    broker = tables.get("broker_result", pd.DataFrame())
    if not broker.empty:
        part = base(broker)
        part["rule"] = [
            f"{RULE_BROKER} {b} {p:g}%" if m == BROKER_METHOD_CAPITAL_PCT and pd.notna(p)
            else f"{RULE_BROKER} {b} Fixed"
            for b, m, p in zip(broker[BROKER_COL], broker["broker_rule"], broker["pct"])
        ]
        part["allocation"] = broker["actual_allocation"]
        part["expected allocation"] = broker["expected_allocation"]
        part["category capital"] = broker["category_capital"]
        part["capital"] = broker[CAPITAL_COL]
        part["status"] = broker["status"]
        part["remark"] = np.where(
            broker["status"] == STATUS_MATCH, "",
            "Allocation differs from the broker rule",
        )
        parts.append(part)

    # --- FIX (CR) exception ---
    fix = tables.get("fix_result", pd.DataFrame())
    if not fix.empty:
        part = base(fix)
        # Show the percentage alongside, since FIX now runs the category rule
        # on a fixed capital rather than setting the allocation directly.
        part["rule"] = fix["pct"].map(
            lambda p: f"{RULE_FIX} {p:g}%" if pd.notna(p) else RULE_FIX
        )
        part["allocation"] = fix["actual_allocation"]
        part["expected allocation"] = fix["expected_allocation"]
        part["category capital"] = fix["category_capital"]
        part["capital"] = fix["fixed_capital"]
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
        ("broker_unresolved", "Broker rule could not be resolved (no capital or no FIX value)"),
        ("fix_invalid", "FIX (CR) value is unusable (0, negative or non-numeric)"),
        ("fix_no_category", "FIX account has no category percentage"),
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


# -----------------------------
# EXPORT LAYOUT (one sheet per rule)
# -----------------------------
# Exactly the columns requested by the business, in order, with `remark` last
# so the Not Checked sheet can say WHY an account was skipped.
EXPORT_COLUMNS = [
    "userid", "alias", "server", "algo", "Rule", "sub category",
    "pct", "capital", "category_capital", "rounded_capital",
    "expected_allocation", "actual_allocation", "difference", "status",
    "operator_name", "remark",
]

# Sheet order. Every sheet is always written, even when empty, so the workbook
# has a stable shape day to day.
EXPORT_RULES = [
    "Previous Day", "Category", "Broker", "Algo", "Jainam", "Fixed", "Not Checked",
]


def build_export(tables: Dict[str, pd.DataFrame], in_scope: pd.DataFrame) -> pd.DataFrame:
    """
    One row per in-scope account with a `Rule` column, in the export layout.

    This is the single source for the downloaded workbook: the Summary sheet is
    this frame, and each rule sheet is a filter on `Rule`.
    """
    def frame(source: pd.DataFrame, rule: str, **cols) -> pd.DataFrame:
        if source is None or source.empty:
            return pd.DataFrame(columns=EXPORT_COLUMNS)
        out = pd.DataFrame(index=source.index)
        out["userid"] = source.get("userid", "")
        out["alias"] = source.get("alias", "")
        out["server"] = source.get("server", "")
        out["algo"] = source.get("algo", np.nan)
        out["Rule"] = rule
        out["sub category"] = source.get(SUBCATEGORY_COL, "")
        out["operator_name"] = source.get("operator_name", "")
        for name in ("pct", "capital", "category_capital", "rounded_capital",
                     "expected_allocation", "actual_allocation", "difference",
                     "status", "remark"):
            value = cols.get(name)
            if value is None:
                out[name] = "" if name in ("status", "remark") else np.nan
            else:
                out[name] = value
        return out[EXPORT_COLUMNS]

    parts: List[pd.DataFrame] = []

    cap = tables.get("result", pd.DataFrame())
    parts.append(frame(
        cap, "Category",
        pct=cap.get("pct"), capital=cap.get(CAPITAL_COL),
        category_capital=cap.get("category_capital"),
        rounded_capital=cap.get("rounded_capital"),
        expected_allocation=cap.get("expected_allocation"),
        actual_allocation=cap.get("actual_allocation"),
        difference=cap.get("difference"), status=cap.get("status"),
    ))

    fix = tables.get("fix_result", pd.DataFrame())
    parts.append(frame(
        fix, "Fixed",
        pct=fix.get("pct"), capital=fix.get("fixed_capital"),
        category_capital=fix.get("category_capital"),
        rounded_capital=fix.get("rounded_capital"),
        expected_allocation=fix.get("expected_allocation"),
        actual_allocation=fix.get("actual_allocation"),
        difference=fix.get("difference"), status=fix.get("status"),
        remark="Capital fixed from the FIX (CR) column",
    ))

    brk = tables.get("broker_result", pd.DataFrame())
    if not brk.empty:
        is_fix_method = brk["broker_rule"] == BROKER_METHOD_FIX
        # A fix_allocation broker derives the allocation straight from FIX (CR)
        # and never touches the running capital. Showing that capital would
        # imply it fed the calculation, so blank it and name the driver in the
        # remark instead -- otherwise the number is untraceable.
        broker_capital = brk[CAPITAL_COL].where(~is_fix_method)
        broker_remark = [
            f"{b} - from FIX (CR) {f:g}" if m == BROKER_METHOD_FIX and pd.notna(f)
            else (f"{b} - from FIX (CR)" if m == BROKER_METHOD_FIX else str(b))
            for b, m, f in zip(brk[BROKER_COL], brk["broker_rule"], brk["fix_cr"])
        ]
    else:
        broker_capital, broker_remark = None, None
    parts.append(frame(
        brk, "Broker",
        pct=brk.get("pct"), capital=broker_capital,
        category_capital=brk.get("category_capital"),
        rounded_capital=brk.get("rounded_capital"),
        expected_allocation=brk.get("expected_allocation"),
        actual_allocation=brk.get("actual_allocation"),
        difference=brk.get("difference"), status=brk.get("status"),
        remark=broker_remark,
    ))

    jai = tables.get("jainam_result", pd.DataFrame())
    parts.append(frame(
        jai, "Jainam",
        expected_allocation=jai.get("expected_allocation"),
        actual_allocation=jai.get("actual_allocation"),
        difference=jai.get("difference"), status=jai.get("status"),
        remark=jai.get("remark"),
    ))

    prev = tables.get("prevday_result", pd.DataFrame())
    parts.append(frame(
        prev, "Previous Day",
        expected_allocation=prev.get("previous_allocation"),
        actual_allocation=prev.get("today_allocation"),
        difference=prev.get("difference"), status=prev.get("status"),
        remark="Compared against the previous day's allocation",
    ))

    new = tables.get("prevday_new", pd.DataFrame())
    parts.append(frame(
        new, "Previous Day",
        actual_allocation=new.get("today_allocation"),
        status=STATUS_NEW_USER,
        remark="Not present in the previous day's All Users sheet",
    ))

    algo = tables.get("algo_result", pd.DataFrame())
    if algo is not None and not algo.empty:
        parts.append(frame(
            algo, "Algo",
            pct=algo.get("pct"), capital=algo.get(CAPITAL_COL),
            category_capital=algo.get("category_capital"),
            rounded_capital=algo.get("rounded_capital"),
            expected_allocation=algo.get("expected_allocation"),
            actual_allocation=algo.get("actual_allocation"),
            difference=algo.get("difference"), status=algo.get("status"),
        ))

    for key, reason in (
        ("excluded_algo", "Algo excluded by rule"),
        ("broker_unresolved", "Broker rule could not be resolved (no capital or no FIX value)"),
        ("fix_invalid", "FIX (CR) value is unusable (0, negative or non-numeric)"),
        ("fix_no_category", "FIX account has no category percentage"),
        ("unknown_subcategory", "SubCategory not defined in the rules file"),
        ("excluded", "SubCategory excluded by rule"),
        ("not_in_running", "Not present in the Running file"),
        ("no_capital", "Capital is blank or <= 0 in the Running file"),
        ("unroutable", "Matched no routing rule for this mode"),
    ):
        src = tables.get(key, pd.DataFrame())
        # Prefer the per-row reason where the check recorded one (broker rules
        # know whether it was the capital or the FIX value that was missing).
        row_reason = reason
        if src is not None and not src.empty and "_reason" in src.columns:
            row_reason = src["_reason"].replace("", np.nan).fillna(reason)
        parts.append(frame(
            src, "Not Checked",
            actual_allocation=pd.to_numeric(src.get("allocation"), errors="coerce")
            if src is not None and not src.empty else None,
            status=STATUS_NOT_CHECKED, remark=row_reason,
        ))

    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame(columns=EXPORT_COLUMNS)

    combined = pd.concat(parts, ignore_index=True)
    order = {STATUS_MISMATCH: 0, STATUS_NEW_USER: 1, STATUS_NOT_CHECKED: 2, STATUS_MATCH: 3}
    combined["_o"] = combined["status"].map(order).fillna(9)
    combined = (
        combined.sort_values(["_o", "Rule", "sub category", "userid"])
        .drop(columns=["_o"]).reset_index(drop=True)
    )
    return combined[EXPORT_COLUMNS]


def export_sheets(
    export: pd.DataFrame, zero_sl: Optional[pd.DataFrame] = None
) -> Dict[str, pd.DataFrame]:
    """
    Summary plus one sheet per rule, in a fixed order.

    The 0 SL sheet is appended last and has its own columns: it checks max_loss
    rather than allocation, so it is not part of the one-row-per-account
    Summary and an account may legitimately appear on both.
    """
    sheets = {"Summary": export}
    for rule in EXPORT_RULES:
        sheets[rule] = (
            export[export["Rule"] == rule].reset_index(drop=True)
            if not export.empty else pd.DataFrame(columns=EXPORT_COLUMNS)
        )
    sheets[RULE_ZERO_SL] = (
        zero_sl if zero_sl is not None else pd.DataFrame(columns=ZERO_SL_COLUMNS)
    )
    return sheets


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
        + len(tables.get("fix_no_category", []))
        + len(tables.get("broker_result", []))
        + len(tables.get("broker_unresolved", []))
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
