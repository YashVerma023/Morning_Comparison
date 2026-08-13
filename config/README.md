# `allocation_rules.json`

Every rule the Allocation Check applies. Editing this file changes what each
account is measured against, so it is validated strictly on load — a bad edit
stops the app with a clear message rather than producing plausible wrong numbers.

Most of it is editable in the app: **Allocation Check → Active rules**. Saving
there rewrites this file and keeps a backup at `allocation_rules.json.bak`.

> **JSON has no comments.** That is why the documentation lives here instead of
> inside the file. The only prose key in the JSON is `_doc`, which points here.

---

## Order of precedence

An account is checked by exactly **one** allocation rule. The first match wins:

```
1. Excluded algo      →  not checked at all      (except MSJ, see below)
2. Broker rule        →  EXCELSTOCK / VMATHRAMCO
3. Jainam             →  SubCategory action 'jexception' (MSJ)
4. Previous Day       →  POS accounts: today's allocation vs yesterday's
5. Fixed              →  account has a FIX (CR) value
6. Category           →  running capital × SubCategory %
```

Previous Day sits **above** Fixed: a POS account is checked against yesterday
even when it carries a FIX (CR) value. FIX therefore applies only to accounts
routed to the capital rule.

Jainam sits **above** Previous Day: a MSJ account goes to the Jainam sheet
whether it is POS or INT.

**0 SL is separate.** It checks `max_loss`, not allocation, so it runs
*alongside* whichever rule above applied. An account can appear on both.

---

## `rounding`

Applied to category capital before it becomes an allocation.

```
rounded  = round(category_capital / basis) × basis
expected = rounded / divisor
```

| Key | Current | Meaning |
|-----|---------|---------|
| `basis` | `2500000` | Round capital to a multiple of 25,00,000 |
| `mode` | `half_up` | Exact halves round **up**. Also accepts `floor`, `ceil` |
| `divisor` | `100` | Rounded capital ÷ 100 = expected allocation |

The tie point is always `basis ÷ 2` — at 25,00,000 that is 12,50,000.
Expected allocations therefore step in `basis / divisor` = 25,000.

`half_up` is deliberate. Python and numpy round halves to **even**, which would
turn 90,00,000 into 80,00,000 instead of 1,00,00,000.

---

## `excluded_servers`

Accounts on these servers are out of scope in every mode and never appear in the
report. Matched case-insensitively.

## `excluded_algos`

Algos skipped by every rule. Those accounts stay **in scope** and appear as
`Not under check` with the remark *"Algo excluded by rule"* — nothing disappears
silently. Values compare numerically, so `8`, `8.0` and `"8"` all match.

**Exception:** Jainam-type accounts (any SubCategory whose action is
`jexception`) are still checked against the Jainam sheet even when their algo is
excluded. Keyed off the action, not the name, so renaming MSJ will not break it.

---

## `dte_filters`

Which accounts each DTE mode looks at, after `excluded_servers` is applied.
`null` means no filter on that column.

| Mode | Running Type | Running Days |
|------|--------------|--------------|
| 0DTE | any | any |
| 1DTE | POS | 1DTE/0DTE, DAILY |
| 4DTE | POS | DAILY |

These are the **Allocation Check** filters and are intentionally independent of
the Login Check filters in `morning_comp.py`.

## `previous_day`

Whether the previous-day All Users sheet is `required`, `optional` or `unused`
per mode. 0DTE refuses to run without it.

## `routing`

Sends accounts to either the `capital` or `previous_day` method. Rules are read
in order and the **first match wins**; an account matching none is reported as
*"Cannot route"*.

---

## `subcategories`

| `action` | Needs | Meaning |
|----------|-------|---------|
| `check` | `pct` | Expected = capital × pct%, rounded |
| `exclude` | `pct: 0` | Not checked |
| `jexception` | — | Checked against the Jainam sheet |

`pct` is a **whole percent**: `100` = 100%, `60` = 60%, `40` = 40%.
**Do not write `0.6` for 60%** — a value between 0 and 1 is rejected as a likely
fraction mistake, because it would deploy 0.6% of capital.

A SubCategory found in the sheet but missing here is reported under
*"SubCategory not defined in the rules file"* and is never silently dropped.

---

## `broker_rules`

Per-broker overrides matched on the All Users `Broker` column, exact name,
case-insensitive. These outrank Fixed, Jainam and Category.

| `method` | Needs | Calculation |
|----------|-------|-------------|
| `capital_pct` | `pct` | `round(running_capital × pct%) / divisor` |
| `fix_allocation` | `multiplier` | `FIX (CR) × multiplier` |

Current: **EXCELSTOCK** takes 25% of running capital. **VMATHRAMCO** derives its
allocation from the FIX column, so `FIX 1 → 1,00,000` and the running capital is
ignored entirely.

An account whose broker rule cannot be resolved — no capital, or no FIX value —
is reported as `Not under check` rather than given an invented number.

---

## `fix`

A positive `FIX (CR)` value **fixes the account's capital**, it does not set the
allocation directly:

```
fixed_capital    = FIX (CR) × capital_multiplier      (FIX 1 → 1,00,00,000)
category_capital = fixed_capital × SubCategory %
expected         = round(category_capital) / divisor
```

So `FIX 1` on a 40% category gives `1,00,00,000 → 40,00,000 → 40,000`.

A blank cell means the account is not fixed. A populated but unusable value
(0, negative, text) is reported as `Not under check`, as is a FIX account whose
SubCategory has no percentage.

## `jainam`

MSJ accounts are checked against the `Jainam` sheet **inside the same All Users
workbook** — no extra upload.

```
expected = ALLOCATION × multiplier      (4 → 4,00,000)
```

All rows are used regardless of `Date`. `exclude_userids` drops the sheet's
trailing `Total` row. `ALLOCATION` of 0 means the expected allocation **is**
zero, so a non-zero actual allocation is a mismatch. A MSJ account absent from
the sheet is also a mismatch.

## `zero_sl`

Accounts with a numeric **0** in the `SL` column must satisfy:

```
max_loss = allocation × max_loss_multiplier      (default 30)
```

A blank SL cell means "no SL recorded" and is not checked. This checks max loss,
so it runs alongside the allocation rules rather than replacing them.

---

## Editing safely

- Validation runs **before** the file is written, so an invalid edit never
  reaches disk and cannot break the next startup.
- The write is atomic, so a crash mid-save cannot leave a half-written file.
- The previous version is kept as `allocation_rules.json.bak`.
- Rejected outright: `pct` between 0 and 1, `pct` above 100, negative values,
  duplicate SubCategory or broker names, unknown `action` or `method`.

**After editing this file by hand, restart the app.** Streamlit caches imported
modules, so a rerun alone may not pick the change up. Edits made through the app
apply immediately.
