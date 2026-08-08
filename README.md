# Morning Sheet Comparison Tool

A Streamlit app for reconciling the **All Users** master sheet against the **Running Users** sheet every morning. It highlights allocation mismatches, missing accounts, fixed-allocation violations, and low-allocation flags — all in one place.

---

## How to Run

```bash
pip install -r requirements.txt
streamlit run morning_comp.py
```

---

## Input Files

### 1. All Users File (Excel — `.xlsx`)
- Must contain a sheet named **`Main`** or **`Mfibain`** (case-insensitive)
- Required columns: `userId`, `alias`, `allocation`, `max_loss`, `server`, `algo`, `Running Type`, `Running Days`, `FIX (CR)`, `SL`, `Operator Name`
- `FIX (CR)` is a **numeric** column (blank = not a fixed account). It replaces the old `Remarks/...` free-text parsing for the Fixed tab.
- `SL` drives the **0 SL** tab. `Operator Name` is appended to every table.

### 2. Running Users File (CSV or Excel)
- Same core columns: `userid`, `alias`, `allocation`, `max_loss`, `server`, `algo`
- Allocation and max_loss values are expected in **raw form** (divided by 100 automatically)

---

## Comparison Modes

Select a mode from the sidebar before uploading files. The mode controls which rows from the All Users sheet participate in the comparison.

| Mode | RunningType filter | RunningDays filter | Base server exclusions |
|------|-------------------|-------------------|----------------------|
| **0DTE** | None | None | Yes |
| **1DTE** | POS, INT | DAILY, 1DTE/0DTE | Yes |
| **4DTE** | POS | DAILY | No |

---

## Data Normalization

| Field | Rule |
|-------|------|
| `userid` | Trimmed, spaces removed, uppercased |
| `server` | Lowercased, trimmed |
| `allocation` | Running file divided by 100, then floored |
| `max_loss` | Running file converted to absolute, then floored |
| `runningtype` / `runningdays` | Lowercased, spaces removed |

---

## Server Exclusions

### All Users sheet (0DTE and 1DTE modes):
- `NOT RUNNING`
- `DLR ACC`
- `Z SERVER`

### Running Users sheet (all modes):
- `Z SERVER`

Excluded accounts appear in the **Extra** tab. In 4DTE mode, no base exclusions are applied (Extra tab is empty).

---

## Tabs

### 1. Difference
Users present in **both** files but with at least one mismatched field:
- `alias`, `allocation`, `max_loss`, `server`, `algo`
- Side-by-side columns: `_all` (All Users) vs `_run` (Running)

### 2. Not Found
Users present in one file but missing from the other.

| userid | server | algo | Not found in |
|--------|--------|------|-------------|
| ABC123 | vs1 | 7 | Running |
| XYZ789 | vs2 | 1 | All User |

### 3. Extra
Accounts excluded from comparison due to their server type (DLR ACC, NOT RUNNING, Z SERVER). Shown with source file label.

### 4. Duplicate
User IDs that appear more than once in either file. Duplicates are **excluded from all comparisons**.

| userid | Found in |
|--------|----------|
| ABC123 | All User |
| XYZ999 | Running |

### 5. Fixed
Accounts that carry a value in the **`FIX (CR)`** column of the main sheet.

> **Changed:** this tab previously parsed FIX instructions out of the free-text
> `Remarks/Algo8 Previous day realised MTM` column. It is now driven **entirely**
> by the dedicated `FIX (CR)` numeric column. Remark text is no longer read, and
> is no longer shown on the tab.

**Rule:**

```
expected_allocation = FIX (CR) × 1,00,000
```

| `FIX (CR)` | Expected allocation |
|-----------|-------------------|
| 1 | 1,00,000 |
| 1.6 | 1,60,000 |
| 3 | 3,00,000 |
| 0.8 | 80,000 |
| N | N × 1,00,000 |

The product is **rounded** to the nearest rupee (not floored) — flooring a binary
float product can land a rupee low, e.g. `1.15 × 1,00,000`.

**Cell handling:**

| `FIX (CR)` cell | Behaviour |
|----------------|-----------|
| Blank / empty / `-` | Account is not on a fixed allocation — skipped silently |
| Positive number (incl. numeric text like `" 1.4 "`) | Checked normally |
| `0` or negative | Excluded, listed in an on-screen warning |
| Non-numeric text | Excluded, listed in an on-screen warning |

An account with a FIX instruction written **only** in the remark text and a blank
`FIX (CR)` cell will **not** be checked. Populating `FIX (CR)` is now the single
source of truth.

**Columns:** `userid`, `alias`, `server`, `algo`, `fix_cr`, `expected_allocation`, `actual_allocation`, `status`

**Colour coding:**
- 🟥 **Red row** — actual allocation does not match the expected FIX amount (mismatch)
- 🟩 **Green row** — actual allocation matches

**Exclusions:** DLR ACC, NOT RUNNING, and Z SERVER accounts are always excluded from this tab regardless of mode.

**Mode behaviour:** The Fixed tab respects the active DTE mode filter — only accounts that pass the RunningType/RunningDays filter for the selected mode are shown.

### 6. Allocation
Flags accounts whose allocation is **below the mode threshold** but above zero. Not applicable in 0DTE mode.

| Mode | Threshold | Action |
|------|-----------|--------|
| **4DTE** | < 1,00,000 | Flag and show |
| **1DTE** | < 60,000 | Flag and show |
| **0DTE** | — | Not applicable |

Columns shown: `server`, `algo`, `userid`, `alias`, `allocation`
All flagged rows are highlighted in **orange**.

### 7. 0 SL
Pivot of accounts running with **no stop-loss**, from the `SL` column.

| algo | server | count of 0 SL accounts | operator_name |
|------|--------|----------------------|--------------|
| 1 | vs3 | 23 | MOHITC |
| 7 | vs4 | 23 | SHIVANSHU |
| 19 | vs22 | 27 | DUSHYANT |

**What counts as 0:** only a **numeric zero** — `0`, `0.0`, `"0"`, `" 0 "`, `"0%"`.
A **blank `SL` cell means "no SL recorded" and is NOT counted.** Non-numeric text
is ignored and logged.

**Scope:** the userid set for the active **DTE mode**. **No server exclusions** —
`NOT RUNNING` / `DLR ACC` / `Z SERVER` accounts still appear here, deliberately,
since an account with no SL matters regardless of where it sits.

Counts are **distinct userid** per algo/server. Rows are highlighted in **purple**.

---

## Operator Name

`Operator Name` from the All Users sheet is appended as the **last column of every
table**, on screen and in the Excel export.

**How it is resolved:** operator is treated as an attribute of the
**`(algo, server)` pair**, not of the individual account. A lookup is built from
the All Users sheet (scoped to the active DTE mode) and applied to every table.

This matters because the **Running Users file has no operator column** — rows
sourced from it (accounts found only in Running, Running duplicates, Running
extras) are resolved through the same algo/server map.

| Situation | Result |
|-----------|--------|
| Pair found in the lookup | Operator filled |
| Pair not found (e.g. a Running row on an unknown algo/server) | **Blank** — never guessed |
| Pair maps to more than one operator | Most frequent wins, warning logged |
| `Operator Name` column missing entirely | Blank everywhere, on-screen warning |

Attachment is done by positional lookup rather than a merge, so **row count and
order are guaranteed unchanged** — a merge on a non-unique key would silently
multiply rows.

---

## Download

Click **Download Full Report** to export all tabs as a single `.xlsx` file:
- `Summary`, `0 SL`, `Difference`, `Not Found`, `Not Found Summary`, `Extra`, `Duplicate`, `Fixed`, `Allocation`
- Every sheet carries `operator_name` as its last column.
- The Fixed sheet preserves red/green row colouring.

---

## Important Notes

- Duplicate user IDs are excluded from all comparisons
- Only `allocation`, `max_loss`, `server`, and `algo` are used for difference detection; `alias` is displayed only
- Matching rows are not shown in the Difference tab
- The Fixed tab reads the `FIX (CR)` column only — remark text is ignored

---

## Tests

```bash
python tests/test_fixed_tab.py          # Fixed tab / FIX (CR)
python tests/test_zero_sl_operator.py   # 0 SL pivot / operator column
```

Both run against `Data/All User Details Daily Updated.xlsx` plus synthetic edge
cases. `test_zero_sl_operator.py` also runs the **full pipeline end to end across
all three DTE modes**, including the Excel export, and asserts that attaching the
operator column never changes a table's row count or order.

Point either at another file with:

```bash
ALL_USERS_FILE=/path/to/file.xlsx python tests/test_zero_sl_operator.py
```

---

*Developed for **MEGASERVE TECHNOLOGIES** by **YV23***
