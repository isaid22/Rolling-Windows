# Rolling-Windows

This repository contains scripts to generate synthetic ATM withdrawal data, aggregate it,
compute rolling-window statistics, and visualize results (CLI + Streamlit UI).

## Quick Start (2023-2024 with $20/$50/$100)

Run these commands from the repository root in the exact order shown.

### 1. Activate environment

```bash
source ~/ai_venv/bin/activate
```

### 2. Generate ATM-level synthetic CSVs

This generates one file per year plus one combined file. The combined filename follows
your selected date range.

```bash
python generate_atm_withdrawal_data.py --start-date 2023-01-01 --end-date 2024-12-31 --seed 20260419
```

Expected combined output:

`atm_withdrawals_2023_2024.csv`

### 3. Verify generated files

```bash
ls -1 atm_withdrawals_2023.csv atm_withdrawals_2024.csv atm_withdrawals_2023_2024.csv
```

### 4. Aggregate daily totals across all ATMs

Produces:
- `daily_bill_pattern_wide.csv`
- `daily_bill_pattern_long.csv`

```bash
python aggregate_daily_bill_patterns.py --input-file atm_withdrawals_2023_2024.csv
```

### 5. Validate denomination columns

Confirm `$100` exists in wide format:

```bash
head -n 1 daily_bill_pattern_wide.csv
```

Confirm `$100` appears in long format:

```bash
head -n 12 daily_bill_pattern_long.csv
```

## Rolling Statistics (CLI)

### Example: $100 denomination, 14-day window

```bash
python rolling_window_bill_stats.py --input-file daily_bill_pattern_wide.csv --denomination 100 --window-days 14 --output-file rolling_stats_100_window14.csv
```

### Plot rolling stats (single line + band)

```bash
python plot_rolling_stats.py --input-file rolling_stats_100_window14.csv --output-file rolling_stats_100_window14_band_stddev.png --bound-mode stddev
```

## UI Demo

The Streamlit app is in `UI-demo/`.

### Install UI dependencies

```bash
pip install -r UI-demo/requirements.txt
```

### Run app

```bash
streamlit run UI-demo/app.py
```

## About py_compile

`py_compile` is a quick syntax check for Python files.

- It parses your script and creates Python bytecode in `__pycache__`.
- It does not run your app logic and does not build an executable binary.

Use it when:

- You edited a Python file and want to quickly catch syntax errors.
- You want a fast validation step before running Streamlit or longer jobs.

Example:

```bash
python -m py_compile UI-demo/app.py
```

## Notes

- `generate_atm_withdrawal_data.py` now emits:
	- `twenty_dollar_bills_withdrawn`
	- `fifty_dollar_bills_withdrawn`
	- `hundred_dollar_bills_withdrawn`
- Combined output filename defaults to:
	- `atm_withdrawals_<startYear>_<endYear>.csv`

