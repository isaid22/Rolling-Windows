# UI Demo: ATM Rolling Window Analytics

This Streamlit app provides a professional, clean UI for exploring ATM withdrawal CSVs and rolling-window statistics.

## Features
- Dropdown to select an existing CSV file from the project root.
- Optional CSV upload for ad-hoc analysis.
- CSV table preview in the app.
- User-controlled rolling window length (n days).
- User-selectable date column.
- Works with arbitrary denomination/value columns (for example 20, 50, 100, or non-ATM series).
- User-selectable series to plot (one or many).
- Rolling trend chart with one moving-average line and an uncertainty band.
- Band mode switch:
  - `stddev`: mean +/- sqrt(variance)
  - `variance`: mean +/- variance
- Explicit variance visibility (metrics panel, variance chart, and rolling stats table).
- Download rolling stats CSV directly from the UI.

## Run
From the repository root:

```bash
source ~/ai_venv/bin/activate
pip install -r UI-demo/requirements.txt
streamlit run UI-demo/app.py
```

## Supported Input CSV Shapes
1. Wide shape:
- One date column (user-selected in UI)
- One or more numeric series columns (user-selected in UI)

2. Long shape:
- One date column (user-selected)
- One series-label column (for example denomination)
- One numeric value column

If the CSV has multiple ATM rows per day, the app automatically aggregates by date.
