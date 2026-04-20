#!/usr/bin/env python3
"""Compute rolling-window moving average and variance for one bill denomination.

Expected input columns (wide format):
- date
- twenty_dollar_bills_withdrawn
- fifty_dollar_bills_withdrawn
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
from collections import deque
from pathlib import Path


DENOMINATION_TO_COLUMN = {
    "20": "twenty_dollar_bills_withdrawn",
    "50": "fifty_dollar_bills_withdrawn",
    "100": "hundred_dollar_bills_withdrawn",
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute n-day rolling moving average and variance for one bill denomination."
        )
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=Path("daily_bill_pattern_wide.csv"),
        help="Input CSV file path (default: daily_bill_pattern_wide.csv).",
    )
    parser.add_argument(
        "--denomination",
        choices=sorted(DENOMINATION_TO_COLUMN.keys()),
        required=True,
        help="Bill denomination to analyze: 20 or 50.",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=7,
        help="Rolling window size in days (default: 7).",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Output CSV path. Default: rolling_stats_<denomination>_window<windowDays>.csv",
    )
    parser.add_argument(
        "--min-periods",
        type=int,
        default=None,
        help=(
            "Minimum observations required before emitting stats. "
            "Default equals --window-days."
        ),
    )
    return parser


def parse_date(value: str) -> dt.date:
    return dt.date.fromisoformat(value)


def load_series(input_file: Path, value_column: str) -> list[tuple[dt.date, int]]:
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    rows: list[tuple[dt.date, int]] = []
    with input_file.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"date", value_column}
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"Input CSV missing required columns: {', '.join(missing)}")

        for row in reader:
            rows.append((parse_date(row["date"]), int(row[value_column])))

    rows.sort(key=lambda x: x[0])
    return rows


def compute_rolling_stats(
    series: list[tuple[dt.date, int]],
    window_days: int,
    min_periods: int,
) -> list[dict[str, object]]:
    if window_days <= 0:
        raise ValueError("--window-days must be >= 1")
    if min_periods <= 0:
        raise ValueError("--min-periods must be >= 1")
    if min_periods > window_days:
        raise ValueError("--min-periods cannot be greater than --window-days")

    window: deque[tuple[dt.date, int]] = deque()
    rolling_sum = 0.0
    rolling_sum_sq = 0.0
    out: list[dict[str, object]] = []

    for day, value in series:
        window.append((day, value))
        rolling_sum += value
        rolling_sum_sq += value * value

        if len(window) > window_days:
            _, dropped = window.popleft()
            rolling_sum -= dropped
            rolling_sum_sq -= dropped * dropped

        n = len(window)
        if n < min_periods:
            continue

        mean = rolling_sum / n
        variance_population = (rolling_sum_sq / n) - (mean * mean)
        variance_population = max(0.0, variance_population)

        out.append(
            {
                "date": day.isoformat(),
                "window_start_date": window[0][0].isoformat(),
                "window_end_date": window[-1][0].isoformat(),
                "window_observations": n,
                "moving_average": round(mean, 4),
                "moving_variance": round(variance_population, 4),
            }
        )

    return out


def write_output(output_file: Path, rows: list[dict[str, object]]) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "date",
        "window_start_date",
        "window_end_date",
        "window_observations",
        "moving_average",
        "moving_variance",
    ]

    with output_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = build_arg_parser().parse_args()
    value_column = DENOMINATION_TO_COLUMN[args.denomination]
    min_periods = args.window_days if args.min_periods is None else args.min_periods

    output_file = (
        args.output_file
        if args.output_file is not None
        else Path(f"rolling_stats_{args.denomination}_window{args.window_days}.csv")
    )

    series = load_series(args.input_file, value_column)
    results = compute_rolling_stats(series, args.window_days, min_periods)
    write_output(output_file, results)

    print(f"Input file: {args.input_file}")
    print(f"Denomination: ${args.denomination}")
    print(f"Window days: {args.window_days}")
    print(f"Min periods: {min_periods}")
    print(f"Rows written: {len(results):,}")
    print(f"Output file: {output_file}")


if __name__ == "__main__":
    main()
