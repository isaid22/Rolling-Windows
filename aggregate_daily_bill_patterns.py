#!/usr/bin/env python3
"""Aggregate daily ATM withdrawals across all ATM IDs by bill denomination.

Input expected columns:
- date
- one or more columns ending with _dollar_bills_withdrawn

Outputs:
- Wide format CSV: one row per day with totals for each denomination column.
- Long format CSV: one row per day per denomination (useful for plotting).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
from collections import defaultdict
from pathlib import Path


def extract_denom_label(column_name: str) -> str:
    if not column_name.endswith("_dollar_bills_withdrawn"):
        return column_name
    prefix = column_name[: -len("_dollar_bills_withdrawn")]
    return "$" + prefix.split("_")[0]


def parse_iso_date(value: str) -> dt.date:
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid date '{value}'. Expected format: YYYY-MM-DD"
        ) from exc


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate all ATM IDs into daily denomination-level withdrawal totals."
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=Path("atm_withdrawals_2024_2025.csv"),
        help="Path to input CSV (default: atm_withdrawals_2024_2025.csv).",
    )
    parser.add_argument(
        "--start-date",
        type=parse_iso_date,
        default=None,
        help="Optional start date filter in YYYY-MM-DD.",
    )
    parser.add_argument(
        "--end-date",
        type=parse_iso_date,
        default=None,
        help="Optional end date filter in YYYY-MM-DD.",
    )
    parser.add_argument(
        "--output-wide",
        type=Path,
        default=Path("daily_bill_pattern_wide.csv"),
        help="Output wide-format CSV (default: daily_bill_pattern_wide.csv).",
    )
    parser.add_argument(
        "--output-long",
        type=Path,
        default=Path("daily_bill_pattern_long.csv"),
        help="Output long-format CSV (default: daily_bill_pattern_long.csv).",
    )
    return parser


def within_range(day: dt.date, start: dt.date | None, end: dt.date | None) -> bool:
    if start and day < start:
        return False
    if end and day > end:
        return False
    return True


def read_and_aggregate(input_file: Path, start: dt.date | None, end: dt.date | None):
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    daily_totals: dict[str, dict[str, int]] = defaultdict(dict)

    with input_file.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        denom_columns = [c for c in fieldnames if c.endswith("_dollar_bills_withdrawn")]
        required = {"date"}
        if not required.issubset(set(fieldnames)):
            missing = sorted(required - set(fieldnames))
            raise ValueError(f"Input CSV missing required columns: {', '.join(missing)}")
        if not denom_columns:
            raise ValueError(
                "Input CSV must include at least one *_dollar_bills_withdrawn column."
            )

        for row in reader:
            day = dt.date.fromisoformat(row["date"])
            if not within_range(day, start, end):
                continue

            day_key = day.isoformat()
            day_bucket = daily_totals[day_key]
            for col in denom_columns:
                day_bucket[col] = day_bucket.get(col, 0) + int(row[col])

    return daily_totals, denom_columns


def write_wide(
    path: Path,
    daily_totals: dict[str, dict[str, int]],
    denom_columns: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["date", *denom_columns, "total_bills_withdrawn"]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for day in sorted(daily_totals):
            row_out = {"date": day}
            total = 0
            for col in denom_columns:
                value = daily_totals[day].get(col, 0)
                row_out[col] = value
                total += value
            row_out["total_bills_withdrawn"] = total
            writer.writerow(row_out)


def write_long(
    path: Path,
    daily_totals: dict[str, dict[str, int]],
    denom_columns: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["date", "denomination", "bills_withdrawn"]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for day in sorted(daily_totals):
            for col in denom_columns:
                writer.writerow(
                    {
                        "date": day,
                        "denomination": extract_denom_label(col),
                        "bills_withdrawn": daily_totals[day].get(col, 0),
                    }
                )


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.start_date and args.end_date and args.end_date < args.start_date:
        raise ValueError("--end-date must be on or after --start-date")

    daily_totals, denom_columns = read_and_aggregate(
        args.input_file,
        args.start_date,
        args.end_date,
    )
    write_wide(args.output_wide, daily_totals, denom_columns)
    write_long(args.output_long, daily_totals, denom_columns)

    print(f"Input file: {args.input_file}")
    print(f"Days aggregated: {len(daily_totals):,}")
    print(f"Denomination columns: {', '.join(denom_columns)}")
    print(f"Wrote wide output: {args.output_wide}")
    print(f"Wrote long output: {args.output_long}")


if __name__ == "__main__":
    main()
