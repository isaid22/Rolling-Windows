#!/usr/bin/env python3
"""Plot daily withdrawal lines for $20 and $50 bills from wide-format CSV."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot daily $20/$50 withdrawal counts from daily_bill_pattern_wide.csv"
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=Path("daily_bill_pattern_wide.csv"),
        help="Input wide-format CSV file (default: daily_bill_pattern_wide.csv).",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("daily_bill_pattern_line_plot.png"),
        help="Output chart image path (default: daily_bill_pattern_line_plot.png).",
    )
    parser.add_argument(
        "--title",
        default="Daily ATM Bill Withdrawals ($20 vs $50)",
        help="Chart title.",
    )
    return parser


def read_series(input_file: Path):
    if not input_file.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_file}. "
            "If your file is named daily_bill_patern_wide.csv, pass --input-file with that exact name."
        )

    dates: list[dt.date] = []
    twenty: list[int] = []
    fifty: list[int] = []

    with input_file.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {
            "date",
            "twenty_dollar_bills_withdrawn",
            "fifty_dollar_bills_withdrawn",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

        for row in reader:
            dates.append(dt.date.fromisoformat(row["date"]))
            twenty.append(int(row["twenty_dollar_bills_withdrawn"]))
            fifty.append(int(row["fifty_dollar_bills_withdrawn"]))

    return dates, twenty, fifty


def plot_lines(
    dates: list[dt.date],
    twenty: list[int],
    fifty: list[int],
    output_file: Path,
    title: str,
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 6), dpi=130)

    ax.plot(
        dates,
        twenty,
        color="#0072B2",
        linewidth=1.4,
        marker="o",
        markersize=2.2,
        label="$20 bills",
    )
    ax.plot(
        dates,
        fifty,
        color="#D55E00",
        linewidth=1.4,
        marker="o",
        markersize=2.2,
        label="$50 bills",
    )

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Bills Withdrawn (Count)")
    ax.grid(True, alpha=0.25)
    ax.legend()

    locator = mdates.AutoDateLocator(minticks=8, maxticks=16)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    fig.tight_layout()
    fig.savefig(output_file)
    plt.close(fig)


def main() -> None:
    args = build_arg_parser().parse_args()
    dates, twenty, fifty = read_series(args.input_file)
    plot_lines(dates, twenty, fifty, args.output_file, args.title)
    print(f"Plotted {len(dates):,} days from {args.input_file}")
    print(f"Saved chart: {args.output_file}")


if __name__ == "__main__":
    main()
