#!/usr/bin/env python3
"""Plot rolling moving-average and moving-variance time series from CSV."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot moving average as one line with an upper/lower uncertainty band."
        )
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=Path("rolling_stats_20_window7.csv"),
        help="Input CSV file (default: rolling_stats_20_window7.csv).",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("rolling_stats_20_window7_plot.png"),
        help="Output image file (default: rolling_stats_20_window7_plot.png).",
    )
    parser.add_argument(
        "--title",
        default="Rolling 7-Day Stats for $20 Bills",
        help="Chart title.",
    )
    parser.add_argument(
        "--bound-mode",
        choices=["variance", "stddev"],
        default="variance",
        help=(
            "Band width source: 'variance' uses mean +/- variance, "
            "'stddev' uses mean +/- sqrt(variance). Default: variance."
        ),
    )
    return parser


def load_rolling_stats(input_file: Path):
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    dates: list[dt.date] = []
    moving_avg: list[float] = []
    moving_var: list[float] = []

    with input_file.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"date", "moving_average", "moving_variance"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

        for row in reader:
            dates.append(dt.date.fromisoformat(row["date"]))
            moving_avg.append(float(row["moving_average"]))
            moving_var.append(float(row["moving_variance"]))

    return dates, moving_avg, moving_var


def plot_series(
    dates: list[dt.date],
    moving_avg: list[float],
    moving_var: list[float],
    output_file: Path,
    title: str,
    bound_mode: str,
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 6), dpi=130)

    if bound_mode == "stddev":
        spread = [math.sqrt(max(0.0, v)) for v in moving_var]
    else:
        spread = [max(0.0, v) for v in moving_var]

    lower = [max(0.0, m - s) for m, s in zip(moving_avg, spread)]
    upper = [m + s for m, s in zip(moving_avg, spread)]

    ax.plot(
        dates,
        moving_avg,
        color="#0072B2",
        linewidth=1.9,
        marker="o",
        markersize=2.0,
        label="Moving Average",
    )
    ax.fill_between(
        dates,
        lower,
        upper,
        color="#56B4E9",
        alpha=0.28,
        label=("Variance Band" if bound_mode == "variance" else "Std Dev Band"),
    )

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Bills Withdrawn")

    ax.grid(True, alpha=0.25)
    locator = mdates.AutoDateLocator(minticks=8, maxticks=16)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    ax.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(output_file)
    plt.close(fig)


def main() -> None:
    args = build_arg_parser().parse_args()
    dates, moving_avg, moving_var = load_rolling_stats(args.input_file)
    plot_series(
        dates,
        moving_avg,
        moving_var,
        args.output_file,
        args.title,
        args.bound_mode,
    )
    print(f"Plotted {len(dates):,} rows from {args.input_file}")
    print(f"Band mode: {args.bound_mode}")
    print(f"Saved chart: {args.output_file}")


if __name__ == "__main__":
    main()
