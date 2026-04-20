#!/usr/bin/env python3
"""Generate synthetic daily ATM bill-withdrawal data for Lower Manhattan.

Creates three CSV files for Jan 1, 2024 through Dec 31, 2025:
- atm_withdrawals_2024.csv
- atm_withdrawals_2025.csv
- atm_withdrawals_2024_2025.csv
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ATMProfile:
    atm_id: str
    area: str
    base_traffic: float
    tourist_weight: float
    office_weight: float
    residential_weight: float
    weather_sensitivity: float


ATM_PROFILES = [
    ATMProfile("LM-ATM-001", "Battery Park", 1.05, 1.20, 0.70, 0.65, 1.20),
    ATMProfile("LM-ATM-002", "Wall Street", 1.25, 0.55, 1.40, 0.45, 1.05),
    ATMProfile("LM-ATM-003", "Broad Street", 1.18, 0.60, 1.35, 0.42, 1.05),
    ATMProfile("LM-ATM-004", "Fulton Center", 1.35, 1.00, 1.20, 0.60, 1.15),
    ATMProfile("LM-ATM-005", "South Street Seaport", 1.22, 1.35, 0.70, 0.55, 1.25),
    ATMProfile("LM-ATM-006", "Tribeca", 0.98, 0.75, 0.90, 1.00, 0.95),
    ATMProfile("LM-ATM-007", "City Hall", 1.08, 0.85, 1.10, 0.70, 1.00),
    ATMProfile("LM-ATM-008", "Chinatown", 1.12, 1.05, 0.85, 0.90, 1.00),
    ATMProfile("LM-ATM-009", "Little Italy", 1.04, 1.25, 0.70, 0.75, 1.10),
    ATMProfile("LM-ATM-010", "SoHo Broadway", 1.30, 1.50, 0.85, 0.65, 1.20),
    ATMProfile("LM-ATM-011", "NoLiTa", 0.96, 1.20, 0.65, 0.80, 1.10),
    ATMProfile("LM-ATM-012", "Canal Street", 1.20, 1.15, 0.95, 0.75, 1.15),
    ATMProfile("LM-ATM-013", "World Trade Center", 1.38, 0.95, 1.45, 0.48, 1.10),
    ATMProfile("LM-ATM-014", "Brookfield Place", 1.28, 0.90, 1.35, 0.52, 1.10),
    ATMProfile("LM-ATM-015", "Bowling Green", 1.10, 0.95, 1.10, 0.60, 1.00),
    ATMProfile("LM-ATM-016", "Staten Ferry Terminal", 1.18, 1.30, 0.70, 0.58, 1.30),
    ATMProfile("LM-ATM-017", "Essex Market", 1.02, 1.00, 0.75, 0.95, 1.00),
    ATMProfile("LM-ATM-018", "Houston Street", 1.06, 1.18, 0.82, 0.82, 1.10),
    ATMProfile("LM-ATM-019", "Delancey Street", 1.00, 1.08, 0.78, 0.92, 1.05),
    ATMProfile("LM-ATM-020", "Pier 17", 1.15, 1.45, 0.62, 0.60, 1.25),
]


MAJOR_EVENTS = {
    dt.date(2024, 1, 1): ("New Year Day", 0.70),
    dt.date(2024, 7, 4): ("Independence Day", 1.30),
    dt.date(2024, 9, 23): ("UN General Assembly", 1.15),
    dt.date(2024, 9, 24): ("UN General Assembly", 1.15),
    dt.date(2024, 9, 25): ("UN General Assembly", 1.15),
    dt.date(2024, 10, 31): ("Halloween", 1.20),
    dt.date(2024, 11, 3): ("NYC Marathon", 1.18),
    dt.date(2024, 11, 28): ("Thanksgiving", 0.78),
    dt.date(2024, 11, 29): ("Black Friday", 1.22),
    dt.date(2024, 12, 24): ("Christmas Eve", 1.05),
    dt.date(2024, 12, 25): ("Christmas", 0.68),
    dt.date(2024, 12, 31): ("New Years Eve", 1.35),
    dt.date(2025, 1, 1): ("New Year Day", 0.72),
    dt.date(2025, 7, 4): ("Independence Day", 1.28),
    dt.date(2025, 9, 22): ("UN General Assembly", 1.15),
    dt.date(2025, 9, 23): ("UN General Assembly", 1.15),
    dt.date(2025, 9, 24): ("UN General Assembly", 1.15),
    dt.date(2025, 10, 31): ("Halloween", 1.20),
    dt.date(2025, 11, 2): ("NYC Marathon", 1.18),
    dt.date(2025, 11, 27): ("Thanksgiving", 0.78),
    dt.date(2025, 11, 28): ("Black Friday", 1.22),
    dt.date(2025, 12, 24): ("Christmas Eve", 1.05),
    dt.date(2025, 12, 25): ("Christmas", 0.68),
    dt.date(2025, 12, 31): ("New Years Eve", 1.35),
}


def date_range(start: dt.date, end: dt.date):
    current = start
    while current <= end:
        yield current
        current += dt.timedelta(days=1)


def month_seasonality_factor(day: dt.date) -> float:
    factors = {
        1: 0.90,
        2: 0.92,
        3: 1.00,
        4: 1.04,
        5: 1.08,
        6: 1.12,
        7: 1.16,
        8: 1.14,
        9: 1.06,
        10: 1.08,
        11: 1.02,
        12: 1.18,
    }
    return factors[day.month]


def day_of_week_factor(day: dt.date, atm: ATMProfile) -> float:
    weekday = day.weekday()  # Monday=0
    if weekday < 5:
        office_bias = 1.00 + 0.14 * atm.office_weight
        tourist_bias = 0.98 + 0.02 * atm.tourist_weight
        return office_bias * tourist_bias
    if weekday == 5:
        return (0.90 + 0.20 * atm.tourist_weight) * (0.95 + 0.08 * atm.residential_weight)
    return (0.82 + 0.26 * atm.tourist_weight) * (0.94 + 0.10 * atm.residential_weight)


def holiday_factor(day: dt.date) -> float:
    if day.month == 12 and 18 <= day.day <= 23:
        return 1.10
    if day.month == 1 and day.day <= 3:
        return 0.92
    return 1.00


def weather_for_day(day: dt.date, rng: random.Random) -> tuple[str, float]:
    # Rough NYC monthly weather tendencies.
    monthly_temp_c = {
        1: 2.0,
        2: 3.0,
        3: 7.0,
        4: 12.0,
        5: 18.0,
        6: 23.0,
        7: 27.0,
        8: 26.0,
        9: 22.0,
        10: 16.0,
        11: 10.0,
        12: 5.0,
    }
    avg_temp = monthly_temp_c[day.month] + rng.gauss(0, 4.5)

    if avg_temp <= 0:
        profile = [("snow", 0.12), ("rain", 0.15), ("cloudy", 0.28), ("clear", 0.45)]
    elif avg_temp >= 25:
        profile = [("thunderstorm", 0.07), ("rain", 0.18), ("humid", 0.30), ("clear", 0.45)]
    else:
        profile = [("rain", 0.22), ("cloudy", 0.36), ("clear", 0.42)]

    pick = rng.random()
    cumulative = 0.0
    condition = "clear"
    for name, p in profile:
        cumulative += p
        if pick <= cumulative:
            condition = name
            break

    impact_by_condition = {
        "clear": 1.04,
        "cloudy": 0.98,
        "rain": 0.86,
        "snow": 0.76,
        "thunderstorm": 0.72,
        "humid": 0.96,
    }

    temp_discomfort = 1.0 - max(0.0, abs(avg_temp - 18.0) - 10.0) * 0.007
    weather_impact = impact_by_condition[condition] * temp_discomfort
    return condition, max(0.65, min(1.08, weather_impact))


def event_factor(day: dt.date) -> tuple[str, float]:
    if day in MAJOR_EVENTS:
        name, factor = MAJOR_EVENTS[day]
        return name, factor
    return "None", 1.0


def poisson_sample(lam: float, rng: random.Random) -> int:
    # Knuth algorithm is fine for this medium-size synthetic generation.
    lam = max(1e-6, lam)
    l_val = math.exp(-lam)
    k = 0
    p = 1.0
    while p > l_val:
        k += 1
        p *= rng.random()
    return k - 1


def generate_rows(start_date: dt.date, end_date: dt.date, seed: int = 20260419):
    rng = random.Random(seed)

    for day in date_range(start_date, end_date):
        season = month_seasonality_factor(day)
        holiday = holiday_factor(day)
        event_name, event_boost = event_factor(day)
        weather_condition, weather_impact = weather_for_day(day, rng)

        for atm in ATM_PROFILES:
            weekday_factor = day_of_week_factor(day, atm)

            annual_growth = 1.00 if day.year == 2024 else 1.04
            effective_weather = 1.0 + (weather_impact - 1.0) * atm.weather_sensitivity

            base_withdrawals = (
                110
                * atm.base_traffic
                * season
                * weekday_factor
                * holiday
                * event_boost
                * effective_weather
                * annual_growth
            )

            random_noise = max(0.75, min(1.30, rng.gauss(1.0, 0.08)))
            expected_withdrawals = max(15.0, base_withdrawals * random_noise)

            total_withdrawals = poisson_sample(expected_withdrawals, rng)

            # Offices trend to smaller/tactical withdrawals; leisure visitors skew larger amounts.
            share_100 = 0.06 + 0.12 * atm.tourist_weight - 0.03 * atm.office_weight
            share_50 = 0.24 + 0.17 * atm.tourist_weight - 0.10 * atm.office_weight
            if day.weekday() >= 5:
                share_100 += 0.01
                share_50 += 0.04
            if weather_condition in {"snow", "thunderstorm"}:
                share_100 += 0.01
                share_50 += 0.03
            if event_name in {"Black Friday", "New Years Eve", "Independence Day"}:
                share_100 += 0.03
                share_50 += 0.05
            share_100 = max(0.02, min(0.28, share_100 + rng.gauss(0, 0.01)))
            share_50 = max(0.12, min(0.62, share_50 + rng.gauss(0, 0.02)))

            hundred_count = min(total_withdrawals, max(0, int(round(total_withdrawals * share_100))))
            remaining_after_100 = max(0, total_withdrawals - hundred_count)
            fifty_count = min(
                remaining_after_100,
                max(0, int(round(remaining_after_100 * share_50))),
            )
            twenty_count = max(0, total_withdrawals - hundred_count - fifty_count)

            yield {
                "date": day.isoformat(),
                "atm_id": atm.atm_id,
                "area": atm.area,
                "weather": weather_condition,
                "major_event": event_name,
                "twenty_dollar_bills_withdrawn": twenty_count,
                "fifty_dollar_bills_withdrawn": fifty_count,
                "hundred_dollar_bills_withdrawn": hundred_count,
            }


def write_csv(path: Path, rows: list[dict]):
    fieldnames = [
        "date",
        "atm_id",
        "area",
        "weather",
        "major_event",
        "twenty_dollar_bills_withdrawn",
        "fifty_dollar_bills_withdrawn",
        "hundred_dollar_bills_withdrawn",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_iso_date(value: str) -> dt.date:
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid date '{value}'. Expected format: YYYY-MM-DD"
        ) from exc


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate synthetic daily ATM withdrawal CSVs for Lower Manhattan."
    )
    parser.add_argument(
        "--start-date",
        type=parse_iso_date,
        default=dt.date(2024, 1, 1),
        help="Start date in YYYY-MM-DD format (default: 2024-01-01).",
    )
    parser.add_argument(
        "--end-date",
        type=parse_iso_date,
        default=dt.date(2025, 12, 31),
        help="End date in YYYY-MM-DD format (default: 2025-12-31).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260419,
        help="Random seed for reproducibility (default: 20260419).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory to write output CSV files (default: script directory).",
    )
    parser.add_argument(
        "--combined-file",
        default=None,
        help=(
            "Filename for combined CSV output "
            "(default: atm_withdrawals_<startYear>_<endYear>.csv)."
        ),
    )
    parser.add_argument(
        "--skip-yearly-files",
        action="store_true",
        help="Only generate the combined file and skip per-year CSV outputs.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.end_date < args.start_date:
        raise ValueError("--end-date must be on or after --start-date")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_file = (
        args.combined_file
        if args.combined_file
        else f"atm_withdrawals_{args.start_date.year}_{args.end_date.year}.csv"
    )

    all_rows = list(generate_rows(args.start_date, args.end_date, seed=args.seed))
    rows_by_year: dict[str, list[dict]] = {}
    for row in all_rows:
        year = row["date"][:4]
        rows_by_year.setdefault(year, []).append(row)

    if not args.skip_yearly_files:
        for year in sorted(rows_by_year):
            yearly_name = f"atm_withdrawals_{year}.csv"
            write_csv(output_dir / yearly_name, rows_by_year[year])
            print(f"Generated {len(rows_by_year[year]):,} rows in {yearly_name}")

    write_csv(output_dir / combined_file, all_rows)
    print(f"Generated {len(all_rows):,} rows in {combined_file}")


if __name__ == "__main__":
    main()
