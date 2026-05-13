#!/usr/bin/env python3
"""Streamlit UI for ATM bill rolling-window analytics."""

from __future__ import annotations

from pathlib import Path
from statistics import NormalDist

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import qualitative
import streamlit as st

st.set_page_config(
    page_title="ATM Rolling Window Analytics",
    page_icon="chart_with_upwards_trend",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Map constants
# ---------------------------------------------------------------------------

MAP_BILL_COLS = [
    "twenty_dollar_bills_withdrawn",
    "fifty_dollar_bills_withdrawn",
    "hundred_dollar_bills_withdrawn",
]
MAP_BILL_LABELS = {
    "twenty_dollar_bills_withdrawn": "$20 Bills",
    "fifty_dollar_bills_withdrawn": "$50 Bills",
    "hundred_dollar_bills_withdrawn": "$100 Bills",
}
MAP_DENOM_COLORS = {
    "twenty_dollar_bills_withdrawn": "#3B82F6",
    "fifty_dollar_bills_withdrawn": "#10B981",
    "hundred_dollar_bills_withdrawn": "#F59E0B",
}


def inject_style() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2rem;
        }
        .app-title {
            font-size: 2.0rem;
            font-weight: 700;
            letter-spacing: 0.2px;
            color: #0F172A;
            margin-bottom: 0.2rem;
        }
        .app-subtitle {
            color: #334155;
            margin-top: 0;
            margin-bottom: 1.1rem;
        }
        .info-card {
            border: 1px solid #D9E3F0;
            background: linear-gradient(130deg, #FFFFFF, #F8FBFF);
            border-radius: 12px;
            padding: 14px 16px;
            margin-bottom: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def list_csv_files(search_root: Path) -> list[Path]:
    return sorted(search_root.glob("*.csv"))


# ---------------------------------------------------------------------------
# Map helpers
# ---------------------------------------------------------------------------


def _has_geo_columns(path: Path) -> bool:
    with path.open("r", encoding="utf-8") as fh:
        header = fh.readline()
    return "latitude" in header and "longitude" in header


def find_geo_csv_files(root: Path) -> list[Path]:
    return [p for p in sorted(root.glob("atm_withdrawals_*.csv")) if _has_geo_columns(p)]


@st.cache_data
def load_geo_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"])


def build_atm_summary(df: pd.DataFrame, bill_cols: list[str]) -> pd.DataFrame:
    agg_dict: dict[str, str] = {col: "sum" for col in bill_cols}
    agg_dict.update({"latitude": "first", "longitude": "first", "area": "first"})
    summary = df.groupby("atm_id", as_index=False).agg(agg_dict)
    summary["total_withdrawals"] = summary[bill_cols].sum(axis=1)
    return summary


def build_map_figure(
    summary: pd.DataFrame,
    size_metric: str,
    map_style: str,
    max_bubble: int,
) -> go.Figure:
    # Normalize sizes linearly from min_px to max_bubble so differences are visible
    # regardless of how tight the value range is.
    vals = summary[size_metric].values.astype(float)
    min_val, max_val = vals.min(), vals.max()
    min_px = max(8, max_bubble * 0.2)
    if max_val > min_val:
        sizes = min_px + (vals - min_val) / (max_val - min_val) * (max_bubble - min_px)
    else:
        sizes = [max_bubble] * len(vals)

    label_map = {
        "total_withdrawals": "Total Withdrawals",
        "twenty_dollar_bills_withdrawn": "$20 Bills",
        "fifty_dollar_bills_withdrawn": "$50 Bills",
        "hundred_dollar_bills_withdrawn": "$100 Bills",
    }
    metric_label = label_map.get(size_metric, size_metric)

    # Build hover text manually
    hover_lines = []
    for _, row in summary.iterrows():
        lines = [
            f"<b>{row['atm_id']}</b>",
            f"Area: {row['area']}",
            f"Lat: {row['latitude']:.4f}  Lon: {row['longitude']:.4f}",
            f"{metric_label}: {int(row[size_metric]):,}",
        ]
        for col in MAP_BILL_COLS:
            if col in summary.columns and col != size_metric:
                lines.append(f"{label_map.get(col, col)}: {int(row[col]):,}")
        hover_lines.append("<br>".join(lines))

    fig = go.Figure(
        go.Scattermapbox(
            lat=summary["latitude"],
            lon=summary["longitude"],
            mode="markers",
            marker=go.scattermapbox.Marker(
                size=list(sizes),
                sizemode="diameter",
                color=vals,
                colorscale="Viridis",
                showscale=True,
                colorbar={"title": metric_label, "thickness": 14},
                opacity=0.82,
            ),
            text=summary["atm_id"],
            customdata=summary[["atm_id"]].values,
            hovertemplate="%{hovertext}<extra></extra>",
            hovertext=hover_lines,
        )
    )
    fig.update_layout(
        height=560,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        mapbox={
            "style": map_style,
            "center": {"lat": 40.715, "lon": -74.006},
            "zoom": 13,
        },
    )
    return fig


def build_map_timeseries_figure(daily: pd.DataFrame) -> go.Figure:
    fig = px.line(
        daily,
        x="date",
        y="total",
        color="atm_id",
        title="Daily Total Withdrawals — Selected ATMs",
        labels={"total": "Bills Withdrawn", "date": "Date", "atm_id": "ATM"},
        template="plotly_white",
    )
    fig.update_layout(
        height=360,
        margin={"l": 10, "r": 10, "t": 45, "b": 10},
        legend={"orientation": "h", "yanchor": "top", "y": -0.18, "x": 0},
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(15,23,42,0.08)")
    return fig


def build_map_denomination_figure(
    daily: pd.DataFrame,
    selected_bills: list[str],
    selected_ids: list[str],
) -> go.Figure:
    melted = daily.melt(
        id_vars=["date", "atm_id"],
        value_vars=selected_bills,
        var_name="denomination",
        value_name="count",
    )
    fig = px.bar(
        melted,
        x="date",
        y="count",
        color="denomination",
        facet_col="atm_id" if len(selected_ids) > 1 else None,
        title="Denomination Breakdown — Selected ATMs",
        labels={"count": "Bills Withdrawn", "date": "Date", "denomination": "Bill"},
        template="plotly_white",
        color_discrete_map=MAP_DENOM_COLORS,
        barmode="stack",
    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_layout(height=340, margin={"l": 10, "r": 10, "t": 45, "b": 10})
    fig.update_xaxes(showgrid=False)
    return fig


def build_map_rolling_variance_figure(rolling_daily: pd.DataFrame, window_days: int) -> go.Figure:
    fig = px.line(
        rolling_daily,
        x="date",
        y="rolling_variance",
        color="atm_id",
        title=f"Rolling Variance of Daily Total Withdrawals ({window_days}-day window)",
        labels={"rolling_variance": "Rolling Variance", "date": "Date", "atm_id": "ATM"},
        template="plotly_white",
    )
    fig.update_layout(
        height=300,
        margin={"l": 10, "r": 10, "t": 45, "b": 10},
        legend={"orientation": "h", "yanchor": "top", "y": -0.2, "x": 0},
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(15,23,42,0.08)")
    return fig


def infer_date_column(columns: list[str]) -> str:
    preferred = ["date", "day", "timestamp", "datetime", "ds"]
    lowered = {c.lower(): c for c in columns}
    for key in preferred:
        if key in lowered:
            return lowered[key]
    return columns[0]


def infer_numeric_columns(df: pd.DataFrame, excluded: set[str]) -> list[str]:
    numeric = []
    for col in df.columns:
        if col in excluded:
            continue
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().any():
            numeric.append(col)
    return numeric


def infer_layout(df: pd.DataFrame, date_col: str) -> str:
    cols = list(df.columns)
    lower_map = {c.lower(): c for c in cols}
    non_date_cols = [c for c in cols if c != date_col]

    if any(c.endswith("_dollar_bills_withdrawn") for c in non_date_cols):
        return "Wide"

    if "denomination" in lower_map and "bills_withdrawn" in lower_map:
        return "Long"

    numeric_cols = infer_numeric_columns(df, excluded={date_col})
    if len(numeric_cols) >= 2:
        return "Wide"
    if len(numeric_cols) == 1 and len(non_date_cols) >= 2:
        return "Long"

    return "Wide"


def choose_long_defaults(df: pd.DataFrame, date_col: str) -> tuple[str, str]:
    cols = list(df.columns)
    non_date_cols = [c for c in cols if c != date_col]
    lowered = {c.lower(): c for c in non_date_cols}

    label_priority = ["denomination", "series", "label", "category", "bill_type", "type"]
    value_priority = ["bills_withdrawn", "value", "count", "amount", "withdrawals"]

    denom_col = None
    for key in label_priority:
        if key in lowered:
            denom_col = lowered[key]
            break

    numeric_candidates = infer_numeric_columns(df, excluded={date_col})

    value_col = None
    for key in value_priority:
        if key in lowered:
            value_col = lowered[key]
            break
    if value_col is None and numeric_candidates:
        value_col = numeric_candidates[0]

    if denom_col is None:
        non_numeric = [c for c in non_date_cols if c not in numeric_candidates]
        if non_numeric:
            denom_col = non_numeric[0]
        elif non_date_cols:
            denom_col = non_date_cols[0]

    if value_col is None and non_date_cols:
        value_col = non_date_cols[0]

    if denom_col == value_col:
        alternatives = [c for c in non_date_cols if c != denom_col]
        if alternatives:
            if denom_col in numeric_candidates:
                denom_col = alternatives[0]
            else:
                value_col = alternatives[0]

    return denom_col, value_col


def prepare_daily_wide(
    df: pd.DataFrame,
    date_col: str,
    data_shape: str,
    selected_series_cols: list[str],
    long_denom_col: str,
    long_value_col: str,
) -> tuple[pd.DataFrame, list[str]]:
    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work = work.dropna(subset=[date_col])

    use_long = data_shape == "Long"
    if data_shape == "Auto-detect":
        use_long = infer_layout(work, date_col) == "Long"

    if use_long:
        if len({date_col, long_denom_col, long_value_col}) < 3:
            raise ValueError(
                "For Long format, date, series label, and value columns must be different. "
                f"Current selection: date={date_col}, label={long_denom_col}, value={long_value_col}."
            )

        long_df = work.loc[:, [date_col, long_denom_col, long_value_col]].copy()
        long_df[long_denom_col] = long_df[long_denom_col].astype(str).str.strip()
        long_df[long_value_col] = pd.to_numeric(long_df[long_value_col], errors="coerce")
        long_df = long_df.dropna(subset=[long_value_col])

        daily = (
            long_df.pivot_table(
                index=date_col,
                columns=long_denom_col,
                values=long_value_col,
                aggfunc="sum",
                fill_value=0,
            )
            .reset_index()
            .rename_axis(None, axis=1)
            .sort_values(date_col)
        )
        series_cols = [c for c in daily.columns if c != date_col]
    else:
        if not selected_series_cols:
            raise ValueError("Select at least one numeric series column for wide-format data.")

        for col in selected_series_cols:
            work[col] = pd.to_numeric(work[col], errors="coerce")

        daily = (
            work.groupby(date_col, as_index=False)[selected_series_cols]
            .sum(min_count=1)
            .fillna(0)
            .sort_values(date_col)
        )
        series_cols = selected_series_cols

    daily = daily.rename(columns={date_col: "date"})
    return daily, series_cols


def compute_rolling(df: pd.DataFrame, column: str, window_days: int) -> pd.DataFrame:
    out = df[["date", column]].copy()
    out = out.rename(columns={column: "value"})
    out["moving_average"] = out["value"].rolling(window=window_days, min_periods=window_days).mean()
    out["moving_variance"] = out["value"].rolling(window=window_days, min_periods=window_days).var(ddof=0)
    out = out.dropna().copy()
    return out


def build_rolling_figure(
    rolling_df: pd.DataFrame,
    series_label: str,
    band_mode: str,
    window_days: int,
    confidence_pct: float,
    bollinger_k: float,
) -> go.Figure:
    fig = go.Figure()

    if band_mode == "stddev":
        spread = rolling_df["moving_variance"].pow(0.5)
        band_label = "Std Dev Band"
    elif band_mode == "bollinger":
        spread = bollinger_k * rolling_df["moving_variance"].pow(0.5)
        band_label = f"Bollinger Band (k={bollinger_k:.2f})"
    elif band_mode == "confidence_interval":
        z_score = NormalDist().inv_cdf(0.5 + (confidence_pct / 100.0) / 2.0)
        spread = z_score * rolling_df["moving_variance"].pow(0.5) / (window_days**0.5)
        band_label = f"{confidence_pct:.1f}% CI Band"
    else:
        spread = rolling_df["moving_variance"]
        band_label = "Variance Band"

    lower = (rolling_df["moving_average"] - spread).clip(lower=0)
    upper = rolling_df["moving_average"] + spread

    fig.add_trace(
        go.Scatter(
            x=rolling_df["date"],
            y=rolling_df["moving_average"],
            mode="lines",
            name=f"{series_label} Moving Average",
            line={"color": "#0C5EA8", "width": 2.3},
            customdata=rolling_df[["moving_variance"]].values,
            hovertemplate=(
                "Date=%{x|%Y-%m-%d}<br>"
                "Moving Average=%{y:.2f}<br>"
                "Moving Variance=%{customdata[0]:.2f}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=rolling_df["date"],
            y=upper,
            mode="lines",
            line={"color": "rgba(12, 94, 168, 0.0)", "width": 0},
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=rolling_df["date"],
            y=lower,
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(12, 94, 168, 0.20)",
            line={"color": "rgba(12, 94, 168, 0.0)", "width": 0},
            name=band_label,
            hovertemplate="Date=%{x|%Y-%m-%d}<br>Lower=%{y:.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        height=420,
        margin={"l": 10, "r": 10, "t": 50, "b": 80},
        template="plotly_white",
        title=f"{series_label} Rolling Window Trend",
        xaxis_title="Date",
        yaxis_title="Bills Withdrawn",
        legend={"orientation": "h", "yanchor": "top", "y": -0.2, "x": 0},
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(15, 23, 42, 0.08)")

    return fig


def build_variance_figure(rolling_df: pd.DataFrame, series_label: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=rolling_df["date"],
            y=rolling_df["moving_variance"],
            mode="lines",
            line={"color": "#D97706", "width": 2.0},
            name="Moving Variance",
            hovertemplate="Date=%{x|%Y-%m-%d}<br>Variance=%{y:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=300,
        margin={"l": 10, "r": 10, "t": 35, "b": 10},
        template="plotly_white",
        title=f"{series_label} Rolling Variance",
        xaxis_title="Date",
        yaxis_title="Variance",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(15, 23, 42, 0.08)")
    return fig


def apply_row_filters(
    df: pd.DataFrame,
    selected_filter_cols: list[str],
    selected_filter_values: dict[str, list[object]],
) -> pd.DataFrame:
    filtered = df.copy()
    for col in selected_filter_cols:
        values = selected_filter_values.get(col, [])
        if values:
            filtered = filtered[filtered[col].isin(values)]
    return filtered


def get_filter_options(df: pd.DataFrame, col: str) -> list[object]:
    series = df[col].dropna()
    if series.empty:
        return []

    try:
        numeric = pd.to_numeric(series, errors="raise")
        return sorted(numeric.unique().tolist())
    except Exception:
        pass

    try:
        return sorted(series.unique().tolist())
    except TypeError:
        # Fallback for mixed non-comparable types.
        return sorted(series.unique().tolist(), key=lambda x: str(x))


def build_combined_rolling_figure(
    rolling_by_series: dict[str, pd.DataFrame],
    band_mode: str,
    show_bands: bool,
    window_days: int,
    confidence_pct: float,
    bollinger_k: float,
) -> go.Figure:
    fig = go.Figure()
    palette = qualitative.Safe + qualitative.Set2 + qualitative.Plotly

    for idx, (series_name, rolling_df) in enumerate(rolling_by_series.items()):
        color = palette[idx % len(palette)]
        fig.add_trace(
            go.Scatter(
                x=rolling_df["date"],
                y=rolling_df["moving_average"],
                mode="lines",
                name=f"{series_name} Moving Average",
                line={"color": color, "width": 2.2},
                customdata=rolling_df[["moving_variance"]].values,
                hovertemplate=(
                    "Series=" + series_name + "<br>"
                    "Date=%{x|%Y-%m-%d}<br>"
                    "Moving Average=%{y:.2f}<br>"
                    "Moving Variance=%{customdata[0]:.2f}<extra></extra>"
                ),
            )
        )

        if show_bands:
            if band_mode == "stddev":
                spread = rolling_df["moving_variance"].pow(0.5)
            elif band_mode == "bollinger":
                spread = bollinger_k * rolling_df["moving_variance"].pow(0.5)
            elif band_mode == "confidence_interval":
                z_score = NormalDist().inv_cdf(0.5 + (confidence_pct / 100.0) / 2.0)
                spread = z_score * rolling_df["moving_variance"].pow(0.5) / (window_days**0.5)
            else:
                spread = rolling_df["moving_variance"]

            lower = (rolling_df["moving_average"] - spread).clip(lower=0)
            upper = rolling_df["moving_average"] + spread

            fig.add_trace(
                go.Scatter(
                    x=rolling_df["date"],
                    y=upper,
                    mode="lines",
                    line={"color": color, "width": 0},
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=rolling_df["date"],
                    y=lower,
                    mode="lines",
                    fill="tonexty",
                    fillcolor="rgba(12, 94, 168, 0.14)",
                    line={"color": color, "width": 0},
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    fig.update_layout(
        height=460,
        margin={"l": 10, "r": 10, "t": 110, "b": 90},
        template="plotly_white",
        title={
            "text": "Rolling Window Trend (Combined Series)",
            "x": 0,
            "xanchor": "left",
            "y": 0.98,
            "yanchor": "top",
            "pad": {"b": 20},
        },
        xaxis_title="Date",
        yaxis_title="Value",
        legend={"orientation": "h", "yanchor": "top", "y": -0.2, "x": 0},
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(15, 23, 42, 0.08)")
    return fig


def resolve_confidence_pct(slider_value: float, text_value: str) -> float:
    text = text_value.strip()
    if not text:
        return slider_value
    try:
        value = float(text)
    except ValueError as exc:
        raise ValueError("Confidence interval text input must be a number (e.g., 95).") from exc
    if not (50.0 <= value < 100.0):
        raise ValueError("Confidence interval must be between 50 and 100 (exclusive).")
    return value


def resolve_bollinger_k(slider_value: float, text_value: str) -> float:
    text = text_value.strip()
    if not text:
        return slider_value
    try:
        value = float(text)
    except ValueError as exc:
        raise ValueError("Bollinger k text input must be a number (e.g., 2.0).") from exc
    if not (0.1 <= value <= 5.0):
        raise ValueError("Bollinger k must be between 0.1 and 5.0.")
    return value


def sync_confidence_from_slider() -> None:
    st.session_state["confidence_pct_text"] = f"{st.session_state['confidence_pct_slider']:.1f}"


def sync_confidence_from_text() -> None:
    text = st.session_state.get("confidence_pct_text", "").strip()
    if not text:
        return
    try:
        value = float(text)
    except ValueError:
        return
    if 50.0 <= value < 100.0:
        st.session_state["confidence_pct_slider"] = value


def sync_bollinger_from_slider() -> None:
    st.session_state["bollinger_k_text"] = f"{st.session_state['bollinger_k_slider']:.2f}"


def sync_bollinger_from_text() -> None:
    text = st.session_state.get("bollinger_k_text", "").strip()
    if not text:
        return
    try:
        value = float(text)
    except ValueError:
        return
    if 0.1 <= value <= 5.0:
        st.session_state["bollinger_k_slider"] = value


def sync_window_from_slider() -> None:
    st.session_state["window_days_text"] = str(st.session_state["window_days_slider"])


def sync_window_from_text() -> None:
    text = st.session_state.get("window_days_text", "").strip()
    if not text:
        return
    try:
        value = int(text)
    except ValueError:
        return
    if 2 <= value <= 60:
        st.session_state["window_days_slider"] = value


def resolve_window_days(slider_value: int, text_value: str) -> int:
    text = text_value.strip()
    if not text:
        return slider_value
    try:
        value = int(text)
    except ValueError as exc:
        raise ValueError("Rolling window text input must be an integer (e.g., 7, 14, 21).") from exc
    if not (2 <= value <= 60):
        raise ValueError("Rolling window length must be between 2 and 60 days.")
    return value


def render_map_tab(project_root: Path) -> None:
    """Render the ATM Map tab content."""
    geo_files = find_geo_csv_files(project_root)
    if not geo_files:
        st.error(
            "No geo-enabled ATM CSVs found. "
            "Re-run `generate_atm_withdrawal_data.py` to produce files with latitude/longitude."
        )
        return

    with st.sidebar:
        st.divider()
        st.subheader("Map Controls")

        csv_options = {p.name: p for p in geo_files}
        map_selected_name = st.selectbox("Map dataset", list(csv_options), key="map_dataset")
        map_df = load_geo_csv(csv_options[map_selected_name])

        min_date = map_df["date"].min().date()
        max_date = map_df["date"].max().date()
        date_range = st.date_input(
            "Date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key="map_date_range",
        )
        if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
            start_d, end_d = date_range
            map_df = map_df[
                (map_df["date"].dt.date >= start_d) & (map_df["date"].dt.date <= end_d)
            ]

        map_selected_bills = st.multiselect(
            "Include denominations",
            options=MAP_BILL_COLS,
            default=MAP_BILL_COLS,
            format_func=lambda x: MAP_BILL_LABELS[x],
            key="map_bills",
        )
        if not map_selected_bills:
            st.warning("Select at least one denomination.")
            return

        # Build slider bounds from ATM totals over the selected date range.
        map_df = map_df.copy()
        map_df["daily_total_withdrawals"] = map_df[map_selected_bills].sum(axis=1)

        deposit_cols = [
            c
            for c in map_df.columns
            if ("deposit" in c.lower()) and pd.api.types.is_numeric_dtype(map_df[c])
        ]
        has_deposit = bool(deposit_cols)
        if has_deposit:
            map_df["daily_total_deposits"] = map_df[deposit_cols].sum(axis=1)

        slider_summary = build_atm_summary(map_df, map_selected_bills)
        if has_deposit:
            slider_summary = slider_summary.merge(
                map_df.groupby("atm_id", as_index=False)["daily_total_deposits"].sum(),
                on="atm_id",
                how="left",
            ).rename(columns={"daily_total_deposits": "total_deposits"})

        withdraw_min = int(slider_summary["total_withdrawals"].min())
        withdraw_max = int(slider_summary["total_withdrawals"].max())
        withdraw_range = st.slider(
            "Total withdrawals (selected date range)",
            min_value=withdraw_min,
            max_value=withdraw_max,
            value=(withdraw_min, withdraw_max),
            key="map_daily_withdraw_range",
        )

        deposit_range: tuple[int, int] | None = None
        if has_deposit:
            deposit_min = int(slider_summary["total_deposits"].min())
            deposit_max = int(slider_summary["total_deposits"].max())
            deposit_range = st.slider(
                "Total deposits (selected date range)",
                min_value=deposit_min,
                max_value=deposit_max,
                value=(deposit_min, deposit_max),
                key="map_daily_deposit_range",
            )
        else:
            st.caption("No deposit columns found in this dataset.")

        map_style = st.selectbox(
            "Map style",
            options=["carto-positron", "open-street-map", "carto-darkmatter"],
            index=0,
            key="map_style",
        )
        size_metric = st.selectbox(
            "Bubble size & colour",
            options=["total_withdrawals"] + map_selected_bills,
            format_func=lambda x: (
                "Total Withdrawals" if x == "total_withdrawals" else MAP_BILL_LABELS[x]
            ),
            key="map_size_metric",
        )
        max_bubble = st.slider("Max bubble size", min_value=15, max_value=70, value=38, key="map_bubble")
        map_window_days = st.slider(
            "Map rolling variance window (days)",
            min_value=2,
            max_value=60,
            value=14,
            key="map_window_days",
        )
        pin_map_on_scroll = st.checkbox(
            "Keep map visible while scrolling",
            value=True,
            key="map_pin_on_scroll",
            help="Experimental: pins the map panel as you scroll the right-side charts.",
        )

    # Apply slider filters to aggregated ATM totals.
    w_low, w_high = withdraw_range
    summary = build_atm_summary(map_df, map_selected_bills)
    if has_deposit:
        summary = summary.merge(
            map_df.groupby("atm_id", as_index=False)["daily_total_deposits"].sum(),
            on="atm_id",
            how="left",
        ).rename(columns={"daily_total_deposits": "total_deposits"})

    summary = summary[
        (summary["total_withdrawals"] >= w_low)
        & (summary["total_withdrawals"] <= w_high)
    ]
    if deposit_range is not None and "total_deposits" in summary.columns:
        d_low, d_high = deposit_range
        summary = summary[
            (summary["total_deposits"] >= d_low)
            & (summary["total_deposits"] <= d_high)
        ]

    if summary.empty:
        st.warning("No ATMs match the current slider filters. Widen the ranges to continue.")
        return

    # Restrict downstream ATM-day details to the ATMs that remain on the map.
    map_df = map_df[map_df["atm_id"].isin(summary["atm_id"])]
    if map_df.empty:
        st.warning("No ATM-day rows remain for the filtered ATM set.")
        return

    if pin_map_on_scroll:
        st.markdown(
            """
            <style>
            .st-key-atm_map_chart {
                position: sticky;
                top: 4.8rem;
                z-index: 10;
                background: #FFFFFF;
                padding-bottom: 0.4rem;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ATMs on Map", len(summary))
    c2.metric("Days in View", f"{map_df['date'].nunique():,}")
    c3.metric("Total Withdrawals", f"{int(summary['total_withdrawals'].sum()):,}")
    busiest = summary.loc[summary["total_withdrawals"].idxmax()]
    c4.metric("Busiest ATM", f"{busiest['atm_id']} ({busiest['area']})")

    col_left, col_right = st.columns([1.25, 1.0], gap="large")

    with col_left:
        map_fig = build_map_figure(summary, size_metric, map_style, max_bubble)
        event = st.plotly_chart(
            map_fig,
            use_container_width=True,
            on_select="rerun",
            selection_mode=["points"],
            key="atm_map_chart",
        )

    selected_ids: list[str] = []
    if event and hasattr(event, "selection") and event.selection.points:
        for pt in event.selection.points:
            cd = pt.get("customdata")
            if cd:
                selected_ids.append(cd[0])
            else:
                idx = pt.get("point_index")
                if idx is not None and 0 <= idx < len(summary):
                    selected_ids.append(summary.iloc[idx]["atm_id"])
    selected_ids = list(dict.fromkeys(selected_ids))

    with col_left:
        with st.expander("Map Selection Snapshot", expanded=True):
            if selected_ids:
                snapshot = (
                    summary[summary["atm_id"].isin(selected_ids)]
                    .sort_values("total_withdrawals", ascending=False)
                    .rename(
                        columns={
                            "atm_id": "ATM ID",
                            "area": "Area",
                            "total_withdrawals": "Total",
                            **MAP_BILL_LABELS,
                        }
                    )
                )
                view_cols = ["ATM ID", "Area", "Total"] + [MAP_BILL_LABELS[c] for c in map_selected_bills]
                st.dataframe(snapshot[view_cols], use_container_width=True, hide_index=True, height=220)
            else:
                top_atms = (
                    summary.sort_values("total_withdrawals", ascending=False)
                    .head(8)
                    .rename(columns={"atm_id": "ATM ID", "area": "Area", "total_withdrawals": "Total"})
                )
                st.dataframe(top_atms[["ATM ID", "Area", "Total"]], use_container_width=True, hide_index=True)

    with col_right:
        st.subheader("ATM Time Series")
        if selected_ids:
            st.caption(f"Selected ATM{'s' if len(selected_ids) > 1 else ''}: {', '.join(selected_ids)}")
            sel_data = map_df[map_df["atm_id"].isin(selected_ids)].copy()

            daily = (
                sel_data.groupby(["date", "atm_id"])[map_selected_bills]
                .sum()
                .reset_index()
            )
            daily["total"] = daily[map_selected_bills].sum(axis=1)

            st.plotly_chart(build_map_timeseries_figure(daily), use_container_width=True)
            if len(map_selected_bills) > 1:
                st.plotly_chart(
                    build_map_denomination_figure(daily, map_selected_bills, selected_ids),
                    use_container_width=True,
                )

            rolling_daily = daily[["date", "atm_id", "total"]].sort_values(["atm_id", "date"]).copy()
            rolling_daily["rolling_variance"] = rolling_daily.groupby("atm_id")["total"].transform(
                lambda s: s.rolling(window=map_window_days, min_periods=map_window_days).var(ddof=0)
            )
            rolling_daily = rolling_daily.dropna(subset=["rolling_variance"])

            if not rolling_daily.empty:
                st.plotly_chart(
                    build_map_rolling_variance_figure(rolling_daily, map_window_days),
                    use_container_width=True,
                )
            else:
                st.info(
                    f"Not enough daily data points for a {map_window_days}-day rolling variance window."
                )
        else:
            st.info("Click one or more ATM bubbles on the map to view time-series details here.")

    with st.expander("All ATMs — aggregated summary", expanded=False):
        disp = summary.sort_values("total_withdrawals", ascending=False).copy()
        disp = disp.rename(
            columns={
                "atm_id": "ATM ID", "area": "Area",
                "latitude": "Latitude", "longitude": "Longitude",
                "total_withdrawals": "Total Withdrawals",
                **MAP_BILL_LABELS,
            }
        )
        st.dataframe(disp, use_container_width=True, hide_index=True)


def main() -> None:
    inject_style()

    if "confidence_pct_slider" not in st.session_state:
        st.session_state["confidence_pct_slider"] = 95.0
    if "confidence_pct_text" not in st.session_state:
        st.session_state["confidence_pct_text"] = ""
    if "window_days_slider" not in st.session_state:
        st.session_state["window_days_slider"] = 7
    if "window_days_text" not in st.session_state:
        st.session_state["window_days_text"] = ""
    if "bollinger_k_slider" not in st.session_state:
        st.session_state["bollinger_k_slider"] = 2.0
    if "bollinger_k_text" not in st.session_state:
        st.session_state["bollinger_k_text"] = ""

    st.markdown('<div class="app-title">ATM Rolling Window Analytics</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="app-subtitle">Explore rolling window trends or switch to the Map tab to see ATM locations.</p>',
        unsafe_allow_html=True,
    )

    project_root = Path(__file__).resolve().parent.parent
    csv_files = list_csv_files(project_root)

    tab_analytics, tab_map = st.tabs(["Rolling Analytics", "ATM Map"])

    with tab_analytics:
        with st.sidebar:
            st.header("Controls")
            selected_path: Path | None = None
            if csv_files:
                csv_options = {p.name: p for p in csv_files}
                selected_name = st.selectbox("Choose input CSV", options=list(csv_options.keys()))
                selected_path = csv_options[selected_name]
            else:
                st.warning("No CSV files found in the project root.")
            uploaded = st.file_uploader("Upload CSV", type=["csv"])

            window_days_slider = st.slider(
                "Rolling window length (days)",
                min_value=2,
                max_value=60,
                key="window_days_slider",
                on_change=sync_window_from_slider,
            )
            window_days_text = st.text_input(
                "Or enter rolling window length",
                key="window_days_text",
                on_change=sync_window_from_text,
                help="Optional manual override between 2 and 60 days.",
            )
            band_mode = st.radio(
                "Band type",
                options=["stddev", "variance", "confidence_interval", "bollinger"],
                index=0,
                horizontal=True,
                format_func=lambda x: {
                    "stddev": "Std Dev",
                    "variance": "Variance",
                    "confidence_interval": "Confidence Interval",
                    "bollinger": "Bollinger",
                }[x],
            )
            confidence_pct_slider = st.slider(
                "Confidence interval (%)",
                min_value=50.0,
                max_value=99.9,
                step=0.1,
                key="confidence_pct_slider",
                on_change=sync_confidence_from_slider,
                help="Used when band type is confidence_interval.",
            )
            confidence_pct_text = st.text_input(
                "Or enter confidence interval (%)",
                key="confidence_pct_text",
                on_change=sync_confidence_from_text,
                help="Optional manual override (e.g., 90, 95, 99).",
            )
            bollinger_k_slider = st.slider(
                "Bollinger k",
                min_value=0.1,
                max_value=5.0,
                step=0.1,
                key="bollinger_k_slider",
                on_change=sync_bollinger_from_slider,
                help="Used when band type is bollinger. Common default is 2.0.",
            )
            bollinger_k_text = st.text_input(
                "Or enter Bollinger k",
                key="bollinger_k_text",
                on_change=sync_bollinger_from_text,
                help="Optional manual override (e.g., 1.5, 2.0, 2.5).",
            )
            show_variance_panel = st.checkbox("Show explicit variance panel", value=True)

        if uploaded is not None:
            source_label = f"Uploaded file: {uploaded.name}"
            raw_df = pd.read_csv(uploaded)
        elif selected_path is not None:
            source_label = f"Selected file: {selected_path.name}"
            raw_df = pd.read_csv(selected_path)
        else:
            st.info("Add or select a CSV to continue.")
            raw_df = None

        if raw_df is not None:
            source_label = (
                f"Uploaded file: {uploaded.name}"
                if uploaded is not None
                else f"Selected file: {selected_path.name}"  # type: ignore[union-attr]
            )
            st.markdown(f'<div class="info-card">{source_label}</div>', unsafe_allow_html=True)

            try:
                window_days = resolve_window_days(window_days_slider, window_days_text)
                confidence_pct = resolve_confidence_pct(confidence_pct_slider, confidence_pct_text)
                bollinger_k = resolve_bollinger_k(bollinger_k_slider, bollinger_k_text)
            except ValueError as exc:
                st.error(str(exc))
                raw_df = None

        if raw_df is not None:
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Rows", f"{len(raw_df):,}")
            col_b.metric("Columns", f"{len(raw_df.columns):,}")
            col_c.metric("Window Days", f"{window_days}")

            with st.expander("Preview selected CSV", expanded=True):
                st.dataframe(raw_df.head(200), use_container_width=True)

            with st.sidebar:
                st.subheader("Data Filters")
                filterable_cols = list(raw_df.columns)
                selected_filter_cols = st.multiselect(
                    "Columns to filter",
                    options=filterable_cols,
                    default=[],
                    help="Choose one or more columns to filter the dataset before time-series prep.",
                )
                selected_filter_values: dict[str, list[object]] = {}
                for col in selected_filter_cols:
                    distinct_values = get_filter_options(raw_df, col)
                    selected_filter_values[col] = st.multiselect(
                        f"Values for {col}",
                        options=distinct_values,
                        default=distinct_values,
                    )

            filtered_df = apply_row_filters(raw_df, selected_filter_cols, selected_filter_values)
            st.caption(f"Rows after filters: {len(filtered_df):,} / {len(raw_df):,}")
            if filtered_df.empty:
                st.warning("No rows left after applying filters. Adjust filter selections.")
            else:
                with st.sidebar:
                    st.subheader("Column Mapping")
                    all_cols = list(raw_df.columns)
                    date_col_default = infer_date_column(all_cols)
                    date_col = st.selectbox(
                        "Date column",
                        options=all_cols,
                        index=all_cols.index(date_col_default),
                    )

                    data_shape = st.radio(
                        "CSV layout",
                        options=["Auto-detect", "Wide", "Long"],
                        index=0,
                        horizontal=True,
                        help="Auto-detect infers layout from columns. Use Wide or Long to override.",
                    )

                    detected_layout = infer_layout(filtered_df, date_col)
                    effective_layout = detected_layout if data_shape == "Auto-detect" else data_shape
                    if data_shape == "Auto-detect":
                        st.caption(f"Detected layout: {detected_layout}")

                    long_denom_default, long_value_default = choose_long_defaults(filtered_df, date_col)
                    long_denom_col = long_denom_default
                    long_value_col = long_value_default
                    if effective_layout == "Long":
                        long_denom_col = st.selectbox(
                            "Long format series label column",
                            options=all_cols,
                            index=all_cols.index(long_denom_default),
                        )
                        long_value_col = st.selectbox(
                            "Long format value column",
                            options=all_cols,
                            index=all_cols.index(long_value_default),
                        )
                        if len({date_col, long_denom_col, long_value_col}) < 3:
                            st.warning(
                                "Long format needs three different columns: date, series label, and value."
                            )

                    candidate_numeric = infer_numeric_columns(filtered_df, excluded={date_col})
                    denom_defaults = [
                        c for c in candidate_numeric if c.endswith("_dollar_bills_withdrawn")
                    ]
                    preferred_defaults = denom_defaults if denom_defaults else [
                        c
                        for c in [
                            "twenty_dollar_bills_withdrawn",
                            "fifty_dollar_bills_withdrawn",
                            "hundred_dollar_bills_withdrawn",
                        ]
                        if c in candidate_numeric
                    ]
                    default_series = preferred_defaults if preferred_defaults else candidate_numeric
                    selected_series_cols = default_series
                    if effective_layout == "Wide":
                        selected_series_cols = st.multiselect(
                            "Wide format series columns",
                            options=candidate_numeric,
                            default=default_series,
                            help="For wide-format CSVs, choose one or more numeric columns to analyze.",
                        )

                try:
                    daily_df, available_series = prepare_daily_wide(
                        filtered_df,
                        date_col=date_col,
                        data_shape=effective_layout,
                        selected_series_cols=selected_series_cols,
                        long_denom_col=long_denom_col,
                        long_value_col=long_value_col,
                    )
                except Exception as exc:
                    st.error(f"Could not prepare daily series from this CSV: {exc}")
                    available_series = []
                    daily_df = pd.DataFrame()

                if not daily_df.empty:
                    daily_df = daily_df.sort_values("date")

                    st.subheader("Prepared Daily Series")
                    st.dataframe(daily_df.head(200), use_container_width=True)

                    with st.sidebar:
                        selected_series = st.multiselect(
                            "Series to plot",
                            options=available_series,
                            default=available_series,
                        )
                        plot_mode = "Separate"
                        show_bands_combined = True
                        if len(selected_series) > 1:
                            plot_mode = st.radio(
                                "Plot mode",
                                options=["Combined", "Separate"],
                                index=0,
                                horizontal=True,
                                help="Combined overlays selected series on one chart with different colors.",
                            )
                            show_bands_combined = st.checkbox(
                                "Show bands on combined plot",
                                value=True,
                            )

                    if not selected_series:
                        st.warning("Select at least one series to draw charts.")
                    else:
                        rolling_by_series: dict[str, pd.DataFrame] = {}
                        for series_name in selected_series:
                            rolling_df = compute_rolling(daily_df, series_name, window_days)
                            if rolling_df.empty:
                                st.warning(
                                    f"Not enough observations for '{series_name}' with a {window_days}-day window."
                                )
                                continue
                            rolling_by_series[series_name] = rolling_df

                        if not rolling_by_series:
                            st.warning("No rolling series available after filtering and window settings.")
                        else:
                            st.subheader("Rolling Statistics Charts")

                            if plot_mode == "Combined" and len(rolling_by_series) > 1:
                                combined_fig = build_combined_rolling_figure(
                                    rolling_by_series,
                                    band_mode,
                                    show_bands_combined,
                                    window_days,
                                    confidence_pct,
                                    bollinger_k,
                                )
                                st.plotly_chart(combined_fig, use_container_width=True)

                            for series_name, rolling_df in rolling_by_series.items():
                                if plot_mode == "Separate" or len(rolling_by_series) == 1:
                                    fig = build_rolling_figure(
                                        rolling_df,
                                        series_name,
                                        band_mode,
                                        window_days,
                                        confidence_pct,
                                        bollinger_k,
                                    )
                                    st.plotly_chart(fig, use_container_width=True)

                                metric_col_1, metric_col_2, metric_col_3 = st.columns(3)
                                metric_col_1.metric("Latest Variance", f"{rolling_df['moving_variance'].iloc[-1]:,.2f}")
                                metric_col_2.metric("Average Variance", f"{rolling_df['moving_variance'].mean():,.2f}")
                                metric_col_3.metric("Max Variance", f"{rolling_df['moving_variance'].max():,.2f}")

                                if show_variance_panel:
                                    variance_fig = build_variance_figure(rolling_df, series_name)
                                    st.plotly_chart(variance_fig, use_container_width=True)
                                    with st.expander(f"{series_name} rolling stats table", expanded=False):
                                        st.dataframe(
                                            rolling_df[["date", "moving_average", "moving_variance"]],
                                            use_container_width=True,
                                            height=260,
                                        )

                                export_cols = ["date", "moving_average", "moving_variance"]
                                csv_bytes = rolling_df[export_cols].to_csv(index=False).encode("utf-8")
                                st.download_button(
                                    label=f"Download {series_name} rolling stats CSV",
                                    data=csv_bytes,
                                    file_name=f"rolling_stats_{series_name}_window{window_days}.csv",
                                    mime="text/csv",
                                )

    with tab_map:
        render_map_tab(project_root)


if __name__ == "__main__":
    main()
