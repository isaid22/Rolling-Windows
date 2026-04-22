#!/usr/bin/env python3
"""Streamlit UI for ATM bill rolling-window analytics."""

from __future__ import annotations

from pathlib import Path
from statistics import NormalDist

import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative
import streamlit as st

st.set_page_config(
    page_title="ATM Rolling Window Analytics",
    page_icon="chart_with_upwards_trend",
    layout="wide",
)


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
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
        template="plotly_white",
        title=f"{series_label} Rolling Window Trend",
        xaxis_title="Date",
        yaxis_title="Bills Withdrawn",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
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
    selected_filter_values: dict[str, list[str]],
) -> pd.DataFrame:
    filtered = df.copy()
    for col in selected_filter_cols:
        values = selected_filter_values.get(col, [])
        if values:
            filtered = filtered[filtered[col].astype(str).isin(values)]
    return filtered


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
        margin={"l": 10, "r": 10, "t": 110, "b": 10},
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
        legend={"orientation": "h", "yanchor": "top", "y": 1.0, "x": 0},
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
        '<p class="app-subtitle">Load any CSV, choose your date and series columns, and visualize rolling window trends with variance controls.</p>',
        unsafe_allow_html=True,
    )

    project_root = Path(__file__).resolve().parent.parent
    csv_files = list_csv_files(project_root)

    with st.sidebar:
        st.header("Controls")
        selected_path: Path | None = None
        if csv_files:
            csv_options = {p.name: p for p in csv_files}
            selected_name = st.selectbox("Choose input CSV", options=list(csv_options.keys()))
            selected_path = csv_options[selected_name]
        else:
            st.warning("No CSV files found in the project root.")

        uploaded = st.file_uploader("Or upload CSV", type=["csv"])

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
        return

    st.markdown(f'<div class="info-card">{source_label}</div>', unsafe_allow_html=True)

    try:
        window_days = resolve_window_days(window_days_slider, window_days_text)
        confidence_pct = resolve_confidence_pct(confidence_pct_slider, confidence_pct_text)
        bollinger_k = resolve_bollinger_k(bollinger_k_slider, bollinger_k_text)
    except ValueError as exc:
        st.error(str(exc))
        return

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
        selected_filter_values: dict[str, list[str]] = {}
        for col in selected_filter_cols:
            distinct_values = sorted(raw_df[col].dropna().astype(str).unique().tolist())
            selected_filter_values[col] = st.multiselect(
                f"Values for {col}",
                options=distinct_values,
                default=distinct_values,
            )

    filtered_df = apply_row_filters(raw_df, selected_filter_cols, selected_filter_values)
    st.caption(f"Rows after filters: {len(filtered_df):,} / {len(raw_df):,}")
    if filtered_df.empty:
        st.warning("No rows left after applying filters. Adjust filter selections.")
        return

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
        return

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
        return

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
        return

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


if __name__ == "__main__":
    main()
