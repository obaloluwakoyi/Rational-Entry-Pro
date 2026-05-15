"""
charts.py — All Plotly figures for Rational Entry Pro
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

ZONE_COL = {"green": "#3fb950", "amber": "#d29922", "red": "#f85149"}


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)

def score_color(s):
    return "#3fb950" if s < 34 else "#d29922" if s < 67 else "#f85149"


def gauge_chart(score: float, zone: str) -> go.Figure:
    color = ZONE_COL[zone]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(score, 1),
        number={"font": {"family": "IBM Plex Mono", "size": 40, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#30363d",
                     "tickfont": {"family": "IBM Plex Mono", "size": 10, "color": "#8b949e"}},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "#0d1117", "borderwidth": 0,
            "steps": [
                {"range": [0,   33], "color": "#0d3622"},
                {"range": [33,  67], "color": "#2b2108"},
                {"range": [67, 100], "color": "#3b1219"},
            ],
            "threshold": {"line": {"color": color, "width": 3}, "thickness": 0.85, "value": score},
        },
        domain={"x": [0, 1], "y": [0, 1]},
    ))
    fig.update_layout(height=230, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(t=20, b=5, l=30, r=30), font={"family": "IBM Plex Mono"})
    return fig


def price_chart(r) -> go.Figure:
    df        = r.df
    close     = df["Close"].squeeze()
    vol       = df["Volume"].squeeze()
    sma50_s   = r.sma50_series
    dates     = df.index.tolist()
    green_mask = close <= sma50_s

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.72, 0.28], vertical_spacing=0.03)

    in_zone, zone_start = False, None
    for i, (d, g) in enumerate(zip(dates, green_mask)):
        if g and not in_zone:
            zone_start = d; in_zone = True
        elif not g and in_zone:
            fig.add_vrect(x0=zone_start, x1=dates[i-1],
                          fillcolor="rgba(63,185,80,0.10)", line_width=0, row=1, col=1)
            in_zone = False
    if in_zone:
        fig.add_vrect(x0=zone_start, x1=dates[-1],
                      fillcolor="rgba(63,185,80,0.10)", line_width=0, row=1, col=1)

    fig.add_hline(y=r.stop_loss, line=dict(color="#f85149", width=0.8, dash="dot"), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=close, mode="lines", name="Price",
        line=dict(color="#58a6ff", width=2),
        hovertemplate="<b>%{x|%b %d %Y}</b><br>$%{y:.2f}<extra></extra>"), row=1, col=1)

    if sma50_s is not None and not sma50_s.empty:
        fig.add_trace(go.Scatter(x=sma50_s.index.tolist(), y=sma50_s,
            mode="lines", name="SMA 50",
            line=dict(color="#f85149", width=1.5, dash="dash"),
            hovertemplate="SMA50: $%{y:.2f}<extra></extra>"), row=1, col=1)

    fig.add_annotation(x=dates[-1], y=r.stop_loss, text=f"  Stop ${r.stop_loss:.2f}",
        showarrow=False, font=dict(size=10, color="#f85149"), xanchor="left", row=1, col=1)

    vol_colors = ["rgba(63,185,80,0.55)" if c >= o else "rgba(248,81,73,0.55)"
                  for c, o in zip(close, df["Open"].squeeze())]
    fig.add_trace(go.Bar(x=dates, y=vol, name="Volume", marker_color=vol_colors,
        hovertemplate="Vol: %{y:,.0f}<extra></extra>"), row=2, col=1)

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d1117",
        font=dict(family="IBM Plex Mono", color="#8b949e", size=11),
        legend=dict(orientation="h", x=0, y=1.06, bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        margin=dict(t=10, b=10, l=0, r=0), height=480, hovermode="x unified",
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor="#21262d", zeroline=False, tickprefix="$", tickfont=dict(size=10)),
        xaxis2=dict(showgrid=False, tickfont=dict(size=10)),
        yaxis2=dict(gridcolor="#21262d", zeroline=False, tickfont=dict(size=10), title="Vol"),
    )
    return fig


def rsi_chart(r) -> go.Figure:
    close = r.df["Close"].squeeze()
    rsi_s = rsi(close, length=14).dropna()
    fig   = go.Figure()
    fig.add_hrect(y0=70, y1=100, fillcolor="rgba(248,81,73,0.08)", line_width=0)
    fig.add_hrect(y0=0,  y1=35,  fillcolor="rgba(63,185,80,0.08)",  line_width=0)
    fig.add_hline(y=70, line=dict(color="#f85149", width=1, dash="dot"))
    fig.add_hline(y=35, line=dict(color="#3fb950", width=1, dash="dot"))
    fig.add_trace(go.Scatter(x=rsi_s.index.tolist(), y=rsi_s, mode="lines", name="RSI 14",
        line=dict(color="#d29922", width=1.8),
        fill="tozeroy", fillcolor="rgba(210,153,34,0.06)",
        hovertemplate="RSI: %{y:.1f}<extra></extra>"))
    fig.add_annotation(x=rsi_s.index[-1], y=70, text="  Overbought",
        showarrow=False, font=dict(size=9, color="#f85149"), xanchor="left")
    fig.add_annotation(x=rsi_s.index[-1], y=35, text="  Oversold",
        showarrow=False, font=dict(size=9, color="#3fb950"), xanchor="left")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d1117",
        font=dict(family="IBM Plex Mono", color="#8b949e", size=11),
        height=200, margin=dict(t=10, b=10, l=0, r=0), showlegend=False,
        yaxis=dict(range=[0, 100], gridcolor="#21262d", tickfont=dict(size=10)),
        xaxis=dict(showgrid=False, tickfont=dict(size=10)),
    )
    return fig


def comparison_bar(results: list) -> go.Figure:
    sr      = sorted(results, key=lambda r: r.risk_score)
    tickers = [r.ticker for r in sr]
    scores  = [r.risk_score for r in sr]
    colors  = [score_color(s) for s in scores]
    fig = go.Figure(go.Bar(
        x=scores, y=tickers, orientation="h", marker_color=colors,
        text=[f"{s:.0f}" for s in scores], textposition="outside",
        hovertemplate="%{y}: %{x:.1f} / 100<extra></extra>",
    ))
    fig.add_vline(x=33, line=dict(color="#3fb950", width=1, dash="dot"))
    fig.add_vline(x=67, line=dict(color="#f85149", width=1, dash="dot"))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d1117",
        font=dict(family="IBM Plex Mono", color="#8b949e", size=12),
        margin=dict(t=10, b=10, l=0, r=40),
        height=max(180, len(tickers) * 48),
        xaxis=dict(range=[0, 115], showgrid=False, tickfont=dict(size=10)),
        yaxis=dict(gridcolor="#21262d", tickfont=dict(size=12, color="#c9d1d9")),
        showlegend=False,
    )
    return fig


def spider_chart(results: list) -> go.Figure:
    categories = ["RSI Signal", "Mean Reversion", "Low Volatility", "Price Momentum", "Safety Score"]
    palette    = ["#58a6ff", "#3fb950", "#d29922", "#f85149", "#bc8cff", "#ffa657"]
    fig        = go.Figure()
    for i, r in enumerate(results):
        vals = [
            max(0, 100 - abs(r.rsi - 50) * 2),
            max(0, 100 - abs(r.ma_dev) * 3),
            max(0, 100 - r.atr / r.price * 1000),
            max(0, min(100, 50 + r.ma_dev * 2)),
            max(0, 100 - r.risk_score),
        ]
        col = palette[i % len(palette)]
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=categories + [categories[0]],
            fill="toself",
            fillcolor=f"rgba({int(col[1:3],16)},{int(col[3:5],16)},{int(col[5:7],16)},0.10)",
            line=dict(color=col, width=2), name=r.ticker,
            hovertemplate=f"<b>{r.ticker}</b><br>%{{theta}}: %{{r:.0f}}<extra></extra>",
        ))
    fig.update_layout(
        polar=dict(
            bgcolor="#0d1117",
            radialaxis=dict(range=[0,100], tickfont=dict(size=9, color="#8b949e"), gridcolor="#21262d"),
            angularaxis=dict(tickfont=dict(size=11, color="#c9d1d9"), gridcolor="#21262d"),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Mono", color="#c9d1d9", size=11),
        legend=dict(font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=20, b=20, l=20, r=20), height=380,
    )
    return fig


def scatter_rsi_vs_dev(results: list) -> go.Figure:
    fig = go.Figure()
    fig.add_shape(type="rect", x0=-30, x1=0,  y0=0,  y1=35,  fillcolor="rgba(63,185,80,0.10)",  line_width=0)
    fig.add_shape(type="rect", x0=15,  x1=60, y0=70, y1=100, fillcolor="rgba(248,81,73,0.10)", line_width=0)
    fig.add_vline(x=0,  line=dict(color="#3fb950", width=1, dash="dot"))
    fig.add_vline(x=15, line=dict(color="#f85149", width=1, dash="dot"))
    fig.add_hline(y=35, line=dict(color="#3fb950", width=1, dash="dot"))
    fig.add_hline(y=70, line=dict(color="#f85149", width=1, dash="dot"))
    zone_col = {"green": "#3fb950", "amber": "#d29922", "red": "#f85149"}
    for r in results:
        fig.add_trace(go.Scatter(
            x=[r.ma_dev], y=[r.rsi], mode="markers+text",
            marker=dict(color=zone_col[r.zone], size=14, line=dict(color="#0d1117", width=2)),
            text=[r.ticker], textposition="top center",
            textfont=dict(size=11, color="#c9d1d9"), name=r.ticker,
            hovertemplate=f"<b>{r.ticker}</b><br>RSI: {r.rsi:.1f}<br>MA Dev: {r.ma_dev:+.1f}%<extra></extra>",
        ))
    fig.add_annotation(x=-15, y=20, text="Green zone", showarrow=False, font=dict(size=10, color="#3fb950"))
    fig.add_annotation(x=30,  y=80, text="Red zone",   showarrow=False, font=dict(size=10, color="#f85149"))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d1117",
        font=dict(family="IBM Plex Mono", color="#8b949e", size=11),
        xaxis=dict(title="MA Deviation (%)", gridcolor="#21262d", zeroline=False),
        yaxis=dict(title="RSI (14-day)",     gridcolor="#21262d", zeroline=False, range=[0, 100]),
        margin=dict(t=10, b=10, l=0, r=0), height=380, showlegend=False,
    )
    return fig
