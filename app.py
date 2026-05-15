"""
app.py — Rational Entry Pro
Multi-ticker Momentum & Safety Scorer with AI Analyst + Q&A Chat
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import time
import io
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib import colors
from engine import compute, CLOUD_PROVIDERS, LOCAL_PROVIDERS, StockResult
from ai_client import AIClient, build_analyst_prompt, build_chat_system, CLOUD_PROVIDERS as AI_CLOUD, LOCAL_PROVIDERS as AI_LOCAL
from charts import gauge_chart, price_chart, rsi_chart, comparison_bar, spider_chart, scatter_rsi_vs_dev

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Rational Entry Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500&display=swap');
html,body,[class*="css"]{font-family:'IBM Plex Sans',sans-serif}
h1,h2,h3,h4{font-family:'IBM Plex Mono',monospace!important}
.main .block-container{padding-top:1.5rem;max-width:1300px}
section[data-testid="stSidebar"]{background:#0d1117;border-right:1px solid #21262d}
section[data-testid="stSidebar"] *{color:#c9d1d9!important}
section[data-testid="stSidebar"] label{color:#8b949e!important;font-size:11px;text-transform:uppercase;letter-spacing:.07em}
div[data-testid="metric-container"]{background:#161b22;border:1px solid #21262d;border-radius:8px;padding:.9rem 1.1rem}
div[data-testid="metric-container"] label{font-family:'IBM Plex Mono',monospace!important;font-size:10px!important;text-transform:uppercase;letter-spacing:.07em;color:#8b949e!important}
div[data-testid="metric-container"] [data-testid="stMetricValue"]{font-family:'IBM Plex Mono',monospace!important;font-size:24px!important;font-weight:600!important}
.sec{font-family:'IBM Plex Mono',monospace;font-size:10px;text-transform:uppercase;letter-spacing:.1em;color:#8b949e;border-bottom:1px solid #21262d;padding-bottom:.35rem;margin:1.2rem 0 .6rem}
.badge-g{background:#0d3622;color:#3fb950;border:1px solid #238636;padding:3px 12px;border-radius:16px;font-size:12px;font-family:'IBM Plex Mono',monospace;font-weight:500}
.badge-a{background:#2b2108;color:#d29922;border:1px solid #9e6a03;padding:3px 12px;border-radius:16px;font-size:12px;font-family:'IBM Plex Mono',monospace;font-weight:500}
.badge-r{background:#3b1219;color:#f85149;border:1px solid #da3633;padding:3px 12px;border-radius:16px;font-size:12px;font-family:'IBM Plex Mono',monospace;font-weight:500}
.scard-g{background:#0d3622;border:1px solid #238636;border-radius:10px;padding:1rem 1.2rem;font-family:'IBM Plex Mono',monospace}
.scard-r{background:#3b1219;border:1px solid #da3633;border-radius:10px;padding:1rem 1.2rem;font-family:'IBM Plex Mono',monospace}
.sval-g{font-size:28px;font-weight:600;color:#56d364}
.sval-r{font-size:28px;font-weight:600;color:#ff7b72}
.slbl{font-size:10px;text-transform:uppercase;letter-spacing:.08em;margin-bottom:5px}
.slbl-g{color:#3fb950}.slbl-r{color:#f85149}
.ssub-g{font-size:10px;color:#3fb950;margin-top:3px}
.ssub-r{font-size:10px;color:#f85149;margin-top:3px}
.insight{background:#161b22;border:1px solid #21262d;border-left:3px solid #1f6feb;border-radius:0 8px 8px 0;padding:1rem 1.2rem;font-size:13px;line-height:1.75;color:#c9d1d9;margin:.5rem 0}
.insight strong{color:#fff;font-weight:500}
.chat-user{background:#1f3a5f;border-radius:12px 12px 4px 12px;padding:.65rem 1rem;font-size:13px;color:#cae3ff;margin:.4rem 0;max-width:85%;margin-left:auto}
.chat-ai{background:#161b22;border:1px solid #21262d;border-radius:4px 12px 12px 12px;padding:.65rem 1rem;font-size:13px;color:#c9d1d9;line-height:1.7;margin:.4rem 0;max-width:90%}
.stock-chip{display:inline-block;padding:3px 10px;border-radius:12px;font-size:11px;font-weight:500;font-family:'IBM Plex Mono',monospace;margin:2px}
.chip-g{background:#0d3622;color:#3fb950;border:1px solid #238636}
.chip-a{background:#2b2108;color:#d29922;border:1px solid #9e6a03}
.chip-r{background:#3b1219;color:#f85149;border:1px solid #da3633}
.rank-row:hover{background:#161b22}
.appbar{font-family:'IBM Plex Mono',monospace;font-size:20px;font-weight:600;color:#58a6ff;letter-spacing:-.01em}
.appbar-sub{font-size:12px;color:#8b949e;margin-bottom:1rem}
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────
def init_state():
    defaults = {
        "results":       {},    # ticker -> StockResult
        "analyst_text":  "",
        "analyst_done":  False,
        "chat_history":  [],    # [{role, content, display}]
        "period":        "200d",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ── Helper: zone badge ─────────────────────────────────────────
def zone_badge(zone: str) -> str:
    labels = {"green": "🟢 GREEN ZONE", "amber": "🟡 AMBER ZONE", "red": "🔴 RED ZONE"}
    cls    = {"green": "badge-g",        "amber": "badge-a",        "red": "badge-r"}
    return f"<span class='{cls[zone]}'>{labels[zone]}</span>"

def score_color(s):
    return "#3fb950" if s < 34 else "#d29922" if s < 67 else "#f85149"

def figure_to_pdf_image(fig, width_px: int, height_px: int, width_in: float, height_in: float) -> Image:
    """Render a Plotly figure into a ReportLab image."""
    img_buffer = io.BytesIO(fig.to_image(format="png", width=width_px, height=height_px, scale=2))
    img_buffer.seek(0)
    return Image(img_buffer, width=width_in * inch, height=height_in * inch)

# ── PDF Generation ────────────────────────────────────────────
def generate_pdf_report(valid_results):
    """Generate a scanner PDF with summary, overview charts, and per-ticker charts."""
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    story = []
    result_list = sorted(valid_results, key=lambda r: r.risk_score)
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#58a6ff'),
        spaceAfter=0.3*inch,
    )
    
    # Title
    story.append(Paragraph("📊 Rational Entry Pro — Scanner Results", title_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Summary table
    summary_data = [["Ticker", "Price", "RSI 14", "SMA 50", "MA Dev", "Risk Score", "Zone"]]
    for r in result_list:
        summary_data.append([
            r.ticker,
            f"${r.price:.2f}",
            f"{r.rsi:.1f}",
            f"${r.sma50:.2f}",
            f"{r.ma_dev:+.1f}%",
            f"{r.risk_score:.0f}/100",
            r.zone.upper(),
        ])
    
    summary_table = Table(summary_data, colWidths=[0.8*inch, 0.9*inch, 0.8*inch, 0.9*inch, 0.8*inch, 1.0*inch, 0.8*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#161b22')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#c9d1d9')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#0d1117')),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.HexColor('#8b949e')),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#0d1117'), colors.HexColor('#161b22')]),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#21262d')),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 0.3*inch))
    story.append(PageBreak())

    story.append(Paragraph("Scanner Chart Report", styles['Heading2']))
    story.append(Spacer(1, 0.15*inch))
    story.append(figure_to_pdf_image(comparison_bar(result_list), width_px=900, height_px=420, width_in=6.5, height_in=2.9))
    story.append(Spacer(1, 0.15*inch))
    story.append(figure_to_pdf_image(scatter_rsi_vs_dev(result_list), width_px=900, height_px=460, width_in=6.5, height_in=3.0))
    story.append(Spacer(1, 0.15*inch))
    story.append(figure_to_pdf_image(spider_chart(result_list), width_px=900, height_px=460, width_in=6.1, height_in=3.0))
    story.append(PageBreak())
    
    # Per-ticker details with charts
    for i, r in enumerate(result_list):
        story.append(Paragraph(f"{r.ticker} - {r.company_name} ({r.sector})", styles['Heading2']))
        story.append(Spacer(1, 0.1*inch))

        # Charts - capture as images
        try:
            story.append(figure_to_pdf_image(gauge_chart(r.risk_score, r.zone), width_px=520, height_px=300, width_in=3.0, height_in=1.75))
            story.append(Spacer(1, 0.1*inch))
            story.append(figure_to_pdf_image(price_chart(r), width_px=900, height_px=500, width_in=6.5, height_in=3.4))
            story.append(Spacer(1, 0.1*inch))
            story.append(figure_to_pdf_image(rsi_chart(r), width_px=900, height_px=320, width_in=6.5, height_in=2.2))
            story.append(Spacer(1, 0.1*inch))
            
            # Metrics
            metrics_text = f"""
            <b>Price:</b> ${r.price:.2f} | <b>RSI:</b> {r.rsi:.1f} | <b>SMA50:</b> ${r.sma50:.2f}<br/>
            <b>MA Dev:</b> {r.ma_dev:+.1f}% | <b>ATR:</b> ${r.atr:.2f} | <b>Stop-Loss:</b> ${r.stop_loss:.2f}<br/>
            <b>Risk Score:</b> {r.risk_score:.0f}/100 | <b>Zone:</b> {r.zone.upper()} | <b>Date:</b> {r.last_date}
            """
            story.append(Paragraph(metrics_text, styles['Normal']))
            
            if i < len(result_list) - 1:
                story.append(PageBreak())
        except Exception as e:
            story.append(Paragraph(f"Error rendering charts for {r.ticker}: {str(e)}", styles['Normal']))
    
    # Build PDF
    doc.build(story)
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()

# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div class='appbar'>📊 RATIONAL ENTRY</div><div class='appbar-sub'>PRO · Multi-Ticker Scanner</div>", unsafe_allow_html=True)

    # ── Tickers ──
    st.markdown("<div class='sec'>Tickers</div>", unsafe_allow_html=True)
    raw = st.text_area(
        "Enter tickers (comma or newline separated)",
        value="AAPL, MSFT, NVDA",
        height=90,
        help="Up to 10 tickers. E.g.: AAPL, TSLA, NVDA, MSFT",
        label_visibility="collapsed",
    )
    period = st.selectbox("Data window", ["100d", "200d", "1y", "2y"], index=1)

    # ── Model ──
    st.markdown("<div class='sec'>AI Model</div>", unsafe_allow_html=True)
    model_type = st.radio("Engine", ["☁️ Cloud API", "💻 Local / No-API"], horizontal=True, label_visibility="collapsed")

    ai_client = None
    model_label = "—"

    if "Cloud" in model_type:
        provider = st.selectbox("Provider", list(AI_CLOUD.keys()))
        meta = AI_CLOUD[provider]
        model_id = st.text_input("Model ID", value=meta["default_model"])
        api_key  = st.text_input("API Key", type="password", placeholder=meta["key_hint"])
        if api_key:
            ai_client   = AIClient("cloud", provider, model_id, api_key=api_key)
            model_label = f"{provider} / {model_id}"
    else:
        provider = st.selectbox("Provider", list(AI_LOCAL.keys()))
        meta = AI_LOCAL[provider]
        base_url  = st.text_input("Base URL",   value=meta["url"])
        model_id  = st.text_input("Model name", value=meta["default_model"])
        ai_client   = AIClient("local", provider, model_id, base_url=base_url)
        model_label = f"{provider} / {model_id}"
        st.caption(f"ℹ️ Make sure {provider} is running before analyzing.")

    st.markdown("---")
    analyze_btn = st.button("⚡ ANALYZE TICKERS", use_container_width=True, type="primary")

    st.markdown("""
    <div style='font-size:11px;color:#8b949e;line-height:1.75;font-family:"IBM Plex Mono",monospace;margin-top:.5rem'>
    <b style='color:#c9d1d9'>ZONES</b><br>
    🟢 RSI&lt;35 or Price≤SMA50<br>
    🟡 Neutral momentum<br>
    🔴 RSI&gt;70 AND Dev&gt;15%<br><br>
    <b style='color:#c9d1d9'>STOP-LOSS</b><br>
    Price − (2 × ATR14)<br><br>
    <b style='color:#c9d1d9'>CACHE</b> 15 min · yfinance
    </div>""", unsafe_allow_html=True)

# ── Run analysis ───────────────────────────────────────────────
if analyze_btn:
    tickers = [t.strip().upper() for t in raw.replace("\n", ",").split(",") if t.strip()]
    tickers = list(dict.fromkeys(tickers))[:10]  # dedupe, max 10

    if not tickers:
        st.error("Enter at least one ticker symbol.")
    else:
        st.session_state["results"]      = {}
        st.session_state["analyst_text"] = ""
        st.session_state["analyst_done"] = False
        st.session_state["period"]       = period

        prog = st.progress(0, text="Starting analysis…")
        for i, t in enumerate(tickers):
            prog.progress((i) / len(tickers), text=f"Fetching {t}…")
            r = compute(t, period)
            st.session_state["results"][t] = r
            time.sleep(0.05)
        prog.progress(1.0, text="Done ✓")
        time.sleep(0.4)
        prog.empty()

# ── Main content ───────────────────────────────────────────────
results: dict = st.session_state["results"]

if not results:
    st.markdown("""
    <div style='text-align:center;padding:4rem 2rem;color:#8b949e'>
      <div style='font-size:48px;margin-bottom:1rem'>📊</div>
      <div style='font-family:"IBM Plex Mono",monospace;font-size:16px;color:#58a6ff;margin-bottom:.5rem'>RATIONAL ENTRY PRO</div>
      <div style='font-size:13px;line-height:1.8'>Enter tickers in the sidebar and click <b style='color:#c9d1d9'>ANALYZE TICKERS</b>.<br>
      Supports up to 10 tickers simultaneously.<br>
      Get AI analyst reports and ask questions about any stock.</div>
    </div>""", unsafe_allow_html=True)
    st.stop()

valid   = {t: r for t, r in results.items() if not r.error}
invalid = {t: r for t, r in results.items() if r.error}

# ── Error notices ──────────────────────────────────────────────
for t, r in invalid.items():
    st.error(f"**{t}**: {r.error}")

if not valid:
    st.stop()

# ═══════════════════════════════════════════════════════════════
#  TABS
# ═══════════════════════════════════════════════════════════════
tab_scanner, tab_compare, tab_analyst, tab_chat = st.tabs([
    "🔍 Scanner", "📊 Compare", "🧠 AI Analyst", "💬 Ask AI"
])

# ───────────────────────────────────────────────────────────────
#  TAB 1: SCANNER
# ───────────────────────────────────────────────────────────────
with tab_scanner:
    result_list = sorted(valid.values(), key=lambda r: r.risk_score)

    # Summary chips
    chips_html = ""
    for r in result_list:
        cls = {"green": "chip-g", "amber": "chip-a", "red": "chip-r"}[r.zone]
        chips_html += f"<span class='stock-chip {cls}'>{r.ticker} · {r.risk_score:.0f}</span>"
    st.markdown(chips_html, unsafe_allow_html=True)
    with st.expander("View full scanner chart report", expanded=True):
        st.caption("Overview graphs are available here, and the same visuals are included in the PDF export.")
        report_col1, report_col2 = st.columns(2)
        with report_col1:
            st.markdown("<div class='sec'>Risk Score Ranking</div>", unsafe_allow_html=True)
            st.plotly_chart(comparison_bar(result_list), use_container_width=True, key="scanner_report_bar")
        with report_col2:
            st.markdown("<div class='sec'>Multi-Dimension Radar</div>", unsafe_allow_html=True)
            st.plotly_chart(spider_chart(result_list), use_container_width=True, key="scanner_report_spider")
        st.markdown("<div class='sec'>RSI vs MA Deviation</div>", unsafe_allow_html=True)
        st.plotly_chart(scatter_rsi_vs_dev(result_list), use_container_width=True, key="scanner_report_scatter")

    st.markdown("")

    # Download button
    df_table = pd.DataFrame([{
        "Ticker":     r.ticker,
        "Company":    r.company_name,
        "Sector":     r.sector,
        "Price":      f"${r.price:.2f}",
        "RSI 14":     f"{r.rsi:.1f}",
        "SMA 50":     f"${r.sma50:.2f}",
        "MA Dev %":   f"{r.ma_dev:+.1f}%",
        "ATR 14":     f"${r.atr:.2f}",
        "Stop-Loss":  f"${r.stop_loss:.2f}",
        "Risk Score": f"{r.risk_score:.0f}/100",
        "Zone":       r.zone.upper(),
        "Last Date":  r.last_date,
    } for r in result_list])
    csv = df_table.to_csv(index=False)
    
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("⬇️ Download Results (CSV)", csv, "scanner_results.csv", "text/csv", use_container_width=True)
    with col2:
        try:
            pdf = generate_pdf_report(result_list)
            st.download_button("⬇️ Download Full Report (PDF + Charts)", pdf, "scanner_results.pdf", "application/pdf", use_container_width=True)
        except Exception as e:
            st.warning(f"PDF generation unavailable: {str(e)}")
    
    st.markdown("")
    for r in result_list:
        st.markdown(f"<div class='sec'>{r.ticker} — {r.company_name} · {r.sector}</div>", unsafe_allow_html=True)

        h1, h2, h3 = st.columns([4, 1, 1])
        with h1:
            st.markdown(f"""
            <div style='font-family:"IBM Plex Mono",monospace'>
              <span style='font-size:22px;font-weight:600;color:#c9d1d9'>${r.price:.2f}</span>
              <span style='font-size:12px;color:#8b949e;margin-left:10px'>as of {r.last_date}</span>
            </div>""", unsafe_allow_html=True)
        with h2:
            st.markdown(zone_badge(r.zone), unsafe_allow_html=True)
        with h3:
            st.markdown(f"<div style='font-size:11px;color:#8b949e;font-family:\"IBM Plex Mono\",monospace'>via {model_label}</div>", unsafe_allow_html=True)

        # Metrics
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        rsi_delta = "Overbought 🔴" if r.rsi > 70 else "Oversold 🟢" if r.rsi < 35 else "Neutral 🟡"
        with c1: st.metric("Price",      f"${r.price:.2f}")
        with c2: st.metric("RSI 14",     f"{r.rsi:.1f}",     delta=rsi_delta,       delta_color="off")
        with c3: st.metric("SMA 50",     f"${r.sma50:.2f}")
        with c4: st.metric("MA Dev",     f"{r.ma_dev:+.1f}%")
        with c5: st.metric("ATR 14",     f"${r.atr:.2f}")
        with c6: st.metric("Risk Score", f"{r.risk_score:.0f}/100",
                            delta="vs 100", delta_color="off")

        # Gauge + Safety
        gcol, s1, s2 = st.columns([2, 1, 1])
        with gcol:
            st.plotly_chart(gauge_chart(r.risk_score, r.zone), use_container_width=True, key=f"gauge_{r.ticker}")
        with s1:
            st.markdown(f"""
            <div class='scard-g'>
              <div class='slbl slbl-g'>✅ Safe Buy Price</div>
              <div class='sval-g'>${r.price:.2f}</div>
              <div class='ssub-g'>Current market price</div>
            </div>""", unsafe_allow_html=True)
        with s2:
            st.markdown(f"""
            <div class='scard-r'>
              <div class='slbl slbl-r'>🛑 Stop-Loss Price</div>
              <div class='sval-r'>${r.stop_loss:.2f}</div>
              <div class='ssub-r'>Price − 2 × ATR(14)</div>
            </div>""", unsafe_allow_html=True)

        cushion     = r.price - r.stop_loss
        cushion_pct = cushion / r.price * 100
        st.caption(f"📐 Risk cushion: **${cushion:.2f}** ({cushion_pct:.1f}%) — volatility-adjusted buffer below current price.")

        # Charts
        ch1, ch2 = st.columns([3, 1])
        with ch1:
            st.plotly_chart(price_chart(r), use_container_width=True, key=f"price_{r.ticker}")
        with ch2:
            st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)
            st.plotly_chart(rsi_chart(r), use_container_width=True, key=f"rsi_{r.ticker}")

        # Insight
        insight_text = {
            "green": f"**{r.ticker}** is showing a rational entry signal. RSI at {r.rsi:.1f} is {'in oversold territory — strong mean-reversion opportunity' if r.rsi < 35 else 'not overextended'}. Price is {abs(r.ma_dev):.1f}% {'below' if r.ma_dev < 0 else 'near'} the 50-day SMA. Stop-loss at ${r.stop_loss:.2f} gives a {cushion_pct:.1f}% risk cushion.",
            "amber": f"**{r.ticker}** is in a neutral zone — RSI at {r.rsi:.1f} with price {r.ma_dev:+.1f}% from SMA50. No strong entry or exit signal. Wait for RSI to drop below 35 or price to touch the SMA50 (${r.sma50:.2f}) for a cleaner setup. Stop-loss: ${r.stop_loss:.2f}.",
            "red":   f"⚠️ **{r.ticker}** is overextended. RSI at {r.rsi:.1f} (overbought) and price {r.ma_dev:.1f}% above SMA50 — both conditions that historically precede pullbacks. Not a rational entry point. Consider waiting for a reset. Stop-loss: ${r.stop_loss:.2f}.",
        }[r.zone]
        st.markdown(f"<div class='insight'>{insight_text}</div>", unsafe_allow_html=True)
        st.divider()

# ───────────────────────────────────────────────────────────────
#  TAB 2: COMPARE
# ───────────────────────────────────────────────────────────────
with tab_compare:
    rv = list(valid.values())

    st.markdown("<div class='sec'>Risk Score Ranking</div>", unsafe_allow_html=True)
    st.plotly_chart(comparison_bar(rv), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='sec'>RSI vs MA Deviation — Quadrant View</div>", unsafe_allow_html=True)
        st.plotly_chart(scatter_rsi_vs_dev(rv), use_container_width=True)
    with c2:
        st.markdown("<div class='sec'>Multi-Dimension Radar</div>", unsafe_allow_html=True)
        st.plotly_chart(spider_chart(rv), use_container_width=True)

    st.markdown("<div class='sec'>Full Comparison Table</div>", unsafe_allow_html=True)
    df_table = pd.DataFrame([{
        "Ticker":     r.ticker,
        "Company":    r.company_name,
        "Sector":     r.sector,
        "Price":      f"${r.price:.2f}",
        "RSI 14":     f"{r.rsi:.1f}",
        "SMA 50":     f"${r.sma50:.2f}",
        "MA Dev %":   f"{r.ma_dev:+.1f}%",
        "ATR 14":     f"${r.atr:.2f}",
        "Stop-Loss":  f"${r.stop_loss:.2f}",
        "Risk Score": f"{r.risk_score:.0f}/100",
        "Zone":       r.zone.upper(),
        "Last Date":  r.last_date,
    } for r in sorted(rv, key=lambda x: x.risk_score)])
    st.dataframe(df_table, use_container_width=True, hide_index=True)

    # Download
    csv = df_table.to_csv(index=False)
    st.download_button("⬇️ Download CSV", csv, "rational_entry_analysis.csv", "text/csv")

# ───────────────────────────────────────────────────────────────
#  TAB 3: AI ANALYST
# ───────────────────────────────────────────────────────────────
with tab_analyst:
    if ai_client is None:
        st.warning("🔑 Enter your API key in the sidebar to enable the AI Analyst.")
    else:
        gen_col, _ = st.columns([1, 3])
        with gen_col:
            gen_btn = st.button("🧠 Generate Analyst Report", type="primary", use_container_width=True)

        if gen_btn or (st.session_state["analyst_text"] and not st.session_state["analyst_done"]):
            with st.spinner("Generating analyst report…"):
                try:
                    prompt = build_analyst_prompt(list(valid.values()))
                    text   = ai_client.chat([{"role": "user", "content": prompt}], max_tokens=2000)
                    st.session_state["analyst_text"] = text
                    st.session_state["analyst_done"] = True
                except Exception as e:
                    st.error(f"AI error: {e}")

        if st.session_state["analyst_text"]:
            # Signal chips
            chips = ""
            for r in valid.values():
                cls = {"green": "chip-g", "amber": "chip-a", "red": "chip-r"}[r.zone]
                lbl = {"green": "BUY SIGNAL", "amber": "NEUTRAL", "red": "CAUTION"}[r.zone]
                chips += f"<span class='stock-chip {cls}'>{r.ticker} · {lbl}</span>"
            st.markdown(chips + "<br><br>", unsafe_allow_html=True)

            # Format report
            text = st.session_state["analyst_text"]
            text = text.replace("**", "**")
            st.markdown(
                f"<div class='insight' style='font-size:14px;line-height:1.9'>"
                f"{text.replace(chr(10), '<br>')}"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.caption(f"Generated by {model_label} · For educational purposes only · Not financial advice.")

            # Download report
            st.download_button(
                "⬇️ Download Report",
                st.session_state["analyst_text"],
                "analyst_report.txt",
                "text/plain",
            )
        elif not gen_btn:
            st.info("Click **Generate Analyst Report** to get a full AI-written analysis of all scanned tickers.")

# ───────────────────────────────────────────────────────────────
#  TAB 4: ASK AI (Chat)
# ───────────────────────────────────────────────────────────────
with tab_chat:
    if ai_client is None:
        st.warning("🔑 Enter your API key or configure a local model in the sidebar to enable the AI chat.")
    else:
        # Suggested questions
        st.markdown("<div class='sec'>Suggested Questions</div>", unsafe_allow_html=True)
        suggestions = [
            "Which ticker is the safest entry right now?",
            "Compare the RSI signals across all stocks.",
            "Which stock has the best risk/reward ratio?",
            "Explain each stop-loss level and why it matters.",
            "Any red flags I should know about?",
            "Which stock should I avoid and why?",
            "What would a 5% portfolio allocation look like?",
        ]
        cols = st.columns(4)
        for i, s in enumerate(suggestions):
            if cols[i % 4].button(s, key=f"sug_{i}", use_container_width=True):
                st.session_state["chat_history"].append({"role": "user", "content": s, "display": s})

        st.markdown("<div class='sec'>Chat</div>", unsafe_allow_html=True)

        # Render history
        for msg in st.session_state["chat_history"]:
            if msg["role"] == "user":
                st.markdown(f"<div class='chat-user'>{msg['display']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='chat-ai'>{msg['display'].replace(chr(10),'<br>')}</div>", unsafe_allow_html=True)

        # If last message is user → get AI response
        history = st.session_state["chat_history"]
        if history and history[-1]["role"] == "user":
            with st.spinner("Thinking…"):
                try:
                    system  = build_chat_system(list(valid.values()))
                    api_msgs = [{"role": "user", "content": system + "\n\n---\n\nConversation so far:"}]
                    for m in history[-10:]:
                        api_msgs.append({"role": m["role"], "content": m["content"]})
                    reply = ai_client.chat(api_msgs, max_tokens=1000, temperature=0.4)
                    st.session_state["chat_history"].append({
                        "role": "assistant", "content": reply, "display": reply
                    })
                    st.rerun()
                except Exception as e:
                    st.error(f"AI error: {e}")

        # Input
        user_input = st.chat_input("Ask about the analyzed stocks…")
        if user_input:
            st.session_state["chat_history"].append({
                "role": "user", "content": user_input, "display": user_input
            })
            st.rerun()

        if st.button("🗑️ Clear chat", key="clear_chat"):
            st.session_state["chat_history"] = []
            st.rerun()

# ── Footer ─────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='font-size:11px;color:#8b949e;text-align:center;font-family:\"IBM Plex Mono\",monospace'>"
    "⚠️ For financial and educational purposes · Data via yfinance"
    "</div>",
    unsafe_allow_html=True,
)
