# pages/sentiment_analysis.py
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import pages.cores.run_sentiment_pipeline as run_sentiment_pipeline

st.set_page_config(page_title="Sentiment", layout="wide")
st.title("📰 Market Sentiment")

# ===========================

# ===========================
colA, colB, colC = st.columns([1, 1, 1])
with colA:
    thres = st.slider("Neutral band (|compound| ≤)", 0.0, 0.2, 0.05, 0.01)
with colB:
    smoothing_dailys = st.slider("Daily smoothinging (EMA dailys)", 1, 14, 7)
with colC:
    run_now = st.button("Re-run pipeline")


if run_now:
    try:
        run_sentiment_pipeline.thres = float(thres)
        run_sentiment_pipeline.main()
        st.success("Pipeline finished. Data refreshed.")
    except Exception as e:
        st.error(f"Pipeline error: {e}")

# ===========================

# ===========================
DATA_DIRS = ["week8/news_clean_data"]  
csv = None
for d in DATA_DIRS:
    p = os.path.join(d, "clean_news_timeseries.csv")
    if os.path.exists(p):
        csv = p
        break
if not csv:
    st.error("clean_news_timeseries.csv not found. Run the pipeline first.")
    st.stop()

df = pd.read_csv(csv)


df = df.copy()
df["date"] = pd.to_datetime(df["date"], errors="coerce")
for col in ["compound", "pos", "neu", "neg"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["date", "compound"]).sort_values("date")
if df.empty:
    st.error("No rows with valid dates/compound were found.")
    st.stop()


daily = (
    df.set_index("date")
      .groupby(pd.Grouper(freq="D"))
      .agg(
          avg_compound=("compound", "mean"),
          n_articles=("compound", "size"),
      )
      .reset_index()
)


daily["avg_pct"] = (daily["avg_compound"] + 1.0) * 50.0
daily["ema"] = daily["avg_pct"].ewm(span=smoothing_dailys, adjust=False).mean()

# ===========================

# ===========================
st.session_state.setdefault("sentiment", {})
st.session_state["sentiment"].update(
    {
        "df": df,
        "daily": daily,
        "params": {"thres": float(thres), "smoothing_dailys": int(smoothing_dailys)},
        "metrics": {
            "dailys": int(len(daily)),
            "articles": int(df.shape[0] if df is not None else 0),
            "avg_last_ema": float(daily["ema"].iloc[-1]) if len(daily) else None,
            "avg_last7_avg_pct": float(daily.tail(7)["avg_pct"].mean()) if len(daily) else None,
        },
        "signals": {
            "avg_last7_avg_pct": float(daily.tail(7)["avg_pct"].mean()) if len(daily) else None,
            "ema_now": float(daily["ema"].iloc[-1]) if len(daily) else None,
        },
    }
)

# ===========================

# ===========================
k1, k2, k3, k4 = st.columns(4)
k1.metric("dailys covered", f"{len(daily):,}")
k2.metric("Articles", f"{int(df.shape[0]):,}")
k3.metric("avg_last EMA", f"{daily['ema'].iloc[-1]:.1f}" if len(daily) else "—")
k4.metric("Neutral band", f"±{thres:.2f}")

# ===========================
# ===========================
tab_overvi, tab_heatmap, tab_distrib, tab_cls, tab_invest = st.tabs(
    ["Overview", "Heatmap", "Distributions", "Confusion", "Investment View"]
)

# ----- Overview -----
with tab_overvi:
    trend_fig = go.Figure()
    trend_fig.add_trace(
        go.Scatter(
            x=daily["date"],
            y=daily["avg_pct"],
            name="Daily",
            mode="lines",
            line=dict(width=1),
            opacity=0.45,
        )
    )
    trend_fig.add_trace(go.Scatter(x=daily["date"], y=daily["ema"], name=f"EMA {smoothing_dailys}", mode="lines"))
    trend_fig.add_hline(y=50, line_dash="dash", annotation_text="Neutral (50)")
    trend_fig.update_layout(
        title="Daily Average Sentiment (0–100)",
        xaxis_title="Date",
        yaxis_title="Index",
        hovermode="x unified",
    )
    st.plotly_chart(trend_fig, use_container_width=True)

    st.session_state["sentiment"]["overview_series"] = {
        "date": daily["date"].astype(str).tolist(),
        "avg_pct": daily["avg_pct"].astype(float).tolist(),
        "ema": daily["ema"].astype(float).tolist(),
    }

# ----- Heatmap -----
with tab_heatmap:
    m = daily.copy()
    m["Year"] = m["date"].dt.year
    m["Month"] = m["date"].dt.strftime("%b")
    heat = (
        m.pivot_table(index="Year", columns="Month", values="avg_pct", aggfunc="mean")
        .reindex(columns=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    )

    fig_hm = go.Figure(
        go.Heatmap(
            z=heat.values,
            x=heat.columns,
            y=heat.index,
            colorscale="RdYlGn",
            zmid=50,
            colorbar=dict(title="%"),
            hovertemplate="Year=%{y}<br>Month=%{x}<br>Avg=%{z:.1f}%<extra></extra>",
        )
    )
    fig_hm.update_layout(title="Monthly Average Sentiment (0–100)", xaxis_title="Month", yaxis_title="Year")
    st.plotly_chart(fig_hm, use_container_width=True)

    st.session_state["sentiment"]["heatmap"] = {
        "index_years": heat.index.tolist(),
        "columns_months": [str(c) for c in heat.columns],
        "values": heat.fillna(np.nan).values.tolist(),
    }

# ----- Distributions -----
with tab_distrib:
    st.caption("Use the slider to see the effect of thresesholding.")
    thres_local = st.slider("Histogram threseshold |compound| ≥", 0.00, 0.20, float(thres), 0.01, key="thres_hist")
    df_thres = df.loc[df["compound"].abs() >= thres_local].copy()

    cA, cB = st.columns(2)
    cA.plotly_chart(px.histogram(df, x="compound", nbins=60, title="Compound (all)"), use_container_width=True)
    cB.plotly_chart(
        px.histogram(df_thres, x="compound", nbins=60, title=f"Compound (|compound| ≥ {thres_local:.2f})"),
        use_container_width=True,
    )

    c1, c2, c3 = st.columns(3)
    for name, holder in zip(["pos", "neu", "neg"], [c1, c2, c3]):
        if name in df.columns:
            holder.plotly_chart(
                px.histogram(df_thres, x=name, nbins=50, title=f"Distribution: {name}"),
                use_container_width=True,
            )

# ----- Confusion -----
with tab_cls:
    st.subheader("Confusion Matrix & classif repo")
    repo_path = "week8/news_figures/classif_repo.txt"
    if os.path.exists(repo_path):
        with open(repo_path, "r", encoding="utf-8", errors="ignore") as f:
            st.text(f.read())
    else:
        st.info("No classif repo found yet.")

# ----- Investment View -----
with tab_invest:
    avg_last7 = float(daily.tail(7)["avg_pct"].mean()) if len(daily) else 50.0
    ema_now = float(daily["ema"].iloc[-1]) if len(daily) else 50.0
    ema_prev = float(daily["ema"].iloc[-8]) if len(daily) >= 8 else float(daily["ema"].iloc[0]) if len(daily) else 50.0
    ema_slope = ema_now - ema_prev

    def map_to_gauge(v: float) -> str:
        if v < 25: return "Extreme Fear"
        if v < 45: return "Fear"
        if v <= 55: return "Neutral"
        if v <= 75: return "Greed"
        return "Extreme Greed"

    current_regime = map_to_gauge(avg_last7)

    ema_component = np.clip((ema_now - 50) * 1.2 + 50, 0, 100)
    slope_component = np.clip(50 + 4 * ema_slope, 0, 100)
    risk_metric = 0.6 * avg_last7 + 0.3 * ema_component + 0.1 * slope_component

    def risk_to_allocation(score: float) -> dict:
        if score < 35: return {"Defensive": 0.70, "Core": 0.25, "Risk-On": 0.05}
        if score < 50: return {"Defensive": 0.50, "Core": 0.40, "Risk-On": 0.10}
        if score < 65: return {"Defensive": 0.30, "Core": 0.55, "Risk-On": 0.15}
        if score < 80: return {"Defensive": 0.15, "Core": 0.55, "Risk-On": 0.30}
        return {"Defensive": 0.10, "Core": 0.45, "Risk-On": 0.45}

    allocation = risk_to_allocation(risk_metric)
    st.session_state["sentiment"].update({"allocation": allocation})

    def get_weight(mapping, key: str, default: float = 0.0) -> float:
        if isinstance(mapping, dict):
            val = mapping.get(key, default)
            if isinstance(val, (int, float, np.floating)): return float(val)
        return float(default)

    risk_on_wt = get_weight(allocation, "Risk-On", 0.0)

    st.markdown(
        f"**current_regime:** `{current_regime}` • **Risk score:** `{risk_metric:.1f}` → "
        f"Defensive `{allocation['Defensive']*100:.0f}%`, "
        f"Core `{allocation['Core']*100:.0f}%`, "
        f"Risk-On `{allocation['Risk-On']*100:.0f}%`"
    )

    st.divider()
    st.subheader("Risk-On Sleeve (from Momentum)")

    momo = st.session_state.get("momentum", {})
    model_curves = momo.get("portfolio_model_curves", {})
    top_model = momo.get("top_model")
    top_curve = momo.get("top_curve")
    top_sharpe = momo.get("top_sharpe")
    sharpe_scores = momo.get("sharpe_ratios", {}) or {}

    if not model_curves or top_curve is None or getattr(top_curve, "empty", True):
        st.info("No Momentum results in session. Go to the Momentum page and run a backtest first.")
    else:
        use_xsltr = False
        if current_regime in {"Greed", "Extreme Greed"} and "XSLTR-current_regime" in model_curves:
            use_xsltr = True

        if use_xsltr:
            ro_name = "XSLTR-current_regime"
            ro_curve = model_curves["XSLTR-current_regime"].copy()
        else:
            ro_name = top_model or "top Model"
            ro_curve = top_curve.copy()

        ro_curve["date"] = pd.to_datetime(ro_curve["date"], errors="coerce")
        ro_curve = ro_curve.dropna(subset=["date"])

        fig_ro = go.Figure()
        fig_ro.add_trace(
            go.Scatter(
                x=ro_curve["date"],
                y=ro_curve["portfolio_value"],
                mode="lines",
                name=f"{ro_name} (Sharpe={sharpe_scores.get(ro_name, float('nan')):.2f})",
                hovertemplate="%{x|%Y-%m-%d}<br>Value=%{y:.2f}<extra></extra>",
            )
        )

        if risk_on_wt > 0:
            fig_ro.add_trace(
                go.Scatter(
                    x=ro_curve["date"],
                    y=ro_curve["portfolio_value"] * risk_on_wt,
                    mode="lines",
                    name=f"{ro_name} × Risk-On weight ({risk_on_wt:.0%})",
                    line=dict(dash="dash"),
                )
            )

        if "XSLTR-current_regime" in model_curves and ro_name != "XSLTR-current_regime":
            xr = model_curves["XSLTR-current_regime"].copy()
            xr["date"] = pd.to_datetime(xr["date"], errors="coerce")
            xr = xr.dropna(subset=["date"])
            fig_ro.add_trace(
                go.Scatter(
                    x=xr["date"],
                    y=xr["portfolio_value"],
                    mode="lines",
                    name=f"XSLTR-current_regime (Sharpe={sharpe_scores.get('XSLTR-current_regime', float('nan')):.2f})",
                    opacity=0.75,
                )
            )

        fig_ro.update_layout(
            title=f"Risk-On Model Performance — {ro_name}",
            xaxis_title="Date",
            yaxis_title="Portfolio Value",
            hovermode="x unified",
            legend_title="Models",
        )
        st.plotly_chart(fig_ro, use_container_width=True)

        st.caption(
            f"current_regime: {current_regime} • Risk-On weight = {risk_on_wt:.0%} • "
            f"top-Sharpe baseline = {top_model} (Sharpe={top_sharpe:.2f})"
        )
