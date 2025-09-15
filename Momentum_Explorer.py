import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import traceback

# Project imports
from pages.cores.crypto_pipeline import build_week_dirs
from pages.cores.runners import multi_symbol_rotation_by_model
from pages.cores.reader import getSymbolToDf
from pages.cores.commons import FEATURES, MODELS
from pages.cores import run_crypto_pipeline

import plotly.graph_objects as go

st.set_page_config(page_title="Momentum Explorer", layout="wide")
st.title("Momentum Explorer Dashboard for Crypto")


m_labels = {
    "ridge": "Ridge",
    "ols": "OLS",
    "elasticnet": "ElasticNet",
}
f_labels = {
    "momentum": "Momentum",
    "rsi": "RSI",
    "volatility": "Volatility",
    "volume_ratio": "Volume Ratio",
    
}


@st.cache_data(show_spinner=False)
def download_data():
    return run_crypto_pipeline.main()

def load_data():
    WEEK_FOLDER = "week5_crypto"
    data_dir = build_week_dirs(WEEK_FOLDER)           
    csv_path = str(Path(data_dir) / "stage_1_crypto_data.csv")
    return getSymbolToDf(path=csv_path, threshold=100)

def _build_price_panel(symbol_to_df: dict, symbols: List[str]) -> pd.DataFrame:
    """Return wide price panel PX[date x symbol] from symbol_to_df."""
    panel = {}
    for sym in symbols:
        df = symbol_to_df.get(sym)
        if df is None or "date" not in df.columns:
            continue
        dfx = df.copy()
        dfx["date"] = pd.to_datetime(dfx["date"], errors="coerce")
        dfx = dfx.dropna(subset=["date"]).sort_values("date")
        price_colmnmn = _pick_price_colmnmn(dfx)
        if not price_colmnmn:
            continue
        s = dfx.set_index("date")[price_colmnmn].astype(float).rename(sym)
        panel[sym] = s
    if not panel:
        raise ValueError("No usable price columns found for selected symbols.")
    PX = pd.concat(panel.values(), axis=1).dropna(how="any")
    return PX


def _pick_price_colmnmn(df: pd.DataFrame) -> Optional[str]:
    for c in ["close", "price", "Close", "Price", "adj_close", "Adj Close"]:
        if c in df.columns:
            return c
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return num_cols[-1] if num_cols else None

def _get_features(px: pd.Series) -> pd.DataFrame:
    r1 = px.pct_change(1)
    r5 = px.pct_change(5)
    r10 = px.pct_change(10)
    r20 = px.pct_change(20)
    vol20 = r1.rolling(20).std()
    mom20 = px.pct_change(20)
    return pd.DataFrame({"r1": r1, "r5": r5, "r10": r10, "r20": r20, "vol20": vol20, "mom20": mom20})

def _cs_zscore(df_xs: pd.DataFrame) -> pd.DataFrame:
    return (df_xs - df_xs.mean()) / (df_xs.std(ddof=0) + 1e-9)

def _sentiment_Current_regime_from_session(ss) -> str:
    sig = ss.get("sentiment", {}).get("signals", {}) if isinstance(ss.get("sentiment", {}), dict) else {}
    last7 = sig.get("last7_avg_pct"); ema = sig.get("ema_now")
    if last7 is None: return "neutral"
    if last7 < 45 or (ema is not None and ema < 50): return "fear"
    if last7 > 75: return "extreme_greed"
    if last7 > 55: return "greed"
    return "neutral"

def xsltr_Current_regime_backtest(
    symbol_to_df: dict,
    symbols: list[str],
    pred_horizon: int = 5,
    initial_capital: float = 10_000.0,
    rebalance_days: int = 5,                 
    rebalance_freq: Optional[str] = None,    
    lookbck_days: int = 120,
    buffer: int = 40,
    session_state=None,
) -> pd.DataFrame:
   
    from sklearn.linear_model import Ridge

    PX = _build_price_panel(symbol_to_df, symbols)
    feat = {sym: _get_features(PX[sym]) for sym in PX.columns}
    F = pd.concat({sym: feat[sym] for sym in PX.columns}, axis=1)

    Current_regime = _sentiment_Current_regime_from_session(session_state or {})
    if Current_regime == "fear":               topk, exposure = 1, 0.4
    elif Current_regime == "greed":            topk, exposure = 5, 1.0
    elif Current_regime == "extreme_greed":    topk, exposure = 6, 1.0
    else:                              topk, exposure = 3, 0.7

    dates = PX.index.to_list()
    if len(PX) < (lookbck_days + buffer + pred_horizon + 2):
        raise ValueError("Not enough history for XSLTR (fetch more symbols).")

    Present_val = initial_capital
    curve = []
    ridge = Ridge(alpha=ridge_alpha, fit_intercept=True)

    start_idx = lookbck_days + buffer
    end_idx = len(dates) - pred_horizon - 1

    if rebalance_freq:
        sched = pd.date_range(dates[start_idx], dates[end_idx], freq=rebalance_freq)
        rebal_dates = [d for d in sched if d in PX.index]
        rebal_indices = [PX.index.get_loc(d) for d in rebal_dates]
    else:
        rebal_indices = list(range(start_idx, end_idx, rebalance_days))

    for t_idx in rebal_indices:
        if t_idx + pred_horizon >= len(dates):
            continue
        t0 = dates[t_idx]
        t1 = dates[t_idx + pred_horizon]

        # Train panel
        rows_X, rows_y = [], []
        for d_idx in range(t_idx - lookbck_days, t_idx):
            d = dates[d_idx]
            d1_idx = d_idx + pred_horizon
            if d1_idx >= len(dates):
                continue
            d1 = dates[d1_idx]

            xs_rows, syms_ok = [], []
            for sym in PX.columns:
                row_all = F.get(sym)
                if row_all is None or d not in row_all.index:
                    continue
                rowv = row_all.loc[d][["r1", "r5", "r10", "r20", "vol20", "mom20"]]
                if rowv.isna().any():
                    continue
                xs_rows.append(rowv.values.astype(float)); syms_ok.append(sym)

            if len(xs_rows) < 3:
                continue

            Xd = pd.DataFrame(xs_rows, index=syms_ok, columns=["r1", "r5", "r10", "r20", "vol20", "mom20"])
            Xd = _cs_zscore(Xd)

            keep, yvals = [], []
            for sym in Xd.index:
                p0 = PX.loc[d, sym]; p1 = PX.loc[d1, sym]
                r = p1 / p0 - 1.0
                if np.isfinite(r):
                    keep.append(sym); yvals.append(r)
            if len(keep) < 3:
                continue

            Xd = Xd.loc[keep]
            rows_X.append(Xd.values)
            rows_y.append(np.array(yvals))

        if not rows_X:
            curve.append((t1, Present_val)); continue

        X_train = np.vstack(rows_X); y_train = np.concatenate(rows_y)
        mask = np.isfinite(X_train).all(axis=1) & np.isfinite(y_train)
        X_train = X_train[mask]; y_train = y_train[mask]
        if len(X_train) < 30:
            curve.append((t1, Present_val)); continue

        ridge.fit(X_train, y_train)

        # Score at t0
        feat_rows = {}
        for sym in PX.columns:
            row_all = F.get(sym)
            if row_all is None or t0 not in row_all.index:
                continue
            row = row_all.loc[t0][["r1", "r5", "r10", "r20", "vol20", "mom20"]]
            if row.isna().any():
                continue
            feat_rows[sym] = row.values.astype(float)

        if len(feat_rows) < 3:
            curve.append((t1, Present_val)); continue

        X0 = pd.DataFrame(feat_rows).T
        X0.columns = ["r1", "r5", "r10", "r20", "vol20", "mom20"]
        X0 = _cs_zscore(X0)

        scores = ridge.predict(X0.values)
        rank = pd.Series(scores, index=X0.index).sort_values(ascending=False)
        k = min(topk, len(rank))
        if k == 0:
            curve.append((t1, Present_val)); continue
        picks = list(rank.index[:k])

        # realising equal-weight period return
        rets = []
        for sym in picks:
            p0 = PX.loc[t0, sym]; p1 = PX.loc[t1, sym]
            r = p1 / p0 - 1.0
            if np.isfinite(r): rets.append(r)
        if not rets:
            curve.append((t1, Present_val)); continue

        Present_val *= (1.0 + float(np.mean(rets)))
        curve.append((t1, Present_val))

    out = pd.DataFrame(curve, columns=["date", "portfolio_value"]).dropna()
    return out


def _rebalance_indices_for_panel(PX: pd.DataFrame, lookbck_days: int, rebalance_freq: str) -> List[int]:
    dates = PX.index.to_list()
    start_idx = lookbck_days
    end_idx = len(dates) - 1
    sched = pd.date_range(dates[start_idx], dates[end_idx], freq=rebalance_freq)
    rebal_dates = [d for d in sched if d in PX.index]
    rebal_indices = [PX.index.get_loc(d) for d in rebal_dates]
    return [i for i in rebal_indices if i < end_idx]

def _safe_pinv(mat: np.ndarray) -> np.ndarray:
    return np.linalg.pinv(mat + 1e-10 * np.eye(mat.shape[0]))

def _weights_MinVar(Sigma: np.ndarray, enforce_nonneg: bool = True) -> np.ndarray:
    n = Sigma.shape[0]
    invS = _safe_pinv(Sigma)
    ones = np.ones(n)
    w = invS @ ones
    denom = float(ones @ w)
    if denom <= 0:
        w = ones / n
    else:
        w = w / denom
    if enforce_nonneg:
        w = np.maximum(w, 0)
        s = w.sum()
        w = (w / s) if s > 0 else np.ones(n) / n
    return w

def _weights_meanvar(Sigma: np.ndarray, mu: np.ndarray, enforce_nonneg: bool = True, cap: float = 0.5) -> np.ndarray:
    invS = _safe_pinv(Sigma)
    w = invS @ mu
    w = np.maximum(w, 0) if enforce_nonneg else w
    if enforce_nonneg:
        w = np.minimum(w, cap)
    s = w.sum()
    w = (w / s) if s > 0 else np.ones_like(w) / len(w)
    return w

def _weights_risk_parity(vol: np.ndarray) -> np.ndarray:
    inv_vol = 1.0 / np.maximum(vol, 1e-12)
    w = inv_vol / inv_vol.sum()
    return w

def _cumret_backtest(
    symbol_to_df: dict,
    symbols: List[str],
    initial_capital: float,
    rebalance_freq: str,
    lookbck_days: int,
    method: str,  
) -> pd.DataFrame:
    
    PX = _build_price_panel(symbol_to_df, symbols)
    ret = PX.pct_change().dropna()

    idxs = _rebalance_indices_for_panel(PX, lookbck_days, rebalance_freq)
    if not idxs:
        raise ValueError("No rebalancing points available. Try shorter frequency or longer history.")

    Present_val = initial_capital
    curve = []

    for j, t_idx in enumerate(idxs):
        t0 = PX.index[t_idx]
        if j + 1 < len(idxs):
            t1_idx = idxs[j + 1]
        else:
            t1_idx = len(PX.index) - 1
        t1 = PX.index[t1_idx]
        if t1_idx <= t_idx:
            continue

        window = ret.iloc[max(0, t_idx - lookbck_days): t_idx]
        if window.shape[0] < max(20, lookbck_days // 4):
            curve.append((t1, Present_val)); continue

        mu = window.mean().values
        Sigma = np.cov(window.values.T)
        vol = window.std().values

        if method == "MinVar":
            w = _weights_MinVar(Sigma, enforce_nonneg=True)
        elif method == "meanvar":
            w = _weights_meanvar(Sigma, mu, enforce_nonneg=True, cap=0.5)
        elif method == "riskparity":
            w = _weights_risk_parity(vol)
        else:
            raise ValueError(f"Unknown method {method}")

        p0 = PX.iloc[t_idx].values
        p1 = PX.iloc[t1_idx].values
        period_ret = (p1 / p0 - 1.0)
        period_ret = np.where(np.isfinite(period_ret), period_ret, 0.0)
        port_r = float(np.dot(w, period_ret))
        Present_val *= (1.0 + port_r)
        curve.append((t1, Present_val))

    return pd.DataFrame(curve, columns=["date", "portfolio_value"]).dropna()


def _annualize_factor_from_index(idx: pd.DatetimeIndex) -> float:
    
    if len(idx) < 2:
        return np.nan
    gaps = np.diff(idx.values).astype("timedelta64[D]").astype(int)
    step_days = int(np.median(gaps)) if len(gaps) else 1
    step_days = max(step_days, 1)
    annual_steps = 252 / step_days
    return float(np.sqrt(annual_steps)), float(step_days)

def compute_curve_metrics(pf_df: pd.DataFrame) -> Tuple[float, float, float]:
    
    MIN_STEPS_FOR_SHARPE = 5           
    MIN_DAYS_FOR_CAGR = 90             
    MIN_POINTS_FOR_MDD = 3              

    if pf_df is None or pf_df.empty or "date" not in pf_df.columns:
        return (np.nan, np.nan, np.nan)

    s = pf_df.copy()
    s["date"] = pd.to_datetime(s["date"], errors="coerce")
    s = s.dropna(subset=["date"])
    if s.empty:
        return (np.nan, np.nan, np.nan)

    Present_val = s.set_index("date")["portfolio_value"].dropna()
    if Present_val.size < 2:
        return (np.nan, np.nan, np.nan)

    # Sharpe
    step_ret = Present_val.pct_change().dropna()
    if len(step_ret) >= MIN_STEPS_FOR_SHARPE and step_ret.std() > 0:
        ann_factor, _ = _annualize_factor_from_index(Present_val.index)
        if not np.isfinite(ann_factor):
            ann_factor = np.sqrt(252.0)
        sharpe = (step_ret.mean() / (step_ret.std() + 1e-12)) * ann_factor
    else:
        sharpe = np.nan

    # CAGR
    total_days = (Present_val.index[-1] - Present_val.index[0]).days
    if total_days >= MIN_DAYS_FOR_CAGR and Present_val.iloc[0] > 0:
        cagr = (Present_val.iloc[-1] / Present_val.iloc[0]) ** (365.25 / total_days) - 1.0
    else:
        cagr = np.nan

    # Max Drawdown
    if len(Present_val) >= MIN_POINTS_FOR_MDD:
        running_max = Present_val.cummax()
        dd = Present_val / running_max - 1.0
        mdd = float(dd.min()) if not dd.empty else np.nan
    else:
        mdd = np.nan

    return (float(sharpe), float(cagr), mdd)

symbol_to_df = load_data()


# Sidebar

with st.sidebar:
    st.header("Parameters")
    available_symbols = list(symbol_to_df.keys())
    available_models = MODELS
    available_features = FEATURES

    symbols = st.multiselect(
        "Select Symbols to Backtest:",
        options=available_symbols,
        default=available_symbols[:3],
    )
    st.session_state.selection_Symbols = list(symbols)

    selected_models = st.multiselect(
        "Select Models to Compare:",
        options=available_models,
        default=["ridge", "ols", "elasticnet"],
        format_func=lambda x: m_labels.get(x, x.replace("_", " ").title()),
    )
    selected_features = st.multiselect(
        "Select Features for the Models:",
        options=available_features,
        default=available_features,
        format_func=lambda x: f_labels.get(x, x.replace("_", " ").title()),
    )

    train_size = st.slider("Training Data Size (in Days):", 30, 250, 120)
    pred_horizon = st.slider("Prediction pred_horizon (in Days):", 1, 20, 5)
    threshold = st.number_input("Prediction Threshold:", 0.0, 1.0, 0.0, step=0.01)
    initial_capital = st.number_input("Initial Capital:", 1000, 1_000_000, 10_000, step=1000)

    rebalance_freq = st.selectbox(
        "Rebalancing Frequency:",
        ["D", "2D", "3D", "W-FRI", "M"],
        index=0,
        help="D = daily, 2D/3D = every 2/3 days, W-FRI = weekly on Fridays, M = month-end."
    )

    cum_strats_to_add = st.multiselect(
        "Add Cumulative-Returns Strategies:",
        ["Min Variance", "Mean Variance", "Risk Parity"],
        default=["Min Variance", "Mean Variance", "Risk Parity"]
    )

    
    run_button = st.button("Run Backtest & Plot")

def _effective_symbols_and_train_size(symbol_to_df: dict, symbols: list[str], train_size: int, pred_h: int):
    ok = []
    min_len = 10**9
    for s in symbols:
        df = symbol_to_df[s]
        if "date" not in df.columns:
            continue
        n = len(df)
        if n <= pred_h + 10:
            continue
        ok.append(s)
        min_len = min(min_len, n)
    if not ok:
        return [], train_size
    max_train = max(30, min_len - pred_h - 5)
    eff_train = int(min(train_size, max_train))
    return ok, eff_train

def _safe_core_backtest(symbol_to_df: dict, symbols: list[str], model_list, features, train_size: int, pred_horizon: int, threshold: float, initial_capital: float):
    portfolio_outputs = {}
    sharpe_ratio = {}
    try:
        return multi_symbol_rotation_by_model(
            symbol_to_df={s: symbol_to_df[s] for s in symbols},
            symbols=symbols,
            model_name_list=model_list,
            features=features,
            train_size=train_size,
            pred_horizon=pred_horizon,
            threshold=threshold,
            initial_capital=initial_capital,
        )
    except Exception as e:
        msg = str(e)
        if "Length of values" not in msg:
            raise
        for m in model_list:
            try:
                pc, sr = multi_symbol_rotation_by_model(
                    symbol_to_df={s: symbol_to_df[s] for s in symbols},
                    symbols=symbols,
                    model_name_list=[m],
                    features=features,
                    train_size=train_size,
                    pred_horizon=pred_horizon,
                    threshold=threshold,
                    initial_capital=initial_capital,
                )
                portfolio_outputs.update(pc)
                sharpe_ratio.update(sr)
            except Exception as ee:
                st.warning(f"Skipped model '{m}': {ee}")
        if not portfolio_outputs:
            raise ValueError("All selected models failed to produce predictions (check features/train size).")
        return portfolio_outputs, sharpe_ratio

if run_button:
    if not symbols or not selected_models or not selected_features:
        st.warning("Please select at least one symbol, model, and feature.")
    else:
        st.subheader("Backtest Results")

        try:
            
            symbols, eff_train_size = _effective_symbols_and_train_size(
                symbol_to_df, symbols, train_size, pred_horizon
            )
            if not symbols:
                st.error("No symbols have enough history for the chosen pred_horizon. Pick others or reduce the pred_horizon.")
                st.stop()

            
            try:
                portfolio_outputs, sharpe_ratio = _safe_core_backtest(
                    symbol_to_df, symbols, selected_models, selected_features,
                    eff_train_size, pred_horizon, threshold, initial_capital
                )
            except Exception:
                st.error("Core backtest failed.")
                st.code(traceback.format_exc())
                raise

            
            

            
            try:
                if "Min Variance" in cum_strats_to_add:
                    mv_curve = _cumret_backtest(
                        symbol_to_df, symbols, initial_capital,
                        rebalance_freq, eff_train_size, method="MinVar"
                    )
                    portfolio_outputs["Min Variance"] = mv_curve
                if "Mean Variance" in cum_strats_to_add:
                    meanv_curve = _cumret_backtest(
                        symbol_to_df, symbols, initial_capital,
                        rebalance_freq, eff_train_size, method="meanvar"
                    )
                    portfolio_outputs["Mean Variance"] = meanv_curve
                if "Risk Parity" in cum_strats_to_add:
                    rp_curve = _cumret_backtest(
                        symbol_to_df, symbols, initial_capital,
                        rebalance_freq, eff_train_size, method="riskparity"
                    )
                    portfolio_outputs["Risk Parity"] = rp_curve
            except Exception as e:
                st.warning(f"Cumulative strategy failed: {e}")

            
            metrics: Dict[str, Dict[str, float]] = {}
            for model_name, pf_df in portfolio_outputs.items():
                sh, cg, mdd = compute_curve_metrics(pf_df)
                metrics[model_name] = {
                    "Sharpe Ratio": float(sh) if np.isfinite(sh) else np.nan,
                    "CAGR (%)": float(cg * 100.0) if np.isfinite(cg) else np.nan,
                    "Max Drawdown (%)": float(mdd * 100.0) if np.isfinite(mdd) else np.nan,
                }
                sharpe_ratio[model_name] = metrics[model_name]["Sharpe Ratio"]

            
            st.session_state["momentum"] = {
                "portfolio_outputs": portfolio_outputs,
                "metrics": metrics,
                "sharpe_ratio": sharpe_ratio,
                "params": {
                    "symbols": symbols,
                    "models": selected_models,
                    "features": selected_features,
                    "train_size": eff_train_size,
                    "pred_horizon": pred_horizon,
                    "threshold": threshold,
                    "initial_capital": initial_capital,
                    "rebalance_freq": rebalance_freq,
                    "cum_strats": cum_strats_to_add,
                },
            }

            
            st.subheader("Model outputs — Ridge, OLS, ElasticNet")
            fig_models = go.Figure()
            desired_models = {"ridge", "ols", "elasticnet"}
            for name, pf_df in portfolio_outputs.items():
                lname = name.lower()
                if lname not in desired_models:
                    continue
                if pf_df is None or len(pf_df) == 0 or "date" not in pf_df.columns:
                    continue
                df_plot = pf_df.copy()
                df_plot["date"] = pd.to_datetime(df_plot["date"], errors="coerce")
                df_plot = df_plot.dropna(subset=["date"])
                if df_plot.empty:
                    continue
                sh = metrics.get(name, {}).get("Sharpe Ratio", np.nan)
                pretty_label = m_labels.get(lname, name.replace("_", " ").title())
                fig_models.add_trace(
                    go.Scattergl(
                        x=df_plot["date"], y=df_plot["portfolio_value"],
                        mode="lines",
                        name=f"{pretty_label} (Sharpe={sh:.2f})",  # <-- R/O/E capitalized
                        hovertemplate="<b>%{fullData.name}</b><br>Date=%{x|%Y-%m-%d}<br>Value=%{y:.2f}<extra></extra>",
                    )
                )

            if len(fig_models.data) == 0:
                st.info("No Ridge/OLS/ElasticNet outputs available to plot.")
            else:
                fig_models.update_layout(
                    title="Model Comparison (Ridge / OLS / ElasticNet)",
                    xaxis_title="Date",
                    yaxis_title="Portfolio Value (Log Scale)",
                    hovermode="x unified",
                    legend_title="Models",
                )
                fig_models.update_yaxes(type="log")
                st.plotly_chart(fig_models, use_container_width=True)

            
            st.subheader("Portfolio outputs — Min Variance, Mean Variance, Risk Parity")
            fig_ports = go.Figure()
            for strat_name in ["Min Variance", "Mean Variance", "Risk Parity"]:
                pf_df = portfolio_outputs.get(strat_name)
                if pf_df is None or len(pf_df) == 0:
                    continue
                df_plot = pf_df.copy()
                if "date" not in df_plot.columns:
                    continue
                df_plot["date"] = pd.to_datetime(df_plot["date"], errors="coerce")
                df_plot = df_plot.dropna(subset=["date"])
                if df_plot.empty:
                    continue
                sh = metrics.get(strat_name, {}).get("Sharpe Ratio", np.nan)
                fig_ports.add_trace(
                    go.Scattergl(
                        x=df_plot["date"], y=df_plot["portfolio_value"],
                        mode="lines",
                        name=f"{strat_name} (Sharpe={sh:.2f})",
                        hovertemplate="<b>%{fullData.name}</b><br>Date=%{x|%Y-%m-%d}<br>Value=%{y:.2f}<extra></extra>",
                    )
                )

            if len(fig_ports.data) == 0:
                st.info("No portfolio strategy outputs available to plot.")
            else:
                fig_ports.update_layout(
                    title="Portfolio Strategies (Min Variance / Mean Variance / Risk Parity)",
                    xaxis_title="Date",
                    yaxis_title="Portfolio Value (Log Scale)",
                    hovermode="x unified",
                    legend_title="Strategies",
                )
                fig_ports.update_yaxes(type="log")
                st.plotly_chart(fig_ports, use_container_width=True)

            
            st.subheader("Performance Metrics")
            if st.session_state["momentum"]["metrics"]:
                metrics_df = pd.DataFrame(st.session_state["momentum"]["metrics"]).T[
                    ["Sharpe Ratio", "CAGR (%)", "Max Drawdown (%)"]
                ].replace([np.inf, -np.inf], np.nan) \
                 .sort_values(by="Sharpe Ratio", ascending=False)

                st.dataframe(
                    metrics_df.style.format({
                        "Sharpe Ratio": lambda v: "—" if pd.isna(v) else f"{v:.2f}",
                        "CAGR (%)": lambda v: "—" if pd.isna(v) else f"{v:.2f}",
                        "Max Drawdown (%)": lambda v: "—" if pd.isna(v) else f"{v:.2f}",
                    })
                )
                if "metrics_df" not in st.session_state:
                    st.session_state.metrics_df = metrics_df
            else:
                st.info("No metrics computed.")

            
            st.subheader("Monthly Returns Heatmap (%)")
            monthly_returns_data = {}
            for model_name, pf_df in st.session_state["momentum"]["portfolio_outputs"].items():
                if pf_df is None or len(pf_df) <= 30:
                    continue
                pf_df = pf_df.copy()
                if "date" not in pf_df.columns:
                    continue
                df_dt = pd.to_datetime(pf_df["date"], errors="coerce")
                pf_df = pf_df.loc[df_dt.notna()]
                pf_df["date"] = pd.to_datetime(pf_df["date"])
                if pf_df.empty:
                    continue
                monthly_pf = pf_df.set_index("date").resample("M")["portfolio_value"].last()
                mr = monthly_pf.pct_change() * 100
                if not mr.empty:
                    monthly_returns_data[model_name] = mr

            if monthly_returns_data:
                monthly_returns_df = pd.DataFrame(monthly_returns_data)
                monthly_returns_df.index = monthly_returns_df.index.strftime("%Y-%m")
                monthly_returns_df.index.name = "Date"

                if "monthly_returns_df" not in st.session_state:
                    st.session_state.monthly_returns_df = monthly_returns_df

                z = monthly_returns_df.T.values
                x = monthly_returns_df.index.tolist()
                y = monthly_returns_df.columns.tolist()
                height_px = max(350, 40 * len(y))

                heatmap = go.Figure(
                    data=go.Heatmap(
                        z=z, x=x, y=y, colorscale="RdYlGn", zmid=0,
                        colorbar=dict(title="%"),
                        hovertemplate="Model/Strategy=%{y}<br>Date=%{x}<br>Return=%{z:.2f}%<extra></extra>",
                        zauto=False,
                    )
                )
                heatmap.update_layout(
                    title="Monthly Returns (%)",
                    xaxis_title="Date",
                    yaxis_title="Model/Strategy",
                    height=height_px, margin=dict(l=60, r=20, t=60, b=60),
                )
                st.plotly_chart(heatmap, use_container_width=True)
            else:
                st.info("Not enough data to generate a monthly returns heatmap.")

        except Exception:
            st.error("Backtest failed.")
            st.code(traceback.format_exc())
