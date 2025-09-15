import os
from typing import Any, Dict
from dotenv import load_dotenv
import google.generativeai as genai
import streamlit as st
import pandas as pd
import numpy as np

# -------------------- Setup --------------------
st.set_page_config(page_title="Helper Bot", layout="centered")
st.title("Helper Bot")
st.subheader("Your AI copilot for market momentum and smart trading decisions")



load_dotenv("pages/cores/.env")
API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key="AIzaSyDgg8IQdCMTs-etJTkpS-7bihUsie9ZueI")


def _safe_get(d: Dict, path: str, default=None):

    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def summarize_df(df: pd.DataFrame, max_rows: int = 30) -> str:
   
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return "None"
    head_csv = df.head(min(max_rows, len(df))).to_csv(index=False)
    return f"[HEAD CSV rows={min(max_rows, len(df))}]\n{head_csv}"

def build_context(ss: Dict[str, Any]) -> str:
    
   
  
    moda = ss.get("momentum", {}) if isinstance(ss.get("momentum", {}), dict) else {}
    top_model   = moda.get("top_model")
    top_sharpe  = moda.get("top_sharpe")
    moda_params  = moda.get("params", {})
    symb      = moda_params.get("symb")
    data_features     = moda_params.get("data_features")
    models_used  = moda_params.get("models")
    initial_cap  = moda_params.get("initial_capital")

   
    top_curve   = moda.get("top_curve")
    curve_summary = "None"
    if isinstance(top_curve, pd.DataFrame) and not top_curve.empty and {"date","portfolio_value"}.issubset(top_curve.columns):
        preview = top_curve[["date","portfolio_value"]].tail(50).copy()
        preview["date"] = pd.to_datetime(preview["date"]).dt.strftime("%Y-%m-%d")
        curve_summary = summarize_df(preview, max_rows=50)

 
    senti_metrics = ss.get("senti_metricsment", {}) if isinstance(ss.get("senti_metricsment", {}), dict) else {}
    indicators     = senti_metrics.get("indicators", {})
    alerts     = senti_metrics.get("alerts", {})
    params_sent = senti_metrics.get("params", {})

 
    summary = senti_metrics.get("summary_series", {})
    ov_preview = {
        "last_date": summary.get("date", [])[-1] if summary.get("date") else None,
        "last_avg_pct": (summary.get("avg_pct", [])[-1] if summary.get("avg_pct") else None),
        "last_ema": (summary.get("ema", [])[-1] if summary.get("ema") else None),
        "len_series": len(summary.get("date", [])) if summary.get("date") else 0,
    }

   
    classification_data      = senti_metrics.get("classification_dataication", {})
    repo_summary  = classification_data.get("repo_summary")
    repo_df    = classification_data.get("repo_df")  
    repo_tbl   = summarize_df(repo_df) if isinstance(repo_df, pd.DataFrame) else (repo_summary or "None")

    heatmap      = senti_metrics.get("heatmap", {})
    heat_years   = heatmap.get("index_years")
    heat_months  = heatmap.get("columns_months")


    dist      = senti_metrics.get("distributions", {})
    threshold_local    = dist.get("threshold_local")
    df_threshold_count = dist.get("df_threshold_count")

    cntxt = f"""
[MOMENTUM]
top_model: {top_model}
top_sharpe: {top_sharpe}
symb: {symb}
models_used: {models_used}
data_features_used: {data_features}
initial_capital: {initial_cap}
top_model_curve_summary: {curve_summary}

[senti_metricsMENT]
params: {params_sent}
indicators: {indicators}
alerts: {alerts}
summary_series_preview: {ov_preview}
heatmap_years: {heat_years}
heatmap_months: {heat_months}
distributions: threshold_used={threshold_local}, df_threshold_rows={df_threshold_count}
allocation_suggestion: {ss.get("alloc", "None")}

[classification_dataICATION]
repo_table_or_text:
{repo_tbl}
"""
    return cntxt.strip()

def system_guidelines() -> str:
    return (
        "You are a smart, helpful and cautious financial analysis assistant. "
       "Use momentum & sentiment metrics to discuss allocations and trade posture to suggest whether or not to invest in the symb mentioned in the context.  "
        "Be concise, avoid jargon when possible, be clear and simple. "
        "Does not provide guarantees or personalized legal/financial advice. "
        "When suggesting actions, explain the rationale from the context (e.g., Sharpe, regime, gauge). "
        "Prefer buckets like Defensive/Core/Risk-On over exact weights. "
    )

def build_model():
    
    return genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        system_guidelines=system_guidelines()
    )

def start_chat():
    
    if "gemini_chat" not in st.session_state:
        model = build_model()
       
        hist = []
        if "messages" in st.session_state:
            for m in st.session_state.messages:
                role = "user" if m.get("role") == "user" else "model"
                content = m.get("content", "")
                if content:
                    hist.append({"role": role, "parts": [content]})

        chat = model.start_chat(history=hist)
        st.session_state.gemini_chat = chat
    return st.session_state.gemini_chat

def friendly_wrap(raw_text: str) -> str:
    return (
        "Great question! \n\n"
        f"{raw_text.strip()}\n\n"
        "_Note: This is educational information, not financial advice. Backtest before deploying._"
    )


if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm your fintech copilot. How can I help?"}
    ]

for msg in st.session_state.messages:
    with st.chat_message("user" if msg["role"] == "user" else "assistant"):
        st.write(msg["content"])

prompt = st.chat_input("Ask me about allocations, model performance, or what the signals imply…")

if prompt:
 
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.write("🤔 Thinking...")

        try:
            
            cntxt = build_context(dict(st.session_state))

            
            chat = start_chat()
            
            response = chat.send_message([
                {"text": f"[CONTEXT]\n{cntxt}"},
                {"text": f"[USER QUESTION]\n{prompt}"}
            ])

            answer = response.text or "(No answer returned.)"
            friendly_answer = friendly_wrap(answer)

        except Exception as e:
            friendly_answer = f"Sorry, I hit an error: {e}"

        placeholder.write(friendly_answer)
        st.session_state.messages.append({"role": "assistant", "content": friendly_answer})

    st.rerun()
