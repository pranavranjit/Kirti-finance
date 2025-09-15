# pages/cores/sentiment_pipeline.py
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.metrics import confusion_matrix, classification_report

# --------------------------- NLTK (quiet one-time)
nltk.download("vader_lexicon", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

# --------------------------- logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
tqdm.pandas()

# --------------------------- directories
def _ensure_dir(root: Path, sub: str | Path) -> Path:
    p = root / sub
    p.mkdir(parents=True, exist_ok=True)
    return p

def build_week_dirs(
    base_dir: str | Path | None = None,
    results_folder: str = "week8",
    data_sub: str = "news_clean_data",
    fig_sub: str = "news_figures",
) -> Dict[str, Path]:
    root = Path(base_dir).expanduser().resolve() if base_dir else Path.cwd().resolve()
    res_root = _ensure_dir(root, results_folder)
    return {
        "data_dir": _ensure_dir(res_root, data_sub),
        "fig_dir": _ensure_dir(res_root, fig_sub),
    }

# --------------------------- Stage 1 (download & cache once)
def fetch_news_range(
    api_key: Optional[str],
    start_dt: datetime,
    end_dt: datetime,
    lang: str = "EN",
) -> pd.DataFrame:
    url = "https://data-api.coindesk.com/news/v1/article/list"
    out: list[pd.DataFrame] = []

    while end_dt > start_dt:
        query_ts = int(end_dt.timestamp())
        query_day = end_dt.strftime("%Y-%m-%d")
        logging.info("Requesting articles up to %s (UTC)", query_day)

        try:
            resp = requests.get(f"{url}?lang={lang}&to_ts={query_ts}", timeout=30)
        except Exception as e:
            logging.error("Request failed: %s", e)
            break
        if not resp.ok:
            logging.error("Request failed with status %s", resp.status_code)
            break

        try:
            payload = resp.json()
        except Exception:
            logging.error("Invalid JSON payload.")
            break

        rows = payload.get("Data", [])
        d = pd.DataFrame(rows)
        if d.empty:
            logging.info("No data returned for %s – stopping.", query_day)
            break

        d["date"] = pd.to_datetime(d["PUBLISHED_ON"], unit="s", utc=True)
        out.append(d[d["date"] >= start_dt])

        # step to just before earliest record we received
        end_dt = datetime.utcfromtimestamp(int(d["PUBLISHED_ON"].min()) - 1)

    news = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    logging.info("Fetched %d articles.", len(news))
    return news

def stage1_load_news(
    api_key: Optional[str],
    start_dt: datetime,
    end_dt: datetime,
    data_dir: Path,
    filename: str = "stage_1_news_raw.csv",
) -> pd.DataFrame:
    tic = time.time()
    logging.info("Stage 1 – downloading news …")

    df = fetch_news_range(api_key, start_dt, end_dt)

    drop_cols = [
        "GUID","PUBLISHED_ON_NS","IMAGE_URL","SUBTITLE","AUTHORS","URL","UPVOTES",
        "DOWNVOTES","SCORE","CREATED_ON","UPDATED_ON","SOURCE_DATA","CATEGORY_DATA",
        "STATUS","SOURCE_ID","TYPE"
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    df.columns = df.columns.str.lower()

    if "id" not in df.columns:
        df["id"] = np.arange(len(df))
    other = [c for c in df.columns if c not in ["date", "id"]]
    df = df[["date", "id"] + other]

    if "sentiment" in df.columns:
        df["positive"] = np.where(df["sentiment"].astype(str).str.upper() == "POSITIVE", 1, 0)
        df = df.drop(columns="sentiment")
    else:
        df["positive"] = np.nan

    out = data_dir / filename
    data_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    logging.info("Saved raw news -> %s (%.2fs)", out, time.time() - tic)
    return df

def ensure_stage1_csv_once(
    *,
    api_key: Optional[str],
    start_dt: datetime,
    end_dt: datetime,
    data_dir: Path,
    filename: str = "stage_1_news_raw.csv",
) -> pd.DataFrame:
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / filename
    if path.exists():
        df = pd.read_csv(path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df
    return stage1_load_news(api_key, start_dt, end_dt, data_dir, filename)

# --------------------------- Stage 2 (pure transform; no I/O)
_STOP = set(stopwords.words("english"))
_LEM = WordNetLemmatizer()

def _preprocess_text(txt: str) -> str:
    tokens = word_tokenize(str(txt))
    keep = [t for t in tokens if t.isalpha() and t.lower() not in _STOP]
    lemmas = [_LEM.lemmatize(t.lower()) for t in keep]
    return " ".join(lemmas)

def stage2_add_columns(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    # ensure expected columns exist
    for col in ["title", "body"]:
        if col not in df.columns:
            df[col] = ""

    # normalize date to tz-naive midnight so Grouper works
    df["date"] = pd.to_datetime(df.get("date"), errors="coerce")
    try:
        if df["date"].dt.tz is not None:
            df["date"] = df["date"].dt.tz_convert(None)
    except Exception:
        pass
    df["date"] = df["date"].dt.normalize()

    # preprocessed/body and combined text
    try:
        df["reviewText"] = df["body"].astype(str).progress_apply(_preprocess_text)
    except Exception:
        df["reviewText"] = df["body"].astype(str).apply(_preprocess_text)

    df["all_text"] = df["title"].fillna("").astype(str) + " " + df["body"].fillna("").astype(str)

    # place all_text next to body for readability
    cols = list(df.columns)
    if "all_text" in cols and "body" in cols:
        cols.insert(cols.index("body") + 1, cols.pop(cols.index("all_text")))
        df = df[cols]

    return df

# --------------------------- VADER + crypto tweaks
CRYPTO_VADER_TERMS = {"hodl": 2.6, "bullish": 1.9, "bearish": -1.9, "rekt": -2.8}
PHRASE_TOKENS_VALENCE = {"rug_pull": -3.2, "all_time_high": 2.6, "all_time_low": -2.6}

def get_vader(custom_unigrams=None, phrase_tokens=None) -> SentimentIntensityAnalyzer:
    a = SentimentIntensityAnalyzer()
    if custom_unigrams:
        a.lexicon.update({k.lower(): float(v) for k, v in custom_unigrams.items()})
    if phrase_tokens:
        a.lexicon.update({k.lower(): float(v) for k, v in phrase_tokens.items()})
    return a

_VADER = get_vader(CRYPTO_VADER_TERMS, PHRASE_TOKENS_VALENCE)

def _vader_scores(txt: str) -> pd.Series:
    return pd.Series(_VADER.polarity_scores(str(txt)))

def _add_word_filters(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["n_words"] = out["all_text"].astype(str).str.split().str.len()
    p25 = out.groupby("date")["n_words"].transform(lambda x: np.percentile(x, 25))
    out["below_p25"] = out["n_words"] < p25
    return out

def _rescale(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["compound_pct"] = (out["compound"] + 1.0) * 50.0
    return out

# --------------------------- Stage 3 (sentiment + plots + final CSV)
def stage3_sentiment_and_plots(
    df_clean: pd.DataFrame,
    dirs: Dict[str, Path],
    *,
    thr: float = 0.05,
) -> pd.DataFrame:
    tic = time.time()
    logging.info("Stage 3 – VADER, filters, plots")

    df = df_clean.copy()
    df[["neg", "neu", "pos", "compound"]] = df["all_text"].progress_apply(_vader_scores)

    df_thr   = df.loc[df["compound"].abs() >= thr].copy()
    df_final = _add_word_filters(df_thr).loc[lambda d: ~d["below_p25"]].copy()
    df_final["sentiment"] = np.where(df_final["compound"] >= thr, "positive", "negative")
    df_final = _rescale(df_final)

    _hist_grid(df,       dirs["fig_dir"] / "hist_raw.png",   "Sentiment – raw sample")
    _hist_grid(df_thr,   dirs["fig_dir"] / "hist_thr.png",   f"Sentiment – |compound| ≥ {thr}")
    _hist_grid(df_final, dirs["fig_dir"] / "hist_final.png", "Sentiment – final sample")
    _daily_line(df_final, dirs["fig_dir"] / "daily_avg_sentiment.png")
    _heatmap(df_final,    dirs["fig_dir"] / "monthly_avg_heatmap.png")
    _fear_greed_gauge(df_final, dirs["fig_dir"] / "fear_greed_gauge.png")

    out_csv = dirs["data_dir"] / "clean_news_timeseries.csv"
    df_final.to_csv(out_csv, index=False)
    logging.info("Wrote final time-series -> %s (%.2fs)", out_csv.name, time.time() - tic)
    return df_final

# ---- plot helpers
def _hist_grid(df: pd.DataFrame, fname: Path, title: str) -> None:
    if df.empty: return
    cols = ["compound", "pos", "neg", "neu"]; labs = ["Compound","Positive","Negative","Neutral"]
    clrs = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D"]
    fig, ax = plt.subplots(2,2, figsize=(14,10)); fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)
    for i, (c,l,col) in enumerate(zip(cols,labs,clrs)):
        r,cc = divmod(i,2); vals = pd.to_numeric(df.get(c, pd.Series(dtype=float)), errors="coerce")
        ax[r,cc].hist(vals.dropna(), bins=50, color=col, alpha=0.7, edgecolor="white")
        ax[r,cc].set_title(l); ax[r,cc].grid(alpha=0.3, linewidth=0.5)
    plt.tight_layout(); plt.savefig(fname, dpi=150); plt.close()

def _daily_line(df: pd.DataFrame, fname: Path) -> None:
    if df.empty: return
    daily = df.groupby("date")["compound_pct"].mean().reset_index()
    daily["date"] = pd.to_datetime(daily["date"]); daily = daily.sort_values("date")
    plt.figure(figsize=(15,8))
    plt.plot(daily["date"], daily["compound_pct"], linewidth=1.5, color="#2E86AB")
    plt.fill_between(daily["date"], daily["compound_pct"], color="#2E86AB", alpha=0.3)
    plt.axhline(50, color="red", linestyle="--", alpha=0.7, label="Neutral (50)")
    plt.title("Daily Average Sentiment (0–100)")
    plt.xlabel("Date"); plt.ylabel("Avg Compound (%)")
    plt.grid(alpha=0.3, linewidth=0.5); plt.legend()
    plt.xticks(rotation=45); plt.tight_layout()
    plt.savefig(fname, dpi=150); plt.close()

def _heatmap(df: pd.DataFrame, fname: Path) -> None:
    if df.empty: return
    daily = df.groupby("date")["compound_pct"].mean().reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    daily["year"] = daily["date"].dt.year; daily["month"] = daily["date"].dt.month
    pivot = daily.groupby(["year","month"])["compound_pct"].mean().unstack().sort_index(ascending=False)
    plt.figure(figsize=(12,8))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlBu_r", center=50,
                cbar_kws={"label":"Avg Sentiment (%)"}, linewidths=0.5, linecolor="white")
    plt.title("Monthly Average Sentiment Heat-map")
    plt.xlabel("Month"); plt.ylabel("Year")
    plt.xticks(ticks=np.arange(12)+0.5,
               labels=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
               rotation=0)
    plt.tight_layout(); plt.savefig(fname, dpi=150); plt.close()

def _fear_greed_gauge(df: pd.DataFrame, fname: Path) -> None:
    if df.empty: return
    dmax = pd.to_datetime(max(df["date"]))
    recent = df[pd.to_datetime(df["date"]) >= dmax - pd.Timedelta(days=6)]
    avg = float(recent["compound_pct"].mean())

    fig, ax = plt.subplots(figsize=(12,8), subplot_kw=dict(projection="polar"))
    colors = ["#8B0000","#FF4500","#FFD700","#90EE90","#006400"]; bounds = [0,20,40,60,80,100]
    for i in range(5):
        t0,t1 = np.pi*(bounds[i]/100), np.pi*(bounds[i+1]/100)
        ax.fill_between(np.linspace(t0,t1,20), 0.5, 1, color=colors[i], alpha=0.8)
    for sc in [0,25,50,75,100]:
        ang = np.pi*(sc/100); ax.plot([ang,ang],[0.5,0.55],"k-",lw=1); ax.text(ang,0.6,f"{sc}",ha="center",va="center",fontsize=10)
    needle = np.pi*(avg/100); ax.plot([needle,needle],[0,0.9],"k-",lw=8); ax.plot(needle,0,"ko",ms=15)

    cat,col = _cat(avg)
    ax.text(np.pi/2, 0.2, f"{avg:.0f}", ha="center", va="center", fontsize=60, weight="bold")
    plt.figtext(0.5,0.15,"Last 7-day Average",ha="center",fontsize=13)
    plt.figtext(0.5,0.10,f"Current Status: {cat}",ha="center",fontsize=15,weight="bold",color=col)

    ax.set_ylim(0,1.3); ax.set_xlim(0,np.pi)
    ax.set_theta_zero_location("W"); ax.set_theta_direction(1)
    ax.grid(False); ax.set_rticks([]); ax.set_thetagrids([])
    plt.tight_layout(); plt.savefig(fname, dpi=150); plt.close()

def _cat(v: float) -> tuple[str, str]:
    if v < 20: return "Extreme Fear", "#8B0000"
    if v < 40: return "Fear", "#FF4500"
    if v < 60: return "Neutral", "#FFD700"
    if v < 80: return "Greed", "#90EE90"
    return "Extreme Greed", "#006400"

# --------------------------- Stage 4 (optional)
def stage4_confusion(df_sent: pd.DataFrame, dirs: Dict[str, Path]) -> None:
    if "positive" not in df_sent.columns or df_sent["positive"].isna().all():
        logging.info("Stage 4 – no ground-truth labels -> skipped"); return

    tic = time.time(); logging.info("Stage 4 – confusion matrix")
    df = df_sent.copy(); df["predicted_positive"] = (df["sentiment"] == "positive").astype(int)

    cm = confusion_matrix(df["positive"], df["predicted_positive"])
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Neg","Pos"], yticklabels=["Neg","Pos"])
    plt.title("Confusion Matrix"); plt.ylabel("Actual"); plt.xlabel("Predicted")
    plt.tight_layout(); (dirs["fig_dir"] / "confusion_matrix.png").parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(dirs["fig_dir"] / "confusion_matrix.png", dpi=150); plt.close()

    rep = classification_report(df["positive"], df["predicted_positive"])
    (dirs["fig_dir"] / "classification_report.txt").write_text(rep)
    logging.info("Stage 4 finished (%.2fs)", time.time() - tic)
