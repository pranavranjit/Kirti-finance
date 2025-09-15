# pages/cores/run_sentiment_pipeline.py
"""
Run stages 1–4 of the CoinDesk sentiment pipeline.

Behavior:
- If week8/news_clean_data/stage_1_news_raw.csv exists, reuse it.
- Otherwise, download once (with LOOKBACK_DAYS window), save it, and reuse thereafter.
- Produce week8/news_clean_data/clean_news_timeseries.csv and plots in week8/news_figures/.
"""
from pathlib import Path
import os
from datetime import datetime, timedelta

# headless matplotlib
import matplotlib
os.environ["MPLBACKEND"] = "Agg"
matplotlib.use("Agg")
import matplotlib.pyplot as mpl

import pandas as pd

from pages.cores.sentiment_pipeline import (
    build_week_dirs,
    ensure_stage1_csv_once,
    stage2_add_columns,
    stage3_sentiment_and_plots,
    stage4_confusion,
)

# --------------------------- user settings
BASE_DIR         = Path(__file__).resolve().parents[2]  # project root (contains week8/)
WHITE_BACKGROUND = True
THR              = 0.05
RAW_FILE         = "stage_1_news_raw.csv"
LOOKBACK_DAYS    = 540
API_KEY          = os.environ.get("COINDESK_API_KEY", None)
# ----------------------------------------

if WHITE_BACKGROUND:
    mpl.rcdefaults()
else:
    mpl.style.use("dark_background")


def main() -> None:
    print(f"[runner] BASE_DIR resolved to: {BASE_DIR}")
    dirs = build_week_dirs(BASE_DIR)  # creates week8/news_clean_data & week8/news_figures
    data_dir = dirs["data_dir"]
    fig_dir  = dirs["fig_dir"]

    end_dt   = datetime.utcnow()
    start_dt = end_dt - timedelta(days=LOOKBACK_DAYS)

    # Stage 1: download-once (or reuse)
    raw = ensure_stage1_csv_once(
        api_key=API_KEY,
        start_dt=start_dt,
        end_dt=end_dt,
        data_dir=data_dir,
        filename=RAW_FILE,
    )
    print(f"[runner] Stage-1 rows: {len(raw):,} ({data_dir / RAW_FILE})")

    # Stage 2: cleaning/no I/O
    clean = stage2_add_columns(raw)
    print(f"[runner] Stage-2 clean shape: {clean.shape}")

    # Stage 3: VADER + plots + final CSV
    sent = stage3_sentiment_and_plots(clean, dirs, thr=THR)
    print(f"[runner] Stage-3 final CSV -> {data_dir / 'clean_news_timeseries.csv'}")
    print(f"[runner] Stage-3 plots     -> {fig_dir}")

    # Stage 4: confusion/classification if labels exist
    stage4_confusion(sent, dirs)
    print("[runner] Stage-4 done")


if __name__ == "__main__":
    main()
