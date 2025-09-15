"""
Execute Stages 1 and 2 of the crypto data pipeline.
Works unchanged on macOS, Windows, or Linux.
"""

from pages.cores.crypto_pipeline import (
    stage1_etl,
    stage2_feature_engineering,
    build_week_dirs,
)

# ---------------------------------------------------------------------------
# User-adjustable inputs -----------------------------------------------------
# Please include your API key #
API_KEY = "446dcc8a6daf40347ca5373fd7a4886d4cd51bb87bedacf648c8de54050c73b2"          # required
PAGES   = [1,2,3]                          # which pages of the top-list to pull
TOP_LIMIT = 100                          # coins per page
HISTORY_LIMIT = 500                     # days of history per coin
CURRENCY = "USD"                         # quote currency
WEEK_FOLDER = "week5_crypto"             # root for outputs
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Stage 1 then Stage 2 with the constants above."""
    data_dir = build_week_dirs(WEEK_FOLDER)
    print(data_dir)
    # Stage 1 – raw daily prices
    df_prices = stage1_etl(
        api_key=API_KEY,
        pages=PAGES,
        top_limit=TOP_LIMIT,
        history_limit=HISTORY_LIMIT,
        currency=CURRENCY,
        data_dir=data_dir,
    )

    # Stage 2 – derived weekly features
    stage2_feature_engineering(
        tidy_prices=df_prices,
        data_dir=data_dir,
    )

    print("Done!")
    print("Data ->", data_dir.resolve())


if __name__ == "__main__":  # required on Windows
    main()
