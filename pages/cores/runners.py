import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .ml import getModelToPreds


def multi_symbol_rotation_by_model(
    symbol_to_df: dict[str, pd.DataFrame],
    symbols: list[str],
    model_name_list: list[str],
    features: list[str],
    train_size=60,
    pred_horizon=3,
    threshold=0.0,
    initial_capital=10000,
):
    """
    Rotational strategy for multiple models:
    For each model, at each date pick the symbol with the highest predicted return.
    Returns a dict of portfolio curves per model and their Sharpe ratios.
    """
    model_symbol_preds = {model: {} for model in model_name_list}

    # Step 1: Get predictions for each symbol and model
    for sym in symbols:
        df = symbol_to_df[sym]
        preds_dict = getModelToPreds(
            df=df,
            models=model_name_list,
            train_size=train_size,
            pred_horizon=pred_horizon,
            features=features,
        )
        for model_name, preds in preds_dict.items():
            pred_df = df.iloc[-len(preds) :].copy()
            pred_df["pred"] = preds
            model_symbol_preds[model_name][sym] = pred_df[["date", "return", "pred"]]

    portfolio_curves = {}
    sharpe_ratios = {}

    # Step 2: For each model, build combined DataFrame of all symbols' preds
    for model_name, sym_data in model_symbol_preds.items():
        all_preds_df = pd.concat(
            [df.assign(symbol=sym).set_index("date") for sym, df in sym_data.items()]
        )

        # Ensure index is datetime and sorted
        all_preds_df.index = pd.to_datetime(all_preds_df.index)
        all_preds_df = all_preds_df.sort_index()

        # Step 3: Find max pred per date (handle duplicates safely)
        max_preds = all_preds_df.groupby("date")["pred"].transform("max")
        best_pred_per_date = all_preds_df[all_preds_df["pred"] == max_preds].copy()

        # If multiple rows per date (ties), keep only the first to avoid duplicates
        best_pred_per_date.reset_index(inplace=True)
        best_pred_per_date = best_pred_per_date.drop_duplicates(
            subset=["date"], keep="first"
        )
        best_pred_per_date.set_index("date", inplace=True)
        best_pred_per_date.index.name = "date"

        # Step 4: Create full date range and reindex
        full_date_range = pd.date_range(
            start=best_pred_per_date.index.min(),
            end=best_pred_per_date.index.max(),
            freq="D",
        )
        best_pred_per_date = best_pred_per_date.reindex(full_date_range)

        # Fill missing values for pred and return with 0
        best_pred_per_date["pred"] = best_pred_per_date["pred"].fillna(0)
        best_pred_per_date["return"] = best_pred_per_date["return"].fillna(0)

        # Step 5: Calculate daily returns applying threshold
        daily_returns = best_pred_per_date.apply(
            lambda row: row["return"] if abs(row["pred"]) >= threshold else 0, axis=1
        )

        # Step 6: Calculate portfolio values
        portfolio_values = (1 + daily_returns).cumprod() * initial_capital

        portfolio_curves[model_name] = pd.DataFrame(
            {"date": portfolio_values.index, "portfolio_value": portfolio_values.values}
        )

        # Step 7: Calculate annualized Sharpe ratio
        mean_ret = np.mean(daily_returns)
        std_ret = np.std(daily_returns)
        sharpe = (mean_ret / std_ret) * np.sqrt(252) if std_ret != 0 else np.nan
        sharpe_ratios[model_name] = sharpe

    return portfolio_curves, sharpe_ratios
