import pandas as pd
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.linear_model import LinearRegression, ElasticNet


def prepare_features_targets(df: pd.DataFrame, feature_cols):
    """
    Shift target column up by lag to predict future returns
    """
    X = df[feature_cols]
    y = df["return_target"]
    return X, y


def train_forecaster(df, feature_cols, model):
    """
    Trains forecasting model to predict
    """
    X, y = prepare_features_targets(df=df, feature_cols=feature_cols)
    model.fit(X, y)
    return model


def predict_model(df, feature_cols, model):
    features, _ = prepare_features_targets(df=df, feature_cols=feature_cols)
    model_preds = model.predict(features)
    return model_preds


def rolling_forecast_pipeline(df, feature_cols, model, train_size=60, pred_horizon=7):
    """
    Rolling retrain pipeline:
    1. Train on `train_size` points
    2. Predict `pred_horizon` points
    3. Retrain on expanded window (train + predicted period)
    """
    preds = []
    idxs = []

    start = 0
    end = train_size

    while end + pred_horizon <= len(df):
        # 1. Train
        train_df = df.iloc[start:end]
        trained_model = train_forecaster(train_df, feature_cols, model)

        # 2. Predict next pred_horizon points
        pred_df = df.iloc[end : end + pred_horizon]
        pred_vals = predict_model(pred_df, feature_cols, trained_model)

        preds.extend(pred_vals)
        idxs.extend(pred_df.index)

        # 3. Expand training window
        end += pred_horizon

    # Return predictions aligned to the original DataFrame index
    pred_series = pd.Series(preds, index=idxs, name="predictions")
    return pred_series.values


def get_model_preds(df, model_name, features, train_size=60, pred_horizon=7):
    if model_name == "ridge":
        model = Ridge(alpha=1.0)
    elif model_name == "ols":
        model = LinearRegression()
    
        
    elif model_name == "elasticnet":
    
        model = ElasticNet(alpha=0.1, l1_ratio=0.5)
    else:
        raise ValueError("Unknown model name")

    preds = rolling_forecast_pipeline(
        df=df,
        feature_cols=features,
        model=model,
        train_size=train_size,
        pred_horizon=pred_horizon,
    )
    return preds


def getModelToPreds(df, models: list[str], features, train_size=60, pred_horizon=7):
    model_to_preds = {}
    for model_name in models:
        preds = get_model_preds(
            df=df,
            model_name=model_name,
            train_size=train_size,
            pred_horizon=pred_horizon,
            features=features,
        )
        model_to_preds[model_name] = preds
    return model_to_preds
