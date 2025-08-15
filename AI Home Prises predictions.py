import argparse
import os
from typing import Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import OneHotEncoder
import joblib

def load_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset file not found at '{path}'. "
            "Please download 'HousePricePrediction.xlsx' from the GfG link "
            "and place it here or provide the correct --data path."
        )
    # Uses openpyxl engine implicitly; ensure it's installed.
    df = pd.read_excel(path)
    return df

def summarize_schema(df: pd.DataFrame) -> Dict[str, int]:
    obj = (df.dtypes == 'object')
    int_ = (df.dtypes == 'int') | (df.dtypes == 'int64')
    fl = (df.dtypes == 'float') | (df.dtypes == 'float64')

    counts = {
        "categorical": int(obj.sum()),
        "integer": int(int_.sum()),
        "float": int(fl.sum()),
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
    }
    return counts

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Drop Id if present
    if 'Id' in df.columns:
        df.drop(['Id'], axis=1, inplace=True)

    # Fill SalePrice NA with mean if present
    if 'SalePrice' in df.columns:
        df['SalePrice'] = df['SalePrice'].fillna(df['SalePrice'].mean())

    # Drop remaining NA rows (as per article)
    df = df.dropna()
    return df

def encode_categoricals(df: pd.DataFrame) -> Tuple[pd.DataFrame, OneHotEncoder, list]:
    s = (df.dtypes == 'object')
    object_cols = list(s[s].index)

    if not object_cols:
        # Nothing to encode
        return df.copy(), None, []

    # Handle scikit-learn version compatibility:
    # - Newer versions use 'sparse_output', older use 'sparse'
    try:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    except TypeError:
        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

    OH_cols = pd.DataFrame(encoder.fit_transform(df[object_cols]), index=df.index)
    try:
        feature_names = list(encoder.get_feature_names_out(object_cols))
    except TypeError:
        # Older versions
        feature_names = list(encoder.get_feature_names(object_cols))

    OH_cols.columns = feature_names
    df_num = df.drop(columns=object_cols)
    df_final = pd.concat([df_num, OH_cols], axis=1)
    return df_final, encoder, object_cols

def split_xy(df_final: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    if 'SalePrice' not in df_final.columns:
        raise KeyError("Expected 'SalePrice' column as target in the dataset.")
    X = df_final.drop(['SalePrice'], axis=1)
    y = df_final['SalePrice']
    return X, y

def train_and_evaluate(X_train, X_valid, y_train, y_valid) -> Dict[str, Dict[str, float]]:
    results = {}

    # 1) SVR
    svr = SVR()
    svr.fit(X_train, y_train)
    pred = svr.predict(X_valid)
    results['SVR'] = {
        'MAPE': float(mean_absolute_percentage_error(y_valid, pred)),
        'MAE': float(mean_absolute_error(y_valid, pred)),
        'R2': float(r2_score(y_valid, pred)),
    }

    # 2) Random Forest
    rfr = RandomForestRegressor(n_estimators=10, random_state=0)
    rfr.fit(X_train, y_train)
    pred = rfr.predict(X_valid)
    results['RandomForest'] = {
        'MAPE': float(mean_absolute_percentage_error(y_valid, pred)),
        'MAE': float(mean_absolute_error(y_valid, pred)),
        'R2': float(r2_score(y_valid, pred)),
    }

    # 3) Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    pred = lr.predict(X_valid)
    results['LinearRegression'] = {
        'MAPE': float(mean_absolute_percentage_error(y_valid, pred)),
        'MAE': float(mean_absolute_error(y_valid, pred)),
        'R2': float(r2_score(y_valid, pred)),
    }

    return results

def pick_best_model(results: Dict[str, Dict[str, float]]) -> str:
    # Lower MAPE is better
    best_model = min(results.items(), key=lambda kv: kv[1]['MAPE'])[0]
    return best_model

def save_artifacts(best_name: str, models: Dict[str, object], encoder, encoded_cols: list, numeric_feature_names: list, outdir: str = '.') -> None:
    os.makedirs(outdir, exist_ok=True)
    # Save model
    joblib.dump(models[best_name], os.path.join(outdir, 'best_model.joblib'))
    # Save encoder & meta
    joblib.dump({
        'encoder': encoder,
        'encoded_object_cols': encoded_cols,
        'numeric_feature_names': numeric_feature_names
    }, os.path.join(outdir, 'preprocess.joblib'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='HousePricePrediction.xlsx',
                        help='Path to the HousePricePrediction.xlsx dataset')
    parser.add_argument('--test_size', type=float, default=0.2, help='Validation split size')
    parser.add_argument('--random_state', type=int, default=0, help='Random state for splitting')
    args = parser.parse_args()

    print('[1/7] Loading dataset...')
    df = load_dataset(args.data)

    print('[2/7] Schema summary:')
    info = summarize_schema(df)
    for k, v in info.items():
        print(f'  {k}: {v}')

    print('[3/7] Cleaning data...')
    df_clean = clean_data(df)

    print('[4/7] Encoding categoricals...')
    df_final, encoder, object_cols = encode_categoricals(df_clean)

    print('[5/7] Splitting X, y...')
    X, y = split_xy(df_final)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, train_size=1 - args.test_size, test_size=args.test_size, random_state=args.random_state
    )

    print('[6/7] Training models...')
    # Train models and keep references for saving
    models = {
        'SVR': SVR().fit(X_train, y_train),
        'RandomForest': RandomForestRegressor(n_estimators=10, random_state=args.random_state).fit(X_train, y_train),
        'LinearRegression': LinearRegression().fit(X_train, y_train),
    }

    results = {}
    # Evaluate
    for name, mdl in models.items():
        pred = mdl.predict(X_valid)
        results[name] = {
            'MAPE': float(mean_absolute_percentage_error(y_valid, pred)),
            'MAE': float(mean_absolute_error(y_valid, pred)),
            'R2': float(r2_score(y_valid, pred)),
        }

    print('\nModel performance (lower MAPE is better):')
    for name, metrics in results.items():
        print(f"- {name}: MAPE={metrics['MAPE']:.6f}, MAE={metrics['MAE']:.2f}, R2={metrics['R2']:.4f}")

    best_name = min(results.items(), key=lambda kv: kv[1]['MAPE'])[0]
    print(f"\n[7/7] Best model by MAPE: {best_name}")

    # Save artifacts
    numeric_features = [c for c in df_clean.columns if c not in object_cols]
    save_artifacts(best_name, models, encoder, object_cols, numeric_features, outdir='artifacts')
    print("Artifacts saved to ./artifacts (best_model.joblib, preprocess.joblib)")

    # Example: predict on first validation row
    example_row = X_valid.iloc[[0]]
    example_pred = models[best_name].predict(example_row)[0]
    print(f"\nExample prediction for 1 validation sample: {example_pred:.2f}")

if __name__ == '__main__':
    main()
