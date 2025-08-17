import argparse
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Set, Dict, Any

import joblib
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import random


@dataclass
class Config:
    pos_csv_path: str
    output_dir: str
    forecast_horizon_days: int = 7
    close_hour_local: int = 22  # NYC close hour
    discount_windows_minutes: Tuple[int, ...] = (120, 60, 30)
    discount_levels: Tuple[float, ...] = (0.8, 0.6, 0.4)  # 20%, 40%, 60% off
    service_level: float = 0.95
    random_state: int = 42
    fd_train_csv_path: Optional[str] = os.path.join('data', 'fd_train.csv')
    fd_meal_info_csv_path: Optional[str] = os.path.join('data', 'fd_meal_info.csv')
    fd_center_info_csv_path: Optional[str] = os.path.join('data', 'fd_fulfilment_center_info.csv')
    sweep_candidates: int = 12


def load_pos_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Normalize column names
    df.columns = [c.strip() for c in df.columns]
    # Parse timestamp
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    elif 'timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['timestamp'])
    else:
        raise ValueError('Expected a Timestamp column in POS data')

    # Ensure required columns
    required = ['Timestamp', 'Item_Name', 'Item_Category', 'Quantity', 'Price_Per_Item']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'Missing required columns in POS data: {missing}')

    # Clean types
    df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce').fillna(0).astype(int)
    df['Price_Per_Item'] = pd.to_numeric(df['Price_Per_Item'], errors='coerce').fillna(np.nan)
    # Drop rows with missing price
    df = df.dropna(subset=['Price_Per_Item'])

    return df


def aggregate_daily_item_demand(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['date'] = df['Timestamp'].dt.date
    df['hour'] = df['Timestamp'].dt.hour
    # Aggregate to item-day
    daily = (
        df.groupby(['date', 'Item_Category'])
          .agg(
              total_qty=('Quantity', 'sum'),
              avg_price=('Price_Per_Item', 'mean'),
              num_txn=('Quantity', 'count')
          )
          .reset_index()
    )
    # Calendar features
    dts = pd.to_datetime(daily['date'])
    daily['dow'] = dts.dt.dayofweek
    daily['is_weekend'] = (daily['dow'] >= 5).astype(int)
    daily['weekofyear'] = dts.dt.isocalendar().week.astype(int)
    daily['month'] = dts.dt.month

    # Build per-item lags and rolling stats
    daily = daily.sort_values(['Item_Category', 'date'])
    for lag in [1, 7, 14, 28]:
        daily[f'lag_{lag}'] = (
            daily.groupby(['Item_Category'])['total_qty']
                 .shift(lag)
        )
    for w in [7, 14, 28]:
        daily[f'rollmean_{w}'] = (
            daily.groupby(['Item_Category'])['total_qty']
                 .shift(1)
                 .rolling(window=w, min_periods=max(1, int(w/2)))
                 .mean()
        )
        daily[f'rollstd_{w}'] = (
            daily.groupby(['Item_Category'])['total_qty']
                 .shift(1)
                 .rolling(window=w, min_periods=max(1, int(w/2)))
                 .std()
        )

    # Drop rows without history
    daily = daily.dropna(subset=['lag_1'])

    return daily


def split_train_val(daily: pd.DataFrame, val_days: int = 28) -> Tuple[pd.DataFrame, pd.DataFrame]:
    last_date = pd.to_datetime(daily['date']).max().date()
    val_start = last_date - timedelta(days=val_days)
    train = daily[pd.to_datetime(daily['date']) < pd.Timestamp(val_start)].copy()
    val = daily[pd.to_datetime(daily['date']) >= pd.Timestamp(val_start)].copy()
    return train, val


def build_preprocessor() -> Tuple[ColumnTransformer, List[str]]:
    numeric_features = [
        'avg_price', 'num_txn', 'dow', 'is_weekend', 'weekofyear', 'month', 'is_holiday',
        'lag_1', 'lag_7', 'lag_14', 'lag_28',
        'rollmean_7', 'rollmean_14', 'rollmean_28',
        'rollstd_7', 'rollstd_14', 'rollstd_28'
    ]
    categorical_features = ['Item_Category']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(with_mean=False), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=True), categorical_features),
        ],
        remainder='drop'
    )
    features = numeric_features + categorical_features
    return preprocessor, features


def build_models(random_state: int = 42) -> Dict[str, Pipeline]:
    preprocessor, _ = build_preprocessor()

    xgb = XGBRegressor(
        n_estimators=800,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        random_state=random_state,
        n_jobs=4,
    )
    lgbm = LGBMRegressor(
        n_estimators=1200,
        max_depth=-1,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        num_leaves=63,
        reg_alpha=0.0,
        reg_lambda=0.0,
        random_state=random_state,
        n_jobs=4,
    )
    rf = RandomForestRegressor(
        n_estimators=600,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=4,
        random_state=random_state,
    )

    return {
        'xgboost': Pipeline(steps=[('prep', preprocessor), ('model', xgb)]),
        'lightgbm': Pipeline(steps=[('prep', preprocessor), ('model', lgbm)]),
        'random_forest': Pipeline(steps=[('prep', preprocessor), ('model', rf)]),
    }


def sample_param_sets(random_state: int, n_samples: int) -> Dict[str, List[Dict[str, Any]]]:
    rng = random.Random(random_state)
    params: Dict[str, List[Dict[str, Any]]] = {
        'xgboost': [],
        'lightgbm': [],
        'random_forest': [],
    }
    for _ in range(n_samples):
        params['xgboost'].append({
            'model__n_estimators': rng.choice([600, 800, 1200]),
            'model__max_depth': rng.choice([4, 6, 8]),
            'model__learning_rate': rng.choice([0.03, 0.05, 0.1]),
            'model__subsample': rng.choice([0.8, 1.0]),
            'model__colsample_bytree': rng.choice([0.8, 1.0]),
        })
        params['lightgbm'].append({
            'model__n_estimators': rng.choice([800, 1200, 1600]),
            'model__learning_rate': rng.choice([0.03, 0.05]),
            'model__num_leaves': rng.choice([31, 63, 127]),
            'model__subsample': rng.choice([0.8, 0.9, 1.0]),
            'model__colsample_bytree': rng.choice([0.8, 0.9, 1.0]),
        })
        params['random_forest'].append({
            'model__n_estimators': rng.choice([400, 600, 800]),
            'model__max_depth': rng.choice([None, 10, 20]),
            'model__min_samples_split': rng.choice([2, 4, 8]),
            'model__min_samples_leaf': rng.choice([1, 2, 4]),
        })
    return params


def sweep_models(daily: pd.DataFrame, random_state: int, n_samples: int, output_dir: Optional[str]) -> Tuple[Pipeline, float, pd.DataFrame]:
    train, val = split_train_val(daily)
    _, features = build_preprocessor()
    base_models = build_models(random_state)
    param_sets = sample_param_sets(random_state, n_samples)
    rows = []
    best_model: Optional[Pipeline] = None
    best_mae = float('inf')
    best_name = ''
    best_params: Dict[str, Any] = {}

    for name, base in base_models.items():
        for ps in param_sets[name]:
            from sklearn.base import clone as sk_clone
            model = sk_clone(base)
            model.set_params(**ps)
            model.fit(train[features], train['total_qty'])
            val_pred = model.predict(val[features])
            mae = mean_absolute_error(val['total_qty'], val_pred)
            rmse = float(np.sqrt(np.mean((val['total_qty'].values - val_pred) ** 2)))
            r2 = float(r2_score(val['total_qty'].values, val_pred))
            rec = {'model': name, 'mae': mae, 'rmse': rmse, 'r2': r2}
            rec.update({k.replace('model__', ''): v for k, v in ps.items()})
            rows.append(rec)
            if mae < best_mae:
                best_mae = mae
                best_model = model
                best_name = name
                best_params = ps

    sweep_df = pd.DataFrame(rows).sort_values('mae')
    if output_dir:
        os.makedirs(os.path.join(output_dir, 'eval'), exist_ok=True)
        sweep_df.to_csv(os.path.join(output_dir, 'eval', 'model_sweep.csv'), index=False)
    print(f"Sweep best: {best_name} MAE={best_mae:.3f} params={best_params}")
    assert best_model is not None
    return best_model, best_mae, sweep_df


def train_and_evaluate(daily: pd.DataFrame, random_state: int = 42, output_dir: Optional[str] = None, sweep_candidates: int = 0) -> Tuple[Pipeline, float, pd.DataFrame]:
    if sweep_candidates and sweep_candidates > 0:
        return sweep_models(daily, random_state, sweep_candidates, output_dir)

    train, val = split_train_val(daily)

    _, features = build_preprocessor()
    models = build_models(random_state)

    rows = []
    best_name = None
    best_mae = float('inf')
    best_model = None

    for name, model in models.items():
        model.fit(train[features], train['total_qty'])
        val_pred = model.predict(val[features])
        mae = mean_absolute_error(val['total_qty'], val_pred)
        rmse = float(np.sqrt(np.mean((val['total_qty'].values - val_pred) ** 2)))
        r2 = float(r2_score(val['total_qty'].values, val_pred))
        rows.append({'model': name, 'mae': mae, 'rmse': rmse, 'r2': r2})
        if mae < best_mae:
            best_mae = mae
            best_model = model
            best_name = name

    comp_df = pd.DataFrame(rows).sort_values('mae')
    if output_dir:
        os.makedirs(os.path.join(output_dir, 'eval'), exist_ok=True)
        comp_df.to_csv(os.path.join(output_dir, 'eval', 'model_comparison.csv'), index=False)

    print(f"Selected model: {best_name} (MAE={best_mae:.3f})")
    assert best_model is not None
    return best_model, best_mae, comp_df


def estimate_forecast_error_std(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    resid = y_true - y_pred
    return float(np.sqrt(np.maximum(np.var(resid), 1e-6)))


def compute_service_level_z(service_level: float) -> float:
    # Inverse CDF for standard normal via scipy-free approximation
    from math import sqrt, log
    # Beasley-Springer/Moro-like approximation
    a = [2.50662823884, -18.61500062529, 41.39119773534, -25.44106049637]
    b = [-8.47351093090, 23.08336743743, -21.06224101826, 3.13082909833]
    c = [0.3374754822726147, 0.9761690190917186, 0.1607979714918209,
         0.0276438810333863, 0.0038405729373609, 0.0003951896511919,
         0.0000321767881768, 0.0000002888167364, 0.0000003960315187]

    if service_level <= 0 or service_level >= 1:
        raise ValueError('service_level must be in (0,1)')

    y = service_level - 0.5
    if abs(y) < 0.42:
        r = y * y
        num = y * (((a[3] * r + a[2]) * r + a[1]) * r + a[0])
        den = (((b[3] * r + b[2]) * r + b[1]) * r + 1.0)
        return num / den
    r = service_level if y > 0 else 1 - service_level
    s = -log(-log(r))
    z = c[0] + s * (c[1] + s * (c[2] + s * (c[3] + s * (c[4] + s * (c[5] + s * (c[6] + s * (c[7] + s * c[8])))))))
    return z if y > 0 else -z


def forecast_next_days(model: Pipeline, daily: pd.DataFrame, horizon_days: int, holidays: Optional[Set[pd.Timestamp]] = None) -> pd.DataFrame:
    # Create future dates per item based on last known date
    last_date = pd.to_datetime(daily['date']).max().date()
    items = daily[['Item_Category']].drop_duplicates().reset_index(drop=True)

    # Prepare base frame with lags computed from history
    hist = daily.copy()
    hist['date'] = pd.to_datetime(hist['date'])

    forecasts = []
    for d in range(1, horizon_days + 1):
        curr_date = last_date + timedelta(days=d)
        df_future = items.copy()
        df_future['date'] = pd.Timestamp(curr_date)
        dts = pd.to_datetime(df_future['date'])
        df_future['dow'] = dts.dt.dayofweek
        df_future['is_weekend'] = (df_future['dow'] >= 5).astype(int)
        df_future['weekofyear'] = dts.dt.isocalendar().week.astype(int)
        df_future['month'] = dts.dt.month
        df_future['is_holiday'] = int(pd.Timestamp(curr_date).normalize() in (holidays or set()))

        # Merge last known price as proxy (could be improved)
        last_price = (hist.sort_values('date')
                           .groupby(['Item_Category'])['avg_price']
                           .last()
                           .reset_index())
        last_num_txn = (hist.sort_values('date')
                            .groupby(['Item_Category'])['num_txn']
                            .last()
                            .reset_index())
        df_future = df_future.merge(last_price, on=['Item_Category'], how='left')
        df_future = df_future.merge(last_num_txn, on=['Item_Category'], how='left')

        # Compute lags/rolls from history
        for lag in [1, 7, 14, 28]:
            lag_df = hist[hist['date'] == (pd.Timestamp(curr_date) - pd.Timedelta(days=lag))][['Item_Category','total_qty']]
            df_future = df_future.merge(
                lag_df.rename(columns={'total_qty': f'lag_{lag}'}),
                on=['Item_Category'], how='left'
            )
        for w in [7, 14, 28]:
            # approximate rolling stats using available history up to previous day
            roll_mean = (hist[hist['date'] < pd.Timestamp(curr_date)]
                            .sort_values('date')
                            .groupby(['Item_Category'])['total_qty']
                            .apply(lambda s: s.tail(w).mean())
                            .reset_index(name=f'rollmean_{w}') )
            roll_std = (hist[hist['date'] < pd.Timestamp(curr_date)]
                           .sort_values('date')
                           .groupby(['Item_Category'])['total_qty']
                           .apply(lambda s: s.tail(w).std())
                           .reset_index(name=f'rollstd_{w}') )
            df_future = df_future.merge(roll_mean, on=['Item_Category'], how='left')
            df_future = df_future.merge(roll_std, on=['Item_Category'], how='left')

        # Fill missing with sensible defaults
        feature_cols = [
            'avg_price', 'num_txn', 'dow', 'is_weekend', 'weekofyear', 'month', 'is_holiday',
            'Item_Category',
            'lag_1', 'lag_7', 'lag_14', 'lag_28',
            'rollmean_7', 'rollmean_14', 'rollmean_28',
            'rollstd_7', 'rollstd_14', 'rollstd_28'
        ]
        for c in ['avg_price', 'num_txn']:
            df_future[c] = df_future[c].fillna(hist[c].median())
        # Fill remaining NA lags/rolls with per-item-category mean total_qty
        default_qty = hist.groupby(['Item_Category'])['total_qty'].mean().reset_index().rename(columns={'total_qty':'_default_qty'})
        df_future = df_future.merge(default_qty, on=['Item_Category'], how='left')
        for c in ['lag_1', 'lag_7', 'lag_14', 'lag_28', 'rollmean_7', 'rollmean_14', 'rollmean_28', 'rollstd_7', 'rollstd_14', 'rollstd_28']:
            df_future[c] = df_future[c].fillna(df_future['_default_qty'])
        df_future = df_future.drop(columns=['_default_qty'])

        preds = model.predict(df_future[feature_cols])
        df_future['predicted_qty'] = np.maximum(preds, 0.0)
        df_future['forecast_date'] = curr_date
        forecasts.append(df_future[['forecast_date', 'Item_Category', 'predicted_qty', 'avg_price']])

        # Append predictions to history for iterative lags if horizon>1
        tmp = df_future[['Item_Category']].copy()
        tmp['date'] = pd.Timestamp(curr_date)
        tmp['total_qty'] = df_future['predicted_qty']
        tmp['avg_price'] = df_future['avg_price']
        tmp['num_txn'] = df_future['num_txn']
        tmp['dow'] = df_future['dow']
        tmp['is_weekend'] = df_future['is_weekend']
        tmp['weekofyear'] = df_future['weekofyear']
        tmp['month'] = df_future['month']
        hist = pd.concat([hist, tmp], ignore_index=True)

    return pd.concat(forecasts, ignore_index=True)


def estimate_price_elasticity(df: pd.DataFrame, global_fallback: float = -1.0) -> pd.DataFrame:
    # Simple log-log regression per item
    results = []
    grouped = df.groupby('Item_Category')
    for item, g in grouped:
        g = g.copy()
        # Avoid non-positive
        g = g[(g['avg_price'] > 0) & (g['total_qty'] > 0)]
        if len(g) < 10:
            elasticity = float(global_fallback)
        else:
            x = np.log(g['avg_price'].values)
            y = np.log(g['total_qty'].values)
            x = np.vstack([np.ones_like(x), x]).T
            beta, *_ = np.linalg.lstsq(x, y, rcond=None)
            elasticity = float(beta[1])
        results.append({'Item_Category': item, 'price_elasticity': elasticity})
    return pd.DataFrame(results)


def compute_safety_stock(daily: pd.DataFrame, model: Pipeline, service_level: float) -> float:
    # Estimate global forecast error std on validation as a fallback
    train, val = split_train_val(daily)
    features = [
        'avg_price', 'num_txn', 'dow', 'is_weekend', 'weekofyear', 'month', 'is_holiday',
        'Item_Category',
        'lag_1', 'lag_7', 'lag_14', 'lag_28',
        'rollmean_7', 'rollmean_14', 'rollmean_28',
        'rollstd_7', 'rollstd_14', 'rollstd_28'
    ]
    val_pred = model.predict(val[features])
    sigma = estimate_forecast_error_std(val['total_qty'].values, val_pred)
    z = compute_service_level_z(service_level)
    ss = z * sigma
    return float(max(ss, 0.0))


def simulate_end_of_day_discounts(
    forecast_df: pd.DataFrame,
    elasticity_df: pd.DataFrame,
    discount_windows_minutes: Tuple[int, ...],
    discount_levels: Tuple[float, ...],
    close_hour_local: int
) -> pd.DataFrame:
    # Approximate: assume base forecast spread uniformly across open hours (e.g., 10:00-22:00)
    # Apply discounts in last windows to boost demand according to elasticity
    open_hour_local = 10
    hours_open = max(close_hour_local - open_hour_local, 1)

    merged = forecast_df.merge(elasticity_df, on='Item_Category', how='left')
    merged['price_elasticity'] = merged['price_elasticity'].fillna(-1.0)

    # Base demand near close assumed proportional to window length
    total_minutes = hours_open * 60
    window_fracs = [w / total_minutes for w in discount_windows_minutes]
    base_window_demands = [merged['predicted_qty'] * f for f in window_fracs]

    # Apply discounts multiplicatively via elasticity
    demand_after_discounts = merged['predicted_qty'].copy().astype(float)
    for base_window_qty, disc in zip(base_window_demands, discount_levels):
        price_ratio = disc  # p_discounted / p_base
        demand_multiplier = np.power(price_ratio, merged['price_elasticity'])
        uplift = base_window_qty * (demand_multiplier - 1.0)
        demand_after_discounts = demand_after_discounts + uplift

    merged['expected_demand_post_discounts'] = np.maximum(demand_after_discounts, 0.0)
    return merged[['forecast_date', 'Item_Category', 'avg_price', 'predicted_qty', 'price_elasticity', 'expected_demand_post_discounts']]


def optimize_discount_schedules(
    forecasts: pd.DataFrame,
    elasticity_df: pd.DataFrame,
    discount_windows_minutes: Tuple[int, ...],
    candidate_schedules: Tuple[Tuple[float, ...], ...],
    close_hour_local: int,
    safety_stock_units: float,
) -> pd.DataFrame:
    """Select the best discount schedule per category-day to minimize expected waste (tie-break: maximize revenue)."""
    open_hour_local = 10
    hours_open = max(close_hour_local - open_hour_local, 1)
    total_minutes = hours_open * 60
    window_fracs = np.array([w / total_minutes for w in discount_windows_minutes])

    df = forecasts.merge(elasticity_df, on='Item_Category', how='left')
    df['price_elasticity'] = df['price_elasticity'].fillna(-1.0)

    # Precompute base window demand and non-window demand
    base_window_demands = np.outer(df['predicted_qty'].values, window_fracs)  # shape (n, W)
    base_rest = df['predicted_qty'].values - base_window_demands.sum(axis=1)

    best_waste = np.full(len(df), np.inf)
    best_revenue = np.full(len(df), -np.inf)
    best_schedule = [None] * len(df)
    best_expected_demand = np.zeros(len(df))

    for sched in candidate_schedules:
        sched = np.array(sched)
        price_ratio = sched  # elementwise per window
        # demand multiplier per row per window: (price_ratio ** elasticity)
        demand_mult = np.power(price_ratio.reshape(1, -1), df['price_elasticity'].values.reshape(-1, 1))
        demand_in_windows = base_window_demands * demand_mult
        total_expected_demand = np.maximum(base_rest + demand_in_windows.sum(axis=1), 0.0)

        # Revenue approximation: base price for rest-of-day, discounted price in windows
        revenue = (df['avg_price'].values * np.maximum(base_rest, 0.0)) + (df['avg_price'].values.reshape(-1, 1) * sched.reshape(1, -1) * demand_in_windows).sum(axis=1)

        # Recommended stock given safety stock
        rec_stock = np.ceil(df['predicted_qty'].values + safety_stock_units)
        expected_waste = np.maximum(rec_stock - total_expected_demand, 0.0)

        # Choose schedule that minimizes waste, tie-breaker by revenue
        better = (expected_waste < best_waste) | ((expected_waste == best_waste) & (revenue > best_revenue))
        best_waste = np.where(better, expected_waste, best_waste)
        best_revenue = np.where(better, revenue, best_revenue)
        best_expected_demand = np.where(better, total_expected_demand, best_expected_demand)
        best_schedule = [tuple(sched) if b else prev for b, prev in zip(better, best_schedule)]

    out = df[['forecast_date', 'Item_Category', 'avg_price', 'predicted_qty', 'price_elasticity']].copy()
    out['recommended_stock'] = np.ceil(out['predicted_qty'] + safety_stock_units).astype(int)
    out['opt_discount_schedule'] = [str(s) for s in best_schedule]
    out['expected_demand_post_discounts'] = best_expected_demand
    out['expected_waste'] = best_waste
    out['expected_revenue'] = best_revenue
    return out


def compute_fd_nyc_elasticity(
    fd_train_csv_path: Optional[str],
    fd_meal_info_csv_path: Optional[str],
    fd_center_info_csv_path: Optional[str],
    nyc_region_codes: Set[int] = frozenset({23, 34, 45, 56, 67}),
) -> Optional[pd.DataFrame]:
    if not (fd_train_csv_path and os.path.exists(fd_train_csv_path)):
        return None
    if not (fd_meal_info_csv_path and os.path.exists(fd_meal_info_csv_path)):
        return None
    if not (fd_center_info_csv_path and os.path.exists(fd_center_info_csv_path)):
        return None
    try:
        usecols = ['week', 'meal_id', 'center_id', 'checkout_price', 'num_orders']
        tr = pd.read_csv(fd_train_csv_path, usecols=usecols)
        meals = pd.read_csv(fd_meal_info_csv_path, usecols=['meal_id', 'category'])
        centers = pd.read_csv(fd_center_info_csv_path, usecols=['center_id', 'region_code'])
        df = tr.merge(meals, on='meal_id', how='left').merge(centers, on='center_id', how='left')
        df = df[(df['checkout_price'] > 0) & (df['num_orders'] > 0)]
        df = df[df['region_code'].isin(list(nyc_region_codes))]
        if df.empty:
            return None
        # Within-meal demeaning to approximate FE; then per-category slope
        df['log_q'] = np.log(df['num_orders'])
        df['log_p'] = np.log(df['checkout_price'])
        df['log_q_dm'] = df['log_q'] - df.groupby('meal_id')['log_q'].transform('mean')
        df['log_p_dm'] = df['log_p'] - df.groupby('meal_id')['log_p'].transform('mean')
        out = []
        for cat, g in df.groupby('category'):
            x = g['log_p_dm'].values.reshape(-1, 1)
            y = g['log_q_dm'].values
            xtx = float(np.dot(x.T, x))
            if xtx <= 0 or len(g) < 20:
                continue
            beta = float(np.dot(x.T, y) / xtx)
            out.append({'fd_category': cat, 'fd_elasticity_nyc': beta})
        if not out:
            return None
        return pd.DataFrame(out)
    except Exception:
        return None


def map_fd_category_to_item_category() -> Dict[str, str]:
    # Heuristic mapping between FD meal categories and our POS Item_Category
    return {
        'Beverages': 'Cool Drinks',
        'Soup': 'Hot Soups',
        'Salad': 'Salads',
        'Rice Bowl': 'Curries',  # approximate
        'Biryani': 'Biryani',
        'Sandwich': 'Roti',      # approximate
        'Other Snacks': 'Pakoras',
        'Starters': 'Pakoras',
        'Desert': 'Ice Cream',   # approximate
        'Pasta': 'Curries',      # approximate
        'Extras': None,          # skip
    }


def blend_elasticities(
    pos_elast_df: pd.DataFrame,
    fd_elast_df: Optional[pd.DataFrame],
    weight_pos: float = 0.7,
) -> pd.DataFrame:
    pos = pos_elast_df.copy()
    pos = pos.rename(columns={'price_elasticity': 'pos_elasticity'})
    if fd_elast_df is None or fd_elast_df.empty:
        pos['price_elasticity'] = pos['pos_elasticity']
        return pos[['Item_Category', 'price_elasticity']]
    # Map FD category to Item_Category
    mapping = map_fd_category_to_item_category()
    fd = fd_elast_df.copy()
    fd['Item_Category'] = fd['fd_category'].map(mapping)
    fd = fd.dropna(subset=['Item_Category'])
    fd = fd.groupby('Item_Category', as_index=False)['fd_elasticity_nyc'].mean()
    merged = pos.merge(fd, on='Item_Category', how='left')
    # Blend where FD available
    merged['price_elasticity'] = np.where(
        merged['fd_elasticity_nyc'].notna(),
        weight_pos * merged['pos_elasticity'] + (1.0 - weight_pos) * merged['fd_elasticity_nyc'],
        merged['pos_elasticity']
    )
    return merged[['Item_Category', 'price_elasticity']]


def build_recommendations(
    forecasts: pd.DataFrame,
    daily: pd.DataFrame,
    model: Pipeline,
    service_level: float,
    discount_windows_minutes: Tuple[int, ...],
    discount_levels: Tuple[float, ...],
    close_hour_local: int,
    output_dir: str,
    global_elasticity: Optional[float] = None,
    fd_meal_info_csv_path: Optional[str] = None,
    fd_center_info_csv_path: Optional[str] = None,
    fd_train_csv_path: Optional[str] = None,
):
    os.makedirs(output_dir, exist_ok=True)

    # POS per-category elasticity
    pos_elasticity_df = estimate_price_elasticity(daily, global_fallback=(global_elasticity if global_elasticity is not None else -1.0))

    # FD NYC per-category elasticity (optional) and blending
    fd_elast_df = compute_fd_nyc_elasticity(fd_train_csv_path, fd_meal_info_csv_path, fd_center_info_csv_path)
    elasticity_df = blend_elasticities(pos_elasticity_df, fd_elast_df, weight_pos=0.7)

    ss = compute_safety_stock(daily, model, service_level)

    # Baseline schedule simulation
    sim = simulate_end_of_day_discounts(
        forecasts, elasticity_df,
        discount_windows_minutes, discount_levels, close_hour_local
    )
    sim['recommended_stock'] = np.ceil(sim['predicted_qty'] + ss).astype(int)
    sim['expected_waste'] = np.maximum(sim['recommended_stock'] - sim['expected_demand_post_discounts'], 0.0)
    sim_path = os.path.join(output_dir, 'stock_recommendations.csv')
    sim.to_csv(sim_path, index=False)

    # Dynamic pricing optimization
    candidate_schedules: Tuple[Tuple[float, ...], ...] = (
        tuple([1.0] * len(discount_windows_minutes)),
        discount_levels,
        tuple(min(1.0, x + 0.1) for x in discount_levels),
        tuple(max(0.1, x - 0.1) for x in discount_levels),
    )
    dyn = optimize_discount_schedules(
        forecasts, elasticity_df,
        discount_windows_minutes, candidate_schedules, close_hour_local, ss
    )
    dyn_path = os.path.join(output_dir, 'dynamic_pricing_recommendations.csv')
    dyn.to_csv(dyn_path, index=False)

    return sim_path, ss


def compute_us_ny_holidays(min_date: pd.Timestamp, max_date: pd.Timestamp) -> Set[pd.Timestamp]:
    cal = USFederalHolidayCalendar()
    hol = cal.holidays(start=min_date, end=max_date).normalize()
    hol_set: Set[pd.Timestamp] = set(pd.to_datetime(hol))
    # Add Black Friday (day after Thanksgiving)
    # Thanksgiving = fourth Thursday in November
    for year in range(min_date.year, max_date.year + 1):
        nov = pd.date_range(start=pd.Timestamp(year=year, month=11, day=1), end=pd.Timestamp(year=year, month=11, day=30), freq='D')
        thursdays = [d for d in nov if d.weekday() == 3]
        if len(thursdays) >= 4:
            thanksgiving = thursdays[3].normalize()
            black_friday = (thanksgiving + pd.Timedelta(days=1)).normalize()
            if min_date <= black_friday <= max_date:
                hol_set.add(black_friday)
    return hol_set


def enrich_with_holidays(daily: pd.DataFrame, holidays: Set[pd.Timestamp]) -> pd.DataFrame:
    daily = daily.copy()
    dts = pd.to_datetime(daily['date']).dt.normalize()
    daily['is_holiday'] = dts.isin(holidays).astype(int) if holidays else 0
    return daily


def estimate_global_elasticity_from_fd(fd_train_csv_path: Optional[str]) -> Optional[float]:
    if not fd_train_csv_path or not os.path.exists(fd_train_csv_path):
        return None
    try:
        usecols = ['week', 'meal_id', 'center_id', 'checkout_price', 'base_price', 'emailer_for_promotion', 'homepage_featured', 'num_orders']
        df = pd.read_csv(fd_train_csv_path, usecols=usecols)
        df = df[(df['checkout_price'] > 0) & (df['num_orders'] > 0)]
        df['log_q'] = np.log(df['num_orders'])
        df['log_p'] = np.log(df['checkout_price'])
        df['log_q_dm'] = df['log_q'] - df.groupby('meal_id')['log_q'].transform('mean')
        df['log_p_dm'] = df['log_p'] - df.groupby('meal_id')['log_p'].transform('mean')
        x = df['log_p_dm'].values.reshape(-1, 1)
        y = df['log_q_dm'].values
        xtx = float(np.dot(x.T, x))
        if xtx <= 0:
            return None
        beta = float(np.dot(x.T, y) / xtx)
        return beta
    except Exception:
        return None


def write_evaluation_report(daily: pd.DataFrame, model: Pipeline, service_level: float, output_dir: str) -> str:
    """Generate validation metrics, per-category breakdowns, and policy coverage using safety stock."""
    eval_dir = os.path.join(output_dir, 'eval')
    os.makedirs(eval_dir, exist_ok=True)

    train, val = split_train_val(daily)
    features = [
        'avg_price', 'num_txn', 'dow', 'is_weekend', 'weekofyear', 'month', 'is_holiday',
        'Item_Category',
        'lag_1', 'lag_7', 'lag_14', 'lag_28',
        'rollmean_7', 'rollmean_14', 'rollmean_28',
        'rollstd_7', 'rollstd_14', 'rollstd_28'
    ]
    y_true = val['total_qty'].values
    y_pred = model.predict(val[features])

    # Metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mask = y_true > 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))) if mask.any() else np.nan
    smape = float(np.mean(2.0 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-9)))
    r2 = float(r2_score(y_true, y_pred))
    bias = float(np.mean(y_pred - y_true))

    # Policy coverage with safety stock
    ss = compute_safety_stock(daily, model, service_level)
    rec_stock = np.ceil(y_pred + ss)
    met_demand = rec_stock >= y_true
    service_level_days = float(np.mean(met_demand))  # fraction of days covered
    fill_rate = float(np.sum(np.minimum(rec_stock, y_true)) / (np.sum(y_true) + 1e-9))

    # Save predictions with errors
    val_out = val.copy()
    val_out['y_pred'] = y_pred
    val_out['abs_error'] = np.abs(val_out['total_qty'] - val_out['y_pred'])
    val_out['ape'] = np.where(val_out['total_qty'] > 0, val_out['abs_error'] / val_out['total_qty'], np.nan)
    val_out['smape'] = 2.0 * np.abs(val_out['total_qty'] - val_out['y_pred']) / (np.abs(val_out['total_qty']) + np.abs(val_out['y_pred']) + 1e-9)
    val_out['rec_stock'] = rec_stock
    val_out['met_demand'] = met_demand.astype(int)
    val_out.to_csv(os.path.join(eval_dir, 'val_predictions.csv'), index=False)

    # Per-category metrics
    per_cat = val_out.groupby('Item_Category').agg(
        mae=('abs_error', 'mean'),
        mape=('ape', 'mean'),
        rmse=('abs_error', lambda s: float(np.sqrt(np.mean(s**2)))),
        support=('total_qty', 'size'),
    ).reset_index()
    per_cat.to_csv(os.path.join(eval_dir, 'per_category_metrics.csv'), index=False)

    # Global metrics CSV
    metrics_rows = [
        ('mae', mae),
        ('rmse', rmse),
        ('mape', mape),
        ('smape', smape),
        ('r2', r2),
        ('bias', bias),
        ('safety_stock_units', ss),
        ('service_level_days', service_level_days),
        ('fill_rate', fill_rate),
    ]
    pd.DataFrame(metrics_rows, columns=['metric', 'value']).to_csv(os.path.join(eval_dir, 'metrics.csv'), index=False)

    return eval_dir


def main():
    parser = argparse.ArgumentParser(description='NYC daily demand, stock, waste, and pricing pipeline')
    parser.add_argument('--pos_csv', type=str, default=os.path.join('data', 'simulated_pos_data_with_seasonal_trends.csv'))
    parser.add_argument('--output_dir', type=str, default=os.path.join('outputs'))
    parser.add_argument('--horizon_days', type=int, default=7)
    parser.add_argument('--close_hour', type=int, default=22)
    parser.add_argument('--service_level', type=float, default=0.95)
    parser.add_argument('--fd_train_csv', type=str, default=os.path.join('data', 'fd_train.csv'))
    parser.add_argument('--fd_meal_info_csv', type=str, default=os.path.join('data', 'fd_meal_info.csv'))
    parser.add_argument('--fd_center_info_csv', type=str, default=os.path.join('data', 'fd_fulfilment_center_info.csv'))
    parser.add_argument('--sweep_candidates', type=int, default=12)
    args = parser.parse_args()

    cfg = Config(
        pos_csv_path=args.pos_csv,
        output_dir=args.output_dir,
        forecast_horizon_days=args.horizon_days,
        close_hour_local=args.close_hour,
        service_level=args.service_level,
        fd_train_csv_path=args.fd_train_csv,
        fd_meal_info_csv_path=args.fd_meal_info_csv,
        fd_center_info_csv_path=args.fd_center_info_csv,
        sweep_candidates=args.sweep_candidates,
    )

    print('Loading POS data...')
    df = load_pos_data(cfg.pos_csv_path)
    print(f'Loaded {len(df):,} POS rows')

    print('Aggregating to category-day (NYC daily context)...')
    daily = aggregate_daily_item_demand(df)

    # US/NY holidays
    min_date = pd.to_datetime(daily['date']).min().normalize()
    max_date = pd.to_datetime(daily['date']).max().normalize() + pd.Timedelta(days=cfg.forecast_horizon_days)
    print('Loading US/NY holidays and enriching features...')
    holidays = compute_us_ny_holidays(min_date, max_date)
    daily = enrich_with_holidays(daily, holidays)
    print(f'Prepared {len(daily):,} category-day rows')

    print('Training model...')
    model, mae, comp_df = train_and_evaluate(daily, cfg.random_state, cfg.output_dir, cfg.sweep_candidates)
    print(f'Validation MAE: {mae:.3f}')

    print('Writing evaluation report...')
    eval_dir = write_evaluation_report(daily, model, cfg.service_level, cfg.output_dir)
    print(f'Eval files written to: {eval_dir}')

    print('Forecasting next days...')
    forecasts = forecast_next_days(model, daily, cfg.forecast_horizon_days, holidays)

    print('Estimating global elasticity from fd_train (optional)...')
    global_elast = estimate_global_elasticity_from_fd(cfg.fd_train_csv_path)
    if global_elast is not None:
        print(f'Global elasticity (fd_train) ~ {global_elast:.3f}')

    print('Building recommendations...')
    rec_path, ss = build_recommendations(
        forecasts, daily, model,
        cfg.service_level,
        cfg.discount_windows_minutes,
        cfg.discount_levels,
        cfg.close_hour_local,
        cfg.output_dir,
        global_elasticity=global_elast,
        fd_meal_info_csv_path=cfg.fd_meal_info_csv_path,
        fd_center_info_csv_path=cfg.fd_center_info_csv_path,
        fd_train_csv_path=cfg.fd_train_csv_path,
    )

    print(f'Recommendations saved to: {rec_path}')
    print(f'Global safety stock units: {ss:.2f}')


if __name__ == '__main__':
    main() 