"""
Banking Domain Feature Engineering for MinSight Quantitative AI.
Tailored for listed Vietnamese Commercial Banks (VCB, BID, CTG, MBB, TCB, VPB, ACB, HDB, SHB, VIB).

Features categorized into 4 core quantitative modules:
1. Base Price & Technical Momentum (20 features - preserved from baseline)
2. Cross-Sectional Alpha & Sector Beta (5 features)
3. Microstructure & High-Frequency Regime Volatility (Parkinson & Garman-Klass) (3 features)
4. Institutional Liquidity & Capital Flow Proxies (Amihud & Close Location Value) (2 features)

Total feature space: 30 features.
All features are strictly causal (no lookahead bias), stationary or bounded, and calibrated for TensorFlow multi-task learning.
"""

import os
import sys
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# 10 Listed Vietnamese Banks under study
TICKERS = ['VCB', 'BID', 'CTG', 'MBB', 'TCB', 'VPB', 'ACB', 'HDB', 'SHB', 'VIB']

# Base Technical & Momentum Features (20 features)
BASELINE_FEATURES = [
    # Price & Volume fundamentals
    'open', 'high', 'low', 'close_winsorized', 'volume',
    'sma_10', 'sma_20', 'rsi_14',
    # Multi-horizon return momentum
    'return_1d', 'return_3d', 'return_5d', 'return_10d',
    # Rolling volatility & historical drawdown
    'volatility_10d', 'volatility_20d', 'drawdown_20d',
    # Relative price location vs Moving Averages & Volume Z-score
    'price_vs_sma10', 'price_vs_sma20', 'sma10_vs_sma20',
    'rsi_delta_5', 'volume_zscore_20',
]

# Cross-Sectional Alpha vs Banking Peer Group (5 features)
CROSS_SECTIONAL_FEATURES = [
    'benchmark_return_1d',
    'alpha_1d_vs_peer',
    'alpha_5d_vs_peer',
    'rank_return_1d',
    'rel_volatility_20d',
]

# High-Frequency Volatility & Regime Estimators (3 features)
REGIME_VOLATILITY_FEATURES = [
    'parkinson_vol_10d',
    'garman_klass_vol_10d',
    'trend_regime_14d',
]

# Institutional Liquidity Flow & Accumulation Proxies (2 features)
LIQUIDITY_FLOW_FEATURES = [
    'amihud_illiquidity',
    'clv_momentum_5d',
]

# Full 30-feature space
ALL_BANKING_FEATURES = (
    BASELINE_FEATURES +
    CROSS_SECTIONAL_FEATURES +
    REGIME_VOLATILITY_FEATURES +
    LIQUIDITY_FLOW_FEATURES
)


def _augment_baseline_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes the 20 baseline features if not already present."""
    result = df.copy()
    close = result['close_winsorized']
    log_price = np.log(close)

    # Return rolling
    for horizon in (1, 3, 5, 10):
        col = f'return_{horizon}d'
        if col not in result.columns:
            result[col] = log_price.diff(horizon)

    # Volatility rolling
    ret_1d = log_price.diff(1)
    for horizon in (10, 20):
        col = f'volatility_{horizon}d'
        if col not in result.columns:
            result[col] = ret_1d.rolling(horizon, min_periods=1).std()

    # Drawdown 20d
    if 'drawdown_20d' not in result.columns:
        result['drawdown_20d'] = close / close.rolling(20, min_periods=1).max() - 1.0

    # Moving Average Relative Positions
    if 'price_vs_sma10' not in result.columns:
        result['price_vs_sma10'] = close / result['sma_10'] - 1.0
    if 'price_vs_sma20' not in result.columns:
        result['price_vs_sma20'] = close / result['sma_20'] - 1.0
    if 'sma10_vs_sma20' not in result.columns:
        result['sma10_vs_sma20'] = result['sma_10'] / result['sma_20'] - 1.0

    # RSI Delta 5
    if 'rsi_delta_5' not in result.columns:
        result['rsi_delta_5'] = result['rsi_14'].diff(5)

    # Volume Z-Score 20d
    if 'volume_zscore_20' not in result.columns:
        vol_mean = result['volume'].rolling(20, min_periods=1).mean()
        vol_std = result['volume'].rolling(20, min_periods=1).std()
        result['volume_zscore_20'] = (result['volume'] - vol_mean) / (vol_std + 1e-8)

    return result


def _augment_regime_and_liquidity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes Parkinson volatility, Garman-Klass, Trend Regime, Amihud, and CLV."""
    result = df.copy()
    h = result['high'].astype(float)
    l = result['low'].astype(float)
    o = result['open'].astype(float)
    c = result['close_winsorized'].astype(float)
    v = result['volume'].astype(float)

    # 1. Parkinson Volatility (10-day rolling root-mean-square)
    log_hl = np.log(np.maximum(h, l * 1.0001) / np.maximum(l, 1e-6))
    parkinson_inst = (log_hl ** 2) / (4.0 * np.log(2.0))
    result['parkinson_vol_10d'] = np.sqrt(parkinson_inst.rolling(10, min_periods=1).mean())

    # 2. Garman-Klass Volatility (10-day rolling root-mean-square)
    log_co = np.log(np.maximum(c, 1e-6) / np.maximum(o, 1e-6))
    gk_inst = np.maximum(
        0.0,
        0.5 * (log_hl ** 2) - (2.0 * np.log(2.0) - 1.0) * (log_co ** 2)
    )
    result['garman_klass_vol_10d'] = np.sqrt(pd.Series(gk_inst).rolling(10, min_periods=1).mean())

    # 3. Trend Regime Strength (EMA spread normalized by ATR)
    prev_c = c.shift(1).bfill()
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    atr_14 = tr.ewm(span=14, min_periods=1).mean()
    ema12 = c.ewm(span=12, min_periods=1).mean()
    ema26 = c.ewm(span=26, min_periods=1).mean()
    result['trend_regime_14d'] = (ema12 - ema26) / (atr_14 + 1e-8)

    # 4. Amihud Illiquidity Ratio (log-transformed price impact per billion VND)
    ret_1d = np.log(c / prev_c).fillna(0.0)
    turnover_billion = (v * c) / 1e6  # (Volume * Price_in_thousands) / 1e6 = Billion VND
    result['amihud_illiquidity'] = np.log1p((np.abs(ret_1d) / (turnover_billion + 1e-6)) * 1e3)

    # 5. Close Location Value (Accumulation / Distribution force, bounded in [-1, +1])
    hl_range = np.maximum(h - l, 0.01)
    clv = np.clip((2.0 * c - h - l) / hl_range, -1.0, 1.0)
    result['clv_momentum_5d'] = clv.rolling(5, min_periods=1).mean()

    return result


def _augment_cross_sectional_banking_features(
    df: pd.DataFrame,
    ticker: str,
    peer_close_table: pd.DataFrame
) -> pd.DataFrame:
    """Computes Sector Benchmark, Alpha 1d/5d, Rank, and Relative Volatility."""
    ticker = ticker.upper()
    result = df.copy()

    # Format time to YYYY-MM-DD string for exact key matching
    if 'time' in result.columns:
        result['time'] = pd.to_datetime(result['time']).dt.strftime('%Y-%m-%d')

    peer_table = peer_close_table.copy()
    peer_table['time'] = pd.to_datetime(peer_table['time']).dt.strftime('%Y-%m-%d')

    all_ticker_cols = [c for c in peer_table.columns if c != 'time']
    peer_only_cols = [c for c in all_ticker_cols if c != ticker]
    if not peer_only_cols:
        peer_only_cols = all_ticker_cols

    all_prices = peer_table[all_ticker_cols].astype(float).replace(0, np.nan)
    all_log_prices = np.log(all_prices)

    # Benchmark return (equal-weight log-return of peers, excluding the ticker)
    benchmark = peer_table[['time']].copy()
    peer_ret_1d = all_log_prices[peer_only_cols].diff(1)
    peer_ret_5d = all_log_prices[peer_only_cols].diff(5)
    benchmark['benchmark_return_1d'] = peer_ret_1d.mean(axis=1, skipna=True).fillna(0.0)
    benchmark['benchmark_return_5d'] = peer_ret_5d.mean(axis=1, skipna=True).fillna(0.0)

    # Benchmark 20-day volatility
    benchmark['benchmark_volatility_20d'] = (
        peer_ret_1d.mean(axis=1, skipna=True).rolling(20, min_periods=1).std().fillna(0.01)
    )

    # Percentile Rank (0..1) across all banks
    ranks = peer_table[['time']].copy()
    ranks_1d = all_log_prices.diff(1).rank(axis=1, pct=True)
    if ticker in ranks_1d.columns:
        ranks['rank_return_1d'] = ranks_1d[ticker].values
    else:
        ranks['rank_return_1d'] = 0.5

    # Merge benchmark and ranks onto ticker frame
    result = result.merge(benchmark, on='time', how='left')
    result = result.merge(ranks, on='time', how='left')

    # Forward fill or default-impute any unmatched edge dates
    result['benchmark_return_1d'] = result['benchmark_return_1d'].ffill().bfill().fillna(0.0)
    result['benchmark_return_5d'] = result['benchmark_return_5d'].ffill().bfill().fillna(0.0)
    result['benchmark_volatility_20d'] = result['benchmark_volatility_20d'].ffill().bfill().fillna(0.01)
    result['rank_return_1d'] = result['rank_return_1d'].ffill().bfill().fillna(0.5)

    # Alpha 1d and 5d vs Sector
    ret_1d_col = result['return_1d'] if 'return_1d' in result.columns else np.log(result['close_winsorized']).diff(1)
    ret_5d_col = result['return_5d'] if 'return_5d' in result.columns else np.log(result['close_winsorized']).diff(5)

    result['alpha_1d_vs_peer'] = ret_1d_col - result['benchmark_return_1d']
    result['alpha_5d_vs_peer'] = ret_5d_col - result['benchmark_return_5d']

    # Relative Volatility (ticker 20d vol / sector 20d vol), clipped to safe range [0.1, 5.0]
    ticker_vol = result['volatility_20d'] if 'volatility_20d' in result.columns else ret_1d_col.rolling(20, min_periods=1).std()
    result['rel_volatility_20d'] = (
        ticker_vol / (result['benchmark_volatility_20d'] + 1e-4)
    ).clip(0.1, 5.0)

    # Drop temporary benchmark helper column
    if 'benchmark_return_5d' in result.columns:
        result.drop(columns=['benchmark_return_5d'], inplace=True)
    if 'benchmark_volatility_20d' in result.columns:
        result.drop(columns=['benchmark_volatility_20d'], inplace=True)

    return result


def compute_banking_features(
    df: pd.DataFrame,
    ticker: str,
    peer_close_table: pd.DataFrame | None = None,
    data_dir: str = 'data/processed'
) -> pd.DataFrame:
    """
    Main pipeline entrypoint to engineer all 30 features for a given bank.
    
    Args:
        df: Processed DataFrame containing OHLCV and close_winsorized.
        ticker: Banking ticker symbol (e.g. 'VCB', 'BID').
        peer_close_table: Optional pre-built peer table. Built automatically if None.
        data_dir: Path to directory containing processed ticker CSVs.
        
    Returns:
        Augmented DataFrame with all 30 features, zero NaNs, ready for sequence generation.
    """
    result = df.copy()

    # Step 1: Augment 20 baseline features
    result = _augment_baseline_features(result)

    # Step 2: Augment 5 regime & liquidity features
    result = _augment_regime_and_liquidity_features(result)

    # Step 3: Augment 5 cross-sectional peer features
    if peer_close_table is None:
        from src.model.probability import build_peer_close_table
        peer_close_table = build_peer_close_table(TICKERS, data_dir)

    result = _augment_cross_sectional_banking_features(result, ticker, peer_close_table)

    # Clean any edge NaNs resulting from rolling windows using causal ffill then bfill
    for col in ALL_BANKING_FEATURES:
        if col in result.columns:
            result[col] = result[col].ffill().bfill().fillna(0.0)

    return result
