# src/feature_engineering.py
import pandas as pd
import numpy as np
import ta
from ta import add_all_ta_features
from ta.utils import dropna
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator, StochasticOscillator, TSIIndicator
from ta.trend import MACD, ADXIndicator, CCIIndicator
from ta.volume import VolumeWeightedAveragePrice, OnBalanceVolumeIndicator

class FeatureEngineer:
    def __init__(self):
        pass
    
    def calculate_technical_indicators(self, df):
        """Calculate various technical indicators from OHLCV data"""
        df = df.copy()
        
        # Ensure we have the required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Calculate returns
        df['ret_1d'] = df['Close'].pct_change()
        df['ret_5d'] = df['Close'].pct_change(5)
        df['ret_10d'] = df['Close'].pct_change(10)
        df['ret_21d'] = df['Close'].pct_change(21)  # approx 1 month
        
        # Calculate volatility
        df['vol_5d'] = df['ret_1d'].rolling(5).std()
        df['vol_10d'] = df['ret_1d'].rolling(10).std()
        df['vol_21d'] = df['ret_1d'].rolling(21).std()
        
        # Calculate momentum indicators
        df['rsi_14'] = RSIIndicator(df['Close'], window=14).rsi()
        df['rsi_21'] = RSIIndicator(df['Close'], window=21).rsi()
        
        # MACD
        macd = MACD(df['Close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # Bollinger Bands
        bb = BollingerBands(df['Close'])
        df['bb_high'] = bb.bollinger_hband()
        df['bb_low'] = bb.bollinger_lband()
        df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['Close']
        
        # Volume indicators
        df['volume_zscore'] = (df['Volume'] - df['Volume'].rolling(20).mean()) / df['Volume'].rolling(20).std()
        df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        
        # On Balance Volume
        df['obv'] = OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
        
        # True Strength Index
        df['tsi'] = TSIIndicator(df['Close']).tsi()
        
        # Add day of week, month, and quarter
        if 'Date' in df.columns:
            df['date'] = pd.to_datetime(df['Date'])
            df['day_of_week'] = df['date'].dt.dayofweek
            df['month'] = df['date'].dt.month
            df['quarter'] = df['date'].dt.quarter
        
        # Drop rows with NaN values created by indicators
        df = df.dropna()
        
        return df
    
    def create_target_variable(self, df, prediction_horizon=1, task_type='classification'):
        """
        Create target variable for prediction
        
        Parameters:
        - prediction_horizon: Number of days ahead to predict
        - task_type: 'classification' for direction, 'regression' for return value
        """
        df = df.copy()
        
        if task_type == 'classification':
            # Predict direction of next day's return
            df['target'] = np.sign(df['Close'].shift(-prediction_horizon) / df['Close'] - 1)
            df['target'] = df['target'].replace({-1: 0, 1: 1})  # Convert to binary: 0=down, 1=up
        elif task_type == 'regression':
            # Predict the actual return
            df['target'] = df['Close'].shift(-prediction_horizon) / df['Close'] - 1
        else:
            raise ValueError("task_type must be 'classification' or 'regression'")
        
        # Drop rows where target is NaN (end of dataset)
        df = df.dropna(subset=['target'])
        
        return df
    
    def add_sentiment_features(self, price_df, sentiment_df):
        """Merge sentiment features with price data and create lagged features"""
        # Ensure dates are in datetime format
        price_df = price_df.copy()
        sentiment_df = sentiment_df.copy()
        
        price_df['date'] = pd.to_datetime(price_df['Date'])
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
        
        # Merge sentiment data
        merged_df = pd.merge(price_df, sentiment_df, on=['date', 'Ticker'], how='left')
        
        # Fill missing sentiment values with mean or zero
        sentiment_cols = ['sent_mean', 'sent_std', 'pct_pos', 'pct_neg', 'tweet_count', 'unique_users']
        for col in sentiment_cols:
            merged_df[col] = merged_df[col].fillna(0)
        
        # Create lagged sentiment features to avoid lookahead bias
        # We use sentiment from previous day to predict today's price movement
        for col in sentiment_cols:
            merged_df[f'{col}_lag1'] = merged_df[col].shift(1)
        
        # Create momentum features for sentiment
        merged_df['sent_momentum_3d'] = merged_df['sent_mean'].rolling(3).mean()
        merged_df['sent_momentum_3d_lag1'] = merged_df['sent_momentum_3d'].shift(1)
        
        # Drop original sentiment columns (we use lagged versions)
        merged_df = merged_df.drop(columns=sentiment_cols + ['sent_momentum_3d'])
        
        return merged_df.dropna()
