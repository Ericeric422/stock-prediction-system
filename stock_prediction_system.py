#!/usr/bin/env python3
"""
Complete Stock Market Prediction System
=====================================

IMPORTANT DISCLAIMER:
This system is for educational and research purposes only. Stock market prediction
is extremely challenging and risky. Past performance does not guarantee future results.
Always use paper trading before risking real money. Never invest more than you can afford to lose.

Author: AI Assistant
License: Educational Use Only
"""

import warnings
warnings.filterwarnings('ignore')

import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
import yfinance as yf
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import json
import pickle
from pathlib import Path

import matplotlib
# Conditionally set backend based on command-line arguments
# If '--start-api' is present, use a non-interactive backend
if '--start-api' in sys.argv:
    matplotlib.use('Agg')

# ML Libraries
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import xgboost as xgb

# Deep Learning
try:
    import tensorflow as tf
    from tensorflow import keras
    from keras.models import Sequential, Model
    from keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Input, MultiHeadAttention, LayerNormalization
    from keras.optimizers import Adam
    from keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
    print("TensorFlow available - Deep learning models enabled")
except ImportError:
    try:
        # Fallback for older TensorFlow versions
        import tensorflow as tf
        from tensorflow.keras.models import Sequential, Model
        from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Input
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping
        TENSORFLOW_AVAILABLE = True
        print("TensorFlow (legacy imports) available - Deep learning models enabled")
    except ImportError:
        TENSORFLOW_AVAILABLE = False
        print("TensorFlow not available. Deep learning models will be skipped.")
        print("To enable deep learning: pip install tensorflow")

# Explainability
try:
    import shap
    SHAP_AVAILABLE = True
    print("SHAP available - Model explanations enabled")
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not available. Model explanations will be skipped.")

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# API
from flask import Flask, jsonify, request, render_template
import threading
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stock_prediction.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class DataCollector:
    """Handles data collection from multiple sources"""
    
    def __init__(self, alpha_vantage_key: str = None):
        self.av_key = alpha_vantage_key or os.getenv('ALPHA_VANTAGE_KEY')
        self.cache_dir = Path('data_cache')
        self.cache_dir.mkdir(exist_ok=True)
        
    def get_stock_data(self, symbol: str, period: str = '5y') -> pd.DataFrame:
        """Fetch stock data from Yahoo Finance"""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            if data.empty:
                logger.warning(f"No data retrieved for {symbol}")
                return pd.DataFrame()
            
            # Add symbol column
            data['Symbol'] = symbol
            logger.info(f"Retrieved {len(data)} rows for {symbol}")
            return data
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_multiple_stocks(self, symbols: List[str], period: str = '5y') -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple stocks"""
        data = {}
        for symbol in symbols:
            stock_data = self.get_stock_data(symbol, period)
            if not stock_data.empty:
                data[symbol] = stock_data
        return data
    
    def get_economic_indicators(self) -> pd.DataFrame:
        """Fetch basic economic indicators (simulated data for demo)"""
        # In production, integrate with FRED API
        # For demo, create simulated data
        dates = pd.date_range(start='2019-01-01', end=datetime.now(), freq='D')
        np.random.seed(42)
        
        economic_data = pd.DataFrame({
            'VIX': np.random.normal(20, 5, len(dates)),
            'DXY': np.random.normal(100, 2, len(dates)),
            '10Y_Treasury': np.random.normal(2.5, 0.5, len(dates)),
            'Oil_Price': np.random.normal(70, 10, len(dates))
        }, index=dates)
        
        return economic_data

class FeatureEngineer:
    """Creates features for machine learning models"""
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        return pd.DataFrame({
            'MACD': macd_line,
            'MACD_Signal': signal_line,
            'MACD_Histogram': histogram
        })
    
    @staticmethod
    def calculate_bollinger_bands(prices: pd.Series, window: int = 20, num_std: int = 2) -> pd.DataFrame:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        
        return pd.DataFrame({
            'BB_Upper': sma + (std * num_std),
            'BB_Lower': sma - (std * num_std),
            'BB_Middle': sma,
            'BB_Width': (sma + (std * num_std)) - (sma - (std * num_std)),
            'BB_Position': (prices - (sma - (std * num_std))) / ((sma + (std * num_std)) - (sma - (std * num_std)))
        })
    
    def create_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive technical indicators"""
        df = df.copy()
        
        # Price-based features
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Price_Change'] = df['Close'] - df['Open']
        df['True_Range'] = np.maximum(df['High'] - df['Low'], 
                                     np.maximum(abs(df['High'] - df['Close'].shift(1)),
                                              abs(df['Low'] - df['Close'].shift(1))))
        
        # Moving averages
        for window in [5, 10, 20, 50, 200]:
            df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()
            df[f'EMA_{window}'] = df['Close'].ewm(span=window).mean()
            df[f'Price_to_SMA_{window}'] = df['Close'] / df[f'SMA_{window}']
        
        # Technical indicators
        df['RSI'] = self.calculate_rsi(df['Close'])
        
        # MACD
        macd_data = self.calculate_macd(df['Close'])
        for col in macd_data.columns:
            df[col] = macd_data[col]
        
        # Bollinger Bands
        bb_data = self.calculate_bollinger_bands(df['Close'])
        for col in bb_data.columns:
            df[col] = bb_data[col]
        
        # Volatility
        for window in [10, 30]:
            df[f'Volatility_{window}'] = df['Returns'].rolling(window=window).std()
        
        # Volume indicators
        df['Volume_SMA_10'] = df['Volume'].rolling(window=10).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_10']
        df['Price_Volume'] = df['Close'] * df['Volume']
        
        # Momentum indicators
        for window in [5, 10, 20]:
            df[f'Momentum_{window}'] = df['Close'] / df['Close'].shift(window)
            df[f'ROC_{window}'] = (df['Close'] - df['Close'].shift(window)) / df['Close'].shift(window)
        
        # Lag features
        for lag in [1, 2, 3, 5]:
            df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
            df[f'Returns_Lag_{lag}'] = df['Returns'].shift(lag)
        
        # Rolling statistics
        for window in [5, 10, 20]:
            df[f'High_Low_Ratio_{window}'] = df['High'].rolling(window=window).mean() / df['Low'].rolling(window=window).mean()
            df[f'Close_Std_{window}'] = df['Close'].rolling(window=window).std()
            df[f'Volume_Std_{window}'] = df['Volume'].rolling(window=window).std()
        
        # Time-based features
        df['Day_of_Week'] = df.index.dayofweek
        df['Month'] = df.index.month
        df['Quarter'] = df.index.quarter
        
        return df
    
    def create_targets(self, df: pd.DataFrame, days_ahead: int = 5) -> pd.DataFrame:
        """Create prediction targets"""
        df = df.copy()
        
        # Future price
        df['Future_Close'] = df['Close'].shift(-days_ahead)
        
        # Binary classification: price direction
        df['Target_Binary'] = (df['Future_Close'] > df['Close']).astype(int)
        
        # Multi-class classification
        df['Price_Change_Pct'] = (df['Future_Close'] - df['Close']) / df['Close']
        df['Target_Multi'] = pd.cut(df['Price_Change_Pct'], 
                                   bins=[-np.inf, -0.02, -0.005, 0.005, 0.02, np.inf],
                                   labels=[0, 1, 2, 3, 4])  # Strong down, down, neutral, up, strong up
        
        # Regression target
        df['Target_Price'] = df['Future_Close']
        
        return df

class ModelTrainer:
    """Handles model training and evaluation"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.results = {}
        self.full_dataset_for_explainer = None
    
    def prepare_data(self, df: pd.DataFrame, target_col: str = 'Target_Binary', 
                    feature_selection: bool = True, n_features: int = 20) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare data for training"""
        # Remove rows with NaN in target
        df_clean = df.dropna(subset=[target_col])
        
        # Select feature columns (exclude target and future columns)
        exclude_cols = ['Symbol', 'Target_Binary', 'Target_Multi', 'Target_Price', 'Future_Close', 'Price_Change_Pct']
        feature_cols = [col for col in df_clean.columns if col not in exclude_cols and not col.startswith('Future_')]
        
        X = df_clean[feature_cols].fillna(0)  # Fill remaining NaNs with 0
        y = df_clean[target_col]
        
        # Remove infinite values
        X = X.replace([np.inf, -np.inf], 0)
        
        # Feature selection
        if feature_selection and len(feature_cols) > n_features:
            selector = SelectKBest(score_func=mutual_info_classif, k=n_features)
            X_selected = selector.fit_transform(X, y)
            selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
            self.feature_selectors[target_col] = selector
            return X_selected, y.values, selected_features
        
        return X.values, y.values, feature_cols
    
    def train_traditional_models(self, X: np.ndarray, y: np.ndarray):
        """Train traditional ML models"""
        # Scale features
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers['traditional'] = scaler
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Models to train
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
            'XGBoost': xgb.XGBClassifier(n_estimators=100, max_depth=6, random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(kernel='rbf', probability=True, random_state=42)
        }
        
        results = {}
        trained_models = {}
        
        for name, model in models.items():
            logger.info(f"Training {name}...")
            scores = []
            
            for train_idx, val_idx in tscv.split(X_scaled):
                X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                model.fit(X_train, y_train)
                score = model.score(X_val, y_val)
                scores.append(score)
            
            avg_score = np.mean(scores)
            std_score = np.std(scores)
            
            results[name] = {
                'cv_score_mean': avg_score,
                'cv_score_std': std_score,
                'scores': scores
            }
            
            # Train final model on all data
            model.fit(X_scaled, y)
            trained_models[name] = model
            
            logger.info(f"{name} CV Score: {avg_score:.4f} (+/- {std_score:.4f})")
        
        self.models.update(trained_models)
        self.results.update(results)
        
        return results
    
    def train_lstm_model(self, X: np.ndarray, y: np.ndarray, sequence_length: int = 60):
        """Train LSTM model"""
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available. Skipping LSTM training.")
            return {}
        
        try:
            # Prepare sequences
            X_sequences, y_sequences = self.create_sequences(X, y, sequence_length)
            
            if len(X_sequences) == 0:
                logger.warning("Not enough data for LSTM sequences")
                return {}
            
            # Split data
            split_idx = int(len(X_sequences) * 0.8)
            X_train, X_val = X_sequences[:split_idx], X_sequences[split_idx:]
            y_train, y_val = y_sequences[:split_idx], y_sequences[split_idx:]
            
            # Build LSTM model
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(sequence_length, X.shape[1])),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            
            model.compile(optimizer=Adam(learning_rate=0.001), 
                         loss='binary_crossentropy', 
                         metrics=['accuracy'])
            
            # Early stopping
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            
            # Train model
            history = model.fit(
                X_train, y_train,
                batch_size=32,
                epochs=50,
                validation_data=(X_val, y_val),
                callbacks=[early_stopping],
                verbose=0
            )
            
            # Evaluate
            val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
            
            self.models['LSTM'] = model
            self.results['LSTM'] = {
                'val_accuracy': val_acc,
                'val_loss': val_loss,
                'history': history.history
            }
            
            logger.info(f"LSTM Validation Accuracy: {val_acc:.4f}")
            return self.results['LSTM']
            
        except Exception as e:
            logger.error(f"Error training LSTM: {e}")
            return {}
    
    def create_sequences(self, X: np.ndarray, y: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM"""
        if len(X) < sequence_length:
            return np.array([]), np.array([])
        
        X_sequences, y_sequences = [], []
        for i in range(sequence_length, len(X)):
            X_sequences.append(X[i-sequence_length:i])
            y_sequences.append(y[i])
        
        return np.array(X_sequences), np.array(y_sequences)
    
    def create_ensemble(self) -> VotingClassifier:
        """Create ensemble model"""
        estimators = []
        for name, model in self.models.items():
            if name != 'LSTM':  # Exclude LSTM from sklearn ensemble
                estimators.append((name, model))
        
        if len(estimators) > 1:
            ensemble = VotingClassifier(estimators=estimators, voting='soft')
            self.models['Ensemble'] = ensemble
            return ensemble
        
        return None

class BacktestingFramework:
    """Comprehensive backtesting and evaluation"""
    
    def __init__(self):
        self.results = {}
    
    def calculate_financial_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate financial performance metrics"""
        if len(returns) == 0 or returns.std() == 0:
            return {}
        
        total_return = (1 + returns).prod() - 1
        annual_return = (1 + returns.mean()) ** 252 - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Win rate
        win_rate = (returns > 0).mean()
        
        return {
            'Total_Return': total_return,
            'Annual_Return': annual_return,
            'Volatility': volatility,
            'Sharpe_Ratio': sharpe_ratio,
            'Max_Drawdown': max_drawdown,
            'Win_Rate': win_rate
        }
    
    def backtest_strategy(self, predictions: np.ndarray, actual_returns: pd.Series, 
                         transaction_cost: float = 0.001) -> Dict[str, float]:
        """Backtest trading strategy"""
        if len(predictions) != len(actual_returns):
            logger.error("Predictions and returns length mismatch")
            return {}
        
        # Simple strategy: go long when prediction is 1, hold cash when 0
        strategy_returns = []
        
        for i, pred in enumerate(predictions):
            if pred == 1:
                # Go long, subtract transaction cost
                ret = actual_returns.iloc[i] - transaction_cost
            else:
                # Hold cash
                ret = 0
            strategy_returns.append(ret)
        
        strategy_returns = pd.Series(strategy_returns, index=actual_returns.index)
        return self.calculate_financial_metrics(strategy_returns)
    
    def comprehensive_evaluation(self, model, X_test: np.ndarray, y_test: np.ndarray, 
                                actual_returns: pd.Series) -> Dict[str, any]:
        """Comprehensive model evaluation"""
        try:
            # Predictions
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                y_pred = (y_pred_proba > 0.5).astype(int)
            else:
                y_pred = model.predict(X_test)
                y_pred_proba = None
            
            # Classification metrics
            accuracy = (y_pred == y_test).mean()
            
            # Financial metrics
            financial_metrics = self.backtest_strategy(y_pred, actual_returns)
            
            # Combine results
            results = {
                'accuracy': accuracy,
                'predictions': y_pred,
                'prediction_probabilities': y_pred_proba,
                'financial_metrics': financial_metrics
            }
            
            # Add AUC if probabilities available
            if y_pred_proba is not None:
                try:
                    auc = roc_auc_score(y_test, y_pred_proba)
                    results['auc'] = auc
                except:
                    results['auc'] = 0.5
            
            return results
            
        except Exception as e:
            logger.error(f"Error in evaluation: {e}")
            return {}

class ProductionPredictor:
    """Production-ready prediction system"""
    
    def __init__(self, model_path: str = 'models'):
        self.model_path = Path(model_path)
        self.model_path.mkdir(exist_ok=True)
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.explainers = {}
        self.feature_names = []
        self.data_collector = DataCollector()
        self.feature_engineer = FeatureEngineer()
        
    def save_models(self, trainer: ModelTrainer, feature_names: List[str]):
        """Save trained models"""
        # Save sklearn models
        for name, model in trainer.models.items():
            if name != 'LSTM':
                model_file = self.model_path / f'{name}_model.pkl'
                with open(model_file, 'wb') as f:
                    pickle.dump(model, f)
                
                # Create and save SHAP explainer for tree models
                if SHAP_AVAILABLE and hasattr(model, 'predict_proba') and name in ['RandomForest', 'XGBoost']:
                    # Need to scale the training data used for the explainer
                    X, _, _ = trainer.prepare_data(trainer.full_dataset_for_explainer, target_col='Target_Binary', feature_selection=True)
                    X_scaled = trainer.scalers['traditional'].transform(X)
                    explainer = shap.TreeExplainer(model, X_scaled)
                    explainer_file = self.model_path / f'{name}_explainer.pkl'
                    with open(explainer_file, 'wb') as f:
                        pickle.dump(explainer, f)
        
        # Save LSTM separately if available
        if 'LSTM' in trainer.models and TENSORFLOW_AVAILABLE:
            lstm_path = self.model_path / 'lstm_model.keras'
            trainer.models['LSTM'].save(lstm_path)
        
        # Save scalers and feature selectors
        for name, scaler in trainer.scalers.items():
            scaler_file = self.model_path / f'{name}_scaler.pkl'
            with open(scaler_file, 'wb') as f:
                pickle.dump(scaler, f)
        
        for name, selector in trainer.feature_selectors.items():
            selector_file = self.model_path / f'{name}_selector.pkl'
            with open(selector_file, 'wb') as f:
                pickle.dump(selector, f)
        
        # Save feature names
        feature_file = self.model_path / 'feature_names.pkl'
        with open(feature_file, 'wb') as f:
            pickle.dump(feature_names, f)
        
        logger.info(f"Models saved to {self.model_path}")
    
    def load_models(self):
        """Load saved models"""
        try:
            # Load sklearn models
            for model_file in self.model_path.glob('*_model.pkl'):
                name = model_file.stem.replace('_model', '')
                with open(model_file, 'rb') as f:
                    self.models[name] = pickle.load(f)
            
            # Load LSTM if available
            lstm_path = self.model_path / 'lstm_model.keras'
            if lstm_path.exists() and TENSORFLOW_AVAILABLE:
                if hasattr(tf.keras, 'models'):
                    self.models['LSTM'] = tf.keras.models.load_model(lstm_path)
                else:
                    # Fallback for different TensorFlow versions
                    from keras.models import load_model
                    self.models['LSTM'] = load_model(lstm_path)
            
            # Load scalers
            for scaler_file in self.model_path.glob('*_scaler.pkl'):
                name = scaler_file.stem.replace('_scaler', '')
                with open(scaler_file, 'rb') as f:
                    self.scalers[name] = pickle.load(f)
            
            # Load feature selectors
            for selector_file in self.model_path.glob('*_selector.pkl'):
                name = selector_file.stem.replace('_selector', '')
                with open(selector_file, 'rb') as f:
                    self.feature_selectors[name] = pickle.load(f)
            
            # Load explainers
            if SHAP_AVAILABLE:
                for explainer_file in self.model_path.glob('*_explainer.pkl'):
                    name = explainer_file.stem.replace('_explainer', '')
                    with open(explainer_file, 'rb') as f:
                        self.explainers[name] = pickle.load(f)
            
            # Load feature names
            feature_file = self.model_path / 'feature_names.pkl'
            if feature_file.exists():
                with open(feature_file, 'rb') as f:
                    self.feature_names = pickle.load(f)
            
            logger.info(f"Models loaded from {self.model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def predict(self, symbol: str, model_name: str = 'RandomForest') -> Dict[str, any]:
        """Make prediction for a stock"""
        try:
            # Get recent data
            data = self.data_collector.get_stock_data(symbol, period='1y')
            # Check for empty or very small dataframes to avoid errors with invalid symbols
            if data.empty or len(data) < 50: # Need at least 50 days for feature calculation
                logger.warning(f"Insufficient data for {symbol}. It might be an invalid ticker or have too little history.")
                return {'error': f'No data available for {symbol}. It may be an invalid symbol or have insufficient history.'}
            
            # Create features
            data_with_features = self.feature_engineer.create_technical_features(data)
            
            # Get latest features
            latest_data = data_with_features.iloc[-1:][self.feature_names].fillna(0)
            latest_data = latest_data.replace([np.inf, -np.inf], 0)
            
            # Apply feature selection first
            if 'Target_Binary' in self.feature_selectors:
                selected_feature_indices = self.feature_selectors['Target_Binary'].get_support(indices=True)
                selected_feature_names = [self.feature_names[i] for i in selected_feature_indices]
                features_to_scale = latest_data[selected_feature_names]
            else:
                features_to_scale = latest_data

            # Scale features
            if 'traditional' in self.scalers:
                features_scaled = self.scalers['traditional'].transform(features_to_scale)
            else:
                features_scaled = features_to_scale.values
            
            # Ensure proper dtype and shape for downstream models (esp. XGBoost)
            try:
                import numpy as _np
                if not isinstance(features_scaled, _np.ndarray):
                    features_scaled = _np.asarray(features_scaled)
                # Cast to float32 to avoid dtype issues with some xgboost builds
                features_scaled = features_scaled.astype(_np.float32, copy=False)
                # Guarantee 2D shape (n_samples, n_features)
                if features_scaled.ndim == 1:
                    features_scaled = features_scaled.reshape(1, -1)
                logger.debug(f"Prediction input shape: {features_scaled.shape}, dtype: {features_scaled.dtype}")
            except Exception as prep_e:
                logger.warning(f"Feature array prep warning for {symbol}: {prep_e}")
            
            # Make prediction
            if model_name not in self.models:
                return {'error': f'Model {model_name} not available'}
            
            model = self.models[model_name]
            
            # Extra diagnostics for XGBoost path
            logger.debug(f"Using model: {model_name} | type: {type(model)}")
            try:
                if hasattr(model, 'predict_proba'):
                    prob_array = model.predict_proba(features_scaled)
                    # Choose correct column for positive class (label 1) when available
                    pos_idx = None
                    try:
                        classes = getattr(model, 'classes_', None)
                        if classes is not None:
                            import numpy as _np
                            # Find index of class label 1 (binary positive)
                            match = _np.where(classes == 1)[0]
                            if len(match) > 0:
                                pos_idx = int(match[0])
                    except Exception as _e:
                        logger.debug(f"Could not read classes_ for {model_name}: {_e}")

                    if pos_idx is None:
                        # Fallback to last column as positive class if unknown
                        try:
                            pos_idx = prob_array.shape[1] - 1
                        except Exception:
                            pos_idx = 0

                    try:
                        prob = float(prob_array[0, pos_idx])
                    except Exception:
                        # Fallback: handle models returning 1-column or list
                        if hasattr(prob_array, 'shape') and len(getattr(prob_array, 'shape', [])) >= 2 and prob_array.shape[-1] == 1:
                            prob = float(prob_array[0, 0])
                        else:
                            prob = float(getattr(prob_array, 'ravel', lambda: [0.5])()[0])
                    prediction = int(prob > 0.5)
                else:
                    prediction = int(model.predict(features_scaled)[0])
                    # If no proba available, set a neutral probability
                    prob = 0.5
            except Exception as pred_e:
                logger.error(f"Prediction error for {symbol} with model {model_name}: {pred_e}")
                return {'error': f'Prediction failed for model {model_name}: {str(pred_e)}'}
            
            # Try to fetch company name for better UX
            company_name = None
            try:
                ticker = yf.Ticker(symbol)
                get_info = getattr(ticker, 'get_info', None)
                info = get_info() if callable(get_info) else getattr(ticker, 'info', {})
                if isinstance(info, dict):
                    company_name = info.get('longName') or info.get('shortName') or info.get('name')
            except Exception:
                pass

            # Prepare compact chart data (last ~180 trading days)
            try:
                chart_df = data_with_features.tail(180).copy()
                # Ensure index is datetime for label formatting
                if not isinstance(chart_df.index, pd.DatetimeIndex):
                    try:
                        chart_df.index = pd.to_datetime(chart_df.index)
                    except Exception:
                        pass

                labels = [
                    (idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx))
                    for idx in chart_df.index
                ]
                # Use None for NaNs to keep array lengths consistent for Chart.js
                def _series(values):
                    return [
                        (None if pd.isna(v) else float(v))
                        for v in values
                    ]

                chart_payload = {
                    'labels': labels,
                    'close': _series(chart_df.get('Close', pd.Series(index=chart_df.index)).values),
                    'sma20': _series(chart_df.get('SMA_20', pd.Series(index=chart_df.index)).values),
                    'sma50': _series(chart_df.get('SMA_50', pd.Series(index=chart_df.index)).values),
                }
            except Exception as _chart_e:
                logger.debug(f"Chart data prep failed for {symbol}: {_chart_e}")
                chart_payload = None

            return {
                'symbol': symbol,
                'prediction': prediction,
                'probability': prob,
                'direction': 'UP' if prediction == 1 else 'DOWN',
                'confidence': abs(prob - 0.5) * 2,
                'timestamp': datetime.now().isoformat(),
                'model_used': model_name,
                'company_name': company_name,
                'chart': chart_payload
            }
            
        except Exception as e:
            logger.error(f"Error making prediction for {symbol}: {e}")
            return {'error': str(e)}

    def explain(self, symbol: str, model_name: str = 'RandomForest') -> Dict[str, any]:
        """Generate SHAP explanation for a prediction"""
        if not SHAP_AVAILABLE:
            return {'error': 'SHAP library not installed. Cannot generate explanation.'}
        
        if model_name not in self.explainers:
            return {'error': f'Explainer for model {model_name} not available. Explanations are only available for RandomForest and XGBoost.'}

        try:
            # Get latest data (same logic as in predict)
            data = self.data_collector.get_stock_data(symbol, period='1y')
            if data.empty:
                return {'error': f'No data available for {symbol}'}
            
            data_with_features = self.feature_engineer.create_technical_features(data)
            latest_data = data_with_features.iloc[-1:][self.feature_names].fillna(0)
            latest_data = latest_data.replace([np.inf, -np.inf], 0)
            
            # Apply feature selection first
            if 'Target_Binary' in self.feature_selectors:
                selected_feature_indices = self.feature_selectors['Target_Binary'].get_support(indices=True)
                selected_feature_names = [self.feature_names[i] for i in selected_feature_indices]
                features_to_scale = latest_data[selected_feature_names]
                feature_names_for_shap = selected_feature_names
            else:
                features_to_scale = latest_data
                feature_names_for_shap = self.feature_names

            # Scale features
            if 'traditional' in self.scalers:
                features_scaled = self.scalers['traditional'].transform(features_to_scale)
            else:
                features_scaled = features_to_scale.values
            
            # Get explainer and shap values
            explainer = self.explainers[model_name]
            
            # Debug log the input features shape
            logger.debug(f"Features shape before SHAP: {features_scaled.shape}")
            
            try:
                shap_values_output = explainer.shap_values(features_scaled)
                logger.debug(f"SHAP output type: {type(shap_values_output)}")
                if hasattr(shap_values_output, 'shape'):
                    logger.debug(f"SHAP output shape: {shap_values_output.shape}")
                elif isinstance(shap_values_output, list):
                    logger.debug(f"SHAP output list length: {len(shap_values_output)}")
                    if len(shap_values_output) > 0 and hasattr(shap_values_output[0], 'shape'):
                        logger.debug(f"First element shape: {shap_values_output[0].shape}")
            except Exception as e:
                logger.error(f"Error in SHAP calculation: {str(e)}")
                return {'error': f'SHAP calculation failed: {str(e)}'}

            # Handle different outputs from SHAP explainer
            try:
                if isinstance(shap_values_output, list) and len(shap_values_output) == 2:
                    # For binary classifiers, it returns a list of two arrays (one for each class)
                    shap_values_for_positive_class = shap_values_output[1]
                    base_value = explainer.expected_value[1] if hasattr(explainer.expected_value, '__len__') and len(explainer.expected_value) > 1 else explainer.expected_value
                else:
                    # For other cases, it might return a single array
                    shap_values_for_positive_class = shap_values_output
                    base_value = explainer.expected_value[0] if hasattr(explainer.expected_value, '__len__') and len(explainer.expected_value) > 0 else explainer.expected_value

                # Debug log the shapes
                if hasattr(shap_values_for_positive_class, 'shape'):
                    logger.debug(f"SHAP values shape before processing: {shap_values_for_positive_class.shape}")
                
                # Ensure we have a 1D array for the prediction
                if hasattr(shap_values_for_positive_class, 'shape') and len(shap_values_for_positive_class.shape) > 1:
                    if shap_values_for_positive_class.shape[0] == 1:
                        # If it's a 2D array with a single row, take the first row
                        shap_values_single_prediction = shap_values_for_positive_class[0]
                        logger.debug("Took first row from 2D array")
                    else:
                        # If it's a 2D array with multiple rows, take the mean across rows
                        shap_values_single_prediction = np.mean(shap_values_for_positive_class, axis=0)
                        logger.debug("Averaged multiple rows")
                else:
                    shap_values_single_prediction = shap_values_for_positive_class
                    logger.debug("Using SHAP values as-is")
                    
                logger.debug(f"Final SHAP values shape: {np.array(shap_values_single_prediction).shape}")
                
            except Exception as e:
                logger.error(f"Error processing SHAP values: {str(e)}")
                return {'error': f'Error processing SHAP values: {str(e)}'}

            try:
                # Ensure we have a numpy array and flatten if necessary
                shap_array = np.array(shap_values_single_prediction)
                
                # If it's still 2D with shape (n, 1), squeeze it
                if len(shap_array.shape) == 2 and shap_array.shape[1] == 1:
                    shap_array = shap_array.flatten()
                # If it's 2D with multiple columns, take the first column
                elif len(shap_array.shape) == 2 and shap_array.shape[1] > 1:
                    shap_array = shap_array[:, 0]  # Take first column
                # If it's 1D but has the wrong length, take the first n elements
                elif len(shap_array) > len(feature_names_for_shap):
                    shap_array = shap_array[:len(feature_names_for_shap)]
                
                logger.debug(f"Final SHAP array shape: {shap_array.shape}, length: {len(shap_array)}")
                logger.debug(f"Feature names count: {len(feature_names_for_shap)}")
                
                # Create the series with proper alignment
                feature_contributions = pd.Series(shap_array, index=feature_names_for_shap[:len(shap_array)])
                
                # Sort by absolute contribution
                sorted_contributions = feature_contributions.abs().sort_values(ascending=False)
                top_features = sorted_contributions.head(10)  # Top 10 features
                
            except Exception as e:
                logger.error(f"Error creating feature contributions: {str(e)}")
                return {'error': f'Error creating feature contributions: {str(e)}'}
            
            explanation = {
                'base_value': base_value,
                'shap_values': feature_contributions.to_dict(),
                'top_features': {
                    feature: feature_contributions[feature] for feature in top_features.index
                }
            }
            
            return explanation

        except Exception as e:
            logger.error(f"Error generating explanation for {symbol}: {e}")
            return {'error': str(e)}

class MonitoringSystem:
    """Model monitoring and alerting"""
    
    def __init__(self):
        self.performance_history = []
        self.alerts = []
    
    def track_performance(self, predictions: Dict[str, any], actual_outcome: int = None):
        """Track model performance"""
        timestamp = datetime.now()
        
        performance_record = {
            'timestamp': timestamp,
            'prediction': predictions.get('prediction'),
            'probability': predictions.get('probability'),
            'symbol': predictions.get('symbol'),
            'actual_outcome': actual_outcome
        }
        
        self.performance_history.append(performance_record)
        
        # Check for alerts
        self._check_performance_alerts()
    
    def _check_performance_alerts(self):
        """Check for performance degradation"""
        if len(self.performance_history) < 100:
            return
        
        # Get recent predictions with outcomes
        recent = [p for p in self.performance_history[-100:] if p['actual_outcome'] is not None]
        
        if len(recent) < 50:
            return
        
        # Calculate accuracy
        correct = sum(1 for p in recent if p['prediction'] == p['actual_outcome'])
        accuracy = correct / len(recent)
        
        # Alert if accuracy drops below threshold
        if accuracy < 0.45:
            alert = {
                'timestamp': datetime.now(),
                'type': 'Performance Alert',
                'message': f'Model accuracy dropped to {accuracy:.2%}',
                'severity': 'HIGH'
            }
            self.alerts.append(alert)
            logger.warning(f"ALERT: {alert['message']}")
    
    def get_performance_summary(self) -> Dict[str, any]:
        """Get performance summary"""
        if not self.performance_history:
            return {'status': 'No data available'}
        
        recent = [p for p in self.performance_history[-100:] if p['actual_outcome'] is not None]
        
        if not recent:
            return {'status': 'No recent outcomes available'}
        
        correct = sum(1 for p in recent if p['prediction'] == p['actual_outcome'])
        accuracy = correct / len(recent)
        
        return {
            'recent_accuracy': accuracy,
            'total_predictions': len(self.performance_history),
            'recent_predictions_with_outcomes': len(recent),
            'active_alerts': len([a for a in self.alerts if 
                                (datetime.now() - a['timestamp']).days < 7])
        }

# Flask API
app = Flask(__name__)
predictor = ProductionPredictor()
monitor = MonitoringSystem()

@app.route('/')
def home():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/predict/<symbol>')
def api_predict(symbol):
    """API endpoint for predictions"""
    model_name = request.args.get('model', 'RandomForest')
    result = predictor.predict(symbol.upper(), model_name)
    # If error, return 400 and do not track
    if isinstance(result, dict) and result.get('error'):
        return jsonify(result), 400

    # Track prediction on success
    monitor.track_performance(result)
    return jsonify(result)

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': len(predictor.models),
        'performance': monitor.get_performance_summary()
    })

@app.route('/performance')
def performance_summary():
    """Get performance summary"""
    return jsonify(monitor.get_performance_summary())

@app.route('/models')
def list_models():
    """List available models and explainers to aid UI and debugging"""
    try:
        return jsonify({
            'models': sorted(list(predictor.models.keys())),
            'explainers': sorted(list(predictor.explainers.keys())),
            'feature_count': len(predictor.feature_names),
            'selectors': sorted(list(predictor.feature_selectors.keys())),
        })
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/explain/<symbol>')
def api_explain(symbol):
    """API endpoint for explanations"""
    model_name = request.args.get('model', 'RandomForest')
    result = predictor.explain(symbol.upper(), model_name)
    if isinstance(result, dict) and result.get('error'):
        return jsonify(result), 400
    return jsonify(result)

def load_config() -> Dict[str, any]:
    """Load configuration from config.json"""
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        logger.info("Configuration loaded from config.json")
        return config
    except FileNotFoundError:
        logger.error("config.json not found. Please create it.")
        sys.exit(1)
    except json.JSONDecodeError:
        logger.error("Error decoding config.json. Please check its format.")
        sys.exit(1)

def main():
    """Main execution function"""
    print("="*80)
    print("STOCK MARKET PREDICTION SYSTEM")
    print("="*80)
    print("IMPORTANT DISCLAIMER:")
    print("This system is for EDUCATIONAL and RESEARCH purposes only!")
    print("Stock market prediction is extremely risky. Never invest money you can't afford to lose.")
    print("Always use paper trading before risking real money!")
    print("="*80)
    
    # Load configuration
    config = load_config()
    SYMBOLS = config.get('symbols', ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN'])
    PREDICTION_DAYS = config.get('prediction_days', 5)
    API_HOST = config.get('api_host', '0.0.0.0')
    API_PORT = config.get('api_port', 5000)
    
    try:
        # Step 1: Data Collection
        print("\nStep 1: Collecting Data...")
        collector = DataCollector()
        
        stock_data = collector.get_multiple_stocks(SYMBOLS, period='2y')
        print(f"Collected data for {len(stock_data)} stocks")
        
        # Combine all stock data
        combined_data = pd.concat(stock_data.values(), ignore_index=False)
        combined_data = combined_data.sort_index()
        
        # Step 2: Feature Engineering
        print("\nStep 2: Feature Engineering...")
        engineer = FeatureEngineer()
        
        # Process each stock separately to maintain data integrity
        processed_stocks = []
        for symbol, data in stock_data.items():
            print(f"  Processing {symbol}...")
            features = engineer.create_technical_features(data)
            targets = engineer.create_targets(features, days_ahead=PREDICTION_DAYS)
            processed_stocks.append(targets)
        
        # Combine processed data
        full_dataset = pd.concat(processed_stocks, ignore_index=False)
        full_dataset = full_dataset.sort_index()
        
        print(f"Created {len(full_dataset.columns)} features")
        print(f"Dataset shape: {full_dataset.shape}")
        
        # Step 3: Data Preparation
        print("\nStep 3: Preparing Data...")
        trainer = ModelTrainer()
        
        # Prepare data
        trainer.full_dataset_for_explainer = full_dataset # Store for SHAP
        # Get all feature names before selection
        _, _, all_feature_names = trainer.prepare_data(full_dataset, target_col='Target_Binary', feature_selection=False)
        # Get data with feature selection for training
        X, y, selected_feature_names = trainer.prepare_data(full_dataset, target_col='Target_Binary', feature_selection=True)
        print(f"Prepared data: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Split data (time-aware)
        split_date = full_dataset.index[int(len(full_dataset) * 0.8)]
        train_data = full_dataset[full_dataset.index <= split_date]
        test_data = full_dataset[full_dataset.index > split_date]
        
        X_train, y_train, _ = trainer.prepare_data(train_data, target_col='Target_Binary')
        X_test, y_test, _ = trainer.prepare_data(test_data, target_col='Target_Binary')
        
        print(f"Train set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Step 4: Model Training
        print("\nStep 4: Training Models...")
        
        # Train traditional models
        traditional_results = trainer.train_traditional_models(X_train, y_train)
        
        # Train LSTM if available
        if TENSORFLOW_AVAILABLE and len(X_train) > 100:
            print("Training LSTM model...")
            lstm_results = trainer.train_lstm_model(X_train, y_train)
        
        # Create ensemble
        ensemble = trainer.create_ensemble()
        if ensemble is not None:
            print("Training ensemble model...")
            if 'traditional' in trainer.scalers:
                X_train_scaled = trainer.scalers['traditional'].transform(X_train)
                ensemble.fit(X_train_scaled, y_train)
                print("Ensemble model trained")
        
        # Step 5: Model Evaluation
        print("\nStep 5: Evaluating Models...")
        backtester = BacktestingFramework()
        
        # Get test period returns for backtesting
        test_returns = test_data['Returns'].dropna()
        
        evaluation_results = {}
        
        for model_name, model in trainer.models.items():
            if model_name == 'LSTM':
                continue  # Skip LSTM for now due to sequence requirement
            
            print(f"  Evaluating {model_name}...")
            
            # Scale test data
            if 'traditional' in trainer.scalers:
                X_test_scaled = trainer.scalers['traditional'].transform(X_test)
            else:
                X_test_scaled = X_test
            
            # Evaluate
            results = backtester.comprehensive_evaluation(
                model, X_test_scaled, y_test, test_returns[:len(y_test)]
            )
            
            evaluation_results[model_name] = results
            
            if results:
                print(f"    {model_name} Accuracy: {results.get('accuracy', 0):.4f}")
                if 'financial_metrics' in results:
                    fm = results['financial_metrics']
                    if fm:
                        print(f"    Sharpe Ratio: {fm.get('Sharpe_Ratio', 0):.4f}")
                        print(f"    Max Drawdown: {fm.get('Max_Drawdown', 0):.4f}")
        
        # Step 6: Results Summary
        print("\nRESULTS SUMMARY")
        print("="*50)
        
        best_model = None
        best_accuracy = 0
        
        for model_name, results in evaluation_results.items():
            if results and 'accuracy' in results:
                accuracy = results['accuracy']
                print(f"\n {model_name}:")
                print(f"  Accuracy: {accuracy:.4f}")
                
                if 'auc' in results:
                    print(f"  AUC: {results['auc']:.4f}")
                
                if 'financial_metrics' in results and results['financial_metrics']:
                    fm = results['financial_metrics']
                    print(f"  Sharpe Ratio: {fm.get('Sharpe_Ratio', 0):.4f}")
                    print(f"  Total Return: {fm.get('Total_Return', 0):.4f}")
                    print(f"  Max Drawdown: {fm.get('Max_Drawdown', 0):.4f}")
                    print(f"  Win Rate: {fm.get('Win_Rate', 0):.4f}")
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = model_name
        
        if best_model:
            print(f"\nBest Model: {best_model} (Accuracy: {best_accuracy:.4f})")
        
        # Step 7: Save and Reload Models for Production
        print("\nStep 7: Saving Models...")
        # Use the global predictor instance to save the newly trained models
        predictor.save_models(trainer, all_feature_names)
        
        # Step 8: Reload models into the global predictor to make them available to the API
        print("\nStep 8: Reloading Production System...")
        predictor.load_models()
        
        # Make sample predictions
        rf_model = trainer.models['RandomForest']
        if hasattr(rf_model, 'feature_importances_'):
            importances = rf_model.feature_importances_
            feature_importance = list(zip(selected_feature_names, importances))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            print("\nTop 10 Most Important Features:")
            for i, (feature, importance) in enumerate(feature_importance[:10]):
                print(f"  {i+1:2d}. {feature:<20} : {importance:.4f}")
        
        # Step 10: Visualization
        print("\nStep 10: Creating Visualizations...")
        create_visualizations(evaluation_results, selected_feature_names, trainer)
        
        # Final warnings and recommendations
        print("\n" + "="*80)
        print("IMPORTANT REMINDERS:")
        print("1. This is a RESEARCH and EDUCATIONAL tool only")
        print("2. Stock market prediction is inherently uncertain")
        print("3. Past performance does NOT guarantee future results")
        print("4. Always use paper trading before risking real money")
        print("5. Never invest more than you can afford to lose")
        print("6. Consider transaction costs and market impact")
        print("7. Regularly retrain models with new data")
        print("8. Monitor model performance continuously")
        print("="*80)
        
        # Optional: Start API server based on command-line argument
        parser = argparse.ArgumentParser(description='Stock Prediction System')
        parser.add_argument('--start-api', action='store_true', help='Start the API server after training.')
        # We need to parse known args, as other args may be passed by other systems.
        args, _ = parser.parse_known_args()

        if not args.start_api:
            # Show plots only if not starting the API server
            plt.show()

        if args.start_api:
            print("Starting API server on http://localhost:5000")
            print("Available endpoints:")
            print("  GET /predict/<symbol>  - Get prediction for stock")
            print("  GET /health          - Health check")
            print("  GET /performance     - Performance summary")
            print("\nPress Ctrl+C to stop the server")
            app.run(debug=False, host=API_HOST, port=API_PORT)
        
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        print(f"\nError: {e}")
        print("Check the log file for detailed error information")

def create_visualizations(results: Dict, feature_names: List[str], trainer: ModelTrainer):
    """Create visualization plots"""
    try:
        plt.style.use('seaborn-v0_8')  # Updated seaborn style
    except:
        plt.style.use('default')
    
    # 1. Model Performance Comparison
    if results:
        model_names = []
        accuracies = []
        sharpe_ratios = []
        
        for model_name, result in results.items():
            if result and 'accuracy' in result:
                model_names.append(model_name)
                accuracies.append(result['accuracy'])
                
                fm = result.get('financial_metrics', {})
                sharpe_ratios.append(fm.get('Sharpe_Ratio', 0))
        
        if model_names:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Accuracy comparison
            bars1 = ax1.bar(model_names, accuracies, alpha=0.7, color='steelblue')
            ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Accuracy')
            ax1.set_ylim(0, 1)
            ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Random Baseline')
            ax1.legend()
            plt.setp(ax1.get_xticklabels(), rotation=45)
            
            # Add value labels on bars
            for bar, acc in zip(bars1, accuracies):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{acc:.3f}', ha='center', va='bottom')
            
            # Sharpe ratio comparison
            bars2 = ax2.bar(model_names, sharpe_ratios, alpha=0.7, color='forestgreen')
            ax2.set_title('Sharpe Ratio Comparison', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Sharpe Ratio')
            ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)
            plt.setp(ax2.get_xticklabels(), rotation=45)
            
            # Add value labels on bars
            for bar, sr in zip(bars2, sharpe_ratios):
                ax2.text(bar.get_x() + bar.get_width()/2, 
                        bar.get_height() + (0.05 if sr > 0 else -0.1),
                        f'{sr:.3f}', ha='center', va='bottom' if sr > 0 else 'top')
            
            plt.tight_layout()
            plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
            print("Saved model comparison plot: model_comparison.png")
    
    # 2. Feature Importance (if Random Forest is available)
    if 'RandomForest' in trainer.models:
        rf_model = trainer.models['RandomForest']
        if hasattr(rf_model, 'feature_importances_') and len(feature_names) > 0:
            importances = rf_model.feature_importances_
            indices = np.argsort(importances)[::-1][:20]  # Top 20 features
            
            plt.figure(figsize=(12, 8))
            plt.title('Top 20 Feature Importances (Random Forest)', fontsize=14, fontweight='bold')
            plt.bar(range(len(indices)), importances[indices], alpha=0.7, color='darkorange')
            plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
            plt.ylabel('Importance')
            plt.tight_layout()
            plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
            print("Saved feature importance plot: feature_importance.png")
    

if __name__ == "__main__":
    main()