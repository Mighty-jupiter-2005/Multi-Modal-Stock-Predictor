# src/model_training.py
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, 
    mean_absolute_error, mean_squared_error,
    confusion_matrix, classification_report,
    matthews_corrcoef
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.results = {}
    
    def prepare_features_target(self, df, target_col='target', exclude_cols=None):
        """Prepare features and target variable from dataframe"""
        if exclude_cols is None:
            exclude_cols = ['Date', 'date', 'Ticker', 'target']
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        X = df[feature_cols]
        y = df[target_col]
        
        return X, y, feature_cols
    
    def train_models(self, X, y, feature_names, models_to_train=None):
        """Train multiple models using time series cross-validation"""
        if models_to_train is None:
            models_to_train = {
                'logistic': LogisticRegression(max_iter=1000, class_weight='balanced'),
                'random_forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
                'lightgbm': lgb.LGBMClassifier(n_estimators=100, random_state=42, class_weight='balanced')
            }
        
        # Initialize time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        results = {}
        feature_importances = {}
        
        for model_name, model in models_to_train.items():
            print(f"Training {model_name}...")
            
            # Store results for each fold
            fold_results = {
                'accuracy': [], 'f1': [], 'roc_auc': [], 'mcc': [],
                'y_true': [], 'y_pred': [], 'y_prob': []
            }
            
            # For tree models, collect feature importances
            if hasattr(model, 'feature_importances_'):
                feature_importances[model_name] = []
            
            # Perform time series cross-validation
            for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')
                mcc = matthews_corrcoef(y_test, y_pred)
                
                if y_prob is not None:
                    roc_auc = roc_auc_score(y_test, y_prob)
                else:
                    roc_auc = None
                
                # Store results
                fold_results['accuracy'].append(accuracy)
                fold_results['f1'].append(f1)
                fold_results['mcc'].append(mcc)
                if roc_auc is not None:
                    fold_results['roc_auc'].append(roc_auc)
                
                fold_results['y_true'].extend(y_test.tolist())
                fold_results['y_pred'].extend(y_pred.tolist())
                if y_prob is not None:
                    fold_results['y_prob'].extend(y_prob.tolist())
                
                # Store feature importances for tree models
                if hasattr(model, 'feature_importances_'):
                    feature_importances[model_name].append(model.feature_importances_)
            
            # Calculate average metrics across folds
            avg_results = {
                'accuracy': np.mean(fold_results['accuracy']),
                'f1': np.mean(fold_results['f1']),
                'mcc': np.mean(fold_results['mcc']),
                'roc_auc': np.mean(fold_results.get('roc_auc', [0])),
            }
            
            # Calculate overall metrics on all predictions
            all_y_true = fold_results['y_true']
            all_y_pred = fold_results['y_pred']
            
            overall_accuracy = accuracy_score(all_y_true, all_y_pred)
            overall_f1 = f1_score(all_y_true, all_y_pred, average='weighted')
            overall_mcc = matthews_corrcoef(all_y_true, all_y_pred)
            
            if fold_results.get('y_prob'):
                overall_roc_auc = roc_auc_score(all_y_true, fold_results['y_prob'])
            else:
                overall_roc_auc = None
            
            results[model_name] = {
                'fold_metrics': avg_results,
                'overall_metrics': {
                    'accuracy': overall_accuracy,
                    'f1': overall_f1,
                    'mcc': overall_mcc,
                    'roc_auc': overall_roc_auc if overall_roc_auc is not None else 0
                },
                'predictions': fold_results
            }
            
            # Store the trained model
            self.models[model_name] = model
            
            print(f"{model_name} - Accuracy: {overall_accuracy:.4f}, F1: {overall_f1:.4f}, MCC: {overall_mcc:.4f}")
        
        # Calculate average feature importances
        for model_name, importances_list in feature_importances.items():
            avg_importances = np.mean(importances_list, axis=0)
            feature_importances[model_name] = dict(zip(feature_names, avg_importances))
        
        self.results = results
        self.feature_importances = feature_importances
        
        return results, feature_importances
    
    def backtest_strategy(self, df, model_name, prob_threshold=0.55):
        """Backtest a trading strategy based on model predictions"""
        if model_name not in self.results:
            raise ValueError(f"No results found for model: {model_name}")
        
        results = self.results[model_name]
        y_true = results['predictions']['y_true']
        y_pred = results['predictions']['y_pred']
        y_prob = results['predictions'].get('y_prob', [])
        
        # If we have probabilities, use them for strategy
        if y_prob:
            # Create strategy: long if probability > threshold, flat otherwise
            strategy_returns = []
            for i, prob in enumerate(y_prob):
                if prob > prob_threshold:
                    # We predict up, so we would have gone long
                    # For simplicity, assume we get the actual return
                    strategy_returns.append(y_true[i] if y_true[i] == 1 else -1)  # -1 if we're wrong
                else:
                    # We stay flat, no return
                    strategy_returns.append(0)
        else:
            # Use binary predictions
            strategy_returns = []
            for i, pred in enumerate(y_pred):
                if pred == 1:
                    strategy_returns.append(y_true[i] if y_true[i] == 1 else -1)
                else:
                    strategy_returns.append(0)
        
        # Calculate strategy metrics
        strategy_returns = np.array(strategy_returns)
        cumulative_returns = np.cumprod(1 + strategy_returns)
        
        # Calculate performance metrics
        total_return = cumulative_returns[-1] - 1
        sharpe_ratio = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)  # Annualized
        
        # Calculate max drawdown
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (peak - cumulative_returns) / peak
        max_drawdown = np.max(drawdown)
        
        return {
            'returns': strategy_returns,
            'cumulative_returns': cumulative_returns,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }
    
    def save_model(self, model_name, path):
        """Save a trained model to disk"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        joblib.dump(self.models[model_name], path)
        print(f"Model saved to {path}")
    
    def load_model(self, model_name, path):
        """Load a trained model from disk"""
        self.models[model_name] = joblib.load(path)
        print(f"Model loaded from {path}")
    
    def plot_feature_importances(self, model_name, top_n=15):
        """Plot feature importances for a model"""
        if model_name not in self.feature_importances:
            raise ValueError(f"No feature importances found for model: {model_name}")
        
        importances = self.feature_importances[model_name]
        sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        
        # Take top N features
        top_features = sorted_importances[:top_n]
        features, importance_vals = zip(*top_features)
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(features)), importance_vals[::-1])
        plt.yticks(range(len(features)), features[::-1])
        plt.xlabel('Importance')
        plt.title(f'Feature Importances - {model_name}')
        plt.tight_layout()
        
        return pl
