# src/evaluation.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    roc_curve, auc, precision_recall_curve
)

class ModelEvaluator:
    def __init__(self, model_trainer):
        self.model_trainer = model_trainer
    
    def generate_report(self, model_name):
        """Generate a comprehensive evaluation report for a model"""
        if model_name not in self.model_trainer.results:
            raise ValueError(f"No results found for model: {model_name}")
        
        results = self.model_trainer.results[model_name]
        y_true = results['predictions']['y_true']
        y_pred = results['predictions']['y_pred']
        y_prob = results['predictions'].get('y_prob', None)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')
        
        # 2. ROC Curve (if probabilities available)
        if y_prob is not None:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
            
            axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            axes[0, 1].set_xlim([0.0, 1.0])
            axes[0, 1].set_ylim([0.0, 1.05])
            axes[0, 1].set_xlabel('False Positive Rate')
            axes[0, 1].set_ylabel('True Positive Rate')
            axes[0, 1].set_title('Receiver Operating Characteristic')
            axes[0, 1].legend(loc="lower right")
        else:
            axes[0, 1].text(0.5, 0.5, 'No probability scores available', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('ROC Curve (Not Available)')
        
        # 3. Precision-Recall Curve (if probabilities available)
        if y_prob is not None:
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            pr_auc = auc(recall, precision)
            
            axes[1, 0].plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
            axes[1, 0].set_xlim([0.0, 1.0])
            axes[1, 0].set_ylim([0.0, 1.05])
            axes[1, 0].set_xlabel('Recall')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].set_title('Precision-Recall Curve')
            axes[1, 0].legend(loc="lower left")
        else:
            axes[1, 0].text(0.5, 0.5, 'No probability scores available', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Precision-Recall Curve (Not Available)')
        
        # 4. Backtest Results
        backtest_results = self.model_trainer.backtest_strategy(None, model_name)
        cumulative_returns = backtest_results['cumulative_returns']
        
        axes[1, 1].plot(cumulative_returns)
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylabel('Cumulative Return')
        axes[1, 1].set_title('Strategy Cumulative Returns')
        axes[1, 1].grid(True)
        
        # Add metrics text
        metrics_text = f"""
        Accuracy: {results['overall_metrics']['accuracy']:.4f}
        F1 Score: {results['overall_metrics']['f1']:.4f}
        MCC: {results['overall_metrics']['mcc']:.4f}
        ROC AUC: {results['overall_metrics']['roc_auc']:.4f if results['overall_metrics']['roc_auc'] else 'N/A'}
        
        Strategy Metrics:
        Total Return: {backtest_results['total_return']:.2%}
        Sharpe Ratio: {backtest_results['sharpe_ratio']:.2f}
        Max Drawdown: {backtest_results['max_drawdown']:.2%}
        """
        
        # Add text box
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        axes[1, 1].text(0.05, 0.95, metrics_text, transform=axes[1, 1].transAxes, fontsize=10,
                       verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        return fig
    
    def compare_models(self):
        """Compare performance of all trained models"""
        model_names = list(self.model_trainer.results.keys())
        metrics = ['accuracy', 'f1', 'mcc', 'roc_auc']
        
        comparison_data = []
        for model_name in model_names:
            model_metrics = self.model_trainer.results[model_name]['overall_metrics']
            comparison_data.append({
                'Model': model_name,
                **model_metrics
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Plot comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        for i, metric in enumerate(metrics):
            ax = axes[i//2, i%2]
            comparison_df.plot.bar(x='Model', y=metric, ax=ax, legend=False)
            ax.set_title(f'{metric.upper()} Comparison')
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return fig, comparison_df
    
    def ablation_study(self, X, y, feature_names, base_model, sentiment_features):
        """Perform ablation study to measure sentiment impact"""
        # Train model with all features
        base_model.fit(X, y)
        all_features_score = base_model.score(X, y)
        
        # Train model without sentiment features
        non_sentiment_features = [f for f in feature_names if f not in sentiment_features]
        X_no_sentiment = X[non_sentiment_features]
        
        base_model.fit(X_no_sentiment, y)
        no_sentiment_score = base_model.score(X_no_sentiment, y)
        
        # Calculate improvement
        improvement = all_features_score - no_sentiment_score
        improvement_pct = (improvement / no_sentiment_score) * 100
        
        return {
            'all_features_score': all_features_score,
            'no_sentiment_score': no_sentiment_score,
            'improvement': improvement,
            'improvement_pct': improvement_pct
        }
