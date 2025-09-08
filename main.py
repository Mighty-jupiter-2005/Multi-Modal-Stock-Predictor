# main.py (Optional command-line interface)
import argparse
import pandas as pd
from src.data_collection import DataCollector
from src.feature_engineering import FeatureEngineer
from src.model_training import ModelTrainer
from src.evaluation import ModelEvaluator

def main():
    parser = argparse.ArgumentParser(description='Multimodal Stock Price Predictor')
    parser.add_argument('--ticker', type=str, default='AAPL', help='Stock ticker symbol')
    parser.add_argument('--start_date', type=str, default='2020-01-01', help='Start date for data collection')
    parser.add_argument('--end_date', type=str, default='2023-01-01', help='End date for data collection')
    parser.add_argument('--train', action='store_true', help='Train models')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate models')
    parser.add_argument('--predict', action='store_true', help='Make prediction')
    
    args = parser.parse_args()
    
    # Initialize components
    data_collector = DataCollector()
    feature_engineer = FeatureEngineer()
    model_trainer = ModelTrainer()
    
    if args.train:
        print(f"Training models for {args.ticker}...")
        
        # Collect data
        stock_data = data_collector.get_stock_data(args.ticker, args.start_date, args.end_date)
        if stock_data is None:
            print("Failed to fetch stock data")
            return
        
        # Engineer features
        featured_data = feature_engineer.calculate_technical_indicators(stock_data)
        featured_data = feature_engineer.create_target_variable(featured_data)
        
        # Prepare for training
        X, y, feature_names = model_trainer.prepare_features_target(featured_data)
        
        # Train models
        results, feature_importances = model_trainer.train_models(X, y, feature_names)
        
        # Save models
        model_trainer.save_model('logistic', 'models/logistic_model.pkl')
        model_trainer.save_model('random_forest', 'models/random_forest_model.pkl')
        model_trainer.save_model('lightgbm', 'models/lightgbm_model.pkl')
        
        print("Training completed and models saved")
    
    if args.evaluate:
        print("Evaluating models...")
        
        # Load models
        try:
            model_trainer.load_model('logistic', 'models/logistic_model.pkl')
            model_trainer.load_model('random_forest', 'models/random_forest_model.pkl')
            model_trainer.load_model('lightgbm', 'models/lightgbm_model.pkl')
        except:
            print("Failed to load models. Train models first with --train")
            return
        
        # Collect data for evaluation
        stock_data = data_collector.get_stock_data(args.ticker, args.start_date, args.end_date)
        if stock_data is None:
            print("Failed to fetch stock data")
            return
        
        # Engineer features
        featured_data = feature_engineer.calculate_technical_indicators(stock_data)
        featured_data = feature_engineer.create_target_variable(featured_data)
        
        # Prepare for evaluation
        X, y, feature_names = model_trainer.prepare_features_target(featured_data)
        
        # Create evaluator
        evaluator = ModelEvaluator(model_trainer)
        
        # Generate reports
        for model_name in model_trainer.models.keys():
            print(f"\nEvaluation for {model_name}:")
            results = model_trainer.results[model_name]['overall_metrics']
            for metric, value in results.items():
                print(f"{metric}: {value:.4f}")
    
    if args.predict:
        print("Making prediction...")
        
        # Load models
        try:
            model_trainer.load_model('lightgbm', 'models/lightgbm_model.pkl')
        except:
            print("Failed to load model. Train models first with --train")
            return
        
        # Get recent data
        recent_data = data_collector.get_stock_data(args.ticker, '2023-01-01', '2023-12-31')
        if recent_data is None:
            print("Failed to fetch recent data")
            return
        
        # Calculate features
        featured_data = feature_engineer.calculate_technical_indicators(recent_data)
        latest_data = featured_data.iloc[-1:].copy()
        
        # Prepare features
        exclude_cols = ['Date', 'date', 'Ticker', 'target']
        feature_cols = [col for col in latest_data.columns if col not in exclude_cols]
        X_latest = latest_data[feature_cols]
        
        # Make prediction
        model = model_trainer.models['lightgbm']
        prediction = model.predict(X_latest)
        probability = model.predict_proba(X_latest)
        
        print(f"Prediction: {'UP' if prediction[0] == 1 else 'DOWN'}")
        print(f"Confidence: {probability[0][1 if prediction[0] == 1 else 0]:.2%}")
        print(f"Current Price: ${latest_data['Close'].iloc[0]:.2f}")

if __name__ == "__main__":
    main()
