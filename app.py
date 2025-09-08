# app.py (Streamlit Dashboard)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import joblib
import os
import sys

# Add src to path
sys.path.append('src')

from src.data_collection import DataCollector
from src.feature_engineering import FeatureEngineer
from src.model_training import ModelTrainer
from src.evaluation import ModelEvaluator

# Set page config
st.set_page_config(
    page_title="Multimodal Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .warning {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

class StockPredictorApp:
    def __init__(self):
        self.data_collector = DataCollector()
        self.feature_engineer = FeatureEngineer()
        self.model_trainer = ModelTrainer()
        self.evaluator = None
        
        # Load pre-trained models if available
        self.models_loaded = False
        self.load_models()
    
    def load_models(self):
        """Load pre-trained models"""
        model_paths = {
            'logistic': 'models/logistic_model.pkl',
            'random_forest': 'models/random_forest_model.pkl',
            'lightgbm': 'models/lightgbm_model.pkl'
        }
        
        for model_name, path in model_paths.items():
            if os.path.exists(path):
                try:
                    self.model_trainer.load_model(model_name, path)
                    self.models_loaded = True
                except:
                    st.warning(f"Could not load model: {model_name}")
        
        if self.models_loaded:
            self.evaluator = ModelEvaluator(self.model_trainer)
    
    def sidebar(self):
        """Create sidebar controls"""
        st.sidebar.title("Configuration")
        
        # Ticker selection
        ticker = st.sidebar.text_input("Stock Ticker", value="AAPL").upper()
        
        # Date range selection
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*2)  # 2 years by default
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=start_date)
        with col2:
            end_date = st.date_input("End Date", value=end_date)
        
        # Model selection
        model_options = ['logistic', 'random_forest', 'lightgbm']
        selected_model = st.sidebar.selectbox("Select Model", model_options, index=2)
        
        # Twitter API configuration (collapsible)
        with st.sidebar.expander("Twitter API Configuration"):
            consumer_key = st.text_input("Consumer Key", type="password")
            consumer_secret = st.text_input("Consumer Secret", type="password")
            access_token = st.text_input("Access Token", type="password")
            access_token_secret = st.text_input("Access Token Secret", type="password")
            
            if st.button("Configure Twitter API"):
                if all([consumer_key, consumer_secret, access_token, access_token_secret]):
                    success = self.data_collector.setup_twitter_api(
                        consumer_key, consumer_secret, access_token, access_token_secret
                    )
                    if success:
                        st.success("Twitter API configured successfully!")
                    else:
                        st.error("Failed to configure Twitter API")
                else:
                    st.error("Please fill all API fields")
        
        return ticker, start_date, end_date, selected_model
    
    def data_collection_tab(self, ticker, start_date, end_date):
        """Data collection and exploration tab"""
        st.header("Data Collection & Exploration")
        
        # Fetch stock data
        with st.spinner("Fetching stock data..."):
            stock_data = self.data_collector.get_stock_data(
                ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
            )
        
        if stock_data is None:
            st.error("Failed to fetch stock data. Please check the ticker symbol and try again.")
            return
        
        # Display raw data
        st.subheader("Raw Stock Data")
        st.dataframe(stock_data.head())
        
        # Basic statistics
        st.subheader("Basic Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Days", len(stock_data))
            st.metric("Average Volume", f"{stock_data['Volume'].mean():,.0f}")
        
        with col2:
            price_change = (stock_data['Close'].iloc[-1] - stock_data['Close'].iloc[0]) / stock_data['Close'].iloc[0] * 100
            st.metric("Price Change", f"{price_change:.2f}%")
            st.metric("High Price", f"${stock_data['High'].max():.2f}")
        
        with col3:
            st.metric("Low Price", f"${stock_data['Low'].min():.2f}")
            volatility = stock_data['Close'].pct_change().std() * np.sqrt(252) * 100  # Annualized
            st.metric("Annualized Volatility", f"{volatility:.2f}%")
        
        # Price chart
        st.subheader("Price Chart")
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=stock_data['Date'],
            open=stock_data['Open'],
            high=stock_data['High'],
            low=stock_data['Low'],
            close=stock_data['Close'],
            name='OHLC'
        ))
        fig.update_layout(
            title=f"{ticker} Stock Price",
            yaxis_title="Price ($)",
            xaxis_title="Date",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Sentiment data collection (if Twitter API is configured)
        if hasattr(self.data_collector, 'twitter_auth') and self.data_collector.twitter_auth:
            if st.button("Collect Sentiment Data"):
                with st.spinner("Collecting tweet sentiment data..."):
                    sentiment_data = self.data_collector.create_sentiment_dataset(
                        ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
                    )
                
                if sentiment_data is not None and len(sentiment_data) > 0:
                    st.subheader("Sentiment Data")
                    st.dataframe(sentiment_data)
                    
                    # Sentiment timeline chart
                    fig = px.line(sentiment_data, x='date', y='sent_mean', 
                                 title='Daily Average Sentiment')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Sentiment distribution
                    fig = px.histogram(sentiment_data, x='sent_mean', 
                                      title='Sentiment Distribution')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No sentiment data collected. This could be due to rate limits or no tweets found.")
        else:
            st.info("Configure Twitter API in the sidebar to collect sentiment data.")
        
        return stock_data
    
    def feature_engineering_tab(self, stock_data, ticker):
        """Feature engineering tab"""
        st.header("Feature Engineering")
        
        with st.spinner("Calculating technical indicators..."):
            featured_data = self.feature_engineer.calculate_technical_indicators(stock_data)
            featured_data = self.feature_engineer.create_target_variable(featured_data)
        
        st.subheader("Engineered Features")
        st.dataframe(featured_data.select_dtypes(include=[np.number]).iloc[:, -10:].head())
        
        # Feature distributions
        st.subheader("Feature Distributions")
        numeric_cols = featured_data.select_dtypes(include=[np.number]).columns.tolist()
        selected_feature = st.selectbox("Select feature to visualize", numeric_cols[-10:])
        
        fig = px.histogram(featured_data, x=selected_feature, 
                          title=f'Distribution of {selected_feature}')
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation heatmap
        st.subheader("Feature Correlations")
        corr_matrix = featured_data[numeric_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu_r',
            zmin=-1,
            zmax=1
        ))
        fig.update_layout(
            title="Feature Correlation Matrix",
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)
        
        return featured_data
    
    def model_training_tab(self, featured_data):
        """Model training and evaluation tab"""
        st.header("Model Training & Evaluation")
        
        # Prepare features and target
        X, y, feature_names = self.model_trainer.prepare_features_target(featured_data)
        
        st.subheader("Dataset Info")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Number of Samples", X.shape[0])
        with col2:
            st.metric("Number of Features", X.shape[1])
        with col3:
            class_balance = np.mean(y) * 100
            st.metric("Class Balance (Up %)", f"{class_balance:.2f}%")
        
        # Train models
        if st.button("Train Models"):
            with st.spinner("Training models..."):
                results, feature_importances = self.model_trainer.train_models(X, y, feature_names)
            
            self.evaluator = ModelEvaluator(self.model_trainer)
            
            # Display results
            st.subheader("Model Performance")
            for model_name, result in results.items():
                with st.expander(f"{model_name} Results"):
                    st.write("Overall Metrics:", result['overall_metrics'])
            
            # Model comparison
            st.subheader("Model Comparison")
            fig, comparison_df = self.evaluator.compare_models()
            st.pyplot(fig)
            st.dataframe(comparison_df)
            
            # Feature importances
            st.subheader("Feature Importances")
            selected_model = st.selectbox("Select model to view feature importances", list(results.keys()))
            
            if selected_model in feature_importances:
                importances = feature_importances[selected_model]
                top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:15]
                
                fig, ax = plt.subplots(figsize=(10, 8))
                features, importance_vals = zip(*top_features)
                ax.barh(range(len(features)), importance_vals[::-1])
                ax.set_yticks(range(len(features)))
                ax.set_yticklabels(features[::-1])
                ax.set_xlabel('Importance')
                ax.set_title(f'Top Features - {selected_model}')
                st.pyplot(fig)
            
            # Detailed evaluation for selected model
            st.subheader("Detailed Evaluation")
            eval_model = st.selectbox("Select model for detailed evaluation", list(results.keys()))
            
            fig = self.evaluator.generate_report(eval_model)
            st.pyplot(fig)
            
            # Ablation study
            st.subheader("Ablation Study")
            st.info("This study measures the impact of sentiment features on model performance.")
            
            # Identify sentiment features (simplified approach)
            sentiment_features = [f for f in feature_names if 'sent' in f.lower() or 'tweet' in f.lower()]
            
            if sentiment_features:
                base_model = LogisticRegression(max_iter=1000, class_weight='balanced')
                ablation_results = self.evaluator.ablation_study(X, y, feature_names, base_model, sentiment_features)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("All Features Score", f"{ablation_results['all_features_score']:.4f}")
                with col2:
                    st.metric("No Sentiment Score", f"{ablation_results['no_sentiment_score']:.4f}")
                with col3:
                    st.metric("Improvement", f"{ablation_results['improvement_pct']:.2f}%")
            else:
                st.warning("No sentiment features found for ablation study.")
    
    def prediction_tab(self, ticker):
        """Prediction and forecasting tab"""
        st.header("Prediction & Forecasting")
        
        if not self.models_loaded:
            st.warning("Please train or load models first.")
            return
        
        # Get recent data for prediction
        end_date = datetime.now()
        start_date = end_date - timedelta(days=100)  # Get enough data for feature calculation
        
        with st.spinner("Fetching recent data..."):
            recent_data = self.data_collector.get_stock_data(
                ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
            )
        
        if recent_data is None:
            st.error("Failed to fetch recent data.")
            return
        
        # Calculate features
        with st.spinner("Calculating features..."):
            featured_data = self.feature_engineer.calculate_technical_indicators(recent_data)
        
        # Get the most recent data point for prediction
        latest_data = featured_data.iloc[-1:].copy()
        
        # Prepare features
        exclude_cols = ['Date', 'date', 'Ticker', 'target']
        feature_cols = [col for col in latest_data.columns if col not in exclude_cols]
        X_latest = latest_data[feature_cols]
        
        st.subheader("Latest Data for Prediction")
        st.dataframe(X_latest)
        
        # Model selection for prediction
        model_name = st.selectbox("Select model for prediction", list(self.model_trainer.models.keys()))
        
        if st.button("Make Prediction"):
            model = self.model_trainer.models[model_name]
            
            # Make prediction
            prediction = model.predict(X_latest)
            probability = model.predict_proba(X_latest) if hasattr(model, 'predict_proba') else None
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Prediction", "UP" if prediction[0] == 1 else "DOWN")
            
            if probability is not None:
                with col2:
                    confidence = probability[0][1] if prediction[0] == 1 else probability[0][0]
                    st.metric("Confidence", f"{confidence:.2%}")
            
            with col3:
                current_price = latest_data['Close'].iloc[0]
                st.metric("Current Price", f"${current_price:.2f}")
            
            # Explanation (for tree-based models)
            if hasattr(model, 'feature_importances_') and hasattr(model, 'predict_proba'):
                st.subheader("Prediction Explanation")
                
                # Get feature importances
                importances = model.feature_importances_
                feature_importance_dict = dict(zip(feature_cols, importances))
                
                # Get SHAP values if possible (simplified approach)
                top_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)[:10]
                
                fig, ax = plt.subplots(figsize=(10, 6))
                features, importance_vals = zip(*top_features)
                ax.barh(range(len(features)), importance_vals[::-1])
                ax.set_yticks(range(len(features)))
                ax.set_yticklabels(features[::-1])
                ax.set_xlabel('Importance')
                ax.set_title('Top Features Influencing Prediction')
                st.pyplot(fig)
    
    def run(self):
        """Run the Streamlit app"""
        st.title("📈 Multimodal Stock Price Predictor")
        
        # Disclaimer
        st.markdown("""
        <div class="warning">
        <strong>Disclaimer:</strong> This application is for educational purposes only. 
        The predictions should not be considered financial advice. Always do your own research 
        and consult with a qualified financial advisor before making investment decisions.
        </div>
        """, unsafe_allow_html=True)
        
        # Sidebar
        ticker, start_date, end_date, selected_model = self.sidebar()
        
        # Main tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "Data Collection", 
            "Feature Engineering", 
            "Model Training", 
            "Prediction"
        ])
        
        with tab1:
            stock_data = self.data_collection_tab(ticker, start_date, end_date)
        
        with tab2:
            if 'stock_data' in locals():
                featured_data = self.feature_engineering_tab(stock_data, ticker)
        
        with tab3:
            if 'featured_data' in locals():
                self.model_training_tab(featured_data)
        
        with tab4:
            self.prediction_tab(ticker)

# Run the app
if __name__ == "__main__":
    app = StockPredictorApp()
    app.run()
