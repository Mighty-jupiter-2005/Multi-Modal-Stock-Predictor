# src/data_collection.py
import yfinance as yf
import pandas as pd
import numpy as np
import tweepy
import nltk
from textblob import TextBlob
import time
import os
from datetime import datetime, timedelta
import json

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class DataCollector:
    def __init__(self):
        self.twitter_auth = None
        
    def setup_twitter_api(self, consumer_key, consumer_secret, access_token, access_token_secret):
        """Set up Twitter API authentication"""
        try:
            auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
            auth.set_access_token(access_token, access_token_secret)
            self.twitter_auth = tweepy.API(auth, wait_on_rate_limit=True)
            print("Twitter API configured successfully")
            return True
        except Exception as e:
            print(f"Error setting up Twitter API: {e}")
            return False
    
    def get_stock_data(self, ticker, start_date, end_date, interval='1d'):
        """Fetch stock price data from Yahoo Finance"""
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date, interval=interval)
            df.reset_index(inplace=True)
            df['Ticker'] = ticker
            return df
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            return None
    
    def get_tweets(self, query, count=100, until=None):
        """Fetch tweets for a given query"""
        if not self.twitter_auth:
            print("Twitter API not configured")
            return []
        
        try:
            tweets = tweepy.Cursor(
                self.twitter_auth.search_tweets,
                q=query,
                lang="en",
                tweet_mode="extended",
                until=until
            ).items(count)
            
            return [tweet for tweet in tweets]
        except Exception as e:
            print(f"Error fetching tweets: {e}")
            return []
    
    def analyze_sentiment(self, text):
        """Analyze sentiment of text using TextBlob"""
        analysis = TextBlob(text)
        return analysis.sentiment.polarity
    
    def collect_daily_sentiment(self, ticker, date, tweet_count=200):
        """Collect and analyze tweets for a ticker on a specific date"""
        next_day = (datetime.strptime(date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
        query = f"${ticker} OR #{ticker} -filter:retweets"
        
        tweets = self.get_tweets(query, count=tweet_count, until=next_day)
        
        if not tweets:
            return None
        
        sentiments = []
        tweet_texts = []
        user_ids = set()
        
        for tweet in tweets:
            # Skip retweets
            if hasattr(tweet, 'retweeted_status'):
                continue
                
            text = tweet.full_text
            sentiment = self.analyze_sentiment(text)
            
            sentiments.append(sentiment)
            tweet_texts.append(text)
            user_ids.add(tweet.user.id)
        
        if not sentiments:
            return None
            
        # Calculate sentiment metrics
        sentiment_series = pd.Series(sentiments)
        result = {
            'date': date,
            'ticker': ticker,
            'sent_mean': sentiment_series.mean(),
            'sent_std': sentiment_series.std(),
            'pct_pos': (sentiment_series > 0).mean(),
            'pct_neg': (sentiment_series < 0).mean(),
            'tweet_count': len(sentiments),
            'unique_users': len(user_ids)
        }
        
        return result
    
    def create_sentiment_dataset(self, ticker, start_date, end_date):
        """Create sentiment dataset for a ticker over a date range"""
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        sentiment_data = []
        
        for date in dates:
            date_str = date.strftime('%Y-%m-%d')
            print(f"Collecting sentiment for {ticker} on {date_str}")
            
            sentiment = self.collect_daily_sentiment(ticker, date_str)
            if sentiment:
                sentiment_data.append(sentiment)
            
            # Be gentle with the Twitter API
            time.sleep(1)
        
        return pd.DataFrame(sentiment_data)
