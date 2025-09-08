# setup.py (Installation script)
from setuptools import setup, find_packages

setup(
    name="multimodal-stock-predictor",
    version="0.1.0",
    description="A multimodal stock price predictor combining technical indicators and Twitter sentiment",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "yfinance==0.2.18",
        "pandas==1.5.3",
        "numpy==1.23.5",
        "scikit-learn==1.2.2",
        "lightgbm==3.3.5",
        "nltk==3.8.1",
        "textblob==0.17.1",
        "tweepy==4.14.0",
        "plotly==5.13.1",
        "streamlit==1.23.1",
        "ta==0.10.2",
        "matplotlib==3.7.1",
        "seaborn==0.12.2",
    ],
    python_requires=">=3.8",
)
