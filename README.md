# 📈 TIME SERIES ANALYSIS WITH CRYPTOCURRENCY

Welcome to the **TIME SERIES ANALYSIS WITH CRYPTOCURRENCY** repository!

This project is built using Python 3.9, Streamlit, and advanced Time Series Forecasting Models such as **ARIMA**, **SARIMA**, **Prophet**, and **LSTM**. [cite_start]The platform provides a complete dashboard for analyzing, visualizing, and forecasting cryptocurrency trends[cite: 5, 20, 28].

**🚀 Live Demo:** [https://time-series-analysis-with-cryptocurrency-e15v.onrender.com](https://time-series-analysis-with-cryptocurrency-e15v.onrender.com)

---

## 📚 Table of Contents
1. [Overview](#-overview)
2. [Features & Modules](#-features-and-modules)
3. [Tech Stack](#-technology-stack)
4. [Prerequisites](#-prerequisites)
5. [Installation](#-installation-for-macoslinux)
6. [Database Setup](#-database-setup-authentication)
7. [Running the App](#-running-the-application)
8. [Usage Guide](#-accessing-the-dashboard)
9. [Screenshots](#-screenshots)

---

## 📝 Overview

The **TIME SERIES ANALYSIS WITH CRYPTOCURRENCY** project is designed to help traders, investors, analysts, and researchers gain insights into crypto market behavior. [cite_start]It leverages data analytics, statistical modeling, and machine learning to predict future price movements based on historical data[cite: 9, 10, 14].

**Core Capabilities:**
* [cite_start]**Data Collection:** Extracts real-time and historical cryptocurrency data using the Yahoo Finance API (`yfinance`)[cite: 16].
* [cite_start]**Forecasting Engine:** Implements advanced models like **ARIMA, SARIMA, Prophet, and LSTM**[cite: 11, 20].
* [cite_start]**Risk Assessment:** Features tools for Volatility Analysis (Bollinger Bands, Rolling Volatility)[cite: 23].
* [cite_start]**Market Sentiment:** Includes a simulation of **NLP-based sentiment analysis** from crypto-related news[cite: 24].
* [cite_start]**Strategy Backtesting:** Evaluates algorithmic strategy performance against a simple Buy & Hold baseline[cite: 27].

---

## ⭐ Features and Modules

The application is logically structured into 10 key modules, accessible via the sidebar navigation:

1.  **Overview/KPIs:** Real-time metrics (Price, Volume, Daily Change) and price history area charts.
2.  [cite_start]**Price Explorer:** Interactive **Candlestick** charts for deep market examination (OHLCV data)[cite: 22].
3.  [cite_start]**Advanced Forecasting:** Run predictions using **Prophet, ARIMA, SARIMA, and LSTM** models[cite: 20].
4.  [cite_start]**Sentiment Analysis:** Simulated news feed with computed AI Sentiment Scores and Market Mood charts[cite: 24].
5.  **Risk & Volatility:** **Bollinger Bands** (Volatility Envelope) and Return Distribution Histograms.
6.  [cite_start]**Technical Indicators:** Visualizations for **RSI** (Relative Strength Index) and **MACD**[cite: 28].
7.  **Correlations:** **Heatmap** (Correlation Matrix) of OHLCV features.
8.  **Feature Importance:** Lagged correlation analysis to determine predictive power.
9.  **Strategy Backtest:** Equity Curve comparing Algo Strategy vs. Buy & Hold.
10. **Data Export:** Download raw ledger data as CSV.

---

## 🛠️ Technology Stack

The dashboard is built entirely in Python using a powerful set of libraries:

| Category | Technology / Library | Purpose |
| :--- | :--- | :--- |
| **Framework & GUI** | `streamlit` | [cite_start]Main application and interactive dashboard creation[cite: 22, 28]. |
| **Data Acquisition** | `yfinance` | [cite_start]Real-time and historical crypto price data extraction[cite: 16, 28]. |
| **Deep Learning** | `tensorflow` / `keras` (LSTM) | [cite_start]Neural network implementation for time series prediction[cite: 20, 28]. |
| **Statistical Modeling** | `prophet`, `statsmodels` | [cite_start]ARIMA, SARIMA, and Facebook Prophet forecasting[cite: 20, 28]. |
| **Data Processing** | `pandas`, `numpy` | [cite_start]Data cleaning, preprocessing, and numerical calculations[cite: 18, 28]. |
| **Visualization** | `plotly` / `plotly.express` | [cite_start]Highly interactive charts and dashboards[cite: 22, 28]. |
| **Utility** | `textblob`, `streamlit-lottie` | [cite_start]Sentiment analysis and animations[cite: 28]. |

---

## ⚙️ Installation (for macOS/Linux)

### Prerequisites
* Python 3.8+
* A stable internet connection (required for data fetching via `yfinance`).

### 1. Clone and Environment Setup

We recommend using a virtual environment to manage dependencies.

```bash
# 1. Clone the Repository
git clone <YOUR-REPOSITORY-URL>
cd crypto-intelligence-dashboard

# 2. Create a Virtual Environment
python3 -m venv venv

# 3. Activate the environment
source venv/bin/activate
