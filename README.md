# 📈 TIME SERIES ANALYSIS WITH CRYPTOCURRENCY

Welcome to the **TIME SERIES ANALYSIS WITH CRYPTOCURRENCY** repository!

This project is built using Python 3.9, Streamlit, and advanced Time Series Forecasting Models such as **ARIMA**, **SARIMA**, **Prophet**, and **LSTM**. The platform provides a complete dashboard for analyzing, visualizing, and forecasting cryptocurrency trends.

**🚀 Live Demo:** [https://time-series-analysis-with-cryptocurrency-e15v.onrender.com](https://time-series-analysis-with-cryptocurrency-e15v.onrender.com)

---

## 📚 Table of Contents
1. [Overview](#-overview)
2. [Features & Modules](#-features-and-modules)
3. [Tech Stack](#-technology-stack)
4. [Prerequisites](#-prerequisites)
5. [Installation](#-installation-via-conda-macoslinux)
6. [Database Setup](#-database-setup-authentication)
7. [Running the App](#-running-the-application)
8. [Usage Guide](#-accessing-the-dashboard)
9. [Screenshots](#-screenshots)

---

## 📝 Overview

The **TIME SERIES ANALYSIS WITH CRYPTOCURRENCY** project is designed to help traders, investors, analysts, and researchers gain insights into crypto market behavior. It leverages data analytics, statistical modeling, and machine learning to predict future price movements based on historical data.

**Core Capabilities:**
* **Data Collection:** Extracts real-time and historical cryptocurrency data using the Yahoo Finance API (`yfinance`).
* **Forecasting Engine:** Implements advanced models like **ARIMA, SARIMA, Prophet, and LSTM**.
* **Risk Assessment:** Features tools for Volatility Analysis (Bollinger Bands, Rolling Volatility).
* **Market Sentiment:** Includes a simulation of **NLP-based sentiment analysis** from crypto-related news.
* **Strategy Backtesting:** Evaluates algorithmic strategy performance against a simple Buy & Hold baseline.

---

## ⭐ Features and Modules

The application is logically structured into 10 key modules, accessible via the sidebar navigation:

1.  **Overview/KPIs:** Real-time metrics (Price, Volume, Daily Change) and price history area charts.
2.  **Price Explorer:** Interactive **Candlestick** charts for deep market examination (OHLCV data).
3.  **Advanced Forecasting:** Run predictions using **Prophet, ARIMA, SARIMA, and LSTM** models.
4.  **Sentiment Analysis:** Simulated news feed with computed AI Sentiment Scores and Market Mood charts.
5.  **Risk & Volatility:** **Bollinger Bands** (Volatility Envelope) and Return Distribution Histograms.
6.  **Technical Indicators:** Visualizations for **RSI** (Relative Strength Index) and **MACD**.
7.  **Correlations:** **Heatmap** (Correlation Matrix) of OHLCV features.
8.  **Feature Importance:** Lagged correlation analysis to determine predictive power.
9.  **Strategy Backtest:** Equity Curve comparing Algo Strategy vs. Buy & Hold.
10. **Data Export:** Download raw ledger data as CSV.

---

## 🛠️ Technology Stack

The dashboard is built entirely in Python using a powerful set of libraries:

| Category | Technology / Library | Purpose |
| :--- | :--- | :--- |
| **Framework & GUI** | `streamlit` | Main application and interactive dashboard creation. |
| **Data Acquisition** | `yfinance` | Real-time and historical crypto price data extraction. |
| **Deep Learning** | `tensorflow` / `keras` (LSTM) | Neural network implementation for time series prediction. |
| **Statistical Modeling** | `prophet`, `statsmodels` | ARIMA, SARIMA, and Facebook Prophet forecasting. |
| **Data Processing** | `pandas`, `numpy` | Data cleaning, preprocessing, and numerical calculations. |
| **Visualization** | `plotly` / `plotly.express` | Highly interactive charts and dashboards. |
| **Utility** | `textblob`, `streamlit-lottie` | Sentiment analysis and animations. |

---

## ⚙️ Installation (via Conda: macOS/Linux)

### Prerequisites
* **Anaconda or Miniconda** installed on your system.
* **Git** to clone the repository.
* A stable internet connection.

### 1. Clone the Repository
Open your terminal and clone the project:

```bash
git clone <YOUR-REPOSITORY-URL>
# Navigate to the project directory
cd TIME-SERIES-ANALYSIS-WITH-CRYPTOCURRENCY
