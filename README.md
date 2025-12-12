# 📈 TIME SERIES ANALYSIS WITH CRYPTOCURRENCY

Welcome to the **TIME SERIES ANALYSIS WITH CRYPTOCURRENCY** repository!

This project is built using **Python 3.9**, **Streamlit**, and advanced **Time Series Forecasting Models** such as **ARIMA**, **SARIMA**, **Prophet**, and **LSTM**. The platform provides a complete dashboard for analyzing, visualizing, and forecasting cryptocurrency trends.

**🚀 Live Demo:** [https://time-series-analysis-with-cryptocurrency-e15v.onrender.com](https://time-series-analysis-with-cryptocurrency-e15v.onrender.com)

---

## 📚 Table of Contents
1. [Overview](#-overview)
2. [Features](#-features)
3. [Tech Stack](#-tech-stack)
4. [Prerequisites](#-prerequisites)
5. [Installation](#-installation)
    - [Creating Conda Environment](#2-creating-conda-environment)
    - [Installing Required Libraries](#3-installing-required-libraries)
6. [Database Setup](#-database-setup-authentication)
7. [Running the App](#-running-the-app)
8. [Project Structure](#-project-structure)
9. [Usage Guide](#-usage-guide)
10. [Screenshots](#-screenshots)
11. [Project Responsibility](#-project-responsibility)

---

## 📝 Overview

The **TIME SERIES ANALYSIS WITH CRYPTOCURRENCY** project is designed to help traders, investors, analysts, and researchers gain insights into crypto market behavior. 
It leverages data analytics, statistical modeling, and machine learning to predict future price movements based on historical data.

**Key Objectives:**
* Analyze cryptocurrency price trends using time series forecasting techniques.
* Predict future price movements using historical data. 
* Provide a Graphical User Interface (GUI) for real-time trends and predictive insights. 

---

## ⭐ Features

The dashboard is logically structured into powerful modules accessible via the sidebar:

1.  **Real-Time & Historical Data Collection:** Extracts live data from API sources like Yahoo Finance (`yfinance`). 
2.  **Data Preprocessing & Exploration:** Cleans and processes data, handles missing values, and visualizes trends using Pandas and Plotly. 
3.  **Forecasting Models (The AI Engine):**
    * **ARIMA & SARIMA:** Classical statistical models for trend projection. 
    * **Facebook Prophet:** Robust forecasting for time-series data with seasonality.
    * **LSTM (Long Short-Term Memory):** Deep learning neural network for complex pattern recognition. 
4.  **Sentiment Analysis:** Analyzes news headlines using NLP (`TextBlob`) to gauge market sentiment (Bullish/Bearish). 
5.  **Risk & Volatility Analysis:** Visualizes market volatility using Bollinger Bands and return distributions. 
6.  **Interactive Dashboard:** A modern GUI built with Streamlit featuring interactive charts and tickers. 
7.  **Secure Authentication:** User login system backed by a local JSON database.
8.  **Data Export:** Capability to download analyzed datasets as CSV files.

---

## 🛠️ Tech Stack

The project relies on a robust Python ecosystem:

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Language** | Python 3.9 | Core programming language. |
| **Frontend/GUI** | Streamlit | [cite_start]Interactive web-based dashboard[cite: 28]. |
| **Data Source** | yfinance | [cite_start]Real-time crypto market data fetcher[cite: 28]. |
| **Data Processing** | Pandas, NumPy | [cite_start]Data manipulation and numerical analysis[cite: 28]. |
| **Visualization** | Plotly, Plotly Express | [cite_start]Interactive financial charting[cite: 28]. |
| **Forecasting** | Prophet, Statsmodels | [cite_start]Statistical time series modeling[cite: 28]. |
| **Deep Learning** | TensorFlow (Keras) | [cite_start]LSTM Neural Network implementation[cite: 28]. |
| **NLP** | TextBlob | [cite_start]Sentiment analysis on text data[cite: 28]. |
| **Utilities** | Streamlit-Lottie | [cite_start]Animation integration[cite: 28]. |

---

## 📋 Prerequisites

Before starting, ensure you have the following:

* **Anaconda or Miniconda** installed on your system.
* **Git** for version control.
* A stable internet connection for fetching real-time data.

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone <YOUR-REPOSITORY-URL>
# Navigate to the project directory
cd TIME-SERIES-ANALYSIS-WITH-CRYPTOCURRENCY
```

### 2. Creating Conda Environment

We recommend using a dedicated Conda environment (Python 3.9 recommended) to manage dependencies.
```bash
# Create a new environment named 'crypto_env'
conda create -n crypto_env python=3.9
# Activate the environment
conda activate crypto_env
```
3. Installing Required Libraries
Install all required packages listed in requirements.txt using pip while your Conda environment is active:
```bash
pip install -r requirements.txt
```
⚠️ Note on Dependencies (macOS):The prophet and tensorflow libraries may require system dependencies like the GCC compiler and OpenSSL. If you encounter errors, install them using Homebrew:

```bash
brew install openssl
```
## ▶️ Running the App
Launch the Streamlit server from the project directory. Ensure your Conda environment is active.
```bash
streamlit run crypto_app.py
```
The dashboard will automatically open in your web browser, typically at (http://localhost:8501).

## 📂 Project Structure
```Plaintext
TIME-SERIES-ANALYSIS-WITH-CRYPTOCURRENCY/
├── crypto_app.py        # Main application file (Streamlit)
├── requirements.txt     # List of python dependencies
├── users_db.json        # JSON database for user credentials
├── assets/              # (Optional) Folder for static assets/images
└── README.md            # Project documentation
```

