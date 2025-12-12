# 📈 TIME SERIES ANALYSIS WITH CRYPTOCURRENCY

Welcome to the **TIME SERIES ANALYSIS WITH CRYPTOCURRENCY** repository!

This project is built using **Python 3.9**, **Streamlit**, and advanced **Time Series Forecasting Models** such as **ARIMA**, **SARIMA**, **Prophet**, and **LSTM**. The platform provides a complete dashboard for analyzing, visualizing, and forecasting cryptocurrency trends.

**🚀 Live on web:** [https://time-series-analysis-with-cryptocurrency-e15v.onrender.com](https://time-series-analysis-with-cryptocurrency-e15v.onrender.com)

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
## 📖 Usage Guide
1. Authentication (Register & Login):
   - Open the app in your browser.
   -  Go to the sidebar "User Access".
   - Register: Click the "Register" tab, enter a new Username and Password, and click "Create Account".
   - Login: Switch to the "Login" tab and enter your newly registered credentials to access the dashboard.
2. Select Asset:
   - In the "Parameter Tuning" section (Sidebar), enter a ticker symbol (e.g., BTC-USD, ETH-USD).
   - Set the Start and End dates.
3. Analyze:
   - Navigate through modules like Price Explorer, Advanced Forecasting, and Risk & Volatility to view charts and predictions.
## 📸 Screenshots
###1. DASHBOARD
<img width="1440" height="784" alt="Screenshot 2025-12-12 at 8 31 22 PM" src="https://github.com/user-attachments/assets/a21f95b1-f9d2-47fa-8aa5-62bcfb4aa38c" />


###2. Register and Login

<img width="231" height="601" alt="Screenshot 2025-12-12 at 8 39 32 PM" src="https://github.com/user-attachments/assets/953cabc1-8231-49a2-84a8-0af3893c3da3" />

   - Register:
<img width="240" height="535" alt="Screenshot 2025-12-12 at 8 39 59 PM" src="https://github.com/user-attachments/assets/28eecf2f-c388-4896-928f-2c2d992cc5ca" />

   - Login:
<img width="246" height="342" alt="Screenshot 2025-12-12 at 8 40 42 PM" src="https://github.com/user-attachments/assets/fc2f909d-8998-4451-a226-9e16498acbcd" />

###3. ⚙️ Parameter Tuning

<img width="235" height="409" alt="Screenshot 2025-12-12 at 8 32 20 PM" src="https://github.com/user-attachments/assets/d06418a3-33c9-4faa-914f-9adfc41f6502" />

###4. Price Movement & Trend
<img width="1187" height="849" alt="Screenshot 2025-12-12 at 8 32 58 PM" src="https://github.com/user-attachments/assets/f849754e-c178-4f55-a244-024d3a6384f9" />

###5. Interactive Candle Anlysis(Price Explorer)

<img width="1142" height="701" alt="Screenshot 2025-12-12 at 8 33 46 PM" src="https://github.com/user-attachments/assets/44d14f94-e85b-4593-9e12-6f978cc6b61a" />

###7.Forecasting Models
  - **Arima**
<img width="1123" height="727" alt="Screenshot 2025-12-12 at 8 34 07 PM" src="https://github.com/user-attachments/assets/3fa5b3fa-b9f5-44e1-990b-d9be6dc38700" />
  - **Sarima**
  <img width="1063" height="732" alt="Screenshot 2025-12-12 at 8 35 09 PM" src="https://github.com/user-attachments/assets/c1bcdb98-68a0-4f06-9fc0-6339974c3b15" />
  - **Prophet**
  <img width="1096" height="693" alt="Screenshot 2025-12-12 at 8 35 32 PM" src="https://github.com/user-attachments/assets/d474d927-fa6d-4b51-af6c-3456b80ccf82" />
  - **LSTM model**
  <img width="1125" height="731" alt="Screenshot 2025-12-12 at 8 35 57 PM" src="https://github.com/user-attachments/assets/d386c0ef-c27a-49cc-816d-80c31c64df01" />
  
###8. Sentiment Analysis

<img width="1106" height="761" alt="Screenshot 2025-12-12 at 8 36 26 PM" src="https://github.com/user-attachments/assets/1a3f8989-fdb9-41ad-ab32-88b1444707f1" />

9###. Risk & Volatility
<img width="1100" height="692" alt="Screenshot 2025-12-12 at 8 36 57 PM" src="https://github.com/user-attachments/assets/7d9be616-1199-4873-a355-91ca2d67d7d4" />
<img width="1095" height="666" alt="Screenshot 2025-12-12 at 8 37 07 PM" src="https://github.com/user-attachments/assets/90be09b0-7bcb-47a0-bd02-2ae8c81efbb5" />

###10. Technical Indicator 
<img width="1161" height="807" alt="Screenshot 2025-12-12 at 8 37 30 PM" src="https://github.com/user-attachments/assets/23830320-5c2a-4d8d-b080-561ddffebb4e" />

###11. Correlation
<img width="1090" height="719" alt="Screenshot 2025-12-12 at 8 37 51 PM" src="https://github.com/user-attachments/assets/e4d97f03-a668-460d-af77-25986bb27e62" />

###11. Feature Importance 
<img width="1122" height="736" alt="Screenshot 2025-12-12 at 8 38 13 PM" src="https://github.com/user-attachments/assets/7b6832fa-6ac3-4f6a-8ca4-573899353a7b" />

###12. Strategy Backtest
<img width="1101" height="762" alt="Screenshot 2025-12-12 at 8 38 37 PM" src="https://github.com/user-attachments/assets/0d64f11c-f6d5-4ea7-af2d-e97b9813ef16" />

###13. Data Export 
<img width="1120" height="607" alt="Screenshot 2025-12-12 at 8 38 49 PM" src="https://github.com/user-attachments/assets/3fe5168a-e86b-4f8e-a7d2-8f6d62ec6102" />

###14. Changing Parameter:
<img width="244" height="365" alt="Screenshot 2025-12-12 at 8 41 33 PM" src="https://github.com/user-attachments/assets/9c752463-0b49-4eba-9ea1-a62d5a73c9ee" />








    


  





