📈 Time Series Analysis with Cryptocurrency

A complete time-series forecasting dashboard built using Python 3.9, Streamlit, ARIMA, SARIMA, Prophet, LSTM, and YFinance.

This project analyzes cryptocurrency price trends using statistical modeling & machine-learning forecasting models and displays everything in an interactive Streamlit dashboard.
🚀 Features
✅ Data Collection

Real-time and historical crypto price data (Yahoo Finance API)

✅ Data Preprocessing

Handling missing values

Smoothing

Normalization (MinMaxScaler)

✅ Exploratory Data Analysis

Trend lines

Candlestick charts

Volume charts

Moving averages
| Model       | Library     | Purpose              |
| ----------- | ----------- | -------------------- |
| **ARIMA**   | statsmodels | Classic forecasting  |
| **SARIMA**  | statsmodels | Seasonality-aware    |
| **Prophet** | Prophet     | Long-term prediction |
| **LSTM**    | TensorFlow  | Deep learning model  |

✅ Sentiment Analysis

Polarity score via TextBlob

✅ Streamlit Dashboard

Interactive visualizations

Model comparison

Forecasts

User login system via users_db.json

📁 Project Structure
Time-Series-Analysis-with-Cryptocurrency/
│── crypto_app.py          → Streamlit Application
│── requirements.txt       → Python dependencies
│── users_db.json          → Local login/user storage
│── README.md              → Documentation
└── data/ (optional)       → Saved CSVs

📦 Installation & Setup
1️⃣ Create and activate Conda environment (Python 3.9)
conda create -n crypto_env python=3.9
conda activate crypto_env
2️⃣ Install dependencies
pip install -r requirements.txt
If Prophet fails to install:
pip install prophet
3️⃣ Optional (recommended for macOS)
xcode-select --install
pip install watchdog
▶️ Run the Streamlit App

Inside your project folder:
streamlit run crypto_app.py
You will see:
Local URL: http://localhost:8501
🌐 Deployment (Render / Streamlit Cloud)
Render Deployment Steps

Push the project to GitHub

Go to https://render.com

Create a New Web Service

Select your GitHub repo

Set:

Build Command

pip install -r requirements.txt


Start Command

streamlit run crypto_app.py --server.port=$PORT --server.address=0.0.0.0

🔐 Login System

Your users_db.json:

{
  "admin": "password123",
  "syamantak06": "1234",
  "say": "123",
  "hello": "helo"
}


You can add/remove users simply by editing the JSON file.

📚 Requirements (from requirements.txt)

streamlit

yfinance

pandas

numpy

plotly

textblob

prophet

statsmodels

scikit-learn

tensorflow

requests

streamlit-lottie

streamlit-autorefresh

💻 Commands You Used
conda create -n crypto_env python=3.9
conda activate crypto_env
pip install -r requirements.txt
streamlit run crypto_app.py
xcode-select --install
pip install watchdog


Stop Streamlit:

CTRL + C
