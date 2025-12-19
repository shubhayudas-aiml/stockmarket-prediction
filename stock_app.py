import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import base64
import os

st.title("Stock Price Predictor App")

# ======================================================
# Optional Background Image (SAFE)
# ======================================================
def get_base64(file_path):
    if not os.path.exists(file_path):
        return None
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img_base64 = get_base64("s2.jpg")

if img_base64:
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{img_base64}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ======================================================
# User Input
# ======================================================
stock = st.text_input("Enter the Stock ID", "GOOG").upper()

# ======================================================
# Robust Stock Data Loader (LIVE + BACKUP)
# ======================================================
@st.cache_data(show_spinner=False)
def load_stock_data(ticker):
    # 1️⃣ Try Yahoo Finance (LIVE)
    try:
        data = yf.download(
            ticker,
            period="max",
            auto_adjust=True,
            threads=False
        )
        if not data.empty:
            return data, "live"
    except:
        pass

    # 2️⃣ Fallback to backup CSV (ROBUST)
    try:
        data = pd.read_csv("backup_GOOG.csv")
        data.columns = [c.strip() for c in data.columns]

        if "Date" in data.columns:
            data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
            data = data.set_index("Date")

        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        data = data[[c for c in required_cols if c in data.columns]]
        data = data.dropna()

        return data, "backup"
    except:
        return pd.DataFrame(), "none"

google_data, source = load_stock_data(stock)

if google_data.empty:
    st.error("Unable to load stock data (live and backup both unavailable).")
    st.stop()

if source == "backup":
    st.info("Live data unavailable. Showing cached historical data.")

# ======================================================
# Keep ASC for ML, DESC for Display
# ======================================================
google_data_asc = google_data.copy()
google_data_desc = google_data_asc.sort_index(ascending=False)

# ======================================================
# Load Trained Model
# ======================================================
model = load_model("Latest_stock_price_model.keras")

# ======================================================
# Display Stock Data
# ======================================================
st.subheader("Stock Data (Latest to Oldest)")
st.write(google_data_desc)

# ======================================================
# Moving Averages
# ======================================================
google_data_asc["MA_100"] = google_data_asc.Close.rolling(100).mean()
google_data_asc["MA_200"] = google_data_asc.Close.rolling(200).mean()
google_data_asc["MA_250"] = google_data_asc.Close.rolling(250).mean()

def plot_ma(title, ma_column):
    fig = plt.figure(figsize=(15,6))
    plt.plot(google_data_asc.Close, label="Actual Close Price", color="orange")
    plt.plot(google_data_asc[ma_column], label=ma_column, color="blue")
    plt.legend()
    plt.title(title)
    st.pyplot(fig)

st.subheader("Original Close Price vs MA (100 Days)")
plot_ma("MA 100 Days", "MA_100")

st.subheader("Original Close Price vs MA (200 Days)")
plot_ma("MA 200 Days", "MA_200")

st.subheader("Original Close Price vs MA (250 Days)")
plot_ma("MA 250 Days", "MA_250")

st.subheader("Original Close Price vs MA (100 & 250 Days)")
fig = plt.figure(figsize=(15,6))
plt.plot(google_data_asc.Close, label="Actual Close Price", color="orange")
plt.plot(google_data_asc.MA_100, label="MA 100 Days", color="green")
plt.plot(google_data_asc.MA_250, label="MA 250 Days", color="blue")
plt.legend()
st.pyplot(fig)

# ======================================================
# Train / Test Split
# ======================================================
splitting_len = int(len(google_data_asc) * 0.8)
x_test = pd.DataFrame(google_data_asc["Close"][splitting_len:])
x_test.columns = ["Close"]

# ======================================================
# Scaling & Sequence Creation
# ======================================================
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(x_test)

x_data, y_data = [], []

for i in range(100, len(scaled_data)):
    x_data.append(scaled_data[i-100:i])
    y_data.append(scaled_data[i])

x_data = np.array(x_data)
y_data = np.array(y_data)

# ======================================================
# Prediction
# ======================================================
predictions = model.predict(x_data)

inv_pred = scaler.inverse_transform(predictions)
inv_y = scaler.inverse_transform(y_data)

# Smooth predictions (visual improvement)
inv_pred_smoothed = (
    pd.Series(inv_pred.reshape(-1))
    .rolling(3)
    .mean()
    .bfill()
    .values
)

# ======================================================
# Prediction DataFrame
# ======================================================
ploting_data = pd.DataFrame(
    {
        "Actual Price": inv_y.reshape(-1),
        "Predicted Price": inv_pred_smoothed
    },
    index=google_data_asc.index[splitting_len+100:]
)

# Align with full timeline
ploting_data = ploting_data.reindex(google_data_asc.index)

# Fill actual prices everywhere
ploting_data["Actual Price"] = google_data_asc["Close"]

# Display latest first
ploting_data = ploting_data.sort_index(ascending=False)

# ======================================================
# Output
# ======================================================
st.subheader("Original vs Predicted Values (Latest to Oldest)")
st.write(ploting_data)

st.subheader("Original vs Predicted Close Price")
fig = plt.figure(figsize=(15,6))
plt.plot(ploting_data["Actual Price"], label="Actual Price", color="orange")
plt.plot(ploting_data["Predicted Price"], label="Predicted Price", color="blue")
plt.legend()
st.pyplot(fig)
