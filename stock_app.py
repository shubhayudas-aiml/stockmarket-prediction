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

# ---------- Background Image (SAFE) ----------
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

# ---------- Input ----------
stock = st.text_input("Enter the Stock ID", "GOOG")

# ---------- Load Data (RELIABLE ON STREAMLIT CLOUD) ----------
google_data = yf.download(stock, period="max")

if google_data is None or google_data.empty:
    st.error("Unable to fetch stock data. Please try again later.")
    st.stop()

# ---------- Keep ASC for ML ----------
google_data_asc = google_data.copy()

# ---------- DESC for display ----------
google_data_desc = google_data_asc.sort_index(ascending=False)

model = load_model("Latest_stock_price_model.keras")

st.subheader("Stock Data (2025 → 2005)")
st.write(google_data_desc)

# ---------- Moving Averages ----------
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

# ---------- Train/Test Split ----------
splitting_len = int(len(google_data_asc) * 0.8)
x_test = pd.DataFrame(google_data_asc["Close"][splitting_len:])
x_test.columns = ["Close"]

# ---------- Scaling ----------
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(x_test)

x_data, y_data = [], []

for i in range(100, len(scaled_data)):
    x_data.append(scaled_data[i-100:i])
    y_data.append(scaled_data[i])

x_data = np.array(x_data)
y_data = np.array(y_data)

# ---------- Prediction ----------
predictions = model.predict(x_data)

inv_pred = scaler.inverse_transform(predictions)
inv_y = scaler.inverse_transform(y_data)

# ---------- Smooth Predictions ----------
inv_pred_smoothed = (
    pd.Series(inv_pred.reshape(-1))
    .rolling(3)
    .mean()
    .bfill()
    .values
)

# ---------- Prediction DataFrame ----------
ploting_data = pd.DataFrame(
    {
        "Actual Price": inv_y.reshape(-1),
        "Predicted Price": inv_pred_smoothed
    },
    index=google_data_asc.index[splitting_len+100:]
)

# Align with full timeline
ploting_data = ploting_data.reindex(google_data_asc.index)

# Fill actual price
ploting_data["Actual Price"] = google_data_asc["Close"]

# Display latest first
ploting_data = ploting_data.sort_index(ascending=False)

# ---------- Output ----------
st.subheader("Original vs Predicted Values (2025 → 2005)")
st.write(ploting_data)

st.subheader("Original vs Predicted Close Price")
fig = plt.figure(figsize=(15,6))
plt.plot(ploting_data["Actual Price"], label="Actual Price", color="orange")
plt.plot(ploting_data["Predicted Price"], label="Predicted Price", color="blue")
plt.legend()
st.pyplot(fig)
