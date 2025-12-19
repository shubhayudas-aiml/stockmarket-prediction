import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import base64

st.title("Stock Price Predictor App")

# ======================================================
# SAFE Background Image (NO FileNotFoundError)
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
# USER INPUT
# ======================================================
stock = st.text_input("Enter Stock Symbol:", "GOOG").upper()

# ======================================================
# RELIABLE DATA FETCH (Streamlit Cloud SAFE)
# ======================================================
@st.cache_data(show_spinner=False)
def load_data(symbol):
    try:
        df = yf.download(
            symbol,
            period="max",
            auto_adjust=True,
            threads=False
        )
        return df
    except:
        return pd.DataFrame()

data = load_data(stock)

if data.empty:
    st.error("Unable to fetch stock data. Please try again later.")
    st.stop()

# Keep ASC for calculations
data_asc = data.copy()

# DESC for display
data = data_asc.sort_index(ascending=False)

st.subheader("Stock Data (Latest → Oldest)")
st.write(data)

# ======================================================
# MOVING AVERAGES (computed on ASC data)
# ======================================================
data_asc['MA_250'] = data_asc.Close.rolling(250).mean()
data_asc['MA_200'] = data_asc.Close.rolling(200).mean()
data_asc['MA_100'] = data_asc.Close.rolling(100).mean()

def plot_graph(title, ma_col, extra_col=None):
    fig = plt.figure(figsize=(15, 6))
    plt.plot(data_asc.Close, label="Actual Close", color="orange")
    plt.plot(data_asc[ma_col], label=ma_col, color="blue")
    if extra_col:
        plt.plot(data_asc[extra_col], label=extra_col, color="green")
    plt.legend()
    plt.title(title)
    st.pyplot(fig)

st.subheader("Moving Average for 250 days")
plot_graph("MA 250 Days", "MA_250")

st.subheader("Moving Average for 200 days")
plot_graph("MA 200 Days", "MA_200")

st.subheader("Moving Average for 100 days")
plot_graph("MA 100 Days", "MA_100")

st.subheader("MA 100 vs MA 250")
plot_graph("MA 100 vs MA 250", "MA_100", "MA_250")

# ======================================================
# MODEL TRAINING (Random Forest)
# ======================================================
df = data_asc[['Close']].copy()
df['Target'] = df['Close'].shift(-1)
df.dropna(inplace=True)

scaler = MinMaxScaler()
scaled = scaler.fit_transform(df)

X = scaled[:, 0].reshape(-1, 1)
y = scaled[:, 1]

model = RandomForestRegressor(random_state=42)
model.fit(X, y)

# Optional save
joblib.dump(model, "model.pkl")

# ======================================================
# PREDICTION
# ======================================================
scaled_pred = model.predict(X)

pred = scaler.inverse_transform(
    np.column_stack((scaled[:, 0], scaled_pred))
)

prediction_df = pd.DataFrame({
    "Actual Price": df['Target'],
    "Predicted Price": pred[:, 1]
}, index=df.index)

prediction_df = prediction_df.sort_index(ascending=False)

# ======================================================
# OUTPUT
# ======================================================
st.subheader("Actual vs Predicted Close Price")
st.write(prediction_df)

fig = plt.figure(figsize=(15, 6))
plt.plot(prediction_df["Actual Price"], label="Actual Price", color="blue")
plt.plot(prediction_df["Predicted Price"], label="Predicted Price", color="orange")
plt.legend()
st.pyplot(fig)
