import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import joblib

st.title("Stock Price Predictor App")

# ======================================================
# USER INPUT
# ======================================================
stock = st.text_input("Enter Stock Symbol:", "GOOG").upper()

# ======================================================
# DATA LOADING (LIVE + FALLBACK)
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
        if not df.empty:
            return df, "live"
    except:
        pass

    # Offline fallback (always works)
    df = pd.read_csv("stock_data_GOOG.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")
    return df, "offline"

data, source = load_data(stock)

if source == "offline":
    st.info("Live data unavailable. Showing historical dataset.")

# ======================================================
# KEEP ASC FOR CALCULATION, DESC FOR DISPLAY
# ======================================================
data_asc = data.copy()
data_desc = data_asc.sort_index(ascending=False)

st.subheader("Stock Data (Latest → Oldest)")
st.write(data_desc)

# ======================================================
# MOVING AVERAGES
# ======================================================
data_asc["MA_100"] = data_asc.Close.rolling(100).mean()
data_asc["MA_200"] = data_asc.Close.rolling(200).mean()
data_asc["MA_250"] = data_asc.Close.rolling(250).mean()

def plot_ma(title, col1, col2=None):
    fig = plt.figure(figsize=(15, 6))
    plt.plot(data_asc.Close, label="Actual Close Price", color="orange")
    plt.plot(data_asc[col1], label=col1, color="blue")
    if col2:
        plt.plot(data_asc[col2], label=col2, color="green")
    plt.legend()
    plt.title(title)
    st.pyplot(fig)

st.subheader("Moving Average (100 Days)")
plot_ma("MA 100 Days", "MA_100")

st.subheader("Moving Average (200 Days)")
plot_ma("MA 200 Days", "MA_200")

st.subheader("Moving Average (250 Days)")
plot_ma("MA 250 Days", "MA_250")

st.subheader("MA 100 vs MA 250")
plot_ma("MA 100 vs MA 250", "MA_100", "MA_250")

# ======================================================
# MODEL TRAINING (RANDOM FOREST)
# ======================================================
df = data_asc[["Close"]].copy()
df["Target"] = df["Close"].shift(-1)
df.dropna(inplace=True)

scaler = MinMaxScaler()
scaled = scaler.fit_transform(df)

X = scaled[:, 0].reshape(-1, 1)
y = scaled[:, 1]

model = RandomForestRegressor(random_state=42)
model.fit(X, y)

joblib.dump(model, "model.pkl")

# ======================================================
# PREDICTION
# ======================================================
scaled_pred = model.predict(X)

pred = scaler.inverse_transform(
    np.column_stack((scaled[:, 0], scaled_pred))
)

prediction_df = pd.DataFrame(
    {
        "Actual Price": df["Target"],
        "Predicted Price": pred[:, 1]
    },
    index=df.index
)

prediction_df = prediction_df.sort_index(ascending=False)

# ======================================================
# OUTPUT TABLE
# ======================================================
st.subheader("Actual vs Predicted Values (Latest → Oldest)")
st.write(prediction_df)

# ======================================================
# ✅ FIXED: OUTPUT GRAPH HEADING ADDED
# ======================================================
st.subheader("Original vs Predicted Close Price")

fig = plt.figure(figsize=(15, 6))
plt.plot(prediction_df["Actual Price"], label="Actual Price", color="blue")
plt.plot(prediction_df["Predicted Price"], label="Predicted Price", color="orange")
plt.legend()
st.pyplot(fig)
