import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

st.title("Stock Price Predictor App")

import base64

def get_base64(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img_base64 = get_base64("sp1.jpg")

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



stock = st.text_input("Enter Stock Ticker", "GOOG")

end = datetime.now()
start = datetime(end.year - 20, end.month, end.day)

google_data = yf.download(stock, start, end)

if google_data.empty:
    st.error("Invalid stock ticker")
    st.stop()

if isinstance(google_data.columns, pd.MultiIndex):
    google_data.columns = ['_'.join(col).strip() for col in google_data.columns.values]

if f"Close_{stock}" in google_data.columns and "Adj Close" not in google_data.columns:
    google_data.rename(columns={f"Close_{stock}": "Adj Close"}, inplace=True)

st.subheader("Stock Data")
st.write(google_data.tail())

model = load_model("Latest_stock_price_model.keras")

google_data["MA_100"] = google_data["Adj Close"].rolling(100).mean()
google_data["MA_200"] = google_data["Adj Close"].rolling(200).mean()
google_data["MA_250"] = google_data["Adj Close"].rolling(250).mean()

def plot_data(data, title):
    fig = plt.figure(figsize=(15,6))
    plt.plot(data)
    plt.title(title)
    st.pyplot(fig)

st.subheader("Moving Averages")
plot_data(google_data[["Adj Close","MA_100"]], "Adj Close & MA 100")
plot_data(google_data[["Adj Close","MA_200"]], "Adj Close & MA 200")
plot_data(google_data[["Adj Close","MA_250"]], "Adj Close & MA 250")

Adj_close_price = google_data[["Adj Close"]]

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(Adj_close_price)

x_data = []
y_data = []

for i in range(100, len(scaled_data)):
    x_data.append(scaled_data[i-100:i])
    y_data.append(scaled_data[i])

x_data = np.array(x_data)
y_data = np.array(y_data)

splitting_len = int(len(x_data) * 0.7)

x_test = x_data[splitting_len:]
y_test = y_data[splitting_len:]

predictions = model.predict(x_test)

inv_predictions = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_test)

ploting_data = pd.DataFrame(
    {
        "Original": inv_y_test.reshape(-1),
        "Predicted": inv_predictions.reshape(-1)
    },
    index=google_data.index[splitting_len+100:]
)

st.subheader("Original vs Predicted Values")
st.write(ploting_data.tail())

st.subheader("Final Prediction Plot")
fig = plt.figure(figsize=(15,6))
plt.plot(pd.concat([Adj_close_price[:splitting_len+100], ploting_data], axis=0))
plt.legend(["Training Data","Original Test Data","Predicted Data"])
st.pyplot(fig)
