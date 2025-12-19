import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import joblib
import base64
from datetime import datetime

st.title("Stock Price Predictor App")

# ====== Background Image =======
def get_base64(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img_base64 = get_base64("s2.jpg")

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

# ========= USER INPUT =========
stock = st.text_input("Enter Stock Symbol:", "GOOG")

# ========= FETCH DATA =========
end = datetime.now()
start = datetime(end.year - 20, end.month, end.day)


data = yf.download(stock, start, end)
data = data.sort_index(ascending=False)

st.subheader("Stock Data")
st.write(data)


# ========= MOVING AVERAGES =========
def plot_graph(figsize, values, full_data, extra_data=0, extra_dataset=None):
    fig = plt.figure(figsize=figsize)
    plt.plot(values, 'orange')
    plt.plot(full_data.Close, 'b')
    if extra_data:
        plt.plot(extra_dataset)
    return fig

data['MA_250'] = data.Close.rolling(250).mean()
data['MA_200'] = data.Close.rolling(200).mean()
data['MA_100'] = data.Close.rolling(100).mean()

st.subheader("Moving Average for 250 days")
st.pyplot(plot_graph((15, 6), data['MA_250'], data))

st.subheader("Moving Average for 200 days")
st.pyplot(plot_graph((15, 6), data['MA_200'], data))

st.subheader("Moving Average for 100 days")
st.pyplot(plot_graph((15, 6), data['MA_100'], data))

st.subheader("Moving Average 100 vs Moving Average 250")
st.pyplot(plot_graph((15, 6), data['MA_100'], data, 1, data['MA_250']))

# ========= MODEL TRAINING =========

# prepare data
df = data[['Close']].copy()
df['Target'] = df['Close'].shift(-1)  # predict next day close
df = df.dropna()

scaler = MinMaxScaler()
scaled = scaler.fit_transform(df)

X = scaled[:, 0].reshape(-1, 1)        # Close
y = scaled[:, 1]                       # Next day Close

# train model
model = RandomForestRegressor()
model.fit(X, y)

# save model (optional)
joblib.dump(model, "model.pkl")

# ========= PREDICTION =========
scaled_pred = model.predict(X)
pred = scaler.inverse_transform(
    np.concatenate([scaled[:, 0].reshape(-1, 1), scaled_pred.reshape(-1, 1)], axis=1)
)

prediction_df = pd.DataFrame({
    "Actual": df['Target'],
    "Predicted": pred[:, 1]
}, index=df.index)



st.subheader("Actual vs Predicted Close Price")
prediction_df = prediction_df.sort_index(ascending=False)
st.write(prediction_df)


fig = plt.figure(figsize=(15, 6))
plt.plot(prediction_df["Actual"], label="Actual", color="blue")
plt.plot(prediction_df["Predicted"], label="Predicted", color="orange")
plt.legend()
st.pyplot(fig)
