# stockmarket-prediction
📈 Stock Market Prediction Web App

🔗 Live Demo: https://s-das-stockprice.streamlit.app

🚀 Project Overview

This project is a machine learning–based stock market prediction web application, developed as a Final Year B.Tech (CSE) project.
It analyzes historical stock market data and predicts future stock prices using trained machine learning models, presented through an interactive Streamlit web interface.

The application enables users to:

Enter a stock symbol

Visualize historical price trends

Compare actual vs predicted prices
—all directly in the browser, without any local setup.

🧠 Key Features

✔ Fetches historical stock market data using yfinance

✔ Performs data preprocessing and normalization

✔ Predicts future stock prices using a trained ML model

✔ Visualizes trends with moving averages and comparison charts

✔ Interactive and user-friendly UI built with Streamlit

✔ Deployed live on Streamlit Cloud

🗂️ Repository Structure
| File / Folder                    | Description                               |
| -------------------------------- | ----------------------------------------- |
| `stock_app.py`                   | Main Streamlit web application            |
| `requirements.txt`               | Python dependencies                       |
| `runtime.txt`                    | Python runtime configuration              |
| `Stock Price.ipynb`              | Data analysis and model training notebook |
| `Latest_stock_price_model.keras` | Trained machine learning model            |
| `stock_price`                    | Scaler file for feature normalization     |
| `s2.jpg`                         | Application UI background image           |

🛠️ Technologies Used

✔Python

✔Streamlit

✔TensorFlow 

✔Keras

✔scikit-learn

✔pandas

numpy

✔matplotlib

✔yfinance

📊 How the System Works

✔ Historical stock data is collected using the yfinance API

✔ Data is cleaned, processed, and scaled

✔ A trained machine learning model predicts future stock prices

✔ Actual and predicted prices are visualized using interactive charts

✔ The final output is delivered through a Streamlit web application

💻 Local Setup (Optional)

This is required only if you want to run the project locally.

1️⃣ Clone the Repository
git clone https://github.com/shubhayudas-aiml/stockmarket-prediction.git

2️⃣ Navigate to the Project Directory
cd stockmarket-prediction

3️⃣ Create a Virtual Environment
python -m venv venv


Activate it:

Windows

venv\Scripts\activate


macOS / Linux

source venv/bin/activate

4️⃣ Install Dependencies
pip install -r requirements.txt

5️⃣ Run the App
streamlit run stock_app.py

🌐 Live Application

👉 Access the deployed app:
https://s-das-stockprice.streamlit.app

🎓 Academic Context

This project was developed as part of a Final Year B.Tech (CSE) Project, focusing on the practical application of machine learning techniques in financial data analysis, along with real-world deployment using Streamlit Cloud.

📬 Feedback & Contributions

Suggestions, improvements, and feedback are welcome.
Feel free to open an issue or submit a pull request.

🙌 Author

Shubhayu Das
AI / Machine Learning Enthusiast | Computer Science Engineer

🔗 GitHub: https://github.com/shubhayudas-aiml

🔗 Live App: https://s-das-stockprice.streamlit.app
