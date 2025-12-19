# stockmarket-prediction
📈 Stock Market Prediction Web App

🔗 Live Demo:
https://s-das-stockprice.streamlit.app

🚀 Project Overview

This project is an AI-powered stock market prediction web application developed as a final year B.Tech project.
It analyzes historical stock market data and predicts future stock prices using machine learning models, presented through an interactive Streamlit web interface.

The application allows users to select a stock symbol, visualize historical price trends, and view predicted prices directly in the browser without any local setup.

🧠 Key Features

✔ Fetches historical stock market data using yfinance

✔ Applies data preprocessing and normalization techniques

✔ Predicts future stock prices using trained ML models

✔ Interactive and user-friendly visualizations

✔ Clean and responsive UI built with Streamlit

✔ Deployed live on Streamlit Cloud for public access

🗂️ Repository Structure
| File / Folder                    | Description                                 |
| -------------------------------- | ------------------------------------------- |
| `stock_app.py`                   | Main Streamlit application                  |
| `requirements.txt`               | Python dependencies                         |
| `runtime.txt`                    | Python version configuration for deployment |
| `Stock Price.ipynb`              | Notebook for data analysis & model training |
| `Latest_stock_price_model.keras` | Trained ML model                            |
| `stock_price`                    | Scaler file for feature normalization       |
| `s2.jpg`                         | Application UI image / asset                |



🛠️ Technologies Used

1. Python

2. Streamlit

3. TensorFlow / Keras

4. scikit-learn

5. pandas
   
6.  numpy

7. matplotlib

8. yfinance

📊 How the System Works

✔Stock market data is collected using the yfinance API

✔Data is cleaned and scaled using preprocessing techniques

✔A trained machine learning model predicts future prices

✔Predictions and historical trends are visualized interactively

✔The final output is served through a Streamlit web application


💻 Local Setup (Optional)

This step is only required if you want to run the project locally.

1️⃣ Clone the Repository
git clone https://github.com/shubhayudas-aiml/stockmarket-prediction.git

2️⃣ Navigate to the Project Directory
cd stockmarket-prediction

3️⃣ Create a Virtual Environment
python -m venv venv

Activate it:

✨Windows
venv\Scripts\activate

✨macOS / Linux
source venv/bin/activate

4️⃣ Install Dependencies
pip install -r requirements.txt

5️⃣ Run the App
streamlit run stock_app.py


🌐 Live Application

👉 Try the live app here:
https://s-das-stockprice.streamlit.app

🎓 Academic Context

This project was developed as part of a Final Year B.Tech (CSE) Project, focusing on the practical application of machine learning in financial data analysis and real-world web deployment using Streamlit Cloud.

📬 Feedback & Contributions

Suggestions, improvements, and feedback are welcome.
Feel free to open an issue or submit a pull request.

🙌 Author

Shubhayu Das
AI / Machine Learning Enthusiast | Computer Science Engineer

🔗 GitHub: https://github.com/shubhayudas-aiml

🔗 Live App: https://s-das-stockprice.streamlit.app
