# Stock Market Price Movement Prediction System

A Machine Learning project that predicts the next-day movement of stock prices using historical S&P 500 stock data. The project includes data preprocessing, feature engineering, model training, evaluation, and an interactive Streamlit web application for real-time predictions.


Dataset

\- S\&P 500 Stock Data from Kaggle

\- Stock used: Apple Inc. (AAPL)

\- 5 years of historical data (\~1258 trading days)


Technologies Used

\- Python

\- Pandas

\- Scikit-learn (Random Forest Classifier)

\- Streamlit (Web App)

\- Plotly (Charts)



\## Features Used

\- Open, High, Low, Close Price

\- Trading Volume

\- 7-day \& 21-day Moving Average

\- Daily Return

\- 7-day Volatility

\- Price Range



\## How to Run

pip install -r requirements.txt

streamlit run app.py



\## Project Structure

\- app.py — Streamlit web application

\- all\_stocks\_5yr.csv — Dataset

\- stock\_prediction\_model.pkl — Trained ML model

\- requirements.txt — Dependencies



\## Disclaimer

For educational purposes only. Not financial advice.

