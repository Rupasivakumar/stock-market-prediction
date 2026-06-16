import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="Stock Market Prediction",
    page_icon="📈",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load('stock_prediction_model.pkl')

@st.cache_data
def load_data():
    df = pd.read_csv('all_stocks_5yr.csv')
    df = df[df['Name'] == 'AAPL'].copy()
    df = df.sort_values('date').reset_index(drop=True)
    df['MA7']         = df['close'].rolling(7).mean()
    df['MA21']        = df['close'].rolling(21).mean()
    df['return']      = df['close'].pct_change()
    df['volatility']  = df['return'].rolling(7).std()
    df['price_range'] = df['high'] - df['low']
    df['Target']      = (df['close'].shift(-1) > df['close']).astype(int)
    df.dropna(inplace=True)
    return df

model = load_model()
df    = load_data()

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Home & Predict", "Stock Charts", "EDA Analysis", "About"]
)

# ── PAGE 1: HOME & PREDICT ──────────────────────────────
if page == "Home & Predict":
    st.title("Stock Market Price Movement Prediction")
    st.markdown("Predicting whether Apple (AAPL) stock will go UP or DOWN the next day")
    st.markdown("---")
    st.header("Predict Next Day Movement")

    latest = df.iloc[-1]
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Enter Stock Values")
        open_price  = st.number_input("Open Price ($)",  value=float(round(latest['open'],  2)), step=0.01)
        high_price  = st.number_input("High Price ($)",  value=float(round(latest['high'],  2)), step=0.01)
        low_price   = st.number_input("Low Price ($)",   value=float(round(latest['low'],   2)), step=0.01)
        close_price = st.number_input("Close Price ($)", value=float(round(latest['close'], 2)), step=0.01)
        volume      = st.number_input("Volume",          value=float(latest['volume']),           step=1000.0)

    with col2:
        st.subheader("Technical Indicators")
        ma7         = st.number_input("MA7 (7-day Moving Avg)",   value=float(round(latest['MA7'],        2)), step=0.01)
        ma21        = st.number_input("MA21 (21-day Moving Avg)", value=float(round(latest['MA21'],       2)), step=0.01)
        ret         = st.number_input("Daily Return",             value=float(round(latest['return'],     5)), step=0.00001, format="%.5f")
        volatility  = st.number_input("Volatility (7-day)",       value=float(round(latest['volatility'], 5)), step=0.00001, format="%.5f")
        price_range = st.number_input("Price Range (High-Low)",   value=float(round(latest['price_range'],2)), step=0.01)

    st.markdown("---")

    if st.button("Predict Tomorrow's Movement", use_container_width=True):
        input_data  = np.array([[open_price, high_price, low_price,
                                  close_price, volume, ma7, ma21,
                                  ret, volatility, price_range]])
        prediction  = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        confidence  = max(probability) * 100

        st.markdown("---")
        if prediction == 1:
            st.success("## Stock Likely To Go UP Tomorrow!")
        else:
            st.error("## Stock Likely To Go DOWN Tomorrow!")

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("UP Probability",   f"{probability[1]*100:.1f}%")
        col_b.metric("DOWN Probability", f"{probability[0]*100:.1f}%")
        col_c.metric("Confidence",       f"{confidence:.1f}%")

        fig_gauge = go.Figure(go.Indicator(
            mode  = "gauge+number",
            value = probability[1] * 100,
            title = {'text': "UP Probability (%)"},
            gauge = {
                'axis':  {'range': [0, 100]},
                'bar':   {'color': "green" if prediction == 1 else "red"},
                'steps': [
                    {'range': [0,  40], 'color': "#ffcccc"},
                    {'range': [40, 60], 'color': "#ffffcc"},
                    {'range': [60,100], 'color': "#ccffcc"},
                ]
            }
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)
        st.info("This is an AI prediction for educational purposes only. Not financial advice.")

# ── PAGE 2: STOCK CHARTS ───────────────────────────────
elif page == "Stock Charts":
    st.header("AAPL Historical Stock Charts")
    total_rows = len(df)
    start_idx, end_idx = st.slider(
        "Select date range (row numbers)",
        0, total_rows - 1,
        (total_rows - 252, total_rows - 1)
    )
    df_filtered = df.iloc[start_idx:end_idx].copy()

    st.subheader("Candlestick Chart")
    fig_candle = go.Figure(data=[go.Candlestick(
        x=df_filtered['date'], open=df_filtered['open'],
        high=df_filtered['high'], low=df_filtered['low'],
        close=df_filtered['close'], name="AAPL"
    )])
    fig_candle.add_trace(go.Scatter(x=df_filtered['date'], y=df_filtered['MA7'],  name='MA7',  line=dict(color='orange', width=1)))
    fig_candle.add_trace(go.Scatter(x=df_filtered['date'], y=df_filtered['MA21'], name='MA21', line=dict(color='blue',   width=1)))
    fig_candle.update_layout(title="AAPL Candlestick with Moving Averages", height=500)
    st.plotly_chart(fig_candle, use_container_width=True)

    st.subheader("Trading Volume")
    fig_vol = px.bar(df_filtered, x='date', y='volume', title="Daily Trading Volume",
                     color='volume', color_continuous_scale='blues')
    st.plotly_chart(fig_vol, use_container_width=True)

    st.subheader("Daily Returns")
    fig_ret = px.line(df_filtered, x='date', y='return', title="Daily Return (%)")
    fig_ret.add_hline(y=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig_ret, use_container_width=True)

# ── PAGE 3: EDA ANALYSIS ──────────────────────────────
elif page == "EDA Analysis":
    st.header("Exploratory Data Analysis")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", len(df))
    col2.metric("Date Range", f"{df['date'].iloc[0][:4]}–{df['date'].iloc[-1][:4]}")
    col3.metric("Max Price",  f"${df['close'].max():.2f}")
    col4.metric("Min Price",  f"${df['close'].min():.2f}")

    st.dataframe(df[['date','open','high','low','close','volume']].tail(10))

    st.subheader("Close Price Distribution")
    fig_dist = px.histogram(df, x='close', nbins=50, title="Distribution of Closing Prices",
                            color_discrete_sequence=['steelblue'])
    st.plotly_chart(fig_dist, use_container_width=True)

    st.subheader("Feature Correlation Heatmap")
    corr_cols = ['open','high','low','close','volume','MA7','MA21','return','volatility','price_range']
    corr = df[corr_cols].corr()
    fig_heat = px.imshow(corr, text_auto=True, title="Feature Correlation Matrix",
                         color_continuous_scale='RdBu_r')
    st.plotly_chart(fig_heat, use_container_width=True)

    st.subheader("UP vs DOWN Days")
    target_counts = df['Target'].value_counts().reset_index()
    target_counts.columns = ['Movement', 'Count']
    target_counts['Movement'] = target_counts['Movement'].map({1:'UP', 0:'DOWN'})
    fig_pie = px.pie(target_counts, names='Movement', values='Count',
                     title="UP vs DOWN Days Distribution",
                     color_discrete_map={'UP':'green','DOWN':'red'})
    st.plotly_chart(fig_pie, use_container_width=True)

    st.subheader("Volatility Over Time")
    fig_vola = px.line(df, x='date', y='volatility', title="7-Day Rolling Volatility")
    st.plotly_chart(fig_vola, use_container_width=True)

# ── PAGE 4: ABOUT ─────────────────────────────────────
elif page == "About":
    st.header("About This Project")
    st.markdown("""
## Stock Market Price Movement Prediction System

### Objective
Predict whether Apple (AAPL) stock will go **UP or DOWN** the next trading day using Machine Learning.

### Dataset
- **Source:** S&P 500 Stock Data — Kaggle
- **Stock Used:** Apple Inc. (AAPL)
- **Records:** ~1,258 trading days (5 years)

### Technologies Used
| Tool | Purpose |
|------|---------|
| Python | Core programming |
| Pandas | Data processing |
| Scikit-learn | ML model (Random Forest) |
| Streamlit | Web application |
| Plotly | Interactive charts |
| Joblib | Model saving/loading |

### Model Details
- **Algorithm:** Random Forest Classifier
- **Trees:** 200
- **Max Depth:** 6
- **Class Weight:** Balanced (fixes prediction bias)

### Features Used
- Open, High, Low, Close Price
- Trading Volume
- 7-day Moving Average (MA7)
- 21-day Moving Average (MA21)
- Daily Return
- 7-day Volatility
- Price Range (High - Low)
    """)

st.markdown("---")
st.markdown("<center>Stock Market Prediction System | Built with Python & Streamlit | Educational Purposes Only</center>",
            unsafe_allow_html=True)