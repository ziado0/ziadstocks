import streamlit as st
import pandas as pd
import numpy as np
import datetime
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import joblib
import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import gdown
import shutil

# Set page configuration
st.set_page_config(
    page_title="Saudi Stock Market Predictor",
    page_icon="📈",
    layout="wide"
)

# Google Drive folder ID
DRIVE_FOLDER_ID = "1hmaj_v6rnsYEkm8HFF4K6_D7kwAkh7nr"

# Path to models directory
MODELS_DIR = "models"

# Function to download models from Google Drive folder
@st.cache_resource
def download_models():
    """Download models from Google Drive if not already downloaded"""
    if not os.path.exists(MODELS_DIR) or len(os.listdir(MODELS_DIR)) == 0:
        with st.spinner("Downloading models from Google Drive... This may take a minute."):
            os.makedirs(MODELS_DIR, exist_ok=True)
            
            # Create a temporary directory for downloads
            temp_dir = "temp_downloads"
            os.makedirs(temp_dir, exist_ok=True)
            
            # Download all files from the Google Drive folder
            folder_url = f"https://drive.google.com/drive/folders/{DRIVE_FOLDER_ID}"
            gdown.download_folder(url=folder_url, output=temp_dir, quiet=False) 
            
            # Move model files to the models directory
            for root, dirs, files in os.walk(temp_dir):
                for dir_name in dirs:
                    if dir_name.endswith("_SR"):  # Model directories like 2222_SR
                        src_dir = os.path.join(root, dir_name)
                        dst_dir = os.path.join(MODELS_DIR, dir_name)
                        if os.path.exists(src_dir) and not os.path.exists(dst_dir):
                            shutil.copytree(src_dir, dst_dir)
            
            # Clean up temporary directory
            shutil.rmtree(temp_dir)
            
            st.success("Models downloaded successfully!")
    return True

# Rest of the code remains the same as in my previous message
# Function to get all available models
def get_available_models():
    """Get all available trained models"""
    # Ensure models are downloaded
    download_models()
    
    models = []
    for item in os.listdir(MODELS_DIR):
        if os.path.isdir(os.path.join(MODELS_DIR, item)) and os.path.exists(os.path.join(MODELS_DIR, item, 'lstm_model.h5')):
            symbol = item.replace('_', '.')
            models.append(symbol)
    return sorted(models)

# Function to get company name from symbol
def get_company_name(symbol):
    """Get company name from symbol using Yahoo Finance"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return info.get('longName', symbol)
    except:
        return symbol

# Function to get real stock data from Yahoo Finance
@st.cache_data(ttl=24*3600)  # Cache for 24 hours
def get_stock_data(symbol, period="1y"):
    """Get real stock data from Yahoo Finance"""
    try:
        data = yf.download(symbol, period=period)
        if data.empty:
            st.error(f"No data available for {symbol}")
            return None
        return data
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return None

# Function to make predictions using LSTM model
def predict_with_lstm(model, scaler, data, time_steps=60, future_days=30):
    """Make predictions using LSTM model"""
    # Extract close price
    df = data['Close'].values.reshape(-1, 1)
    
    # Scale the data
    df_scaled = scaler.transform(df)
    
    # Use last time_steps days for prediction
    X_predict = df_scaled[-time_steps:].reshape(1, time_steps, 1)
    
    # Make predictions for future days
    predictions = []
    current_batch = X_predict.reshape((1, time_steps, 1))
    
    for i in range(future_days):
        current_pred = model.predict(current_batch, verbose=0)[0]
        predictions.append(current_pred)
        current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)
    
    # Inverse transform to get actual prices
    predictions = np.array(predictions).reshape(-1, 1)
    predictions = scaler.inverse_transform(predictions)
    
    # Create future dates
    last_date = data.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days, freq='B')
    
    # Create DataFrame with predictions
    future_df = pd.DataFrame(index=future_dates, data=predictions, columns=['Predicted_Close'])
    
    return future_df

# Function to load model and make predictions
def predict_stock(symbol, data):
    """Load model and make predictions for different timeframes"""
    if data is None or len(data) < 60:
        st.error(f"Not enough data for {symbol} to make predictions")
        return None
    
    # Load model and scaler
    model_dir = os.path.join(MODELS_DIR, symbol.replace('.', '_'))
    model_path = os.path.join(model_dir, 'lstm_model.h5')
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        st.error(f"Model or scaler not found for {symbol}")
        return None
    
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    
    # Make predictions for different timeframes
    predictions = {}
    timeframes = [1, 5, 10, 20]  # days - next day, 5 days, 10 days, 1 month (20 trading days)
    
    # Get future predictions
    future_df = predict_with_lstm(model, scaler, data, future_days=max(timeframes))
    
    # Extract predictions for each timeframe
    latest_price = data['Close'].iloc[-1]
    
    for days in timeframes:
        predicted_price = future_df['Predicted_Close'].iloc[days-1]
        predicted_change = ((predicted_price - latest_price) / latest_price) * 100
        direction = "Up" if predicted_price > latest_price else "Down"
        
        predictions[days] = {
            'predicted_price': round(float(predicted_price), 2),
            'predicted_change': round(float(predicted_change), 2),
            'direction': direction
        }
    
    return predictions, future_df

# Function to create interactive chart
def create_interactive_chart(historical_data, future_data=None):
    """Create interactive chart with historical and predicted data"""
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True)
    
    # Add historical data
    fig.add_trace(
        go.Scatter(
            x=historical_data.index,
            y=historical_data['Close'],
            mode='lines',
            name='Historical',
            line=dict(color='blue')
        )
    )
    
    # Add future predictions if available
    if future_data is not None:
        fig.add_trace(
            go.Scatter(
                x=future_data.index,
                y=future_data['Predicted_Close'],
                mode='lines',
                name='Predicted',
                line=dict(color='red', dash='dash')
            )
        )
    
    # Update layout
    fig.update_layout(
        title='Stock Price History and Prediction',
        xaxis_title='Date',
        yaxis_title='Price (SAR)',
        legend=dict(x=0, y=1, traceorder='normal'),
        height=500
    )
    
    return fig

# Main app
def main():
    # Header
    st.title("Saudi Stock Market Predictor")
    
    # Get available models
    available_models = get_available_models()
    
    if not available_models:
        st.error("No trained models found. Please check the Google Drive link.")
        return
    
    # Sidebar
    st.sidebar.title("Options")
    
    # Stock selection
    selected_symbol = st.sidebar.selectbox(
        "Select a stock",
        options=available_models,
        format_func=lambda x: f"{x} - {get_company_name(x)}"
    )
    
    # Historical data range selection (this is just for input data)
    st.sidebar.subheader("Historical Data Range")
    historical_range = st.sidebar.radio(
        "Select historical data range to analyze:",
        options=["3 Months", "6 Months", "1 Year", "2 Years"],
        index=2  # Default to 1 Year
    )
    
    # Map selection to period for Yahoo Finance
    period_map = {
        "3 Months": "3mo",
        "6 Months": "6mo",
        "1 Year": "1y",
        "2 Years": "2y"
    }
    
    # About section in sidebar
    st.sidebar.markdown("---")
    st.sidebar.title("About")
    st.sidebar.info(
        "This app predicts Saudi stock market movements using "
        "LSTM neural networks trained on historical data. "
        "The predictions are provided for next day, 5 days, 10 days, and 1 month timeframes."
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("© 2025 ZiadStocks")
    
    # Main content
    if st.sidebar.button("Load Stock Data & Generate Predictions"):
        with st.spinner(f"Loading data for {selected_symbol}..."):
            # Get real data from Yahoo Finance
            data = get_stock_data(selected_symbol, period=period_map[historical_range])
            
            if data is not None and not data.empty:
                # Display stock info
                st.header(f"{get_company_name(selected_symbol)} ({selected_symbol})")
                
                # Display current price and date
                latest_price = data['Close'].iloc[-1]
                latest_date = data.index[-1].strftime('%Y-%m-%d')
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Latest Price", f"{latest_price:.2f} SAR")
                with col2:
                    st.metric("As of", latest_date)
                
                # Make predictions
                with st.spinner("Generating predictions..."):
                    predictions_result = predict_stock(selected_symbol, data)
                    
                    if predictions_result:
                        predictions, future_df = predictions_result
                        
                        # Display interactive chart
                        st.subheader("Price History and Prediction")
                        chart = create_interactive_chart(data, future_df)
                        st.plotly_chart(chart, use_container_width=True)
                        
                        # Highlight the prediction timeframes
                        st.header("Price Predictions")
                        st.markdown("### These are the exact timeframes you requested:")
                        
                        # Display predictions in columns
                        cols = st.columns(4)
                        
                        timeframes = [1, 5, 10, 20]
                        timeframe_names = ['Next Day', '5 Days', '10 Days', '1 Month']
                        
                        for i, (days, name) in enumerate(zip(timeframes, timeframe_names)):
                            pred = predictions[days]
                            
                            with cols[i]:
                                st.subheader(name)
                                
                                # Predicted price
                                st.markdown(f"**Price:** {pred['predicted_price']:.2f} SAR")
                                
                                # Predicted change
                                change_color = "green" if pred["direction"] == "Up" else "red"
                                change_sign = "+" if pred["direction"] == "Up" else ""
                                st.markdown(f"**Change:** <span style='color:{change_color}'>{change_sign}{pred['predicted_change']:.2f}%</span>", unsafe_allow_html=True)
                                
                                # Direction
                                direction_icon = "↑" if pred["direction"] == "Up" else "↓"
                                st.markdown(f"**Trend:** <span style='color:{change_color}'>{direction_icon} {pred['direction']}</span>", unsafe_allow_html=True)
                        
                        # Disclaimer
                        st.markdown("---")
                        st.caption(
                            "Disclaimer: The predictions provided by this tool are based on LSTM neural networks trained on historical data. "
                            "Financial markets are inherently unpredictable, and these forecasts should not be used as the sole basis "
                            "for investment decisions. Always conduct your own research and consider consulting with a financial advisor "
                            "before making investment choices."
                        )
            else:
                st.error(f"No data available for {selected_symbol}")
    else:
        # Welcome message
        st.markdown("""
        ## Welcome to the Saudi Stock Market Predictor!
        
        This application predicts stock movements in the Saudi market (Tadawul) using LSTM neural networks.
        
        ### Features:
        - Real-time data from Yahoo Finance
        - LSTM neural network predictions
        - Predictions for exactly the timeframes you requested:
          - Next day
          - 5 days
          - 10 days
          - 1 month
        - Interactive price charts
        - Pre-trained models for all Saudi stocks
        
        ### How to use:
        1. Select a stock from the dropdown menu in the sidebar
        2. Choose a historical data range to analyze (this is just input data for the model)
        3. Click "Load Stock Data & Generate Predictions" to see predictions for all four timeframes
        """)
        
        # Simple info message
        st.info("Select a stock from the sidebar and click the button to see predictions for next day, 5 days, 10 days, and 1 month.")

if __name__ == "__main__":
    main()
