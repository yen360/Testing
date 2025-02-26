# Install required libraries if needed:
# pip install yfinance plotly pandas numpy

import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -------------------------------
# 1. Data Fetching using Yahoo Finance
# -------------------------------
ticker = 'AAPL'
start_date = '2018-01-01'
end_date = '2023-01-01'

# Download historical data
data = yf.download(ticker, start=start_date, end=end_date)
data.reset_index(inplace=True)  # Ensure Date is a column for Plotly

# -------------------------------
# 2. Indicator Calculations
# -------------------------------

# --- a. Moving Average Crossover (SMA 50 & SMA 200) ---
short_window = 50
long_window  = 200
data['SMA_short'] = data['Close'].rolling(window=short_window).mean()
data['SMA_long']  = data['Close'].rolling(window=long_window).mean()

# Generate crossover signals (1 when short > long; diff = +1 means buy, -1 means sell)
data['MA_signal'] = np.where(data['SMA_short'] > data['SMA_long'], 1, 0)
data['MA_cross']  = data['MA_signal'].diff()
buy_signals_ma  = data.loc[data['MA_cross'] == 1]
sell_signals_ma = data.loc[data['MA_cross'] == -1]

# --- b. Relative Strength Index (RSI) ---
rsi_period = 14
delta = data['Close'].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
avg_gain = gain.rolling(window=rsi_period).mean()
avg_loss = loss.rolling(window=rsi_period).mean()
rs = avg_gain / avg_loss
data['RSI'] = 100 - (100 / (1 + rs))
# (RSI signals can be further processed as needed, e.g., crossing above 30 or below 70)

# --- c. MACD (12-day EMA, 26-day EMA, and Signal line 9) ---
exp12 = data['Close'].ewm(span=12, adjust=False).mean()
exp26 = data['Close'].ewm(span=26, adjust=False).mean()
data['MACD'] = exp12 - exp26
data['MACD_signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
data['MACD_cross'] = np.where(data['MACD'] > data['MACD_signal'], 1, 0)
data['MACD_cross'] = pd.Series(data['MACD_cross']).diff()
buy_signals_macd  = data.loc[data['MACD_cross'] == 1]
sell_signals_macd = data.loc[data['MACD_cross'] == -1]

# --- d. Bollinger Bands (20-day) ---
bb_period = 20
data['BB_mid']   = data['Close'].rolling(bb_period).mean()
data['BB_std']   = data['Close'].rolling(bb_period).std()
data['BB_upper'] = data['BB_mid'] + 2 * data['BB_std']
data['BB_lower'] = data['BB_mid'] - 2 * data['BB_std']
# Identify reversal signals: price moving from outside back inside the band
buy_signals_bb, sell_signals_bb = [], []
for i in range(1, len(data)):
    if data['Close'].iloc[i-1] < data['BB_lower'].iloc[i-1] and data['Close'].iloc[i] > data['BB_lower'].iloc[i]:
        buy_signals_bb.append(data.iloc[i])
    if data['Close'].iloc[i-1] > data['BB_upper'].iloc[i-1] and data['Close'].iloc[i] < data['BB_upper'].iloc[i]:
        sell_signals_bb.append(data.iloc[i])
buy_signals_bb = pd.DataFrame(buy_signals_bb)
sell_signals_bb = pd.DataFrame(sell_signals_bb)

# --- e. Stochastic Oscillator (14 period %K, 3 period %D) ---
data['Lowest14'] = data['Low'].rolling(window=14).min()
data['Highest14'] = data['High'].rolling(window=14).max()
data['%K'] = (data['Close'] - data['Lowest14']) / (data['Highest14'] - data['Lowest14']) * 100
data['%D'] = data['%K'].rolling(window=3).mean()
buy_signals_stoch, sell_signals_stoch = [], []
for i in range(1, len(data)):
    if (data['%K'].iloc[i] > data['%D'].iloc[i] and 
        data['%K'].iloc[i-1] <= data['%D'].iloc[i-1] and 
        data['%K'].iloc[i-1] < 20):
        buy_signals_stoch.append(data.iloc[i])
    if (data['%K'].iloc[i] < data['%D'].iloc[i] and 
        data['%K'].iloc[i-1] >= data['%D'].iloc[i-1] and 
        data['%K'].iloc[i-1] > 80):
        sell_signals_stoch.append(data.iloc[i])
buy_signals_stoch = pd.DataFrame(buy_signals_stoch)
sell_signals_stoch = pd.DataFrame(sell_signals_stoch)

# --- f. Volume Spike Analysis ---
vol_window = 20
data['Vol_avg'] = data['Volume'].rolling(window=vol_window).mean()
data['Vol_std'] = data['Volume'].rolling(window=vol_window).std()
data['Vol_spike'] = data['Volume'] > (data['Vol_avg'] + 2 * data['Vol_std'])
# Signal: if there's a volume spike and the price increased from previous day, consider a buy signal;
# if price decreased, consider a sell signal.
buy_signals_vol = data[(data['Vol_spike']) & (data['Close'].diff() > 0)]
sell_signals_vol = data[(data['Vol_spike']) & (data['Close'].diff() < 0)]

# --- g. Trend Detection ---
# Here, we use the slope of the short-term SMA (50-day) over the last 20 days as a simple trend indicator.
def calculate_trend(sma_series, window=20):
    if len(sma_series.dropna()) < window:
        return None
    recent = sma_series.dropna().iloc[-window:]
    x = np.arange(window)
    slope = np.polyfit(x, recent, 1)[0]
    return slope

trend_slope = calculate_trend(data['SMA_short'], window=20)
if trend_slope is None:
    trend_direction = "Not enough data"
else:
    trend_direction = "Uptrend" if trend_slope > 0 else "Downtrend" if trend_slope < 0 else "Sideways"

# -------------------------------
# 3. Plotting with Plotly
# -------------------------------
# We'll create a two-row subplot: the upper for price (candlestick + signals) and lower for volume bars.

fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                    vertical_spacing=0.02,
                    row_heights=[0.7, 0.3],
                    subplot_titles=(f"{ticker} Price with Trading Signals and Trend Detection", "Volume"))

# --- a. Candlestick Chart ---
fig.add_trace(go.Candlestick(x=data['Date'],
                             open=data['Open'],
                             high=data['High'],
                             low=data['Low'],
                             close=data['Close'],
                             name="Candlestick"), row=1, col=1)

# --- b. Overlay Moving Averages ---
fig.add_trace(go.Scatter(x=data['Date'], y=data['SMA_short'], mode='lines',
                         name=f"SMA {short_window}", line=dict(color='blue', width=1)),
              row=1, col=1)
fig.add_trace(go.Scatter(x=data['Date'], y=data['SMA_long'], mode='lines',
                         name=f"SMA {long_window}", line=dict(color='orange', width=1)),
              row=1, col=1)

# --- c. Overlay Bollinger Bands ---
fig.add_trace(go.Scatter(x=data['Date'], y=data['BB_upper'], mode='lines',
                         name="BB Upper", line=dict(color='grey', dash='dot')),
              row=1, col=1)
fig.add_trace(go.Scatter(x=data['Date'], y=data['BB_lower'], mode='lines',
                         name="BB Lower", line=dict(color='grey', dash='dot')),
              row=1, col=1)

# --- d. Helper function to add signal markers ---
def add_signals(df, marker_symbol, color, name):
    if df.empty:
        return
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['Close'],
        mode='markers',
        marker_symbol=marker_symbol,
        marker=dict(color=color, size=10),
        name=name,
        hovertemplate='%{x}<br>Price: %{y:.2f}'
    ), row=1, col=1)

# Add signals from various strategies:
add_signals(buy_signals_ma, marker_symbol="triangle-up", color="green", name="MA Buy")
add_signals(sell_signals_ma, marker_symbol="triangle-down", color="red", name="MA Sell")
add_signals(buy_signals_macd, marker_symbol="circle", color="darkgreen", name="MACD Buy")
add_signals(sell_signals_macd, marker_symbol="circle", color="darkred", name="MACD Sell")
add_signals(buy_signals_bb, marker_symbol="diamond", color="lime", name="BB Buy")
add_signals(sell_signals_bb, marker_symbol="diamond", color="magenta", name="BB Sell")
add_signals(buy_signals_stoch, marker_symbol="star", color="cyan", name="Stoch Buy")
add_signals(sell_signals_stoch, marker_symbol="star", color="purple", name="Stoch Sell")
add_signals(buy_signals_vol, marker_symbol="arrow-up", color="forestgreen", name="Volume Buy")
add_signals(sell_signals_vol, marker_symbol="arrow-down", color="firebrick", name="Volume Sell")

# --- e. Volume Bar Chart ---
fig.add_trace(go.Bar(x=data['Date'], y=data['Volume'], name="Volume", marker_color='lightblue'),
              row=2, col=1)

# --- f. Add Trend Annotation ---
fig.add_annotation(
    x=data['Date'].iloc[-1],
    y=data['Close'].iloc[-1],
    text=f"Trend: {trend_direction}",
    showarrow=True,
    arrowhead=1,
    ax=-40, ay=-40,
    font=dict(color="black", size=12),
    row=1, col=1
)

# --- g. Final Layout Adjustments ---
fig.update_layout(title=f"{ticker} Trading Signals with Volume Analysis and Trend Detection",
                  xaxis_rangeslider_visible=False,
                  hovermode="x unified",
                  template="plotly_white")
fig.update_xaxes(title_text="Date", row=2, col=1)
fig.update_yaxes(title_text="Price", row=1, col=1)
fig.update_yaxes(title_text="Volume", row=2, col=1)

fig.show()
