import yfinance as yf
import pandas as pd

# Test fetching data
symbol = "AAPL"
start_date = "2023-01-01"
end_date = "2024-10-08"

try:
    data = yf.download(symbol, start=start_date, end=end_date, interval="1d")
    if data.empty:
        print(f"No data found for {symbol}")
    else:
        print(f"Data fetched successfully:\n{data.head()}")
except Exception as e:
    print(f"Error: {e}")
    