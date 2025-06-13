#!/usr/bin/env python3
"""
Download test data for backtesting
"""

import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta
from rich.console import Console

console = Console()

def download_stock_data(symbol, start_date='2022-01-01', end_date='2024-12-31'):
    """Download stock data using yfinance"""
    try:
        console.print(f"[INFO] Downloading {symbol} data from {start_date} to {end_date}...", style="yellow")
        
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date)
        
        if data.empty:
            console.print(f"[ERROR] No data downloaded for {symbol}", style="red")
            return None
        
        # Reset index to make Date a column
        data = data.reset_index()
        
        # Ensure we have the required columns
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            if col not in data.columns:
                console.print(f"[ERROR] Missing column {col} in {symbol} data", style="red")
                return None
        
        console.print(f"[OK] Downloaded {len(data)} days of {symbol} data", style="green")
        return data
        
    except Exception as e:
        console.print(f"[ERROR] Failed to download {symbol}: {e}", style="red")
        return None

def main():
    """Download test data for backtesting"""
    console.print("=" * 50)
    console.print("DOWNLOADING TEST DATA", style="bold blue")
    console.print("=" * 50)
    
    symbols = ['AAPL', 'NVDA', 'MSFT', 'GOOGL', 'TSLA']
    
    # Create data directory
    os.makedirs('test_data', exist_ok=True)
    
    for symbol in symbols:
        data = download_stock_data(symbol)
        
        if data is not None:
            # Save to CSV
            filename = f"test_data/{symbol}_data.csv"
            data.to_csv(filename, index=False)
            console.print(f"[SAVED] {symbol} data saved to {filename}", style="green")
        else:
            console.print(f"[FAILED] Could not download {symbol} data", style="red")
    
    console.print("\n[COMPLETE] Data download finished!", style="bold blue")

if __name__ == '__main__':
    main()