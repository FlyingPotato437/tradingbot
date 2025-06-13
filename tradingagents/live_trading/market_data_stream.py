import threading
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
import yfinance as yf
import pandas as pd
from dataclasses import dataclass
from collections import deque


@dataclass
class MarketDataPoint:
    """Represents a single market data point"""
    symbol: str
    price: float
    volume: int
    timestamp: datetime
    bid: float = 0.0
    ask: float = 0.0
    high: float = 0.0
    low: float = 0.0
    open: float = 0.0


class LiveMarketDataStream:
    """Streams live market data for trading symbols"""
    
    def __init__(self, symbols: List[str], update_interval: int = 30):
        self.symbols = set(symbols)
        self.update_interval = update_interval  # seconds
        self.is_running = False
        
        # Data storage
        self.latest_data: Dict[str, MarketDataPoint] = {}
        self.historical_data: Dict[str, deque] = {symbol: deque(maxlen=1000) for symbol in symbols}
        
        # Threading
        self.data_thread = None
        self.data_lock = threading.Lock()
        
        # Callbacks
        self.data_callbacks: List[Callable] = []
        
        # Setup logging
        self.logger = logging.getLogger(__name__)

    def start(self):
        """Start the market data stream"""
        if self.is_running:
            self.logger.warning("Market data stream is already running")
            return
        
        self.logger.info("Starting market data stream...")
        self.is_running = True
        
        # Start data collection thread
        self.data_thread = threading.Thread(target=self._data_collection_loop, daemon=True)
        self.data_thread.start()
        
        self.logger.info(f"Market data stream started for {len(self.symbols)} symbols")

    def stop(self):
        """Stop the market data stream"""
        if not self.is_running:
            return
        
        self.logger.info("Stopping market data stream...")
        self.is_running = False
        
        # Wait for thread to finish
        if self.data_thread:
            self.data_thread.join(timeout=5)
        
        self.logger.info("Market data stream stopped")

    def _data_collection_loop(self):
        """Main data collection loop"""
        while self.is_running:
            try:
                self._fetch_latest_data()
                time.sleep(self.update_interval)
            except Exception as e:
                self.logger.error(f"Error in data collection loop: {e}")
                time.sleep(60)  # Wait before retrying on error

    def _fetch_latest_data(self):
        """Fetch latest market data for all symbols"""
        if not self.symbols:
            return
        
        try:
            # Create space-separated string of symbols for yfinance
            symbols_str = ' '.join(self.symbols)
            
            # Fetch data using yfinance
            tickers = yf.Tickers(symbols_str)
            
            with self.data_lock:
                for symbol in self.symbols:
                    try:
                        ticker = tickers.tickers[symbol]
                        
                        # Get latest price data
                        hist = ticker.history(period="1d", interval="1m")
                        if hist.empty:
                            continue
                        
                        latest_row = hist.iloc[-1]
                        
                        # Get additional info
                        info = ticker.info
                        
                        # Create market data point
                        data_point = MarketDataPoint(
                            symbol=symbol,
                            price=float(latest_row['Close']),
                            volume=int(latest_row['Volume']),
                            timestamp=datetime.now(),
                            high=float(latest_row['High']),
                            low=float(latest_row['Low']),
                            open=float(latest_row['Open']),
                            bid=info.get('bid', float(latest_row['Close']) * 0.999),
                            ask=info.get('ask', float(latest_row['Close']) * 1.001)
                        )
                        
                        # Update latest data
                        self.latest_data[symbol] = data_point
                        
                        # Add to historical data
                        if symbol not in self.historical_data:
                            self.historical_data[symbol] = deque(maxlen=1000)
                        self.historical_data[symbol].append(data_point)
                        
                        # Trigger callbacks
                        for callback in self.data_callbacks:
                            try:
                                callback(symbol, data_point)
                            except Exception as e:
                                self.logger.error(f"Error in data callback: {e}")
                        
                    except Exception as e:
                        self.logger.error(f"Error fetching data for {symbol}: {e}")
            
            self.logger.debug(f"Updated market data for {len(self.symbols)} symbols")
            
        except Exception as e:
            self.logger.error(f"Error fetching market data: {e}")

    def get_latest_data(self, symbol: str) -> Optional[MarketDataPoint]:
        """Get latest market data for a symbol"""
        with self.data_lock:
            return self.latest_data.get(symbol)

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol"""
        data = self.get_latest_data(symbol)
        return data.price if data else None

    def get_historical_data(self, symbol: str, minutes: int = 60) -> List[MarketDataPoint]:
        """Get historical data for a symbol"""
        with self.data_lock:
            if symbol not in self.historical_data:
                return []
            
            # Calculate cutoff time
            cutoff_time = datetime.now() - timedelta(minutes=minutes)
            
            # Filter data by time
            return [point for point in self.historical_data[symbol] 
                   if point.timestamp >= cutoff_time]

    def get_price_change(self, symbol: str, minutes: int = 60) -> Optional[Dict]:
        """Get price change over specified time period"""
        historical = self.get_historical_data(symbol, minutes)
        current = self.get_latest_data(symbol)
        
        if not historical or not current:
            return None
        
        old_price = historical[0].price
        current_price = current.price
        
        change = current_price - old_price
        change_pct = (change / old_price) * 100 if old_price > 0 else 0
        
        return {
            'symbol': symbol,
            'current_price': current_price,
            'old_price': old_price,
            'change': change,
            'change_pct': change_pct,
            'time_period_minutes': minutes
        }

    def get_market_summary(self) -> Dict:
        """Get summary of all market data"""
        with self.data_lock:
            summary = {
                'symbols_count': len(self.symbols),
                'last_update': max([data.timestamp for data in self.latest_data.values()]) if self.latest_data else None,
                'symbols': {}
            }
            
            for symbol, data in self.latest_data.items():
                change_1h = self.get_price_change(symbol, 60)
                
                summary['symbols'][symbol] = {
                    'price': data.price,
                    'volume': data.volume,
                    'bid': data.bid,
                    'ask': data.ask,
                    'spread': data.ask - data.bid,
                    'high': data.high,
                    'low': data.low,
                    'change_1h': change_1h['change_pct'] if change_1h else 0,
                    'last_update': data.timestamp
                }
            
            return summary

    def add_symbol(self, symbol: str):
        """Add a new symbol to track"""
        with self.data_lock:
            if symbol not in self.symbols:
                self.symbols.add(symbol)
                self.historical_data[symbol] = deque(maxlen=1000)
                self.logger.info(f"Added symbol {symbol} to market data stream")

    def remove_symbol(self, symbol: str):
        """Remove a symbol from tracking"""
        with self.data_lock:
            if symbol in self.symbols:
                self.symbols.remove(symbol)
                if symbol in self.latest_data:
                    del self.latest_data[symbol]
                if symbol in self.historical_data:
                    del self.historical_data[symbol]
                self.logger.info(f"Removed symbol {symbol} from market data stream")

    def add_data_callback(self, callback: Callable):
        """Add a callback function to be called when new data arrives"""
        self.data_callbacks.append(callback)

    def remove_data_callback(self, callback: Callable):
        """Remove a data callback"""
        if callback in self.data_callbacks:
            self.data_callbacks.remove(callback)

    def get_volatility(self, symbol: str, minutes: int = 60) -> Optional[float]:
        """Calculate volatility for a symbol over specified time period"""
        historical = self.get_historical_data(symbol, minutes)
        
        if len(historical) < 2:
            return None
        
        prices = [point.price for point in historical]
        returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
        
        if not returns:
            return None
        
        # Calculate standard deviation of returns
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        volatility = variance ** 0.5
        
        # Annualize volatility (approximate)
        trading_periods_per_year = 252 * 390  # 252 trading days, 390 minutes per day
        annualized_volatility = volatility * (trading_periods_per_year ** 0.5)
        
        return annualized_volatility

    def get_volume_profile(self, symbol: str, minutes: int = 60) -> Optional[Dict]:
        """Get volume profile for a symbol"""
        historical = self.get_historical_data(symbol, minutes)
        
        if not historical:
            return None
        
        total_volume = sum(point.volume for point in historical)
        avg_volume = total_volume / len(historical) if historical else 0
        
        # Calculate volume-weighted average price (VWAP)
        if total_volume > 0:
            vwap = sum(point.price * point.volume for point in historical) / total_volume
        else:
            vwap = 0
        
        return {
            'symbol': symbol,
            'total_volume': total_volume,
            'avg_volume': avg_volume,
            'vwap': vwap,
            'time_period_minutes': minutes,
            'data_points': len(historical)
        }

    def is_market_open(self) -> bool:
        """Check if the market is currently open (simplified)"""
        now = datetime.now()
        
        # Check if it's a weekday
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        # Check if it's during market hours (9:30 AM - 4:00 PM ET)
        market_start = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_end = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        return market_start <= now <= market_end

    def get_connection_status(self) -> Dict:
        """Get status of the market data connection"""
        last_update_times = [data.timestamp for data in self.latest_data.values()]
        
        if last_update_times:
            last_update = max(last_update_times)
            time_since_update = (datetime.now() - last_update).total_seconds()
        else:
            last_update = None
            time_since_update = float('inf')
        
        # Consider connection healthy if updated within 2x the update interval
        is_healthy = time_since_update < (self.update_interval * 2)
        
        return {
            'is_running': self.is_running,
            'is_healthy': is_healthy,
            'symbols_tracked': len(self.symbols),
            'symbols_with_data': len(self.latest_data),
            'last_update': last_update,
            'seconds_since_update': time_since_update,
            'update_interval': self.update_interval,
            'market_open': self.is_market_open()
        }