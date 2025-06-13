import logging
import time
from datetime import datetime
from typing import Dict, Optional, List
from dataclasses import dataclass
from enum import Enum
import os

try:
    import alpaca_trade_api as tradeapi
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    # Create dummy tradeapi for compatibility
    class DummyAPI:
        pass
    tradeapi = DummyAPI()

from .trade_executor import OrderStatus, OrderType, Order


class AlpacaExecutor:
    """Alpaca-specific trade executor for live stock trading"""
    
    def __init__(self, paper_trading: bool = True):
        self.paper_trading = paper_trading
        self.api = None
        self.logger = logging.getLogger(__name__)
        
        # Order tracking
        self.orders: Dict[str, Order] = {}
        self.filled_orders: List[Order] = []
        
        # Initialize Alpaca API
        self._initialize_alpaca()

    def _initialize_alpaca(self):
        """Initialize Alpaca API connection"""
        if not ALPACA_AVAILABLE:
            self.logger.error("Alpaca Trade API not installed. Run: pip install alpaca-trade-api")
            return False
        
        try:
            # Get API credentials from environment
            api_key = os.getenv('ALPACA_API_KEY')
            api_secret = os.getenv('ALPACA_SECRET_KEY')
            
            if not api_key or not api_secret:
                self.logger.error("Alpaca API credentials not found in environment variables")
                self.logger.error("Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables")
                return False
            
            # Set base URL based on paper trading
            if self.paper_trading:
                base_url = "https://paper-api.alpaca.markets"
            else:
                base_url = "https://api.alpaca.markets"
            
            # Initialize API
            self.api = tradeapi.REST(
                api_key,
                api_secret,
                base_url,
                api_version='v2'
            )
            
            # Test connection
            account = self.api.get_account()
            self.logger.info(f"Alpaca API connected - Account status: {account.status}")
            
            if account.trading_blocked:
                self.logger.warning("Trading is blocked on this account")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Alpaca API: {e}")
            return False

    def execute_trade(self, symbol: str, action: str, quantity: int, 
                     price: Optional[float] = None, order_type: OrderType = OrderType.MARKET) -> Optional[str]:
        """Execute a trade through Alpaca"""
        
        if not self.api:
            self.logger.error("Alpaca API not initialized")
            return None
        
        try:
            # Validate inputs
            if action not in ['BUY', 'SELL']:
                self.logger.error(f"Invalid action: {action}")
                return None
            
            if quantity <= 0:
                self.logger.error(f"Invalid quantity: {quantity}")
                return None
            
            # Convert to Alpaca order parameters
            side = 'buy' if action == 'BUY' else 'sell'
            
            if order_type == OrderType.MARKET:
                order_class = 'simple'
                time_in_force = 'day'
                order_params = {
                    'symbol': symbol,
                    'qty': quantity,
                    'side': side,
                    'type': 'market',
                    'time_in_force': time_in_force
                }
            elif order_type == OrderType.LIMIT:
                if not price:
                    self.logger.error("Limit price required for limit orders")
                    return None
                order_params = {
                    'symbol': symbol,
                    'qty': quantity,
                    'side': side,
                    'type': 'limit',
                    'limit_price': price,
                    'time_in_force': 'day'
                }
            else:
                self.logger.error(f"Unsupported order type: {order_type}")
                return None
            
            # Submit order to Alpaca
            self.logger.info(f"Submitting {action} order: {quantity} shares of {symbol}")
            alpaca_order = self.api.submit_order(**order_params)
            
            # Create our order object
            order = Order(
                order_id=alpaca_order.id,
                symbol=symbol,
                action=action,
                quantity=quantity,
                order_type=order_type,
                price=price,
                status=OrderStatus.PENDING
            )
            
            # Store order
            self.orders[alpaca_order.id] = order
            
            self.logger.info(f"Order submitted successfully: {alpaca_order.id}")
            return alpaca_order.id
            
        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")
            return None

    def get_order_status(self, order_id: str) -> Optional[Order]:
        """Get current status of an order"""
        if not self.api:
            return None
        
        try:
            # Get order from Alpaca
            alpaca_order = self.api.get_order(order_id)
            
            if order_id in self.orders:
                order = self.orders[order_id]
                
                # Update status based on Alpaca status
                if alpaca_order.status == 'filled':
                    order.status = OrderStatus.FILLED
                    order.filled_quantity = int(alpaca_order.filled_qty)
                    order.average_fill_price = float(alpaca_order.filled_avg_price or 0)
                    order.fill_timestamp = datetime.now()
                    
                    # Move to filled orders if not already there
                    if order not in self.filled_orders:
                        self.filled_orders.append(order)
                        
                elif alpaca_order.status == 'partially_filled':
                    order.status = OrderStatus.PARTIAL
                    order.filled_quantity = int(alpaca_order.filled_qty)
                    order.average_fill_price = float(alpaca_order.filled_avg_price or 0)
                    
                elif alpaca_order.status == 'canceled':
                    order.status = OrderStatus.CANCELLED
                    
                elif alpaca_order.status == 'rejected':
                    order.status = OrderStatus.REJECTED
                
                return order
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting order status: {e}")
            return None

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order"""
        if not self.api:
            return False
        
        try:
            self.api.cancel_order(order_id)
            
            if order_id in self.orders:
                self.orders[order_id].status = OrderStatus.CANCELLED
            
            self.logger.info(f"Order cancelled: {order_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error cancelling order: {e}")
            return False

    def get_pending_orders(self) -> List[Order]:
        """Get all pending orders"""
        return [order for order in self.orders.values() if order.status == OrderStatus.PENDING]

    def get_filled_orders(self, symbol: str = None) -> List[Order]:
        """Get filled orders, optionally filtered by symbol"""
        if symbol:
            return [order for order in self.filled_orders if order.symbol == symbol]
        return self.filled_orders.copy()

    def get_account_info(self) -> Optional[Dict]:
        """Get account information"""
        if not self.api:
            return None
        
        try:
            account = self.api.get_account()
            positions = self.api.list_positions()
            
            return {
                'account_number': account.account_number,
                'status': account.status,
                'trading_blocked': account.trading_blocked,
                'equity': float(account.equity),
                'cash': float(account.cash),
                'buying_power': float(account.buying_power),
                'portfolio_value': float(account.portfolio_value),
                'positions_count': len(positions),
                'day_trade_count': account.day_trade_count,
                'pattern_day_trader': account.pattern_day_trader
            }
            
        except Exception as e:
            self.logger.error(f"Error getting account info: {e}")
            return None

    def get_positions(self) -> List[Dict]:
        """Get current positions"""
        if not self.api:
            return []
        
        try:
            positions = self.api.list_positions()
            
            result = []
            for pos in positions:
                result.append({
                    'symbol': pos.symbol,
                    'quantity': int(pos.qty),
                    'side': pos.side,
                    'market_value': float(pos.market_value),
                    'cost_basis': float(pos.cost_basis),
                    'unrealized_pnl': float(pos.unrealized_pnl),
                    'unrealized_pnl_pct': float(pos.unrealized_plpc) * 100,
                    'avg_entry_price': float(pos.avg_entry_price),
                    'current_price': float(pos.current_price or 0)
                })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")
            return []

    def get_buying_power(self) -> float:
        """Get current buying power"""
        if not self.api:
            return 0.0
        
        try:
            account = self.api.get_account()
            return float(account.buying_power)
        except Exception as e:
            self.logger.error(f"Error getting buying power: {e}")
            return 0.0

    def is_market_open(self) -> bool:
        """Check if market is currently open"""
        if not self.api:
            return False
        
        try:
            clock = self.api.get_clock()
            return clock.is_open
        except Exception as e:
            self.logger.error(f"Error checking market status: {e}")
            return False

    def get_market_hours(self) -> Optional[Dict]:
        """Get market hours for today"""
        if not self.api:
            return None
        
        try:
            clock = self.api.get_clock()
            calendar = self.api.get_calendar(start=clock.timestamp.date(), end=clock.timestamp.date())
            
            if calendar:
                today = calendar[0]
                return {
                    'is_open': clock.is_open,
                    'next_open': today.open,
                    'next_close': today.close,
                    'current_time': clock.timestamp
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting market hours: {e}")
            return None

    def place_bracket_order(self, symbol: str, quantity: int, entry_price: float,
                           stop_loss: float, take_profit: float) -> Optional[str]:
        """Place a bracket order with stop loss and take profit"""
        if not self.api:
            return None
        
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=quantity,
                side='buy',
                type='limit',
                limit_price=entry_price,
                time_in_force='day',
                order_class='bracket',
                stop_loss={'stop_price': stop_loss},
                take_profit={'limit_price': take_profit}
            )
            
            self.logger.info(f"Bracket order submitted: {order.id}")
            return order.id
            
        except Exception as e:
            self.logger.error(f"Error placing bracket order: {e}")
            return None

    def get_historical_bars(self, symbol: str, timeframe: str = '1Day', limit: int = 100) -> Optional[List[Dict]]:
        """Get historical price bars"""
        if not self.api:
            return None
        
        try:
            bars = self.api.get_bars(symbol, timeframe, limit=limit)
            
            result = []
            for bar in bars:
                result.append({
                    'timestamp': bar.t,
                    'open': float(bar.o),
                    'high': float(bar.h),
                    'low': float(bar.l),
                    'close': float(bar.c),
                    'volume': int(bar.v)
                })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting historical bars: {e}")
            return None

    def validate_symbol(self, symbol: str) -> bool:
        """Validate if symbol is tradeable"""
        if not self.api:
            return False
        
        try:
            asset = self.api.get_asset(symbol)
            return asset.tradable and asset.status == 'active'
        except Exception as e:
            self.logger.error(f"Symbol validation failed for {symbol}: {e}")
            return False

    def get_execution_stats(self) -> Dict:
        """Get execution statistics"""
        total_orders = len(self.orders)
        filled_orders = len([o for o in self.orders.values() if o.status == OrderStatus.FILLED])
        rejected_orders = len([o for o in self.orders.values() if o.status == OrderStatus.REJECTED])
        cancelled_orders = len([o for o in self.orders.values() if o.status == OrderStatus.CANCELLED])
        
        return {
            'total_orders': total_orders,
            'filled_orders': filled_orders,
            'rejected_orders': rejected_orders,
            'cancelled_orders': cancelled_orders,
            'fill_rate_pct': (filled_orders / total_orders * 100) if total_orders > 0 else 0,
            'broker': 'Alpaca',
            'paper_trading': self.paper_trading,
            'api_connected': self.api is not None
        }