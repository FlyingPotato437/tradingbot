import time
import uuid
import logging
from datetime import datetime
from typing import Dict, Optional, List
from dataclasses import dataclass
from enum import Enum
import yfinance as yf


class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


@dataclass
class Order:
    """Represents a trading order"""
    order_id: str
    symbol: str
    action: str  # 'BUY' or 'SELL'
    quantity: int
    order_type: OrderType
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    average_fill_price: float = 0.0
    timestamp: datetime = None
    fill_timestamp: Optional[datetime] = None
    commission: float = 1.0  # Simplified commission

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class TradeExecutor:
    """Handles trade execution with support for both paper trading and live trading"""
    
    def __init__(self, paper_trading: bool = True, broker_api=None):
        self.paper_trading = paper_trading
        self.broker_api = broker_api  # Would be actual broker API in production
        
        # Order tracking
        self.orders: Dict[str, Order] = {}
        self.filled_orders: List[Order] = []
        
        # Paper trading simulation
        self.slippage_pct = 0.001  # 0.1% slippage simulation
        self.commission = 1.0  # $1 per trade
        
        # Setup logging
        self.logger = logging.getLogger(__name__)

    def execute_trade(self, symbol: str, action: str, quantity: int, 
                     price: Optional[float] = None, order_type: OrderType = OrderType.MARKET) -> Optional[str]:
        """Execute a trade and return order ID"""
        try:
            # Generate order ID
            order_id = f"ORDER_{uuid.uuid4().hex[:8].upper()}"
            
            # Create order
            order = Order(
                order_id=order_id,
                symbol=symbol,
                action=action,
                quantity=quantity,
                order_type=order_type,
                price=price
            )
            
            # Store order
            self.orders[order_id] = order
            
            if self.paper_trading:
                # Execute paper trade immediately
                success = self._execute_paper_trade(order)
            else:
                # Execute real trade through broker API
                success = self._execute_real_trade(order)
            
            if success:
                self.logger.info(f"Order executed: {order_id} - {action} {quantity} {symbol}")
                return order_id
            else:
                self.logger.error(f"Order failed: {order_id}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")
            return None

    def _execute_paper_trade(self, order: Order) -> bool:
        """Execute a paper trade (simulation)"""
        try:
            # Get current market price
            current_price = self._get_current_price(order.symbol)
            if not current_price:
                order.status = OrderStatus.REJECTED
                return False
            
            # Apply slippage for market orders
            if order.order_type == OrderType.MARKET:
                if order.action == 'BUY':
                    fill_price = current_price * (1 + self.slippage_pct)
                else:  # SELL
                    fill_price = current_price * (1 - self.slippage_pct)
            else:
                fill_price = order.price or current_price
            
            # Fill the order
            order.status = OrderStatus.FILLED
            order.filled_quantity = order.quantity
            order.average_fill_price = fill_price
            order.fill_timestamp = datetime.now()
            
            # Add to filled orders
            self.filled_orders.append(order)
            
            self.logger.info(f"Paper trade filled: {order.symbol} {order.action} {order.quantity} @ ${fill_price:.2f}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error executing paper trade: {e}")
            order.status = OrderStatus.REJECTED
            return False

    def _execute_real_trade(self, order: Order) -> bool:
        """Execute a real trade through broker API"""
        # This would integrate with a real broker API like Alpaca, Interactive Brokers, etc.
        # For now, we'll just simulate it
        self.logger.warning("Real trading not implemented - using paper trading simulation")
        return self._execute_paper_trade(order)

    def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price for a symbol"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d", interval="1m")
            if not data.empty:
                return float(data['Close'].iloc[-1])
            return None
        except Exception as e:
            self.logger.error(f"Error getting price for {symbol}: {e}")
            return None

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order"""
        try:
            if order_id not in self.orders:
                return False
            
            order = self.orders[order_id]
            if order.status == OrderStatus.PENDING:
                order.status = OrderStatus.CANCELLED
                self.logger.info(f"Order cancelled: {order_id}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error cancelling order {order_id}: {e}")
            return False

    def get_order_status(self, order_id: str) -> Optional[Order]:
        """Get status of an order"""
        return self.orders.get(order_id)

    def get_pending_orders(self) -> List[Order]:
        """Get all pending orders"""
        return [order for order in self.orders.values() if order.status == OrderStatus.PENDING]

    def get_filled_orders(self, symbol: str = None) -> List[Order]:
        """Get filled orders, optionally filtered by symbol"""
        if symbol:
            return [order for order in self.filled_orders if order.symbol == symbol]
        return self.filled_orders.copy()

    def place_stop_loss_order(self, symbol: str, quantity: int, stop_price: float) -> Optional[str]:
        """Place a stop loss order"""
        return self.execute_trade(
            symbol=symbol,
            action='SELL',
            quantity=quantity,
            price=stop_price,
            order_type=OrderType.STOP
        )

    def place_take_profit_order(self, symbol: str, quantity: int, limit_price: float) -> Optional[str]:
        """Place a take profit order"""
        return self.execute_trade(
            symbol=symbol,
            action='SELL',
            quantity=quantity,
            price=limit_price,
            order_type=OrderType.LIMIT
        )

    def place_limit_order(self, symbol: str, action: str, quantity: int, limit_price: float) -> Optional[str]:
        """Place a limit order"""
        return self.execute_trade(
            symbol=symbol,
            action=action,
            quantity=quantity,
            price=limit_price,
            order_type=OrderType.LIMIT
        )

    def get_execution_stats(self) -> Dict:
        """Get execution statistics"""
        total_orders = len(self.orders)
        filled_orders = len([o for o in self.orders.values() if o.status == OrderStatus.FILLED])
        rejected_orders = len([o for o in self.orders.values() if o.status == OrderStatus.REJECTED])
        cancelled_orders = len([o for o in self.orders.values() if o.status == OrderStatus.CANCELLED])
        
        # Calculate average fill time for paper trades (always immediate)
        avg_fill_time = 0.0  # Immediate for paper trading
        
        # Calculate total commission paid
        total_commission = sum(o.commission for o in self.orders.values() if o.status == OrderStatus.FILLED)
        
        return {
            'total_orders': total_orders,
            'filled_orders': filled_orders,
            'rejected_orders': rejected_orders,
            'cancelled_orders': cancelled_orders,
            'fill_rate_pct': (filled_orders / total_orders * 100) if total_orders > 0 else 0,
            'avg_fill_time_seconds': avg_fill_time,
            'total_commission': total_commission,
            'paper_trading': self.paper_trading
        }

    def simulate_market_conditions(self, high_volatility: bool = False, poor_liquidity: bool = False):
        """Simulate different market conditions for paper trading"""
        if high_volatility:
            self.slippage_pct = 0.005  # 0.5% slippage in volatile markets
        elif poor_liquidity:
            self.slippage_pct = 0.003  # 0.3% slippage in illiquid markets
        else:
            self.slippage_pct = 0.001  # Normal 0.1% slippage

    def get_order_book(self, symbol: str) -> Dict:
        """Get simulated order book for a symbol (paper trading only)"""
        if not self.paper_trading:
            return {}
        
        try:
            current_price = self._get_current_price(symbol)
            if not current_price:
                return {}
            
            # Simulate order book
            bid_price = current_price * 0.999
            ask_price = current_price * 1.001
            
            return {
                'symbol': symbol,
                'bid': bid_price,
                'ask': ask_price,
                'spread': ask_price - bid_price,
                'spread_pct': ((ask_price - bid_price) / current_price) * 100,
                'last_price': current_price,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting order book for {symbol}: {e}")
            return {}

    def enable_real_trading(self, broker_api):
        """Enable real trading with broker API"""
        self.paper_trading = False
        self.broker_api = broker_api
        self.logger.warning("Real trading enabled - USE WITH CAUTION!")

    def enable_paper_trading(self):
        """Switch back to paper trading"""
        self.paper_trading = True
        self.broker_api = None
        self.logger.info("Switched to paper trading mode")

    def get_trading_summary(self) -> Dict:
        """Get comprehensive trading summary"""
        orders_by_symbol = {}
        for order in self.filled_orders:
            if order.symbol not in orders_by_symbol:
                orders_by_symbol[order.symbol] = {'buy_orders': 0, 'sell_orders': 0, 'total_volume': 0}
            
            if order.action == 'BUY':
                orders_by_symbol[order.symbol]['buy_orders'] += 1
            else:
                orders_by_symbol[order.symbol]['sell_orders'] += 1
            
            orders_by_symbol[order.symbol]['total_volume'] += order.filled_quantity
        
        return {
            'execution_stats': self.get_execution_stats(),
            'orders_by_symbol': orders_by_symbol,
            'most_traded_symbols': sorted(orders_by_symbol.keys(), 
                                        key=lambda x: orders_by_symbol[x]['total_volume'], 
                                        reverse=True)[:5],
            'trading_mode': 'Paper Trading' if self.paper_trading else 'Live Trading'
        }