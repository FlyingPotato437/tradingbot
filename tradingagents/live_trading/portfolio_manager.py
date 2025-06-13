import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging


@dataclass
class Position:
    """Represents a stock position"""
    symbol: str
    quantity: int
    avg_cost: float
    current_price: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)

    def update_price(self, price: float):
        """Update current price and calculate market value"""
        self.current_price = price
        self.market_value = self.quantity * price
        self.unrealized_pnl = (price - self.avg_cost) * self.quantity
        self.last_updated = datetime.now()


@dataclass
class Trade:
    """Represents a completed trade"""
    symbol: str
    action: str  # 'BUY' or 'SELL'
    quantity: int
    price: float
    timestamp: datetime
    order_id: str
    commission: float = 0.0


class PortfolioManager:
    """Manages portfolio positions, cash, and P&L tracking"""
    
    def __init__(self, initial_capital: float = 100000.0, max_position_size: float = 0.1):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.max_position_size = max_position_size
        
        # Portfolio tracking
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[Trade] = []
        
        # Performance tracking
        self.total_realized_pnl = 0.0
        self.total_commission = 0.0
        self.peak_portfolio_value = initial_capital
        self.max_drawdown = 0.0
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Load existing portfolio if available
        self._load_portfolio()

    def update_position(self, symbol: str, action: str, quantity: int, price: float, order_id: str = None):
        """Update position based on a trade"""
        try:
            # Calculate commission (simplified - $1 per trade)
            commission = 1.0
            self.total_commission += commission
            
            # Record the trade
            trade = Trade(
                symbol=symbol,
                action=action,
                quantity=quantity,
                price=price,
                timestamp=datetime.now(),
                order_id=order_id or f"ORDER_{len(self.trade_history) + 1}",
                commission=commission
            )
            self.trade_history.append(trade)
            
            if action == 'BUY':
                self._handle_buy(symbol, quantity, price, commission)
            elif action == 'SELL':
                self._handle_sell(symbol, quantity, price, commission)
                
            # Update portfolio metrics
            self._update_portfolio_metrics()
            
            # Save portfolio state
            self._save_portfolio()
            
            self.logger.info(f"Position updated: {action} {quantity} {symbol} at ${price}")
            
        except Exception as e:
            self.logger.error(f"Error updating position: {e}")

    def _handle_buy(self, symbol: str, quantity: int, price: float, commission: float):
        """Handle a buy order"""
        total_cost = (quantity * price) + commission
        
        if self.cash < total_cost:
            raise ValueError(f"Insufficient cash: need ${total_cost}, have ${self.cash}")
        
        self.cash -= total_cost
        
        if symbol in self.positions:
            # Update existing position
            position = self.positions[symbol]
            total_shares = position.quantity + quantity
            total_cost_basis = (position.quantity * position.avg_cost) + (quantity * price)
            position.avg_cost = total_cost_basis / total_shares
            position.quantity = total_shares
        else:
            # Create new position
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                avg_cost=price,
                current_price=price
            )

    def _handle_sell(self, symbol: str, quantity: int, price: float, commission: float):
        """Handle a sell order"""
        if symbol not in self.positions:
            raise ValueError(f"No position in {symbol} to sell")
            
        position = self.positions[symbol]
        
        if position.quantity < quantity:
            raise ValueError(f"Insufficient shares: need {quantity}, have {position.quantity}")
        
        # Calculate proceeds
        proceeds = (quantity * price) - commission
        self.cash += proceeds
        
        # Calculate realized P&L
        cost_basis = quantity * position.avg_cost
        realized_pnl = proceeds - cost_basis
        self.total_realized_pnl += realized_pnl
        position.realized_pnl += realized_pnl
        
        # Update position
        position.quantity -= quantity
        
        # Remove position if fully sold
        if position.quantity == 0:
            del self.positions[symbol]

    def update_market_prices(self, price_data: Dict[str, float]):
        """Update current market prices for all positions"""
        for symbol, price in price_data.items():
            if symbol in self.positions:
                self.positions[symbol].update_price(price)
        
        self._update_portfolio_metrics()

    def _update_portfolio_metrics(self):
        """Update portfolio-level metrics"""
        current_value = self.get_total_value()
        
        # Update peak value and drawdown
        if current_value > self.peak_portfolio_value:
            self.peak_portfolio_value = current_value
        
        # Calculate current drawdown
        current_drawdown = (self.peak_portfolio_value - current_value) / self.peak_portfolio_value
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown

    def get_total_value(self) -> float:
        """Get total portfolio value"""
        positions_value = sum(pos.market_value for pos in self.positions.values())
        return self.cash + positions_value

    def get_available_capital(self) -> float:
        """Get available cash for trading"""
        return self.cash

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a specific symbol"""
        return self.positions.get(symbol)

    def get_all_positions(self) -> Dict[str, Position]:
        """Get all current positions"""
        return self.positions.copy()

    def get_portfolio_summary(self) -> Dict:
        """Get comprehensive portfolio summary"""
        total_value = self.get_total_value()
        positions_value = sum(pos.market_value for pos in self.positions.values())
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        
        return {
            'total_value': total_value,
            'cash': self.cash,
            'positions_value': positions_value,
            'positions_count': len(self.positions),
            'total_realized_pnl': self.total_realized_pnl,
            'total_unrealized_pnl': total_unrealized_pnl,
            'total_pnl': self.total_realized_pnl + total_unrealized_pnl,
            'total_return_pct': ((total_value - self.initial_capital) / self.initial_capital) * 100,
            'max_drawdown_pct': self.max_drawdown * 100,
            'total_trades': len(self.trade_history),
            'total_commission': self.total_commission,
            'positions': {symbol: {
                'quantity': pos.quantity,
                'avg_cost': pos.avg_cost,
                'current_price': pos.current_price,
                'market_value': pos.market_value,
                'unrealized_pnl': pos.unrealized_pnl,
                'unrealized_pnl_pct': ((pos.current_price - pos.avg_cost) / pos.avg_cost) * 100 if pos.avg_cost > 0 else 0
            } for symbol, pos in self.positions.items()}
        }

    def can_buy(self, symbol: str, quantity: int, price: float) -> bool:
        """Check if we can afford to buy a position"""
        total_cost = (quantity * price) + 1.0  # Include commission
        
        if total_cost > self.cash:
            return False
        
        # Check position size limit
        position_value = quantity * price
        total_portfolio_value = self.get_total_value()
        position_pct = position_value / total_portfolio_value
        
        return position_pct <= self.max_position_size

    def can_sell(self, symbol: str, quantity: int) -> bool:
        """Check if we can sell a position"""
        if symbol not in self.positions:
            return False
        return self.positions[symbol].quantity >= quantity

    def get_trade_history(self, symbol: str = None, days: int = None) -> List[Trade]:
        """Get trade history, optionally filtered by symbol and/or days"""
        trades = self.trade_history
        
        if symbol:
            trades = [t for t in trades if t.symbol == symbol]
        
        if days:
            cutoff_date = datetime.now() - timedelta(days=days)
            trades = [t for t in trades if t.timestamp >= cutoff_date]
        
        return trades

    def get_performance_metrics(self) -> Dict:
        """Get detailed performance metrics"""
        if not self.trade_history:
            return {}
        
        # Calculate win rate
        profitable_trades = [t for t in self.trade_history if self._get_trade_pnl(t) > 0]
        win_rate = len(profitable_trades) / len(self.trade_history) if self.trade_history else 0
        
        # Calculate average trade P&L
        trade_pnls = [self._get_trade_pnl(t) for t in self.trade_history]
        avg_trade_pnl = sum(trade_pnls) / len(trade_pnls) if trade_pnls else 0
        
        return {
            'total_trades': len(self.trade_history),
            'profitable_trades': len(profitable_trades),
            'win_rate_pct': win_rate * 100,
            'avg_trade_pnl': avg_trade_pnl,
            'best_trade': max(trade_pnls) if trade_pnls else 0,
            'worst_trade': min(trade_pnls) if trade_pnls else 0,
            'total_commission': self.total_commission
        }

    def _get_trade_pnl(self, trade: Trade) -> float:
        """Calculate P&L for a specific trade (simplified)"""
        # This is a simplified calculation
        # In practice, you'd need to match buy/sell pairs
        return 0  # Placeholder

    def _save_portfolio(self):
        """Save portfolio state to disk"""
        try:
            portfolio_data = {
                'cash': self.cash,
                'initial_capital': self.initial_capital,
                'total_realized_pnl': self.total_realized_pnl,
                'total_commission': self.total_commission,
                'peak_portfolio_value': self.peak_portfolio_value,
                'max_drawdown': self.max_drawdown,
                'positions': {
                    symbol: {
                        'symbol': pos.symbol,
                        'quantity': pos.quantity,
                        'avg_cost': pos.avg_cost,
                        'current_price': pos.current_price,
                        'realized_pnl': pos.realized_pnl
                    } for symbol, pos in self.positions.items()
                },
                'trade_history': [
                    {
                        'symbol': t.symbol,
                        'action': t.action,
                        'quantity': t.quantity,
                        'price': t.price,
                        'timestamp': t.timestamp.isoformat(),
                        'order_id': t.order_id,
                        'commission': t.commission
                    } for t in self.trade_history
                ]
            }
            
            os.makedirs('portfolio_data', exist_ok=True)
            with open('portfolio_data/portfolio.json', 'w') as f:
                json.dump(portfolio_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving portfolio: {e}")

    def _load_portfolio(self):
        """Load portfolio state from disk"""
        try:
            if os.path.exists('portfolio_data/portfolio.json'):
                with open('portfolio_data/portfolio.json', 'r') as f:
                    data = json.load(f)
                
                self.cash = data.get('cash', self.initial_capital)
                self.total_realized_pnl = data.get('total_realized_pnl', 0.0)
                self.total_commission = data.get('total_commission', 0.0)
                self.peak_portfolio_value = data.get('peak_portfolio_value', self.initial_capital)
                self.max_drawdown = data.get('max_drawdown', 0.0)
                
                # Load positions
                for symbol, pos_data in data.get('positions', {}).items():
                    self.positions[symbol] = Position(
                        symbol=pos_data['symbol'],
                        quantity=pos_data['quantity'],
                        avg_cost=pos_data['avg_cost'],
                        current_price=pos_data['current_price'],
                        realized_pnl=pos_data.get('realized_pnl', 0.0)
                    )
                
                # Load trade history
                for trade_data in data.get('trade_history', []):
                    self.trade_history.append(Trade(
                        symbol=trade_data['symbol'],
                        action=trade_data['action'],
                        quantity=trade_data['quantity'],
                        price=trade_data['price'],
                        timestamp=datetime.fromisoformat(trade_data['timestamp']),
                        order_id=trade_data['order_id'],
                        commission=trade_data.get('commission', 0.0)
                    ))
                
                self.logger.info("Portfolio loaded from disk")
                
        except Exception as e:
            self.logger.error(f"Error loading portfolio: {e}")

    def reset_portfolio(self):
        """Reset portfolio to initial state"""
        self.cash = self.initial_capital
        self.positions.clear()
        self.trade_history.clear()
        self.total_realized_pnl = 0.0
        self.total_commission = 0.0
        self.peak_portfolio_value = self.initial_capital
        self.max_drawdown = 0.0
        
        # Delete saved portfolio file
        try:
            if os.path.exists('portfolio_data/portfolio.json'):
                os.remove('portfolio_data/portfolio.json')
        except:
            pass
        
        self.logger.info("Portfolio reset to initial state")